# Moved from BUG-Julia@b155550:src/RSVDCBEBondUpdate/bug/krylov_enrich.jl, src/RSVDCBEBondUpdate/bug/krylov_closure.jl, src/RSVDCBEBondUpdate/bug/krylov_closure_loop.jl.

# ── from src/RSVDCBEBondUpdate/bug/krylov_enrich.jl ──

# ── THE HALF-SWEEP KRYLOV STAGE OF `cbe_bug_step!` (plan §4, Tests C and D) ─────────────────

_tagW(i::Int) = "cbeW,$i"
_tagZ(j::Int) = "cbeZ,$j"

# The residual below which the Krylov space has genuinely CLOSED -- an invariant subspace, not a
# convergence judgement.
const _KRY_BREAKDOWN = 1e-13

# ── STAGE 3: residual-driven K/L Krylov enrichment (plan §4, Tests C and D) ──
# Krylov vectors live on one-site tensors, but the plan's projector `I - QQ'` acts on the frame (an
# isometry on `(link, site)`): a vector adds only through the part of its local range the frame
# lacks. `Q0` is the CBE output `U_ex`, so old directions are kept verbatim; the seed `K_ex` already
# lies in `U_old`.

const _TAG_K, _TAG_KR = "cbeK,L", "cbeK,R"

"Safety bound on the adaptive Krylov depth. Reaching it with weight left is REPORTED, not hidden."
const _KRY_MAXDEPTH = 64

# The root solve's Krylov basis per substep. A NUMERICAL cap, not an
# accuracy one: `substeps = 0` splits `tau` whenever this basis cannot reach `tol_expv`.
const _ROOT_MAXITER = 30

# Relative Hermitian part of `tau*H_k` below which the Krylov propagator counts as unitary: CGS2
# leaves a Hermitian generator's Hessenberg Hermitian to ~1e-15, and a loss term far below this
# gives the same weight in both branches anyway.
const _KRY_SKEW_TOL = 1e-12

"""
    _kry_weight(Hm, k, tau; state_seed = true) -> Float64

The weight the newest Krylov vector `v_{k+1}` carries in `exp(tau*H) v_1`, from the Arnoldi
Hessenberg after `k` steps: `|tau * beta_k * [phi_1(tau*H_k)]_{k,1}|`, read from one `(k+1)x(k+1)`
exponential. This is Saad's a posteriori error of the `k`-dimensional approximation, so it is also
the closure test: below the allowance, no later direction matters more.

`state_seed` is false when `v_1` is a closure frontier rather than the state's own tensor; it
matters only for a non-unitary propagator.

The propagator is unitary when `tau*H_k` is skew-Hermitian, not whenever `real(tau) == 0`: a
no-jump generator `H - (i/2)Q` at `tau = -im*dt` contracts, and its weight must be relative to the
decayed state, not to the start norm.
"""
function _kry_weight(Hm::Matrix{ComplexF64}, k::Int, tau::ComplexF64; state_seed::Bool = true)
    A = zeros(ComplexF64, k + 1, k + 1)
    A[1:k, 1:k] .= tau .* @view Hm[1:k, 1:k]
    A[k + 1, k] = tau * Hm[k + 1, k]
    Ak = @view A[1:k, 1:k]
    norm(Ak + Ak') <= _KRY_SKEW_TOL * norm(Ak) && return abs(exp(A)[k + 1, 1])
    # Non-unitary (imaginary time, Lindblad, no-jump loss): weigh relative to the state's own
    # growth `g = ||exp(tau*H_k) e_1||`, which is `exp(dbeta*|E|)` unshifted and overflows at large dbeta. The
    # shift `s` cancels in that ratio exactly.
    s = maximum(real, eigvals(A[1:k, 1:k]))
    E = exp(A - s * I)
    state_seed && return abs(E[k + 1, 1]) / max(norm(@view E[1:k, 1]), floatmin(Float64))
    # A closure frontier is not the state: it decays under the shifted generator, so dividing by its
    # own growth inflated its directions by `e^{dbeta h11}`, enough to block a converged step. Its
    # weight is the absolute one, `|E[k+1,1]| e^s`, in the shifted frame, which never under-weights
    # in imaginary time (the shift is the energy there).
    w = abs(E[k + 1, 1])
    return w == 0 ? 0.0 : exp(log(w) + s)
end

"""
    _extend_frame(Q, R, v, side, weight, allowance; room) -> (Q, n_added, fits)

Append to the frame `Q` the local directions of the residual `R = (I - QQ') v` that the
allowance requires: the smallest number whose weighted discarded tail `weight * sqrt(sum s^2)`
fits (Test A's rule, `_tail_rank`). Orthogonalised within each charge sector by the block SVD,
cut only at the numerical floor `_rank_cut`, re-orthogonalised against `Q` before the direct
sum -- the same guard `cbe_expand` applies to its own appended block.

`room` is what the work ceiling still allows on the frame leg (plan §5): at most that many are
appended, and `fits = false` reports that the allowance needed more.
"""
function _extend_frame(Q, R, v, side::Symbol, weight::Float64, allowance::Float64;
                       room::Int = typemax(Int))
    legs = side === :left ? (1, 2) : (2, 3)
    cut = _rank_cut(0.0, v, norm(R))
    res = svd(R, legs, _TAG_K, _TAG_KR; cutoff = cut, get_lists = true)
    nk = length(res.kept_list)
    k, tail = _tail_rank(res.kept_list, weight, allowance, min(nk, room);
                         base = _sq_weight(res.trunc_list))
    fits = tail <= allowance
    k == 0 && return (Q, 0, fits)
    k < nk && (res = svd(R, legs, _TAG_K, _TAG_KR; Nkeep = k, cutoff = cut))
    U = side === :left ? _reortho_left(Q, to_concrete(res.U)) :
                         _reortho_right(Q, to_concrete(permutedims(res.U, (3, 1, 2))))
    # roundoff in a saturated sector must not become a column (`_cap_to_local_room`)
    U = _cap_to_local_room(Q, U, side)
    U === nothing && return (Q, 0, fits)
    fl = side === :left ? 3 : 1
    Qn = to_concrete(oplus([Q, to_concrete(setitag(U, fl, Q.inds[fl].itags))], (fl,)))
    return (_check_frame(Qn, side, "_extend_frame"), leg_dim(U, fl), fits)
end

"""
    _krylov_enrich(v0, applyH, Q0, side, tau, alloc; workcap = 0, maxdepth = _KRY_MAXDEPTH,
                   amp = 1.0, state_seed = true, fixed_depth = 0, recycled = ())
        -> (Q, stats)

Grow the frame `Q0` by the Krylov directions of `applyH` from seed `v0` that the step needs. Each
direction is scored by `weight * ||(I - QQ') v_k||`: `eta_1` is Test C, `eta_{k>=2}` Test D. Scores
above `alloc.krylov` are appended. A small `eta_1` is not a stop (`H e1 = e2`, `H e2 = e1 + e3` with
`Q0 = [e1, e2]`). `capped` with `met = false` means the work ceiling or depth bound stopped the
search with weight left. `amp` is the seed's norm relative to the state. `stats.basis` and
`stats.weights` are the Arnoldi vectors and their weights, the last vector the active frontier: what
a later round recycles. With `recycled` only the complement of the previous solve's input is
extended ([`_krylov_basis`](@ref)).

`fixed_depth = m > 0` is the conventional-control ablation (plan Figs. 2 and 4): exactly `m` steps,
every new local direction appended (numerical floor only), no weight stop; `met` then reports
whether the weight had fallen to the allowance by depth `m`.
"""
function _krylov_enrich(v0, applyH, Q0, side::Symbol, tau::ComplexF64,
                        alloc::BasisAllocation; workcap::Int = 0,
                        maxdepth::Int = _KRY_MAXDEPTH, amp::Float64 = 1.0,
                        state_seed::Bool = true, fixed_depth::Int = 0, recycled = ())
    side in (:left, :right) || throw(ArgumentError("side must be :left or :right"))
    perp = side === :left ? perp_component : perp_component_right
    fl = side === :left ? 3 : 1
    fixed = fixed_depth > 0
    fixed && (maxdepth = fixed_depth)
    allow = fixed ? 0.0 : alloc.krylov
    Q = Q0
    eta_c, eta_d, added, added_d = 0.0, 0.0, 0, 0
    function score(v, weight, first::Bool)          # Tests C/D on one direction; false = capped
        R = to_concrete(perp(Q, v))
        eta = weight * norm(R)
        first ? (eta_c = eta) : (eta_d = max(eta_d, eta))
        eta > allow || return true
        # The ceiling bounds the frame WIDTH, not the number of extensions: a block may not carry
        # the frame past it either.
        room = workcap > 0 ? workcap - leg_dim(Q, fl) : typemax(Int)
        room <= 0 && return false
        Q, nadd, fits = _extend_frame(Q, R, v, side, weight, allow; room = room)
        added += nadd
        first || (added_d += nadd)
        # an allowance of 0 never "fits" (the floor's tail): only the adaptive stop caps here
        return fits || fixed
    end
    kb = _krylov_basis(v0, applyH, tau, alloc.krylov, score; recycled = recycled,
                       maxdepth = maxdepth, amp = amp, state_seed = state_seed, fixed = fixed)
    return (Q, (eta_c = eta_c, eta_d = eta_d, added = added, n_h1 = kb.n_h1,
                d_after_c = eta_c <= alloc.krylov && added_d > 0,
                capped = kb.capped, met = kb.closed && !kb.capped, basis = kb.basis,
                weights = kb.weights, n_recycled = kb.n_recycled, rec_closed = kb.rec_closed,
                rec_overlap = kb.rec_overlap, rec_defect = kb.rec_defect))
end

# The recurrence runs on the local tensors and, for the matrix tests, on plain vectors.
_kinner(a, b) = tensor_inner(a, b)
_kinner(a::Vector{<:Number}, b::Vector{<:Number}) = dot(a, b)
_kconc(x) = to_concrete(x)
_kconc(x::Vector{<:Number}) = x

"A recycled vector whose part outside the block is below this fraction of it adds nothing new."
const _RECYCLE_FLOOR = 1e-8

"""
    _krylov_basis(v0, applyH, tau, allowance, score; recycled = (), maxdepth, amp, state_seed,
                  fixed = false)
        -> (basis, weights, n_h1, closed, capped, n_recycled, rec_closed, rec_overlap, rec_defect)

The local Krylov recurrence of the frame stages. `score(v, weight, first) -> fits` takes each
direction to the frame (`first` = Test C's direction), `false` stopping the search as capped.
`basis[1]` is always the solve's input `v0 / ||v0||`, the last vector the active frontier.

Without `recycled` it is Arnoldi from `v0` ([`_arnoldi_chain`](@ref)). With `recycled` -- the
previous solve's input first, then its kept Krylov vectors, all in this round's gauge -- the input
is split as `v0 = alpha p + v_c` and only the complement `v_c` is extended
([`_rk_complement`](@ref)); when the previous block no longer holds `p`'s evolution it is Arnoldi
from `v0` after all.
"""
function _krylov_basis(v0, applyH, tau::ComplexF64, allowance::Float64, score;
                       recycled = (), maxdepth::Int = _KRY_MAXDEPTH, amp::Float64 = 1.0,
                       state_seed::Bool = true, fixed::Bool = false)
    b0 = norm(v0)
    b0 > 0 || return (basis = Any[], weights = Float64[], n_h1 = 0, closed = true,
                      capped = false, n_recycled = 0, rec_closed = false, rec_overlap = 0.0,
                      rec_defect = 0.0)
    # `tau = 0` asks the unit-time closure question, as `basis_allocation` does.
    teff = iszero(tau) ? one(tau) : tau
    u0 = _kconc((1.0 / b0) * v0)
    rc = (n_h1 = 0, n_recycled = 0, rec_overlap = 0.0, rec_defect = 0.0)
    if !isempty(recycled)
        rc = _rk_complement(u0, recycled, applyH, teff, allowance, score; maxdepth = maxdepth,
                            amp = amp, state_seed = state_seed)
        haskey(rc, :basis) && return rc
    end
    ch = _arnoldi_chain(u0, applyH, teff, allowance, score; maxdepth = maxdepth, amp = amp,
                        state_seed = state_seed, fixed = fixed)
    return (basis = ch.basis, weights = ch.weights, n_h1 = rc.n_h1 + ch.n_h1, closed = ch.closed,
            capped = ch.capped, n_recycled = rc.n_recycled, rec_closed = false,
            rec_overlap = rc.rec_overlap, rec_defect = rc.rec_defect)
end

"""
    _arnoldi_chain(u, applyH, tau, allowance, score; maxdepth, amp, state_seed, fixed)
        -> (basis, weights, n_h1, closed, capped)

Arnoldi from the unit seed `u`: each new vector weighed by `amp` times Saad's estimate
(`_kry_weight`) and offered to `score`, closed once that weight is within `allowance`. `fixed`
runs all `maxdepth` steps (the `fixed_depth` ablation of [`_krylov_enrich`](@ref)).
"""
function _arnoldi_chain(u, applyH, tau::ComplexF64, allowance::Float64, score;
                        maxdepth::Int = _KRY_MAXDEPTH, amp::Float64 = 1.0,
                        state_seed::Bool = true, fixed::Bool = false)
    nh, capped, closed = 0, false, false
    vs, ws = Any[u], Float64[amp]
    Hm = zeros(ComplexF64, maxdepth + 1, maxdepth + 1)
    for k in 1:maxdepth
        w = applyH(vs[end])
        nh += 1
        # ⛔ CGS2 against EVERY previous vector, the correction accumulated lazily and subtracted
        # ONCE per pass: `2m` materialisations per solve instead of `m^2` (a per-vector update
        # allocated 28.6 GiB per step at L=20, chi=32, d=4). Classical Gram-Schmidt repeated
        # twice is as stable as modified ("twice is enough"); the coefficients accumulate
        # across both passes, so column `k` of `Hm` is the true projection of `H v_k`.
        for _pass in 1:2
            corr = nothing
            for (t, x) in pairs(vs)
                ov = _kinner(x, w)
                Hm[t, k] += ov
                corr = corr === nothing ? ov * x : corr + ov * x
            end
            corr === nothing || (w = _kconc(w - corr))
        end
        b = norm(w)
        b <= _KRY_BREAKDOWN && (closed = true; break)    # invariant: nothing further exists
        Hm[k + 1, k] = b
        v = _kconc((1.0 / b) * w)
        push!(vs, v)
        weight = amp * _kry_weight(Hm, k, tau; state_seed = state_seed)
        push!(ws, weight)
        score(v, weight, k == 1) || (capped = true; break)
        weight <= allowance && (closed = true; fixed || break)
    end
    closed || fixed || (capped = true)                    # depth bound with weight left
    return (basis = vs, weights = ws, n_h1 = nh, closed = closed, capped = capped)
end

"""
    _rk_complement(u0, recycled, applyH, tau, allowance, score; maxdepth, amp, state_seed)

The recycled solve (plan §1.2: "transport/reorthogonalize old useful directions into the new gauge,
and extend only from active frontiers"; §1.3: `tol_basis` carries "the recycled local Krylov basis
defect"). `recycled[1]` is the previous solve's input `p`, the rest its kept Krylov vectors. By
linearity `exp(tau H) u0 = alpha exp(tau H) p + exp(tau H) v_c` with `alpha = <p, u0>` and the
complement `v_c = u0 - alpha p`: the previous block `B = orth[p, recycled...]` already spans the
first term if its defect for `p` under THIS round's `H1`, `d = tau (HB - BA) phi_1(tau A) e1` with
`A = B'HB`, is small, and only `v_c` is new.

- `amp |alpha|` within the allowance: `p` carries nothing, nothing is tested (no `H1`).
- `amp |alpha| ||d|| <= allowance / 2` (one `H1` per block vector): the block joins the frame
  weighted by `amp |alpha| |exp(tau A) e1|`, and a fresh Arnoldi chain runs from `v_c` alone with
  `amp ||v_c||` and the allowance left -- unless `v_c`'s whole evolution fits in it (`2 amp ||v_c||`
  bounds it for a contraction), when `v_c` only joins the frame. Returns the solve, `rec_closed`.
- otherwise the block no longer holds `p`'s evolution (the generator moved between the rounds) and
  the caller runs Arnoldi from `u0`; returns only the counts.

The previous block is only ever tested, never extended: a block whose every vector leaves a
residual, continued one direction per step, ran to the depth bound in the matrix tests. Skipping
the test is no option either: with the generator moved by 1e-6 relative between the solves, the
complement's chain alone left errors 10x the allowance (1e-3: 1e4x).
"""
function _rk_complement(u0, recycled, applyH, tau::ComplexF64, allowance::Float64, score;
                        maxdepth::Int = _KRY_MAXDEPTH, amp::Float64 = 1.0,
                        state_seed::Bool = true)
    none = (n_h1 = 0, n_recycled = 0, rec_overlap = 0.0, rec_defect = 0.0)
    np = norm(recycled[1])
    np > 0 || return none
    p = _kconc((1.0 / np) * recycled[1])
    al = _kinner(p, u0)
    ampr = amp * abs(al)
    ampr > allowance || return merge(none, (rec_overlap = ampr,))
    B = Any[p]
    for c in Iterators.drop(recycled, 1)
        x = _orth_against(B, c)
        nx = norm(x)
        nx > _RECYCLE_FLOOR * max(norm(c), eps()) && push!(B, _kconc((1.0 / nx) * x))
    end
    HB = Any[_kconc(applyH(x)) for x in B]
    nh = length(B)
    A = ComplexF64[_kinner(B[r], HB[c]) for r in eachindex(B), c in eachindex(B)]
    g, ph = _exp_phi1_e1(tau .* A)
    dr = ampr * norm(_kconc(tau * (_lincomb(HB, ph) - _lincomb(B, A * ph))))
    counts = (n_h1 = nh, n_recycled = length(B), rec_overlap = ampr, rec_defect = dr)
    dr <= allowance / 2 || return counts
    gw = ampr .* abs.(g)
    capped, closed = false, true
    for (k, x) in pairs(B)
        score(x, gw[k], false) || (capped = true; break)
    end
    basis, weights = vcat(Any[u0], B), vcat(amp, gw)
    vc = _kconc(u0 - al * p)
    nc = norm(vc)
    rest = allowance - dr
    if !capped && nc > 0
        uc = _kconc((1.0 / nc) * vc)
        if 2 * amp * nc <= rest
            score(uc, amp * nc, false) || (capped = true)
            push!(basis, uc)
            push!(weights, amp * nc)
        else
            ch = _arnoldi_chain(uc, applyH, tau, rest, score; maxdepth = maxdepth,
                                amp = amp * nc, state_seed = state_seed)
            nh += ch.n_h1
            capped, closed = ch.capped, ch.closed
            append!(basis, ch.basis)
            append!(weights, ch.weights)
        end
    end
    return (basis = basis, weights = weights, n_h1 = nh, closed = closed, capped = capped,
            n_recycled = length(B), rec_closed = true, rec_overlap = ampr, rec_defect = dr)
end

"`sum_k c[k] vs[k]`."
function _lincomb(vs, c)
    acc = nothing
    for (k, v) in pairs(vs)
        iszero(c[k]) && continue
        acc = acc === nothing ? c[k] * v : acc + c[k] * v
    end
    return acc === nothing ? _kconc(0.0 * vs[1]) : _kconc(acc)
end

"`(exp(A) e1, phi_1(A) e1)` from one augmented exponential: `exp([A e1; 0 0])`."
function _exp_phi1_e1(A::AbstractMatrix)
    n = size(A, 1)
    Z = zeros(ComplexF64, n + 1, n + 1)
    Z[1:n, 1:n] .= A
    Z[1, n + 1] = 1
    E = exp(Z)
    return E[1:n, 1], E[1:n, n + 1]
end

"""
    _merge_frame(Q, U, side) -> Q'

`Q` grown by the part of the frame `U` it does not span, cut only at the numerical floor (the
`krylov_reuse = false` ablation: the CBE directions join after the Krylov stage instead of
seeding it).
"""
function _merge_frame(Q, U, side::Symbol)
    R = to_concrete(side === :left ? perp_component(Q, U) : perp_component_right(Q, U))
    # `U` inside `Q` (no CBE direction this step): Telum drops the cancelled blocks and its SVD of
    # a blockless tensor throws (hq34b 17436300)
    norm(R) == 0 && return Q
    return first(_extend_frame(Q, R, U, side, 1.0, 0.0))
end

"Fold one `_krylov_enrich` verdict into the step ledger (plan §8 Krylov diagnostics)."
function _record_krylov!(ledger::Union{Nothing, AdaptiveLedger}, st, alloc::BasisAllocation)
    ledger === nothing && return nothing
    ledger.max_krylov_c = max(ledger.max_krylov_c, st.eta_c)
    ledger.max_krylov_d = max(ledger.max_krylov_d, st.eta_d)
    ledger.max_krylov   = max(ledger.max_krylov, st.eta_c, st.eta_d)
    ledger.n_krylov += st.added
    ledger.n_h1     += st.n_h1
    st.eta_c <= alloc.krylov && (ledger.n_c_pass += 1)
    st.d_after_c && (ledger.n_d_after_c += 1)
    st.capped && (ledger.workcap_reached = true)
    st.met || _unmet!(ledger, :krylov)
    return nothing
end

# ── from src/RSVDCBEBondUpdate/bug/krylov_closure.jl ──

# ── GENERAL-ORDER KRYLOV CLOSURE (order-p condition) ──
# The order-p error is O(|tau|^(p+1)) once `(I - P) A B^j psi = 0` for j < p, with B = P A P.
# Frontiers are judged by `(I - P) A F_j`, not by a root residual (which concerns B, not A).

"A frontier below this fraction of the state norm is the zero vector (e.g. `H psi = 0`)."
const _FRONTIER_FLOOR = 1e-14

"Internal bound on closure passes. Reaching it with the frames still growing is `:closure`."
_closure_maxpass(order::Int) = 2 * order + 2

# Each enriching frontier sweep re-tags its frames with a new generation, so a frame and the
# frame it grew from never share a tag inside one overlap tensor.
_tagWg(i::Int, g::Int) = g == 0 ? _tagW(i) : "cbeW$g,$i"
_tagZg(j::Int, g::Int) = g == 0 ? _tagZ(j) : "cbeZ$g,$j"

"""
    KLFrames(W, Z, c, lenv, renv, gen)

The K/L frames of one step and the root environments they imply: `W[1..c]` left frames,
`Z[c..L-1]` right frames (`Z[j]` absorbs site `j+1`), `lenv` the environment at link `c` through
`W[1..c-1]`, `renv` the one at link `c+2` through `Z[c+1..L-1]`. `gen` counts the enrichments,
so a [`KrylovFrontier`](@ref) built in older frames is refused rather than silently misread.

`lcache` / `rcache` hold the environments every frontier sweep of ONE generation shares
([`_side_cache!`](@ref)); each half-sweep writes only its own, so the parallel halves never
write the same field.
"""
mutable struct KLFrames
    W::Vector{Any}
    Z::Vector{Any}
    c::Int
    lenv::Any
    renv::Any
    gen::Int
    lcache::Any
    rcache::Any
    # The generators' energy shift (`shift_h`): the root `B - eshift` and the frontier sweeps'
    # one-site Krylov operators. `0.0` is the unshifted step.
    eshift::Float64
end

KLFrames(W, Z, c, lenv, renv, gen) = KLFrames(W, Z, c, lenv, renv, gen, nothing, nothing, 0.0)

"""
Threads below which the closure keeps its frontier sweeps in sequence. Each sweep already runs
its two halves concurrently under `parallel = true`, so two concurrent sweeps need four threads
to gain anything; below that the 2-thread production path stays exactly what it was.
"""
const _FRONTIER_THREADS = 4

_frontier_parallel(parallel::Bool) = parallel && Threads.nthreads() >= _FRONTIER_THREADS

"""
Whether a measuring sweep runs its bonds as concurrent tasks ([`_bond_measurements`](@ref)) when
[`_frontier_parallel`](@ref) holds. A switch, not a tolerance: it changes no result at
`fold = false` and exists so a timing job can A/B it in one build (`bug_profile_2d.jl`,
`BP_BOND_TASKS`). Off by default: at 4 threads, D = 512 it gained nothing (job 17287169).
"""
const _BOND_TASKS = Ref(false)

"""
    _prime_caches!(fr, psi, mpo, core)

Build this generation's shared environments ([`_side_cache!`](@ref)) before frontier sweeps run
concurrently, so the concurrent sweeps only READ them. `core` is any root core: the cached
parts do not depend on it. The two sides are separate fields and are built concurrently.
"""
function _prime_caches!(fr::KLFrames, psi::SymMPS, mpo, core)
    Fm = _frontier_mps(psi, fr, core)
    t = Threads.@spawn _side_cache!(fr, Fm, mpo, :right)
    _side_cache!(fr, Fm, mpo, :left)
    fetch(t)
    return fr
end

"""
    _side_cache!(fr, Fm, mpo, side) -> (chain, stack)

The environments of one half-sweep that do not depend on the frontier, built once per frame
generation. ⛔ EVERY FRONTIER OF A GENERATION IS `(W ⊗ Z) core` WITH ONLY THE CORE DIFFERENT, so
re-gauging `Fm` to site 1 (left half) changes sites `1..c` only: the right stack on links
`c+1..L+1` and the frame chain `<W[1..i-1]| H |W[1..i-1]>` (the non-enriching sweep's `lch`)
are the same for every frontier. Rebuilding them per sweep was ~1/3 of a measuring sweep
(bug4 10x4 D=256 profile, job 17283685). `chain[i]` is the left channels at bond `i` (`i = 1..c`),
`stack` the right-stack entries on links `>= c+1`; the right half mirrors both (`chain[j]` the
right channels at bond `j = c..L-1`, `stack` the left-stack entries on links `<= c`).
"""
function _side_cache!(fr::KLFrames, Fm::SymMPS, mpo, side::Symbol)
    L, c = length(Fm), fr.c
    cur = side === :left ? fr.lcache : fr.rcache
    valid = cur !== nothing && cur.gen == fr.gen
    chain = valid ? cur.chain : nothing          # an enriching sweep seeds the chain only
    E = valid ? cur.stack : nothing
    (chain !== nothing && E !== nothing) && return (chain, E)
    if chain === nothing
        chain = Vector{Any}(undef, L - 1)
        ch = boundary_channels(mpo)
        if side === :left
            for i in 1:c
                chain[i] = ch
                i < c && (ch = push_left_channels(ch, mpo, fr.W[i], i))
            end
        else
            for j in (L - 1):-1:c
                chain[j] = ch
                j > c && (ch = push_right_channels(ch, mpo, fr.Z[j], j + 1))
            end
        end
    end
    E = Any[missing for _ in 1:(L + 1)]
    env = nothing
    if side === :left
        E[L + 1] = nothing
        for k in L:-1:(c + 1)
            env = _mpo_right_step(env, Fm[k], mpo[k])
            E[k] = env
        end
        fr.lcache = (gen = fr.gen, chain = chain, stack = E)
    else
        E[1] = nothing
        for i in 1:(c - 1)
            env = _mpo_left_step(env, Fm[i], mpo[i])
            E[i + 1] = env
        end
        fr.rcache = (gen = fr.gen, chain = chain, stack = E)
    end
    return (chain, E)
end

"""
    KrylovFrontier(core, degree, gen)

A projected Krylov frontier `F_j = (tau^j / j!) B^j S0`: its coefficients `core` in the root
basis of frames generation `gen` (a block-sparse `(b_L, b_R)` tensor, so its U(1) sectors travel
with it) and its degree `j`. The cached environments are the frames' own.
"""
struct KrylovFrontier
    core::Any
    degree::Int
    gen::Int
end

"""
    frames_from_state(psi, mpo, c) -> KLFrames

The state's OWN Schmidt frames at root bond `c` -- the subspace every step starts from, before
any expansion. Used to start a closure from scratch (`include_zero = true`) and by the tests.
"""
function frames_from_state(psi::SymMPS, mpo, c::Int)
    L = length(psi)
    1 <= c <= L - 1 || throw(ArgumentError("root must be a bond in 1:$(L - 1), got $c"))
    p = copy(psi)
    canonical!(p, c)
    W = Vector{Any}(undef, c)
    for i in 1:(c - 1)
        W[i] = p[i]
    end
    W[c] = to_concrete(svd(p[c], (1, 2); cutoff = 0.0).U)
    q = copy(psi)
    canonical!(q, c + 1)
    Z = Vector{Any}(undef, L - 1)
    Z[c] = to_concrete(svd(q[c + 1], (1,); cutoff = 0.0).Vd)
    for j in (c + 1):(L - 1)
        Z[j] = q[j + 1]
    end
    # Frame tags on both ends of every frame link, as the half-sweeps write them.
    for i in 1:c
        W[i] = to_concrete(setitag(W[i], 3, _tagW(i)))
        i < c && (W[i + 1] = to_concrete(setitag(W[i + 1], 1, _tagW(i))))
    end
    for j in c:(L - 1)
        Z[j] = to_concrete(setitag(Z[j], 1, _tagZ(j)))
        j > c && (Z[j - 1] = to_concrete(setitag(Z[j - 1], 3, _tagZ(j))))
    end
    lenv = boundary_channels(mpo)
    for i in 1:(c - 1)
        lenv = push_left_channels(lenv, mpo, W[i], i)
    end
    renv = boundary_channels(mpo)
    for j in (L - 1):-1:(c + 1)
        renv = push_right_channels(renv, mpo, Z[j], j + 1)
    end
    return KLFrames(W, Z, c, lenv, renv, 0)
end

"`<W[1..c] | psi>` on the left of the root: `(b_L, link_{c+1})`. A full contraction, any gauge."
function _overlap_left(psi::SymMPS, W, c::Int)
    AL = nothing
    for i in 1:c
        T = AL === nothing ? psi[i] : to_concrete!(contract(AL, (2,), psi[i], (1,)))
        AL = to_concrete!(contract(W[i]', (1, 2), T, (1, 2)))         # (b_L, link_{i+1})
    end
    return AL
end

"`<Z[c..L-1] | psi>` on the right of the root: `(link_{c+1}, b_R)`. A full contraction, any gauge."
function _overlap_right(psi::SymMPS, Z, c::Int)
    AR = nothing
    for j in (length(psi) - 1):-1:c
        B = AR === nothing ? psi[j + 1] : to_concrete!(contract(psi[j + 1], (3,), AR, (1,)))
        AR = to_concrete!(contract(B, (2, 3), Z[j]', (2, 3)))         # (link_j, b_j)
    end
    return AR
end

"""
    _projection_miss(psi, W, Z, c) -> Float64

`||(I - P) psi||` for `P` the projector onto the root space of `W[1..c]`, `Z[c..L-1]`, measured
DIRECTLY. `I - P = (I - P_L) ⊗ 1 + P_L ⊗ (I - P_R)`, and each side telescopes into orthogonal
one-site pieces `(1 - W_i W_i') T_i` / `B_j (1 - Z_j' Z_j)` that are formed explicitly, so the
roundoff floor is ~eps. The difference of squares `||psi||^2 - ||P psi||^2` cancels to
~sqrt(eps), the size of a tight `tol_basis`. The rest of `psi` beyond each piece enters through
a square-root factor of its Gram matrix (an SVD's `U S` / `S V'`), so any gauge works.
"""
function _projection_miss(psi::SymMPS, W, Z, c::Int)
    L = length(psi)
    R = Vector{Any}(undef, L)                    # R[k] R[k]' = Gram of psi[k..L] on link_k
    for k in L:-1:2
        T = k == L ? psi[L] : to_concrete!(contract(psi[k], (3,), R[k + 1], (1,)))
        res = svd(T, (1,); cutoff = 0.0)
        R[k] = to_concrete(res.U * res.S)
    end
    miss2 = 0.0
    AL = nothing
    for i in 1:c
        T = AL === nothing ? psi[i] : to_concrete!(contract(AL, (2,), psi[i], (1,)))
        miss2 += norm(to_concrete!(contract(perp_component(W[i], T), (3,), R[i + 1], (1,))))^2
        AL = to_concrete!(contract(W[i]', (1, 2), T, (1, 2)))         # (b_i, link_{i+1})
    end
    G = Vector{Any}(undef, L - 1)                # G[j]' G[j] = Gram of P_L psi on link_{j+1}
    res = svd(AL, (1,); cutoff = 0.0)
    G[c] = to_concrete(res.S * res.Vd)
    for j in (c + 1):(L - 1)
        res = svd(to_concrete!(contract(G[j - 1], (2,), psi[j], (1,))), (1, 2); cutoff = 0.0)
        G[j] = to_concrete(res.S * res.Vd)
    end
    AR = nothing
    for j in (L - 1):-1:c
        B = AR === nothing ? psi[j + 1] : to_concrete!(contract(psi[j + 1], (3,), AR, (1,)))
        miss2 += norm(to_concrete!(contract(G[j], (2,), perp_component_right(Z[j], B), (1,))))^2
        AR = to_concrete!(contract(B, (2, 3), Z[j]', (2, 3)))         # (link_{j+1}, b_j)
    end
    return sqrt(miss2)
end

"`S = (W ⊗ Z)' psi`, the root coefficients of `P psi` (plan: `project_root`)."
_project_root(psi::SymMPS, fr::KLFrames) =
    to_concrete!(contract(_overlap_left(psi, fr.W, fr.c), (2,),
                          _overlap_right(psi, fr.Z, fr.c), (1,)))

"`B = P H P` at the root, from the frames' cached environments (plan: `root_generator`)."
_root_generator(mpo, fr::KLFrames) =
    shift_h(zero_site_h(mpo, fr.c, fr.lenv, fr.renv, fr.W[fr.c], fr.Z[fr.c]), fr.eshift)

_scaled(a::Number, x::Array{<:Number}) = a .* x
_scaled(a::Number, x) = to_concrete(a * x)

"""
    _projected_frontiers(applyB, S0, tau, m) -> [F_1, ..., F_m]

`F_j = (tau^j / j!) B^j S0` by the recursion `F_j = (tau/j) B F_{j-1}`. Generic in the vector
type (root cores here, plain vectors in the dense unit tests).
"""
function _projected_frontiers(applyB, S0, tau::Number, m::Int)
    Fs = Any[]
    F = S0
    for j in 1:m
        F = _scaled(tau / j, applyB(F))
        push!(Fs, F)
    end
    return Fs
end

"""
    _frontier_mps(psi, fr, core) -> SymMPS

The Galerkin vector `(W ⊗ Z) core` as an MPS on `psi`'s link tags, centre at the root -- the
same assembly `cbe_bug_step!` uses for the evolved core.
"""
function _frontier_mps(psi::SymMPS, fr::KLFrames, core)
    L, c = length(psi), fr.c
    T = Vector{Any}(undef, L)
    for i in 1:(c - 1)
        T[i] = _relink(fr.W[i], psi[i].inds[1].itags, psi[i].inds[3].itags)
    end
    T[c] = _relink(to_concrete(fr.W[c] * core), psi[c].inds[1].itags, psi[c].inds[3].itags)
    for j in c:(L - 1)
        T[j + 1] = _relink(fr.Z[j], psi[j + 1].inds[1].itags, psi[j + 1].inds[3].itags)
    end
    return SymMPS(T, c)
end

"""
    _frame_onto_left(U0, K, right) -> BondFrame

`_frame_from` with the LEFT isometry prescribed: `U0` is the current frame (embedded in the grown
basis of the bond before), which contains the range of the carry `K`, so `U0 S0 V0` is the exact
two-site block. `old_rank` is `U0`'s width -- the side being grown.
"""
function _frame_onto_left(U0, K, right)
    res_r = svd(right, (1,), "bV,L", "bV,R"; cutoff = 0.0)
    V0 = to_concrete(res_r.Vd)
    S0 = to_concrete(to_concrete!(contract(U0', (1, 2), K, (1, 2))) *
                     to_concrete(res_r.U * res_r.S))
    return BondFrame(U0, S0, V0, K.inds[1], K.inds[2], K.inds[3], right.inds[2],
                     right.inds[3], leg_dim(U0, 3), copy(K.spaces[3]))
end

"Row mirror of [`_frame_onto_left`](@ref): the right isometry `V0` is prescribed."
function _frame_onto_right(left, B, V0)
    res_l = svd(left, (1, 2), "bU,L", "bU,R"; cutoff = 0.0)
    U0 = to_concrete(res_l.U)
    S0 = to_concrete(to_concrete(res_l.S * res_l.Vd) *
                     to_concrete!(contract(B, (2, 3), V0', (2, 3))))
    return BondFrame(U0, S0, V0, left.inds[1], left.inds[2], left.inds[3], B.inds[2],
                     B.inds[3], leg_dim(V0, 1), copy(left.spaces[3]))
end

"`||(I - U0 U0') X||` for a two-site `X = (link_l, site_l, site_r, link_r)`."
_left_leak(U0, X) = norm(perp_component(U0, X))

"`||X (I - V0' V0)||` for a two-site `X`, the right frame's miss."
function _right_leak(V0, X)
    ov = contract(X, (3, 4), V0', (2, 3))                               # (link_l, site_l, bond)
    return norm(to_concrete(X - to_concrete!(contract(ov, (3,), V0, (1,)))))
end

"""
    _bond_action(expansion, f, mpo, i, lch, rch, U0, V0, side, fold, rng) -> (skl, skr, leak, nh2)

One frontier bond's sketch closures and outside-space action on `side`. With `fold = false` the
exact two-site action `X = H Theta_F` is formed once, the leak is exact, and the closures share `X`
(`nh2 = 1`). With `fold = true` (`fold_omega = true` in `cbe_bug_step!`) `X` is never formed: the
probe is folded into the environment contractions, and the leak is Test B's holdout estimate at a
quarter of the frame width. This is the large-chi path, since `X` lives on the frame width (~2chi)
and the exact action costs far more than the folded probe. The closures and the estimate come from
the basis-selection strategy `expansion`.
"""
function _bond_action(expansion::BasisExpansion, f::BondFrame, mpo, i::Int, lch, rch, U0, V0,
                      side::Symbol, fold::Bool, rng::AbstractRNG)
    if fold
        skl, skr = folded_sketch_closures(expansion, f, mpo, i, lch, rch)
        n = max(8, cld(leg_dim(side === :left ? U0 : V0, side === :left ? 3 : 1), 4))
        return (skl, skr, sampled_leak(expansion, f, skl, skr, side, n, rng), 0)
    end
    X = apply_h_two_site(frame_theta(f), mpo, i, lch, rch)
    skl, skr = sketch_closures(expansion, f, () -> X)
    return (skl, skr, side === :left ? _left_leak(U0, X) : _right_leak(V0, X), 1)
end

"""
    _bond_measurements(Fm, mpo, fr, chain, stack, side; expansion, fold, nsample, rng) -> Vector

A MEASURING half-sweep's bonds, CONCURRENTLY (parallel plan (b)). With the frames fixed, the only
bond-to-bond dependency is the carry (`C = W[i]' K` on the left, `D = B Z[j]'` on the right): a
cheap sequential chain. Each bond's frame, two-site action and leak (~1/3 of a measuring sweep,
`_bond_action`) then depend on nothing else, so they run one task per bond, each with its own RNG
drawn in bond order from `rng`. Entry `i` (a bond of `side`) is `(K, leak, nh2, sleak)`, `K` the
carry the sequential loop would have built there; `sleak = 0` unless `nsample > 0`.
"""
function _bond_measurements(Fm::SymMPS, mpo, fr::KLFrames, chain, stack, side::Symbol;
                            expansion::BasisExpansion, fold::Bool, nsample::Int,
                            rng::AbstractRNG)
    L, c = length(Fm), fr.c
    bonds = side === :left ? (1:c) : ((L - 1):-1:c)
    carry = Vector{Any}(undef, L - 1)
    X = nothing
    for i in bonds
        if side === :left
            carry[i] = X === nothing ? Fm[i] : to_concrete!(contract(X, (2,), Fm[i], (1,)))
            X = to_concrete!(contract(fr.W[i]', (1, 2), carry[i], (1, 2)))
        else
            carry[i] = X === nothing ? Fm[i + 1] :
                       to_concrete!(contract(Fm[i + 1], (3,), X, (1,)))
            X = to_concrete!(contract(carry[i], (2, 3), fr.Z[i]', (2, 3)))
        end
    end
    rngs = Vector{Any}(undef, L - 1)
    for i in bonds
        rngs[i] = MersenneTwister(rand(rng, UInt64))
    end
    function one(i)
        f, a = if side === :left
            f = _frame_onto_left(fr.W[i], carry[i], Fm[i + 1])
            (f, _bond_action(expansion, f, mpo, i, chain[i], right_channels(stack, i + 2),
                             fr.W[i], nothing, :left, fold, rngs[i]))
        else
            f = _frame_onto_right(Fm[i], carry[i], fr.Z[i])
            (f, _bond_action(expansion, f, mpo, i, left_channels(stack, i), chain[i], nothing,
                             fr.Z[i], :right, fold, rngs[i]))
        end
        skl, skr, leak, nh2 = a
        sl = nsample > 0 ? sampled_leak(expansion, f, skl, skr, side, nsample, rngs[i]) : 0.0
        return (K = carry[i], leak = leak, nh2 = nh2, sleak = sl)
    end
    tasks = Dict(i => Threads.@spawn(one(i)) for i in bonds)
    out = Vector{Any}(undef, L - 1)
    for i in bonds
        out[i] = fetch(tasks[i])
    end
    return out
end

"How many charge sectors `new` carries on `leg` that `old` did not (plan §3.5)."
function _new_sectors(old, new, leg::Int)
    have = Set(first(s) for s in old.spaces[leg])
    return count(s -> !(first(s) in have), new.spaces[leg])
end

"""
    _frontier_expand(f, mpo, i, lch, rch, side, ckw, ledger, rng) -> CBEExpansion

A frontier bond's CBE (plan T4.1): `side`'s frame grows by [`expand_frame`](@ref) with the
step's strategy `ckw.expansion`, exactly as in the half-sweeps, so the Krylov stage after it
starts from those directions. `ckw` carries `cbe_bug_step!`'s expansion settings.
"""
function _frontier_expand(f::BondFrame, mpo, i::Int, lch, rch, side::Symbol,
                          ckw::NamedTuple, ledger::AdaptiveLedger, rng::AbstractRNG)
    return expand_frame(ckw.expansion, f, mpo, i, lch, rch, side; ledger = ledger, rng = rng,
                         alloc = get(ckw, :alloc, nothing), workcap = get(ckw, :workcap, 0),
                         growth = get(ckw, :growth, ADAPTIVE_GROWTH),
                         exact = get(ckw, :exact, false), version = i,
                         cap_order = get(ckw, :cap_order, :sector),
                         dd_width = get(ckw, :dd_width, 0),
                         dd_policy = get(ckw, :dd_policy, :adaptive),
                         dd_memory = get(ckw, :dd_memory, nothing))
end

# Tag every bond entry a frontier sweep logged, after `_log_bond!` wrote it.
_tag_last!(lg::AdaptiveLedger, extra::NamedTuple) =
    (lg.bond_log[end] = merge(lg.bond_log[end], extra); lg)

# ── from src/RSVDCBEBondUpdate/bug/krylov_closure_loop.jl ──

# krylov_closure_loop.jl: the closure loop of `order = p` steps (from `krylov_closure.jl`, included
# right after it).

"""
    _krylov_closure!(fr, psi, mpo, S0, tau, alloc, ckw; order, workcap, nrm, parallel, rng,
                     ledger, include_zero = false, maxpass, checks = false) -> NamedTuple

Close the frontiers `j = 0..order-1` of the frames `fr` in place, for `psi` with root coefficients
`S0`. `checks = false` (default) runs one enriching sweep per frontier, each rebuilt in the frames
the previous one grew, and each sweep measures its own leaks. `checks = true` first measures every
frontier on the incoming frames ([`_measure_frontiers`](@ref)) and returns `fr` untouched if all
close; otherwise it enriches and then measures `j = 0` on the final frames. Non-convergence is
`:closure`, never a silent pass; `capped` marks a stop at the working-rank limit
([`_closure_loop!`](@ref)).
"""
function _krylov_closure!(fr::KLFrames, psi::SymMPS, mpo, S0, tau::ComplexF64,
                          alloc::BasisAllocation, ckw::NamedTuple; order::Int,
                          workcap::Int = 0, nrm::Float64 = 1.0, parallel::Bool = false,
                          rng::AbstractRNG = MersenneTwister(0x5EED),
                          ledger::Union{Nothing, AdaptiveLedger} = nothing,
                          include_zero::Bool = false,
                          maxpass::Int = _closure_maxpass(order), fold::Bool = false,
                          checks::Bool = false)
    t0 = time_ns()
    lg = AdaptiveLedger()
    lg.order = order
    sweepkw = (workcap = workcap, nrm = nrm, parallel = parallel, rng = rng, fold = fold)
    checks || return _single_pass_closure!(fr, psi, mpo, S0, tau, alloc, ckw, lg, t0;
                                           order = order, include_zero = include_zero,
                                           ledger = ledger, sweepkw...)
    # CHECK FIRST: what the incoming frames (the ordinary CBE + local-Krylov half-sweeps)
    # already capture of the global Krylov sequence. If they close every frontier, no closure
    # sweep runs and the frames stay exactly as the half-sweeps built them.
    pre = _measure_frontiers(fr, psi, mpo, S0, tau, alloc, ckw, order; sweepkw...)
    merge_ledger!(lg, pre.ledger)
    nroot = Ref(pre.n_root)
    S = Ref{Any}(S0)
    final_leak = Dict{Int, Float64}(j - 1 => pre.leak[j] for j in 1:order)
    gen0 = fr.gen
    res = if pre.met && (pre.met0 || !include_zero)
        (passes = 0, converged = true, exhausted = false, capped = false, log = Any[])
    else
        H0 = Ref{Any}(_root_generator(mpo, fr))
        applyB = x -> (nroot[] += 1; apply_zero_site(H0[], x))
        frontier = j -> KrylovFrontier(j == 0 ? S[] :
                                       _projected_frontiers(applyB, S[], tau, j)[j], j, fr.gen)
        # ⚠ SPECULATIVE FINAL MEASUREMENT (>= _FRONTIER_THREADS threads). A later pass usually
        # adds nothing and is then the last; its last sweep cannot change the frames unless it
        # adds. So `j = 0` is measured CONCURRENTLY on a snapshot of the frames, and kept only
        # if the loop ends in that generation -- otherwise it is discarded and redone below.
        spec = Ref{Any}(nothing)
        r0kw = (workcap = workcap, nrm = nrm, parallel = parallel, fold = fold, enrich = false)
        sweep! = (F, j, pass) -> begin
            if _frontier_parallel(parallel) && pass >= 2 && j == order - 1
                spec[] === nothing || fetch(spec[].task)
                snap = KLFrames(copy(fr.W), copy(fr.Z), fr.c, fr.lenv, fr.renv, fr.gen,
                                fr.lcache, fr.rcache, fr.eshift)
                F0 = KrylovFrontier(S[], 0, snap.gen)
                rs0 = MersenneTwister(rand(rng, UInt64))
                spec[] = (gen = snap.gen, task = Threads.@spawn(
                    _frontier_sweep!(snap, psi, mpo, F0, tau, alloc, ckw; r0kw...,
                                     rng = rs0, pass = pass + 1)))
            end
            r = _frontier_sweep!(fr, psi, mpo, F, tau, alloc, ckw; sweepkw..., pass = pass)
            merge_ledger!(lg, r.ledger)
            final_leak[j] = r.leak
            if r.added > 0
                H0[] = _root_generator(mpo, fr)
                S[] = _project_root(psi, fr)
            end
            r
        end
        loop = _closure_loop!(frontier, sweep!, order; maxpass = maxpass,
                              include_zero = include_zero)
        sr = spec[] === nothing ? nothing : (gen = spec[].gen, r = fetch(spec[].task))
        if fr.gen == gen0
            # Nothing grew: the frames and `S0` are the ones `pre` measured `j = 0` on, so its
            # measurement IS the final one (`final_leak[0]` already holds it).
        elseif sr !== nothing && sr.gen == fr.gen
            merge_ledger!(lg, sr.r.ledger)
            final_leak[0] = sr.r.leak
        else
            r0 = _frontier_sweep!(fr, psi, mpo, frontier(0), tau, alloc, ckw; r0kw...,
                                  rng = rng, pass = loop.passes + 1)
            merge_ledger!(lg, r0.ledger)
            final_leak[0] = r0.leak
        end
        loop
    end
    lg.n_passes = res.passes
    lg.leak = [final_leak[j] for j in 0:(order - 1)]
    lg.defect = sum(lg.leak[j + 1] / (j + 1) for j in 0:(order - 1))
    lg.n_root_frontier = nroot[]
    res.converged || _unmet!(lg, :closure)
    lg.t_closure = (time_ns() - t0) / 1e9
    ledger === nothing || merge_ledger!(ledger, lg)
    return (S0 = S[], passes = res.passes, converged = res.converged,
            exhausted = res.exhausted, capped = res.capped, log = res.log, ledger = lg,
            n_root = nroot[],
            n_h1 = lg.n_h1_closure)
end

"""
    _single_pass_closure!(fr, psi, mpo, S0, tau, alloc, ckw, lg, t0; order, include_zero,
                          ledger, sweepkw...) -> NamedTuple

[`_krylov_closure!`](@ref) with `checks = false`: one enriching sweep per frontier, each rebuilt
in the frames the previous sweep grew, and nothing measured before or after. Same return fields.
"""
function _single_pass_closure!(fr::KLFrames, psi::SymMPS, mpo, S0, tau::ComplexF64,
                               alloc::BasisAllocation, ckw::NamedTuple, lg::AdaptiveLedger,
                               t0::UInt64; order::Int, include_zero::Bool,
                               ledger::Union{Nothing, AdaptiveLedger}, sweepkw...)
    nroot = Ref(0)
    S = Ref{Any}(S0)
    H0 = Ref{Any}(_root_generator(mpo, fr))
    applyB = x -> (nroot[] += 1; apply_zero_site(H0[], x))
    leak = fill(NaN, order)                      # NaN: not measured (j = 0 unless include_zero)
    log = Any[]
    met, capped = true, false
    for j in (include_zero ? 0 : 1):(order - 1)
        core = j == 0 ? S[] : _projected_frontiers(applyB, S[], tau, j)[j]
        r = _frontier_sweep!(fr, psi, mpo, KrylovFrontier(core, j, fr.gen), tau, alloc, ckw;
                             sweepkw..., enrich = true, pass = 1)
        merge_ledger!(lg, r.ledger)
        push!(log, (pass = 1, frontier = j, added = r.added, leak = r.leak, met = r.met))
        leak[j + 1] = r.leak                     # before this sweep's enrichment
        met &= r.met
        capped |= r.capped
        if r.added > 0
            H0[] = _root_generator(mpo, fr)
            S[] = _project_root(psi, fr)
        end
    end
    lg.n_passes = 1
    lg.leak = leak
    lg.defect = sum((leak[j + 1] / (j + 1) for j in 0:(order - 1) if !isnan(leak[j + 1]));
                    init = 0.0)
    lg.n_root_frontier = nroot[]
    met || _unmet!(lg, :closure)
    lg.t_closure = (time_ns() - t0) / 1e9
    ledger === nothing || merge_ledger!(ledger, lg)
    return (S0 = S[], passes = 1, converged = met, exhausted = false, capped = capped && !met,
            log = log, ledger = lg, n_root = nroot[], n_h1 = lg.n_h1_closure)
end
