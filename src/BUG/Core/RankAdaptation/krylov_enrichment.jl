# Moved from BUG-Julia@b155550:src/RSVDCBEBondUpdate/bug/krylov_enrich.jl, src/RSVDCBEBondUpdate/bug/krylov_closure.jl, src/RSVDCBEBondUpdate/bug/krylov_closure_loop.jl.
# krylov_enrichment.jl: the half-sweep Krylov stage -- weight estimate, frame extension, `_krylov_enrich`, and the frame merge / ledger fold around it.

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
    A[k+1, k] = tau * Hm[k+1, k]
    Ak = @view A[1:k, 1:k]
    norm(Ak + Ak') <= _KRY_SKEW_TOL * norm(Ak) && return abs(exp(A)[k+1, 1])
    # Non-unitary (imaginary time, Lindblad, no-jump loss): weigh relative to the state's own
    # growth `g = ||exp(tau*H_k) e_1||`, which is `exp(dbeta*|E|)` unshifted and overflows at large dbeta. The
    # shift `s` cancels in that ratio exactly.
    s = maximum(real, eigvals(A[1:k, 1:k]))
    E = exp(A - s * I)
    state_seed && return abs(E[k+1, 1]) / max(norm(@view E[1:k, 1]), floatmin(Float64))
    # A closure frontier is not the state: it decays under the shifted generator, so dividing by its
    # own growth inflated its directions by `e^{dbeta h11}`, enough to block a converged step. Its
    # weight is the absolute one, `|E[k+1,1]| e^s`, in the shifted frame, which never under-weights
    # in imaginary time (the shift is the energy there).
    w = abs(E[k+1, 1])
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
function _extend_frame(
    Q,
    R,
    v,
    side::Symbol,
    weight::Float64,
    allowance::Float64;
    room::Int = typemax(Int),
)
    legs = side === :left ? (1, 2) : (2, 3)
    cut = _rank_cut(0.0, v, norm(R))
    res = svd(R, legs, _TAG_K, _TAG_KR; cutoff = cut, get_lists = true)
    nk = length(res.kept_list)
    k, tail = _tail_rank(
        res.kept_list,
        weight,
        allowance,
        min(nk, room);
        base = _sq_weight(res.trunc_list),
    )
    fits = tail <= allowance
    k == 0 && return (Q, 0, fits)
    k < nk && (res = svd(R, legs, _TAG_K, _TAG_KR; Nkeep = k, cutoff = cut))
    U =
        side === :left ? _reortho_left(Q, to_concrete(res.U)) :
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
function _krylov_enrich(
    v0,
    applyH,
    Q0,
    side::Symbol,
    tau::ComplexF64,
    alloc::BasisAllocation;
    workcap::Int = 0,
    maxdepth::Int = _KRY_MAXDEPTH,
    amp::Float64 = 1.0,
    state_seed::Bool = true,
    fixed_depth::Int = 0,
    recycled = (),
)
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
    kb = _krylov_basis(
        v0,
        applyH,
        tau,
        alloc.krylov,
        score;
        recycled = recycled,
        maxdepth = maxdepth,
        amp = amp,
        state_seed = state_seed,
        fixed = fixed,
    )
    return (
        Q,
        (
            eta_c = eta_c,
            eta_d = eta_d,
            added = added,
            n_h1 = kb.n_h1,
            d_after_c = eta_c <= alloc.krylov && added_d > 0,
            capped = kb.capped,
            met = kb.closed && !kb.capped,
            basis = kb.basis,
            weights = kb.weights,
            n_recycled = kb.n_recycled,
            rec_closed = kb.rec_closed,
            rec_overlap = kb.rec_overlap,
            rec_defect = kb.rec_defect,
        ),
    )
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
function _record_krylov!(ledger::Union{Nothing,AdaptiveLedger}, st, alloc::BasisAllocation)
    ledger === nothing && return nothing
    ledger.max_krylov_c = max(ledger.max_krylov_c, st.eta_c)
    ledger.max_krylov_d = max(ledger.max_krylov_d, st.eta_d)
    ledger.max_krylov = max(ledger.max_krylov, st.eta_c, st.eta_d)
    ledger.n_krylov += st.added
    ledger.n_h1 += st.n_h1
    st.eta_c <= alloc.krylov && (ledger.n_c_pass += 1)
    st.d_after_c && (ledger.n_d_after_c += 1)
    st.capped && (ledger.workcap_reached = true)
    st.met || _unmet!(ledger, :krylov)
    return nothing
end
