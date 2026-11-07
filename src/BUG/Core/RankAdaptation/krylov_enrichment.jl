# The half-sweep Krylov stage: the Krylov weight (a residual integral), the frame extension,
# `_krylov_enrich`, and the ledger fold after it. The local recurrence it
# drives (`_krylov_basis`) is in `krylov_recurrence.jl`, included right after this file.

# ── THE HALF-SWEEP KRYLOV STAGE OF `cbe_bug_step!` (Tests C and D) ───────────────────────────

# Bond tags of the left (`W`) and right (`Z`) Krylov frames at bond `bond`.
_tagW(bond::Int) = "cbeW,$bond"
_tagZ(bond::Int) = "cbeZ,$bond"

# The residual below which the Krylov space has genuinely CLOSED -- an invariant subspace, not a
# convergence judgement.
const _KRY_BREAKDOWN = 1e-13

# ── STAGE 3: residual-driven K/L Krylov enrichment (Tests C and D) ──
# Krylov vectors live on one-site tensors, but the projector `I - QQ'` acts on the frame (an
# isometry on `(link, site)`): a vector adds only through the part of its local range the frame
# lacks. `_krylov_enrich`'s `start_frame` is the CBE output `U_ex`, so old directions are kept
# verbatim; the seed `K_ex` already lies in `U_old`.

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

"Most Simpson panels of one residual integral (a numerical cap: `2 x 4096` nodes)."
const _RESIDUAL_MAX_PANELS = 4096

"""
    _kry_weight(Hm, krylov_step, tau; state_seed = true) -> Float64

The weight the newest Krylov vector `v_{k+1}` carries in `exp(tau*H) v_1`, from the Arnoldi
Hessenberg `Hm` after `k = krylov_step` steps, and so the closure test: below the allowance, no
later direction matters more. With `f(s) = [exp(s tau H_k)]_{k,1}` and `beta_k = Hm[k+1, k]` it is
the residual integral `|tau| beta_k int_0^1 |f(s)| ds`, the variation-of-constants bound on
`||exp(tau A) v_1 - V_k exp(tau H_k) e_1||` for a unitary propagator. Saad's scalar
`|tau beta_k int_0^1 f(s) ds|` is never larger, and cancels when `f` turns its phase over the step:
`A = -i[2pi 1; 1 0]`, `v = e1`, `tau = 1` gives it 4e-17 against a true error of 0.155.

Three normalisations: absolute for a unitary propagator and for a closure frontier
(`state_seed = false`), relative to the state's own growth `||exp(tau H_k) e_1||` for the state's
seed. The propagator is unitary when `tau*H_k` is skew-Hermitian, not whenever `real(tau) == 0`: a
no-jump generator `H - (i/2)Q` at `tau = -im*dt` contracts, and its weight must be relative to the
decayed state, not to the start norm. A closure frontier is not the state: it decays under the
shifted generator, so dividing by its own growth would inflate its directions by `e^{dbeta h11}`.

The integral is [`_residual_integral`](@ref)'s.
"""
function _kry_weight(
    Hm::Matrix{ComplexF64},
    krylov_step::Int,
    tau::ComplexF64;
    state_seed::Bool = true,
)
    beta = abs(Hm[krylov_step+1, krylov_step])
    beta == 0 && return 0.0
    # |f(s)| is the last entry of exp(s tau H_k) e_1; the defect it leaves carries beta_k
    return beta * _residual_integral(
        column -> abs(column[krylov_step]),
        tau .* Hm[1:krylov_step, 1:krylov_step],
        tau;
        state_seed = state_seed,
    )
end

"""
    _residual_integral(defect_norm, generator, tau; state_seed = true) -> Float64

`|tau| int_0^1 defect_norm(exp(s M) e_1) ds` for `M = generator` (`tau` times a projected
operator), in the normalisations of [`_kry_weight`](@ref): absolute for a unitary propagator and
for a closure frontier, relative to the state's own growth `||exp(M) e_1||` for the state's seed.
`defect_norm` maps the coefficient column `c(s) = exp(s M) e_1` to the norm of the defect it
leaves, and must be homogeneous of degree one (`defect_norm(a c) = |a| defect_norm(c)`).

Composite Simpson on an even number of intervals with `||M - sigma||_1 / intervals <= 1/4`, so the
phase is resolved; `sigma` is the largest real part of `M`'s spectrum (0 when unitary), the
integrand is evaluated on `exp(s (M - sigma))` so no exponential overflows, and the sum is taken
in log space.
"""
function _residual_integral(
    defect_norm,
    generator::AbstractMatrix,
    tau::ComplexF64;
    state_seed::Bool = true,
)
    unitary = norm(generator + generator') <= _KRY_SKEW_TOL * norm(generator)
    sigma = unitary ? 0.0 : maximum(real, eigvals(generator))
    shifted = generator - sigma * I
    intervals = 2 * clamp(ceil(Int, 2 * opnorm(shifted, 1)), 8, _RESIDUAL_MAX_PANELS)
    spacing = 1.0 / intervals
    propagator = exp(spacing .* shifted)
    column = zeros(ComplexF64, size(generator, 1))   # exp(s * shifted) e_1, from s = 0
    column[1] = 1
    log_terms = Vector{Float64}(undef, intervals + 1)
    for node = 0:intervals
        node > 0 && (column = propagator * column)
        simpson_weight = node == 0 || node == intervals ? 1.0 : (isodd(node) ? 4.0 : 2.0)
        magnitude = defect_norm(column)
        # the defect at s: e^{s sigma} defect_norm(exp(s shifted) e_1), weighted by Simpson's rule
        log_terms[node+1] =
            magnitude == 0 ? -Inf :
            log(simpson_weight * spacing / 3 * magnitude) + node * spacing * sigma
    end
    peak = maximum(log_terms)
    peak == -Inf && return 0.0
    log_weight = log(abs(tau)) + peak + log(sum(exp(term - peak) for term in log_terms))
    (unitary || !state_seed) && return exp(log_weight)
    # the state's own seed: relative to its growth ||exp(M) e_1|| = e^sigma ||exp(shifted) e_1||,
    # and `column` now holds exp(shifted) e_1
    return exp(log_weight - sigma - log(max(norm(column), floatmin(Float64))))
end

"""
    _extend_frame(frame, residual, direction, side, weight, allowance; room)
        -> (frame, n_added, fits)

Append to `frame` the local directions of `residual = (I - QQ') direction` (`Q` = `frame`) that
the allowance requires: the smallest number whose weighted discarded tail `weight * sqrt(sum s^2)`
fits (Test A's rule, `_tail_rank`). Orthogonalised within each charge sector by the block SVD,
cut only at the numerical floor `_rank_cut`, re-orthogonalised against `frame` before the direct
sum -- the same guard `dd_cbe_expand` applies to its own appended block (`_cap_to_local_room`).

`room` is what the work ceiling still allows on the frame leg: at most that many are
appended, and `fits = false` reports that the allowance needed more.
"""
function _extend_frame(
    frame,
    residual,
    direction,
    side::Symbol,
    weight::Float64,
    allowance::Float64;
    room::Int = typemax(Int),
)
    # The legs `svd` keeps on its `U`: the frame's (link, site) pair.
    legs = side === :left ? (1, 2) : (2, 3)
    cut = _rank_cut(0.0, direction, norm(residual))
    res = svd(residual, legs, _TAG_K, _TAG_KR; cutoff = cut, get_lists = true)
    nk = length(res.kept_list)
    keep_rank, tail = _tail_rank(
        res.kept_list,
        weight,
        allowance,
        min(nk, room);
        base = _sq_weight(res.trunc_list),
    )
    fits = tail <= allowance
    keep_rank == 0 && return (frame, 0, fits)
    keep_rank < nk && (res = svd(residual, legs, _TAG_K, _TAG_KR; Nkeep = keep_rank, cutoff = cut))
    # A right frame carries its bond leg first, so the SVD's new leg moves to the front.
    new_block =
        side === :left ? _reortho_left(frame, TLArray(res.U)) :
        _reortho_right(frame, TLArray(permutedims(res.U, (3, 1, 2))))
    # roundoff in a saturated sector must not become a column (`_cap_to_local_room`)
    new_block = _cap_to_local_room(frame, new_block, side)
    new_block === nothing && return (frame, 0, fits)
    fl = side === :left ? 3 : 1
    # Direct sum on the frame leg, the new block relabelled with the frame's own bond tag.
    grown_frame = TLArray(
        oplus([frame, TLArray(setitag(new_block, fl, Telum.inds(frame)[fl].itags))], (fl,)),
    )
    return (_check_frame(grown_frame, side, "_extend_frame"), leg_dim(new_block, fl), fits)
end

"""
    _append_ranked(frame, candidates, side, allowance, workcap) -> (frame, n_added, fits)

The ranked selection of one local Krylov stage (`alloc.krylov_rank > 0`). `candidates` are the
weighted residuals `weight * (I - QQ') v_k` of every direction the chain found above the allowance,
all outside the same `frame`. One SVD of their direct sum ranks the local directions by their weight
in the step, and the fewest top ones whose discarded tail fits `allowance` join the frame. The
sequential rule instead appends each vector's own directions in chain order, each against its own
allowance, until the work ceiling stops it; on a nonlocal generator that fills the stage frame to
the ceiling, far above the rank the step keeps. `fits = false`: the work ceiling cut the selection
with more than `allowance` left.
"""
function _append_ranked(frame, candidates, side::Symbol, allowance::Float64, workcap::Int)
    isempty(candidates) && return (frame, 0, true)
    fl = side === :left ? 3 : 1
    room = workcap > 0 ? workcap - leg_dim(frame, fl) : typemax(Int)
    # the residuals' own bond leg (to the rest of the chain) is the leg the block is summed over
    block = length(candidates) == 1 ? candidates[1] : TLArray(oplus(candidates, (fl,)))
    room <= 0 && return (frame, 0, norm(block) <= allowance)
    # the weights are folded into the block, so the tail rule runs at weight 1
    return _extend_frame(frame, block, block, side, 1.0, allowance; room = room)
end

"""
    _krylov_enrich(seed, applyH, start_frame, side, tau, alloc; workcap = 0,
                   maxdepth = _KRY_MAXDEPTH, amp = 1.0, state_seed = true)
        -> (frame, stats)

Grow the frame `start_frame` by the Krylov directions of `applyH` from `seed` that the step needs.
Each direction is scored by `weight * ||(I - QQ') v_k||` (`Q` the frame so far): `eta_1` is Test C,
`eta_{k>=2}` Test D. Scores above `alloc.krylov` are appended. A small `eta_1` is not a stop
(`H e1 = e2`, `H e2 = e1 + e3` with `start_frame = [e1, e2]`). `capped` with `met = false` means
the work ceiling or depth bound stopped the search with weight left. `amp` is the seed's norm
relative to the state. `stats.basis` and `stats.weights` are the Arnoldi vectors and their
weights, the last vector the active frontier.

`alloc.krylov_rank > 0` ranks instead of appending in chain order: every direction above the
allowance is only collected while the chain runs (the closure test is unchanged), and
[`_append_ranked`](@ref) keeps the top ones within `alloc.krylov_rank` at the end.
"""
function _krylov_enrich(
    seed,
    applyH,
    start_frame,
    side::Symbol,
    tau::ComplexF64,
    alloc::BasisAllocation;
    workcap::Int = 0,
    maxdepth::Int = _KRY_MAXDEPTH,
    amp::Float64 = 1.0,
    state_seed::Bool = true,
)
    side in (:left, :right) || throw(ArgumentError("side must be :left or :right"))
    perp = side === :left ? perp_component : perp_component_right
    fl = side === :left ? 3 : 1
    allow = alloc.krylov
    ranked = alloc.krylov_rank > 0
    # the ranked rule's collected residuals, and how many came from Test D directions
    candidates, n_test_d_candidates = Any[], 0
    frame = start_frame
    eta_c, eta_d, added, added_d = 0.0, 0.0, 0, 0
    # Tests C/D on one direction, growing `frame` in place; false = capped.
    function score(direction, weight, first::Bool)
        residual = TLArray(perp(frame, direction))
        eta = weight * norm(residual)
        first ? (eta_c = eta) : (eta_d = max(eta_d, eta))
        eta > allow || return true
        if ranked
            # `frame` stays `start_frame`: every candidate is measured against the same frame
            push!(candidates, TLArray(weight * residual))
            first || (n_test_d_candidates += 1)
            return true
        end
        # The ceiling bounds the frame WIDTH, not the number of extensions: a block may not carry
        # the frame past it either.
        room = workcap > 0 ? workcap - leg_dim(frame, fl) : typemax(Int)
        room <= 0 && return false
        frame, nadd, fits =
            _extend_frame(frame, residual, direction, side, weight, allow; room = room)
        added += nadd
        first || (added_d += nadd)
        return fits
    end
    kb = _krylov_basis(
        seed,
        applyH,
        tau,
        alloc.krylov,
        score;
        maxdepth = maxdepth,
        amp = amp,
        state_seed = state_seed,
    )
    capped = kb.capped
    if ranked
        frame, added, fits =
            _append_ranked(start_frame, candidates, side, alloc.krylov_rank, workcap)
        added_d = n_test_d_candidates > 0 ? added : 0
        capped |= !fits
    end
    return (
        frame,
        (
            eta_c = eta_c,
            eta_d = eta_d,
            added = added,
            n_h1 = kb.n_h1,
            d_after_c = eta_c <= alloc.krylov && added_d > 0,
            capped = capped,
            met = kb.closed && !capped,
            basis = kb.basis,
            weights = kb.weights,
        ),
    )
end

"Fold one `_krylov_enrich` verdict into the step ledger's Krylov diagnostics."
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
