# The half-sweep Krylov stage: the Saad weight estimate, the frame extension, `_krylov_enrich`, and
# the frame merge and ledger fold around it. The local recurrence it drives (`_krylov_basis`) is in
# `krylov_recurrence.jl`, included right after this file.

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

"""
    _kry_weight(Hm, krylov_step, tau; state_seed = true) -> Float64

The weight the newest Krylov vector `v_{k+1}` carries in `exp(tau*H) v_1`, from the Arnoldi
Hessenberg `Hm` after `k = krylov_step` steps: `|tau * beta_k * [phi_1(tau*H_k)]_{k,1}|`, read
from one `(k+1)x(k+1)` exponential. This is Saad's a posteriori error of the `k`-dimensional
approximation, so it is also the closure test: below the allowance, no later direction matters more.

`state_seed` is false when `v_1` is a closure frontier rather than the state's own tensor; it
matters only for a non-unitary propagator.

The propagator is unitary when `tau*H_k` is skew-Hermitian, not whenever `real(tau) == 0`: a
no-jump generator `H - (i/2)Q` at `tau = -im*dt` contracts, and its weight must be relative to the
decayed state, not to the start norm.
"""
function _kry_weight(
    Hm::Matrix{ComplexF64},
    krylov_step::Int,
    tau::ComplexF64;
    state_seed::Bool = true,
)
    # `tau * H_k` bordered by the subdiagonal entry `tau * beta_k`: its exponential's first
    # column carries both the approximation (rows 1..k) and the newest vector's weight (row k+1).
    augmented = zeros(ComplexF64, krylov_step + 1, krylov_step + 1)
    augmented[1:krylov_step, 1:krylov_step] .= tau .* @view Hm[1:krylov_step, 1:krylov_step]
    augmented[krylov_step+1, krylov_step] = tau * Hm[krylov_step+1, krylov_step]
    leading_block = @view augmented[1:krylov_step, 1:krylov_step]
    norm(leading_block + leading_block') <= _KRY_SKEW_TOL * norm(leading_block) &&
        return abs(exp(augmented)[krylov_step+1, 1])
    # Non-unitary (imaginary time, Lindblad, no-jump loss): weigh relative to the state's own
    # growth `g = ||exp(tau*H_k) e_1||`, which is `exp(dbeta*|E|)` unshifted and overflows at large
    # dbeta. The `shift` cancels in that ratio exactly.
    shift = maximum(real, eigvals(augmented[1:krylov_step, 1:krylov_step]))
    shifted_exp = exp(augmented - shift * I)
    state_seed && return abs(shifted_exp[krylov_step+1, 1]) /
           max(norm(@view shifted_exp[1:krylov_step, 1]), floatmin(Float64))
    # A closure frontier is not the state: it decays under the shifted generator, so dividing by its
    # own growth inflated its directions by `e^{dbeta h11}`, enough to block a converged step. Its
    # weight is the absolute one, `|shifted_exp[k+1,1]| e^shift`, in the shifted frame, which never
    # under-weights in imaginary time (the shift is the energy there).
    frontier_weight = abs(shifted_exp[krylov_step+1, 1])
    return frontier_weight == 0 ? 0.0 : exp(log(frontier_weight) + shift)
end

"""
    _extend_frame(frame, residual, direction, side, weight, allowance; room)
        -> (frame, n_added, fits)

Append to `frame` the local directions of `residual = (I - QQ') direction` (`Q` = `frame`) that
the allowance requires: the smallest number whose weighted discarded tail `weight * sqrt(sum s^2)`
fits (Test A's rule, `_tail_rank`). Orthogonalised within each charge sector by the block SVD,
cut only at the numerical floor `_rank_cut`, re-orthogonalised against `frame` before the direct
sum -- the same guard `cbe_expand` applies to its own appended block.

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
        side === :left ? _reortho_left(frame, to_concrete(res.U)) :
        _reortho_right(frame, to_concrete(permutedims(res.U, (3, 1, 2))))
    # roundoff in a saturated sector must not become a column (`_cap_to_local_room`)
    new_block = _cap_to_local_room(frame, new_block, side)
    new_block === nothing && return (frame, 0, fits)
    fl = side === :left ? 3 : 1
    # Direct sum on the frame leg, the new block relabelled with the frame's own bond tag.
    grown_frame = to_concrete(
        oplus([frame, to_concrete(setitag(new_block, fl, frame.inds[fl].itags))], (fl,)),
    )
    return (_check_frame(grown_frame, side, "_extend_frame"), leg_dim(new_block, fl), fits)
end

"""
    _krylov_enrich(seed, applyH, start_frame, side, tau, alloc; workcap = 0,
                   maxdepth = _KRY_MAXDEPTH, amp = 1.0, state_seed = true, fixed_depth = 0,
                   recycled = ())
        -> (frame, stats)

Grow the frame `start_frame` by the Krylov directions of `applyH` from `seed` that the step needs.
Each direction is scored by `weight * ||(I - QQ') v_k||` (`Q` the frame so far): `eta_1` is Test C,
`eta_{k>=2}` Test D. Scores above `alloc.krylov` are appended. A small `eta_1` is not a stop
(`H e1 = e2`, `H e2 = e1 + e3` with `start_frame = [e1, e2]`). `capped` with `met = false` means
the work ceiling or depth bound stopped the search with weight left. `amp` is the seed's norm
relative to the state. `stats.basis` and `stats.weights` are the Arnoldi vectors and their
weights, the last vector the active frontier: what a later round recycles. With `recycled` only
the complement of the previous solve's input is extended ([`_krylov_basis`](@ref)).

`fixed_depth = m > 0` is the conventional-control ablation: exactly `m` steps,
every new local direction appended (numerical floor only), no weight stop; `met` then reports
whether the weight had fallen to the allowance by depth `m`.
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
    fixed_depth::Int = 0,
    recycled = (),
)
    side in (:left, :right) || throw(ArgumentError("side must be :left or :right"))
    perp = side === :left ? perp_component : perp_component_right
    fl = side === :left ? 3 : 1
    fixed = fixed_depth > 0
    fixed && (maxdepth = fixed_depth)
    allow = fixed ? 0.0 : alloc.krylov
    frame = start_frame
    eta_c, eta_d, added, added_d = 0.0, 0.0, 0, 0
    # Tests C/D on one direction, growing `frame` in place; false = capped.
    function score(direction, weight, first::Bool)
        residual = to_concrete(perp(frame, direction))
        eta = weight * norm(residual)
        first ? (eta_c = eta) : (eta_d = max(eta_d, eta))
        eta > allow || return true
        # The ceiling bounds the frame WIDTH, not the number of extensions: a block may not carry
        # the frame past it either.
        room = workcap > 0 ? workcap - leg_dim(frame, fl) : typemax(Int)
        room <= 0 && return false
        frame, nadd, fits =
            _extend_frame(frame, residual, direction, side, weight, allow; room = room)
        added += nadd
        first || (added_d += nadd)
        # an allowance of 0 never "fits" (the floor's tail): only the adaptive stop caps here
        return fits || fixed
    end
    kb = _krylov_basis(
        seed,
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
        frame,
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
    _merge_frame(frame, other_frame, side) -> grown frame

`frame` grown by the part of `other_frame` it does not span, cut only at the numerical floor (the
`krylov_reuse = false` ablation: the CBE directions join after the Krylov stage instead of
seeding it).
"""
function _merge_frame(frame, other_frame, side::Symbol)
    residual = to_concrete(
        side === :left ? perp_component(frame, other_frame) :
        perp_component_right(frame, other_frame),
    )
    # `other_frame` inside `frame` (no CBE direction this step): Telum drops the cancelled blocks
    # and its SVD of a blockless tensor throws (hq34b 17436300)
    norm(residual) == 0 && return frame
    return first(_extend_frame(frame, residual, other_frame, side, 1.0, 0.0))
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
