# ── the two half-sweeps of `cbe_bug_step!` ──
# Both only build bases (they never evolve a site tensor) and share no data, so they run
# concurrently on their own gauges. `psi` stays L-canonical so `AL`, `AR` and the assembly read one
# gauge; `copy` is safe because every mutation replaces a slot.

"""
    _HalfAcc

Everything ONE half-sweep accumulates, held per half-sweep rather than in closure variables.

This exists because the half-sweeps can now run concurrently. Counters shared between them
(`nmv`, `epre`, the phase timers) are read-modify-write from two tasks, so as closure `Ref`s they
are a data race -- and a racy `+=` on a Float64 does not crash, it silently loses increments, so
the failure would show up as a profile that quietly under-reports and never as an error.
"""
mutable struct _HalfAcc
    nmv::Int
    ngrow::Int
    epre::Float64
    efnl::Float64
    peak::Int
    tcbe::Float64
    tkry::Float64
    tsplit::Float64
    tframe::Float64
    tenv::Float64
    kc::KernelCounters      # this half-sweep's kernel counters
end
# An empty accumulator: zero counts and timers, fresh kernel counters.
_HalfAcc() = _HalfAcc(0, 0, 0.0, 0.0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, KernelCounters())

"""
`build(args...)`, an environment build or push, charged to the half-sweep's `tenv` and
`kc.b_env`; returns the environment. The arguments are passed, not captured: a closure over the
sweep's reassigned `lch`/`rch` boxes them.
"""
function _env_timed!(acc::_HalfAcc, build, args...; kw...)
    environment, dt, db = _measure(() -> build(args...; kw...))
    acc.tenv += dt
    acc.kc.b_env += db
    return environment
end

"""
    _HalfSweepCtx

What both half-sweeps of one step READ and never write: the operator, `tau`, the generator shift,
the root bond `root_link`, the step's allocation, its basis-selection strategy (`expansion`) and that
strategy's settings (`exkw`), the work ceiling, the `krylov_reuse` and `krylov_depth` ablations,
the states the trial space must contain (`keep_states`, each admitted at its `keep_tols` entry)
and the `krylov_recycle` cache (`nothing` when off), which round 1 fills. `cap_keep`
(`work_ceiling_rule = :bond`): a tolerance-admitted kept state may only fill the frame up to the
bond's work ceiling, so no frame exceeds it; exactly kept states (tol 0, e.g. Heston's mass) stay
a containment guarantee and are never cut.
"""
struct _HalfSweepCtx{M,E,A,X<:BasisExpansion,K,S,R}
    mpo::M
    tau::ComplexF64
    eshift::E
    root_link::Int
    alloc::A
    expansion::X
    exkw::K
    wcap::WorkCeiling
    krylov_reuse::Bool
    krylov_depth::Int
    keep_states::S
    keep_tols::Vector{Float64}   # per kept state: 0 = contained exactly, > 0 = `_union_fresh`'s tol
    recycle::R
    cap_keep::Bool
end

"""
Time one bond expansion into the half-sweep's own accumulator: `side`'s frame grows by the step's
strategy (`expand_frame`; DD-QR CBE by default) into the bond's own `ledger`;
`nn[bond]` records what it admitted. Returns the expansion.
"""
function _expand_bond!(
    acc::_HalfAcc,
    cx::_HalfSweepCtx,
    nn,
    frame,
    bond,
    lch,
    rch,
    side::Symbol,
    ledger,
    rng,
)
    start_time = time_ns()
    ex = expand_frame(
        cx.expansion,
        frame,
        cx.mpo,
        bond,
        lch,
        rch,
        side;
        cx.exkw...,
        workcap = _cap_at(cx.wcap, bond),
        ledger = ledger,
        rng = rng,
        kc = acc.kc,
        version = bond,
    )
    acc.tcbe += (time_ns() - start_time) / 1e9
    acc.epre = max(acc.epre, ex.err_pre)
    acc.efnl = max(acc.efnl, ex.err_fnl)
    nn[bond] = side === :left ? ex.n_new_l : ex.n_new_r
    return ex
end

# Peak is tracked per half-sweep and summed, not scanned across the frame arrays: scanning reads
# arrays the other half-sweep is writing concurrently, and costs O(L) walks per bond. The reported
# number is therefore an upper bound, counting both halves' frames as live at once, which keeps the
# two arms' working sets comparable when they run in parallel.
function _peak!(acc::_HalfAcc, held::Int, ts...)
    acc.peak = max(acc.peak, held + sum(tensor_elements(tensor) for tensor in ts; init = 0))
    return nothing
end

# A state inside the span of a frame with isometry defect d leaves a residual of size d ||X||,
# above `_preselect_*`'s 100 eps floor; normalised, it is a full column that still lies in the
# span, and each later union widens the frame again (qt1f, QTT Heston order-1 fallback: a right
# frame 48 wide on a 32-dim bond, ||Q'Q - I|| = 4, mass 1 -> 5.5). A direction is appended only
# when one more projection keeps at least this share of it (twice is enough), tested on the first
# projection's basis itself: `_preselect_*`'s RepairOrtho re-SVDs the second projection relative
# to its own largest value, which turns the ~eps ||X|| rounding of an in-span residual into full
# columns genuinely outside the frame (qt1g/qt1h: a 13-wide frame grew to 16 from a state inside
# it). The first cut is the preselection's numerical floor, measured against the whole state.
const _UNION_KEEP = 0.5

"""
    _union_fresh(frame, candidates, side; tol = 0, room = typemax(Int)) -> directions or nothing

Orthonormal directions of `candidates` outside the frame `frame` that survive a second
projection, or `nothing`.

`tol = 0` admits every direction above the numerical floor: `candidates` lies in the new span
exactly (the MPDO identity, Heston's mass state). `tol > 0` admits only directions carrying more
than about `tol * ||candidates||` of it (the SVD cutoff is relative to the largest singular value
of the normalised complement, so the threshold sits between `tol ||candidates|| / sqrt(rank)` and
`tol ||candidates||`, never above): the DD acceptance rule applied to a predictor state, which is
how the QTT step's stage fields enter its basis. `room` bounds how many directions are appended,
the strongest first (the work ceiling's headroom, `_keep_room`); `room <= 0` admits none.
"""
function _union_fresh(frame, candidates, side::Symbol; tol::Float64 = 0.0, room::Int = typemax(Int))
    room <= 0 && return nothing
    left = side === :left
    perp = left ? perp_component : perp_component_right
    # The legs the frame spans, and the tags of the new bond leg's two ends.
    legs, left_tag, right_tag = left ? ((1, 2), _TAG_L, _TAG_LR) : ((2, 3), _TAG_R, _TAG_RR)
    # The SVD puts the new bond leg last; a right frame wants it first.
    orient = vectors -> TLArray(left ? vectors : permutedims(vectors, (3, 1, 2)))
    residual = perp(frame, candidates)
    residual_norm = norm(residual)
    residual_norm <= _CBE_EPS * max(norm(candidates), 1.0) && return nothing
    # the whole complement below the tolerance: nothing to admit
    residual_norm <= tol * norm(candidates) && return nothing
    # First projection: normalise, cut at the preselection's floor relative to the whole state, or
    # at the admission tolerance when that is higher.
    first_cut =
        max(_rank_cut(0.0, candidates, residual_norm), tol * norm(candidates) / residual_norm)
    orthonormal_basis = orient(
        svd(
            TLArray((1.0 / residual_norm) * residual),
            legs,
            left_tag,
            right_tag;
            cutoff = first_cut,
        ).U,
    )
    # Second projection of that basis: keep only directions that retain `_UNION_KEEP` of it.
    residual = perp(frame, orthonormal_basis)
    res = svd(residual, legs, left_tag, right_tag; cutoff = 1e-13, get_lists = true)
    smax = maximum(first, res.kept_list; init = 0.0)
    smax < _UNION_KEEP && return nothing
    # the SVD orders the surviving directions by weight, so `Nkeep` keeps the strongest ones
    limit = room < typemax(Int) ? (Nkeep = room,) : NamedTuple()
    return orient(svd(residual, legs, left_tag, right_tag; cutoff = _UNION_KEEP / smax, limit...).U)
end

"How many directions a kept state with tolerance `tol` may still append to a frame `width` wide."
_keep_room(cx, bond::Int, width::Int, tol::Float64) =
    cx.cap_keep && tol > 0 ? _cap_at(cx.wcap, bond) - width : typemax(Int)

"""
Union `krylov_frame` (the basis update's output) with `cbe_frame` (the CBE frame), on the LEFT.

`cbe_frame` FIRST so the CBE directions are kept verbatim and only the part of `krylov_frame`
outside them is appended -- `_preselect_left` returns `nothing` when there is nothing to append,
which is the ordinary case once the two already span the same space.
"""
function _union_left(cbe_frame, krylov_frame; tol::Float64 = 0.0, room::Int = typemax(Int))
    fresh = _union_fresh(cbe_frame, krylov_frame, :left; tol = tol, room = room)
    fresh === nothing && return cbe_frame
    return TLArray(
        oplus([cbe_frame, TLArray(setitag(fresh, 3, Telum.inds(cbe_frame)[3].itags))], (3,)),
    )
end
"Row mirror of [`_union_left`](@ref): the bond leg is 1 and the direct sum is over it."
function _union_right(cbe_frame, krylov_frame; tol::Float64 = 0.0, room::Int = typemax(Int))
    fresh = _union_fresh(cbe_frame, krylov_frame, :right; tol = tol, room = room)
    fresh === nothing && return cbe_frame
    return TLArray(
        oplus([cbe_frame, TLArray(setitag(fresh, 1, Telum.inds(cbe_frame)[1].itags))], (1,)),
    )
end

"""
    _half_left!(psiL, left_frames, expanded, nn, acc, hledger, rng, cx) -> lenv_root

PHASE B(a), LEFT -> ROOT: writes `left_frames[1:c]`, `expanded[1:c]` and `nn`, logs every bond
into `hledger`, and returns the left environment at link `c` in the NEW basis (captured before
the push through the root frame).

THE SWEEP IS INTERLEAVED: CBE expands bond `bond`, then that bond's frame is built and the
environment pushed through it immediately -- the reference's own ordering
(TDVPSweepCBE1Si.m:30/:34), so the environment chain runs through the NEW frames and bond
`bond+1`'s selection sees the widening at bond `bond`.
"""
function _half_left!(
    psiL,
    left_frames,
    expanded,
    nn,
    acc::_HalfAcc,
    hledger,
    rng,
    cx::_HalfSweepCtx,
)
    mpo, root_link = cx.mpo, cx.root_link
    canonical!(psiL, 1)
    _recycle_begin!(cx.recycle, :left, psiL, 0)
    rstack = _env_timed!(acc, right_env_stack, psiL, mpo; downto = 3)
    lch = boundary_channels(mpo)
    lroot = lch
    carry = nothing                   # the original amplitude in the new frames, `(b_i, link)`
    kcar = Any[nothing for _ in cx.keep_states]   # each kept state's carry, `(b_i, link)`
    held = 0                          # frames this sweep has accumulated, running total
    for bond = 1:root_link
        # The bond's left site with everything to its left already in the new frames.
        carried_site =
            carry === nothing ? psiL[bond] : contract(carry, (2,), psiL[bond], (1,))
        _fr = (frame_start = time_ns();
        built_frame = _frame_from(carried_site, psiL[bond+1]);
        acc.tframe += (time_ns() - frame_start) / 1e9;
        built_frame)
        bondlg = AdaptiveLedger()                               # this bond's own ledger
        ex = _expand_bond!(
            acc,
            cx,
            nn,
            _fr,
            bond,
            lch,
            right_channels(rstack, bond + 2),
            :left,
            bondlg,
            rng,
        )
        _peak!(acc, held + _state_stored(psiL), ex.U_ex, ex.V_ex)
        # ── stage 3: the frame grows from `U_ex` by Tests C/D ──────────────────────────────
        # The CBE frame is KEPT (old directions verbatim, CBE directions reused) and a Krylov
        # direction is appended only when its weighted residual outside the frame matters.
        # The right site projected onto the expanded right isometry, then joined to the left one.
        right_projected = contract(psiL[bond+1], (2, 3), ex.V_ex', (2, 3))
        Kex = contract(carried_site, (3,), right_projected, (1,))
        rch1 = push_right_channels(right_channels(rstack, bond + 2), mpo, ex.V_ex, bond + 1)
        site_operator =
            OneSiteOp(shift_h(one_site_h(mpo, bond, lch, rch1), cx.eshift), bond, acc.kc)
        tk0 = time_ns()
        Wkry, kst = _krylov_enrich(
            Kex,
            vector -> (acc.nmv += 1; apply_one_site(site_operator, vector)),
            cx.krylov_reuse ? ex.U_ex : _fr.U0,
            :left,
            cx.tau,
            cx.alloc;
            workcap = _cap_at(cx.wcap, bond),
            fixed_depth = cx.krylov_depth,
        )
        cx.krylov_reuse || (Wkry = _merge_frame(Wkry, ex.U_ex, :left))
        _recycle_store!(cx.recycle, :left, bond, kst, ex.V_ex, cx.alloc.krylov)
        acc.tkry += (time_ns() - tk0) / 1e9
        _record_krylov!(bondlg, kst, cx.alloc)
        _peak!(acc, held + _state_stored(psiL), Wkry)
        left_frames[bond] = TLArray(setitag(Wkry, 3, _tagW(bond)))
        # Union every kept state first: a later union widens `left_frames[bond]`, so a carry taken
        # before it would hold a stale bond leg.
        KXs = [
            kcar[state_index] === nothing ? kept_state[bond] :
            contract(kcar[state_index], (2,), kept_state[bond], (1,)) for
            (state_index, kept_state) in pairs(cx.keep_states)
        ]
        for (state_index, KX) in pairs(KXs)
            tol = cx.keep_tols[state_index]
            room = _keep_room(cx, bond, leg_dim(left_frames[bond], 3), tol)
            left_frames[bond] = _union_left(left_frames[bond], KX; tol = tol, room = room)
        end
        for (state_index, KX) in pairs(KXs)
            kcar[state_index] = contract(left_frames[bond]', (1, 2), KX, (1, 2))
        end
        _check_frame(left_frames[bond], :left, "half-sweep W[$bond]")
        expanded[bond] = leg_dim(left_frames[bond], 3)
        _log_bond!(hledger, bondlg, bond, :left, expanded[bond])
        held += tensor_elements(left_frames[bond])

        # (d) carry the ORIGINAL amplitude into the new basis. `carried_site`, not `Kex`: the
        # frame touches only legs 1-2, so this lands on `(b_i, link_i)` and composes with
        # `psiL[bond+1]`.
        carry = contract(left_frames[bond]', (1, 2), carried_site, (1, 2))
        bond == root_link && (lroot = lch)         # captured BEFORE pushing through the root frame
        bond < root_link &&
            (lch = _env_timed!(acc, push_left_channels, lch, mpo, left_frames[bond], bond))
    end
    _recycle_commit!(cx.recycle, :left, true)
    return lroot
end

"""
    _half_right!(psiR, right_frames, expanded, nn, acc, hledger, rng, cx) -> (renv_root, AR)

Right-to-root phase: writes `right_frames[c:L-1]`, `expanded[c+1:L-1]` and `nn`, and returns the
right environment at link `c+2` in the new basis together with the sweep's final carry.

The final carry IS `AR`: the recursion here and an `AR` loop after the sweeps would be the same
recursion over the same tensors in the same gauge, so `AR` is simply where `carry` ends. `AL`
gets no such shortcut and is recomputed, since the left carry is built in the 1-canonical gauge
and the two meet over link `c+1`; contracting them would be silently wrong.
"""
function _half_right!(
    psiR,
    right_frames,
    expanded,
    nn,
    acc::_HalfAcc,
    hledger,
    rng,
    cx::_HalfSweepCtx,
)
    mpo, root_link = cx.mpo, cx.root_link
    site_count = length(psiR)
    canonical!(psiR, site_count)
    # a shallow copy: the assembly later replaces `psi`'s slots, the cache must keep these
    _recycle_begin!(cx.recycle, :right, cx.recycle === nothing ? nothing : copy(psiR), 0)
    lstack = _env_timed!(acc, left_env_stack, psiR, mpo; upto = site_count - 2)
    rch = boundary_channels(mpo)
    rroot = rch
    carry = nothing                   # the original amplitude in the new frames, `(link, b_j)`
    kcar = Any[nothing for _ in cx.keep_states]   # each kept state's carry, `(link, b_j)`
    held = 0
    for bond = (site_count-1):-1:root_link
        # The bond's right site with everything to its right already in the new frames.
        carried_site =
            carry === nothing ? psiR[bond+1] :
            contract(psiR[bond+1], (3,), carry, (1,))
        _fr = (frame_start = time_ns();
        built_frame = _frame_from(psiR[bond], carried_site);
        acc.tframe += (time_ns() - frame_start) / 1e9;
        built_frame)
        bondlg = AdaptiveLedger()                               # this bond's own ledger
        ex = _expand_bond!(
            acc,
            cx,
            nn,
            _fr,
            bond,
            left_channels(lstack, bond),
            rch,
            :right,
            bondlg,
            rng,
        )
        _peak!(acc, held + _state_stored(psiR), ex.U_ex, ex.V_ex)
        # Row mirror of the left sweep's Stage 3; see the comment there.
        # (b_L, link_j)
        left_projected = contract(ex.U_ex', (1, 2), psiR[bond], (1, 2))
        Bex = contract(left_projected, (2,), carried_site, (1,))
        lch1 = push_left_channels(left_channels(lstack, bond), mpo, ex.U_ex, bond)
        site_operator =
            OneSiteOp(shift_h(one_site_h(mpo, bond + 1, lch1, rch), cx.eshift), bond, acc.kc)
        tk0 = time_ns()
        Zkry, kst = _krylov_enrich(
            Bex,
            vector -> (acc.nmv += 1; apply_one_site(site_operator, vector)),
            cx.krylov_reuse ? ex.V_ex : _fr.V0,
            :right,
            cx.tau,
            cx.alloc;
            workcap = _cap_at(cx.wcap, bond),
            fixed_depth = cx.krylov_depth,
        )
        cx.krylov_reuse || (Zkry = _merge_frame(Zkry, ex.V_ex, :right))
        _recycle_store!(cx.recycle, :right, bond, kst, ex.U_ex, cx.alloc.krylov)
        acc.tkry += (time_ns() - tk0) / 1e9
        _record_krylov!(bondlg, kst, cx.alloc)
        _peak!(acc, held + _state_stored(psiR), Zkry)
        right_frames[bond] = TLArray(setitag(Zkry, 1, _tagZ(bond)))
        BXs = [
            kcar[state_index] === nothing ? kept_state[bond+1] :
            contract(kept_state[bond+1], (3,), kcar[state_index], (1,)) for
            (state_index, kept_state) in pairs(cx.keep_states)
        ]
        for (state_index, BX) in pairs(BXs)     # all unions before any carry, as on the left
            tol = cx.keep_tols[state_index]
            room = _keep_room(cx, bond, leg_dim(right_frames[bond], 1), tol)
            right_frames[bond] = _union_right(right_frames[bond], BX; tol = tol, room = room)
        end
        for (state_index, BX) in pairs(BXs)
            kcar[state_index] = contract(BX, (2, 3), right_frames[bond]', (2, 3))
        end
        _check_frame(right_frames[bond], :right, "half-sweep Z[$bond]")
        bond != root_link && (expanded[bond] = leg_dim(right_frames[bond], 1))
        _log_bond!(hledger, bondlg, bond, :right, leg_dim(right_frames[bond], 1))
        held += tensor_elements(right_frames[bond])

        # (link_j, b_j)
        carry = contract(carried_site, (2, 3), right_frames[bond]', (2, 3))
        bond == root_link && (rroot = rch)
        bond > root_link &&
            (rch = _env_timed!(acc, push_right_channels, rch, mpo, right_frames[bond], bond + 1))
    end
    _recycle_commit!(cx.recycle, :right, true)
    return (rroot, carry)               # `carry` is (link_{c+1}, b_R)
end
