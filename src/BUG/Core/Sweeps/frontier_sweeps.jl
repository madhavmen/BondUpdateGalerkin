# frontier_sweeps.jl: the two frontier half-sweeps (`_frontier_left`, `_frontier_right`) and the
# polar re-orthonormalisation of the frames they grow. The sweep over both sides, the
# measurements built on it and the closure loop live in `frontier_closure.jl`, included next.

const _TAG_POL_A, _TAG_POL_B = "cbePol,A", "cbePol,B"

"""
    _polar_frame(frame, side) -> the orthonormal frame nearest `frame`

`U V'` from `frame = U S V'`, which has the same span, width and outgoing leg as `frame`. The
frontier sweeps carry the old frame into each grown bond as `U0 = Pm W[i]` with `Pm = Wn'U0`, and
that is the transfer map only when `Wn` is orthonormal. A bond that adds nothing keeps `Wn = U0`,
so with Gram `G = I + E` the map is `G` itself and the next frame's Gram is about `G^2`: the
defect doubles to triples per site and again in every closure generation. With the polar factor,
`Pm = G^{1/2}`, so the defect is passed on, not squared. MEASURED (A2 L = 48, 2 excitations,
closure_checks, generation 10): ×2.9-3.0 per site, from 2.5e-5 at site 31 to 0.015 at site 25,
then 16-17 in the root frames; at A2 L = 32 it reached 7.5e5, the root solve returned NaN and the
root split found no singular values (dr6v, dr7s, dr6k).
"""
function _polar_frame(frame, side::Symbol)
    res = svd(frame, side === :left ? (1, 2) : (1,), _TAG_POL_A, _TAG_POL_B; cutoff = 0.0)
    # `U S^0 Vd`: S with unit singular values carries the tags and arrows that join U to Vd
    unit_values = res.S
    # 1 for a kept singular value, 0 for an exact zero (so a rank-deficient frame stays so)
    unit(value) = iszero(value) ? zero(value) : one(value)
    for block in eachindex(unit_values.RMTs)
        unit_values.isdefined[block] || continue
        matrix_block = unit_values.RMTs[block]
        matrix_block isa Telum.DiagRMT ? (matrix_block.diag .= unit.(matrix_block.diag)) :
        (unit_values.RMTs[block] = map(unit, matrix_block))
    end
    return to_concrete(res.U * (unit_values * res.Vd))
end

"""
    _frontier_left(Fm, mpo, fr, tau, alloc, ckw; enrich, gen, workcap, amp, rng, nsample)

The left half-sweep of one frontier MPS `Fm` (centre at the root). Bond by bond the frame
`fr.left_frames[bond]` is embedded into the grown basis of bond `bond-1`, and the frontier's two-site
outside action is measured. If it exceeds the bond's share and `enrich` is set,
[`_frontier_expand`](@ref) and `_krylov_enrich` append validated directions. Returns the new
frames (generation `gen`), the root environment, the bond ledger and the per-bond misses. A
measuring sweep (`enrich = false`) grows nothing and uses the cached frame chain
([`_side_cache!`](@ref)).
"""
function _frontier_left(
    Fm::SymMPS,
    mpo,
    fr::KLFrames,
    tau::ComplexF64,
    alloc::BasisAllocation,
    ckw::NamedTuple;
    enrich::Bool,
    gen::Int,
    workcap::WorkCeiling,
    amp::Float64,
    rng::AbstractRNG,
    nsample::Int = 0,
    fold::Bool = false,
    bonds::Bool = false,
)
    root_link, site_count = fr.root_link, length(Fm)
    canonical!(Fm, 1)
    chain, zst = _side_cache!(fr, Fm, mpo, :left)
    right_envs = copy(zst)                       # links > root_link: shared by every frontier
    env = right_envs[root_link+1]
    for site = root_link:-1:3                    # links 3..root_link: own gauge
        env = _mpo_right_step(env, Fm[site], mpo[site])
        right_envs[site] = env
    end
    rstack = MPORightEnvStack(right_envs)
    # a measuring sweep may take every bond's action up front, in parallel (`_bond_measurements`)
    pre =
        bonds && !enrich ?
        _bond_measurements(
            Fm,
            mpo,
            fr,
            chain,
            rstack,
            :left;
            expansion = ckw.expansion,
            fold = fold,
            nsample = nsample,
            rng = rng,
        ) : nothing
    lch = boundary_channels(mpo)
    lenv = lch
    lchs = Vector{Any}(undef, site_count - 1)
    rec = enrich ? get(ckw, :recycle, nothing) : nothing
    rcur = _recycle_entries(rec, :left, fr.gen)
    Eov = rcur === nothing ? nothing : _tail_overlaps(Fm, rcur.tail, :left)
    _recycle_begin!(rec, :left, Fm, gen)
    # `carry` (new frame bond, frontier link) gauges the frontier into the grown frames,
    # `Pm` (new bond, old bond) carries the old frame into them
    carry, Pm = nothing, nothing
    Wn = Vector{Any}(undef, root_link)
    lg = AdaptiveLedger()
    leak, sleak = zeros(root_link), zeros(root_link)
    added, newsec, nskip, nh2 = 0, 0, 0, 0
    nh1 = Ref(0)
    for bond = 1:root_link
        enrich || (lch = chain[bond])
        old_frame = _check_frame(
            Pm === nothing ? fr.left_frames[bond] :
            to_concrete!(contract(Pm, (2,), fr.left_frames[bond], (1,))),
            :left,
            "frontier U0 = Pm W[$bond] (gen $(fr.gen))",
        )
        if pre === nothing
            # the frontier's site tensor in the grown left gauge, then its two-site action
            site_core =
                carry === nothing ? Fm[bond] : to_concrete!(contract(carry, (2,), Fm[bond], (1,)))
            two_site_frame = _frame_onto_left(old_frame, site_core, Fm[bond+1])
            rch = right_channels(rstack, bond + 2)
            skl, skr, leak[bond], two_site_calls = _bond_action(
                ckw.expansion,
                two_site_frame,
                mpo,
                bond,
                lch,
                rch,
                old_frame,
                nothing,
                :left,
                fold,
                rng,
            )
            nsample > 0 && (
                sleak[bond] =
                    sampled_leak(ckw.expansion, two_site_frame, skl, skr, :left, nsample, rng)
            )
        else
            site_core, leak[bond], two_site_calls, sleak[bond] = pre[bond]
        end
        nh2 += two_site_calls
        bl = AdaptiveLedger()
        quiet = indicator(alloc, leak[bond]) <= alloc.per_bond
        nrec, rclosed, rov, rdef = 0, false, 0.0, 0.0
        Wi = if enrich && !quiet
            ex = _frontier_expand(two_site_frame, mpo, bond, lch, rch, :left, ckw, bl, rng)
            # the next site projected onto the expansion's right directions: (link, b_ex)
            right_projected = to_concrete!(contract(Fm[bond+1], (2, 3), ex.V_ex', (2, 3)))
            Kex = to_concrete!(contract(site_core, (3,), right_projected, (1,)))
            one_site_op = shift_h(
                one_site_h(mpo, bond, lch, push_right_channels(rch, mpo, ex.V_ex, bond + 1)),
                fr.eshift,
            )
            reuse = get(ckw, :krylov_reuse, true)
            recycle_entry = rcur === nothing ? nothing : get(rcur.entries, bond, nothing)
            Pmi = Pm
            krylov_frame, st = _recycled_or_fresh(
                rec,
                :left,
                bond,
                recycle_entry,
                () -> _recycle_transport(recycle_entry, :left, Pmi, ex.V_ex, Eov, bond),
                Kex,
                vector -> (nh1[] += 1; apply_one_site(one_site_op, vector)),
                reuse ? ex.U_ex : old_frame,
                tau,
                alloc;
                round = gen + 1,
                workcap = _cap_at(workcap, bond),
                amp = amp,
                state_seed = false,
                fixed_depth = get(ckw, :krylov_depth, 0),
            )
            _recycle_store!(rec, :left, bond, st, ex.V_ex, alloc.krylov)
            nrec = get(st, :n_recycled, 0)
            rclosed, rov, rdef = get(st, :rec_closed, false),
            get(st, :rec_overlap, 0.0),
            get(st, :rec_defect, 0.0)
            _record_krylov!(bl, st, alloc)
            reuse ? krylov_frame : _merge_frame(krylov_frame, ex.U_ex, :left)
        else
            nskip += 1
            old_frame
        end
        Wn[bond] =
            enrich ? _polar_frame(to_concrete(setitag(Wi, 3, _tagWg(bond, gen))), :left) : old_frame
        _check_frame(Wn[bond], :left, "frontier Wn[$bond] (gen $gen)")
        nadd = leg_dim(Wn[bond], 3) - leg_dim(old_frame, 3)
        nsec = _new_sectors(old_frame, Wn[bond], 3)
        added += nadd
        newsec += nsec
        _log_bond!(lg, bl, bond, :left, leg_dim(Wn[bond], 3))
        _tag_last!(
            lg,
            (
                leak = indicator(alloc, leak[bond]),
                skipped = quiet || !enrich,
                added = nadd,
                new_sectors = nsec,
                width_before = leg_dim(old_frame, 3),
                recycled = nrec,
                rec_closed = rclosed,
                rec_overlap = rov,
                rec_defect = rdef,
            ),
        )
        enrich && (Pm = to_concrete!(contract(Wn[bond]', (1, 2), old_frame, (1, 2))))  # (new, old)
        pre === nothing && (carry = to_concrete!(contract(Wn[bond]', (1, 2), site_core, (1, 2))))
        lchs[bond] = lch
        bond == root_link && (lenv = lch)
        enrich && bond < root_link && (lch = push_left_channels(lch, mpo, Wn[bond], bond))
    end
    return (
        frames = Wn,
        env = lenv,
        chain = lchs,
        ledger = lg,
        leak = leak,
        sleak = sleak,
        added = added,
        new_sectors = newsec,
        skipped = nskip,
        n_h2 = nh2,
        n_h1 = nh1[],
    )
end

"Row mirror of [`_frontier_left`](@ref): grows `fr.right_frames[L-1..c]`; returns the same NamedTuple."
function _frontier_right(
    Fm::SymMPS,
    mpo,
    fr::KLFrames,
    tau::ComplexF64,
    alloc::BasisAllocation,
    ckw::NamedTuple;
    enrich::Bool,
    gen::Int,
    workcap::WorkCeiling,
    amp::Float64,
    rng::AbstractRNG,
    nsample::Int = 0,
    fold::Bool = false,
    bonds::Bool = false,
)
    root_link, site_count = fr.root_link, length(Fm)
    canonical!(Fm, site_count)
    chain, wst = _side_cache!(fr, Fm, mpo, :right)
    left_envs = copy(wst)                        # links <= root_link: shared by every frontier
    env = left_envs[root_link]
    for site = root_link:(site_count-2)          # links root_link+1..L-1: own gauge
        env = _mpo_left_step(env, Fm[site], mpo[site])
        left_envs[site+1] = env
    end
    lstack = MPOLeftEnvStack(left_envs)
    # a measuring sweep may take every bond's action up front, in parallel (`_bond_measurements`)
    pre =
        bonds && !enrich ?
        _bond_measurements(
            Fm,
            mpo,
            fr,
            chain,
            lstack,
            :right;
            expansion = ckw.expansion,
            fold = fold,
            nsample = nsample,
            rng = rng,
        ) : nothing
    rch = boundary_channels(mpo)
    renv = rch
    rchs = Vector{Any}(undef, site_count - 1)
    rec = enrich ? get(ckw, :recycle, nothing) : nothing
    rcur = _recycle_entries(rec, :right, fr.gen)
    Eov = rcur === nothing ? nothing : _tail_overlaps(Fm, rcur.tail, :right)
    _recycle_begin!(rec, :right, Fm, gen)
    # `carry` (frontier link, new frame bond) gauges the frontier into the grown frames,
    # `Pm` (old bond, new bond) carries the old frame into them
    carry, Pm = nothing, nothing
    Zn = Vector{Any}(undef, site_count - 1)
    lg = AdaptiveLedger()
    leak, sleak = zeros(site_count - 1), zeros(site_count - 1)
    added, newsec, nskip, nh2 = 0, 0, 0, 0
    nh1 = Ref(0)
    for bond = (site_count-1):-1:root_link
        enrich || (rch = chain[bond])
        old_frame = _check_frame(
            Pm === nothing ? fr.right_frames[bond] :
            to_concrete!(contract(fr.right_frames[bond], (3,), Pm, (1,))),
            :right,
            "frontier V0 = Z[$bond] Pm (gen $(fr.gen))",
        )
        if pre === nothing
            # the frontier's site tensor in the grown right gauge, then its two-site action
            site_core =
                carry === nothing ? Fm[bond+1] :
                to_concrete!(contract(Fm[bond+1], (3,), carry, (1,)))
            two_site_frame = _frame_onto_right(Fm[bond], site_core, old_frame)
            lch = left_channels(lstack, bond)
            skl, skr, leak[bond], two_site_calls = _bond_action(
                ckw.expansion,
                two_site_frame,
                mpo,
                bond,
                lch,
                rch,
                nothing,
                old_frame,
                :right,
                fold,
                rng,
            )
            nsample > 0 && (
                sleak[bond] =
                    sampled_leak(ckw.expansion, two_site_frame, skl, skr, :right, nsample, rng)
            )
        else
            site_core, leak[bond], two_site_calls, sleak[bond] = pre[bond]
        end
        nh2 += two_site_calls
        bl = AdaptiveLedger()
        quiet = indicator(alloc, leak[bond]) <= alloc.per_bond
        nrec, rclosed, rov, rdef = 0, false, 0.0, 0.0
        Zj = if enrich && !quiet
            ex = _frontier_expand(two_site_frame, mpo, bond, lch, rch, :right, ckw, bl, rng)
            # the previous site projected onto the expansion's left directions: (b_ex, link)
            left_projected = to_concrete!(contract(ex.U_ex', (1, 2), Fm[bond], (1, 2)))
            Bex = to_concrete!(contract(left_projected, (2,), site_core, (1,)))
            one_site_op = shift_h(
                one_site_h(mpo, bond + 1, push_left_channels(lch, mpo, ex.U_ex, bond), rch),
                fr.eshift,
            )
            reuse = get(ckw, :krylov_reuse, true)
            recycle_entry = rcur === nothing ? nothing : get(rcur.entries, bond, nothing)
            Pmj = Pm
            krylov_frame, st = _recycled_or_fresh(
                rec,
                :right,
                bond,
                recycle_entry,
                () -> _recycle_transport(recycle_entry, :right, Pmj, ex.U_ex, Eov, bond),
                Bex,
                vector -> (nh1[] += 1; apply_one_site(one_site_op, vector)),
                reuse ? ex.V_ex : old_frame,
                tau,
                alloc;
                round = gen + 1,
                workcap = _cap_at(workcap, bond),
                amp = amp,
                state_seed = false,
                fixed_depth = get(ckw, :krylov_depth, 0),
            )
            _recycle_store!(rec, :right, bond, st, ex.U_ex, alloc.krylov)
            nrec = get(st, :n_recycled, 0)
            rclosed, rov, rdef = get(st, :rec_closed, false),
            get(st, :rec_overlap, 0.0),
            get(st, :rec_defect, 0.0)
            _record_krylov!(bl, st, alloc)
            reuse ? krylov_frame : _merge_frame(krylov_frame, ex.V_ex, :right)
        else
            nskip += 1
            old_frame
        end
        Zn[bond] =
            enrich ? _polar_frame(to_concrete(setitag(Zj, 1, _tagZg(bond, gen))), :right) :
            old_frame
        _check_frame(Zn[bond], :right, "frontier Zn[$bond] (gen $gen)")
        nadd = leg_dim(Zn[bond], 1) - leg_dim(old_frame, 1)
        nsec = _new_sectors(old_frame, Zn[bond], 1)
        added += nadd
        newsec += nsec
        _log_bond!(lg, bl, bond, :right, leg_dim(Zn[bond], 1))
        _tag_last!(
            lg,
            (
                leak = indicator(alloc, leak[bond]),
                skipped = quiet || !enrich,
                added = nadd,
                new_sectors = nsec,
                width_before = leg_dim(old_frame, 1),
                recycled = nrec,
                rec_closed = rclosed,
                rec_overlap = rov,
                rec_defect = rdef,
            ),
        )
        enrich && (Pm = to_concrete!(contract(old_frame, (2, 3), Zn[bond]', (2, 3))))  # (old, new)
        pre === nothing && (carry = to_concrete!(contract(site_core, (2, 3), Zn[bond]', (2, 3))))
        rchs[bond] = rch
        bond == root_link && (renv = rch)
        enrich && bond > root_link && (rch = push_right_channels(rch, mpo, Zn[bond], bond + 1))
    end
    return (
        frames = Zn,
        env = renv,
        chain = rchs,
        ledger = lg,
        leak = leak,
        sleak = sleak,
        added = added,
        new_sectors = newsec,
        skipped = nskip,
        n_h2 = nh2,
        n_h1 = nh1[],
    )
end
