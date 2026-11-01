# Moved from BUG-Julia@b155550:src/RSVDCBEBondUpdate/bug/frontier_sweeps.jl.

# frontier_sweeps.jl: the frontier half-sweeps and their measurements (from `krylov_closure.jl`,
# included right after it).

const _TAG_POL_A, _TAG_POL_B = "cbePol,A", "cbePol,B"

"""
    _polar_frame(F, side) -> the orthonormal frame nearest `F`

`U V'` from `F = U S V'`, which has the same span, width and outgoing leg as `F`. The frontier
sweeps carry the old frame into each grown bond as `U0 = Pm W[i]` with `Pm = Wn'U0`, and that is
the transfer map only when `Wn` is orthonormal. A bond that adds nothing keeps `Wn = U0`, so with
Gram `G = I + E` the map is `G` itself and the next frame's Gram is about `G^2`: the defect
doubles to triples per site and again in every closure generation. With the polar factor,
`Pm = G^{1/2}`, so the defect is passed on, not squared. MEASURED (A2 L = 48, 2 excitations,
closure_checks, generation 10): ×2.9-3.0 per site, from 2.5e-5 at site 31 to 0.015 at site 25,
then 16-17 in the root frames; at A2 L = 32 it reached 7.5e5, the root solve returned NaN and the
root split found no singular values (dr6v, dr7s, dr6k).
"""
function _polar_frame(F, side::Symbol)
    res = svd(F, side === :left ? (1, 2) : (1,), _TAG_POL_A, _TAG_POL_B; cutoff = 0.0)
    # `U S^0 Vd`: S with unit singular values carries the tags and arrows that join U to Vd
    S0 = res.S
    unit(x) = iszero(x) ? zero(x) : one(x)
    for k in eachindex(S0.RMTs)
        S0.isdefined[k] || continue
        r = S0.RMTs[k]
        r isa Telum.DiagRMT ? (r.diag .= unit.(r.diag)) : (S0.RMTs[k] = map(unit, r))
    end
    return to_concrete(res.U * (S0 * res.Vd))
end

"""
    _frontier_left(Fm, mpo, fr, tau, alloc, ckw; enrich, gen, workcap, amp, rng, nsample)

The left half-sweep of one frontier MPS `Fm` (centre at the root). Bond by bond the frame
`fr.W[i]` is embedded into the grown basis of bond `i-1`, and the frontier's two-site outside
action is measured. If it exceeds the bond's share and `enrich` is set, [`_frontier_expand`](@ref)
and `_krylov_enrich` append validated directions. Returns the new frames (generation `gen`), the
root environment, the bond ledger and the per-bond misses. A measuring sweep (`enrich = false`)
grows nothing and uses the cached frame chain ([`_side_cache!`](@ref)).
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
    workcap::Int,
    amp::Float64,
    rng::AbstractRNG,
    nsample::Int = 0,
    fold::Bool = false,
    bonds::Bool = false,
)
    c, L = fr.c, length(Fm)
    canonical!(Fm, 1)
    chain, zst = _side_cache!(fr, Fm, mpo, :left)
    E = copy(zst)                                # links >= c+1: shared by every frontier
    env = E[c+1]
    for k = c:-1:3                              # links 3..c: this frontier's own gauge
        env = _mpo_right_step(env, Fm[k], mpo[k])
        E[k] = env
    end
    rstack = MPORightEnvStack(E)
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
    lchs = Vector{Any}(undef, L - 1)
    rec = enrich ? get(ckw, :recycle, nothing) : nothing
    rcur = _recycle_entries(rec, :left, fr.gen)
    Eov = rcur === nothing ? nothing : _tail_overlaps(Fm, rcur.tail, :left)
    _recycle_begin!(rec, :left, Fm, gen)
    C, Pm = nothing, nothing
    Wn = Vector{Any}(undef, c)
    lg = AdaptiveLedger()
    leak, sleak = zeros(c), zeros(c)
    added, newsec, nskip, nh2 = 0, 0, 0, 0
    nh1 = Ref(0)
    for i = 1:c
        enrich || (lch = chain[i])
        U0 = _check_frame(
            Pm === nothing ? fr.W[i] : to_concrete!(contract(Pm, (2,), fr.W[i], (1,))),
            :left,
            "frontier U0 = Pm W[$i] (gen $(fr.gen))",
        )
        if pre === nothing
            K = C === nothing ? Fm[i] : to_concrete!(contract(C, (2,), Fm[i], (1,)))
            f = _frame_onto_left(U0, K, Fm[i+1])
            rch = right_channels(rstack, i + 2)
            skl, skr, leak[i], k2 =
                _bond_action(ckw.expansion, f, mpo, i, lch, rch, U0, nothing, :left, fold, rng)
            nsample > 0 &&
                (sleak[i] = sampled_leak(ckw.expansion, f, skl, skr, :left, nsample, rng))
        else
            K, leak[i], k2, sleak[i] = pre[i]
        end
        nh2 += k2
        bl = AdaptiveLedger()
        quiet = indicator(alloc, leak[i]) <= alloc.per_bond
        nrec, rclosed, rov, rdef = 0, false, 0.0, 0.0
        Wi = if enrich && !quiet
            ex = _frontier_expand(f, mpo, i, lch, rch, :left, ckw, bl, rng)
            M = to_concrete!(contract(Fm[i+1], (2, 3), ex.V_ex', (2, 3)))
            Kex = to_concrete!(contract(K, (3,), M, (1,)))
            H1 = shift_h(
                one_site_h(mpo, i, lch, push_right_channels(rch, mpo, ex.V_ex, i + 1)),
                fr.eshift,
            )
            reuse = get(ckw, :krylov_reuse, true)
            e = rcur === nothing ? nothing : get(rcur.entries, i, nothing)
            Pmi = Pm
            Q, st = _recycled_or_fresh(
                rec,
                :left,
                i,
                e,
                () -> _recycle_transport(e, :left, Pmi, ex.V_ex, Eov, i),
                Kex,
                x -> (nh1[] += 1; apply_one_site(H1, x)),
                reuse ? ex.U_ex : U0,
                tau,
                alloc;
                round = gen + 1,
                workcap = workcap,
                amp = amp,
                state_seed = false,
                fixed_depth = get(ckw, :krylov_depth, 0),
            )
            _recycle_store!(rec, :left, i, st, ex.V_ex, alloc.krylov)
            nrec = get(st, :n_recycled, 0)
            rclosed, rov, rdef = get(st, :rec_closed, false),
            get(st, :rec_overlap, 0.0),
            get(st, :rec_defect, 0.0)
            _record_krylov!(bl, st, alloc)
            reuse ? Q : _merge_frame(Q, ex.U_ex, :left)
        else
            nskip += 1
            U0
        end
        Wn[i] = enrich ? _polar_frame(to_concrete(setitag(Wi, 3, _tagWg(i, gen))), :left) : U0
        _check_frame(Wn[i], :left, "frontier Wn[$i] (gen $gen)")
        nadd = leg_dim(Wn[i], 3) - leg_dim(U0, 3)
        nsec = _new_sectors(U0, Wn[i], 3)
        added += nadd
        newsec += nsec
        _log_bond!(lg, bl, i, :left, leg_dim(Wn[i], 3))
        _tag_last!(
            lg,
            (
                leak = indicator(alloc, leak[i]),
                skipped = quiet || !enrich,
                added = nadd,
                new_sectors = nsec,
                width_before = leg_dim(U0, 3),
                recycled = nrec,
                rec_closed = rclosed,
                rec_overlap = rov,
                rec_defect = rdef,
            ),
        )
        enrich && (Pm = to_concrete!(contract(Wn[i]', (1, 2), U0, (1, 2))))  # (new_i, old_i)
        pre === nothing && (C = to_concrete!(contract(Wn[i]', (1, 2), K, (1, 2))))
        lchs[i] = lch
        i == c && (lenv = lch)
        enrich && i < c && (lch = push_left_channels(lch, mpo, Wn[i], i))
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

"Row mirror of [`_frontier_left`](@ref): grows `fr.Z[L-1..c]`."
function _frontier_right(
    Fm::SymMPS,
    mpo,
    fr::KLFrames,
    tau::ComplexF64,
    alloc::BasisAllocation,
    ckw::NamedTuple;
    enrich::Bool,
    gen::Int,
    workcap::Int,
    amp::Float64,
    rng::AbstractRNG,
    nsample::Int = 0,
    fold::Bool = false,
    bonds::Bool = false,
)
    c, L = fr.c, length(Fm)
    canonical!(Fm, L)
    chain, wst = _side_cache!(fr, Fm, mpo, :right)
    E = copy(wst)                                # links <= c: shared by every frontier
    env = E[c]
    for i = c:(L-2)                           # links c+1..L-1: this frontier's own gauge
        env = _mpo_left_step(env, Fm[i], mpo[i])
        E[i+1] = env
    end
    lstack = MPOLeftEnvStack(E)
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
    rchs = Vector{Any}(undef, L - 1)
    rec = enrich ? get(ckw, :recycle, nothing) : nothing
    rcur = _recycle_entries(rec, :right, fr.gen)
    Eov = rcur === nothing ? nothing : _tail_overlaps(Fm, rcur.tail, :right)
    _recycle_begin!(rec, :right, Fm, gen)
    D, Pm = nothing, nothing
    Zn = Vector{Any}(undef, L - 1)
    lg = AdaptiveLedger()
    leak, sleak = zeros(L - 1), zeros(L - 1)
    added, newsec, nskip, nh2 = 0, 0, 0, 0
    nh1 = Ref(0)
    for j = (L-1):-1:c
        enrich || (rch = chain[j])
        V0 = _check_frame(
            Pm === nothing ? fr.Z[j] : to_concrete!(contract(fr.Z[j], (3,), Pm, (1,))),
            :right,
            "frontier V0 = Z[$j] Pm (gen $(fr.gen))",
        )
        if pre === nothing
            B = D === nothing ? Fm[j+1] : to_concrete!(contract(Fm[j+1], (3,), D, (1,)))
            f = _frame_onto_right(Fm[j], B, V0)
            lch = left_channels(lstack, j)
            skl, skr, leak[j], k2 =
                _bond_action(ckw.expansion, f, mpo, j, lch, rch, nothing, V0, :right, fold, rng)
            nsample > 0 &&
                (sleak[j] = sampled_leak(ckw.expansion, f, skl, skr, :right, nsample, rng))
        else
            B, leak[j], k2, sleak[j] = pre[j]
        end
        nh2 += k2
        bl = AdaptiveLedger()
        quiet = indicator(alloc, leak[j]) <= alloc.per_bond
        nrec, rclosed, rov, rdef = 0, false, 0.0, 0.0
        Zj = if enrich && !quiet
            ex = _frontier_expand(f, mpo, j, lch, rch, :right, ckw, bl, rng)
            M = to_concrete!(contract(ex.U_ex', (1, 2), Fm[j], (1, 2)))
            Bex = to_concrete!(contract(M, (2,), B, (1,)))
            H1 = shift_h(
                one_site_h(mpo, j + 1, push_left_channels(lch, mpo, ex.U_ex, j), rch),
                fr.eshift,
            )
            reuse = get(ckw, :krylov_reuse, true)
            e = rcur === nothing ? nothing : get(rcur.entries, j, nothing)
            Pmj = Pm
            Q, st = _recycled_or_fresh(
                rec,
                :right,
                j,
                e,
                () -> _recycle_transport(e, :right, Pmj, ex.U_ex, Eov, j),
                Bex,
                x -> (nh1[] += 1; apply_one_site(H1, x)),
                reuse ? ex.V_ex : V0,
                tau,
                alloc;
                round = gen + 1,
                workcap = workcap,
                amp = amp,
                state_seed = false,
                fixed_depth = get(ckw, :krylov_depth, 0),
            )
            _recycle_store!(rec, :right, j, st, ex.U_ex, alloc.krylov)
            nrec = get(st, :n_recycled, 0)
            rclosed, rov, rdef = get(st, :rec_closed, false),
            get(st, :rec_overlap, 0.0),
            get(st, :rec_defect, 0.0)
            _record_krylov!(bl, st, alloc)
            reuse ? Q : _merge_frame(Q, ex.V_ex, :right)
        else
            nskip += 1
            V0
        end
        Zn[j] = enrich ? _polar_frame(to_concrete(setitag(Zj, 1, _tagZg(j, gen))), :right) : V0
        _check_frame(Zn[j], :right, "frontier Zn[$j] (gen $gen)")
        nadd = leg_dim(Zn[j], 1) - leg_dim(V0, 1)
        nsec = _new_sectors(V0, Zn[j], 1)
        added += nadd
        newsec += nsec
        _log_bond!(lg, bl, j, :right, leg_dim(Zn[j], 1))
        _tag_last!(
            lg,
            (
                leak = indicator(alloc, leak[j]),
                skipped = quiet || !enrich,
                added = nadd,
                new_sectors = nsec,
                width_before = leg_dim(V0, 1),
                recycled = nrec,
                rec_closed = rclosed,
                rec_overlap = rov,
                rec_defect = rdef,
            ),
        )
        enrich && (Pm = to_concrete!(contract(V0, (2, 3), Zn[j]', (2, 3))))  # (old_j, new_j)
        pre === nothing && (D = to_concrete!(contract(B, (2, 3), Zn[j]', (2, 3))))
        rchs[j] = rch
        j == c && (renv = rch)
        enrich && j > c && (rch = push_right_channels(rch, mpo, Zn[j], j + 1))
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
