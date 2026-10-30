# Moved from BUG-Julia@b155550:src/RSVDCBEBondUpdate/expansion/cbe_core.jl, src/RSVDCBEBondUpdate/expansion/cbe_expand.jl, src/RSVDCBEBondUpdate/expansion/cbe_options.jl, src/RSVDCBEBondUpdate/expansion/cbe_bond_update.jl, src/RSVDCBEBondUpdate/operators/mpo_env.jl, src/RSVDCBEBondUpdate/operators/mpo_fold.jl, src/RSVDCBEBondUpdate/expansion/bond_expansion.jl.

# ── from src/RSVDCBEBondUpdate/expansion/cbe_core.jl ──

"""
    _unprime(t, leg) -> TLArray

Drop one prime level from `leg`. An environment's bra leg is primed to keep it distinct
from the ket while both are open (`henv.jl`); once it has replaced a link leg it must
carry that link's original prime level again.
"""
#
# `to_concrete!`, not `to_concrete`: `prime` already returns a deep copy the caller owns, so a second
# copy is pure waste.
_unprime(t, leg::Int) = to_concrete!(prime(t, leg; inc = -1))

"""
    _env_on_link(T, E, leg) -> TLArray

Contract environment `E` onto `leg` of `T`, leaving the bra leg in `leg`'s place,
unprimed. If both carry an op-leg -- `T`'s last, `E`'s third -- they are contracted too
(the channel closes).

`T` is `K0`-shaped `(link_l, site_l, bond [, op])` with `leg = 1`, or `V0`-shaped
`(bond, site_r, link_r [, op])` with `leg = 3`. In both cases the contracted link leg is
at an END of the tensor, so the bra comes back out in exactly the position it vacated
and no permutation is needed beyond the two written below.
"""
function _env_on_link(T, E, leg::Int)
    nT, nE = length(T.inds), length(E.inds)
    out = if nE == 3 && nT == 4
        contract(T, (leg, nT), E, (2, 3))
    elseif nE == 2 && nT == 3
        contract(T, (leg,), E, (2,))
    else
        error("_env_on_link: rank-$nT tensor against a rank-$nE environment leaves an " *
              "operator channel dangling")
    end
    # `out` = T's remaining legs in order, then E's bra leg.
    #   leg == 1 (K0): (site_l, bond, bra) -> bra first
    #   leg == 3 (V0): (bond, site_r, bra) -> already in place
    if leg == 1
        return _unprime(to_concrete(permutedims(out, (3, 1, 2))), 1)
    elseif leg == 3
        return _unprime(to_concrete(out), 3)
    end
    throw(ArgumentError("_env_on_link: leg must be 1 or 3, got $leg"))
end

# ── project-first sketches: `Y = (P_perp (H Theta)) Om'` ──
# `P_perp` and `Om` act on disjoint legs, so the order is exact; `_preselect_*` projects again
# (idempotent) because it also orthonormalises and returns the residual.

# `H*Theta` is built once per `cbe_expand` and shared by its three sketch calls; the arithmetic is
# the same as building it per call.

"""
    _sketch_closures(f, buildHT) -> (skl, skr)

The `(skl, skr)` pair [`cbe_expand`](@ref) wants, sharing one lazily built `H*Theta` and one
projection per side. Lazy because `cbe_expand` often returns before sketching (saturated frames).
"""
function _sketch_closures(f::BondFrame, buildHT; share::Bool = true)
    HT = Ref{Any}(nothing)
    AL = Ref{Any}(nothing)          # P_perp^L (H Theta) -- wanted by BOTH `skl` calls
    AR = Ref{Any}(nothing)          # (H Theta) P_perp^R
    # ⚠ `share = false` IS THE OLD BEHAVIOUR, KEPT AS AN A/B AND NOTHING ELSE. A performance claim
    # that can only be reproduced by checking out an old commit is not reproducible in practice;
    # this makes "how much did sharing buy?" a switch that `benchmarks/rsvd_parallel.jl` can flip
    # inside ONE process -- which matters here, because a cold Julia process spends ~10 minutes
    # compiling before it can time anything.
    _ht() = share ? (HT[] === nothing && (HT[] = buildHT()); HT[]) : buildHT()
    function _al()
        share && AL[] !== nothing && return AL[]
        H = _ht()
        c = contract(f.U0', (1, 2), H, (1, 2))             # (bond, site_r, link_r)
        A = to_concrete(H - to_concrete!(contract(f.U0, (3,), c, (1,))))
        share && (AL[] = A)
        return A
    end
    function _ar()
        share && AR[] !== nothing && return AR[]
        H = _ht()
        c = contract(H, (3, 4), f.V0', (2, 3))             # (link_l, site_l, bond)
        A = to_concrete(H - to_concrete!(contract(c, (3,), f.V0, (1,))))
        share && (AR[] = A)
        return A
    end
    skl = Om -> to_concrete!(contract(_al(), (3, 4), Om', (2, 3)))       # (link_l, site_l, g)
    skr = Om -> to_concrete(permutedims(contract(_ar(), (1, 2), Om', (1, 2)), (3, 1, 2)))
    return skl, skr
end

"""
    sketch_h_left(f, h, i, lenv, renv, Om) -> TLArray

`Y = (P_perp (H Theta)) Om†`, legs `(link_l, site_l, g)`: the discarded-space component of
`H Theta` sketched from the right, with `Om` in `V0`'s layout. A standalone call rebuilds
`H*Theta`; sweeps use the cached closures.
"""
sketch_h_left(f::BondFrame, h::XXZChain, i::Int,
              lch::ChannelSet, rch::ChannelSet, Om) =
    first(_sketch_closures(f, () -> apply_h_two_site(frame_theta(f), h, i, lch, rch)))(Om)

"""
    sketch_h_right(f, h, i, lenv, renv, Om) -> TLArray

Mirror of [`sketch_h_left`](@ref): the right matricization of `(H Theta) P_perp^R` with
`P_perp^R = I - V0' V0`, sketched from the left. `Om` carries `U0`'s layout
`(link_l, site_l, g)` and the result is `(g, site_r, link_r)`.
"""
sketch_h_right(f::BondFrame, h::XXZChain, i::Int,
               lch::ChannelSet, rch::ChannelSet, Om) =
    last(_sketch_closures(f, () -> apply_h_two_site(frame_theta(f), h, i, lch, rch)))(Om)

# ── the sector-graded Gaussian sketch (`GaussRandMat`, `RSVDpreBE0SiQS.m:557`) ──
# `CompRatio·Dpre` columns in the frame's complement (per sector, by complement dimension) and the
# rest inside the frame, so empty-but-reachable sectors are sampled through `H` by construction.

"""
    _randomize_like(F, rng) -> TLArray

`F`'s sparsity pattern with Gaussian payloads. Only the blocks the symmetry allows are
ever touched.
"""
function _randomize_like(F, rng::AbstractRNG)
    T = eltype(F) <: Complex ? eltype(F) : ComplexF64
    return TLArray(symm(F), copy(F.qlabels), copy(F.wmatdata), copy(F.wmatinfo),
                   [randn(rng, T, size(r)) for r in F.RMTs], F.inds, F.spaces)
end

"""
    _alloc_columns(dims, budget) -> Dict

Split `budget` columns across sectors in proportion to `dims`, capped per sector at its
own dimension. `ceil` per sector as in `GaussRandMat`, so a small budget still reaches
every sector that has room rather than rounding the small ones to zero -- which is the
whole point when the sector we most need is the empty one.
"""
function _alloc_columns(dims::Dict, budget::Int)
    tot = sum(values(dims); init = 0)
    (tot == 0 || budget <= 0) && return Dict(q => 0 for q in keys(dims))
    return Dict(q => min(d, ceil(Int, d / tot * budget)) for (q, d) in dims)
end

"""
    _boost!(cols, boost, cap) -> Dict

Add `boost[q]` columns to `cols[q]`, capped at `cap[q]`, and return what did not fit.

A sector absent from `cap` is not reachable from this side, so it is skipped rather than
created: Test B targets a sector by its residual, and a residual can never name a charge the
fusion forbids, but the guard keeps that true by construction instead of by argument.
"""
function _boost!(cols::Dict, boost, cap::Dict)
    left = Dict{Any, Int}()
    for (q, b) in boost
        haskey(cap, q) || continue
        have = get(cols, q, 0)
        add = clamp(b, 0, max(cap[q] - have, 0))
        cols[q] = have + add
        b > add && (left[q] = b - add)
    end
    return left
end

"Keep `counts[q]` columns of sector `q` on `leg`, dropping sectors allotted none."
function _trim_per_sector(Q, leg::Int, counts::Dict)
    keep = s -> (n = get(counts, s, 0); n <= 0 ? nothing : 1:n)
    any(v > 0 for v in values(counts)) || return nothing
    return to_concrete(getsub(Q, leg, keep))
end

"""
    full_local_basis(frame, side; tag="cbeF") -> TLArray

The complete local basis for one side of a bond, in the frame's layout, with `g` the full fused
dimension. Fed to the same sketch closures it turns RSVD-CBE into plain CBE (the A/B reference).
"""
function full_local_basis(frame, side::Symbol; tag::AbstractString = "cbeF")
    side in (:left, :right) || throw(ArgumentError("side must be :left or :right"))
    a, b = side === :left ? (1, 2) : (2, 3)
    F = fusion_basis(frame, a, b; tag = tag)
    # `fusion_basis` returns (a, b, fused); a right frame carries its bond leg first.
    return side === :left ? F : to_concrete(permutedims(F, (3, 1, 2)))
end

"""
    _frame_sector_dims(frame, side, F, where) -> (dfull, dT)

Per charge sector of `F`'s fused leg: the local dimension and the frame's width in it. Throws if
the frame is wider than its local space or holds a sector the space lacks. Labels are aligned to
the fused leg's arrow (`align_charge`).
"""
function _frame_sector_dims(frame, side::Symbol, F, where::AbstractString)
    fl = side === :left ? 3 : 1
    dfull = Dict(q => d for (q, d) in F.spaces[3])
    syms_ = symm(frame)
    fdir, bdir = F.inds[3].dir, frame.inds[fl].dir
    dT = Dict(q => sector_dim(frame, fl, align_charge(syms_, q, fdir, bdir)) for q in keys(dfull))
    for (q, d) in dT
        d <= dfull[q] || throw(ErrorException(
            "$where: the frame holds $d directions in sector $q but the fused local space has " *
            "only $(dfull[q]) -- a frame cannot exceed its local space (frame width " *
            "$(leg_dim(frame, fl)), local $(sum(values(dfull))))"))
    end
    sum(values(dT); init = 0) == leg_dim(frame, fl) || throw(ErrorException(
        "$where: the frame holds $(leg_dim(frame, fl)) directions but only " *
        "$(sum(values(dT); init = 0)) lie in sectors its local space has"))
    return dfull, dT
end

"""
    _cap_to_local_room(Q, U, side) -> U, a trimmed U, or nothing

At most `dim_local(q) - width_Q(q)` leading columns of `U` per sector, so `[Q | U]` cannot exceed
its local space (Telum's cutoff is relative to the global sigma_max, so roundoff in a saturated
sector can otherwise survive). Returns `U` itself when it fits.
"""
function _cap_to_local_room(Q, U, side::Symbol)
    U === nothing && return nothing
    fl = side === :left ? 3 : 1
    a, b = side === :left ? (1, 2) : (2, 3)
    F = fusion_basis(Q, a, b; tag = "cbeRoom")
    dfull, dT = _frame_sector_dims(Q, side, F, "_cap_to_local_room")
    syms_ = symm(Q)
    counts = Dict{Any, Int}()
    over = false
    for (q, d) in dfull
        uq = align_charge(syms_, q, F.inds[3].dir, U.inds[fl].dir)
        du = sector_dim(U, fl, uq)
        du == 0 && continue
        counts[uq] = min(du, d - dT[q])
        over |= counts[uq] < du
    end
    over || return U
    return _trim_per_sector(U, fl, counts)
end

"""
Debug switch, off by default: [`_check_frame`](@ref) asserts at every frame-growth site that the
grown frame still fits its local space. It builds a fusion basis per call, so it is for
localising a non-isometric frame (IT-1 L=16 tol_basis 1e-7, s37test 17341472), not production.
"""
const _FRAME_CHECKS = Ref(false)

"`frame` unchanged; with `_FRAME_CHECKS[]`, first [`_frame_sector_dims`](@ref) at site `where`."
function _check_frame(frame, side::Symbol, where::AbstractString)
    (_FRAME_CHECKS[] && frame !== nothing) || return frame
    a, b = side === :left ? (1, 2) : (2, 3)
    _frame_sector_dims(frame, side, fusion_basis(frame, a, b; tag = "cbeChk"), where)
    return frame
end

# ── from src/RSVDCBEBondUpdate/expansion/cbe_expand.jl ──

# cbe_expand.jl: `cbe_expand`, the two-sided SVD selection (DMRG / TDVP-CBE) (from `cbe_core.jl`, included right after it).

"""
    cbe_expand(f, skl, skr; kwargs...) -> CBEExpansion

The selection half of controlled bond expansion, with the H-action supplied as two closures:
`skl(Om)` gives `(H Theta) Om†` in `U0`'s layout (the left candidates), `skr(Om)` the mirror.
The gate path and the environment path share this selection code. The keywords are internal
settings of the callers (DMRG, TDVP-CBE, the gate path), not a public API.
"""
function cbe_expand(f::BondFrame, skl, skr;
                    # Concrete internal defaults (the TDVP-CBE baselines' fixed-count selection); callers that need
                    # other values pass them.
                    dex::Int = 0,
                    growth::Float64 = 2.0,
                    dover::Union{Nothing, Int} = nothing,
                    comp_ratio::Union{Float64, Nothing} = 0.5,
                    sulz_cap::Bool = false,
                    rmax::Int = 0,
                    preselect_only::Bool = false,
                    exact::Bool = false,
                    stol_pre::Float64 = 1e-10,
                    stol_fnl::Float64 = 1e-13,
                    # The step's allocation and ledger. `nothing` is the LEGACY path: fixed
                    # counts, no residual tests, exactly the historical selection. An
                    # allocation switches on Tests A/B and the rank comes off the spectrum.
                    alloc::Union{Nothing, BasisAllocation} = nothing,
                    ledger::Union{Nothing, AdaptiveLedger} = nothing,
                    workcap::Int = 0,
                    # Adaptive path only (`alloc !== nothing`): Test B sets the sketch width --
                    # see `_certified_range(...; thin)`. `false` is the old full-rank widening.
                    thin_search::Bool = true,
                    rng::AbstractRNG = MersenneTwister(0x5EED))
    U0, V0 = f.U0, f.V0
    r = f.old_rank
    none() = CBEExpansion(U0, V0, 0, 0, 0.0, 0.0)

    # Room to expand per side: the fused local dimension minus the frame's rank. Abelian charges use the
    # product of leg dimensions (exact there); SU(2) counts multiplets, so it must ask `reachable_sectors`.
    fused_dim(t, a, b) = symmetry_mode() === :SU2 ?
        sum(d for (_, d) in reachable_sectors(t, a, b); init = 0) :
        leg_dim(t, a) * leg_dim(t, b)
    room_l = fused_dim(U0, 1, 2) - r
    room_r = fused_dim(V0, 2, 3) - r
    # Growth is neighbourhood-coupled as in `RSVDpreBE0SiQS.m` (target `growth` x the neighbouring max
    # rank), so a wide bond pulls its narrow neighbours up; `growth = 2.0` rather than the DMRG 1.1.
    dmax = max(leg_dim(U0, 1), r, leg_dim(V0, 3))
    budget = dex <= 0 ? max(ceil(Int, growth * dmax) - r, 1) : dex
    # No `2r` (Sulz) bound by default: growth is limited by `room` and by `maxdim` at the truncation.
    # `sulz_cap = true` keeps the global bound for an A/B.
    rm = rmax > 0 ? rmax : r
    sulz_cap && (budget = min(budget, max(2 * rm - r, 0)))

    # At `dex = 0` / `dover = nothing` the admitted count and the sketch width are read off the
    # preselection spectrum (range finder) instead of fixed guesses; positive values pin them.
    cap_l, cap_r = dex > 0 ? (min(budget, room_l), min(budget, room_r)) : (room_l, room_r)
    if sulz_cap
        lim_ex = max(2 * rm - r, 0)
        cap_l, cap_r = min(cap_l, lim_ex), min(cap_r, lim_ex)
    end
    # PLAN §5: THE WORK CEILING ALSO BOUNDS WHAT IS RETAINED, not only how wide the sketch is.
    # A frame of width `r` may grow to `workcap` and no further; `workcap >= 2r`, so the
    # augmented rank-`2r` basis always fits. Where this cap is the binding one and the
    # candidates fill it, `workcap_reached` is set; whether accuracy suffered is Test A's call
    # (the directions it cut are in its tail), so a cap is reported, never silently accepted.
    wbind_l = wbind_r = false
    if alloc !== nothing && workcap > 0
        wk = max(workcap - r, 0)
        wbind_l, wbind_r = wk < cap_l, wk < cap_r
        cap_l, cap_r = min(cap_l, wk), min(cap_r, wk)
    end
    (cap_l <= 0 && cap_r <= 0) && return none()      # both frames already complete

    npre = preselect_only ? max(budget, 1) :
           dover === nothing ? ceil(Int, 1.2 * budget) : budget + dover
    # The probe for the LEFT expansion lives on the RIGHT frame's fused space and vice versa,
    # so that is where each side's width ceiling comes from. Widening past it buys nothing:
    # `sector_graded_sketch` caps every sector at its own dimension and simply returns fewer
    # columns, which `_range_find` reads as "the local space is exhausted".
    full_l, full_r = room_l + r, room_r + r

    # ── preselection: sketch H from each side, project off the frame ──
    # The probe goes on `H Theta`'s opposite leg, so the expanded side stays exact; `exact = true` uses
    # the complete fused basis (plain CBE) and makes the probe settings inert.
    probe(frame, side, w, boost) = exact ? full_local_basis(frame, side) :
                                   sector_graded_sketch(frame, side, w; boost = boost,
                                                        comp_ratio = comp_ratio, rng = rng)
    # An explicit `dover` pins the width, and `exact` has no width to pin -- the probe is the
    # complete basis, so there is nothing to widen and the first pass already sees everything.
    wcap_l, wcap_r = dover === nothing && !exact ? (full_r, full_l) : (npre, npre)
    # THE WORK CEILING IS THE ONE RESOURCE LIMIT, and it is what keeps the certified loop an
    # RSVD. Test B widens until the residual fits, and at a bond whose spectrum does not decay
    # that is satisfied only by the complete fused basis -- exactly the cost the sketch exists
    # to avoid. `workcap` (`work_ceiling`, set by the caller from `maxdim`) stops the
    # doubling there and the shortfall is REPORTED as `workcap_reached` rather than paid for.
    if alloc !== nothing && workcap > 0
        wcap_l, wcap_r = min(wcap_l, workcap), min(wcap_r, workcap)
    end
    # The exact probe IS the complete local basis, so there is no range left for Test B to
    # find, and its holdout would be drawn from that same deterministic basis.
    alloc_b = exact ? nothing : alloc

    QL, err_l, njl = if cap_l > 0
        _certified_range((w, b) -> probe(V0, :right, w, b), 1,
                         Y -> _preselect_left(U0, skl(Y), stol_pre), 3,
                         X -> perp_component(U0, X), skl, :left,
                         npre, wcap_l, cap_l, alloc_b, ledger; thin = thin_search)
    else
        (nothing, 0.0, 0)
    end

    QR, err_r, njr = if cap_r > 0
        _certified_range((w, b) -> probe(U0, :left, w, b), 3,
                         Y -> _preselect_right(V0, skr(Y), stol_pre), 1,
                         X -> perp_component_right(V0, X), skr, :right,
                         npre, wcap_r, cap_r, alloc_b, ledger; thin = thin_search)
    else
        (nothing, 0.0, 0)
    end
    err_pre = sqrt(err_l^2 + err_r^2)

    (QL === nothing && QR === nothing) && return none()

    # THE BUDGET IS READ OFF THE SPECTRUM HERE, not guessed before it: `n_j` counts the
    # candidate directions that cleared `stol_pre` in the preselection SVD, and the per-side
    # cap is the hard local limit (plus an explicit `dex`/`sulz_cap` when one was asked for).
    dex_l, dex_r = min(njl, cap_l), min(njr, cap_r)
    ledger === nothing || ((wbind_l && njl >= cap_l) || (wbind_r && njr >= cap_r)) &&
        (ledger.workcap_reached = true)

    # ── final selection: rank the candidates by their weight in H Theta ──
    # Per side, as the reference's directional CBE does: the joint ranking is used only where it can
    # serve a side's whole budget, otherwise the side keeps its own weight-ordered candidates.
    joint, resj, err_fnl = 0, nothing, 0.0
    if !preselect_only && njl > 0 && njr > 0
        UMU = to_concrete(permutedims(contract(skl(QR), (1, 2), QL', (1, 2)), (2, 1)))
        # EMPTY: the left and right candidate spaces share no charge sector, so no pair of them
        # can carry amplitude (the reference's `Empty` branch, :461; the same constraint
        # `pairable_charges` enforces for the random fill). Now only disables the JOINT ranking.
        if length(UMU.qlabels) > 0 && norm(UMU) >= 1e-11
            joint = min(dex_l, dex_r, njl, njr)
            if joint > 0
                resj = svd(UMU, (1,); Nkeep = joint, cutoff = stol_fnl, get_lists = true)
                err_fnl = _trunc_weight(resj)
                # Test A: keep the fewest ranked directions whose discarded weight fits the bond's `tol_basis`
                # share; a shortfall is reported, not hidden.
                if alloc !== nothing
                    k, tail = _tail_rank(resj.kept_list, alloc.scale, alloc.tail, joint;
                                         base = _sq_weight(resj.trunc_list))
                    if ledger !== nothing
                        ledger.max_tail = max(ledger.max_tail, tail)
                        tail <= alloc.tail || _unmet!(ledger, :tail)
                    end
                    if k < joint
                        joint = k
                        resj = k > 0 ?
                               svd(UMU, (1,); Nkeep = k, cutoff = stol_fnl, get_lists = true) :
                               nothing
                        err_fnl = resj === nothing ? 0.0 : _trunc_weight(resj)
                    end
                    # THE TAIL RANK MUST BIND BOTH SIDES, not just the joint rotation. The
                    # per-side fallback below trims to `dex_*`, which is a headroom bound; a
                    # side with more headroom than `k` would otherwise admit directions the
                    # tail test just certified as unnecessary, and the test would silently
                    # apply to one side only.
                    dex_l, dex_r = min(dex_l, joint), min(dex_r, joint)
                end
            end
        end
    end

    # Test A on a side the joint ranking did not cover: rank its candidates by their exact weight against
    # the opposite side's full local basis (only on such bonds).
    oneL = alloc !== nothing && !preselect_only && njl > 0 && resj === nothing && dex_l > 0
    oneR = alloc !== nothing && !preselect_only && njr > 0 && resj === nothing && dex_r > 0
    UL = oneL ? _one_sided_tail(to_concrete(contract(QL', (1, 2),
                                                     skl(full_local_basis(V0, :right)), (1, 2))),
                                alloc, dex_l, stol_fnl, ledger) : nothing
    UR = oneR ? _one_sided_tail(to_concrete(contract(QR', (2, 3),
                                                     skr(full_local_basis(U0, :left)), (2, 3))),
                                alloc, dex_r, stol_fnl, ledger) : nothing

    TLC = if njl == 0
        nothing
    elseif resj !== nothing && dex_l <= joint
        to_concrete!(contract(QL, (3,), resj.U, (1,)))
    elseif oneL
        UL === nothing ? nothing : to_concrete!(contract(QL, (3,), UL, (1,)))
    else
        _trim_total(QL, 3, dex_l)
    end
    TRC = if njr == 0
        nothing
    elseif resj !== nothing && dex_r <= joint
        to_concrete!(contract(resj.Vd, (2,), QR, (1,)))
    elseif oneR
        UR === nothing ? nothing : to_concrete!(contract(UR, (1,), QR, (1,)))
    else
        _trim_total(QR, 1, dex_r)
    end

    # Re-establish orthonormality and orthogonality to the frame before the direct sum. See
    # the guard block above `_trim_total` for why this is not redundant with the preselection's
    # `RepairOrtho`: that pass ran on `Q`, this one runs on what the final selection made of it.
    TLC = _guarded_block(U0, TLC, :left)
    TRC = _guarded_block(V0, TRC, :right)

    if ledger !== nothing
        ledger.n_selected += (TLC === nothing ? 0 : leg_dim(TLC, 3)) +
                             (TRC === nothing ? 0 : leg_dim(TRC, 1))
    end

    (TLC === nothing && TRC === nothing) && return CBEExpansion(U0, V0, 0, 0, err_pre, 0.0)

    ltag, rtag = U0.inds[3].itags, V0.inds[1].itags
    U_ex, n_l = if TLC === nothing
        (U0, 0)
    else
        (to_concrete(oplus([U0, to_concrete(setitag(TLC, 3, ltag))], (3,))),
         leg_dim(TLC, 3))
    end
    V_ex, n_r = if TRC === nothing
        (V0, 0)
    else
        (to_concrete(oplus([V0, to_concrete(setitag(TRC, 1, rtag))], (1,))),
         leg_dim(TRC, 1))
    end

    # No rotation of the expanded frame: an expansion is lossless, and rank grows only where the state
    # is evolved in the widened space.
    if sulz_cap
        # Only checked when the bound is deliberately switched on, and in its GLOBAL form
        # (`2*rmax`) -- the per-bond reading was wrong, see the budget block above.
        lim = 2 * rm
        leg_dim(U_ex, 3) <= lim || error(
            "CBE expansion broke the global Sulz bound on the left: " *
            "$(leg_dim(U_ex, 3)) > 2*rmax = $lim")
        leg_dim(V_ex, 1) <= lim || error(
            "CBE expansion broke the global Sulz bound on the right: " *
            "$(leg_dim(V_ex, 1)) > 2*rmax = $lim")
    end

    return CBEExpansion(_check_frame(U_ex, :left, "cbe_expand U_ex"),
                        _check_frame(V_ex, :right, "cbe_expand V_ex"), n_l, n_r, err_pre, err_fnl)
end

# ── orthonormality guard on the appended blocks ──
# Project the frame out, then orthonormalise (Telum has no QR, so `svd(...).U`); cutoff 1e-14
# drops columns that collapse, and a dropped column shows in `n_new_*`.
_reortho_left(U0, C) = _reortho(C, X -> perp_component(U0, X),
                                R -> svd(R, (1, 2), _TAG_OL, _TAG_OLR; cutoff = 1e-14).U)

_reortho_right(V0, C) = _reortho(C, X -> perp_component_right(V0, X),
                                 R -> permutedims(
                                     svd(R, (2, 3), _TAG_OR, _TAG_ORR; cutoff = 1e-14).U,
                                     (3, 1, 2)))

function _reortho(C, perp, orth)
    C === nothing && return nothing
    R = to_concrete(perp(C))
    norm(R) <= _CBE_EPS * max(norm(C), 1.0) && return nothing
    return to_concrete(orth(R))
end

"""
    _guarded_block(frame, C, side) -> the block `C` cleared for `[frame | C]`, or nothing

Orthonormal and orthogonal to `frame` (`_reortho_*`), then at most the local room of each sector
(`_cap_to_local_room`). The second half is not redundant with the first: `_reortho`'s SVD cutoff is
relative to the GLOBAL sigma_max, so in a sector the frame already fills, its orthogonality loss
survives the cut and is renormalised into a fake unit column. MEASURED (IT_FRAME_CHECKS, IT-1 L=20
domain wall, panel, bug4): `cbe_expand V_ex` held 10 directions in a sector of local dimension 9
(d50f_trap 17346775, trapezoid, tol_basis 1e-5) and `cbe_expand U_ex` 253 in the full half-chain
S^z = 0 sector of 252 (d50f_rad3 17346774, radau3, 1e-3); the closure frontier's Krylov enrichment
then refused the frame and the runs crashed (it1r 17345037, 17345039).
"""
_guarded_block(frame, C, side::Symbol) =
    _cap_to_local_room(frame, side === :left ? _reortho_left(frame, C) :
                                               _reortho_right(frame, C), side)


"Keep at most `n` columns of `Q` on `leg`, walking sectors in Telum's order."
function _trim_total(Q, leg::Int, n::Int)
    n <= 0 && return nothing
    leg_dim(Q, leg) <= n && return Q
    left = n
    keep = Dict{Any, Int}()
    for (q, d) in Q.spaces[leg]
        keep[q] = min(d, left)
        left -= keep[q]
    end
    any(v > 0 for v in values(keep)) || return nothing
    return to_concrete(getsub(Q, leg, s -> (k = get(keep, s, 0); k == 0 ? nothing : 1:k)))
end

# ── from src/RSVDCBEBondUpdate/expansion/cbe_options.jl ──

# cbe_options.jl: prebuilt-stack wrappers, the shared run configuration, the S step and the sketched gate action (from `cbe_core.jl`, included right after it).

# ── prebuilt-stack convenience wrappers ──────────────────────────────────────
#
# The kernels above take the channels on the two links they actually use, so a sweep can
# CARRY them (O(1) per bond) instead of rebuilding a stack (O(L) per bond). These wrappers
# keep the stack-indexed call sites -- the tests, and anything not sweeping -- unchanged.

sketch_h_left(f::BondFrame, h::XXZChain, i::Int,
              lenv::LeftEnvStack, renv::RightEnvStack, Om) =
    sketch_h_left(f, h, i, left_channels(lenv, i), right_channels(renv, i + 2), Om)

sketch_h_right(f::BondFrame, h::XXZChain, i::Int,
               lenv::LeftEnvStack, renv::RightEnvStack, Om) =
    sketch_h_right(f, h, i, left_channels(lenv, i), right_channels(renv, i + 2), Om)

cbe_expand(f::BondFrame, h::XXZChain, i::Int,
           lenv::LeftEnvStack, renv::RightEnvStack; kwargs...) =
    cbe_expand(f, h, i, left_channels(lenv, i), right_channels(renv, i + 2); kwargs...)

"""
    sketch_bond_left(f, gate, Om) -> TLArray

`Y = (gate Theta) Om†`, legs `(link_l, site_l, g)` -- the left matricization of the gate
action sketched from the right. `Om` carries `V0`'s layout `(g, site_r, link_r)`, i.e. it is
a candidate right frame with `g` columns instead of `r`; `Om = f.V0` recovers the exact left
factor of `gate Theta`.

Contraction order: absorb `Om` into the gate's right BRA leg, then close the gate's right
KET leg and `link_r` against `V0`, then `K0 = U0 S0`. Nothing rank-4 is built.
"""
function sketch_bond_left(f::BondFrame, gate, Om)
    _check_gate(f, gate)
    tl, tr = f.site_l.itags, f.site_r.itags
    HT = apply_gate(gate, frame_theta(f), tl, tr)       # (link_l, site_l, site_r, link_r)
    c  = contract(f.U0', (1, 2), HT, (1, 2))            # (bond, site_r, link_r)
    A  = to_concrete(HT - to_concrete!(contract(f.U0, (3,), c, (1,))))
    return to_concrete!(contract(A, (3, 4), Om', (2, 3)))          # (link_l, site_l, g)
end

"""
    sketch_bond_right(f, gate, Om) -> TLArray

Mirror of [`sketch_bond_left`](@ref): `Om` carries `U0`'s layout `(link_l, site_l, g)` and
the result is `(g, site_r, link_r)`.
"""
function sketch_bond_right(f::BondFrame, gate, Om)
    _check_gate(f, gate)
    tl, tr = f.site_l.itags, f.site_r.itags
    HT = apply_gate(gate, frame_theta(f), tl, tr)
    c  = contract(HT, (3, 4), f.V0', (2, 3))            # (link_l, site_l, bond)
    A  = to_concrete(HT - to_concrete!(contract(c, (3,), f.V0, (1,))))
    Y  = contract(A, (1, 2), Om', (1, 2))               # (site_r, link_r, g)
    return to_concrete(permutedims(Y, (3, 1, 2)))       # (g, site_r, link_r)
end

"""
    cbe_expand_bond(f, gate; kwargs...) -> CBEExpansion

Controlled bond expansion of the frames at one bond, with the bare two-site gate standing in
for the environment-dressed effective Hamiltonian. Keywords are [`cbe_expand`](@ref)'s,
unchanged -- this is the same selection with different sketch closures.

The sketch probes the PROJECTOR (`P_perp (H Theta)`), which is the only path; there is no
`project_first` switch any more.
"""
cbe_expand_bond(f::BondFrame, gate; kwargs...) =
    cbe_expand(f,
               Om -> sketch_bond_left(f, gate, Om),
               Om -> sketch_bond_right(f, gate, Om); kwargs...)

# ── from src/RSVDCBEBondUpdate/expansion/cbe_bond_update.jl ──

# cbe_bond_update.jl: the gate-based CBE bond update and its sweep (from `cbe_core.jl`, included right after it).

# ── the bond update ──────────────────────────────────────────────────────────

"""
    cbe_bond_update(f::BondFrame, gate, tau; kwargs...) -> NamedTuple

One CBE-basis Galerkin update on the bond `f` snapshots, a drop-in for
`BondUpdateBUG.kls_bond_update` (same fields, plus `err_pre` / `err_fnl`). `maxdim`,
`trunc_thresh`, `s_tau`, `maxiter`, `tol` as there; `expand = false` freezes the frames; other
keywords forward to `cbe_expand`. `s_iters = 1` is one expansion then Lanczos, `n > 1` repeats
the expansion, `-g <= 0` uses [`_expanding_krylov`](@ref) with `g` growth passes.
"""
function cbe_bond_update(f::BondFrame, gate, tau::ComplexF64;
                         maxdim::Int = 200,
                         trunc_thresh::Float64 = 1e-14,
                         s_tau::Union{Nothing, ComplexF64} = nothing,
                         expand::Bool = true,
                         maxiter::Int = 30,
                         tol::Float64 = 1e-15,
                         s_iters::Int = 1,
                         iter_tol::Float64 = 1e-12,
                         s_reorth::Bool = false,
                         s_probe::Symbol = :residual,
                         rng::AbstractRNG = MersenneTwister(0x5EED),
                         expand_kwargs...)
    tau_s = s_tau === nothing ? tau : s_tau
    tl, tr = f.site_l.itags, f.site_r.itags
    theta0 = frame_theta(f)

    # Operator applications are counted here (`expv` returns only the vector), so cost can be split from
    # the number of steps.
    nmv = Ref(0)

    # The two closures that are all a mode gets to change. The gate path supplies the bare
    # two-site gate here (its environments are the identity by canonical form -- see
    # `CBEBugOptions`); the MPO path supplies the environment-dressed `apply_h_two_site`.
    applyH = th -> apply_gate(gate, th, tl, tr)
    expand_at = fr -> cbe_expand_bond(fr, gate; rng = rng, expand_kwargs...)

    # The Galerkin step in a GIVEN basis. Note `S_start` is rebuilt from `theta0` every time it
    # is called -- see the loop below for why that is the whole correctness argument.
    function galerkin(U_aug, V_aug)
        S_start = to_concrete!(contract(contract(U_aug', (1, 2), theta0, (1, 2)),
                                       (2, 3), V_aug', (2, 3)))
        function apply_s(x)
            nmv[] += 1
            theta = to_concrete((U_aug * x) * V_aug)
            evolved = apply_gate(gate, theta, tl, tr)
            proj = contract(U_aug', (1, 2), evolved, (1, 2))
            return to_concrete!(contract(proj, (2, 3), V_aug', (2, 3)))
        end
        return expv(apply_s, tau_s, S_start; hermitian = true, maxiter = maxiter,
                    tol = tol, reorth = s_reorth)
    end

    U_aug, V_aug = f.U0, f.V0
    err_pre, err_fnl, n_iter, iter_weight = 0.0, 0.0, 0, 0.0
    local S_new

    if !expand
        S_new = galerkin(U_aug, V_aug)
    elseif s_iters <= 0
        # ── GROWTH INSIDE THE LANCZOS ────────────────────────────────────────────────
        # `s_iters <= 0` selects the expanding-basis solver; see `_expanding_krylov`.
        U_aug, V_aug, S_new, info = _expanding_krylov(
            f, applyH, expand_at, Propagate(tau_s), theta0, nmv;
            maxiter = maxiter, tol = tol, grow_iters = -s_iters, probe = s_probe)
        err_pre, err_fnl = info.err_pre, info.err_fnl
        n_iter, iter_weight = info.n_grow, info.beta_final
    else
        # ── one expansion, then a plain Lanczos (the default); `n > 1` repeats it ──
        # Each pass recomputes the budget from the current rank, compounding the growth schedule;
        # `galerkin` always starts from the original `theta0`, so the exponential is applied once.
        probe = f
        for k in 1:s_iters
            ex = expand_at(probe)
            U_prev, V_prev = U_aug, V_aug
            U_aug, V_aug = ex.U_ex, ex.V_ex
            err_pre, err_fnl, n_iter = ex.err_pre, ex.err_fnl, k

            S_new = galerkin(U_aug, V_aug)

            k == s_iters && break
            (ex.n_new_l + ex.n_new_r) == 0 && break
            iter_weight = _new_direction_weight(U_prev, V_prev, U_aug, V_aug, S_new)
            iter_weight < iter_tol && break

            probe = _reframe(f, U_aug, S_new, V_aug)
        end
    end

    res = svd(S_new, (1,); cutoff = max(trunc_thresh, 1e-14), Nkeep = maxdim,
              get_lists = true)
    tag = f.link_mid.itags
    left_core  = to_concrete(setitag(to_concrete(U_aug * res.U), 3, tag))
    right_core = to_concrete(setitag(to_concrete((res.S * res.Vd) * V_aug), 1, tag))

    svals = [s for (s, _, _, _) in res.kept_list]
    return (; left_core, right_core, U_aug, V_aug, S_new,
            n_new_k = leg_dim(U_aug, 3) - f.old_rank,
            n_new_l = leg_dim(V_aug, 1) - f.old_rank,
            keep = leg_dim(left_core, 3),
            aug_k = leg_dim(U_aug, 3), aug_l = leg_dim(V_aug, 1),
            svals, discarded = _trunc_weight(res),
            err_pre, err_fnl, n_matvec = nmv[], n_iter, iter_weight)
end

"""
    _expanding_krylov(f, applyH, expand, solver, theta0, nmv; ...) -> (U, V, S, info)

Lanczos for `exp(tau*H_eff)` whose basis grows with the Krylov space: before `H` is applied to
`v_k`, the frames are expanded with `v_k` as the probe, so `H v_k` fits. Stops when `beta_k < tol`
or `grow_iters` passes have run. Full reorthogonalisation is required because the projected
operator changes at every growth pass.
"""
function _expanding_krylov(f::BondFrame, applyH, expand, solver::SStepSolver, theta0, nmv;
                           maxiter::Int, tol::Float64, grow_iters::Int,
                           probe::Symbol = :residual)
    project(U, V, th) = to_concrete!(contract(contract(U', (1, 2), th, (1, 2)),
                                             (2, 3), V', (2, 3)))

    # Embed a core from the (Uo,Vo) basis into (Un,Vn) by computed overlaps (robust to a rotation by the
    # orthonormality guard); the overlaps are built once per growth pass, not per stored vector.
    embed_ops(Uo, Vo, Un, Vn) =
        (to_concrete!(contract(Un', (1, 2), Uo, (1, 2))),         # PL (new_l, old_l)
         to_concrete!(contract(Vo, (2, 3), Vn', (2, 3))))         # PR (old_r, new_r)

    embed_with(PL, PR, S) =
        to_concrete!(contract(contract(PL, (2,), S, (1,)), (2,), PR, (1,)))

    U, V = f.U0, f.V0
    S0 = project(U, V, theta0)
    beta0 = norm(S0)
    err_pre, err_fnl, n_grow = 0.0, 0.0, 0
    # `value = NaN` is NOT decoration: every other exit carries it (the Ritz value for
    # `LowestEigen`), and a caller that reads `info.value` -- the ground-state solve does --
    # would hit an UndefVarError on this branch if the field were simply absent here.
    beta0 == 0 && return (U, V, S0,
                          (; err_pre, err_fnl, n_grow, beta_final = 0.0, ranks = Int[],
                             value = NaN))

    basis = Any[to_concrete((1.0 / beta0) * S0)]
    alpha, betas, ranks = Float64[], Float64[], Int[leg_dim(U, 3)]
    b_last = 0.0

    image = nothing        # the PREVIOUS iteration's output, H*v_{k-1}, before orthogonalisation

    for it in 1:maxiter
        # ── step 0: grow the basis around the probe ──
        # `:residual` (default) probes with the orthonormal Lanczos vector; `:image` uses `H v_{k-1}` and
        # measured no different. The Lanczos recurrence itself is unchanged.
        if it <= grow_iters
            pv = (probe === :image && image !== nothing) ? image : basis[it]
            ex = expand(_reframe(f, U, pv, V))
            if (ex.n_new_l + ex.n_new_r) > 0
                Uo, Vo = U, V
                U, V = ex.U_ex, ex.V_ex
                PL, PR = embed_ops(Uo, Vo, U, V)
                for k in eachindex(basis)
                    basis[k] = embed_with(PL, PR, basis[k])
                end
                image === nothing || (image = embed_with(PL, PR, image))
                err_pre, err_fnl, n_grow = ex.err_pre, ex.err_fnl, n_grow + 1
            end
        end
        push!(ranks, leg_dim(U, 3))

        nmv[] += 1
        theta = to_concrete((U * basis[it]) * V)
        w = project(U, V, applyH(theta))
        image = w          # this iteration's output, carried to the next expansion

        a = real(tensor_inner(basis[it], w))
        push!(alpha, a)
        w = to_concrete(w + (-a) * basis[it])
        it > 1 && (w = to_concrete(w + (-betas[it - 1]) * basis[it - 1]))
        for _pass in 1:2, u in basis
            w = to_concrete(w - tensor_inner(u, w) * u)
        end

        # `b_last` is the TERMINATING residual, not the last accepted one. Reporting
        # `betas[end]` instead would show the last beta that was large enough to continue --
        # the opposite of the stopping signal it is meant to document.
        b = norm(w)
        b_last = b
        b < tol && break
        it == maxiter && break
        push!(betas, b)
        push!(basis, to_concrete((1.0 / b) * w))
    end

    # THE ONLY MODE-DEPENDENT LINE IN THE WHOLE FUNCTION. `coeff` already carries `beta0` for
    # the propagator and deliberately does not for the eigensolve, so the accumulation below
    # is identical either way.
    coeff, value = _reduce_krylov(solver, alpha, betas, beta0)
    S = to_concrete(coeff[1] * basis[1])
    for k in 2:length(coeff)
        S = to_concrete(S + coeff[k] * basis[k])
    end
    return (U, V, S, (; err_pre, err_fnl, n_grow, beta_final = b_last, ranks, value))
end

"""
    _new_direction_weight(U_prev, V_prev, U_aug, V_aug, S) -> Float64

Fraction of the evolved core's norm on the directions the last pass added,
`sqrt(‖S‖² - ‖P S Q‖²)/‖S‖`; small means the Krylov space has stopped delivering.
"""
function _new_direction_weight(U_prev, V_prev, U_aug, V_aug, S)
    n = norm(S)
    n == 0 && return 0.0
    PL = to_concrete!(contract(U_prev', (1, 2), U_aug, (1, 2)))     # (prev_l, new_l)
    PR = to_concrete!(contract(V_aug, (2, 3), V_prev', (2, 3)))     # (new_r, prev_r)
    S_old = to_concrete!(contract(contract(PL, (2,), S, (1,)), (2,), PR, (1,)))
    return sqrt(max(n^2 - norm(S_old)^2, 0.0)) / n
end

"""
    _reframe(f, U, S, V) -> BondFrame

`f` with frames and core replaced, so the next sketch probes the evolved state. `old_rank` is the
larger frame rank, so `room` stays a lower bound on each side.
"""
_reframe(f::BondFrame, U, S, V) =
    BondFrame(U, S, V, f.link_l, f.site_l, f.link_mid, f.site_r, f.link_r,
              max(leg_dim(U, 3), leg_dim(V, 1)), f.link_mid_space)

cbe_bond_update(f::BondFrame, gate, tau::Number; kwargs...) =
    cbe_bond_update(f, gate, ComplexF64(tau); kwargs...)

# ── from src/RSVDCBEBondUpdate/operators/mpo_env.jl ──

# The CBE sketch, unchanged in substance: only `H Theta` comes from the MPO now. The
# projector and the sketch are the reference's (`RSVDpreBE0SiQS.m`), and nothing in the
# selection knows how `H` is stored.

"See the [`sketch_h_left`](@ref) for a term list; this is the same sketch on MPO
environments. ⚠ A standalone call rebuilds `H*Theta`; inside a sweep the shared cache in
[`_sketch_closures`](@ref) is what runs."
sketch_h_left(f::BondFrame, mpo::MPO, i::Int, lenv::MPOLink, renv::MPOLink, Om) =
    first(_sketch_closures(f,
        () -> apply_h_two_site(frame_theta(f), mpo, i, lenv, renv)))(Om)

"Mirror of [`sketch_h_left`](@ref) on MPO environments."
sketch_h_right(f::BondFrame, mpo::MPO, i::Int, lenv::MPOLink, renv::MPOLink, Om) =
    last(_sketch_closures(f,
        () -> apply_h_two_site(frame_theta(f), mpo, i, lenv, renv)))(Om)

# ── from src/RSVDCBEBondUpdate/operators/mpo_fold.jl ──

# mpo_fold.jl: fold-omega-first: the sketch that never forms `H*Theta` (from `mpo.jl`, included right after it).

# ── fold-omega-first: the sketch that never forms H*Theta ──
# Contracting `Om` in first makes the cost scale with the sketch width g, not d*chi. It is exact, since
# P_perp and Om act on disjoint legs and commute.
# The three sketches use different Om and cannot share one folded chain, so it wins only for small dex (3g < chi).

"""
    _fold_sketch_closures(f, mpo, i, lenv, renv) -> (skl, skr)

The `(skl, skr)` pair [`cbe_expand`](@ref) wants, with `Om` folded into the contraction chain
instead of applied to a finished `H*Theta`.

`skl(Om)` takes `Om` in `V0`'s layout `(g, site_r, link_r)` and returns `(link_l, site_l, g)`;
`skr(Om)` takes `U0`'s layout `(link_l, site_l, g)` and returns `(g, site_r, link_r)`. Both
return the DISCARDED-space component, i.e. with `P_perp` already applied -- identical to
[`_sketch_closures`](@ref)'s output.
"""
function _fold_sketch_closures(f::BondFrame, mpo::MPO, i::Int,
                               lenv::MPOLink, renv::MPOLink)
    W1, W2 = mpo[i], mpo[i + 1]
    # ⚠ `Theta` IS SHARED, `H*Theta` IS NOT. `frame_theta` is `U0*S0*V0` -- an `O(chi^2 d^2)`
    # object the frame already implies, with no `H` in it. Rebuilding it per call would be pure
    # waste; caching the thing this function exists NOT to build would defeat the purpose.
    Theta = frame_theta(f)                       # (ℓ_l, s_l, s_r, ℓ_r)

    # ── LEFT: fold `Om` through renv and W2, then Theta, then W1 and lenv ────────────────
    function skl(Om)
        Omd = Om'                                # (g, s_bra_r, bra_r)
        # 1. the right cap. At the boundary `renv.E` is absent and `W2`'s `w_r` is the dim-1
        #    leg left by trimming, dropped rather than contracted -- the same asymmetry
        #    `apply_h_two_site` handles, and the reason this is not one expression.
        C = if renv.E === nothing
            X = to_concrete!(contract(Omd, (2,), W2, (3,)))   # (g,bra_r,w_mid,s_ket_r,w_r)
            to_concrete(deleteSingleton(X, 5))               # (g, bra_r, w_mid, s_ket_r)
        else
            X = to_concrete!(contract(Omd, (3,), renv.E, (1,)))  # (g, s_bra_r, w, ket_r)
            # W2 = (w_mid, s_ket_r, s_bra_r, w_r); close `s_bra_r` and `w_r`.
            to_concrete!(contract(X, (2, 3), W2, (3, 4)))     # (g, ket_r, w_mid, s_ket_r)
        end
        # 2. into Theta: close the ket site and ket link on the right.
        D = to_concrete!(contract(Theta, (3, 4), C, (4, 2)))  # (ℓ_l, s_l, g, w_mid)
        # 3. W1 = (w_l, s_ket_l, s_bra_l, w_mid); close the ket site and the shared MPO leg.
        E = to_concrete!(contract(D, (2, 4), W1, (2, 4)))     # (ℓ_l, g, w_l, s_bra_l)
        Y = if lenv.E === nothing
            X = to_concrete(deleteSingleton(E, 3))           # (ℓ_l, g, s_bra_l)
            to_concrete(permutedims(X, (1, 3, 2)))           # (ℓ_l, s_bra_l, g)
        else
            # lenv.E = (bra, w, ket); close the ket link and `w_l`.
            X = to_concrete!(contract(E, (1, 3), lenv.E, (3, 2)))  # (g, s_bra_l, bra)
            _unprime(to_concrete(permutedims(X, (3, 2, 1))), 1)   # (bra, s_bra_l, g)
        end
        return _project_left(f, Y)
    end

    # ── RIGHT: the mirror -- fold `Om` through lenv and W1 first ────────────────────────
    function skr(Om)
        Omd = Om'                                # (bra_l, s_bra_l, g)
        C = if lenv.E === nothing
            X = to_concrete!(contract(Omd, (2,), W1, (3,)))   # (bra_l,g,w_l,s_ket_l,w_mid)
            to_concrete(deleteSingleton(X, 3))               # (bra_l, g, s_ket_l, w_mid)
        else
            X = to_concrete!(contract(Omd, (1,), lenv.E, (1,)))  # (s_bra_l, g, w, ket_l)
            # W1 = (w_l, s_ket_l, s_bra_l, w_mid); close `s_bra_l` and `w_l`.
            Z = to_concrete!(contract(X, (1, 3), W1, (3, 1)))    # (g, ket_l, s_ket_l, w_mid)
            to_concrete(permutedims(Z, (2, 1, 3, 4)))           # (ket_l, g, s_ket_l, w_mid)
        end
        # into Theta: close the ket link and ket site on the left.
        D = to_concrete!(contract(Theta, (1, 2), C, (1, 3)))  # (s_r, ℓ_r, g, w_mid)
        # W2 = (w_mid, s_ket_r, s_bra_r, w_r); close the ket site and the shared MPO leg.
        E = to_concrete!(contract(D, (1, 4), W2, (2, 1)))     # (ℓ_r, g, s_bra_r, w_r)
        Y = if renv.E === nothing
            X = to_concrete(deleteSingleton(E, 4))           # (ℓ_r, g, s_bra_r)
            to_concrete(permutedims(X, (2, 3, 1)))           # (g, s_bra_r, ℓ_r)
        else
            X = to_concrete!(contract(E, (1, 4), renv.E, (3, 2)))  # (g, s_bra_r, bra_r)
            _unprime(X, 3)
        end
        return _project_right(f, Y)
    end

    return skl, skr
end

"`P_perp^L Y` with `P_perp^L = I - U0 U0'`, for `Y` of layout `(link_l, site_l, g)`."
function _project_left(f::BondFrame, Y)
    c = to_concrete!(contract(f.U0', (1, 2), Y, (1, 2)))      # (bond, g)
    return to_concrete(Y - to_concrete!(contract(f.U0, (3,), c, (1,))))
end

"`Y P_perp^R` with `P_perp^R = I - V0' V0`, for `Y` of layout `(g, site_r, link_r)`."
function _project_right(f::BondFrame, Y)
    c = to_concrete!(contract(Y, (2, 3), f.V0', (2, 3)))      # (g, bond)
    return to_concrete(Y - to_concrete!(contract(c, (2,), f.V0, (1,))))
end

"""
The bond expansion at bond `i`, driven by MPO environments.

`fold_omega = true` selects [`_fold_sketch_closures`](@ref), which folds the probe into the
contraction instead of building `H*Theta`. ⚠ It makes `share_ht` MEANINGLESS -- there is no
shared object to cache -- so passing both is refused rather than silently ignoring one.
"""
function cbe_expand(f::BondFrame, mpo::MPO, i::Int,
                    lenv::MPOLink, renv::MPOLink;
                    # ⛔ `nothing`, NOT `true`, SO "NOT PASSED" IS DISTINGUISHABLE FROM "PASSED
                    # true". With a `Bool` default the contradiction check below cannot fire at
                    # all -- `share_ht` is a NAMED kwarg, so it never appears in `kwargs...` and
                    # `haskey(kwargs, :share_ht)` is false even when the caller passed it. That
                    # is a guard that reads as protection and provides none.
                    share_ht::Union{Nothing, Bool} = nothing,
                    fold_omega::Bool = false, kwargs...)
    if fold_omega
        # ⛔ REFUSE THE CONTRADICTION RATHER THAN RESOLVING IT. A caller who sets `fold_omega`
        # alone is asserting nothing about sharing and is not warned; a caller who ALSO passes
        # `share_ht` explicitly believes something false about the run they are about to time,
        # and silently honouring one of the two is how a benchmark reports the wrong arm.
        share_ht === nothing || throw(ArgumentError(
            "fold_omega = true never forms H*Theta, so there is nothing for share_ht to " *
            "share; pass one or the other"))
        return cbe_expand(f, _fold_sketch_closures(f, mpo, i, lenv, renv)...; kwargs...)
    end
    return cbe_expand(f,
                      _sketch_closures(f,
                          () -> apply_h_two_site(frame_theta(f), mpo, i, lenv, renv);
                          share = share_ht === nothing ? true : share_ht)...; kwargs...)
end

# ── from src/RSVDCBEBondUpdate/expansion/bond_expansion.jl ──

# ── the per-bond expansion, at the reference's granularity ──
# One function for every sweep: expand the bond, keep the state unchanged, and push the far
# environment through the widened tensor. The expansion is lossless; new columns carry no amplitude yet.

"""
    BondExpansionInfo

What one [`expand_bond!`](@ref) did at one bond, with field names following the reference's `Info`
struct so Julia and MATLAB traces can be read side by side.

  - `dim_in`, `dim_out`: bond dimension before and after, counting MULTIPLETS (not comparable with a
    U(1) number under SU(2)). Equal means the expansion declined.
  - `n_new`: directions admitted. `err_pre`/`err_fnl`: weight discarded (watch `err_fnl`).
"""
struct BondExpansionInfo
    bond::Int
    dir::Symbol
    dim_in::Int
    dim_out::Int
    n_new::Int
    err_pre::Float64
    err_fnl::Float64
end

"""
    expand_bond!(psi, mpo, i, dir, lch, rch; exkw...) -> (env_far, info)

Expand bond `(i, i+1)` in place, leaving the state unchanged, and return the environment on the side
the sweep is heading toward, pushed through the widened frame. `dir = :right` widens `psi[i+1]`, so
the centre ends on `i` and the returned environment is the right one at link `i+1`; `:left` mirrors it.

`lch` and `rch` are the caller's current environments, so the call costs one push and no rebuild.
"""
function expand_bond!(psi::SymMPS, mpo::MPO, i::Int, dir::Symbol, lch, rch; exkw...)
    dir in (:right, :left) || throw(ArgumentError("dir must be :right or :left, got $dir"))
    L = length(psi)
    1 <= i <= L - 1 || throw(ArgumentError("bond must be in 1:$(L - 1), got $i"))
    psi.center in (i, i + 1) || throw(ArgumentError(
        "expand_bond! needs the orthogonality centre on site $i or $(i + 1), " *
        "found it on $(psi.center)"))

    dim_in = leg_dim(psi[i], 3)
    f  = _frame_from(psi[i], psi[i + 1])
    ex = cbe_expand(f, mpo, i, lch, rch; exkw...)

    if dir === :right
        _absorb_right!(psi, i, ex.V_ex)              # psi[i+1] = V_ex, centre -> i
        env = push_right_channels(rch, mpo, psi[i + 1], i + 1)
        n_new = ex.n_new_r
    else
        _absorb_left!(psi, i, ex.U_ex)               # psi[i] = U_ex, centre -> i+1
        env = push_left_channels(lch, mpo, psi[i], i)
        n_new = ex.n_new_l
    end

    return env, BondExpansionInfo(i, dir, dim_in, leg_dim(psi[i], 3), n_new,
                                  ex.err_pre, ex.err_fnl)
end

# ── the local eigensolve the DMRG sweep needs ──
# Same Krylov space as `lanczos_expv`, reduced to the lowest Ritz pair. It adds a Ritz-residual exit
# (beta_k |c_k|): `lanczos_expv` stops only on breakdown, which wastes the full `maxiter` in DMRG.

"""
    lanczos_lowest(apply, x; maxiter=30, tol=1e-13, restol=1e-10, reorth=true)
        -> (vector, ritz_value, krylov_dim)

Lowest eigenvector of a Hermitian `apply(v) = A*v` in the Krylov space of `x`, normalised. The Ritz
value is returned too, since recomputing it costs another application. It is a variational upper
bound, because the subspace is fixed.

`reorth = true` is the default: without it the recurrence loses orthogonality silently, giving a Ritz
value below the true eigenvalue that looks like better convergence.
"""
function lanczos_lowest(apply, x; maxiter::Int = 30, tol::Float64 = 1e-13,
                        restol::Float64 = 1e-10, reorth::Bool = true)
    xc = to_concrete(ComplexF64(1.0) * x)
    beta0 = norm(xc)
    beta0 == 0 && throw(ArgumentError("lanczos_lowest: start vector has zero norm"))

    v = to_concrete((1.0 / beta0) * xc)
    basis = Any[v]
    alpha = Float64[]
    betas = Float64[]

    w = apply(v)
    a = real(tensor_inner(v, w))
    push!(alpha, a)
    w = to_concrete(w + (-a) * v)

    coeff, value = _reduce_krylov(LowestEigen(), alpha, betas, beta0)
    for _ in 2:maxiter
        b = norm(w)
        b < tol && break
        push!(betas, b)
        v = to_concrete((1.0 / b) * w)
        push!(basis, v)
        w = apply(v)
        a = real(tensor_inner(v, w))
        push!(alpha, a)
        w = w + (-a) * v + (-b) * basis[end - 1]
        if reorth
            for _pass in 1:2, u in basis
                w = w - tensor_inner(u, w) * u
            end
        end
        w = to_concrete(w)
        coeff, value = _reduce_krylov(LowestEigen(), alpha, betas, beta0)
        # Ritz residual: `beta_k |c_k|` bounds `||A v - theta v||`. Cheap -- both factors are
        # already computed -- and it is the ANSWER's convergence, not the basis's.
        b * abs(coeff[end]) < restol && break
    end

    out = to_concrete(coeff[1] * basis[1])
    for k in 2:length(coeff)
        out = to_concrete(out + coeff[k] * basis[k])
    end
    n = norm(out)
    n > 0 && (out = to_concrete((1.0 / n) * out))
    return out, real(value), length(alpha)
end
