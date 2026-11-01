# Moved from BUG-Julia@b155550:src/RSVDCBEBondUpdate/operators/zero_site_core.jl.
# The final rank truncation of a step, and the deterministic rank and frame helpers the Core
# sweeps share with `BUG/Randomised` (preselection, Test A tail, per-sector room,
# reorthonormalisation).

# ── truncation sweep (Sulz algorithm 1) ──────────────────────────────────────

"""
    truncate_sweep!(psi; maxdim, cutoff) -> psi

Compress every bond after the Galerkin steps. It must run only after every bond has had its step:
earlier, it would remove directions that are merely not yet populated, which pinned the rank at 2
in the single-centre version.

One left-to-right truncating pass, then one right-to-left, ending with the centre at site 1.
"""
function truncate_sweep!(psi::SymMPS; maxdim::Int = 200, cutoff::Float64 = 1e-12)
    site_count = length(psi)
    canonical!(psi, 1)
    for site = 1:(site_count-1)
        res = svd(psi[site], (1, 2); cutoff = cutoff, Nkeep = maxdim)
        tag = psi[site].inds[3].itags
        # Keep `U` on this site and push `S Vd` (bond x right link) into the next one.
        weight_block = to_concrete(res.S * res.Vd)
        psi[site] = to_concrete(setitag(res.U, 3, tag))
        psi[site+1] = to_concrete(setitag(contract(weight_block, (2,), psi[site+1], (1,)), 1, tag))
        psi.center = site + 1
    end
    for site = site_count:-1:2
        res = svd(psi[site], (1,); cutoff = cutoff, Nkeep = maxdim)
        tag = psi[site-1].inds[3].itags
        # Mirror: keep `Vd` here and push `U S` (left link x bond) into the previous site.
        weight_block = to_concrete(res.U * res.S)
        psi[site] = to_concrete(setitag(res.Vd, 1, tag))
        psi[site-1] = to_concrete(setitag(contract(psi[site-1], (3,), weight_block, (1,)), 3, tag))
        psi.center = site - 1
    end
    return psi
end

"""
    _cut_divisor(site_count, root) -> Float64

`weight_tol ||psi|| / _cut_divisor(site_count, root)` is the absolute allowance of EVERY Schmidt
cut under the budgeted rule: what [`truncate_recursive!`](@ref) gives each bond of the BUG's S-step
result, and -- USER 2026-10-17, the singular values discarded must be the same for both -- what
`tdvp2_step!(...; weight_tol)` gives each two-site update, at the BUG's default root
`_bug_root_link(site_count) + 1`. One function, so the two rules cannot drift apart.
"""
_cut_divisor(site_count::Int, root::Int) = sqrt(root - 1) + sqrt(site_count - root)

"""
    _budget_svd(tensor, split, maxdim, share_of) -> (res, capped)

ONE bond's budgeted cut, the rule `truncate_recursive!` applies with `weight_tol` and the
adaptive CBE-DMRG applies at every split: keep the SMALLEST rank whose discarded Schmidt norm
fits the absolute allowance `share_of(total_weight)` (`total_weight` = this tensor's total squared
weight), never more than `maxdim`. `capped` reports that `maxdim`, not the allowance, set the rank.
"""
function _budget_svd(tensor, split, maxdim::Int, share_of)
    res = svd(tensor, split; cutoff = 0.0, Nkeep = maxdim, get_lists = true)
    share = share_of(_sq_weight(res.kept_list) + _sq_weight(res.trunc_list))
    nk = length(res.kept_list)
    # `trunc_list` holds what `maxdim` (and exact zeros) already removed: it is spent.
    keep_rank, tail = _tail_rank(res.kept_list, 1.0, share, nk; base = _sq_weight(res.trunc_list))
    capped = tail > share && !isempty(res.trunc_list)
    keep_rank = max(keep_rank, 1)  # a bond never closes; a zero state has nothing to keep anyway
    keep_rank < nk && (res = svd(tensor, split; cutoff = 0.0, Nkeep = keep_rank, get_lists = true))
    return res, capped
end

"""
    truncate_recursive!(psi, root; maxdim, cutoff, weight_tol) -> (psi, discarded, cumulative, capped)

Recursive root-to-leaves rank truncation of the rank-adaptive tree integrator (arXiv:2201.10291
§4.2, Algorithm 7). `cutoff` applies Telum's relative cut at each bond; `discarded` is the worst
single bond. `weight_tol` is a shared budget on `||psi - psi_trunc|| / ||psi||`: each bond gets an
equal share, and `cumulative` bounds the error by the triangle inequality across the two subtrees.
`capped` is true when `maxdim` binds past the budget; `bond_weights` receives each bond's share.
"""
function truncate_recursive!(
    psi::SymMPS,
    root::Int;
    maxdim::Int = 200,
    cutoff::Float64 = 1e-12,
    weight_tol::Union{Nothing,Float64} = nothing,
    bond_weights::Union{Nothing,Vector{Float64}} = nothing,
)
    site_count = length(psi)
    bond_weights === nothing ||
        length(bond_weights) == site_count - 1 ||
        throw(DimensionMismatch("bond_weights needs $(site_count - 1) entries"))
    1 <= root <= site_count || throw(ArgumentError("root $root outside 1:$site_count"))
    weight_tol === nothing ||
        weight_tol >= 0 ||
        throw(ArgumentError("weight_tol must be non-negative, got $weight_tol"))
    worst = 0.0
    cum = [0.0, 0.0]           # absolute squared weight removed, left / right subtree
    nrm2 = -1.0                # ||psi||^2, read off the first (centre) SVD
    capped = false
    sdiv = _cut_divisor(site_count, root)

    # One bond's SVD under whichever rule is active. `split` is the legs `svd` keeps on `U`.
    function cut(tensor, split)
        weight_tol === nothing &&
            return svd(tensor, split; cutoff = cutoff, Nkeep = maxdim, get_lists = true)
        res, cut_capped = _budget_svd(
            tensor,
            split,
            maxdim,
            function (total_weight)
                nrm2 < 0 && (nrm2 = total_weight)
                return sdiv > 0 ? weight_tol * sqrt(max(nrm2, 0.0)) / sdiv : 0.0
            end,
        )
        capped |= cut_capped
        return res
    end
    # Record one cut: the worst relative weight, the subtree's squared total, the bond's share.
    function account!(res, side, bond)
        worst = max(worst, _trunc_weight(res))
        cum[side] += _sq_weight(res.trunc_list)
        bond_weights === nothing || (bond_weights[bond] = _sq_weight(res.trunc_list))
        nrm2 < 0 && (nrm2 = _sq_weight(res.kept_list) + _sq_weight(res.trunc_list))
    end

    # ── left subtree, root-to-leaf: bonds root-1, root-2, … 1 ──
    for site = root:-1:2
        res = cut(psi[site], (1,))
        account!(res, 1, site - 1)            # splits bond (site-1, site)
        tag = psi[site-1].inds[3].itags
        # `A = U S Vd`: keep `Vd` here and hand `U S` outward, so the child receives the weight
        # and its own SVD is a genuine Schmidt decomposition.
        psi[site] = to_concrete(setitag(res.Vd, 1, tag))
        psi[site-1] = to_concrete(
            setitag(contract(psi[site-1], (3,), to_concrete(res.U * res.S), (1,)), 3, tag),
        )
        psi.center = site - 1
    end

    # With the weight carried out to site 1 the centre has to come back before the other subtree
    # can be cut. LOSSLESS: every bond crossed was just cut, and cutting it again is exactly the
    # compounding this routine exists to avoid.
    canonical!(psi, root)

    # ── right subtree, root-to-leaf: bonds root, root+1, … site_count-1 ──
    for site = root:(site_count-1)
        res = cut(psi[site], (1, 2))
        account!(res, 2, site)                # splits bond (site, site+1)
        tag = psi[site].inds[3].itags
        psi[site] = to_concrete(setitag(res.U, 3, tag))
        psi[site+1] = to_concrete(
            setitag(contract(to_concrete(res.S * res.Vd), (2,), psi[site+1], (1,)), 1, tag),
        )
        psi.center = site + 1
    end

    cumulative = nrm2 > 0 ? (sqrt(cum[1]) + sqrt(cum[2])) / sqrt(nrm2) : 0.0
    bond_weights === nothing || (bond_weights .= nrm2 > 0 ? sqrt.(bond_weights ./ nrm2) : 0.0)
    return psi, worst, cumulative, capped
end

# ── from BUG/Randomised/sketch.jl: deterministic, used by the Core sweeps ──

"Degeneracy-weighted squared singular weight of a Telum SVD list, `Σ σ² · deg`."
_sq_weight(list) = sum(Float64(sigma)^2 * Float64(deg) for (sigma, deg, _, _) in list; init = 0.0)

"Relative weight discarded by an SVD, `sqrt(Σcut σ² / Σall σ²)`."
function _trunc_weight(res)
    kept, cut = _sq_weight(res.kept_list), _sq_weight(res.trunc_list)
    tot = kept + cut
    return tot == 0.0 ? 0.0 : sqrt(cut / tot)
end

# Roundoff floor of the CBE rank tests and orthonormality guards, relative to the norms compared.
const _CBE_EPS = 100 * eps(Float64)

# Distinct bond tags for the two preselections: the final-selection matrix carries both bond legs,
# and a TLArray may not hold two identical indices.
const _TAG_L, _TAG_LR = "cbeL,L", "cbeL,R"

const _TAG_R, _TAG_RR = "cbeR,L", "cbeR,R"

# Distinct again for the final orthonormality guard, for the same reason: its output is a
# fresh bond leg, and reusing a preselection tag risks the duplicate-index error above.
const _TAG_OL, _TAG_OLR = "cbeO,L", "cbeO,R"

const _TAG_OR, _TAG_ORR = "cbeP,L", "cbeP,R"

"""
    _rank_cut(stol, sketch, outside_norm) -> Float64

The preselection SVD's relative cutoff: `stol > 0` cuts at `stol`; `stol <= 0` is the numerical
rank test, with the floor `_CBE_EPS` measured against `||sketch||` (not `||P_perp sketch||`,
`outside_norm`, which cancels).
"""
_rank_cut(stol::Float64, sketch, outside_norm::Real) =
    stol > 0 ? stol : _CBE_EPS * max(norm(sketch), outside_norm) / outside_norm

"""
    _preselect_left(left_frame, sketch, stol) -> (new_directions, err) or (nothing, 0.0)

Orthonormal basis of the part of the sketched action `sketch` lying outside `left_frame`, with the
`RepairOrtho` second pass. `sketch` and `new_directions` carry `(link_l, site_l, bond)`.

`sketch` is normalised before the SVD because `cutoff` is relative to the largest singular
value: the new directions are O(tau)-small in absolute terms, and thresholding relative to
their own scale is what keeps a meaningful tolerance (the trap recorded in `augment.jl`).
"""
function _preselect_left(left_frame, sketch, stol::Float64)
    outside_part = perp_component(left_frame, sketch)
    outside_norm = norm(outside_part)
    outside_norm <= _CBE_EPS * max(norm(sketch), 1.0) && return (nothing, 0.0)
    res = svd(
        to_concrete((1.0 / outside_norm) * outside_part),
        (1, 2),
        _TAG_L,
        _TAG_LR;
        cutoff = _rank_cut(stol, sketch, outside_norm),
        get_lists = true,
    )
    new_directions = to_concrete(res.U)
    repaired_part = perp_component(left_frame, new_directions)  # RepairOrtho
    norm(repaired_part) <= _CBE_EPS * norm(new_directions) && return (nothing, 0.0)
    new_directions = to_concrete(svd(repaired_part, (1, 2), _TAG_L, _TAG_LR; cutoff = 1e-13).U)
    return (new_directions, _trunc_weight(res))
end

"""
    _preselect_right(right_frame, sketch, stol) -> (new_directions, err) or (nothing, 0.0)

Row mirror: `sketch` and `new_directions` carry `(bond, site_r, link_r)`, the frame leg is 1 and
the orthonormalisation is over `(site_r, link_r)`.
"""
function _preselect_right(right_frame, sketch, stol::Float64)
    outside_part = perp_component_right(right_frame, sketch)
    outside_norm = norm(outside_part)
    outside_norm <= _CBE_EPS * max(norm(sketch), 1.0) && return (nothing, 0.0)
    res = svd(
        to_concrete((1.0 / outside_norm) * outside_part),
        (2, 3),
        _TAG_R,
        _TAG_RR;
        cutoff = _rank_cut(stol, sketch, outside_norm),
        get_lists = true,
    )
    # Telum returns `U` as (site_r, link_r, bond); move the new bond leg to the front.
    new_directions = to_concrete(permutedims(res.U, (3, 1, 2)))
    repaired_part = perp_component_right(right_frame, new_directions)  # RepairOrtho
    norm(repaired_part) <= _CBE_EPS * norm(new_directions) && return (nothing, 0.0)
    new_directions = to_concrete(
        permutedims(svd(repaired_part, (2, 3), _TAG_R, _TAG_RR; cutoff = 1e-13).U, (3, 1, 2)),
    )
    return (new_directions, _trunc_weight(res))
end

"""
    _tail_rank(list, scale, allowance, kmax; base = 0.0) -> (keep_rank, tail)

Test A: the smallest `keep_rank <= kmax` whose discarded tail
`scale * sqrt(base + sum_{j>keep_rank} sigma_j^2 * deg_j)` fits inside `allowance`, and that tail.
Degeneracies count (SU(2)); `base` is weight the SVD already dropped. Returns `kmax` when nothing
fits, and the caller records the budget as unmet.
"""
function _tail_rank(list, scale::Float64, allowance::Float64, kmax::Int; base::Float64 = 0.0)
    entry_count = length(list)
    kmax = min(kmax, entry_count)
    kmax <= 0 && return (0, scale * sqrt(base + _sq_weight(list)))
    weights = [Float64(sigma)^2 * Float64(deg) for (sigma, deg, _, _) in list]
    # Weight below `kmax` is discarded whatever `keep_rank` we choose, so it belongs to the floor.
    tail = base + sum(weights[(kmax+1):entry_count]; init = 0.0)
    keep_rank = kmax
    # Drop the smallest kept value while the grown tail still fits the allowance.
    while keep_rank > 0
        trial_tail = tail + weights[keep_rank]
        scale * sqrt(trial_tail) <= allowance || break
        tail = trial_tail
        keep_rank -= 1
    end
    return (keep_rank, scale * sqrt(tail))
end

# ── from BUG/Randomised/cbe.jl: deterministic, used by the Core sweeps ──

"Keep `counts[sector]` columns of each `sector` on `leg`, dropping sectors allotted none."
function _trim_per_sector(tensor, leg::Int, counts::Dict)
    # Per-sector selector for `getsub`: the leading `counts[sector]` columns, or none.
    keep = sector -> (kept_width = get(counts, sector, 0); kept_width <= 0 ? nothing : 1:kept_width)
    any(allotted > 0 for allotted in values(counts)) || return nothing
    return to_concrete(getsub(tensor, leg, keep))
end

"""
    _frame_sector_dims(frame, side, fused, where) -> (dfull, dT)

Per charge sector of `fused`'s fused leg: the local dimension and the frame's width in it. Throws
if the frame is wider than its local space or holds a sector the space lacks. Labels are aligned to
the fused leg's arrow (`align_charge`).
"""
function _frame_sector_dims(frame, side::Symbol, fused, where::AbstractString)
    fl = side === :left ? 3 : 1
    dfull = Dict(charge => space_dim for (charge, space_dim) in fused.spaces[3])
    syms_ = symm(frame)
    fdir, bdir = fused.inds[3].dir, frame.inds[fl].dir
    dT = Dict(
        charge => sector_dim(frame, fl, align_charge(syms_, charge, fdir, bdir)) for
        charge in keys(dfull)
    )
    for (charge, frame_width) in dT
        frame_width <= dfull[charge] || throw(
            ErrorException(
                "$where: the frame holds $frame_width directions in sector $charge but the fused " *
                "local space has only $(dfull[charge]) -- a frame cannot exceed its local space " *
                "(frame width $(leg_dim(frame, fl)), local $(sum(values(dfull))))",
            ),
        )
    end
    sum(values(dT); init = 0) == leg_dim(frame, fl) || throw(
        ErrorException(
            "$where: the frame holds $(leg_dim(frame, fl)) directions but only " *
            "$(sum(values(dT); init = 0)) lie in sectors its local space has",
        ),
    )
    return dfull, dT
end

"""
    _cap_to_local_room(frame, candidates, side) -> candidates, a trimmed candidates, or nothing

At most `dim_local(q) - width_frame(q)` leading columns of `candidates` per sector `q`, so
`[frame | candidates]` cannot exceed its local space (Telum's cutoff is relative to the global
sigma_max, so roundoff in a saturated sector can otherwise survive). Returns `candidates` itself
when it fits.
"""
function _cap_to_local_room(frame, candidates, side::Symbol)
    candidates === nothing && return nothing
    fl = side === :left ? 3 : 1
    # The frame's (link, site) legs, which fuse into its local space.
    first_leg, second_leg = side === :left ? (1, 2) : (2, 3)
    fused = fusion_basis(frame, first_leg, second_leg; tag = "cbeRoom")
    dfull, dT = _frame_sector_dims(frame, side, fused, "_cap_to_local_room")
    syms_ = symm(frame)
    counts = Dict{Any,Int}()
    over = false
    for (charge, local_width) in dfull
        uq = align_charge(syms_, charge, fused.inds[3].dir, candidates.inds[fl].dir)
        du = sector_dim(candidates, fl, uq)
        du == 0 && continue
        counts[uq] = min(du, local_width - dT[charge])
        over |= counts[uq] < du
    end
    over || return candidates
    return _trim_per_sector(candidates, fl, counts)
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
    first_leg, second_leg = side === :left ? (1, 2) : (2, 3)
    fused = fusion_basis(frame, first_leg, second_leg; tag = "cbeChk")
    _frame_sector_dims(frame, side, fused, where)
    return frame
end

# ── orthonormality guard on the appended blocks ──
# Project the frame out, then orthonormalise (Telum has no QR, so `svd(...).U`); cutoff 1e-14
# drops columns that collapse, and a dropped column shows in `n_new_*`.

# The appended left block `block`, orthonormal and orthogonal to `left_frame`, or `nothing`.
_reortho_left(left_frame, block) = _reortho(
    block,
    columns -> perp_component(left_frame, columns),
    outside -> svd(outside, (1, 2), _TAG_OL, _TAG_OLR; cutoff = 1e-14).U,
)

# Row mirror of `_reortho_left`: the new bond leg is moved back to the front after the SVD.
_reortho_right(right_frame, block) = _reortho(
    block,
    rows -> perp_component_right(right_frame, rows),
    outside ->
        permutedims(svd(outside, (2, 3), _TAG_OR, _TAG_ORR; cutoff = 1e-14).U, (3, 1, 2)),
)

# `orth(perp(block))`, or `nothing` when the block lies (to roundoff) inside the frame.
function _reortho(block, perp, orth)
    block === nothing && return nothing
    outside_part = to_concrete(perp(block))
    norm(outside_part) <= _CBE_EPS * max(norm(block), 1.0) && return nothing
    return to_concrete(orth(outside_part))
end
