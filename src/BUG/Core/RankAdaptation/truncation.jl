# Moved from BUG-Julia@b155550:src/RSVDCBEBondUpdate/operators/zero_site_core.jl.

# ── truncation sweep (Sulz algorithm 1) ──────────────────────────────────────

"""
    truncate_sweep!(psi; maxdim, cutoff) -> psi

Compress every bond after the Galerkin steps. It must run only after every bond has had its step:
earlier, it would remove directions that are merely not yet populated, which pinned the rank at 2
in the single-centre version.

One left-to-right truncating pass, then one right-to-left, ending with the centre at site 1.
"""
function truncate_sweep!(psi::SymMPS; maxdim::Int = 200, cutoff::Float64 = 1e-12)
    L = length(psi)
    canonical!(psi, 1)
    for i in 1:(L - 1)
        res = svd(psi[i], (1, 2); cutoff = cutoff, Nkeep = maxdim)
        tag = psi[i].inds[3].itags
        M = to_concrete(res.S * res.Vd)
        psi[i]     = to_concrete(setitag(res.U, 3, tag))
        psi[i + 1] = to_concrete(setitag(contract(M, (2,), psi[i + 1], (1,)), 1, tag))
        psi.center = i + 1
    end
    for i in L:-1:2
        res = svd(psi[i], (1,); cutoff = cutoff, Nkeep = maxdim)
        tag = psi[i - 1].inds[3].itags
        M = to_concrete(res.U * res.S)
        psi[i]     = to_concrete(setitag(res.Vd, 1, tag))
        psi[i - 1] = to_concrete(setitag(contract(psi[i - 1], (3,), M, (1,)), 3, tag))
        psi.center = i - 1
    end
    return psi
end

"""
    _cut_divisor(L, root) -> Float64

`weight_tol ||psi|| / _cut_divisor(L, root)` is the absolute allowance of EVERY Schmidt cut under
the budgeted rule: what [`truncate_recursive!`](@ref) gives each bond of the BUG's S-step result,
and -- USER 2026-10-17, the singular values discarded must be the same for both -- what
`tdvp2_step!(...; weight_tol)` gives each two-site update, at the BUG's default root
`_bug_root_link(L) + 1`. One function, so the two rules cannot drift apart.
"""
_cut_divisor(L::Int, root::Int) = sqrt(root - 1) + sqrt(L - root)

"""
    _budget_svd(T, split, maxdim, share_of) -> (res, capped)

ONE bond's budgeted cut, the rule `truncate_recursive!` applies with `weight_tol` and the
adaptive CBE-DMRG applies at every split: keep the SMALLEST rank whose discarded Schmidt norm
fits the absolute allowance `share_of(n2)` (`n2` = this tensor's total squared weight), never
more than `maxdim`. `capped` reports that `maxdim`, not the allowance, set the rank.
"""
function _budget_svd(T, split, maxdim::Int, share_of)
    res = svd(T, split; cutoff = 0.0, Nkeep = maxdim, get_lists = true)
    share = share_of(_sq_weight(res.kept_list) + _sq_weight(res.trunc_list))
    nk = length(res.kept_list)
    # `trunc_list` holds what `maxdim` (and exact zeros) already removed: it is spent.
    k, tail = _tail_rank(res.kept_list, 1.0, share, nk; base = _sq_weight(res.trunc_list))
    capped = tail > share && !isempty(res.trunc_list)
    k = max(k, 1)            # a bond never closes; a zero state has nothing to keep anyway
    k < nk && (res = svd(T, split; cutoff = 0.0, Nkeep = k, get_lists = true))
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
function truncate_recursive!(psi::SymMPS, root::Int; maxdim::Int = 200,
                             cutoff::Float64 = 1e-12,
                             weight_tol::Union{Nothing, Float64} = nothing,
                             bond_weights::Union{Nothing, Vector{Float64}} = nothing)
    L = length(psi)
    bond_weights === nothing || length(bond_weights) == L - 1 ||
        throw(DimensionMismatch("bond_weights needs $(L - 1) entries"))
    1 <= root <= L || throw(ArgumentError("root $root outside 1:$L"))
    weight_tol === nothing || weight_tol >= 0 ||
        throw(ArgumentError("weight_tol must be non-negative, got $weight_tol"))
    worst = 0.0
    cum = [0.0, 0.0]           # absolute squared weight removed, left / right subtree
    nrm2 = -1.0                # ||psi||^2, read off the first (centre) SVD
    capped = false
    sdiv = _cut_divisor(L, root)

    # One bond's SVD under whichever rule is active. `split` is the legs `svd` keeps on `U`.
    function cut(T, split)
        weight_tol === nothing &&
            return svd(T, split; cutoff = cutoff, Nkeep = maxdim, get_lists = true)
        res, c = _budget_svd(T, split, maxdim, function (n2)
            nrm2 < 0 && (nrm2 = n2)
            return sdiv > 0 ? weight_tol * sqrt(max(nrm2, 0.0)) / sdiv : 0.0
        end)
        capped |= c
        return res
    end
    function account!(res, side, bond)
        worst = max(worst, _trunc_weight(res))
        cum[side] += _sq_weight(res.trunc_list)
        bond_weights === nothing || (bond_weights[bond] = _sq_weight(res.trunc_list))
        nrm2 < 0 && (nrm2 = _sq_weight(res.kept_list) + _sq_weight(res.trunc_list))
    end

    # ── left subtree, root-to-leaf: bonds root-1, root-2, … 1 ──
    for i in root:-1:2
        res = cut(psi[i], (1,))
        account!(res, 1, i - 1)               # splits bond (i-1, i)
        tag = psi[i - 1].inds[3].itags
        # `A = U S Vd`: keep `Vd` here and hand `U S` outward, so the child receives the weight
        # and its own SVD is a genuine Schmidt decomposition.
        psi[i]     = to_concrete(setitag(res.Vd, 1, tag))
        psi[i - 1] = to_concrete(setitag(
            contract(psi[i - 1], (3,), to_concrete(res.U * res.S), (1,)), 3, tag))
        psi.center = i - 1
    end

    # With the weight carried out to site 1 the centre has to come back before the other subtree
    # can be cut. LOSSLESS: every bond crossed was just cut, and cutting it again is exactly the
    # compounding this routine exists to avoid.
    canonical!(psi, root)

    # ── right subtree, root-to-leaf: bonds root, root+1, … L-1 ──
    for i in root:(L - 1)
        res = cut(psi[i], (1, 2))
        account!(res, 2, i)                   # splits bond (i, i+1)
        tag = psi[i].inds[3].itags
        psi[i]     = to_concrete(setitag(res.U, 3, tag))
        psi[i + 1] = to_concrete(setitag(
            contract(to_concrete(res.S * res.Vd), (2,), psi[i + 1], (1,)), 1, tag))
        psi.center = i + 1
    end

    cumulative = nrm2 > 0 ? (sqrt(cum[1]) + sqrt(cum[2])) / sqrt(nrm2) : 0.0
    bond_weights === nothing || (bond_weights .= nrm2 > 0 ? sqrt.(bond_weights ./ nrm2) : 0.0)
    return psi, worst, cumulative, capped
end

# ── from BUG/Randomised/sketch.jl: deterministic, used by the Core sweeps ──

"Degeneracy-weighted squared singular weight of a Telum SVD list, `Σ σ² · deg`."
_sq_weight(list) = sum(Float64(s)^2 * Float64(deg) for (s, deg, _, _) in list; init = 0.0)

"Relative weight discarded by an SVD, `sqrt(Σcut σ² / Σall σ²)`."
function _trunc_weight(res)
    kept, cut = _sq_weight(res.kept_list), _sq_weight(res.trunc_list)
    tot = kept + cut
    return tot == 0.0 ? 0.0 : sqrt(cut / tot)
end

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
    _rank_cut(stol, Y, n) -> Float64

The preselection SVD's relative cutoff: `stol > 0` cuts at `stol`; `stol <= 0` is the numerical
rank test, with the floor `_CBE_EPS` measured against `||Y||` (not `||P_perp Y||`, which cancels).
"""
_rank_cut(stol::Float64, Y, n::Real) =
    stol > 0 ? stol : _CBE_EPS * max(norm(Y), n) / n

"""
    _preselect_left(U0, Y, stol) -> (Q, err) or (nothing, 0.0)

Orthonormal basis of the part of the sketched action `Y` lying outside `U0`, with the
`RepairOrtho` second pass. `Y` and `Q` carry `(link_l, site_l, bond)`.

`Y` is normalised before the SVD because `cutoff` is relative to the largest singular
value: the new directions are O(tau)-small in absolute terms, and thresholding relative to
their own scale is what keeps a meaningful tolerance (the trap recorded in `augment.jl`).
"""
function _preselect_left(U0, Y, stol::Float64)
    R = perp_component(U0, Y)
    n = norm(R)
    n <= _CBE_EPS * max(norm(Y), 1.0) && return (nothing, 0.0)
    res = svd(to_concrete((1.0 / n) * R), (1, 2), _TAG_L, _TAG_LR;
              cutoff = _rank_cut(stol, Y, n), get_lists = true)
    Q = to_concrete(res.U)
    R2 = perp_component(U0, Q)            # RepairOrtho
    norm(R2) <= _CBE_EPS * norm(Q) && return (nothing, 0.0)
    Q = to_concrete(svd(R2, (1, 2), _TAG_L, _TAG_LR; cutoff = 1e-13).U)
    return (Q, _trunc_weight(res))
end

"""
    _preselect_right(V0, Y, stol) -> (Q, err) or (nothing, 0.0)

Row mirror: `Y` and `Q` carry `(bond, site_r, link_r)`, the frame leg is 1 and the
orthonormalisation is over `(site_r, link_r)`.
"""
function _preselect_right(V0, Y, stol::Float64)
    R = perp_component_right(V0, Y)
    n = norm(R)
    n <= _CBE_EPS * max(norm(Y), 1.0) && return (nothing, 0.0)
    res = svd(to_concrete((1.0 / n) * R), (2, 3), _TAG_R, _TAG_RR;
              cutoff = _rank_cut(stol, Y, n), get_lists = true)
    Q = to_concrete(permutedims(res.U, (3, 1, 2)))
    R2 = perp_component_right(V0, Q)      # RepairOrtho
    norm(R2) <= _CBE_EPS * norm(Q) && return (nothing, 0.0)
    Q = to_concrete(permutedims(svd(R2, (2, 3), _TAG_R, _TAG_RR; cutoff = 1e-13).U,
                                (3, 1, 2)))
    return (Q, _trunc_weight(res))
end

"""
    _tail_rank(list, scale, allowance, kmax; base = 0.0) -> (k, tail)

Test A: the smallest `k <= kmax` whose discarded tail
`scale * sqrt(base + sum_{j>k} sigma_j^2 * deg_j)` fits inside `allowance`, and that tail.
Degeneracies count (SU(2)); `base` is weight the SVD already dropped. Returns `kmax` when nothing
fits, and the caller records the budget as unmet.
"""
function _tail_rank(list, scale::Float64, allowance::Float64, kmax::Int; base::Float64 = 0.0)
    n = length(list)
    kmax = min(kmax, n)
    kmax <= 0 && return (0, scale * sqrt(base + _sq_weight(list)))
    w = [Float64(s)^2 * Float64(deg) for (s, deg, _, _) in list]
    # Weight below `kmax` is discarded whatever `k` we choose, so it belongs to the floor.
    tail = base + sum(w[(kmax + 1):n]; init = 0.0)
    k = kmax
    while k > 0
        t = tail + w[k]
        scale * sqrt(t) <= allowance || break
        tail = t
        k -= 1
    end
    return (k, scale * sqrt(tail))
end

# ── from BUG/Randomised/cbe.jl: deterministic, used by the Core sweeps ──

"Keep `counts[q]` columns of sector `q` on `leg`, dropping sectors allotted none."
function _trim_per_sector(Q, leg::Int, counts::Dict)
    keep = s -> (n = get(counts, s, 0); n <= 0 ? nothing : 1:n)
    any(v > 0 for v in values(counts)) || return nothing
    return to_concrete(getsub(Q, leg, keep))
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
