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
