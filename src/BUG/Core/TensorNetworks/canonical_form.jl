# Canonical form of `SymMPS`: exact (truncation-free) centre moves and isometry checks.
#
# Moved from BUG-Julia@b155550:src/BondUpdateBUG/symmetric_mps_canonical.jl. Its spin observables
# (`site_expval`, `apply_sz!`, `sz_overlaps`) moved to QMB/Observables/local_observables.jl.

"""
    move_right!(psi, site)

Move the orthogonality centre from `site` to `site+1`. `psi[site]` becomes a left
isometry; `S·Vd` is absorbed into `psi[site+1]`.
"""
function move_right!(psi::SymMPS, site::Int)
    res = svd(psi[site], (1, 2); cutoff = 0.0)
    # U  : (L,i, S,i, svdL)      -- the left isometry, with i = site
    # Vd : (svdR, L,i+1)
    carried = res.S * res.Vd                                  # (svdL, L,i+1)
    next_tensor = contract(carried, (2,), psi[site+1], (1,))  # (svdL, S,i+1, L,i+2)
    psi[site] = to_concrete(setitag(res.U, 3, "L,$(site + 1)"))
    psi[site+1] = to_concrete(setitag(next_tensor, 1, "L,$(site + 1)"))
    return psi
end

"""
    move_left!(psi, site)

Move the orthogonality centre from `site` to `site-1`. `psi[site]` becomes a right
isometry; `U·S` is absorbed into `psi[site-1]`.
"""
function move_left!(psi::SymMPS, site::Int)
    res = svd(psi[site], (1,); cutoff = 0.0)
    # U  : (L,i, svdL), with i = site
    # Vd : (svdR, S,i, L,i+1)    -- the right isometry
    carried = res.U * res.S                                       # (L,i, svdR)
    previous_tensor = contract(psi[site-1], (3,), carried, (1,))  # (L,i-1, S,i-1, svdR)
    psi[site-1] = to_concrete(setitag(previous_tensor, 3, "L,$site"))
    psi[site] = to_concrete(setitag(res.Vd, 1, "L,$site"))
    return psi
end

"""
    canonical!(psi, site)

Sweep the orthogonality centre to site `site`. Norm-preserving.
"""
function canonical!(psi::SymMPS, site::Int)
    1 <= site <= length(psi) || throw(BoundsError(psi, site))
    while psi.center < site
        move_right!(psi, psi.center)
        psi.center += 1
    end
    while psi.center > site
        move_left!(psi, psi.center)
        psi.center -= 1
    end
    return psi
end

"""
    left_gram(site_tensor) -> TLArray

`A†A` for `A = site_tensor`, traced over `(link_l, site)`, leaving the two `link_r` legs open.
`site_tensor` is a left isometry iff this is the identity.

`A' * A` will NOT do: `*` is greedy and closes the bond too, collapsing to a
rank-0 scalar. Priming keeps it open.
"""
left_gram(site_tensor) = contract(prime(site_tensor, 3)', (1, 2), site_tensor, (1, 2))

"""
    right_gram(site_tensor) -> TLArray

`AA†` for `A = site_tensor`, traced over `(site, link_r)`, leaving the two `link_l` legs open.
"""
right_gram(site_tensor) = contract(prime(site_tensor, 1)', (2, 3), site_tensor, (2, 3))

"""
    left_isometry_defect(site_tensor) -> Float64

`||A (I - A'A)||_F` for `A = site_tensor`, over the `(link_l, site)` legs: zero exactly for a left
isometry, computed without cancellation. The alternatives are worse: `norm(left_gram(A) - 1)`
cannot align legs for frames from `oplus`, and the Frobenius expansion
`||G-I||^2 = ||G||^2 - 2||A||^2 + d` subtracts quantities of size `d`, flooring the defect near
1e-8. `A - A(A'A)` is a difference of O(1) tensors whose true value is small, so its norm keeps
full precision.
"""
left_isometry_defect(site_tensor) = norm(perp_component(site_tensor, site_tensor))

"""
    right_isometry_defect(site_tensor) -> Float64

`||(I - AA') A||_F` for `A = site_tensor`, over the `(site, link_r)` legs. Mirror of
[`left_isometry_defect`](@ref); the bond is leg 1.
"""
right_isometry_defect(site_tensor) = norm(perp_component_right(site_tensor, site_tensor))
