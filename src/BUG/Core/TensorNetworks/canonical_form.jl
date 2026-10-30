# Canonical form of `SymMPS`: exact (truncation-free) centre moves and isometry checks.
#
# Moved from BUG-Julia@b155550:src/BondUpdateBUG/symmetric_mps_canonical.jl. Its spin observables
# (`site_expval`, `apply_sz!`, `sz_overlaps`) moved to QMB/Observables/local_observables.jl.

"""
    move_right!(psi, i)

Move the orthogonality centre from `i` to `i+1`. `psi[i]` becomes a left
isometry; `S·Vd` is absorbed into `psi[i+1]`.
"""
function move_right!(psi::SymMPS, i::Int)
    res = svd(psi[i], (1, 2); cutoff = 0.0)
    # U  : (L,i, S,i, svdL)      -- the left isometry
    # Vd : (svdR, L,i+1)
    M = res.S * res.Vd                       # (svdL, L,i+1)
    B = contract(M, (2,), psi[i+1], (1,))  # (svdL, S,i+1, L,i+2)
    psi[i] = to_concrete(setitag(res.U, 3, "L,$(i + 1)"))
    psi[i+1] = to_concrete(setitag(B, 1, "L,$(i + 1)"))
    return psi
end

"""
    move_left!(psi, i)

Move the orthogonality centre from `i` to `i-1`. `psi[i]` becomes a right
isometry; `U·S` is absorbed into `psi[i-1]`.
"""
function move_left!(psi::SymMPS, i::Int)
    res = svd(psi[i], (1,); cutoff = 0.0)
    # U  : (L,i, svdL)
    # Vd : (svdR, S,i, L,i+1)    -- the right isometry
    M = res.U * res.S                          # (L,i, svdR)
    B = contract(psi[i-1], (3,), M, (1,))    # (L,i-1, S,i-1, svdR)
    psi[i-1] = to_concrete(setitag(B, 3, "L,$i"))
    psi[i] = to_concrete(setitag(res.Vd, 1, "L,$i"))
    return psi
end

"""
    canonical!(psi, i)

Sweep the orthogonality centre to site `i`. Norm-preserving.
"""
function canonical!(psi::SymMPS, i::Int)
    1 <= i <= length(psi) || throw(BoundsError(psi, i))
    while psi.center < i
        move_right!(psi, psi.center)
        psi.center += 1
    end
    while psi.center > i
        move_left!(psi, psi.center)
        psi.center -= 1
    end
    return psi
end

"""
    left_gram(A) -> TLArray

`A†A` traced over `(link_l, site)`, leaving the two `link_r` legs open. `A` is
a left isometry iff this is the identity.

`A' * A` will NOT do: `*` is greedy and closes the bond too, collapsing to a
rank-0 scalar. Priming keeps it open.
"""
left_gram(A) = contract(prime(A, 3)', (1, 2), A, (1, 2))

"""
    right_gram(A) -> TLArray

`AA†` traced over `(site, link_r)`, leaving the two `link_l` legs open.
"""
right_gram(A) = contract(prime(A, 1)', (2, 3), A, (2, 3))

"""
    left_isometry_defect(A) -> Float64

`||A (I - A'A)||_F` over the `(link_l, site)` legs: zero exactly for a left isometry, computed
without cancellation. The alternatives are worse: `norm(left_gram(A) - 1)` cannot align legs for
frames from `oplus`, and the Frobenius expansion `||G-I||^2 = ||G||^2 - 2||A||^2 + d` subtracts
quantities of size `d`, flooring the defect near 1e-8. `A - A(A'A)` is a difference of O(1) tensors
whose true value is small, so its norm keeps full precision.
"""
left_isometry_defect(A) = norm(perp_component(A, A))

"""
    right_isometry_defect(A) -> Float64

`||(I - AA') A||_F` over the `(site, link_r)` legs. Mirror of
[`left_isometry_defect`](@ref); the bond is leg 1.
"""
right_isometry_defect(A) = norm(perp_component_right(A, A))
