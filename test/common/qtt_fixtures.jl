# Independent references for the QTT tests.
#
# The grid re-indexing, random TTs, the stencil references (dense matrices) and the Heston grid.
# Dense site-basis objects are small-grid only.

using LinearAlgebra, Random

"A site-basis dense operator re-indexed onto the grid index (`g` a grid or an ordering)."
function _grid_matrix(g, M)
    p = qtt_site_permutation(g)
    G = zeros(eltype(M), length(p), length(p))
    G[p, p] .= M
    return G
end

"A site-basis dense vector re-indexed onto the grid index (`g` a grid or an ordering)."
function _grid_vector(g, v)
    p = qtt_site_permutation(g)
    w = zeros(eltype(v), length(p))
    w[p] .= vec(v)
    return w
end

_relerr(a, b) = norm(a - b) / max(norm(b), 1e-300)

"A random TT with bond ranks `ranks` (length `L - 1`), complex unless `T` says otherwise."
function _random_tt(ranks::AbstractVector{<:Integer}; rng = MersenneTwister(1), T = ComplexF64)
    r = [1; collect(ranks); 1]
    return [randn(rng, T, r[k], 2, r[k+1]) ./ sqrt(2 * r[k]) for k = 1:(length(r)-1)]
end

"Generic ranks `min(chi, 2^k, 2^(L-k))`."
_tt_ranks(L, chi) = [min(chi, 2^k, 2^(L - k)) for k = 1:(L-1)]

"""
The dense `N x N` finite-difference matrix of derivative order `order` (1 or 2) and spacing `h`,
written row by row with the boundary condition `bc` (`:periodic`, `:dirichlet0`, `:onesided`).
"""
function _ref_d(N::Int, h::Real, order::Int, bc::Symbol)
    D = zeros(N, N)
    # (column offset, weight) of the central stencil
    stencil =
        order == 1 ? ((1, 1 / (2h)), (-1, -1 / (2h))) : ((1, 1 / h^2), (0, -2 / h^2), (-1, 1 / h^2))
    for row = 1:N, (offset, weight) in stencil
        col = row + offset
        if bc === :periodic
            col = mod1(col, N)
        elseif !(1 <= col <= N)
            continue
        end
        D[row, col] += weight
    end
    if bc === :onesided
        # the first and last rows become the second-order one-sided stencils
        D[1, :] .= 0
        D[N, :] .= 0
        if order == 1
            D[1, 1:3] .= [-3, 4, -1] ./ (2h)
            D[N, (N-2):N] .= [1, -4, 3] ./ (2h)
        else
            D[1, 1:4] .= [2, -5, 4, -1] ./ h^2
            D[N, (N-3):N] .= [-1, 4, -5, 2] ./ h^2
        end
    end
    return D
end

"`Ax (x) Ay` on the grid index `ix + 1 + Nx iy` (`nothing` = identity)."
function _ref2(g, Ax, Ay)
    Nx, Ny = 2^g.nx, 2^g.ny
    return kron(
        Ay === nothing ? Matrix(1.0I, Ny, Ny) : Ay,
        Ax === nothing ? Matrix(1.0I, Nx, Nx) : Ax,
    )
end

"The Heston test parameters: r 0.03, q 0.01, kappa 2, theta 0.09, sigma 0.4, rho -0.7."
const _HP = HestonParams(0.03, 0.01, 2.0, 0.09, 0.4, -0.7)

"Its grid: log-price `x` in [-1, 1], variance in [0, 0.5], end points on the grid."
_heston_grid(nx, ny; kw...) =
    qtt_grid(nx, ny; x = (-1.0, 1.0), y = (0.0, 0.5), periodic_x = false, periodic_y = false, kw...)
