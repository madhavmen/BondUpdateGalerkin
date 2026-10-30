# Independent references for the QTT tests.
#
# Moved from BUG-Julia@b155550:tests/common/qtt_fixtures.jl (the grid re-indexing and random TTs).
# The stencil references, the environment references and the Heston grid follow with the operators
# and models they check (derivatives and models, PR 7). Dense site-basis objects are small-grid only.

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
