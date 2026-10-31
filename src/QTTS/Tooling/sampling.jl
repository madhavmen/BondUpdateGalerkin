# Normalisation in the physical quadrature and selective sampling of QTT fields.
#
# Moved from BUG-Julia@b155550:src/RSVDCBEBondUpdate/qtt/qtt_observables.jl (`qtt_normalize!`,
# `qtt_sample`). Physical quantities use the discrete L2 inner product with the weight `dx dy` kept
# outside the cores; large fields are read point by point, never densified.

"Rescale `psi` in place so that `dx dy sum |psi|^2 = N`."
function qtt_normalize!(psi::SymMPS, g::QTTGrid; N::Real = 1.0)
    physical_norm = qtt_l2norm(psi, g)
    physical_norm > 0 || throw(ArgumentError("qtt_normalize!: zero state"))
    psi[psi.center] = to_concrete(psi[psi.center] * (sqrt(N) / physical_norm))
    return psi
end

"""
    qtt_sample(A, g; stride = 1) -> (xs, ys, M)

The field on every `stride`-th grid point in each direction, one core contraction per point: a
reduced-resolution image of a field of any size.
"""
function qtt_sample(A::AbstractVector, g::QTTGrid; stride::Int = 1)
    x_indices = 0:stride:(2^g.nx-1)
    y_indices = 0:stride:(2^g.ny-1)
    xs, ys = qtt_coords(g)
    M = [qtt_value(A, g, ix, iy) for ix in x_indices, iy in y_indices]
    return xs[x_indices.+1], ys[y_indices.+1], M
end
