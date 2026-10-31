# Two-dimensional operators assembled from one-dimensional chains on a grid.
#
# Moved from BUG-Julia@b155550:src/RSVDCBEBondUpdate/qtt/qtt_models.jl (`qtt_d`, `qtt_coord_diag`,
# `qtt_op` and their grid helpers). Every model operator is a sum of Kronecker products of 1D chains
# (`derivatives.jl`) placed by `qtt_embed`, as operator cores; it becomes a solver MPO only through
# `qtt_to_mpo`.

"Number of bits of grid axis `axis` (`:x` or `:y`)."
_axis_bits(g::QTTGrid, axis::Symbol) = axis === :x ? g.nx : g.ny

"Grid spacing along `axis`."
_axis_step(g::QTTGrid, axis::Symbol) = axis === :x ? g.dx : g.dy

"Grid coordinates along `axis`."
_axis_coords(g::QTTGrid, axis::Symbol) = axis === :x ? first(qtt_coords(g)) : last(qtt_coords(g))

"""
    qtt_d(g, axis, order; bc) -> operator cores

The 1D finite-difference chain of derivative order `order` (1 or 2) along `axis` (`:x` or `:y`)
with boundary condition `bc` ([`qtt_derivative`](@ref)).
"""
qtt_d(g::QTTGrid, axis::Symbol, order::Int; bc::Symbol) =
    qtt_derivative(_axis_bits(g, axis), _axis_step(g, axis); order = order, bc = bc)

"`diag(f(c))` for the coordinate `c` of `axis`, as a 1D chain."
qtt_coord_diag(g::QTTGrid, axis::Symbol, f = identity) = qtt_diag_1d(f.(_axis_coords(g, axis)))

"""
    qtt_op(g, terms; tol = 1e-13) -> operator cores

`sum_t c_t (X_t (x) Y_t)` on the grid for `terms = [(c, X, Y), ...]` of 1D chains, `X` acting on
`x` and `Y` on `y` (`nothing` = the identity on that axis), compressed once at relative `tol`.
"""
qtt_op(g::QTTGrid, terms; tol::Real = 1e-13) = op_lincomb(
    [
        (coefficient, qtt_embed(g; x = x_chain, y = y_chain)) for
        (coefficient, x_chain, y_chain) in terms
    ];
    tol = tol,
)
