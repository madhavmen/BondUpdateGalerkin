# One-dimensional quantics shift and finite-difference operators.
#
# Moved from BUG-Julia@b155550:src/RSVDCBEBondUpdate/qtt/qtt_ops1d.jl.
#
# Matrix-free MPOs on `n` bits in logical order (bit 0 first), built from the binary carry
# automaton: the shift `(S+ psi)_i = psi_{i+1}` carries from the least significant bit up, so its
# bond is the carry (dimension 2) and no object of size `2^n` is ever formed. Finite differences are
# sums of shifts.
#
# Boundary conditions are explicit, never the MPS open-chain default:
#   :periodic   -- the shift wraps
#   :dirichlet0 -- the shift drops what leaves the box (zero outside)
#   :onesided   -- :dirichlet0 plus rank-1 patches so that the first and last rows are the
#                  second-order one-sided stencils, which need no boundary data (Heston: at v = 0
#                  the variance coefficients vanish, leaving the degenerate first-order equation)

const QTT_BCS = (:periodic, :dirichlet0, :onesided)

"""
    qtt_shift(n; bc = :periodic) -> operator cores

`(S psi)_i = psi_{i+1}` on `n` bits. `:periodic` wraps `i = 2^n - 1` onto `0`; otherwise the row
`2^n - 1` is zero. The backward shift is `op_transpose(qtt_shift(n; bc))`.
"""
function qtt_shift(n::Int; bc::Symbol = :periodic)
    n >= 1 || throw(ArgumentError("qtt_shift needs n >= 1"))
    # carry_core[c_in + 1, o + 1, i + 1, c_out + 1] = 1 iff o + c_in = i + 2 c_out: the input bit is
    # the output bit plus the incoming carry, and the carry passes up to the next bit.
    carry_core = zeros(Float64, 2, 2, 2, 2)
    for carry_in = 0:1, out_bit = 0:1
        total = out_bit + carry_in
        carry_core[carry_in+1, out_bit+1, (total%2)+1, (total÷2)+1] = 1.0
    end
    cores = [copy(carry_core) for _ = 1:n]
    cores[1] = carry_core[2:2, :, :, :]                  # the +1 enters at bit 0
    top_core = cores[n]
    # the carry out of the top bit wraps (periodic) or leaves the box
    cores[n] = bc === :periodic ? sum(top_core; dims = 4) : top_core[:, :, :, 1:1]
    return cores
end

"`sum (row, col, value)` of rank-1 corrections on `n` bits, compressed."
function _qtt_patch(n::Int, entries)
    terms = [(value, op_ket_bra(n, row, col)) for (row, col, value) in entries if value != 0]
    return op_lincomb(terms)
end

"""
    qtt_derivative(n, h; order = 1, bc = :periodic) -> operator cores

Second-order finite differences on `2^n` points of spacing `h`: `order = 1` the central
`(S+ - S-) / 2h`, `order = 2` the three-point `(S+ - 2I + S-) / h^2`, with the boundary condition
`bc` (one of `QTT_BCS`, see the file header). `:onesided` needs at least four points.
"""
function qtt_derivative(n::Int, h::Real; order::Int = 1, bc::Symbol = :periodic)
    bc in QTT_BCS || throw(ArgumentError("bc must be one of $QTT_BCS, got :$bc"))
    order in (1, 2) || throw(ArgumentError("order must be 1 or 2, got $order"))
    shift_bc = bc === :periodic ? :periodic : :dirichlet0
    forward = qtt_shift(n; bc = shift_bc)
    backward = op_transpose(forward)
    identity_op = op_identity(n)
    central =
        order == 1 ? op_lincomb([(1 / (2h), forward), (-1 / (2h), backward)]) :
        op_lincomb([(1 / h^2, forward), (-2 / h^2, identity_op), (1 / h^2, backward)])
    bc === :onesided || return central
    npoints = 2^n
    npoints >= 4 || throw(ArgumentError(":onesided stencils need at least 4 points, got $npoints"))
    last_row = npoints - 1
    # patch = (one-sided stencil) - (the truncated central row), on the first and last rows
    patch =
        order == 1 ?
        [
            (0, 0, -3 / (2h)),
            (0, 1, 3 / (2h)),
            (0, 2, -1 / (2h)),
            (last_row, last_row, 3 / (2h)),
            (last_row, last_row - 1, -3 / (2h)),
            (last_row, last_row - 2, 1 / (2h)),
        ] :
        [
            (0, 0, 4 / h^2),
            (0, 1, -6 / h^2),
            (0, 2, 4 / h^2),
            (0, 3, -1 / h^2),
            (last_row, last_row, 4 / h^2),
            (last_row, last_row - 1, -6 / h^2),
            (last_row, last_row - 2, 4 / h^2),
            (last_row, last_row - 3, -1 / h^2),
        ]
    return op_lincomb([(1.0, central), (1.0, _qtt_patch(n, patch))])
end

"""
    qtt_diag_1d(vals; tol = 1e-14) -> operator cores

`diag(vals)` on `n = log2(length(vals))` bits, from the 1D TT-SVD of the samples (rank 2 for a
linear function, 3 for a quadratic).
"""
qtt_diag_1d(vals::AbstractVector; tol::Real = 1e-14) = op_diag(tt_from_samples(vals; tol = tol))
