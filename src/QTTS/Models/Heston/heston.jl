# The Heston pricing generator on a log-price x variance grid.
#
# Moved from BUG-Julia@b155550:src/RSVDCBEBondUpdate/qtt/qtt_models.jl (Heston part).
#
# Backward formulation in time to maturity tau, u_tau = A_H u with
#   A_H = v/2 d_xx + rho sigma v d_xv + sigma^2 v/2 d_vv + (r - q - v/2) d_x + kappa (theta - v) d_v - r,
# x = log S, v the variance on [0, v_max]. The forward density uses the discrete transpose of A_H
# without the discount, which conserves sum_i p_i exactly because every stencil annihilates
# constants. Boundaries are the `:onesided` stencils of `qtt_derivative`: at v = 0 the variance
# coefficients vanish and only the first-order (degenerate) equation remains, needing no data.

"""
    HestonParams(r, q, kappa, theta, sigma, rho)

Rate, dividend yield, mean reversion, long-run variance, volatility of variance and correlation.
[`feller`](@ref) checks `2 kappa theta >= sigma^2`.
"""
struct HestonParams
    r::Float64
    q::Float64
    kappa::Float64
    theta::Float64
    sigma::Float64
    rho::Float64
end

"Whether the Feller condition `2 kappa theta >= sigma^2` holds (variance stays positive)."
feller(p::HestonParams) = 2 * p.kappa * p.theta >= p.sigma^2

"""
    heston_op(g, p; adjoint = false, bc = :onesided) -> operator cores

`A_H` on the grid `g` (`y` = variance `v`, starting at `v = 0`; both axes with end points on the
grid), or with `adjoint = true` the forward-density operator `A_H'` (discrete transpose, no
discount), which conserves `sum p`. The mixed term is `D_x (x) (V D_v)`, the product of the two
first-derivative chains with the variance factor on the rows of `D_v`.
"""
function heston_op(g::QTTGrid, p::HestonParams; adjoint::Bool = false, bc::Symbol = :onesided)
    !g.periodic_x && !g.periodic_y || throw(
        ArgumentError(
            "heston_op: build the grid with periodic_x = periodic_y = false (end points on the grid)",
        ),
    )
    abs(g.y0) < 1e-14 || throw(ArgumentError("heston_op: the variance grid must start at v = 0"))
    first_x, second_x = qtt_d(g, :x, 1; bc = bc), qtt_d(g, :x, 2; bc = bc)
    first_v, second_v = qtt_d(g, :y, 1; bc = bc), qtt_d(g, :y, 2; bc = bc)
    variance = qtt_coord_diag(g, :y)
    # the variance factor multiplies the rows of the v-derivatives
    variance_first_v = op_compress(op_compose(variance, first_v))
    variance_second_v = op_compress(op_compose(variance, second_v))
    terms = Any[
        (0.5, second_x, variance),
        (p.rho * p.sigma, first_x, variance_first_v),
        (0.5 * p.sigma^2, nothing, variance_second_v),
        (p.r - p.q, first_x, nothing),
        (-0.5, first_x, variance),
        (p.kappa * p.theta, nothing, first_v),
        (-p.kappa, nothing, variance_first_v),
    ]
    adjoint || push!(terms, (-p.r, nothing, op_identity(g.ny)))
    generator = qtt_op(g, terms)
    return adjoint ? op_transpose(generator) : generator
end

"""
    heston_mpo(g, p; kwargs...) -> MPO

[`heston_op`](@ref) as a solver MPO. `cbe_bug_step!(psi, mpo, dt; hermitian = false)` with real
`dt` integrates `u_tau = A_H u` over `dt` (requires `set_symmetry!(:none)`).
"""
heston_mpo(g::QTTGrid, p::HestonParams; kwargs...) = qtt_to_mpo(heston_op(g, p; kwargs...))
