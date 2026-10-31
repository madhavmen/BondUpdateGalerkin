# Heston on a quantics tensor train, stepped by the BUG engine and checked against exact solutions.
#
#   julia --project examples/QTTS/heston_quickstart.jl [OUTPUT_DIR]
#
# Backward Heston in time to maturity, u_tau = A_H u, on log-price x in [-1, 1] and variance
# v in [0, 0.5], 32 x 32 points. From u(0) = exp(a x + b v) the exact solution stays
# exponential-affine, u = exp(a x + B(tau) v + C(tau)), with B, C from the Riccati equations
# (`heston_riccati`). Two errors are kept apart:
#   - BUG error: the QTT solution against the exact flow of the same discrete operator,
#     exp(tau A_H) u(0) (a dense 1024 x 1024 exponential). This is what the integrator controls.
#   - spatial error: that discrete flow against the affine solution. The generator closes every
#     boundary with second-order one-sided stencils and no boundary data; this closure is not
#     uniformly stable, so the error does not fall at second order on these grids (8.6e-5 on
#     16 x 16, 5.8e-5 on 32 x 32). It is a property of the discretisation, not of BUG.
#
# Expected output: both errors, the bond dimension the correlation term builds, then `HESTON-OK`
# (exit code 0). With OUTPUT_DIR, the solution and the affine one are written to
# OUTPUT_DIR/heston_quickstart.csv.

using LinearAlgebra
using BondUpdateGalerkin.BUG, BondUpdateGalerkin.QTTS

const OUTPUT_DIR = length(ARGS) >= 1 ? ARGS[1] : nothing
const PARAMS = HestonParams(0.03, 0.01, 2.0, 0.09, 0.4, -0.7)   # r, q, kappa, theta, sigma, rho
const SLOPE_X, SLOPE_V = 0.5, -1.0
const TAU, STEPS, BITS = 0.2, 10, 5

set_symmetry!(:none)                                  # a quantics field carries no charge
clear_fusion_cache!()

g = qtt_grid(BITS, BITS; x = (-1.0, 1.0), y = (0.0, 0.5), periodic_x = false, periodic_y = false)
initial = qtt_separable(g, x -> exp(SLOPE_X * x), v -> exp(SLOPE_V * v))
generator = heston_op(g, PARAMS)

# u_tau = A_H u as a linear flow; A_H is not Hermitian
flow = LinearFlow(generator)
psi = qtt_mps(initial)
step_size = TAU / STEPS
for step = 0:(STEPS-1)
    # third order in time (Radau IIA root)
    qtt_bug_step!(psi, flow, step * step_size, step_size; order = 3, tol_trunc = 1e-12, maxdim = 64)
end

# the three fields as vectors in the QTT site order (the order op_dense acts in)
numerical = vec(tt_dense(qtt_cores(psi)))
discrete_flow = exp(TAU .* Matrix(op_dense(generator))) * vec(tt_dense(initial))
B, C = heston_riccati(PARAMS, SLOPE_X, SLOPE_V, TAU)
affine = vec(tt_dense(heston_affine_field(g, SLOPE_X, B, C)))

relative_max_error(approximation, reference) =
    maximum(abs.(approximation .- reference)) / maximum(abs.(reference))
bug_error = relative_max_error(numerical, discrete_flow)
spatial_error = relative_max_error(discrete_flow, affine)
println("BUG vs the exact discrete flow: relative max error $bug_error")
println("discrete flow vs the affine solution: relative max error $spatial_error")
println("max bond $(maximum(bond_dims(psi)))")

if OUTPUT_DIR !== nothing
    mkpath(OUTPUT_DIR)
    xs, vs = qtt_coords(g)
    numerical_grid = qtt_values(qtt_cores(psi), g)
    affine_grid = qtt_values(heston_affine_field(g, SLOPE_X, B, C), g)
    lines = ["x,v,numerical,affine"]
    for (ix, x) in enumerate(xs), (iv, v) in enumerate(vs)
        push!(lines, join(string.((x, v, real(numerical_grid[ix, iv]), affine_grid[ix, iv])), ","))
    end
    write(joinpath(OUTPUT_DIR, "heston_quickstart.csv"), join(lines, "\n") * "\n")
end

# BUG reproduces the discrete flow well below the spatial error, which stays small
passed = bug_error < 1e-5 && spatial_error < 1e-3
println(passed ? "HESTON-OK" : "HESTON-FAILED")
passed || exit(1)
