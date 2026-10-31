# Heston on a quantics tensor train, stepped by the BUG engine and checked against an exact solution.
#
#   julia --project examples/QTTS/heston_quickstart.jl [OUTPUT_DIR]
#
# Backward Heston in time to maturity, u_tau = A_H u, on log-price x in [-1, 1] and variance
# v in [0, 0.5]. From u(0) = exp(a x + b v) the exact solution stays exponential-affine,
# u = exp(a x + B(tau) v + C(tau)), with B, C from the Riccati equations (`heston_riccati`). The
# discrete generator uses second-order one-sided stencils at every boundary (no boundary data), so
# the QTT solution converges to the affine one at second order in the grid spacing.
#
# Expected output: the relative max error on a 16 x 16 and a 32 x 32 grid, their ratio (about 4
# for second order), the bond dimension the correlation term builds, then `HESTON-OK` (exit
# code 0). With OUTPUT_DIR, the 32 x 32 solution and the exact one are written to
# OUTPUT_DIR/heston_quickstart.csv.

using LinearAlgebra
using BondUpdateGalerkin.BUG, BondUpdateGalerkin.QTTS

const OUTPUT_DIR = length(ARGS) >= 1 ? ARGS[1] : nothing
const PARAMS = HestonParams(0.03, 0.01, 2.0, 0.09, 0.4, -0.7)   # r, q, kappa, theta, sigma, rho
const SLOPE_X, SLOPE_V = 0.5, -1.0
const TAU, STEPS = 0.2, 10

set_symmetry!(:none)                                  # a quantics field carries no charge
clear_fusion_cache!()

"Evolve the affine initial field on 2^bits x 2^bits points; return grid, solution and exact field."
function solve_on_grid(bits)
    g = qtt_grid(bits, bits; x = (-1.0, 1.0), y = (0.0, 0.5), periodic_x = false, periodic_y = false)
    initial = qtt_separable(g, x -> exp(SLOPE_X * x), v -> exp(SLOPE_V * v))
    generator = heston_mpo(g, PARAMS)
    psi = qtt_mps(initial)
    for _ = 1:STEPS
        # real dt: u(tau + dt) = exp(dt A_H) u; A_H is not Hermitian
        cbe_bug_step!(psi, generator, ComplexF64(TAU / STEPS); hermitian = false, tol_basis = 1e-9, tol_trunc = 1e-12, maxdim = 64)
    end
    B, C = heston_riccati(PARAMS, SLOPE_X, SLOPE_V, TAU)
    exact = qtt_values(heston_affine_field(g, SLOPE_X, B, C), g)
    return g, qtt_values(qtt_cores(psi), g), exact, maximum(bond_dims(psi))
end

errors = Float64[]
for bits in (4, 5)
    g, numerical, exact, max_bond = solve_on_grid(bits)
    relative_error = maximum(abs.(numerical .- exact)) / maximum(abs.(exact))
    push!(errors, relative_error)
    println("$(2^bits) x $(2^bits): relative max error $relative_error, max bond $max_bond")
    if bits == 5 && OUTPUT_DIR !== nothing
        mkpath(OUTPUT_DIR)
        xs, vs = qtt_coords(g)
        lines = ["x,v,numerical,exact"]
        for (ix, x) in enumerate(xs), (iv, v) in enumerate(vs)
            push!(lines, join(string.((x, v, real(numerical[ix, iv]), exact[ix, iv])), ","))
        end
        write(joinpath(OUTPUT_DIR, "heston_quickstart.csv"), join(lines, "\n") * "\n")
    end
end
ratio = errors[1] / errors[2]
println("error ratio for halved spacing: $ratio (second order: ~4)")
passed = ratio > 2.5 && errors[2] < 1e-2
println(passed ? "HESTON-OK" : "HESTON-FAILED")
passed || exit(1)
