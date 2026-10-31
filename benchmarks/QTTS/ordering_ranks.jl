# Grouped against interleaved QTT ordering on a small nonlinear GP run (implementation plan WP5:
# the shedding study uses grouped ordering; interleaved is measured on small setups).
#
# The same rank-rich start (an oscillator superposition) is evolved by the order-2 nonlinear BUG
# step in both orderings, at the same tolerances and rank cap. Per step: the largest bond, the bond
# at the x|y split of the grouped ordering, the error against dense RK4 of the same finite-difference
# system, and the wall time. Equal tolerances, so the comparison is rank at matched accuracy.
#
#   julia --project benchmarks/QTTS/ordering_ranks.jl
#
# ENV: OR_OUT (output directory, required), OR_N (bits per axis, 5), OR_T (1.0), OR_H (0.05),
#      OR_MAXDIM (128). Resource needs are not yet measured.

include(joinpath(@__DIR__, "common.jl"))

haskey(ENV, "OR_OUT") || error("set OR_OUT to the output directory")
const OUTPUT_DIR = ENV["OR_OUT"]
const BITS = env_int("OR_N", 5)
const FINAL_TIME = env_float("OR_T", 1.0)
const STEP = env_float("OR_H", 0.05)
const MAXDIM = env_int("OR_MAXDIM", 128)
mkpath(OUTPUT_DIR)

rows = NamedTuple[]
for order in (:grouped, :interleaved)
    g = qtt_grid(BITS, BITS; x = (-6.0, 6.0), y = (-6.0, 6.0), order = order)
    model = gp_model(g; gnl = 1.0, gamma = 0.1, mu = 0.0, omega = 1.0, rho_tol = 1e-12)
    start = rank_rich_state(g)
    psi = qtt_mps(start)
    reference = grid_vec(start, g)
    wall = 0.0
    for step = 1:round(Int, FINAL_TIME / STEP)
        t = (step - 1) * STEP
        wall += @elapsed qtt_bug_step!(
            psi,
            GPFlow(model),
            t,
            STEP;
            order = 2,
            maxdim = MAXDIM,
            tol_basis = 1e-8,
            tol_trunc = 1e-10,
        )
        reference = gp_dense_rk4(model, reference, t, t + STEP; dt = 1e-4)
        values = grid_vec(qtt_cores(psi), g)
        push!(
            rows,
            (
                ordering = order,
                t = step * STEP,
                chi_max = maximum(bond_dims(psi)),
                chi_x_split = bond_dims(psi)[BITS],
                err_state = norm(values - reference) / norm(reference),
                wall = wall,
            ),
        )
        println(rows[end])
    end
end
write_csv(joinpath(OUTPUT_DIR, "ordering_ranks.csv"), rows)
println("ORDERING-DONE")
