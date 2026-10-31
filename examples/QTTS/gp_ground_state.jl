# The Gross-Pitaevskii ground state in a harmonic trap, by imaginary-time BUG on a quantics tensor
# train, cross-checked against QTT gradient descent.
#
#   julia --project examples/QTTS/gp_ground_state.jl [OUTPUT_DIR]
#
# H[psi] = -Delta/2 + (x^2 + y^2)/2 + g |psi|^2 on a periodic 16 x 16 box [-6, 6)^2 with g = 1. The
# nonlinear density is rebuilt from the state at every Runge-Kutta stage (`qtt_bug_step!`), and the
# state is renormalised in the physical quadrature after every step. Gradient descent on the
# normalised energy (`gp_gradient_descent!`) is an independent method for the same discrete
# problem, so the two energies must agree.
#
# Expected output: the energy, chemical potential and stationarity residual of both methods, then
# `GP-OK` (exit code 0) when the energies agree to 1e-6 relative and the BUG energy decreased at
# every accepted step. With OUTPUT_DIR, the BUG ledger is written to OUTPUT_DIR/gp_ground_state.csv.

using LinearAlgebra
using BondUpdateGalerkin.BUG, BondUpdateGalerkin.QTTS

const OUTPUT_DIR = length(ARGS) >= 1 ? ARGS[1] : nothing

set_symmetry!(:none)
clear_fusion_cache!()
g = qtt_grid(4, 4; x = (-6.0, 6.0), y = (-6.0, 6.0))
model = gp_model(g; gnl = 1.0, gamma = 0.0, mu = 0.0, omega = 1.0, rho_tol = 1e-12)
# start from a displaced Gaussian so relaxation has work to do
start() = qtt_mps(qtt_separable(g, x -> exp(-(x - 0.5)^2 / 2), y -> exp(-y^2 / 3)))

bug_state = start()
ledger = gp_ground_state!(bug_state, model, uniform_schedule(0.5, 30); order = 3, maxdim = 64)
accepted = filter(entry -> entry.accepted, ledger)
bug_final = gp_observables(model, qtt_cores(bug_state), 0.0)

descent_state = start()
descent = gp_gradient_descent!(descent_state, model; eta = 0.05, maxiter = 2000, tol = 1e-8)
descent_final = gp_observables(model, qtt_cores(descent_state), 0.0)

println("imaginary-time BUG: E = $(bug_final.E), mu = $(bug_final.mu), R_mu = $(bug_final.R_mu), rejected steps $(count(entry -> !entry.accepted, ledger))")
println("gradient descent:   E = $(descent_final.E), mu = $(descent_final.mu), R_mu = $(descent_final.R_mu), iterations $(length(descent) - 1)")

if OUTPUT_DIR !== nothing
    mkpath(OUTPUT_DIR)
    lines = ["tau,h,accepted,reason,E,mu,R_mu,chi"]
    for entry in ledger
        push!(lines, join(string.((entry.tau, entry.h, entry.accepted, entry.reason, entry.E, entry.mu, entry.R_mu, entry.chi)), ","))
    end
    write(joinpath(OUTPUT_DIR, "gp_ground_state.csv"), join(lines, "\n") * "\n")
end

monotone = all(k -> accepted[k+1].E <= accepted[k].E + 1e-10 * abs(accepted[k].E), 1:(length(accepted)-1))
agree = abs(bug_final.E - descent_final.E) < 1e-6 * abs(descent_final.E)
passed = monotone && agree
println(passed ? "GP-OK" : "GP-FAILED")
passed || exit(1)
