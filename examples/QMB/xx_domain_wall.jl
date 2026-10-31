# The XX chain released from a domain wall: growing entanglement, BUG against two-site TDVP.
#
#   julia --project examples/QMB/xx_domain_wall.jl [OUTPUT_DIR] [L] [T]
#
#   H = J sum_l (S^x_l S^x_{l+1} + S^y_l S^y_{l+1}),  |psi(0)> = |up ... up down ... down>,
#   observable M_L(t) = sum_{l <= L/2} <S^z_l>, the magnetisation still in the left half.
#
# Jordan-Wigner makes this free fermions, so the exact M_L(t) is an L x L matrix exponential at any
# L and t. The front travels at about J sites per unit time and reaches the boundary at
# t ~ L / (2J); the bond dimension grows until then. Both integrators advance the same state under
# the same MPO with the same time step and rank cap.
#
# Expected output: one line per sample time with both errors and both maximum bonds, then
# `DOMAIN-WALL-OK` (exit code 0) when both stay within 1e-3 of the exact answer. With OUTPUT_DIR,
# the table is written to OUTPUT_DIR/xx_domain_wall.csv. Defaults: L = 16, T = 4, dt = 0.05.
# Ported from BUG-Julia@b155550:examples/xx_domain_wall.jl.

using LinearAlgebra
using BondUpdateGalerkin.BUG, BondUpdateGalerkin.QMB, BondUpdateGalerkin.Baselines

const OUTPUT_DIR = length(ARGS) >= 1 ? ARGS[1] : nothing
const CHAIN_LENGTH = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 16
const FINAL_TIME = length(ARGS) >= 3 ? parse(Float64, ARGS[3]) : 4.0
const TIME_STEP, MAXDIM, SAMPLE_EVERY = 0.05, 64, 10

"Exact M_L(t): free fermions with the left half occupied."
function exact_left_magnetisation(chain_length, t; J = 1.0)
    hopping = zeros(ComplexF64, chain_length, chain_length)
    for site = 1:(chain_length-1)
        hopping[site, site+1] = hopping[site+1, site] = J / 2
    end
    propagator = exp(-im * t * hopping)
    occupied = 1:(chain_length÷2)
    profile = [sum(abs2(propagator[site, k]) for k in occupied) - 0.5 for site = 1:chain_length]
    return sum(profile[1:(chain_length÷2)])
end

left_magnetisation(psi) =
    sum(sz_expectation(psi, site) for site = 1:(CHAIN_LENGTH÷2)) / real(overlap(psi, psi))

set_symmetry!(:U1)
hamiltonian = xxz_mpo(CHAIN_LENGTH; J = 1.0, delta = 0.0)
bug_state = domain_wall_state(CHAIN_LENGTH)
tdvp_state = domain_wall_state(CHAIN_LENGTH)
tau = ComplexF64(-im * TIME_STEP)

lines = ["t,exact,bug,tdvp2,bug_error,tdvp2_error,bug_max_bond,tdvp2_max_bond"]
worst_error = 0.0
println("front reaches the boundary at t ~ $(CHAIN_LENGTH / 2); this run ends at $FINAL_TIME")
for step = 1:round(Int, FINAL_TIME / TIME_STEP)
    cbe_bug_step!(bug_state, hamiltonian, tau; order = 2, maxdim = MAXDIM)
    tdvp2_step!(tdvp_state, hamiltonian, tau; maxdim = MAXDIM, weight_tol = 1e-7)
    step % SAMPLE_EVERY == 0 || continue
    t = step * TIME_STEP
    exact = exact_left_magnetisation(CHAIN_LENGTH, t)
    bug, tdvp = left_magnetisation(copy(bug_state)), left_magnetisation(copy(tdvp_state))
    row = (t, exact, bug, tdvp, abs(bug - exact), abs(tdvp - exact),
           maximum(bond_dims(bug_state)), maximum(bond_dims(tdvp_state)))
    global worst_error = max(worst_error, row[5], row[6])
    push!(lines, join(string.(row), ","))
    println("t = $t  BUG error $(row[5]) (bond $(row[7]))  TDVP2 error $(row[6]) (bond $(row[8]))")
end

if OUTPUT_DIR !== nothing
    mkpath(OUTPUT_DIR)
    write(joinpath(OUTPUT_DIR, "xx_domain_wall.csv"), join(lines, "\n") * "\n")
end
passed = worst_error < 1e-3
println(passed ? "DOMAIN-WALL-OK" : "DOMAIN-WALL-FAILED")
passed || exit(1)
