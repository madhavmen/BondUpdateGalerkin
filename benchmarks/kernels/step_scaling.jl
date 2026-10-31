# Cost of one BUG step against the bond dimension (implementation plan Section 7: no unexplained
# scaling degradation in chi). The kernel baseline that performance PRs compare against.
#
# A Heisenberg chain is first evolved (untimed) until its bonds reach the cap, so every timed step
# runs at the target chi. Then each method takes REPEATS steps at that chi; the median wall time
# and the allocations of one step are reported. Compilation is excluded by a warm-up step on a
# short chain. Both methods use the same MPO, step, cap and per-cut truncation rule.
#
#   julia --project benchmarks/kernels/step_scaling.jl OUTPUT_DIR [L] [CHI_LIST]
#
# Defaults: L = 32, CHI_LIST = "16,32,64". Writes OUTPUT_DIR/step_scaling.csv. Record the machine
# and thread count with every result; compare only runs on the same hardware.

using BondUpdateGalerkin.BUG, BondUpdateGalerkin.QMB, BondUpdateGalerkin.Baselines
using LinearAlgebra

isempty(ARGS) && error("usage: step_scaling.jl OUTPUT_DIR [L] [CHI_LIST]")
const OUTPUT_DIR = ARGS[1]
const CHAIN_LENGTH = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 32
const CHI_LIST = length(ARGS) >= 3 ? parse.(Int, split(ARGS[3], ",")) : [16, 32, 64]
const TIME_STEP, TOL_TRUNC, REPEATS = 0.05, 1e-8, 5
const TAU = ComplexF64(-im * TIME_STEP)

const METHODS = (
    "bug_order2" =>
        (psi, hamiltonian, chi) -> cbe_bug_step!(
            psi,
            hamiltonian,
            TAU;
            order = 2,
            maxdim = chi,
            tol_trunc = TOL_TRUNC,
        ),
    "tdvp2" =>
        (psi, hamiltonian, chi) ->
            tdvp2_step!(psi, hamiltonian, TAU; maxdim = chi, weight_tol = TOL_TRUNC),
)

"A Neel state evolved by TDVP2 until its largest bond reaches `chi` (at most 200 steps)."
function state_at_chi(hamiltonian, chi)
    psi = neel_state(CHAIN_LENGTH)
    for _ = 1:200
        maximum(bond_dims(psi)) >= chi && break
        tdvp2_step!(psi, hamiltonian, TAU; maxdim = chi, weight_tol = 1e-12)
    end
    return psi
end

set_symmetry!(:U1)
mkpath(OUTPUT_DIR)
hamiltonian = xxz_mpo(CHAIN_LENGTH; J = 1.0, delta = 1.0)
lines = ["method,L,chi_target,chi_reached,median_seconds,bytes_per_step,threads"]
for (name, step!) in METHODS
    warm_up_hamiltonian = xxz_mpo(6; J = 1.0, delta = 1.0)
    step!(neel_state(6), warm_up_hamiltonian, 4)                 # compilation only
    for chi in CHI_LIST
        start = state_at_chi(hamiltonian, chi)
        seconds = Float64[]
        bytes = 0
        for repeat = 1:REPEATS
            psi = copy(start)
            stats = @timed step!(psi, hamiltonian, chi)
            push!(seconds, stats.time)
            repeat == 1 && (bytes = stats.bytes)
        end
        median_seconds = sort(seconds)[cld(REPEATS, 2)]
        fields = (
            name,
            CHAIN_LENGTH,
            chi,
            maximum(bond_dims(start)),
            round(median_seconds; digits = 4),
            bytes,
            Threads.nthreads(),
        )
        push!(lines, join(string.(fields), ","))
        println(lines[end])
    end
end
write(joinpath(OUTPUT_DIR, "step_scaling.csv"), join(lines, "\n") * "\n")
