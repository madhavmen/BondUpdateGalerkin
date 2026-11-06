# Same-problem comparison of the second-order BUG step and two-site TDVP.
#
# Parity: one XX chain (U(1), open boundaries, J = 1), one Neel start, one time step, one rank cap,
# and one per-cut truncation rule (`weight_tol` for TDVP2 is the BUG's own `tol_trunc` budget,
# applied at every two-site cut). Both are second order in dt. The reference is the exact free-
# fermion S^z profile. Compilation is excluded by a warm-up run of each method on a short chain;
# the timed runs are repeated and the median is reported.
#
# Usage: julia --project benchmarks/QMB/ClosedSystem/xx_bug_vs_tdvp2.jl OUTPUT_DIR [L] [STEPS]
# Writes OUTPUT_DIR/xx_bug_vs_tdvp2.csv and prints the same table. Defaults: L = 16, 40 steps of
# 0.05. Its run time has not been measured yet; record it here after the first run.

using BondUpdateGalerkin.BUG, BondUpdateGalerkin.QMB, BondUpdateGalerkin.Baselines
using LinearAlgebra

include(joinpath(@__DIR__, "..", "..", "..", "test", "common", "free_fermion.jl"))

isempty(ARGS) && error("usage: xx_bug_vs_tdvp2.jl OUTPUT_DIR [L] [STEPS]")
const OUTPUT_DIR = ARGS[1]
const CHAIN_LENGTH = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 16
const STEPS = length(ARGS) >= 3 ? parse(Int, ARGS[3]) : 40
const TIME_STEP = 0.05
const MAXDIM = 64
const TOL_TRUNC = 1e-8
const REPEATS = 3

"<S^z_j> of every site, normalised by the state's own norm."
sz_profile(psi) = real.(magnetisation(copy(psi))) ./ real(overlap(psi, psi))

"The two methods under one signature: `step!(psi, hamiltonian)` advances by TIME_STEP."
const METHODS = (
    "bug_order2" =>
        (psi, hamiltonian) -> cbe_bug_step!(
            psi,
            hamiltonian,
            ComplexF64(-im * TIME_STEP);
            order = 2,
            maxdim = MAXDIM,
            tol_trunc = TOL_TRUNC,
        ),
    "tdvp2" =>
        (psi, hamiltonian) -> tdvp2_step!(
            psi,
            hamiltonian,
            ComplexF64(-im * TIME_STEP);
            maxdim = MAXDIM,
            weight_tol = TOL_TRUNC,
        ),
)

"Evolve the Neel state of `chain_length` sites for `steps` steps; return error, bond and seconds."
function run_method(step!, chain_length, steps)
    hamiltonian = xxz_mpo(chain_length; J = 1.0, delta = 0.0)
    psi = neel_state(chain_length)
    start = time_ns()
    for _ = 1:steps
        step!(psi, hamiltonian)
    end
    seconds = (time_ns() - start) / 1e9
    exact =
        xx_free_fermion_sz(chain_length, steps * TIME_STEP; J = 1.0, occupied = 1:2:chain_length)
    return (
        error = maximum(abs.(sz_profile(psi) .- exact)),
        max_bond = maximum(bond_dims(psi)),
        seconds = seconds,
    )
end

set_symmetry!(:U1)
mkpath(OUTPUT_DIR)
rows = String["method,L,steps,dt,maxdim,tol_trunc,max_sz_error,max_bond,median_seconds"]
for (name, step!) in METHODS
    run_method(step!, 6, 2)                                   # warm-up: compilation only
    results = [run_method(step!, CHAIN_LENGTH, STEPS) for _ = 1:REPEATS]
    median_seconds = sort([r.seconds for r in results])[cld(REPEATS, 2)]
    last_result = results[end]
    fields = (
        name,
        CHAIN_LENGTH,
        STEPS,
        TIME_STEP,
        MAXDIM,
        TOL_TRUNC,
        last_result.error,
        last_result.max_bond,
        round(median_seconds; digits = 3),
    )
    line = join(string.(fields), ",")
    push!(rows, line)
    println(line)
end
write(joinpath(OUTPUT_DIR, "xx_bug_vs_tdvp2.csv"), join(rows, "\n") * "\n")
