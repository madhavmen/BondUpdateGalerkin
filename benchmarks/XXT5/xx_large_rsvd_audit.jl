# Larger-system audit of the DD rank-selection modes added on experiments/oldcap-fast-rsvd-xx
# (`DDQRExpansion(mode = :rank_fast)` and `mode = :rank_tail`), against the manual fixed-width
# settings, the original `:rank_adaptive`, the sequential/batched adaptive controllers and TDVP2.
#
# What the branch claims, and what this run measures to check it:
# - `:rank_fast` keeps an independent holdout but needs fewer MPO sketch columns than
#   `:rank_adaptive` (smaller holdout, at most 64 trial columns per bond however large chi is),
#   at the same accuracy  -> dd_probes, dd_holdout, per-bond probe maxima vs chi, errors.
# - `:rank_tail` uses the same single sketch as the fixed width (8 or 12 columns), no holdout, and
#   keeps fewer directions; its accuracy is only guarded by the global Krylov closure
#   -> per-bond probes equal to the fixed width, zero holdout, dd_selected, errors, budget status.
# - every BUG arm stays inside the old per-bond cap min(maxdim, ceil(1.2 * d * chi_old))
#   -> cap_violations (must be 0).
#
# Model: spin-1/2 nearest-neighbour XX chain, J = 1, open ends, U(1), Neel start. Its exact
# solution is free fermions, so the S^z profile and every bond current are graded exactly at any L
# (test/common/free_fermion.jl), no exact diagonalisation. Order 2, closure_checks = true, as in
# xx_T5_matched_work.jl; tol_trunc = 1e-3 * tol_basis, tol_root = 1e-10.
#
# All arms of one (L, tol_basis, seed) run in ONE process, after a one-step warm-up of each arm
# at L = 8 (compilation excluded), so wall-clock ratios between arms come from the same node.
# Operator counters are the primary cost measure; they never depend on the node.
#
# Usage (repository root): julia --project benchmarks/XXT5/xx_large_rsvd_audit.jl OUTPUT_DIR
# Environment: AUDIT_L=64 AUDIT_TOL=1e-3 AUDIT_SEED=0 AUDIT_STEPS=50 AUDIT_DT=0.1
#              AUDIT_MAXDIM=512 AUDIT_SAMPLE_EVERY=5 AUDIT_POLICIES="name,...,tdvp2"
# Writes OUTPUT_DIR/steps.csv (one row per arm and step) and OUTPUT_DIR/bonds.csv (one row per
# bond-log entry of every BUG step).

using BondUpdateGalerkin.BUG, BondUpdateGalerkin.QMB, BondUpdateGalerkin.Baselines
using LinearAlgebra, Random, Printf
include(joinpath(@__DIR__, "..", "..", "test", "common", "free_fermion.jl"))
include(joinpath(@__DIR__, "xx_policy_specs.jl"))

isempty(ARGS) && error("usage: xx_large_rsvd_audit.jl OUTPUT_DIR")
const OUTPUT_DIR = ARGS[1]
const CHAIN_LENGTH = parse(Int, get(ENV, "AUDIT_L", "64"))
const BASIS_TOL = parse(Float64, get(ENV, "AUDIT_TOL", "1e-3"))
const TRUNC_TOL = 1e-3 * BASIS_TOL
const SEED = parse(Int, get(ENV, "AUDIT_SEED", "0"))
const STEP_COUNT = parse(Int, get(ENV, "AUDIT_STEPS", "50"))
const TIME_STEP = parse(Float64, get(ENV, "AUDIT_DT", "0.1"))
const MAXDIM = parse(Int, get(ENV, "AUDIT_MAXDIM", "512"))
const SAMPLE_EVERY = parse(Int, get(ENV, "AUDIT_SAMPLE_EVERY", "5"))
const SPIN_TWICE = 1                 # spin 1/2: the only XX chain with a free-fermion solution
const SITE_DIM = SPIN_TWICE + 1

# The arms added on oldcap-fast-rsvd-xx and the baselines its workflow compares them with.
const DEFAULT_POLICIES = [
    "RSVD_pilot2_Kadapt", "RSVD_pilot3_Kadapt", "RSVD_pilot4_Kadapt", "RSVD_pilot3_margin8",
    "RSVD_rank_standard_Kadapt", "RSVD_fixed8rank4_Kadapt", "RSVD_fixed12rank8_Kadapt",
    "RSVD_sequential_Kadapt", "RSVD_batched_Kadapt", "RSVD_tail12_factor01",
    "RSVD_tail12_factor005", "RSVD_tail8_factor01", "tdvp2",
]
const SELECTED_POLICIES =
    filter(!isempty, strip.(split(get(ENV, "AUDIT_POLICIES", join(DEFAULT_POLICIES, ",")), ",")))

"Neel state |up, down, up, ...> of the spin-1/2 chain, built in the same space as `spin_xx_mpo`."
neel_spin_state(chain_length) = sector_product_state(
    spin_space(SPIN_TWICE),
    [(((isodd(site) ? SPIN_TWICE : -SPIN_TWICE),),) for site = 1:chain_length],
)

"Exact S^z profile and bond currents at time `t`: the up sites (odd) are the occupied fermions."
function exact_observables(chain_length, t)
    correlation = xx_open_correlation(chain_length, t; J = 1.0, occupied = 1:2:chain_length)
    return (
        sz = [real(correlation[site, site]) - 0.5 for site = 1:chain_length],
        current = xx_open_current(correlation; J = 1.0),
    )
end

"Errors of the MPS against the exact solution; every expectation value divided by the MPS norm."
function graded_observables(psi, hamiltonian, current_mpos, t)
    chain_length = length(psi)
    exact = exact_observables(chain_length, t)
    norm_squared = real(overlap(psi, psi))
    sz = spin_sz_profile(psi, SPIN_TWICE) ./ norm_squared
    current = [real(mpo_energy(psi, current_mpo)) / norm_squared for current_mpo in current_mpos]
    return (
        sz_error = maximum(abs.(sz .- exact.sz)),
        current_error = maximum(abs.(current .- exact.current)),
        # The Neel state has <H> = 0 and the evolution conserves it exactly.
        energy_drift = abs(real(mpo_energy(psi, hamiltonian)) / norm_squared),
        norm_error = abs(sqrt(norm_squared) - 1),
    )
end

const STEP_HEADER = [
    "L", "basis_tol", "seed", "policy", "step", "t", "wall_s", "D_max", "D_stage",
    "D_root_left", "D_root_right", "h1", "h1_closure", "root", "root_frontier", "h2_leak",
    "stream_cols", "stream_calls", "dd_probes", "dd_holdout", "dd_selected", "n_krylov",
    "max_entry_probes", "max_entry_holdout", "max_entry_selected", "workcap_entries",
    "cap_violations", "budget_status", "unmet", "sz_error", "current_error", "energy_drift",
    "norm_error",
]
const BOND_HEADER = [
    "L", "basis_tol", "seed", "policy", "step", "bond", "side", "pass", "chi_old", "ceiling",
    "width", "n_probes", "n_holdout", "n_selected", "n_krylov", "workcap", "met",
]
format_field(value::AbstractFloat) = isfinite(value) ? @sprintf("%.6e", value) : string(value)
format_field(value) = replace(string(value), ',' => ';')
write_row(io, fields) = (println(io, join(format_field.(fields), ",")); flush(io))

const NO_SAMPLE = (sz_error = NaN, current_error = NaN, energy_drift = NaN, norm_error = NaN)
is_sample_step(step) = step == 1 || step % SAMPLE_EVERY == 0 || step == STEP_COUNT

"""
Count every frame wider than the per-bond cap `ceilings`: the left frames, the final bonds, each
augmented K/L frame in the bond log and the right root frame. The branch's arms all use the
`:bond` rule, `min(maxdim, ceil(1.2 * d * chi_old))`.
"""
function cap_violations(info, psi, ceilings)
    count = sum(info.expanded .> ceilings) + sum(bond_dims(psi) .> ceilings)
    count += sum(entry.width > ceilings[entry.bond] for entry in info.ledger.bond_log; init = 0)
    return count + (info.root_right > ceilings[info.root])
end

"Evolve the Neel state with one BUG arm; one `steps` row per step, one `bonds` row per log entry."
function run_bug_arm(io; steps_io, bonds_io, chain_length, step_count, name, expansion, extra)
    hamiltonian = spin_xx_mpo(chain_length, SPIN_TWICE; J = 1.0)
    current_mpos = [spin_current_mpo(chain_length, SPIN_TWICE, bond) for bond = 1:(chain_length-1)]
    psi = neel_spin_state(chain_length)
    rng = MersenneTwister(SEED + 71)
    keywords = merge((order = 2, closure_checks = true), extra)
    for step = 1:step_count
        chi_old = bond_dims(psi)
        start = time_ns()
        info = cbe_bug_step!(
            psi,
            hamiltonian,
            ComplexF64(-im * TIME_STEP);
            maxdim = MAXDIM,
            tol_basis = BASIS_TOL,
            tol_trunc = TRUNC_TOL,
            tol_root = 1e-10,
            expansion = expansion,
            rng = rng,
            keywords...,
        )
        wall = (time_ns() - start) / 1e9
        io === nothing && continue                    # warm-up: compilation only
        ledger = info.ledger
        counters = info.record.counters
        ceilings = bond_work_ceilings(chi_old, SITE_DIM, MAXDIM; factor = extra.bond_ceiling_factor)
        log = ledger.bond_log
        graded = is_sample_step(step) ?
                 graded_observables(psi, hamiltonian, current_mpos, step * TIME_STEP) : NO_SAMPLE
        write_row(
            steps_io,
            (
                chain_length, BASIS_TOL, SEED, name, step, step * TIME_STEP, wall,
                maximum(bond_dims(psi)), info.record.D_stage, info.expanded[info.root],
                info.root_right, counters.n_one_site, ledger.n_h1_closure, counters.n_root,
                ledger.n_root_frontier, ledger.n_h2_leak, counters.n_stream_cols,
                counters.n_stream, ledger.n_probes, ledger.n_holdout, ledger.n_selected,
                ledger.n_krylov, maximum((entry.n_probes for entry in log); init = 0),
                maximum((entry.n_holdout for entry in log); init = 0),
                maximum((entry.n_selected for entry in log); init = 0),
                count(entry -> entry.workcap, log), cap_violations(info, psi, ceilings),
                budget_status(ledger), join(string.(ledger.unmet), ";"), graded.sz_error,
                graded.current_error, graded.energy_drift, graded.norm_error,
            ),
        )
        for entry in log
            write_row(
                bonds_io,
                (
                    chain_length, BASIS_TOL, SEED, name, step, entry.bond, entry.side,
                    get(entry, :pass, 0), chi_old[entry.bond], ceilings[entry.bond], entry.width,
                    entry.n_probes, entry.n_holdout, entry.n_selected, entry.n_krylov,
                    entry.workcap, entry.met,
                ),
            )
        end
        @printf(
            "RUN L=%d tol=%.0e %s seed=%d step=%d wall=%.2fs D=%d probes=%d holdout=%d sz_err=%.2e\n",
            chain_length, BASIS_TOL, name, SEED, step, wall, maximum(bond_dims(psi)),
            ledger.n_probes, ledger.n_holdout, graded.sz_error
        )
    end
end

"Two-site TDVP with the BUG arms' truncation budget at every cut; same rows, counters zero."
function run_tdvp2_arm(io; steps_io, chain_length, step_count)
    hamiltonian = spin_xx_mpo(chain_length, SPIN_TWICE; J = 1.0)
    current_mpos = [spin_current_mpo(chain_length, SPIN_TWICE, bond) for bond = 1:(chain_length-1)]
    psi = neel_spin_state(chain_length)
    for step = 1:step_count
        start = time_ns()
        tdvp2_step!(psi, hamiltonian, ComplexF64(-im * TIME_STEP); maxdim = MAXDIM, weight_tol = TRUNC_TOL)
        wall = (time_ns() - start) / 1e9
        io === nothing && continue
        graded = is_sample_step(step) ?
                 graded_observables(psi, hamiltonian, current_mpos, step * TIME_STEP) : NO_SAMPLE
        zeros_for_counters = zeros(Int, 19)
        write_row(
            steps_io,
            (
                chain_length, BASIS_TOL, SEED, "tdvp2", step, step * TIME_STEP, wall,
                maximum(bond_dims(psi)), zeros_for_counters..., "reference", "",
                graded.sz_error, graded.current_error, graded.energy_drift, graded.norm_error,
            ),
        )
        @printf(
            "TDVP2 L=%d tol=%.0e step=%d wall=%.2fs D=%d sz_err=%.2e\n",
            chain_length, BASIS_TOL, step, wall, maximum(bond_dims(psi)), graded.sz_error
        )
    end
end

function main()
    set_symmetry!(:U1)
    mkpath(OUTPUT_DIR)
    policies = Dict(name => (expansion, extra) for (name, expansion, extra) in policy_specs(BASIS_TOL))
    unknown = filter(name -> name != "tdvp2" && !haskey(policies, name), SELECTED_POLICIES)
    isempty(unknown) || error("unknown policies: $unknown")
    for name in SELECTED_POLICIES
        name == "tdvp2" && continue
        get(policies[name][2], :work_ceiling_rule, nothing) === :bond ||
            error("$name does not use the per-bond :bond cap this audit checks")
    end
    # Warm-up of every arm on a short chain: compilation is not counted in any wall time.
    for name in SELECTED_POLICIES
        if name == "tdvp2"
            run_tdvp2_arm(nothing; steps_io = nothing, chain_length = 8, step_count = 1)
        else
            expansion, extra = policies[name]
            run_bug_arm(nothing; steps_io = nothing, bonds_io = nothing, chain_length = 8,
                step_count = 1, name, expansion, extra)
        end
    end
    println("WARMUP_DONE")
    steps_io = open(joinpath(OUTPUT_DIR, "steps.csv"), "w")
    bonds_io = open(joinpath(OUTPUT_DIR, "bonds.csv"), "w")
    write_row(steps_io, STEP_HEADER)
    write_row(bonds_io, BOND_HEADER)
    for name in SELECTED_POLICIES
        if name == "tdvp2"
            # TDVP2 draws no random numbers, so one seed is enough.
            SEED == 0 && run_tdvp2_arm(true; steps_io, chain_length = CHAIN_LENGTH, step_count = STEP_COUNT)
        else
            expansion, extra = policies[name]
            run_bug_arm(true; steps_io, bonds_io, chain_length = CHAIN_LENGTH,
                step_count = STEP_COUNT, name, expansion, extra)
        end
    end
    close(steps_io)
    close(bonds_io)
    println("XX_LARGE_RSVD_AUDIT_COMPLETE L=$CHAIN_LENGTH tol=$BASIS_TOL seed=$SEED output=$OUTPUT_DIR")
end

main()
