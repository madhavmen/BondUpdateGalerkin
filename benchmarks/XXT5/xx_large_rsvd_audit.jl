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
# - every BUG arm stays inside the per-bond cap min(ceil(1.2 * d * D_b), D_max), D_b the bond's own
#   dimension at the start of the step and D_max = AUDIT_MAXDIM the global bond-dimension ceiling
#   -> cap_violations (must be 0).
#
# Model: nearest-neighbour spin-S XX chain, J = 1, open ends, U(1), d = 2S + 1 (AUDIT_TWOS = 2S),
# from a Neel or a domain-wall product state (AUDIT_INITIAL). Order 2, closure_checks = true, as in
# xx_T5_matched_work.jl; tol_trunc = 1e-3 * tol_basis, tol_root = 1e-10.
#
# References. Spin 1/2 is free fermions: the S^z profile and every bond current are exact at any L
# (test/common/free_fermion.jl). Spin 1 and 3/2 have no closed form, so AUDIT_MODE=reference first
# writes a converged TDVP2 run (dt / AUDIT_REF_SUBSTEPS, weight_tol 1e-12, maxdim AUDIT_REF_MAXDIM)
# to OUTPUT_DIR/reference.csv, and the arm runs read it through AUDIT_REFERENCE. A reference run at
# spin 1/2 also prints its own error against free fermions: that calibrates the TDVP2 reference.
#
# All arms of one (L, tol_basis, seed) run in ONE process, after a one-step warm-up of each arm
# at L = 8 (compilation excluded), so wall-clock ratios between arms come from the same node.
# Operator counters are the primary cost measure; they never depend on the node.
#
# Usage (repository root): julia --project benchmarks/XXT5/xx_large_rsvd_audit.jl OUTPUT_DIR
# Environment: AUDIT_MODE=arms|reference AUDIT_TWOS=1 AUDIT_INITIAL=neel|domain_wall AUDIT_L=64
#              AUDIT_TOL=1e-3 AUDIT_SEED=0 AUDIT_STEPS=50 AUDIT_DT=0.1 AUDIT_MAXDIM=256
#              AUDIT_SAMPLE_EVERY=5 AUDIT_POLICIES="name,...,tdvp2" AUDIT_REFERENCE=path
#              AUDIT_REF_SUBSTEPS=10 AUDIT_REF_MAXDIM=1024
# Arm runs write OUTPUT_DIR/steps.csv (one row per arm and step) and OUTPUT_DIR/bonds.csv (one row
# per bond-log entry of every BUG step).

using BondUpdateGalerkin.BUG, BondUpdateGalerkin.QMB, BondUpdateGalerkin.Baselines
using LinearAlgebra, Random, Printf
include(joinpath(@__DIR__, "..", "..", "test", "common", "free_fermion.jl"))
include(joinpath(@__DIR__, "xx_policy_specs.jl"))

isempty(ARGS) && error("usage: xx_large_rsvd_audit.jl OUTPUT_DIR")
const OUTPUT_DIR = ARGS[1]
const MODE = get(ENV, "AUDIT_MODE", "arms")
const SPIN_TWICE = parse(Int, get(ENV, "AUDIT_TWOS", "1"))
const SITE_DIM = SPIN_TWICE + 1
const INITIAL = get(ENV, "AUDIT_INITIAL", "neel")
const CHAIN_LENGTH = parse(Int, get(ENV, "AUDIT_L", "64"))
const BASIS_TOL = parse(Float64, get(ENV, "AUDIT_TOL", "1e-3"))
const TRUNC_TOL = 1e-3 * BASIS_TOL
const SEED = parse(Int, get(ENV, "AUDIT_SEED", "0"))
const STEP_COUNT = parse(Int, get(ENV, "AUDIT_STEPS", "50"))
const TIME_STEP = parse(Float64, get(ENV, "AUDIT_DT", "0.1"))
# D_max: the global bond-dimension ceiling. The per-bond working cap is min(ceil(1.2 d D_b), D_max)
# and the final compression keeps at most D_max (USER 11-10: D_max = 256).
const MAXDIM = parse(Int, get(ENV, "AUDIT_MAXDIM", "256"))
const SAMPLE_EVERY = parse(Int, get(ENV, "AUDIT_SAMPLE_EVERY", "5"))
const REFERENCE_PATH = get(ENV, "AUDIT_REFERENCE", "")
const REFERENCE_SUBSTEPS = parse(Int, get(ENV, "AUDIT_REF_SUBSTEPS", "10"))
const REFERENCE_MAXDIM = parse(Int, get(ENV, "AUDIT_REF_MAXDIM", "1024"))
const REFERENCE_WEIGHT_TOL = 1e-12
# cbe_bug_step!'s `parallel`: false (serial), true (the K and L half-sweeps run concurrently) or
# full (`:full`, the parallel K/L/S integrator, itself a different step). Needs `julia -t 2` or more.
const PARALLEL_SETTING = get(ENV, "AUDIT_PARALLEL", "false")
const PARALLEL =
    PARALLEL_SETTING == "false" ? false :
    PARALLEL_SETTING == "true" ? true :
    PARALLEL_SETTING == "full" ? :full :
    error("AUDIT_PARALLEL must be false, true or full, got $PARALLEL_SETTING")
PARALLEL === false ||
    Threads.nthreads() >= 2 ||
    error("AUDIT_PARALLEL=$PARALLEL_SETTING needs julia -t 2 or more, have $(Threads.nthreads())")
MODE in ("arms", "reference") || error("AUDIT_MODE must be arms or reference, got $MODE")
INITIAL in ("neel", "domain_wall") ||
    error("AUDIT_INITIAL must be neel or domain_wall, got $INITIAL")
SPIN_TWICE >= 1 || error("AUDIT_TWOS must be positive")

# AUDIT_SET=branch: the branch's own arms (policy_specs). AUDIT_SET=krylov_factorial: each RSVD
# option of the branch crossed with each adaptive local Krylov solver (USER 11-10: the bond grows
# far past the exact chi, so look at how the RSVD options behave with Lanczos and Arnoldi).
const POLICY_SET = get(ENV, "AUDIT_SET", "branch")
POLICY_SET in ("branch", "krylov_factorial") ||
    error("AUDIT_SET must be branch or krylov_factorial, got $POLICY_SET")

# The RSVD options exactly as the branch's arms set them (xx_policy_specs.jl, RSVD_* entries).
const RSVD_OPTIONS = [
    ("fixed8rank4", (mode = :fixed, fixed_width = 8, fixed_rank = 4)),
    ("fixed12rank8", (mode = :fixed, fixed_width = 12, fixed_rank = 8)),
    (
        "rank_adaptive",
        (mode = :rank_adaptive, rank_initial_width = 4, holdout_width = 4, acceptance_factor = 0.7),
    ),
    (
        "rank_fast3",
        (mode = :rank_fast, rank_initial_width = 3, holdout_width = 2, acceptance_factor = 0.6),
    ),
    ("rank_tail12", (mode = :rank_tail, fixed_width = 12, tail_factor = 0.10)),
    ("rank_tail8", (mode = :rank_tail, fixed_width = 8, tail_factor = 0.10)),
]
# The adaptive local Krylov solvers: the recurrence stops once the residual weight of the next
# direction is within its allowance. `ranked` adds krylov_rank_tol = tol_basis, which keeps only
# the Krylov directions that carry weight (_append_ranked) instead of every accepted vector.
const KRYLOV_OPTIONS = [
    ("lanczos", (krylov_mode = :lanczos,), false),
    ("arnoldi", (krylov_mode = :arnoldi,), false),
    ("lanczos_ranked", (krylov_mode = :lanczos,), true),
]
const FACTORIAL_NAMES =
    ["$(rsvd_name)+$(krylov_name)" for (rsvd_name, _) in RSVD_OPTIONS for (krylov_name, _, _) in KRYLOV_OPTIONS]

"Every RSVD option x every adaptive Krylov solver, all under the per-bond :bond cap 1.2 d D_b."
function krylov_factorial_specs(basis_tol)
    common = (work_ceiling_rule = :bond, bond_ceiling_factor = 1.2)
    return [
        (
            "$(rsvd_name)+$(krylov_name)",
            DDQRExpansion(; rsvd..., krylov...),
            ranked ? merge(common, (krylov_rank_tol = basis_tol,)) : common,
        ) for (rsvd_name, rsvd) in RSVD_OPTIONS for (krylov_name, krylov, ranked) in KRYLOV_OPTIONS
    ]
end
policy_set_specs(basis_tol) =
    POLICY_SET == "krylov_factorial" ? krylov_factorial_specs(basis_tol) : policy_specs(basis_tol)

# The arms added on oldcap-fast-rsvd-xx and the baselines its workflow compares them with.
const BRANCH_POLICIES = [
    "RSVD_pilot2_Kadapt",
    "RSVD_pilot3_Kadapt",
    "RSVD_pilot4_Kadapt",
    "RSVD_pilot3_margin8",
    "RSVD_rank_standard_Kadapt",
    "RSVD_fixed8rank4_Kadapt",
    "RSVD_fixed12rank8_Kadapt",
    "RSVD_sequential_Kadapt",
    "RSVD_batched_Kadapt",
    "RSVD_tail12_factor01",
    "RSVD_tail12_factor005",
    "RSVD_tail8_factor01",
    "tdvp2",
]
const DEFAULT_POLICIES =
    POLICY_SET == "krylov_factorial" ? vcat(FACTORIAL_NAMES, "tdvp2") : BRANCH_POLICIES
const SELECTED_POLICIES =
    filter(!isempty, strip.(split(get(ENV, "AUDIT_POLICIES", join(DEFAULT_POLICIES, ",")), ",")))

"Sites starting at S^z = +S: the odd sites (Neel) or the left half (domain wall)."
up_sites(chain_length) = INITIAL == "neel" ? (1:2:chain_length) : (1:(chain_length÷2))

"The initial product state, built in the same spin space as `spin_xx_mpo`."
function initial_state(chain_length)
    up = up_sites(chain_length)
    return sector_product_state(
        spin_space(SPIN_TWICE),
        [(((site in up ? SPIN_TWICE : -SPIN_TWICE),),) for site = 1:chain_length],
    )
end

"Free-fermion S^z profile and bond currents at time `t` (spin 1/2 only): up sites are occupied."
function free_fermion_observables(chain_length, t)
    correlation = xx_open_correlation(chain_length, t; J = 1.0, occupied = up_sites(chain_length))
    return (
        sz = [real(correlation[site, site]) - 0.5 for site = 1:chain_length],
        current = xx_open_current(correlation; J = 1.0),
    )
end

"""
S^z profile and bond currents of an MPS, each divided by the MPS norm, plus <H> and the norm.

The current is `-mpo_energy(psi, spin_current_mpo(...))`. Measured (smoke job 17875862, L = 16 Neel):
`mpo_energy` of `spin_current_mpo` is exactly `-j_b` of the free-fermion `xx_open_current` for TDVP2
and all 12 BUG arms, while their S^z profiles agree (TDVP2 to 1e-7). The free-fermion sign is the
physical one: site 1 starts up, so `d<S^z_1>/dt = -j_1 < 0` needs `j_1 > 0`, and `mpo_energy` gives
`j_1 < 0`. The sign defect is in the QMB current operator or its expectation, not in the integrators.
"""
function measured_observables(psi, hamiltonian, current_mpos)
    norm_squared = real(overlap(psi, psi))
    return (
        sz = spin_sz_profile(psi, SPIN_TWICE) ./ norm_squared,
        current = [
            -real(mpo_energy(psi, current_mpo)) / norm_squared for current_mpo in current_mpos
        ],
        energy = real(mpo_energy(psi, hamiltonian)) / norm_squared,
        norm_squared = norm_squared,
    )
end

"`reference.csv` rows `quantity,index,step,value` -> Dict(step => (sz, current))."
function load_reference(path, chain_length)
    sz_by_step = Dict{Int,Vector{Float64}}()
    current_by_step = Dict{Int,Vector{Float64}}()
    for line in Iterators.drop(eachline(path), 1)
        quantity, index, step, value = split(line, ",")
        quantity in ("sz", "current") || continue
        table = quantity == "sz" ? sz_by_step : current_by_step
        entry_count = quantity == "sz" ? chain_length : chain_length - 1
        profile = get!(() -> fill(NaN, entry_count), table, parse(Int, step))
        profile[parse(Int, index)] = parse(Float64, value)
    end
    reference = Dict(
        step => (sz = sz_by_step[step], current = current_by_step[step]) for
        step in keys(sz_by_step)
    )
    all(entry -> !any(isnan, entry.sz) && !any(isnan, entry.current), values(reference)) ||
        error("reference $path is incomplete or was written for another L")
    return reference
end

const REFERENCE =
    MODE == "arms" && !isempty(REFERENCE_PATH) ? load_reference(REFERENCE_PATH, CHAIN_LENGTH) :
    nothing
MODE == "arms" &&
    SPIN_TWICE != 1 &&
    REFERENCE === nothing &&
    error(
        "spin $(SPIN_TWICE)/2 has no free-fermion solution: run AUDIT_MODE=reference and pass AUDIT_REFERENCE",
    )

"The values every arm is graded against at `step`."
reference_observables(chain_length, step) =
    REFERENCE === nothing ? free_fermion_observables(chain_length, step * TIME_STEP) :
    REFERENCE[step]

"Errors of the MPS against the reference at `step`."
function graded_observables(psi, hamiltonian, current_mpos, step)
    reference = reference_observables(length(psi), step)
    measured = measured_observables(psi, hamiltonian, current_mpos)
    return (
        sz_error = maximum(abs.(measured.sz .- reference.sz)),
        current_error = maximum(abs.(measured.current .- reference.current)),
        # Both product states have <H> = 0 and the evolution conserves it exactly.
        energy_drift = abs(measured.energy),
        norm_error = abs(sqrt(measured.norm_squared) - 1),
    )
end

const STEP_HEADER = [
    "initial",
    "twoS",
    "L",
    "basis_tol",
    "seed",
    "policy",
    "step",
    "t",
    "wall_s",
    "alloc_bytes",
    "D_max",
    "D_stage",
    "D_root_left",
    "D_root_right",
    "h1",
    "h1_closure",
    "root",
    "root_frontier",
    "h2_leak",
    "stream_cols",
    "stream_calls",
    "dd_probes",
    "dd_holdout",
    "dd_selected",
    "n_krylov",
    "max_entry_probes",
    "max_entry_holdout",
    "max_entry_selected",
    "workcap_entries",
    "cap_violations",
    "budget_status",
    "unmet",
    "sz_error",
    "current_error",
    "energy_drift",
    "norm_error",
    # Step time by component (cbe_bug_step!'s own timers): DD sketch-and-select, half-sweep Krylov
    # frames, frame SVDs, environments, root solve, final truncation, global Krylov closure.
    "t_cbe",
    "t_kry",
    "t_frame",
    "t_env",
    "t_root",
    "t_trunc",
    "t_closure",
    # Every bond dimension after the step, ";"-separated, at sampled steps only.
    "bond_profile",
    # cbe_bug_step!'s `parallel` setting and the Julia thread count of the run.
    "parallel",
    "threads",
]
const BOND_HEADER = [
    "initial",
    "twoS",
    "L",
    "basis_tol",
    "seed",
    "policy",
    "step",
    "bond",
    "side",
    "pass",
    "chi_old",
    "ceiling",
    "width",
    "n_probes",
    "n_holdout",
    "n_selected",
    "n_krylov",
    "workcap",
    "met",
]
# 13 significant digits: reference values are compared at errors far below 1e-6.
format_field(value::AbstractFloat) = isfinite(value) ? @sprintf("%.12e", value) : string(value)
format_field(value) = replace(string(value), ',' => ';')
write_row(io, fields) = (println(io, join(format_field.(fields), ",")); flush(io))

const NO_SAMPLE = (sz_error = NaN, current_error = NaN, energy_drift = NaN, norm_error = NaN)
is_sample_step(step) = step == 1 || step % SAMPLE_EVERY == 0 || step == STEP_COUNT
bond_profile(psi, step) = is_sample_step(step) ? join(bond_dims(psi), ";") : ""

"The chain's Hamiltonian and its bond-current operators, both as MPOs."
model_mpos(chain_length) = (
    spin_xx_mpo(chain_length, SPIN_TWICE; J = 1.0),
    [spin_current_mpo(chain_length, SPIN_TWICE, bond) for bond = 1:(chain_length-1)],
)

"""
Count every frame wider than the per-bond cap `ceilings`: the left frames, the final bonds, each
augmented K/L frame in the bond log and the right root frame. The branch's arms all use the
`:bond` rule, `min(ceil(1.2 * d * chi_old), D_max)` per bond.
"""
function cap_violations(info, psi, ceilings)
    count = sum(info.expanded .> ceilings) + sum(bond_dims(psi) .> ceilings)
    count += sum(entry.width > ceilings[entry.bond] for entry in info.ledger.bond_log; init = 0)
    return count + (info.root_right > ceilings[info.root])
end

"Evolve the initial state with one BUG arm; one `steps` row per step, one `bonds` row per log entry."
function run_bug_arm(io; steps_io, bonds_io, chain_length, step_count, name, expansion, extra)
    hamiltonian, current_mpos = model_mpos(chain_length)
    psi = initial_state(chain_length)
    rng = MersenneTwister(SEED + 71)
    keywords = merge((order = 2, closure_checks = true, parallel = PARALLEL), extra)
    for step = 1:step_count
        chi_old = bond_dims(psi)
        timed = @timed cbe_bug_step!(
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
        io === nothing && continue                    # warm-up: compilation only
        info = timed.value
        ledger = info.ledger
        counters = info.record.counters
        ceilings = bond_work_ceilings(chi_old, SITE_DIM, MAXDIM; factor = extra.bond_ceiling_factor)
        log = ledger.bond_log
        graded =
            is_sample_step(step) ? graded_observables(psi, hamiltonian, current_mpos, step) :
            NO_SAMPLE
        write_row(
            steps_io,
            (
                INITIAL,
                SPIN_TWICE,
                chain_length,
                BASIS_TOL,
                SEED,
                name,
                step,
                step * TIME_STEP,
                timed.time,
                timed.bytes,
                maximum(bond_dims(psi)),
                info.record.D_stage,
                info.expanded[info.root],
                info.root_right,
                counters.n_one_site,
                ledger.n_h1_closure,
                counters.n_root,
                ledger.n_root_frontier,
                ledger.n_h2_leak,
                counters.n_stream_cols,
                counters.n_stream,
                ledger.n_probes,
                ledger.n_holdout,
                ledger.n_selected,
                ledger.n_krylov,
                maximum((entry.n_probes for entry in log); init = 0),
                maximum((entry.n_holdout for entry in log); init = 0),
                maximum((entry.n_selected for entry in log); init = 0),
                count(entry -> entry.workcap, log),
                cap_violations(info, psi, ceilings),
                budget_status(ledger),
                join(string.(ledger.unmet), ";"),
                graded.sz_error,
                graded.current_error,
                graded.energy_drift,
                graded.norm_error,
                info.t_cbe,
                info.t_kry,
                info.t_frame,
                info.t_env,
                info.t_root,
                info.t_trunc,
                ledger.t_closure,
                bond_profile(psi, step),
                PARALLEL_SETTING,
                Threads.nthreads(),
            ),
        )
        for entry in log
            write_row(
                bonds_io,
                (
                    INITIAL,
                    SPIN_TWICE,
                    chain_length,
                    BASIS_TOL,
                    SEED,
                    name,
                    step,
                    entry.bond,
                    entry.side,
                    get(entry, :pass, 0),
                    chi_old[entry.bond],
                    ceilings[entry.bond],
                    entry.width,
                    entry.n_probes,
                    entry.n_holdout,
                    entry.n_selected,
                    entry.n_krylov,
                    entry.workcap,
                    entry.met,
                ),
            )
        end
        @printf(
            "RUN %s d=%d L=%d tol=%.0e %s seed=%d step=%d wall=%.2fs D=%d probes=%d holdout=%d sz_err=%.2e\n",
            INITIAL,
            SITE_DIM,
            chain_length,
            BASIS_TOL,
            name,
            SEED,
            step,
            timed.time,
            maximum(bond_dims(psi)),
            ledger.n_probes,
            ledger.n_holdout,
            graded.sz_error
        )
    end
end

"Two-site TDVP with the BUG arms' truncation budget at every cut; same rows, counters zero."
function run_tdvp2_arm(io; steps_io, chain_length, step_count)
    hamiltonian, current_mpos = model_mpos(chain_length)
    psi = initial_state(chain_length)
    for step = 1:step_count
        timed = @timed tdvp2_step!(
            psi,
            hamiltonian,
            ComplexF64(-im * TIME_STEP);
            maxdim = MAXDIM,
            weight_tol = TRUNC_TOL,
        )
        io === nothing && continue
        graded =
            is_sample_step(step) ? graded_observables(psi, hamiltonian, current_mpos, step) :
            NO_SAMPLE
        zeros_for_counters = zeros(Int, 19)
        write_row(
            steps_io,
            (
                INITIAL,
                SPIN_TWICE,
                chain_length,
                BASIS_TOL,
                SEED,
                "tdvp2",
                step,
                step * TIME_STEP,
                timed.time,
                timed.bytes,
                maximum(bond_dims(psi)),
                zeros_for_counters...,
                "reference",
                "",
                graded.sz_error,
                graded.current_error,
                graded.energy_drift,
                graded.norm_error,
                zeros(7)...,
                bond_profile(psi, step),
                "false",
                Threads.nthreads(),
            ),
        )
        @printf(
            "TDVP2 %s d=%d L=%d tol=%.0e step=%d wall=%.2fs D=%d sz_err=%.2e\n",
            INITIAL,
            SITE_DIM,
            chain_length,
            BASIS_TOL,
            step,
            timed.time,
            maximum(bond_dims(psi)),
            graded.sz_error
        )
    end
end

"""
The converged TDVP2 reference: `REFERENCE_SUBSTEPS` substeps per audit step, weight_tol 1e-12,
maxdim `REFERENCE_MAXDIM`. Writes S^z and the bond currents at every sampled step; at spin 1/2 it
also reports its own worst error against free fermions, which calibrates the reference.
"""
function write_reference()
    hamiltonian, current_mpos = model_mpos(CHAIN_LENGTH)
    psi = initial_state(CHAIN_LENGTH)
    substep = TIME_STEP / REFERENCE_SUBSTEPS
    worst_calibration = 0.0
    open(joinpath(OUTPUT_DIR, "reference.csv"), "w") do io
        write_row(io, ["quantity", "index", "step", "value"])
        for step = 1:STEP_COUNT
            for _ = 1:REFERENCE_SUBSTEPS
                tdvp2_step!(
                    psi,
                    hamiltonian,
                    ComplexF64(-im * substep);
                    maxdim = REFERENCE_MAXDIM,
                    weight_tol = REFERENCE_WEIGHT_TOL,
                )
            end
            is_sample_step(step) || continue
            measured = measured_observables(psi, hamiltonian, current_mpos)
            for (site, value) in enumerate(measured.sz)
                write_row(io, ("sz", site, step, value))
            end
            for (bond, value) in enumerate(measured.current)
                write_row(io, ("current", bond, step, value))
            end
            calibration = NaN
            if SPIN_TWICE == 1
                exact = free_fermion_observables(CHAIN_LENGTH, step * TIME_STEP)
                calibration = max(
                    maximum(abs.(measured.sz .- exact.sz)),
                    maximum(abs.(measured.current .- exact.current)),
                )
                worst_calibration = max(worst_calibration, calibration)
            end
            @printf(
                "REFERENCE %s d=%d L=%d step=%d D=%d energy=%.2e calibration_vs_free_fermions=%.2e\n",
                INITIAL,
                SITE_DIM,
                CHAIN_LENGTH,
                step,
                maximum(bond_dims(psi)),
                measured.energy,
                calibration
            )
        end
    end
    println("REFERENCE_WORST_CALIBRATION=", SPIN_TWICE == 1 ? worst_calibration : NaN)
end

function run_arms()
    policies = Dict(
        name => (expansion, extra) for (name, expansion, extra) in policy_set_specs(BASIS_TOL)
    )
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
            run_bug_arm(
                nothing;
                steps_io = nothing,
                bonds_io = nothing,
                chain_length = 8,
                step_count = 1,
                name,
                expansion,
                extra,
            )
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
            SEED == 0 &&
                run_tdvp2_arm(true; steps_io, chain_length = CHAIN_LENGTH, step_count = STEP_COUNT)
        else
            expansion, extra = policies[name]
            run_bug_arm(
                true;
                steps_io,
                bonds_io,
                chain_length = CHAIN_LENGTH,
                step_count = STEP_COUNT,
                name,
                expansion,
                extra,
            )
        end
    end
    close(steps_io)
    close(bonds_io)
end

function main()
    set_symmetry!(:U1)
    mkpath(OUTPUT_DIR)
    MODE == "reference" ? write_reference() : run_arms()
    println(
        "XX_LARGE_RSVD_AUDIT_COMPLETE mode=$MODE $INITIAL d=$SITE_DIM L=$CHAIN_LENGTH tol=$BASIS_TOL seed=$SEED output=$OUTPUT_DIR",
    )
end

main()
