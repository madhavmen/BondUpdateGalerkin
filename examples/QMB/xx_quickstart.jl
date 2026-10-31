# The rank-adaptive BUG step in one page: three tolerances, one resource limit, and a status.
#
#   julia --project examples/QMB/xx_quickstart.jl [OUTPUT_DIR] [L]
#
# Evolves the XX chain from a Neel state with `cbe_bug_step!`, reads each step's verdict with
# `budget_status`, and checks the S^z profile against the exact free-fermion solution. Then it asks
# for something infeasible (maxdim = 4) to show what a refused accuracy claim looks like.
#
# Expected output: three summary lines and `QUICKSTART-OK` (exit code 0). With OUTPUT_DIR, the
# final profiles are written to OUTPUT_DIR/xx_quickstart.csv. Defaults: L = 20, t = 1, dt = 0.05.
# Ported from BUG-Julia@b155550:examples/adaptive_quickstart.jl.

using LinearAlgebra
using BondUpdateGalerkin.BUG, BondUpdateGalerkin.QMB

const OUTPUT_DIR = length(ARGS) >= 1 ? ARGS[1] : nothing
const CHAIN_LENGTH = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 20
const TIME_STEP, FINAL_TIME = 0.05, 1.0

# ── exact reference: free fermions after Jordan-Wigner ──
# J (SxSx + SySy) is (J/2)(c+_l c_{l+1} + h.c.); an up spin is an occupied site, S^z = n - 1/2.
function exact_sz(chain_length, t; J = 1.0, occupied = 1:2:chain_length)
    hopping = zeros(ComplexF64, chain_length, chain_length)
    for site = 1:(chain_length-1)
        hopping[site, site+1] = hopping[site+1, site] = J / 2
    end
    propagator = exp(-im * t * hopping)
    return [sum(abs2(propagator[site, k]) for k in occupied) - 0.5 for site = 1:chain_length]
end

set_symmetry!(:U1)                                       # S^z_tot is conserved
hamiltonian = xxz_mpo(CHAIN_LENGTH; J = 1.0, delta = 0.0) # delta = 0 is the XX chain
sz_profile(psi) = magnetisation(copy(psi)) ./ norm(copy(psi))^2

"Evolve the Neel state to FINAL_TIME; return the max error vs exact, the bonds, the statuses."
function evolve(; settings...)
    psi = neel_state(CHAIN_LENGTH)
    statuses = Symbol[]
    for _ = 1:round(Int, FINAL_TIME / TIME_STEP)
        # parallel half-sweeps, as in the legacy quickstart the error claims below were made for
        info = cbe_bug_step!(
            psi,
            hamiltonian,
            ComplexF64(-im * TIME_STEP);
            parallel = true,
            settings...,
        )
        push!(statuses, budget_status(info.ledger))
    end
    profile = sz_profile(psi)
    return (
        error = maximum(abs.(profile .- exact_sz(CHAIN_LENGTH, FINAL_TIME))),
        bonds = bond_dims(psi),
        statuses = statuses,
        profile = profile,
    )
end

println("1) defaults (tol_basis 1e-5, tol_trunc 1e-7, tol_expv 1e-10, maxdim 128)")
default_run = evolve()
println(
    "   error vs exact $(default_run.error), max bond $(maximum(default_run.bonds)), all steps met: $(all(==(:met), default_run.statuses))",
)

println("2) looser truncation and root tolerance (target ~1e-4)")
loose_run = evolve(tol_basis = 1e-5, tol_trunc = 1e-3, tol_expv = 1e-2)
println(
    "   error vs exact $(loose_run.error), max bond $(maximum(loose_run.bonds)), all steps met: $(all(==(:met), loose_run.statuses))",
)

println("3) an infeasible request: tight tolerances with maxdim = 4")
capped_run = evolve(tol_basis = 1e-6, tol_trunc = 1e-8, maxdim = 4)
println(
    "   error vs exact $(capped_run.error), steps budget_unmet: $(count(==(:budget_unmet), capped_run.statuses)) of $(length(capped_run.statuses))",
)

if OUTPUT_DIR !== nothing
    mkpath(OUTPUT_DIR)
    exact = exact_sz(CHAIN_LENGTH, FINAL_TIME)
    lines = ["site,exact,default,loose,capped"]
    for site = 1:CHAIN_LENGTH
        values = (
            site,
            exact[site],
            default_run.profile[site],
            loose_run.profile[site],
            capped_run.profile[site],
        )
        push!(lines, join(string.(values), ","))
    end
    write(joinpath(OUTPUT_DIR, "xx_quickstart.csv"), join(lines, "\n") * "\n")
end

# the three claims of this page, checked rather than asserted in prose
passed =
    default_run.error < 1e-4 &&
    all(==(:met), default_run.statuses) &&
    loose_run.error < 1e-3 &&
    maximum(loose_run.bonds) <= maximum(default_run.bonds) &&
    any(==(:budget_unmet), capped_run.statuses)
println(passed ? "QUICKSTART-OK" : "QUICKSTART-FAILED")
passed || exit(1)
