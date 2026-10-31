# Regression against the WP0 fixtures frozen from BUG-Julia b155550 (test/fixtures/b155550): the
# moved engine must give the same scientific answers as the legacy one (implementation plan, WP3
# exit gate and PR 4 proof "unchanged QMB numerical results").
#
# Each case reruns the fixture script's own calls (docs/dev/tools/make_fixtures_b155550.jl) with the
# same seed. The randomised sketches and the thread count make the runs differ at the truncation
# tolerance, never bit for bit, so every quantity is compared with a tolerance:
#   - against the frozen run: S^z profiles to 1e-5, energies to 1e-6 relative;
#   - against the exact answer: no worse than twice the frozen error plus 1e-6;
#   - every step's budget status as frozen ("met").
# To keep the suite near five minutes, the U(1) runs and the imaginary-time run compare their
# first steps only (each frozen row is checked on its own, so a prefix is a valid comparison).

using Test, LinearAlgebra
import JSON, Random
using BondUpdateGalerkin.BUG, BondUpdateGalerkin.QMB

include(joinpath(@__DIR__, "..", "..", "common", "dense_reference.jl"))
include(joinpath(@__DIR__, "..", "..", "common", "free_fermion.jl"))

const FIXTURE_DIR = joinpath(@__DIR__, "..", "..", "fixtures", "b155550")
const COMPARED_STEPS = 6                       # of the 20 real-time and 8 imaginary-time rows
read_fixture(name) = JSON.parsefile(joinpath(FIXTURE_DIR, name * ".json"))

"<S^z_j> of every site, normalised by the state's own norm (the fixture script's `sz_profile`)."
sz_profile(psi) = real.(magnetisation(copy(psi))) ./ real(overlap(psi, psi))

"The budget status of a step's info, as the fixture script records it."
step_status(info) = hasproperty(info, :ledger) ? String(budget_status(info.ledger)) : nothing

@testset "XX chain under U(1) against b155550: $name" for (name, order, midpoint) in (
    ("xx_u1_bug_order1", 1, false),
    ("xx_u1_bug_order2", 2, false),
    ("xx_u1_bug_midpoint", 1, true),
)
    frozen = read_fixture(name)
    set_symmetry!(:U1)
    clear_fusion_cache!()
    Random.seed!(1234)
    chain_length, time_step = frozen["L"], frozen["dt"]
    hamiltonian = xxz_mpo(chain_length; J = 1.0, delta = 0.0)
    psi = neel_state(chain_length)
    for (step, frozen_row) in enumerate(first(frozen["trajectory"], COMPARED_STEPS))
        info =
            midpoint ? cbe_bug_midpoint_step!(psi, hamiltonian, ComplexF64(-im * time_step)) :
            cbe_bug_step!(psi, hamiltonian, ComplexF64(-im * time_step); order = order)
        exact_sz =
            xx_free_fermion_sz(chain_length, step * time_step; J = 1.0, occupied = 1:2:chain_length)
        sz = sz_profile(psi)
        @test maximum(abs.(sz .- Float64.(frozen_row["sz"]))) <= 1e-5
        @test maximum(abs.(sz .- exact_sz)) <= 2 * frozen_row["err_exact"] + 1e-6
        @test step_status(info) == frozen_row["status"]
    end
end

@testset "XX chain without symmetry against b155550" begin
    frozen = read_fixture("xx_none_bug_order2")
    set_symmetry!(:none)
    clear_fusion_cache!()
    Random.seed!(1234)
    chain_length, time_step, steps = frozen["L"], frozen["dt"], frozen["steps"]
    hamiltonian = xxz_mpo(chain_length; J = 1.0, delta = 0.0)
    psi = neel_state(chain_length)
    for frozen_row in frozen["trajectory"]
        info = cbe_bug_step!(psi, hamiltonian, ComplexF64(-im * time_step); order = 2)
        @test step_status(info) == frozen_row["status"]
        @test sqrt(real(overlap(psi, psi))) ≈ frozen_row["norm"] atol = 1e-8
    end
    final_state = dense_state(psi)
    frozen_state =
        ComplexF64.(frozen["final_state"]["re"]) .+ im .* ComplexF64.(frozen["final_state"]["im"])
    exact_state =
        exp(
            -im * steps * time_step * Matrix(dense_heisenberg(chain_length; J = 1.0, delta = 0.0)),
        ) * dense_state(neel_state(chain_length))
    @test norm(final_state - frozen_state) <= 1e-5 * norm(frozen_state)
    @test norm(final_state - exact_state) / norm(exact_state) <= 2 * frozen["err_dense"] + 1e-6
    set_symmetry!(:U1)
end

@testset "Heisenberg imaginary time against b155550" begin
    frozen = read_fixture("heisenberg_ite")
    set_symmetry!(:U1)
    clear_fusion_cache!()
    Random.seed!(1234)
    chain_length = frozen["L"]
    hamiltonian = xxz_mpo(chain_length; J = 1.0, delta = 1.0)
    psi = neel_state(chain_length)
    previous_energy = Inf
    for frozen_row in first(frozen["trajectory"], COMPARED_STEPS ÷ 2)
        history = ite_evolve!(psi, hamiltonian, uniform_beta_steps(0.25, 2); order = 2)
        energy_now = real(mpo_energy(psi, hamiltonian)) / real(overlap(psi, psi))
        @test energy_now ≈ frozen_row["energy"] rtol = 1e-6
        @test energy_now < previous_energy                     # imaginary time lowers the energy
        @test energy_now >= frozen["e_exact_ground"] - 1e-10   # and never below the ground state
        @test [String(record.status) for record in history] == frozen_row["status"]
        previous_energy = energy_now
    end
end
