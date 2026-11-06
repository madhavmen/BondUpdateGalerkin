# One-axis twisting: the analytic operator checks (the closed MPO against pair_mpo's automaton and
# the dense (sum S^z)^2/2) and the initial state.

using Test, LinearAlgebra
using BondUpdateGalerkin.BUG, BondUpdateGalerkin.QMB

include(joinpath(@__DIR__, "..", "..", "common", "dense_reference.jl"))

@testset "OAT: the operator" begin
    set_symmetry!(:none)
    L = 8
    mpo = oat_mpo(L)
    # Virtual dimension 3, INDEPENDENT of L -- the whole cost claim of the `λ = 1` self-loop.
    @test mpo_virtual_dims(mpo) == [1, 3, 3, 3, 3, 3, 3, 3, 1]
    @test mpo_virtual_dims(oat_mpo(16)) == [1; fill(3, 15); 1]

    # The SAME operator through `pair_mpo`'s completely different `O(L)` automaton.
    pairs = oat_mpo_pairs(L)
    @test maximum(mpo_virtual_dims(pairs)) > 3            # it really is the expensive route
    for st in (neel_state(L), domain_wall_state(L), x_polarized_state(L))
        @test real(mpo_energy(copy(st), mpo)) ≈ real(mpo_energy(copy(st), pairs)) atol = 1e-12
    end

    # And against the dense `(Σ S^z)^2 / 2`, MINUS the c-number `oat_mpo` drops.
    Hd = dense_total_sz(L)^2 / 2
    for st in (neel_state(L), domain_wall_state(L), x_polarized_state(L))
        v = dense_state(copy(st))
        @test real(mpo_energy(copy(st), mpo)) ≈ real(v' * Hd * v) - oat_energy_shift(L) atol = 1e-11
    end
end

@testset "OAT: the initial state and the observable" begin
    set_symmetry!(:none)
    L = 10
    psi = x_polarized_state(L)
    @test bond_dims(psi) == ones(Int, L - 1)              # the paper's "an MPS with D = 1"
    @test norm(psi) ≈ 1.0 atol = 1e-12
    @test sx_profile(copy(psi)) ≈ fill(0.5, L) atol = 1e-12
    @test total_sx(copy(psi)) ≈ L / 2 atol = 1e-12         # S^x_tot(0) = (L/2) cos^(L-1)(0)
    @test all(abs.([sz_expectation(copy(psi), j) for j = 1:L]) .< 1e-12)

    # `S^x`, not `-i S^y`: the sign trap named in `sx_operator`'s docstring, checked on a state
    # that is NOT symmetric under spin flip so the two would actually differ.
    @test Matrix(sx_operator().RMTs[1]) ≈ [0 0.5; 0.5 0] atol = 1e-12
    up = product_state(fill(:up, L))
    @test all(abs.(sx_profile(copy(up))) .< 1e-12)        # ⟨↑|S^x|↑⟩ = 0

    # The :none-only guards must THROW rather than return a symmetric zero.
    set_symmetry!(:U1)
    @test_throws ArgumentError x_polarized_state(L)
    @test_throws ArgumentError sx_operator()
    set_symmetry!(:none)
end
