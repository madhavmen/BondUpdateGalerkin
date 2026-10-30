# The essential checks of the quantum-jump engine on a 4-site chain, against dense operators.
#
# Written for PR 6b in place of BUG-Julia@b155550:tests/sweeps/test_dissipative_t4.jl, whose J0/U13
# sets need a 454-line fixture file, tdvp2 (a Baseline) and an 80-trajectory ensemble. What stays
# pinned: a site-emission jump is sqrt(gamma_j) sigma^-_j on the dense state (through BUG's
# `factor_product` with QMB's `SiteOp` method and through `apply_jump`), its weight is the jumped
# state's squared norm, the weights add up to the no-jump decay operator Q, and a seeded trajectory
# is reproducible with normalised conditional states.

using Test, LinearAlgebra
using LurCGT, Telum
using BondUpdateGalerkin.BUG, BondUpdateGalerkin.QMB

include(joinpath(@__DIR__, "..", "..", "..", "common", "dense_reference.jl"))

@testset "quantum jumps: site emission on a 4-site XX chain" begin
    set_symmetry!(:U1)
    chain_length = 4
    loss_rates = [0.3, 0.5, 0.7, 0.9]
    hamiltonian = xxz_mpo(chain_length; J = 1.0, delta = 0.0)
    # Q = sum_j gamma_j sigma^+_j sigma^-_j = sum_j gamma_j |up><up|_j, the no-jump decay.
    spin = local_space()
    up_projector = to_concrete(spin.Sz + 0.5 * spin.I)
    loss_mpo = onsite_mpo(
        chain_length,
        [to_concrete(rate * up_projector) for rate in loss_rates];
        Iloc = spin.I,
    )
    channels = site_emission_channels(chain_length; gamma = loss_rates)

    # An entangled state with weight on every site: the Neel state after a short XX evolution.
    psi = neel_state(chain_length)
    for _ = 1:2
        cbe_bug_step!(psi, hamiltonian, ComplexF64(-0.3im); maxdim = 16)
    end
    dense_psi = dense_state(psi)

    @testset "each jump is sqrt(gamma_j) sigma^-_j on the dense state" begin
        @test length(channels) == chain_length
        for (site, channel) in enumerate(channels)
            @test channel.factors[1] isa SiteOp                     # a local jump needs no MPO
            expected = sqrt(loss_rates[site]) * _embed(_SM, site, chain_length) * dense_psi
            scale = max(norm(expected), 1e-3)
            # `factor_product` is BUG's function; QMB adds the method for its `SiteOp`.
            @test norm(dense_state(factor_product(channel.factors[1], psi)) - expected) <
                  1e-12 * scale
            @test abs(jump_weight(channel, psi) - norm(expected)^2) < 1e-12 * scale^2
            jumped, discarded = apply_jump(channel, psi)
            @test discarded < 1e-14
            @test norm(dense_state(jumped) - expected) < 1e-10 * scale
        end
        # The jump weights add up to <psi|Q|psi>: the decay the no-jump generator carries.
        @test sum(jump_weight(channel, psi) for channel in channels) ≈
              real(mpo_energy(psi, loss_mpo)) atol = 1e-12
    end

    @testset "a seeded trajectory is reproducible, its conditional states normalised" begin
        no_jump_generator = no_jump_mpo(hamiltonian, loss_mpo)
        time_grid = [0.0, 0.5, 1.0]
        norm_squared(state) = real(overlap(state, state))
        run_trajectory(seed) = jump_trajectory(
            neel_state(chain_length),
            no_jump_generator,
            channels,
            time_grid;
            dt = 0.1,
            seed = seed,
            observe = norm_squared,
        )
        first_run, repeated_run = run_trajectory(7), run_trajectory(7)
        @test [(event.t, event.channel) for event in first_run.events] == [(event.t, event.channel) for event in repeated_run.events]
        @test all(abs(observed - 1) < 1e-10 for observed in first_run.obs)
        @test first_run.njumps[end] == length(first_run.events)
        @test first_run.times == time_grid
    end
end
