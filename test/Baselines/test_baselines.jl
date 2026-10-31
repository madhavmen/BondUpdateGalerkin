# The comparison methods (implementation plan WP6): each baseline against an independent reference
# on the same problem the BUG tests use.
#
# From BUG-Julia@b155550: the `baseline_tdvp2_xx` fixture (TDVP2 on the XX chain), the CBE-DMRG
# chain case of tests/sweeps/test_adaptive_dmrg.jl, the unitarity check of the one-site TDVP-CBE
# in tests/sweeps/test_sweeps.jl, and a QTT TDVP2 step against the dense exponential.

using Test, LinearAlgebra
import JSON, Random
using BondUpdateGalerkin.BUG, BondUpdateGalerkin.QMB, BondUpdateGalerkin.QTTS
using BondUpdateGalerkin.Baselines

include(joinpath(@__DIR__, "..", "common", "dense_reference.jl"))
include(joinpath(@__DIR__, "..", "common", "free_fermion.jl"))
include(joinpath(@__DIR__, "..", "common", "qtt_fixtures.jl"))

const FIXTURE_DIR = joinpath(@__DIR__, "..", "fixtures", "b155550")

"<S^z_j> of every site, normalised by the state's own norm (the fixture script's `sz_profile`)."
sz_profile(psi) = real.(magnetisation(copy(psi))) ./ real(overlap(psi, psi))

@testset "baselines" begin
    @testset "TDVP2 on the XX chain against b155550 and free fermions" begin
        frozen = JSON.parsefile(joinpath(FIXTURE_DIR, "baseline_tdvp2_xx.json"))
        set_symmetry!(:U1)
        clear_fusion_cache!()
        chain_length, time_step = frozen["L"], frozen["dt"]
        hamiltonian = xxz_mpo(chain_length; J = 1.0, delta = 0.0)
        psi = neel_state(chain_length)
        for (step, frozen_row) in enumerate(frozen["trajectory"])
            tdvp2_step!(
                psi,
                hamiltonian,
                ComplexF64(-im * time_step);
                maxdim = 64,
                trunc_thresh = 1e-12,
            )
            exact_sz = xx_free_fermion_sz(
                chain_length,
                step * time_step;
                J = 1.0,
                occupied = 1:2:chain_length,
            )
            sz = sz_profile(psi)
            @test maximum(abs.(sz .- Float64.(frozen_row["sz"]))) <= 1e-5
            @test maximum(abs.(sz .- exact_sz)) <= 2 * frozen_row["err_exact"] + 1e-6
        end
        # the fourth-order composition beats the second-order step at the same coarse dt
        errors = map((2, 4)) do order
            state = neel_state(chain_length)
            for _ = 1:5
                tdvp2_composed_step!(
                    state,
                    hamiltonian,
                    ComplexF64(-im * 0.1);
                    order = order,
                    maxdim = 64,
                )
            end
            maximum(
                abs.(
                    sz_profile(state) .-
                    xx_free_fermion_sz(chain_length, 0.5; J = 1.0, occupied = 1:2:chain_length)
                ),
            )
        end
        @info "TDVP2 order 2 / order 4 errors at dt 0.1: $errors"
        @test errors[2] < errors[1]
    end

    @testset "one-site TDVP with CBE: unitary, close to free fermions" begin
        set_symmetry!(:U1)
        clear_fusion_cache!()
        chain_length, time_step, steps = 8, 0.05, 10
        exact_sz = xx_free_fermion_sz(
            chain_length,
            steps * time_step;
            J = 1.0,
            occupied = 1:2:chain_length,
        )
        mpo_state = neel_state(chain_length)
        chain_state = neel_state(chain_length)
        initial_norm = norm(mpo_state)
        hamiltonian = xxz_mpo(chain_length; J = 1.0, delta = 0.0)
        chain = xxz_chain(chain_length; J = 1.0, delta = 0.0)
        for _ = 1:steps
            tdvp_cbe1s_step!(
                mpo_state,
                hamiltonian,
                -im * time_step;
                maxdim = 64,
                trunc_thresh = 1e-14,
            )
            tdvp1_cbe_step!(chain_state, chain, -im * time_step; maxdim = 64, trunc_thresh = 1e-14)
        end
        for state in (mpo_state, chain_state)
            @test norm(state) ≈ initial_norm atol = 1e-9                     # real time is unitary
            @test maximum(bond_dims(state)) > 1                              # CBE grew the rank
            @test maximum(abs.(sz_profile(state) .- exact_sz)) < 1e-2
        end
    end

    @testset "CBE-DMRG: exact ground energy, variational" begin
        set_symmetry!(:U1)
        clear_fusion_cache!()
        chain_length = 8
        hamiltonian = xxz_mpo(chain_length; J = 1.0, delta = 1.0)
        exact_energy = eigen(Hermitian(Matrix(dense_heisenberg(chain_length)))).values[1]
        psi = neel_state(chain_length)
        dmrg_run = dmrg_cbe1s!(
            psi,
            hamiltonian;
            n_sweeps = 20,
            etol = 1e-11,
            tol_basis = 1e-6,
            tol_trunc = 1e-10,
            maxdim = 64,
        )
        @test dmrg_run.energies[end] ≈ exact_energy atol = 1e-8
        @test dmrg_run.energies[end] >= exact_energy - 1e-9                     # variational
        @test maximum(dmrg_run.max_bond_dims) > 1                               # CBE grew the rank from Neel
        @test length(dmrg_run.ledgers[end].bond_log) == 2 * (chain_length - 1)  # one entry per bond visit
    end

    @testset "QTT TDVP2 at full rank is the exact flow" begin
        set_symmetry!(:none)
        clear_fusion_cache!()
        g = qtt_grid(3, 3; x = (-5.0, 5.0), y = (-5.0, 5.0))
        m = gp_model(g; gnl = 0.0, gamma = 0.1, mu = 0.0, omega = 1.0)
        # complete bonds [2, 4, 8, 4, 2]: the tangent-space projector is the identity
        initial = _random_tt(_tt_ranks(6, 64); rng = Random.MersenneTwister(5))
        psi = qtt_mps(initial)
        flow = GPFlow(m)
        for k = 0:3
            qtt_tdvp2_step!(psi, flow, 0.05k, 0.05; maxdim = 64, tol_trunc = 1e-14)
        end
        exact = exp(0.2 * (-1im - m.gamma) * op_dense(m.H0)) * vec(tt_dense(initial))
        @test _relerr(vec(tt_dense(qtt_cores(psi))), exact) < 1e-8
    end
end
