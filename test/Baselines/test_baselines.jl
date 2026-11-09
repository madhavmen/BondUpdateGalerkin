# The comparison methods: each baseline against an independent reference on the same problem the
# BUG tests use -- TDVP2 on the XX chain against its frozen run and free fermions, and a QTT TDVP2
# step against the dense exponential.

using Test, LinearAlgebra
import JSON, Random
using BondUpdateGalerkin.BUG, BondUpdateGalerkin.QMB, BondUpdateGalerkin.QTTS
using BondUpdateGalerkin.Baselines

include(joinpath(@__DIR__, "..", "common", "dense_reference.jl"))
include(joinpath(@__DIR__, "..", "common", "free_fermion.jl"))
include(joinpath(@__DIR__, "..", "common", "qtt_fixtures.jl"))

const FIXTURE_DIR = joinpath(@__DIR__, "..", "fixtures", "b155550")
# the first rows of the 20-step frozen TDVP2 run (each row is checked on its own), to keep the
# suite near five minutes
const COMPARED_STEPS = 3

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
        for (step, frozen_row) in enumerate(first(frozen["trajectory"], COMPARED_STEPS))
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
