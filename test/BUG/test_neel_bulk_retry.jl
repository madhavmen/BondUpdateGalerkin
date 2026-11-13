# Regression checks for opt-in Neel bulk caps and transactional failure.
# julia --project test/BUG/test_neel_bulk_retry.jl
using Test, LinearAlgebra, Random
using BondUpdateGalerkin.BUG, BondUpdateGalerkin.QMB
set_symmetry!(:U1)
@testset "Neel bulk cap and transactional rollback" begin
    @test bond_work_ceilings(ones(Int, 5), 2, 128; factor=1.2) == fill(3,5)
    @test bulk_work_ceilings(ones(Int, 5), 2, 128) == fill(8,5)
    @test bulk_work_ceilings([128], 2, 128) == [160]
    @test all(bulk_work_ceilings([2,4,16],2,128) .>=
              bond_work_ceilings([2,4,16],2,128;factor=1.2))
    psi = neel_state(6)
    before = deepcopy(psi)
    mpo = xxz_mpo(6; J=1.0, delta=0.0)
    result = cbe_bug_bulk_retry_step!(
        psi, mpo, ComplexF64(-0.1im);
        modes=(:bond,),
        order=2, closure_checks=true,
        tol_basis=1e-12, tol_trunc=1e-14,
        tol_root=1e-12, maxdim=1,
        expansion=DDQRExpansion(mode=:fixed, fixed_width=8,
                                fixed_rank=4, krylov_mode=:lanczos),
        rng=MersenneTwister(3),
    )
    @test !result.accepted
    @test length(result.history)==1
    @test result.history[1].cap == :bond
    @test bond_dims(psi)==bond_dims(before)
    @test abs(overlap(before,psi)-overlap(before,before)) < 1e-12
end

@testset "Neel: failed bond cap is retried and discarded before bulk success" begin
    psi = neel_state(6)
    reference = deepcopy(psi)
    mpo = xxz_mpo(6; J=1.0, delta=0.0)
    policy = DDQRExpansion(mode=:rank_tail, fixed_width=8,
                           tail_factor=0.1, krylov_mode=:lanczos)
    result = cbe_bug_bulk_retry_step!(
        psi, mpo, ComplexF64(-0.1im);
        modes=(:bond,:bulk),
        order=2, closure_checks=true,
        tol_basis=1e-2, tol_trunc=1e-5, tol_root=1e-10,
        maxdim=64, expansion=policy,
        rng=MersenneTwister(71),
    )
    @test result.accepted
    @test length(result.history) == 2
    @test result.history[1].cap == :bond
    @test !result.history[1].accepted
    @test result.history[2].cap == :bulk
    @test result.history[2].accepted
    @test result.h1 == sum(x.h1 for x in result.history)
    @test result.stream_columns == sum(x.stream_columns for x in result.history)
    @test norm(psi) > 0
    @test abs(overlap(reference,psi)) < abs(overlap(reference,reference)) + 1e-10
end
println("NEEL_BULK_RETRY_TEST_FINISHED")
