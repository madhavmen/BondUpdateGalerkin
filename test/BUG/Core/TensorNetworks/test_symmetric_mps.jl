# Moved from BUG-Julia@b155550:tests/BondUpdateBUG/test_symmetric_mps.jl (container and canonical
# form; its observable testsets are in test/QMB/Observables/test_local_observables.jl).

using Test, LinearAlgebra, LurCGT, Telum
using BondUpdateGalerkin.BUG, BondUpdateGalerkin.QMB

@testset "SymMPS" begin

    @testset "domain wall is normalised and product" begin
        psi = domain_wall_state(6)
        @test length(psi) == 6
        @test isapprox(norm(psi), 1.0; atol = 1e-14)
        @test all(bond_dims(psi) .== 1)          # product state
    end

    @testset "site tensors have the documented leg layout" begin
        psi = domain_wall_state(6)
        for i in 1:6
            A = psi[i]
            @test length(A.inds) == 3
            @test A.inds[1].itags == "L,$i"
            @test A.inds[2].itags == "S,$i"
            @test A.inds[3].itags == "L,$(i + 1)"
            # the physical leg keeps the full local space, so local operators
            # stay contractible against it
            @test leg_dim(A, 2) == 2
        end
        # bonds carry opposite arrows -- the invariant canonical! maintains
        for i in 1:5
            @test psi[i].inds[3].dir != psi[i + 1].inds[1].dir
        end
    end

    @testset "canonical! preserves norm and moves the centre" begin
        psi = domain_wall_state(6)
        n0 = norm(psi)
        canonical!(psi, 3)
        @test psi.center == 3
        @test isapprox(norm(psi), n0; atol = 1e-13)
        canonical!(psi, 6)                       # sweep back
        @test psi.center == 6
        @test isapprox(norm(psi), n0; atol = 1e-13)
    end

    @testset "canonical! keeps the bond arrow invariant" begin
        psi = neel_state(6)
        canonical!(psi, 2)                       # forces left moves
        for i in 1:5
            @test psi[i].inds[3].dir != psi[i + 1].inds[1].dir
            @test psi[i].inds[3].itags == psi[i + 1].inds[1].itags
        end
    end

    @testset "canonical! makes every tensor left of the centre a left isometry" begin
        psi = domain_wall_state(6)
        canonical!(psi, 4)
        for i in 1:3
            gram = left_gram(psi[i])             # NOT psi[i]' * psi[i] -- see below
            @test isapprox(norm(gram - 1), 0.0; atol = 1e-12)
        end
    end

    @testset "canonical! makes every tensor right of the centre a right isometry" begin
        psi = neel_state(6)
        canonical!(psi, 2)
        for i in 3:6
            gram = right_gram(psi[i])
            @test isapprox(norm(gram - 1), 0.0; atol = 1e-12)
        end
    end

    @testset "`*` is greedy -- the reason left_gram primes the bond" begin
        psi = domain_wall_state(4)
        A = psi[1]
        @test length((A' * A).inds) == 0         # bond closed, rank-0 scalar
        @test length(left_gram(A).inds) == 2     # bond kept open
    end
end
