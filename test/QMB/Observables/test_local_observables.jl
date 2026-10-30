# Moved from BUG-Julia@b155550:tests/BondUpdateBUG/test_symmetric_mps.jl (observable testsets).
# Not yet moved: the dimer-state case (dimer_state moves with QMB/Tooling/gates.jl) and the
# sz_overlaps testset, which evolves its states with xxz_mpo and tdvp2_step!.

using Test, LinearAlgebra, LurCGT, Telum
using BondUpdateGalerkin.BUG, BondUpdateGalerkin.QMB

@testset "spin-1/2 local observables" begin

    @testset "observables read the domain wall correctly" begin
        psi = domain_wall_state(6)               # up up up down down down
        @test isapprox(total_sz(psi), 0.0; atol = 1e-14)
        @test isapprox(sz_expectation(psi, 1), +0.5; atol = 1e-14)
        @test isapprox(sz_expectation(psi, 3), +0.5; atol = 1e-14)
        @test isapprox(sz_expectation(psi, 4), -0.5; atol = 1e-14)
        @test isapprox(sz_expectation(psi, 6), -0.5; atol = 1e-14)
    end

    @testset "observables read the Neel state correctly" begin
        psi = neel_state(6)
        @test isapprox(total_sz(psi), 0.0; atol = 1e-14)
        for j = 1:6
            @test isapprox(sz_expectation(psi, j), isodd(j) ? +0.5 : -0.5; atol = 1e-14)
        end
    end

    @testset "observables survive a canonicalisation round trip" begin
        psi = neel_state(6)
        canonical!(psi, 1)
        canonical!(psi, 6)
        canonical!(psi, 3)
        @test isapprox(norm(psi), 1.0; atol = 1e-13)
        for j = 1:6
            @test isapprox(sz_expectation(psi, j), isodd(j) ? +0.5 : -0.5; atol = 1e-12)
        end
    end

    @testset "odd length and non-zero total Sz" begin
        psi = product_state([:up, :up, :down])
        @test length(psi) == 3
        @test isapprox(norm(psi), 1.0; atol = 1e-14)
        @test isapprox(total_sz(psi), +0.5; atol = 1e-13)
    end

    # `apply_sz!` PRODUCES A NEW STATE, which is a strictly harder job than measuring one, and it
    # lived unexercised in a benchmark until `hs_structure_factor.jl` needed it too. Its docstring
    # warns that the obvious implementation -- contracting `local_space().Sz` onto the site leg --
    # returns a correctly SHAPED tensor whose site sectors have been re-sorted, i.e. a wrong state
    # that measures right. These checks are chosen to fail on exactly that.
    @testset "apply_sz! builds S^z_j|psi>, not merely something of the right shape" begin
        for (name, psi0) in (
            ("neel", neel_state(6)),
            ("domain wall", domain_wall_state(6)),
            # ⛔ ASYMMETRIC on purpose: Neel, the dimer and the domain wall are all
            # invariant under reversal-composed-with-flip, so a convention error
            # that reverses the chain passes on every one of them.
            ("asymmetric product", product_state([:up, :up, :down, :up, :down, :down])),
        )
            @testset "$name" begin
                for j in (1, 3, 6)
                    psi = deepcopy(psi0)
                    phi = deepcopy(psi0)
                    apply_sz!(phi, j)

                    # (S^z)^2 = I/4 for spin-1/2, so this is EXACT for ANY normalised state and
                    # any site -- no reference state needed, and it pins the overall scale that a
                    # sector mix-up would leave intact.
                    @test isapprox(norm(phi)^2, 0.25; atol = 1e-12)

                    # THE ONE THAT CATCHES THE RE-SORT. `site_expval` closes the loop with the
                    # same tensor, so the re-sort cancels there and it stays right; routing the
                    # new state through `overlap` against the ORIGINAL does not cancel, so the
                    # orthogonal component the wrong route introduces shows up here as a mismatch.
                    @test isapprox(real(overlap(psi, phi)), sz_expectation(psi0, j); atol = 1e-12)
                    @test isapprox(imag(overlap(psi, phi)), 0.0; atol = 1e-12)
                end
            end
        end

        # On a product state with a definite spin at `j`, S^z_j acts as a pure scalar, so the
        # result must be PARALLEL to the input -- |<psi|phi>| = ||phi|| exactly. Any leaked
        # orthogonal component breaks this even when the norm above is still 1/4.
        psi0 = product_state([:up, :up, :down, :up, :down, :down])
        for (j, s) in ((1, +0.5), (3, -0.5), (4, +0.5))
            phi = deepcopy(psi0)
            apply_sz!(phi, j)
            @test isapprox(overlap(psi0, phi), s; atol = 1e-12)
            @test isapprox(abs(overlap(psi0, phi)), norm(phi); atol = 1e-12)
        end

        # S^z_j commutes with the total charge, so the result stays in the SAME U(1) sector.
        # `total_sz` here is a ratio because `apply_sz!` deliberately does not renormalise.
        psi = neel_state(6)
        phi = deepcopy(psi)
        apply_sz!(phi, 3)
        @test isapprox(total_sz(phi) / norm(phi)^2, total_sz(psi); atol = 1e-11)

        # Idempotence up to the scalar: applying twice is (S^z)^2 = 1/4, i.e. the state comes back
        # parallel to the original with norm 1/4.
        phi2 = deepcopy(psi)
        apply_sz!(phi2, 2)
        apply_sz!(phi2, 2)
        @test isapprox(norm(phi2), 0.25; atol = 1e-12)
        @test isapprox(abs(overlap(psi, phi2)), 0.25; atol = 1e-12)
    end
end
