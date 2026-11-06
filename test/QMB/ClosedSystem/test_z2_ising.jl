# The Z2 Ising local space, the states and the analytic energy; test_tfim.jl pins the Hamiltonian
# against a dense reference.

using Test, LinearAlgebra, Telum
using BondUpdateGalerkin.BUG, BondUpdateGalerkin.QMB

# Z2 symmetry for the transverse-field Ising model.
#
#     H = -J Σ Z_i Z_{i+1} - h Σ X_i
#
# The Z2 local space is built BELOW Telum's IROP layer (see `ising_local_space` in QMB/Tooling/local_spaces.jl:
# `getweight_irop` derives an operator's charge from a Lie-algebra commutator, and the Z2 flip
# admits no such charge because +1 and -1 are the same mod 2). That means this file rests on
# Telum-PRIVATE API, so these tests are the tripwire: if a Telum upgrade changes the private
# contract, this must fail loudly rather than quietly producing a wrong Hamiltonian.
#
# The tests are therefore on PHYSICS that an incorrect charge assignment could not fake:
# the Pauli algebra and an analytic energy.

@testset "Z2 transverse-field Ising" begin

    @testset "the local space is a genuine Z2 space" begin
        set_symmetry!(:Z2)
        q = ising_local_space()
        @test Set(propertynames(q)) == Set((:I, :X, :Z))
        # Two charge sectors, dim 1 each -- the Z2 parity of sigma^x.
        @test length(q.X.spaces[1]) == 2
        # THE FLIP CARRIES AN OP LEG, and that is the whole reason Z2 is representable here:
        # both matrix elements share ONE op-leg charge because 1 = 0+1 and 0 = 1+1 mod 2.
        # Under U(1) this would need charges +1 and -1 and would not be one operator at all.
        @test length(q.Z.inds) == 3
        @test length(q.Z.spaces[3]) == 1
        # `local_space` is the HEISENBERG space and must refuse :Z2 rather than return something
        # spin-like -- the TFIM has no Sp/Sz/Sm.
        @test_throws ArgumentError local_space(:Z2)
    end

    @testset "states, and the sector they live in" begin
        for sym in (:Z2, :none)
            set_symmetry!(sym)
            for L in (4, 8)
                p = ising_polarised_state(L)
                @test norm(p) ≈ 1.0 atol = 1e-12
                @test bond_dims(p) == ones(Int, L - 1)      # a genuine product state
                @test x_profile(p) ≈ ones(L) atol = 1e-12   # every spin along +x
                k = ising_kink_state(L)
                @test x_profile(k) ≈ [i <= L ÷ 2 ? 1.0 : -1.0 for i = 1:L] atol = 1e-12
            end
        end
    end

    @testset "the analytic energy of the polarised state" begin
        # |+...+> has <Z_i Z_j> = 0 (Z is off-diagonal in this basis) and <X_j> = 1, so
        # E = -h*L exactly, for any J. A wrong ZZ charge assignment would break this.
        for sym in (:Z2, :none), L in (4, 8), h in (0.5, 1.0, 2.0)
            set_symmetry!(sym)
            psi = ising_polarised_state(L)
            g = ising_bond_gates(psi; J = 1.0, h = h)
            @test energy(psi, g) ≈ -h * L atol = 1e-10
        end
    end
end
