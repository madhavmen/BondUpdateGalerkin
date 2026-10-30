# MPO storage: an on-site sum and the exact MPO-MPS product, with and without U(1) symmetry. The
# reference is total S^z, measured by the observables, so no basis order is assumed.

using Test, LinearAlgebra, LurCGT, Telum
import BondUpdateGalerkin
using BondUpdateGalerkin.BUG, BondUpdateGalerkin.QMB

include(joinpath(@__DIR__, "..", "..", "..", "common", "dense_reference.jl"))

@testset "onsite_mpo and mpo_product: $sym" for sym in (:U1, :none)
    set_symmetry!(sym)
    clear_fusion_cache!()
    q = local_space()
    L = 5
    H = onsite_mpo(L, [q.Sz for _ = 1:L]; Iloc = q.I)
    @test length(H) == L
    @test mpo_virtual_dims(H) == [1, 2, 2, 2, 2, 1]
    # a product state is an eigenvector of total S^z
    for spins in ([:up, :up, :down, :up, :down], [:down, :down, :down, :up, :down])
        psi = product_state(spins)
        Hpsi = mpo_product(H, psi)
        @test bond_dimensions(Hpsi) == fill(2, L - 1)          # product of the two bonds
        @test norm(dense_state(Hpsi) - total_sz(psi) * dense_state(psi)) < 1e-12
        @test inner_product(psi, Hpsi) ≈ total_sz(psi) atol = 1e-12
    end
    @test_throws DimensionMismatch mpo_product(H, product_state([:up, :down, :up]))
    @test_throws DimensionMismatch onsite_mpo(L, [q.Sz for _ = 1:3]; Iloc = q.I)
    @test_throws ArgumentError onsite_mpo(1, [q.Sz]; Iloc = q.I)
end

set_symmetry!(:U1)
