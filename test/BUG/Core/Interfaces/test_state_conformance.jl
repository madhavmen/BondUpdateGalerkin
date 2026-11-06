# State-contract conformance (docs/dev/interfaces.md). The same checks run
# against every state type and symmetry; a new state type is added to `conformance_states` and
# must pass unchanged. A QTT field state is a no-symmetry `SymMPS`, so it runs under `:none`.

using Test, LinearAlgebra, LurCGT, Telum
import BondUpdateGalerkin
using BondUpdateGalerkin.BUG, BondUpdateGalerkin.QMB, BondUpdateGalerkin.QTTS

include(joinpath(@__DIR__, "..", "..", "..", "common", "dense_reference.jl"))
include(joinpath(@__DIR__, "..", "..", "..", "common", "qtt_fixtures.jl"))

"Direct sum of two states along the links: an entangled state with multi-sector bonds."
function mps_sum(a::SymMPS, b::SymMPS)
    L = length(a)
    ts = Any[]
    for i = 1:L
        dims = i == 1 ? (3,) : (i == L ? (1,) : (1, 3))
        push!(ts, TLArray(oplus(a[i], b[i], dims)))
    end
    return canonical!(canonical!(SymMPS(ts, L), 1), L)
end

"Named states of the current symmetry mode: product, and entangled with bond dimension > 1."
function conformance_states()
    states = [
        "neel" => neel_state(6),
        "odd-length product" => product_state([:up, :up, :down, :up, :down]),
        "sum of two products" => mps_sum(
            product_state([:up, :down, :up, :down, :up, :down]),
            product_state([:down, :up, :up, :down, :down, :up]),
        ),
    ]
    symmetry_mode() === :none &&
        push!(states, "qtt field" => qtt_mps(_random_tt(_tt_ranks(6, 3)); center = 2))
    return states
end

const TRAITS = Dict(:U1 => U1Symmetry, :none => NoSymmetry)

@testset "state contract: $sym" for sym in (:U1, :none)
    set_symmetry!(sym)
    clear_fusion_cache!()
    for (name, psi0) in conformance_states()
        @testset "$name" begin
            psi = copy(psi0)
            L = nsites(psi)
            @test L == length(psi)
            @test symmetry(psi) isa TRAITS[sym]
            @test all(physical_dimension(psi, i) == 2 for i = 1:L)
            bd = bond_dimensions(psi)
            @test length(bd) == L - 1
            @test bd == [bond_dimension(psi, b) for b = 1:(L-1)]
            name in ("sum of two products", "qtt field") && @test maximum(bd) > 1   # non-vacuous

            v = dense_state(psi)
            n2 = real(inner_product(psi, psi))
            @test n2 ≈ real(dot(v, v)) rtol = 1e-12
            @test inner_product(psi, psi) ≈ overlap(psi, psi)

            # canonicalize! moves the centre, keeps the state, and leaves isometries behind it
            for c in (1, L, (L + 1) ÷ 2)
                canonicalize!(psi, c)
                @test orthogonality_center(psi) == c
                @test norm(dense_state(psi) - v) <= 1e-12 * norm(v)
                @test all(left_isometry_defect(psi[i]) < 1e-12 for i = 1:(c-1))
                @test all(right_isometry_defect(psi[i]) < 1e-12 for i = (c+1):L)
                @test norm(psi)^2 ≈ n2 rtol = 1e-12
            end

            # copy: changing the copy's gauge leaves the original's tensors and centre alone
            a = copy(psi0)
            tensors, centre = copy(a.tensors), orthogonality_center(a)
            b = copy(a)
            canonicalize!(b, centre == 1 ? L : 1)
            @test orthogonality_center(a) == centre
            @test all(a[i] === tensors[i] for i = 1:L)
            @test inner_product(a, b) ≈ n2 rtol = 1e-12
        end
    end

    @testset "inner product against dense vectors" begin
        a = neel_state(6)
        b = conformance_states()[3].second
        @test inner_product(a, b) ≈ dot(dense_state(a), dense_state(b)) atol = 1e-12
        @test inner_product(b, a) ≈ conj(inner_product(a, b)) atol = 1e-12
    end
end

@testset "U(1): different total charge gives an exact zero" begin
    set_symmetry!(:U1)
    a = neel_state(6)
    b = product_state([:up, :up, :up, :up, :down, :down])
    @test inner_product(a, b) === 0.0 + 0.0im
end

@testset "symmetry_trait" begin
    @test BondUpdateGalerkin.BUG.symmetry_trait(()) isa NoSymmetry
    @test BondUpdateGalerkin.BUG.symmetry_trait((U1,)) isa U1Symmetry
    @test BondUpdateGalerkin.BUG.symmetry_trait((Z{2},)) isa Z2Symmetry
    @test BondUpdateGalerkin.BUG.symmetry_trait((SU{2},)) isa SU2Symmetry
    @test_throws ArgumentError BondUpdateGalerkin.BUG.symmetry_trait((U1, U1))
end

set_symmetry!(:U1)
