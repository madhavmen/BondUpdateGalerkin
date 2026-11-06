# QTT fields: sampled functions through every ordering, separable embedding, and the round trip
# through the BUG state type in several gauges.
#
# The periodic-shift wrap is checked in test_operators.jl with an independently built shift.

using Test, LinearAlgebra
using BondUpdateGalerkin.BUG, BondUpdateGalerkin.QTTS

include(joinpath(@__DIR__, "..", "..", "common", "qtt_fixtures.jl"))

set_symmetry!(:none)
clear_fusion_cache!()

const ORDERS = ((:grouped, :lsb_first), (:grouped, :msb_first), (:interleaved, :lsb_first))

@testset "fields, round trips, weights ($o, $e)" for (o, e) in ORDERS
    for (nx, ny) in ((2, 2), (3, 2), (3, 4), (4, 4))
        g = qtt_grid(nx, ny; x = (-1.0, 1.0), y = (0.0, 2.0), order = o, endian = e)
        L = qtt_length(g)
        # field -> QTT -> grid values, and single-point evaluation
        f(x, y) = exp(-x^2) * cos(3y) + 0.3im * sin(x * y) + 0.1x
        C = qtt_function(g, f)
        xs, ys = qtt_coords(g)
        M = [f(x, y) for x in xs, y in ys]
        @test _relerr(qtt_values(C, g), M) < 1e-12
        @test abs(qtt_value(C, g, 2^nx - 1, 1) - M[end, 2]) < 1e-12
        # separable construction == full TT-SVD
        S = qtt_separable(g, x -> exp(-x^2), y -> cos(y))
        @test _relerr(qtt_values(S, g), [exp(-x^2) * cos(y) for x in xs, y in ys]) < 1e-12
        @test maximum(tt_ranks(S)[g.nx:g.nx]; init = 1) == 1 || o !== :grouped
        # Telum round trip in several gauges; the physical L2 norm keeps the weight outside
        for c in (1, L ÷ 2 + 1, L)
            psi = qtt_mps(C; center = c)
            @test orthogonality_center(psi) == c
            @test _relerr(tt_dense(qtt_cores(psi)), tt_dense(C)) < 1e-12
            @test abs(qtt_l2norm(psi, g) - sqrt(qtt_cell(g)) * norm(M)) < 1e-10 * norm(M)
            @test abs(norm(psi) - norm(M)) < 1e-10 * norm(M)
        end
        psi = qtt_mps(C; center = 1)
        canonical!(psi, L)
        @test all(left_isometry_defect(psi[k]) < 1e-12 for k = 1:(L-1))
        canonical!(psi, 1)
        @test all(right_isometry_defect(psi[k]) < 1e-12 for k = 2:L)
        @test symmetry(psi) isa NoSymmetry
        @test bond_dimensions(psi) == tt_ranks(qtt_cores(psi))
    end
end

@testset "TT basics" begin
    C = _random_tt(_tt_ranks(7, 5); rng = Random.MersenneTwister(3))
    A = tt_dense(C)
    @test _relerr(tt_dense(tt_from_dense(A)), A) < 1e-12
    D, disc = tt_compress(C; tol = 0.0)
    @test _relerr(tt_dense(D), A) < 1e-12 && disc < 1e-12
    @test tt_dot(C, C) ≈ dot(vec(A), vec(A))
    @test tt_norm(C) ≈ norm(A)
    B = _random_tt(_tt_ranks(7, 3); rng = Random.MersenneTwister(4))
    @test tt_dot(B, C) ≈ dot(vec(tt_dense(B)), vec(A))
    @test _relerr(tt_dense(tt_add(B, C)), tt_dense(B) + A) < 1e-12
    @test _relerr(tt_dense(tt_lincomb([(2.0, B), (-1im, C)])), 2 * tt_dense(B) - 1im * A) < 1e-12
    @test _relerr(tt_dense(tt_scale(C, 0.5)), 0.5 * A) < 1e-14
    @test _relerr(tt_dense(tt_conj(C)), conj(A)) < 1e-14
    bits = [1, 0, 1, 1, 0, 0, 1]
    @test tt_evaluate(C, bits) ≈ A[(bits .+ 1)...]
    # a truncated compression reports a bound that holds
    T, bound = tt_compress(C; tol = 1e-2)
    @test _relerr(tt_dense(T), A) <= bound + 1e-12
    @test maximum(tt_ranks(T)) <= maximum(tt_ranks(C))
end

@testset "1D and 3D embedding ($o, $e)" for (o, e) in ORDERS
    # one axis: the logical chain lands on the ordering's sites
    n = 5
    ord1 = QTTOrdering((n,); order = o, endian = e)
    vals = [sin(0.3k) + 0.1k^2 for k = 0:(2^n-1)]
    E = qtt_embed(ord1; x = tt_from_samples(vals))
    @test _relerr(_grid_vector(ord1, tt_dense(E)), vals) < 1e-12
    # three axes, separable: the outer product on the grid index (axis 1 fastest)
    ord3 = QTTOrdering((2, 1, 3); order = o, endian = e)
    fx, fy, fz = [1.0, 2.0, -1.0, 0.5], [0.3, -2.0], collect(1.0:8.0) .^ 2
    E3 = qtt_embed(ord3; x = tt_from_samples(fx), y = tt_from_samples(fy), z = tt_from_samples(fz))
    ref = vec([a * b * c for a in fx, b in fy, c in fz])
    @test _relerr(_grid_vector(ord3, tt_dense(E3)), ref) < 1e-12
    # a missing axis is the constant 1
    E2 = qtt_embed(ord3; y = tt_from_samples(fy))
    @test _relerr(_grid_vector(ord3, tt_dense(E2)), vec([b for a in fx, b in fy, c in fz])) < 1e-12
    @test_throws ArgumentError qtt_embed(ord3; w = tt_from_samples(fy))
    @test_throws ArgumentError qtt_embed(ord3)
    @test_throws DimensionMismatch qtt_embed(ord3; x = tt_from_samples(fz))
end

@testset "adapter refuses a symmetric mode" begin
    set_symmetry!(:U1)
    @test_throws ArgumentError qtt_mps(_random_tt(_tt_ranks(4, 2)))
    set_symmetry!(:none)
end
