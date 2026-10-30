# Bit orderings and grids: every bit pattern round-trips exactly, in every layout, for one, two and
# three axes (implementation plan WP2: "ordering is round-trip exact for indexing").

using Test
using BondUpdateGalerkin.QTTS

const LAYOUTS = ((:grouped, :lsb_first), (:grouped, :msb_first), (:interleaved, :lsb_first),
    (:interleaved, :msb_first))

@testset "QTTOrdering round trips ($o, $e)" for (o, e) in LAYOUTS
    for bits in ((3,), (5,), (2, 2), (3, 2), (2, 4), (4, 4), (1, 2, 1), (2, 1, 3))
        ord = QTTOrdering(bits; order = o, endian = e)
        L = qtt_length(ord)
        @test L == sum(bits)
        @test sort(reduce(vcat, ord.sites)) == collect(1:L)
        seen = Set{NTuple{length(bits),Int}}()
        for idx in Iterators.product((0:(2^n-1) for n in bits)...)
            b = qtt_bits(ord, idx...)
            @test all(in((0, 1)), b)
            @test qtt_point(ord, b) == idx
            push!(seen, idx)
        end
        @test length(seen) == 2^L
        p = qtt_site_permutation(ord)
        @test sort(p) == collect(1:2^L)
        # site 1 fastest on the configuration, axis 1 fastest on the grid index
        for c in (0, 1, 2^L - 1, 2^L ÷ 3)
            b = [(c >> (s - 1)) & 1 for s = 1:L]
            idx = qtt_point(ord, b)
            stride = 1
            g = 1
            for a in eachindex(bits)
                g += idx[a] * stride
                stride *= 2^bits[a]
            end
            @test p[c+1] == g
        end
    end
end

@testset "QTTOrdering layouts written out" begin
    # grouped LSB-first is the identity map: x bits first, then y, least significant first
    g = QTTOrdering((3, 2))
    @test g.axes == (:x, :y)
    @test qtt_sites(g, :x) == [1, 2, 3] && qtt_sites(g, :y) == [4, 5]
    @test qtt_site_permutation(g) == collect(1:32)
    @test qtt_site_permutation(QTTOrdering((1, 2, 1))) == collect(1:16)
    # grouped MSB-first reverses each axis
    m = QTTOrdering((2, 2); endian = :msb_first)
    @test qtt_sites(m, :x) == [2, 1] && qtt_sites(m, :y) == [4, 3]
    # interleaved alternates until the shorter axis runs out
    i = QTTOrdering((2, 3); order = :interleaved)
    @test qtt_sites(i, :x) == [1, 3] && qtt_sites(i, :y) == [2, 4, 5]
    @test qtt_sites(i, 2) == qtt_sites(i, :y)
    @test qtt_bits(i, 2, 5) == [0, 1, 1, 0, 1]
    # named axes
    a = QTTOrdering((2, 2); axes = (:s, :v))
    @test qtt_sites(a, :v) == [3, 4]
    @test QTTOrdering((2, 2, 2)).axes == (:x, :y, :z)
end

@testset "QTTOrdering rejects bad input" begin
    @test_throws ArgumentError QTTOrdering((2, 2); order = :snake)
    @test_throws ArgumentError QTTOrdering((2, 2); endian = :middle)
    @test_throws ArgumentError QTTOrdering((2, 0))
    @test_throws ArgumentError QTTOrdering((2, 2); axes = (:x, :x))
    @test_throws ArgumentError qtt_sites(QTTOrdering((2, 2)), :z)
    @test_throws BoundsError qtt_bits(QTTOrdering((2, 2)), 4, 0)
    @test_throws DimensionMismatch qtt_point(QTTOrdering((2, 2)), [0, 1, 0])
end

@testset "QTTGrid geometry" begin
    g = qtt_grid(3, 2; x = (-1.0, 1.0), y = (0.0, 2.0))
    xs, ys = qtt_coords(g)
    @test length(xs) == 8 && length(ys) == 4
    @test g.dx ≈ 2 / 8 && g.dy ≈ 2 / 4                      # periodic: right end excluded
    @test xs[1] == -1.0 && xs[end] ≈ 1.0 - g.dx
    @test qtt_cell(g) ≈ g.dx * g.dy
    h = qtt_grid(3, 2; x = (-1.0, 1.0), y = (0.0, 2.0), periodic_x = false, periodic_y = false)
    hx, hy = qtt_coords(h)
    @test hx[end] ≈ 1.0 && hy[end] ≈ 2.0                     # both end points on the grid
    @test qtt_length(g) == 5
    gi = qtt_grid(2, 3; order = :interleaved, endian = :msb_first)
    @test gi.ordering.sites == QTTOrdering((2, 3); order = :interleaved, endian = :msb_first).sites
    @test qtt_point(gi, qtt_bits(gi, 3, 6)) == (3, 6)
    @test_throws ArgumentError qtt_grid(0, 2)
    @test_throws ArgumentError qtt_grid(2, 2; x = (1.0, -1.0))
end
