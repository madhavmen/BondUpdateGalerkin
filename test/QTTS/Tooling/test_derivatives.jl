# Derivative MPOs and 2D operator assembly against dense stencils.
#
# The periodic-shift wrap, the stencil, coordinate and mixed-derivative checks, plus a second-order convergence check of the
# one-sided boundary stencils on a smooth function.

using Test, LinearAlgebra
using BondUpdateGalerkin.BUG, BondUpdateGalerkin.QTTS

include(joinpath(@__DIR__, "..", "..", "common", "qtt_fixtures.jl"))

set_symmetry!(:none)
clear_fusion_cache!()

const DERIVATIVE_LAYOUTS = ((:grouped, :lsb_first), (:interleaved, :msb_first))

@testset "QTT derivatives and operator assembly" begin
    @testset "1D chains are the dense stencils" begin
        for nbits in (2, 4), (order, bc) in Iterators.product((1, 2), QTT_BCS)
            # the one-sided second derivative needs four points
            bc === :onesided && 2^nbits < 4 && continue
            spacing = 0.3
            chain_matrix = op_dense(qtt_derivative(nbits, spacing; order = order, bc = bc))
            @test _relerr(chain_matrix, _ref_d(2^nbits, spacing, order, bc)) < 1e-12
        end
        @test_throws ArgumentError qtt_derivative(1, 0.1; order = 2, bc = :onesided)
        @test_throws ArgumentError qtt_derivative(3, 0.1; bc = :neumann)
        @test_throws ArgumentError qtt_derivative(3, 0.1; order = 3)
    end

    @testset "periodic shift wraps on grid values ($order, $endian)" for (order, endian) in
                                                                         DERIVATIVE_LAYOUTS
        g = qtt_grid(3, 3; order = order, endian = endian)
        field = qtt_function(g, (x, y) -> x + 10y + 0.5x * y)
        shift_x = qtt_embed(g; x = qtt_shift(3; bc = :periodic))
        shifted = qtt_values(first(tt_compress(op_apply(shift_x, field))), g)
        @test _relerr(shifted, circshift(qtt_values(field, g), (-1, 0))) < 1e-12
    end

    @testset "2D derivatives, coordinates and the mixed term ($order, $endian)" for (
        order,
        endian,
    ) in DERIVATIVE_LAYOUTS
        nbits = 3
        npoints = 2^nbits
        g = qtt_grid(nbits, nbits; x = (-2.0, 2.0), y = (-2.0, 2.0), order = order, endian = endian)
        xs, _ = qtt_coords(g)
        for (derivative_order, bc) in
            ((1, :periodic), (2, :periodic), (1, :dirichlet0), (2, :onesided))
            reference = _ref_d(npoints, g.dx, derivative_order, bc)
            along_x = qtt_embed(g; x = qtt_d(g, :x, derivative_order; bc = bc))
            along_y = qtt_embed(g; y = qtt_d(g, :y, derivative_order; bc = bc))
            @test _relerr(_grid_matrix(g, op_dense(along_x)), _ref2(g, reference, nothing)) < 1e-12
            @test _relerr(_grid_matrix(g, op_dense(along_y)), _ref2(g, nothing, reference)) < 1e-12
        end
        coordinate_x = qtt_embed(g; x = qtt_coord_diag(g, :x))
        @test _relerr(
            _grid_matrix(g, op_dense(coordinate_x)),
            _ref2(g, diagm(collect(xs)), nothing),
        ) < 1e-12
        # d_xy as one Kronecker term, and a two-term sum assembled by qtt_op
        first_periodic = _ref_d(npoints, g.dx, 1, :periodic)
        mixed =
            qtt_embed(g; x = qtt_d(g, :x, 1; bc = :periodic), y = qtt_d(g, :y, 1; bc = :periodic))
        @test _relerr(_grid_matrix(g, op_dense(mixed)), _ref2(g, first_periodic, first_periodic)) <
              1e-12
        laplacian = qtt_op(
            g,
            [
                (1.0, qtt_d(g, :x, 2; bc = :periodic), nothing),
                (1.0, nothing, qtt_d(g, :y, 2; bc = :periodic)),
            ],
        )
        second_periodic = _ref_d(npoints, g.dx, 2, :periodic)
        @test _relerr(
            _grid_matrix(g, op_dense(laplacian)),
            _ref2(g, second_periodic, nothing) + _ref2(g, nothing, second_periodic),
        ) < 1e-12
    end

    @testset "one-sided stencils converge at second order, boundaries included" begin
        # sin on [0, 1] with both end points on the grid; the error of each derivative is measured
        # in the max norm over every point, so a first-order boundary row would show as rate 1
        max_errors = Dict{Int,Vector{Float64}}(1 => Float64[], 2 => Float64[])
        for nbits in (5, 6)
            npoints = 2^nbits
            spacing = 1 / (npoints - 1)
            points = spacing .* (0:(npoints-1))
            samples = sin.(points)
            for (derivative_order, exact) in ((1, cos.(points)), (2, -sin.(points)))
                chain = qtt_derivative(nbits, spacing; order = derivative_order, bc = :onesided)
                approximation = op_dense(chain) * samples
                push!(max_errors[derivative_order], maximum(abs.(approximation - exact)))
            end
        end
        # halving h (31 -> 63 intervals) divides a second-order error by ~4.1
        for derivative_order in (1, 2)
            ratio = max_errors[derivative_order][2] / max_errors[derivative_order][1]
            @test 0.15 < ratio < 0.35
        end
    end
end
