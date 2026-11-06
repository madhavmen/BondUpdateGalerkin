# Heston 2D on QTT: the generator against a dense
# reference, the forward adjoint's mass conservation, the degenerate v = 0 rows, a manufactured
# affine solution at second order, the BUG step against the dense exponential, and the references.

using Test, LinearAlgebra, Random
using BondUpdateGalerkin.BUG, BondUpdateGalerkin.QTTS

include(joinpath(@__DIR__, "..", "..", "..", "common", "qtt_fixtures.jl"))

set_symmetry!(:none)
clear_fusion_cache!()

"The dense Heston generator on grid `g`, term by term from the dense one-sided stencils."
function dense_heston(g, p)
    npoints_x, npoints_v = 2^g.nx, 2^g.ny
    first_x, second_x = _ref_d(npoints_x, g.dx, 1, :onesided), _ref_d(npoints_x, g.dx, 2, :onesided)
    first_v, second_v = _ref_d(npoints_v, g.dy, 1, :onesided), _ref_d(npoints_v, g.dy, 2, :onesided)
    variance = diagm(collect(last(qtt_coords(g))))
    return 0.5 * _ref2(g, second_x, variance) +
           p.rho * p.sigma * _ref2(g, first_x, variance * first_v) +
           0.5 * p.sigma^2 * _ref2(g, nothing, variance * second_v) +
           (p.r - p.q) * _ref2(g, first_x, nothing) - 0.5 * _ref2(g, first_x, variance) +
           p.kappa * p.theta * _ref2(g, nothing, first_v) -
           p.kappa * _ref2(g, nothing, variance * first_v) - p.r * I
end

@testset "Heston 2D QTT" begin
    @testset "generator and forward adjoint vs dense ($order, $endian)" for (order, endian) in (
        (:grouped, :lsb_first),
        (:interleaved, :lsb_first),
    )
        g = _heston_grid(3, 3; order = order, endian = endian)
        npoints_x = 2^g.nx
        reference = dense_heston(g, _HP)
        generator = _grid_matrix(g, op_dense(heston_op(g, _HP)))
        @test _relerr(generator, reference) < 1e-11
        forward = _grid_matrix(g, op_dense(heston_op(g, _HP; adjoint = true)))
        @test _relerr(forward, Matrix(transpose(reference + _HP.r * I))) < 1e-11
        # every stencil annihilates constants, so the forward operator conserves sum p
        @test norm(ones(size(forward, 1))' * forward) < 1e-9 * norm(forward)
        # at v = 0 (the first npoints_x grid rows) no diffusion or mixed term survives
        no_vol_of_vol = HestonParams(_HP.r, _HP.q, _HP.kappa, _HP.theta, 0.0, 0.0)
        degenerate = _grid_matrix(g, op_dense(heston_op(g, no_vol_of_vol)))
        @test _relerr(generator[1:npoints_x, :], degenerate[1:npoints_x, :]) < 1e-11
        # the BUG MPO applies the same matrix to a complex state
        state = _random_tt(_tt_ranks(qtt_length(g), 4); rng = MersenneTwister(3))
        applied = qtt_apply(heston_mpo(g, _HP), qtt_mps(state); tol_trunc = 0.0)
        @test _relerr(
            vec(tt_dense(qtt_cores(applied))),
            op_dense(heston_op(g, _HP)) * vec(tt_dense(state)),
        ) < 1e-10
        @test_throws ArgumentError heston_op(qtt_grid(3, 3), _HP)
        @test_throws ArgumentError heston_op(
            qtt_grid(3, 3; y = (0.1, 0.5), periodic_x = false, periodic_y = false),
            _HP,
        )
    end

    @testset "manufactured affine solution at second order" begin
        # A_H exp(a x + b v) = lambda(v) exp(a x + b v) with lambda linear in v; the discrete
        # generator must reproduce it with an O(h^2) error at every point, boundaries included
        slope_x, slope_v = 0.5, -1.0
        p = _HP
        lambda(v) =
            (slope_x^2 - slope_x) * v / 2 +
            p.rho * p.sigma * slope_x * slope_v * v +
            p.sigma^2 * slope_v^2 * v / 2 +
            (p.r - p.q) * slope_x +
            p.kappa * (p.theta - v) * slope_v - p.r
        max_errors = Float64[]
        for nbits in (4, 5)
            g = _heston_grid(nbits, nbits)
            xs, vs = qtt_coords(g)
            field = qtt_separable(g, x -> exp(slope_x * x), v -> exp(slope_v * v))
            applied = qtt_values(op_apply(heston_op(g, p), field), g)
            exact = [lambda(v) * exp(slope_x * x + slope_v * v) for x in xs, v in vs]
            push!(max_errors, maximum(abs.(applied - exact)) / maximum(abs.(exact)))
        end
        @test max_errors[2] < 0.35 * max_errors[1]
        @test max_errors[2] < 1e-2
    end

    @testset "BUG step vs the dense exponential; the x|v bond grows" begin
        g = _heston_grid(4, 4)
        initial = qtt_separable(g, x -> exp(0.5x), v -> exp(-2v) * (1 + v))
        generator = heston_op(g, _HP)
        exact = exp(0.1 * op_dense(generator)) * vec(tt_dense(initial))
        errors = Float64[]
        for tol_basis in (1e-3, 1e-8)
            psi = qtt_mps(initial; center = 1)
            for _ = 1:5
                cbe_bug_step!(
                    psi,
                    qtt_to_mpo(generator),
                    ComplexF64(0.02);
                    tol_basis = tol_basis,
                    tol_trunc = 1e-12,
                    maxdim = 64,
                    hermitian = false,
                )
            end
            push!(errors, norm(vec(tt_dense(qtt_cores(psi))) - exact) / norm(exact))
            # the correlation term couples x and v, so the separable start must gain rank
            @test bond_dims(psi)[g.nx] > 1
        end
        @test errors[2] <= max(errors[1], 1e-10)
        @test errors[2] < 1e-5
    end

    @testset "references" begin
        @test feller(_HP)
        @test !feller(HestonParams(0.0, 0.0, 1.0, 0.01, 1.0, 0.0))
        # without vol-of-vol the Riccati equation is linear: B(t) = b e^{-kt} + c (1 - e^{-kt})
        p = HestonParams(0.03, 0.01, 2.0, 0.09, 0.0, 0.0)
        slope_x, slope_v, tau = 0.5, -1.0, 0.7
        stationary = (slope_x^2 - slope_x) / (2 * p.kappa)
        decay = exp(-p.kappa * tau)
        exact_B = slope_v * decay + stationary * (1 - decay)
        integral_B = stationary * tau + (slope_v - stationary) * (1 - decay) / p.kappa
        exact_C = ((p.r - p.q) * slope_x - p.r) * tau + p.kappa * p.theta * integral_B
        B, C = heston_riccati(p, slope_x, slope_v, tau)
        @test B ≈ exact_B atol = 1e-12
        @test C ≈ exact_C atol = 1e-12
        # a deep in-the-money call is its forward minus the discounted strike, to ~1e-7
        q = HestonParams(0.03, 0.0, 2.0, 0.04, 0.3, -0.5)
        @test heston_call(q, 1.0, 0.5, 0.04, 0.5) ≈ 1.0 - 0.5 * exp(-0.03 * 0.5) atol = 1e-4
        @test heston_call(q, 1.0, 0.9, 0.04, 0.5) > heston_call(q, 1.0, 1.1, 0.04, 0.5)
        # mass and affine moment of a separable density
        g = _heston_grid(4, 4)
        density = qtt_separable(g, x -> 1.0, v -> 1.0)
        @test heston_mass(density, g) ≈ qtt_cell(g) * 2^8
        xs, vs = qtt_coords(g)
        @test heston_moment(density, g, 0.5, -1.0) ≈
              qtt_cell(g) * sum(exp(0.5x) for x in xs) * sum(exp(-v) for v in vs)
        affine = heston_affine_field(g, 0.5, -1.0, 0.2)
        @test _relerr(qtt_values(affine, g), [exp(0.5x - v + 0.2) for x in xs, v in vs]) < 1e-12
    end
end
