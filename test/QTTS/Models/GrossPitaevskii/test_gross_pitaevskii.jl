# The Gross-Pitaevskii nonlinear right-hand side on QTT.
#
# The density, the
# zip-up products and their variational fit against dense, H[psi, t] and F(psi) against dense, the
# normalised energy with the paddle term, the density rebuilt from the
# trial state, the real-linear JVP against finite differences, and H0 against the dense stencils.
# The frozen-density step is `frozen_flow`.

using Test, LinearAlgebra, Random
using BondUpdateGalerkin.BUG, BondUpdateGalerkin.QTTS

include(joinpath(@__DIR__, "..", "..", "..", "common", "qtt_fixtures.jl"))

set_symmetry!(:none)
clear_fusion_cache!()

@testset "Gross-Pitaevskii nonlinear RHS" begin
    @testset "H0 with trap, mu and rotation vs dense stencils" begin
        g = qtt_grid(3, 3; x = (-2.0, 2.0), y = (-2.0, 2.0))
        npoints = 2^3
        xs, ys = qtt_coords(g)
        m = gp_model(g; gnl = 0.0, gamma = 0.1, mu = 0.7, omega = 1.3, Omega = 0.4)
        second = _ref_d(npoints, g.dx, 2, :periodic)
        first_order = _ref_d(npoints, g.dx, 1, :periodic)
        X, Y = diagm(collect(xs)), diagm(collect(ys))
        reference =
            -0.5 * (_ref2(g, second, nothing) + _ref2(g, nothing, second)) +
            (1.3^2 / 2) * (_ref2(g, X^2, nothing) + _ref2(g, nothing, Y^2)) - 0.7 * I +
            0.4im * (_ref2(g, X, first_order) - _ref2(g, first_order, Y))
        @test _relerr(_grid_matrix(g, op_dense(m.H0)), reference) < 1e-12
        @test _relerr(op_dense(m.H0), op_dense(op_adjoint(m.H0))) < 1e-12        # Hermitian
        @test_throws ArgumentError gp_model(qtt_grid(3, 3; periodic_x = false))
    end

    @testset "density, products, H[psi], F(psi) and the real-linear JVP" begin
        g = qtt_grid(3, 3; x = (-2.0, 2.0), y = (-2.0, 2.0))
        paddle = GPPaddle(4.0, -1.0, 1.0, 0.5, 0.5, 0.6, 0.8, 0.0, 0.15)
        m = gp_model(
            g;
            gnl = 1.5,
            gamma = 0.1,
            mu = 0.4,
            omega = 1.0,
            paddle = paddle,
            rho_tol = 1e-13,
        )
        state_one = _random_tt(_tt_ranks(6, 4); rng = MersenneTwister(41))
        state_two = _random_tt(_tt_ranks(6, 4); rng = MersenneTwister(42))
        values_one, values_two = vec(tt_dense(state_one)), vec(tt_dense(state_two))
        t = 0.2
        linear = op_dense(gp_linear_op(m, t))
        dense_field(v) = (-1im - m.gamma) .* (linear * v .+ m.g .* abs2.(v) .* v)
        # the density QTT is the elementwise |psi|^2, with a bounded error
        rho, info = gp_density(m, state_one)
        @test _relerr(vec(tt_dense(rho)), abs2.(values_one)) < 1e-11
        @test info.err_bound < 1e-11
        exact_density, _ = tt_compress(tt_hadamard(state_one, tt_conj(state_one)); tol = 1e-13)
        @test _relerr(vec(tt_dense(rho)), vec(tt_dense(exact_density))) < 1e-11
        @test gp_density_audit(m, state_one) < 1e-10
        # the same zip-up for any product: psi^2 (the Jacobian's conj term) and A1 .* A2
        squared, squared_info = tt_product(state_one, state_one; tol = 1e-13)
        @test _relerr(vec(tt_dense(squared)), values_one .^ 2) < 1e-11
        @test squared_info.err_bound < 1e-11
        mixed, _ = tt_product(state_one, state_two; tol = 1e-13)
        @test _relerr(vec(tt_dense(mixed)), values_one .* values_two) < 1e-11
        # the variational fit, warm-started from a different product of the same ranks
        fitted, fit_info = tt_hadamard_fit(state_one, tt_conj(state_one), mixed; tol = 1e-12)
        @test _relerr(vec(tt_dense(fitted)), abs2.(values_one)) < 1e-10
        @test fit_info.sweeps >= 1 && fit_info.chi == maximum(tt_ranks(fitted))
        fitted_mixed, _ = tt_hadamard_fit(state_one, state_two, rho; tol = 1e-12)
        @test _relerr(vec(tt_dense(fitted_mixed)), values_one .* values_two) < 1e-10
        # the normalised energy is the functional imaginary time descends, paddle included:
        # E N = F_gc + mu N
        observed = gp_observables(m, state_one, t)
        @test observed.E_pad != 0
        @test observed.E * observed.N ≈ observed.F_gc + m.mu * observed.N rtol = 1e-10
        # H[psi, t], its zip-up application and F(psi) against dense
        H1, _ = gp_hamiltonian_op(m, state_one, t)
        @test _relerr(op_dense(H1), linear + m.g * Diagonal(abs2.(values_one))) < 1e-11
        applied, _ = op_zipup(H1, state_one; tol = 1e-13)
        @test _relerr(vec(tt_dense(applied)), op_dense(H1) * values_one) < 1e-11
        @test _relerr(vec(tt_dense(gp_vector_field(m, state_one, t))), dense_field(values_one)) <
              1e-10
        # g = 0 is the linear path exactly
        linear_model = gp_model(g; gnl = 0.0, gamma = 0.1, mu = 0.4, omega = 1.0, paddle = paddle)
        @test _relerr(op_dense(first(gp_hamiltonian_op(linear_model, state_one, t))), linear) <
              1e-13
        # the density follows the trial state; a frozen one does not
        H2, _ = gp_hamiltonian_op(m, state_two, t)
        @test _relerr(op_dense(H2), linear + m.g * Diagonal(abs2.(values_two))) < 1e-11
        frozen, _ = gp_hamiltonian_op(m, state_two, t; rho = rho)
        @test _relerr(op_dense(frozen), linear + m.g * Diagonal(abs2.(values_one))) < 1e-11
        @test _relerr(op_dense(frozen), op_dense(H2)) > 1e-3
        frozen_generator = first(flow_op(frozen_flow(GPFlow(m), state_one, t), state_two, t))
        @test _relerr(op_dense(frozen_generator), op_dense(H1)) < 1e-12
        # the real-linear JVP against central differences of the dense field, both directions
        direction = _random_tt(_tt_ranks(6, 2); rng = MersenneTwister(43))
        direction_values = vec(tt_dense(direction))
        jvp = vec(tt_dense(gp_jvp(m, state_one, direction, t)))
        step = 1e-6
        difference =
            (
                dense_field(values_one .+ step .* direction_values) .-
                dense_field(values_one .- step .* direction_values)
            ) ./ (2step)
        @test _relerr(jvp, difference) < 1e-7
        jvp_imaginary = vec(tt_dense(gp_jvp(m, state_one, tt_scale(direction, 1im), t)))
        difference_imaginary =
            (
                dense_field(values_one .+ 1im * step .* direction_values) .-
                dense_field(values_one .- 1im * step .* direction_values)
            ) ./ (2step)
        @test _relerr(jvp_imaginary, difference_imaginary) < 1e-7
        @test _relerr(jvp_imaginary, 1im .* jvp) > 1e-3                         # not complex-linear
        @test _relerr(
            vec(tt_dense(gp_jvp(linear_model, state_one, direction, t))),
            (-1im - 0.1) .* (linear * direction_values),
        ) < 1e-11
    end

    @testset "observables and vortex count" begin
        g = qtt_grid(4, 4; x = (-4.0, 4.0), y = (-4.0, 4.0))
        m = gp_model(g; gnl = 0.0, gamma = 0.0, mu = 0.0, omega = 1.0)
        # the harmonic-oscillator ground state: E = mu = 1 up to discretisation, residual small
        ground = qtt_separable(g, x -> pi^(-0.25) * exp(-x^2 / 2), y -> pi^(-0.25) * exp(-y^2 / 2))
        observed = gp_observables(m, ground, 0.0)
        @test observed.N ≈ 1 atol = 1e-6
        @test observed.E ≈ 1 atol = 5e-2
        @test observed.mu ≈ observed.E atol = 1e-10
        @test observed.R_mu < 0.2
        # one positive vortex of (z - z0) exp(-|z|^2 / 4), placed inside a plaquette (at a grid
        # point the four surrounding plaquettes would be masked by the vanishing density)
        xs, ys = qtt_coords(g)
        vortex = [((x - 0.25) + 1im * (y - 0.25)) * exp(-(x^2 + y^2) / 4) for x in xs, y in ys]
        n_plus, n_minus, _ = vortex_count(vortex)
        @test (n_plus, n_minus) == (1, 0)
        @test vortex_count(conj.(vortex))[1:2] == (0, 1)
    end
end
