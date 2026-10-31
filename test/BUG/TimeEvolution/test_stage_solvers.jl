# The implicit Runge-Kutta stage solver in the root space (implementation plan WP3/WP5).
#
# From BUG-Julia@b155550:tests/sweeps/test_qtt_task2.jl V01 (stage-solver part): in complete frames
# the Galerkin stage solve is the full RK map, so on a linear flow the trapezoid and the two Radau
# IIA methods must reproduce their stability functions R(h G) applied to the state exactly, with
# and without the linear-part preconditioner, in real and in imaginary time. The frames are the
# dense QTT frames of QTTS, the interface implementation the solver is used with.

using Test, LinearAlgebra, Random
using BondUpdateGalerkin.BUG, BondUpdateGalerkin.QTTS

include(joinpath(@__DIR__, "..", "..", "common", "qtt_fixtures.jl"))

set_symmetry!(:none)
clear_fusion_cache!()

"The stability function `R(z)` of each root method."
const STABILITY = Dict(
    :trapezoid => z -> (1 + z / 2) / (1 - z / 2),
    :radau2 => z -> (1 + z / 3) / (1 - 2z / 3 + z^2 / 6),
    :radau3 => z -> (1 + 2z / 5 + z^2 / 20) / (1 - 3z / 5 + 3z^2 / 20 - z^3 / 60),
)

"`R(h M)` by the eigendecomposition of `M` (diagonalisable test generators)."
function rational_map(method, M, h)
    F = eigen(h .* M)
    return F.vectors * Diagonal(STABILITY[method].(F.values)) / F.vectors
end

@testset "RK stage solver" begin
    @test qtt_root_for_order(2) === :trapezoid
    @test qtt_root_for_order(4) === :radau3
    @test_throws ArgumentError qtt_root_for_order(5)
    # every tableau is stiffly accurate and consistent: the last row is b, the rows sum to c
    for (_, tableau) in RK_TABLEAUX
        @test tableau.c[end] == 1.0
        @test vec(sum(tableau.A; dims = 2)) ≈ tableau.c
    end

    g = qtt_grid(3, 3; x = (-5.0, 5.0), y = (-5.0, 5.0))
    m = gp_model(g; gnl = 0.0, gamma = 0.1, mu = 0.0, omega = 1.0)
    phi0(x) = pi^(-0.25) * exp(-x^2 / 2)
    phi1(x) = sqrt(2) * x * phi0(x)
    initial = tt_lincomb([
        (1.0, qtt_separable(g, phi0, phi0)),
        (0.25, qtt_separable(g, phi1, phi0)),
        (0.15im, qtt_separable(g, phi0, phi1)),
    ])
    initial_values = vec(tt_dense(initial))
    h = 0.1
    # complete frames: full-rank isometries at the centre bond (8 x 8 = the whole 64-dimensional space)
    rng = MersenneTwister(13)
    left_cores = qtt_cores(qtt_mps(_random_tt(_tt_ranks(6, 64); rng = rng); center = 6))
    right_cores = qtt_cores(qtt_mps(_random_tt(_tt_ranks(6, 64); rng = rng); center = 1))
    frames = QTTFrames(
        [ComplexF64.(core) for core in left_cores[1:3]],
        [ComplexF64.(core) for core in right_cores[4:6]],
    )
    @test size(frames.W[3], 3) == 8 && size(frames.Z[1], 1) == 8
    @test frame_defect(frames) < 1e-12
    @test frame_miss(frames, initial) < 1e-12
    projected = frame_project(frames, nothing, initial)

    @testset "real time, $method" for method in (:trapezoid, :radau2, :radau3)
        generator = (-1im - m.gamma) .* op_dense(m.H0)
        exact = rational_map(method, generator, h) * initial_values
        Y, stats = rk_stage_solve(GPFlow(m), frames, projected, 0.0, h, method; tol = 1e-13)
        @test stats.converged && stats.newton <= 2                 # linear: Newton exact in one pass
        @test stats.precond === :linear_part
        @test _relerr(vec(tt_dense(frame_state(frames, Y[end]))), exact) < 1e-10
        plain, plain_stats = rk_stage_solve(
            GPFlow(m),
            frames,
            projected,
            0.0,
            h,
            method;
            tol = 1e-13,
            precond = :none,
        )
        @test plain_stats.precond === :none && plain_stats.converged
        @test _relerr(plain[end], Y[end]) < 1e-10
        @test stats.gmres <= plain_stats.gmres
        # LinearFlow of the same generator is the same map
        linear, _ = rk_stage_solve(
            LinearFlow(m.H0, -1im - m.gamma),
            frames,
            projected,
            0.0,
            h,
            method;
            tol = 1e-13,
        )
        @test _relerr(linear[end], Y[end]) < 1e-10
    end

    @testset "imaginary time, $method" for method in (:trapezoid, :radau2, :radau3)
        flow = GPFlow(m; imaginary = true, normalized = false)
        @test flow.pref == -1 &&
              !flow_normalized(flow) &&
              flow_normalized(GPFlow(m; imaginary = true))
        exact = rational_map(method, -op_dense(m.H0), h) * initial_values
        Y, stats = rk_stage_solve(flow, frames, projected, 0.0, h, method; tol = 1e-13)
        @test stats.converged
        @test _relerr(vec(tt_dense(frame_state(frames, Y[end]))), exact) < 1e-10
    end
end
