# The implicit Runge-Kutta stage solver in the root space.
#
# In complete frames
# the Galerkin stage solve is the full RK map, so on a linear flow the trapezoid and the two Radau
# IIA methods must reproduce their stability functions R(h G) applied to the state exactly, with
# the separable, the dense and no preconditioner, in real and in imaginary time. On a nonlinear
# flow every stage path (per-summand environments or the full operator, variational or zip-up
# density, recycled GMRES, lagged psi^2, fixed or inexact-Newton GMRES tolerance) solves the same
# equations. The separable preconditioner is checked against its dense inverse, the Kronecker sum
# against the root matrix of a separable operator (and a rotating control), GCRO-DR against the
# residual, the Newton stall exit against running to maxnewton. The frames are the dense QTT
# frames of QTTS, the interface implementation the solver is used with.

using Test, LinearAlgebra, Random
using BondUpdateGalerkin.BUG, BondUpdateGalerkin.QTTS
import BondUpdateGalerkin.BUG: _separable_precond, _gcro_real, GMRESRecycle

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
        @test stats.precond === :separable && stats.envs === :summand
        @test _relerr(vec(tt_dense(frame_state(frames, Y[end]))), exact) < 1e-10
        # the returned fields are the stage fields at the returned stages
        @test length(stats.fields) == length(Y)
        dense, dense_stats = rk_stage_solve(
            GPFlow(m),
            frames,
            projected,
            0.0,
            h,
            method;
            tol = 1e-13,
            precond = :dense,
        )
        @test dense_stats.precond === :linear_part && dense_stats.converged
        @test _relerr(dense[end], Y[end]) < 1e-10
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

    @testset "root environments: Kronecker sum and the separable inverse" begin
        envs = frame_root_envs(frames, m.H0)
        root_matrix = root_envs_matrix(envs)
        @test root_matrix ≈ frame_operator_matrix(frames, m.H0)
        kron_sum(Ml, Mr) =
            kron(Matrix(1.0I, size(Mr, 1), size(Mr, 1)), Ml) +
            kron(Mr, Matrix(1.0I, size(Ml, 1), size(Ml, 1)))
        Ml, Mr = root_envs_kron_sum(envs)
        # kinetic + trap is a Kronecker sum across the grouped x|y root bond: exact
        @test norm(kron_sum(Ml, Mr) - root_matrix) < 1e-10 * norm(root_matrix)
        # control: the rotation couples x and y, so its nearest Kronecker sum is not the operator
        rotating = gp_model(g; gnl = 0.0, gamma = 0.1, mu = 0.0, omega = 1.0, Omega = 0.5)
        rotating_envs = frame_root_envs(frames, rotating.H0)
        rotating_matrix = root_envs_matrix(rotating_envs)
        @test norm(kron_sum(root_envs_kron_sum(rotating_envs)...) - rotating_matrix) >
              1e-3 * norm(rotating_matrix)
        # the separable preconditioner is the inverse of I - h A (x) pref (Ml (+) Mr)
        pref = -1im - m.gamma
        rng_rhs = MersenneTwister(7)
        for method in (:trapezoid, :radau2, :radau3)
            butcher = RK_TABLEAUX[method].A
            nstages = size(butcher, 1)
            rhs = [randn(rng_rhs, ComplexF64, size(Ml, 1), size(Mr, 1)) for _ = 1:nstages]
            solved = _separable_precond(Ml, Mr, pref, butcher, h)(rhs)
            system =
                Matrix{ComplexF64}(I, nstages * length(rhs[1]), nstages * length(rhs[1])) -
                h * kron(butcher, pref * kron_sum(Ml, Mr))
            @test system * vcat(vec.(solved)...) ≈ vcat(vec.(rhs)...) rtol = 1e-10
        end
    end

    @testset "GCRO-DR: recycled directions, solves to tolerance" begin
        rng_system = MersenneTwister(3)
        dimension = 60
        base =
            Matrix(1.0I, dimension, dimension) +
            0.3 * randn(rng_system, dimension, dimension) / sqrt(dimension)
        recycle = GMRESRecycle(4)
        for shift in (0.0, 0.01, 0.02)
            operator = base + shift * I
            target = randn(rng_system, dimension)
            solution, _, relres =
                _gcro_real(x -> operator * x, target, recycle; tol = 1e-12, restart = 10)
            @test norm(operator * solution - target) <= 1e-10 * norm(target)
            @test 0 < size(recycle.directions, 2) <= 4
        end
        # the later solves re-imaged the recycled directions under their own operators
        @test recycle.extra_products > 0
    end

    nonlinear = gp_model(g; gnl = 1.0, gamma = 0.1, mu = 0.0, omega = 1.0, rho_tol = 1e-14)
    # real time, and the normalised imaginary-time flow (whose Jacobian carries D mu)
    @testset "nonlinear stages: every path solves the same equations" for normalised in
                                                                          (false, true)
        flow = normalised ? GPFlow(nonlinear; imaginary = true) : GPFlow(nonlinear)
        @test flow_normalized(flow) == normalised
        # the full path's Jacobian has no D mu: a Newton-like iteration, given room to converge
        reference, reference_stats = rk_stage_solve(
            flow,
            frames,
            projected,
            0.0,
            h,
            :radau2;
            tol = 1e-12,
            envs = :full,
            precond = :dense,
            gmres_tol = 1e-14,
            maxnewton = 60,
        )
        @test reference_stats.converged && reference_stats.env_builds == -1
        @test reference_stats.forcing === :fixed
        variants = (
            NamedTuple(),
            (density = :zipup,),
            (gmres_recycle = 3,),
            (jacobian_psi2 = :lagged,),
            (precond = :none,),
            (gmres_tol = 1e-14,),
        )
        for keywords in variants
            Y, stats =
                rk_stage_solve(flow, frames, projected, 0.0, h, :radau2; tol = 1e-12, keywords...)
            @test stats.converged
            @test stats.env_builds > 0
            @test _relerr(Y[end], reference[end]) < 1e-9
        end
    end

    @testset "Newton stall exit" begin
        flow = GPFlow(nonlinear)
        # an unreachable tolerance: the residual stagnates at roundoff
        _, stalled =
            rk_stage_solve(flow, frames, projected, 0.0, h, :radau2; tol = 1e-30, maxnewton = 15)
        @test !stalled.converged && stalled.newton < 15
        _, unlimited = rk_stage_solve(
            flow,
            frames,
            projected,
            0.0,
            h,
            :radau2;
            tol = 1e-30,
            maxnewton = 15,
            newton_stall = Inf,
        )
        @test !unlimited.converged && unlimited.newton == 15
    end

    @test_throws ArgumentError rk_stage_solve(
        GPFlow(m),
        frames,
        projected,
        0.0,
        h,
        :radau2;
        precond = :unknown,
    )
    @test_throws ArgumentError rk_stage_solve(
        GPFlow(m),
        frames,
        projected,
        0.0,
        h,
        :radau2;
        envs = :all,
    )
end
