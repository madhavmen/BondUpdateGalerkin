# The nonlinear QTT BUG step: a convergence test on an exact nonlinear
# solution, the order lost by a frozen density, the per-step tolerances, the step's solver
# routing, stage-basis rules and ledger, and the frozen b155550 QTT runs (Heston backward pricing,
# GP real time and ground state). The frozen Heston run was made by Newton-GMRES stages with exact
# stage keeps at fixed tolerances, so it passes those settings; the default rational root is
# checked against them on one step.
#
# The plane wave runs at order 2 and two step sizes; the frozen runs are the fixtures
# `qtt_heston_backward` and `qtt_gp` of docs/dev/tools/make_fixtures_b155550.jl.

using Test, LinearAlgebra
import JSON
using BondUpdateGalerkin.BUG, BondUpdateGalerkin.QTTS

include(joinpath(@__DIR__, "..", "..", "common", "qtt_fixtures.jl"))

set_symmetry!(:none)
clear_fusion_cache!()

const FIXTURE_DIR = joinpath(@__DIR__, "..", "..", "fixtures", "b155550")

"A frozen complex vector `{re, im}`."
frozen_vector(entry) = ComplexF64.(entry["re"]) .+ 1im .* ComplexF64.(entry["im"])

"Run `nsteps` steps of `step!(psi, t, h)` from the cores `initial`; return the final dense vector."
function run_steps(step!, initial, h, nsteps)
    psi = qtt_mps(initial)
    for k = 0:(nsteps-1)
        step!(psi, k * h, h)
    end
    return vec(tt_dense(qtt_cores(psi)))
end

"Observed convergence rates `log(e_k / e_{k+1}) / log(h_k / h_{k+1})`."
rates(step_sizes, errors) = [
    log(errors[k] / errors[k+1]) / log(step_sizes[k] / step_sizes[k+1]) for k = 1:(length(errors)-1)
]

@testset "nonlinear QTT BUG step" begin
    @testset "per-step tolerances" begin
        @test qtt_step_tolerance(1e-6, 0.1, 2) == 1e-6
        @test qtt_step_tolerance(1e-6, 0.05, 2) ≈ 1e-6 * 0.5^3
        @test qtt_step_tolerance(1e-6, -0.05, 2) ≈ 1e-6 * 0.5^3
        @test qtt_step_tolerance(1e-6, 0.5, 2) == 1e-6                   # never looser
        @test qtt_step_tolerance(1e-6, 1e-4, 4) == 1e-14                 # floored
        @test qtt_step_tolerance(1e-16, 1e-4, 4) == 1e-16                # already below the floor
        @test qtt_step_tolerance(1e-6, 0.05, 2; scaling = :fixed) == 1e-6
        @test qtt_step_tolerance(1e-6, 0.05, 2; h_ref = 0.05) == 1e-6
        @test_throws ArgumentError qtt_step_tolerance(1e-6, 0.1, 2; scaling = :bogus)
        @test_throws ArgumentError qtt_step_tolerance(1e-6, 0.1, 2; h_ref = 0.0)
    end

    @testset "one GP step: routing, stage paths, stage basis, ledger" begin
        g = qtt_grid(3, 3; x = (-5.0, 5.0), y = (-5.0, 5.0))
        m = gp_model(g; gnl = 1.0, gamma = 0.1, mu = 0.0, omega = 1.0, rho_tol = 1e-13)
        initial = qtt_separable(g, x -> exp(-(x - 0.5)^2 / 2), y -> exp(-y^2 / 2) + 0im)
        function gp_step(; keywords...)
            psi = qtt_mps(initial)
            report = qtt_bug_step!(
                psi,
                GPFlow(m),
                0.0,
                0.1;
                order = 2,
                maxdim = 64,
                tol_basis = 1e-10,
                tol_trunc = 1e-12,
                keywords...,
            )
            return vec(tt_dense(qtt_cores(psi))), report
        end
        state, report = gp_step()
        @test report.converged && report.solver === :newton_gmres
        @test length(report.stats) == 2                                  # npc = order - 1 = 1
        @test report.tol_step.stage == 1e-10 && report.tol_step.scaling === :order
        @test report.newton > 0 && report.gmres > 0 && report.env_builds > 0
        @test 0 <= report.f_rho <= 1 && 0 <= report.f_cbe <= 1
        @test report.t_step >= report.t_setup + report.t_keep + report.t_trunc
        @test report.dd.probes >= 0 && report.chi <= 64
        # the full-operator path with zipped-up densities, and exact stage keeps, reach the same
        # step
        full_state, _ =
            gp_step(; stage_envs = :full, stage_density = :zipup, stage_field_rho = :rebuild)
        @test _relerr(full_state, state) < 1e-8
        exact_state, _ = gp_step(; stage_basis = :exact)
        @test _relerr(exact_state, state) < 1e-6
        # an unreachable stage tolerance: the step reports failure after one root solve
        _, failed = gp_step(; tol_stage = 1e-30)
        @test !failed.converged && length(failed.stats) == 1
        @test_throws ArgumentError gp_step(; stage_solver = :krylov_rational)
        @test_throws ArgumentError gp_step(; stage_basis = :none)
    end

    @testset "damped nonlinear plane wave: order 2, lost by a frozen density" begin
        # psi = sqrt(rho(t)) exp(i (k.x + phi(t))) solves the discrete damped GP exactly, with
        # alpha the discrete kinetic eigenvalue: rho' = -2 gamma (alpha + g rho) rho
        g = qtt_grid(3, 3; x = (0.0, 2pi), y = (0.0, 2pi))
        gamma, nonlinearity, rho0 = 0.5, 1.0, 0.8
        m = gp_model(g; gnl = nonlinearity, gamma = gamma, mu = 0.0, omega = 0.0, rho_tol = 1e-14)
        discrete_k2(k, spacing) = (2 - 2cos(k * spacing)) / spacing^2
        alpha = (discrete_k2(1, g.dx) + discrete_k2(1, g.dy)) / 2
        rho(t) =
            rho0 * exp(-2gamma * alpha * t) /
            (1 + nonlinearity * rho0 / alpha * (1 - exp(-2gamma * alpha * t)))
        phase(t) = log(rho(t) / rho0) / (2gamma)
        initial = qtt_separable(g, x -> sqrt(rho0) * exp(1im * x), y -> exp(1im * y))
        final_time = 0.4
        exact = vec(
            tt_dense(
                qtt_separable(
                    g,
                    x -> sqrt(rho(final_time)) * exp(1im * (x + phase(final_time))),
                    y -> exp(1im * y),
                ),
            ),
        )
        step_sizes = [0.1, 0.05]
        consistent = [
            norm(
                run_steps(
                    (psi, t, h) -> qtt_bug_step!(
                        psi,
                        GPFlow(m),
                        t,
                        h;
                        order = 2,
                        tol_trunc = 1e-14,
                        tol_stage = 1e-13,
                    ),
                    initial,
                    h,
                    round(Int, final_time / h),
                ) - exact,
            ) / norm(exact) for h in step_sizes
        ]
        # the frozen flow holds H[psi_n, t_n] through the step: first order at best
        frozen = [
            norm(
                run_steps(
                    (psi, t, h) -> qtt_bug_step!(
                        psi,
                        frozen_flow(GPFlow(m), qtt_cores(psi), t),
                        t,
                        h;
                        order = 2,
                        tol_trunc = 1e-14,
                        tol_stage = 1e-13,
                    ),
                    initial,
                    h,
                    round(Int, final_time / h),
                ) - exact,
            ) / norm(exact) for h in step_sizes
        ]
        @info "plane wave errors: consistent $consistent (rate $(rates(step_sizes, consistent))), frozen $frozen (rate $(rates(step_sizes, frozen)))"
        @test only(rates(step_sizes, consistent)) > 1.8 || maximum(consistent) < 1e-11
        @test only(rates(step_sizes, frozen)) < 1.5
        @test consistent[end] < frozen[end]
    end

    @testset "frozen b155550 run: Heston backward pricing (order 3)" begin
        legacy = JSON.parsefile(joinpath(FIXTURE_DIR, "qtt_heston_backward.json"))
        bits = legacy["N"]
        params = HestonParams(0.03, 0.0, 1.5, 0.09, 0.5, -0.7)
        g = qtt_grid(
            bits,
            bits;
            x = (-3.0, 3.0),
            y = (0.0, 1.0),
            periodic_x = false,
            periodic_y = false,
        )
        softplus(z) = z > 30 ? z : log1p(exp(z))
        initial =
            qtt_separable(g, x -> 0.02 * softplus((exp(x) - 1.0) / 0.02), v -> 1.0; tol = 1e-12)
        flow = LinearFlow(heston_op(g, params))
        psi = qtt_mps(initial)
        h = legacy["h"]
        # the frozen run's settings: stage equations, exact stage keeps, fixed tolerances
        legacy_settings = (stage_solver = :newton_gmres, stage_basis = :exact, tol_scaling = :fixed)
        for k = 0:(legacy["steps"]-1)
            qtt_bug_step!(
                psi,
                flow,
                k * h,
                h;
                order = legacy["order"],
                maxdim = 64,
                tol_trunc = 1e-10,
                legacy_settings...,
            )
        end
        # one step by the default for a linear flow, the unshifted rational root, against the stages
        rational_state, newton_state = qtt_mps(initial), qtt_mps(initial)
        # (fixed tolerances: the rational root's Arnoldi cap meets tol_stage = 1e-9)
        comparison =
            (order = 3, maxdim = 64, tol_trunc = 1e-10, tol_stage = 1e-9, tol_scaling = :fixed)
        rational = qtt_bug_step!(rational_state, flow, 0.0, h; comparison...)
        qtt_bug_step!(newton_state, flow, 0.0, h; comparison..., stage_solver = :newton_gmres)
        @test rational.solver === :krylov_rational && rational.converged
        @test _relerr(
            vec(tt_dense(qtt_cores(rational_state))),
            vec(tt_dense(qtt_cores(newton_state))),
        ) < 1e-6
        final = vec(tt_dense(qtt_cores(psi)))
        reference = frozen_vector(legacy["final_state"])
        @info "Heston fixture: relative difference $(_relerr(final, reference)), chi $(maximum(bond_dims(psi))) (frozen $(legacy["trajectory"][end]["chi"]))"
        @test _relerr(final, reference) < 1e-6
    end

    @testset "frozen b155550 run: GP real time and ground state (order 2)" begin
        legacy = JSON.parsefile(joinpath(FIXTURE_DIR, "qtt_gp.json"))
        g = qtt_grid(3, 3; x = (-5.0, 5.0), y = (-5.0, 5.0))
        m = gp_model(g; gnl = 1.0, gamma = 0.1, mu = 0.0, omega = 1.0, rho_tol = 1e-13)
        phi0(x) = pi^(-0.25) * exp(-x^2 / 2)
        phi1(x) = sqrt(2) * x * phi0(x)
        initial = tt_lincomb([
            (1.0, qtt_separable(g, phi0, phi0)),
            (0.25, qtt_separable(g, phi1, phi0)),
            (0.15im, qtt_separable(g, phi0, phi1)),
        ])
        psi = qtt_mps(initial)
        for k = 0:3
            qtt_bug_step!(
                psi,
                GPFlow(m),
                k * 0.1,
                0.1;
                order = 2,
                maxdim = 64,
                tol_basis = 1e-10,
                tol_trunc = 1e-12,
            )
        end
        @test _relerr(vec(tt_dense(qtt_cores(psi))), frozen_vector(legacy["final_state"])) < 1e-6
        reported = Ref(0)
        history = gp_ground_state!(
            qtt_mps(initial),
            m,
            uniform_schedule(0.5, 10);
            order = 2,
            maxdim = 64,
            on_step = entry -> (reported[] += 1),
        )
        @test reported[] == length(history)                       # every ledger entry, as recorded
        @test issorted([entry.newton for entry in history]) && history[end].newton > 0
        energies = [entry.E for entry in history if entry.accepted]
        frozen_energies = Float64.(legacy["ground_state_energies"])
        @test length(energies) == length(frozen_energies)
        @test maximum(abs.(energies .- frozen_energies)) < 1e-6
        # imaginary time lowers the energy monotonically
        @test all(
            k -> energies[k+1] <= energies[k] + 1e-10 * abs(energies[k]),
            1:(length(energies)-1),
        )
        # particle number N, and growth: one accepted step per entry, at most `growth` times the
        # last accepted one
        relaxed = qtt_mps(initial)
        grown = gp_ground_state!(
            relaxed,
            m,
            [0.05, 0.5, 0.5];
            order = 2,
            maxdim = 64,
            N = 2.0,
            growth = 2,
        )
        steps = [entry.h for entry in grown if entry.accepted && entry.reason === :ok]
        @test length(steps) == 3
        @test all(k -> steps[k+1] <= 2 * steps[k] + 1e-14, 1:(length(steps)-1))
        @test qtt_l2norm(relaxed, g)^2 ≈ 2.0 rtol = 1e-10
        @test_throws ArgumentError gp_ground_state!(qtt_mps(initial), m, [0.1]; growth = 0.5)
    end
end
