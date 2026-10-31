# The nonlinear QTT BUG step (implementation plan WP5): a convergence test on an exact nonlinear
# solution, the order lost by a frozen density, and the frozen b155550 QTT runs (Heston backward
# pricing, GP real time and ground state).
#
# From BUG-Julia@b155550:tests/sweeps/test_qtt_task2.jl V02 (plane wave, cut to order 2 and two step
# sizes) and the fixtures `qtt_heston_backward`, `qtt_gp` of docs/dev/tools/make_fixtures_b155550.jl.

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
        for k = 0:(legacy["steps"]-1)
            qtt_bug_step!(
                psi,
                flow,
                k * h,
                h;
                order = legacy["order"],
                maxdim = 64,
                tol_trunc = 1e-10,
            )
        end
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
        history =
            gp_ground_state!(qtt_mps(initial), m, uniform_schedule(0.5, 10); order = 2, maxdim = 64)
        energies = [entry.E for entry in history if entry.accepted]
        frozen_energies = Float64.(legacy["ground_state_energies"])
        @test length(energies) == length(frozen_energies)
        @test maximum(abs.(energies .- frozen_energies)) < 1e-6
        # imaginary time lowers the energy monotonically
        @test all(
            k -> energies[k+1] <= energies[k] + 1e-10 * abs(energies[k]),
            1:(length(energies)-1),
        )
    end
end
