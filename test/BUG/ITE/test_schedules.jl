# The imaginary-time schedules (uniform, log-panel and geometric grids).

using Test
using BondUpdateGalerkin.BUG

@testset "IT-U1: grids" begin
    for (bmax, P, k) in ((8.0, 5, 4), (1.0, 1, 4), (30.0, 7, 2))
        s = log_panel_steps(bmax, P; per_panel = k)
        g = beta_grid(s)
        @test length(s) == P * k
        @test all(>(0), s)
        @test g[1] == 0.0
        @test g[end] ≈ bmax rtol = 1e-14
        @test all(diff(g) .> 0)                                     # no gaps, no duplicates
        panels = [s[(j-1)*k+1:j*k] for j = 1:P]
        @test all(all(x -> x ≈ p[1], p) for p in panels)            # uniform inside a panel
        widths = [sum(p) for p in panels]
        P >= 2 && @test widths[2] ≈ widths[1]                       # first two panels equal
        P >= 3 && @test all(widths[j+1] ≈ 2 * widths[j] for j = 2:(P-1))   # then doubling
        @test s[1] ≈ bmax / (2.0^(P - 1) * k)                       # early resolution
        @test maximum(s) ≈ (P == 1 ? bmax : bmax / 2) / k           # coarsest step: per_panel
        @test log_panel_steps(bmax, P; per_panel = k) == s          # reproducible
    end
    u = uniform_beta_steps(2.0, 8)
    @test all(==(0.25), u) && beta_grid(u)[end] ≈ 2.0
    @test_throws ArgumentError log_panel_steps(1.0, 0)
    @test_throws ArgumentError log_panel_steps(-1.0, 3)
    @test_throws ArgumentError uniform_beta_steps(1.0, 0)
end
