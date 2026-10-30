# Moved from BUG-Julia@b155550:tests/sweeps/test_adaptive_api.jl, UT01 (the budget object, the
# allocation rule, the work ceiling, the ledger, and the public step refusing a retired knob).
# Not moved: the refusals of the DMRG/TDVP baselines (they move with Baselines) and UT02 (needs the
# adaptive bond fixtures).

using Test
using BondUpdateGalerkin.BUG, BondUpdateGalerkin.QMB

@testset "UT01: public API cleanup and knob migration" begin
    @testset "the PUBLIC entry refuses the mixture before touching the state" begin
        # What a caller meets is `cbe_bug_step!`, and the refusal must happen there, before
        # any evolution.
        set_symmetry!(:U1)
        mpo, psi = xxz_mpo(6; J = 1.0, delta = 0.0), neel_state(6)
        dims0 = bond_dims(psi)
        dt = ComplexF64(-0.05im)
        # The fixed-knob path is removed from the solver: a retired knob is no keyword.
        @test_throws MethodError cbe_bug_step!(psi, mpo, dt; tol_basis = 1e-5, dex = 4)
        @test_throws MethodError cbe_bug_step!(psi, mpo, dt; trunc_thresh = 1e-10)
        @test_throws MethodError cbe_bug_midpoint_step!(
            psi,
            mpo,
            dt;
            tol_expv = 1e-10,
            stol_pre = 1e-10,
        )
        @test bond_dims(psi) == dims0                    # refused, not half-run
    end

    @testset "the budget object validates its three tolerances" begin
        b = AdaptiveBudget(1e-5, 1e-7, 1e-10)
        @test b.basis == 1e-5
        @test b.trunc == 1e-7
        @test b.expv == 1e-10

        # A non-positive basis or expv tolerance is a caller error, not a "keep everything"
        # instruction: both divide or bound a search, and 0 would make every residual fail.
        @test_throws ArgumentError AdaptiveBudget(0.0, 1e-7, 1e-10)
        @test_throws ArgumentError AdaptiveBudget(-1e-5, 1e-7, 1e-10)
        @test_throws ArgumentError AdaptiveBudget(1e-5, 1e-7, 0.0)
        # `trunc = 0` IS meaningful -- "discard nothing" -- so it must be accepted.
        @test AdaptiveBudget(1e-5, 0.0, 1e-10).trunc == 0.0
        @test_throws ArgumentError AdaptiveBudget(1e-5, -1e-7, 1e-10)
    end
    @testset "the allocation rule is the documented one" begin
        b = AdaptiveBudget(1e-4, 1e-7, 1e-10)
        a = basis_allocation(b, 19; tau = 0.05, nrm = 1.0)       # L = 20 -> 19 bonds
        @test a.per_bond ≈ 1e-4 / sqrt(19)
        @test a.sample ≈ a.per_bond / 2
        @test a.tail ≈ a.per_bond / 2
        @test a.krylov ≈ a.per_bond / 2
        @test a.scale ≈ 0.05

        # More bonds means a tighter per-bond allowance, and in quadrature rather than
        # linearly -- four times the bonds halves the allowance.
        a4 = basis_allocation(b, 76; tau = 0.05, nrm = 1.0)
        @test a4.per_bond ≈ a.per_bond / 2

        # The norm makes the comparison relative.
        @test basis_allocation(b, 19; tau = 0.05, nrm = 2.0).scale ≈ 0.025
    end

    @testset "tau = 0 is a basis question, not a degenerate one" begin
        # A zero timestep is how a caller asks "is this subspace closed under H" with no
        # evolution. Scaling by |tau| there would make every residual negligible and return
        # the unexpanded frame -- a silent refusal to expand, which is the failure mode.
        a = basis_allocation(AdaptiveBudget(1e-5, 1e-7, 1e-10), 11; tau = 0, nrm = 1.0)
        @test a.scale == 1.0
        @test indicator(a, 0.25) == 0.25
    end

    @testset "the indicator applies the scale exactly once" begin
        a = basis_allocation(AdaptiveBudget(1e-5, 1e-7, 1e-10), 9; tau = 0.1, nrm = 4.0)
        @test a.scale ≈ 0.025
        @test indicator(a, 8.0) ≈ 0.2
    end

    @testset "the work ceiling bounds growth without freezing a wide bond" begin
        # Twice the resource limit is the headroom; the `2r` floor is what keeps a bond
        # already at or above `maxdim` from being pinned at its current rank.
        @test work_ceiling(128, 10) == 256
        @test work_ceiling(128, 200) == 400        # r > maxdim: the floor governs
        @test work_ceiling(0, 0) == 1              # never zero, which would admit nothing
        # the factor is the local dimension: the whole one-site space of a d = 4 site
        @test work_ceiling(128, 10, 4) == 512
        @test work_ceiling(128, 200, 4) == 800
    end

    @testset "the ledger starts clean and merges conservatively" begin
        a, b = AdaptiveLedger(), AdaptiveLedger()
        @test a.tolerance_met
        @test !a.workcap_reached

        a.max_sample, a.n_probes = 1e-6, 12
        b.max_sample, b.n_probes = 3e-6, 7
        b.workcap_reached = true
        b.tolerance_met = false
        merge_ledger!(a, b)

        # Indicators take the WORST of the two, counts ADD, and both status flags are
        # pessimistic: one half-sweep failing its budget fails the step.
        @test a.max_sample == 3e-6
        @test a.n_probes == 19
        @test a.workcap_reached
        @test !a.tolerance_met
    end
end
