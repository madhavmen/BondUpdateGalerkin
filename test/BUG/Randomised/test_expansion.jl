# Moved from BUG-Julia@b155550:tests/sweeps/test_adaptive_rsvd.jl, UT05 (Test A, the retained-rank
# tail test, on plain singular-value lists), and tests/sweeps/test_dd_cbe.jl (the two-spin
# entangling step). Not moved: the testsets that need the adaptive and DD bond fixtures.

using Test, LinearAlgebra
import BondUpdateGalerkin
using BondUpdateGalerkin.BUG, BondUpdateGalerkin.QMB

# The legacy tests reach the expansion internals through this alias.
const RC = BondUpdateGalerkin.BUG

# A Telum `kept_list` entry is `(sigma, degeneracy, _, _)`; only the first two are read.
_mklist(sigmas, degs = ones(Int, length(sigmas))) =
    [(sigmas[i], degs[i], nothing, nothing) for i in eachindex(sigmas)]


@testset "UT05: Test A, the retained-rank tail test" begin

    @testset "the plan's spectrum (10, 9, 3, 0.1): the smallest k that fits" begin
        list = _mklist([10.0, 9.0, 3.0, 0.1])
        # tails: k=3 -> 0.1, k=2 -> sqrt(9.01) = 3.0017, k=1 -> sqrt(90.01) = 9.487
        @test first(RC._tail_rank(list, 1.0, 1.0, 4)) == 3
        @test first(RC._tail_rank(list, 1.0, 4.0, 4)) == 2
        @test first(RC._tail_rank(list, 1.0, 9.6, 4)) == 1
        @test first(RC._tail_rank(list, 1.0, 0.05, 4)) == 4
    end

    @testset "it keeps the smallest rank whose tail fits the allowance" begin
        # Weights 1, 1e-2, 1e-4, 1e-6: each tail is dominated by its first discarded entry.
        list = _mklist([1.0, 1e-2, 1e-4, 1e-6])
        k, tail = RC._tail_rank(list, 1.0, 2e-4, 4)
        @test k == 2
        @test tail <= 2e-4
        k, _ = RC._tail_rank(list, 1.0, 5e-5, 4)
        @test k == 3
        k, _ = RC._tail_rank(list, 1.0, 2e-2, 4)
        @test k == 1
    end

    @testset "the retained rank is monotone in the allowance" begin
        list = _mklist([1.0, 0.5, 0.2, 0.05, 0.01, 1e-3])
        ks = [first(RC._tail_rank(list, 1.0, a, 6))
              for a in (1e-6, 1e-3, 1e-2, 0.1, 0.5, 2.0)]
        # A looser budget can never demand MORE rank.
        @test issorted(ks; rev = true)
        @test ks[1] == 6        # unmeetable: nothing may be discarded
        @test ks[end] == 0      # everything fits inside the allowance
    end

    @testset "degeneracy is counted, not ignored" begin
        # One entry standing for 100 states carries 100x the weight of a lone one. Ranking on
        # sigma alone would discard it as "the same size as its neighbour" and lose 100 states.
        plain = _mklist([1.0, 0.1])
        multi = _mklist([1.0, 0.1], [1, 100])
        a = 0.5                       # fits a lone 0.1 (0.1 <= 0.5) but not 100 of them (1.0)
        @test first(RC._tail_rank(plain, 1.0, a, 2)) == 1
        @test first(RC._tail_rank(multi, 1.0, a, 2)) == 2
    end

    @testset "weight the SVD already cut is part of the tail" begin
        # ⛔ THE REGRESSION THIS GUARDS: measuring only the surviving list reports a rank as
        # certified while the weight below the SVD's own cutoff has already breached the
        # budget. The tail must be read against the WHOLE spectrum.
        list = _mklist([1.0, 1e-3])
        base = (0.5)^2
        k_nobase, _ = RC._tail_rank(list, 1.0, 0.1, 2)
        k_base, tail = RC._tail_rank(list, 1.0, 0.1, 2; base = base)
        @test k_nobase == 1
        @test k_base == 2
        @test tail > 0.1
    end

    @testset "the scale is applied, so the test is on the step's error" begin
        list = _mklist([1.0, 1e-2])
        @test first(RC._tail_rank(list, 1.0, 2e-2, 2)) == 1
        @test first(RC._tail_rank(list, 10.0, 2e-2, 2)) == 2
    end

    @testset "an unmeetable allowance returns kmax, never more" begin
        list = _mklist([1.0, 1.0, 1.0, 1.0])
        k, tail = RC._tail_rank(list, 1.0, 1e-12, 2)
        @test k == 2
        @test tail > 1e-12
    end

    @testset "degenerate inputs do not throw" begin
        @test first(RC._tail_rank(_mklist(Float64[]), 1.0, 1e-3, 4)) == 0
        @test first(RC._tail_rank(_mklist([1.0]), 1.0, 1e-3, 0)) == 0
    end

end

@testset "two-spin entangling step: DD finds |1> on both sides, exact to roundoff" begin
    set_symmetry!(:U1)
    mpo = xxz_mpo(2; J = 1.0, delta = 0.0)
    psi = product_state([:up, :down])
    t = 0.3
    info = cbe_bug_step!(psi, mpo, -im * t; tol_basis = 1e-10, tol_trunc = 1e-14,
                         tol_expv = 1e-13, maxdim = 4)
    @test info.record.policy === :dd_qr_streamed
    a = abs(overlap(product_state([:up, :down]), psi)) / norm(psi)
    b = abs(overlap(product_state([:down, :up]), psi)) / norm(psi)
    @test abs(a - cos(t / 2)) < 1e-10
    @test abs(b - sin(t / 2)) < 1e-10
end
