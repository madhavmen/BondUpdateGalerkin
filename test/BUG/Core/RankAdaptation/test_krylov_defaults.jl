# The XX defaults of `cbe_bug_step!` and the local Krylov recurrence that picks Lanczos or Arnoldi:
#   - `DDQRExpansion()` is the single-sketch `:rank_tail` selection with 8 columns and
#     `krylov_mode = :auto`;
#   - `resolve_krylov_recurrence`: `:auto` is Lanczos on a Hermitian step and Arnoldi otherwise, an
#     explicit recurrence is kept, an unknown one refused; an unresolved `:auto` runs Arnoldi;
#   - `_krylov_stop`, the stopping rule both recurrences share (adaptive and fixed depth);
#   - the defaults reach the step: a default step is bitwise the explicit-Lanczos step, with
#     `hermitian = false` bitwise the explicit-Arnoldi step, and every frame stays within the default
#     per-bond ceiling `ceil(1.2 d D_b)`.
# U(1) XX chain, L = 8; seconds.

using Test, LinearAlgebra, Random
using BondUpdateGalerkin.BUG, BondUpdateGalerkin.QMB
import BondUpdateGalerkin.BUG: resolve_krylov_recurrence, krylov_controls, _krylov_stop

include(joinpath(@__DIR__, "..", "..", "..", "common", "dense_reference.jl"))

@testset "XX defaults and the local Krylov recurrence" begin
    @testset "DDQRExpansion defaults" begin
        default_policy = DDQRExpansion()
        @test default_policy.mode === :rank_tail
        @test default_policy.fixed_width == 8
        @test default_policy.tail_factor == 0.10
        @test default_policy.krylov_mode === :auto
        @test default_policy.krylov_fixed_depth == 0          # the adaptive stop
        @test krylov_controls(default_policy).recurrence === :arnoldi   # unresolved: any generator
    end

    @testset "resolve_krylov_recurrence" begin
        @test resolve_krylov_recurrence(DDQRExpansion(), true).krylov_mode === :lanczos
        @test resolve_krylov_recurrence(DDQRExpansion(), false).krylov_mode === :arnoldi
        explicit_arnoldi = DDQRExpansion(krylov_mode = :arnoldi)
        @test resolve_krylov_recurrence(explicit_arnoldi, true) === explicit_arnoldi
        # every other field survives the resolution
        resolved = resolve_krylov_recurrence(DDQRExpansion(mode = :fixed, fixed_rank = 3), true)
        @test (resolved.mode, resolved.fixed_width, resolved.fixed_rank) == (:fixed, 8, 3)
        @test_throws ArgumentError resolve_krylov_recurrence(DDQRExpansion(krylov_mode = :gmres), true)
    end

    @testset "_krylov_stop" begin
        @test _krylov_stop(1e-3, 1e-2, 1, 0) == (true, true)       # adaptive: within the allowance
        @test _krylov_stop(1e-1, 1e-2, 1, 0) == (false, false)     # adaptive: go on
        @test _krylov_stop(1e-3, 1e-2, 2, 4) == (false, false)     # fixed depth not reached yet
        @test _krylov_stop(1e-3, 1e-2, 4, 4) == (true, true)
        @test _krylov_stop(1e-1, 1e-2, 4, 4) == (true, false)      # never claims a missed tolerance
    end

    @testset "the defaults reach the step" begin
        set_symmetry!(:U1)
        clear_fusion_cache!()
        chain_length, site_dim = 8, 2
        hamiltonian = xxz_mpo(chain_length; J = 1.0, delta = 0.0)
        time_step = ComplexF64(-0.1im)
        # an entangled start, so every stage of the step has work
        start_state = domain_wall_state(chain_length)
        for _ = 1:3
            cbe_bug_step!(start_state, hamiltonian, time_step; order = 2)
        end
        function one_step(; keywords...)
            state = copy(start_state)
            info = cbe_bug_step!(
                state,
                hamiltonian,
                time_step;
                order = 2,
                tol_basis = 1e-4,
                rng = MersenneTwister(7),
                keywords...,
            )
            return (state = state, info = info)
        end
        default_step = one_step()
        lanczos_step = one_step(expansion = DDQRExpansion(krylov_mode = :lanczos))
        @test norm(dense_state(default_step.state) - dense_state(lanczos_step.state)) == 0
        non_hermitian_default = one_step(hermitian = false)
        non_hermitian_arnoldi =
            one_step(hermitian = false, expansion = DDQRExpansion(krylov_mode = :arnoldi))
        @test norm(dense_state(non_hermitian_default.state) - dense_state(non_hermitian_arnoldi.state)) == 0
        # the default per-bond ceiling: every frame within min(ceil(1.2 d D_b), maxdim)
        ceilings = bond_work_ceilings(bond_dims(start_state), site_dim, 128; factor = 1.2)
        @test all(default_step.info.expanded .<= ceilings)
        @test all(entry.width <= ceilings[entry.bond] for entry in default_step.info.ledger.bond_log)
        @test all(bond_dims(default_step.state) .<= ceilings)
    end
end
