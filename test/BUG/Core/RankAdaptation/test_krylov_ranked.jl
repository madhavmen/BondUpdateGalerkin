# `cbe_bug_step!`'s `krylov_rank_tol`: each local Krylov stage ranks the directions its chain finds
# by their weight in the step and keeps only the top ones, instead of appending them in chain order
# up to the work ceiling. Sized to run in seconds (Heisenberg, L = 8, U(1), exact selection).
#
#   - `krylov_rank = 0` (every constructor without it) is the sequential rule;
#   - order 2 and 3 steps: a tight rank
#     tolerance is as accurate as the sequential step, a loose one gives a stage frame no wider and
#     is still accurate to the looser share;
#   - a negative tolerance is refused, not read as "off".

using Test, LinearAlgebra
using BondUpdateGalerkin.BUG, BondUpdateGalerkin.QMB

include(joinpath(@__DIR__, "..", "..", "..", "common", "dense_reference.jl"))

@testset "ranked Krylov selection (krylov_rank_tol)" begin
    @testset "krylov_rank = 0 is the default" begin
        @test BasisAllocation(1.0, 1.0, 1.0, 1e-6, 1.0).krylov_rank == 0.0
        @test BasisAllocation(1.0, 1.0, 1.0, 1e-6, 1.0, :linear).krylov_rank == 0.0
        budget = AdaptiveBudget(1e-8, 1e-10, 1e-10)
        @test basis_allocation(budget, 9).krylov_rank == 0.0
        # apportioned like `tol_krylov`: the same share of its own tolerance
        allocation = basis_allocation(budget, 9; rule = :linear, krylov = 1e-6, krylov_rank = 1e-6)
        @test allocation.krylov_rank ≈ allocation.krylov
        @test_throws ArgumentError basis_allocation(budget, 9; krylov_rank = -1.0)
    end

    set_symmetry!(:U1)
    clear_fusion_cache!()
    chain_length = 8
    hamiltonian = xxz_mpo(chain_length; J = 1.0, delta = 1.0)
    time_step = ComplexF64(-0.2im)
    step_settings =
        (tol_basis = 1e-8, tol_root = 1e-12, tol_trunc = 1e-12, maxdim = 64, exact = true)

    # entangle the Neel state first, so the local Krylov stages have directions to rank
    start_state = neel_state(chain_length)
    for _ = 1:2
        cbe_bug_step!(start_state, hamiltonian, time_step; step_settings..., order = 2)
    end
    start_vector = dense_state(start_state)
    start_norm = norm(start_vector)
    exact_vector =
        exp(time_step * Matrix(dense_heisenberg(chain_length; J = 1.0, delta = 1.0))) * start_vector

    # one step from `start_state`: (relative error against exp(tau H), D_stage)
    function step_error_and_stage(; keywords...)
        state = copy(start_state)
        info = cbe_bug_step!(state, hamiltonian, time_step; step_settings..., keywords...)
        return norm(dense_state(state) - exact_vector) / start_norm, info.record.D_stage
    end

    for order in (2, 3)
        @testset "order $order: sequential vs ranked" begin
            plain_error, plain_stage = step_error_and_stage(; order = order)
            tight_error, tight_stage = step_error_and_stage(; order = order, krylov_rank_tol = 1e-12)
            loose_error, loose_stage = step_error_and_stage(; order = order, krylov_rank_tol = 1e-6)
            @test plain_error < 1e-6                     # sanity: the reference arm is accurate
            @test tight_error <= 10 * plain_error + 1e-10
            @test loose_stage <= plain_stage
            @test loose_stage <= tight_stage
            @test loose_error < 1e-4
        end
    end

    @test_throws ArgumentError cbe_bug_step!(
        copy(start_state),
        hamiltonian,
        time_step;
        step_settings...,
        order = 2,
        krylov_rank_tol = -1.0,
    )
end
