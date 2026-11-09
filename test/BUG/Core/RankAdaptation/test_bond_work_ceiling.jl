# `cbe_bug_step!`'s `work_ceiling_rule = :bond`: every bond's DD-QR CBE and local Krylov stage is
# capped at `min(2*d*D_b, maxdim)` of that bond, its dimension at the start of the step, instead of
# one `max(d*maxdim, d*D_max)` for every bond; `bond_ceiling_factor = f` replaces `2*d*D_b` by
# `ceil(f*d*D_b)`. Sized to run in seconds (Heisenberg, L = 8, U(1), exact selection).
#
#   - `bond_work_ceilings` values, never 0; `_cap_at` reads one bond's ceiling from either form;
#   - from a product state the frames and bonds stay within 2d, and D_work is the largest ceiling;
#   - from an entangled state, orders 2 and 3: every frame within its bond's ceiling, a stage no
#     wider than under the global rule, the global rule bitwise the default;
#   - the maxdim side binding, and the factor 1.2;
#   - an unknown rule, and a factor without the per-bond rule or not above 1, are refused.

using Test, LinearAlgebra
using BondUpdateGalerkin.BUG, BondUpdateGalerkin.QMB
import BondUpdateGalerkin.BUG: _cap_at

include(joinpath(@__DIR__, "..", "..", "..", "common", "dense_reference.jl"))

@testset "per-bond work ceiling (work_ceiling_rule = :bond)" begin
    @testset "bond_work_ceilings" begin
        @test bond_work_ceilings([1, 3, 8], 2, 1024) == [4, 12, 32]
        @test bond_work_ceilings([0, 5], 4, 1024) == [1, 40]    # never 0, which admits nothing
        @test bond_work_ceilings([1, 3, 8, 40], 2, 16) == [4, 12, 16, 16]
        # ceil(f*d*D_b) scales with physical site dimension and avoids rounding artifacts.
        @test bond_work_ceilings([1, 5, 10, 65], 2, 1024; factor = 1.2) == [3, 12, 24, 156]
        @test bond_work_ceilings([1, 5, 10], 3, 1024; factor = 1.2) == [4, 18, 36]
        @test bond_work_ceilings([1, 5, 10], 4, 1024; factor = 1.2) == [5, 24, 48]
        @test bond_work_ceilings([0, 65], 2, 70; factor = 1.2) == [1, 70]
        @test _cap_at(1024, 3) == 1024
        @test _cap_at([2, 6, 16], 2) == 6
    end

    set_symmetry!(:U1)
    clear_fusion_cache!()
    chain_length, site_dim = 8, 2
    hamiltonian = xxz_mpo(chain_length; J = 1.0, delta = 1.0)
    time_step = ComplexF64(-0.2im)
    step_settings =
        (tol_basis = 1e-8, tol_root = 1e-12, tol_trunc = 1e-12, maxdim = 64, exact = true)

    @testset "from a product state every bond stays within 2d" begin
        state = neel_state(chain_length)
        start_bonds = bond_dims(state)
        @test all(==(1), start_bonds)                         # fixture guard: a product state
        info = cbe_bug_step!(
            state,
            hamiltonian,
            time_step;
            step_settings...,
            order = 2,
            work_ceiling_rule = :bond,
        )
        ceilings = bond_work_ceilings(start_bonds, site_dim, 64)
        @test all(info.expanded .<= ceilings)
        @test all(bond_dims(state) .<= ceilings)
        @test info.record.D_work == maximum(ceilings)
    end

    # entangle the Neel state first, so the bonds differ
    start_state = neel_state(chain_length)
    for _ = 1:2
        cbe_bug_step!(start_state, hamiltonian, time_step; step_settings..., order = 2)
    end
    start_bonds = bond_dims(start_state)
    ceilings = bond_work_ceilings(start_bonds, site_dim, 64)

    # one step from `start_state`: its frame widths, D_stage and the evolved state
    function ceiling_step(; keywords...)
        state = copy(start_state)
        info = cbe_bug_step!(state, hamiltonian, time_step; step_settings..., keywords...)
        return (widths = info.expanded, stage = info.record.D_stage, state = state)
    end

    for order in (2, 3)
        @testset "order $order: per-bond vs global ceiling" begin
            global_rule = ceiling_step(; order = order)
            explicit_global = ceiling_step(; order = order, work_ceiling_rule = :maxdim)
            per_bond = ceiling_step(; order = order, work_ceiling_rule = :bond)
            # the global rule is the default, bitwise
            @test norm(dense_state(explicit_global.state) - dense_state(global_rule.state)) == 0
            @test all(per_bond.widths .<= ceilings)
            @test all(bond_dims(per_bond.state) .<= ceilings)
            @test per_bond.stage <= global_rule.stage
        end
    end

    @testset "maxdim binds: every frame within min(2*d*D_b, maxdim)" begin
        small_maxdim = max(maximum(start_bonds), 2)
        capped_ceilings = bond_work_ceilings(start_bonds, site_dim, small_maxdim)
        @test any(capped_ceilings .< ceilings)                # fixture guard: the cap binds
        capped = ceiling_step(; order = 2, maxdim = small_maxdim, work_ceiling_rule = :bond)
        @test all(capped.widths .<= capped_ceilings)
        @test all(bond_dims(capped.state) .<= small_maxdim)
    end

    @testset "bond_ceiling_factor = 1.2: every frame within ceil(1.2*d*D_b)" begin
        tight_ceilings = bond_work_ceilings(start_bonds, site_dim, 64; factor = 1.2)
        @test all(tight_ceilings .< ceilings)                 # fixture guard: tighter everywhere
        for order in (2, 3)
            per_bond = ceiling_step(; order = order, work_ceiling_rule = :bond)
            tight =
                ceiling_step(; order = order, work_ceiling_rule = :bond, bond_ceiling_factor = 1.2)
            @test all(tight.widths .<= tight_ceilings)
            @test all(bond_dims(tight.state) .<= tight_ceilings)
            @test tight.stage <= per_bond.stage
        end
    end

    @testset "refused settings" begin
        refused(; keywords...) = cbe_bug_step!(
            copy(start_state),
            hamiltonian,
            time_step;
            step_settings...,
            order = 2,
            keywords...,
        )
        @test_throws ArgumentError refused(; work_ceiling_rule = :site)
        # a factor needs the per-bond rule to scale, and must widen the bond
        @test_throws ArgumentError refused(; bond_ceiling_factor = 1.2)
        @test_throws ArgumentError refused(; work_ceiling_rule = :bond, bond_ceiling_factor = 1.0)
    end
end
