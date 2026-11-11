# `cbe_bug_step!`'s `work_ceiling_rule = :bond` (the default): every bond's DD-QR CBE and local
# Krylov stage is capped at `min(ceil(f*d*D_b), maxdim)` of that bond, its dimension at the start of
# the step, with `f = bond_ceiling_factor` (default 1.2); an explicit `f = 0` is `min(2*d*D_b,
# maxdim)`. `:maxdim` is one `max(d*maxdim, d*D_max)` for every bond. Sized to run in seconds
# (Heisenberg, L = 8, U(1), exact selection).
#
#   - `bond_work_ceilings` values, never 0; `_cap_at` reads one bond's ceiling from either form;
#   - from a product state the frames and bonds stay within ceil(1.2 d) by default, 2d at f = 0,
#     and D_work is the largest ceiling;
#   - from an entangled state, orders 2 and 3: the default is `:bond` at 1.2 bitwise, every frame
#     within its bond's ceiling, a stage no wider than under the global rule;
#   - the maxdim side binding, and the factor 1.2 against 2d;
#   - an unknown rule, and a factor without a per-bond rule or not above 1, are refused.

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

    @testset "from a product state every bond stays within its ceiling (factor $factor)" for (
        factor,
        rule_keywords,
    ) in ((1.2, (;)), (0.0, (work_ceiling_rule = :bond, bond_ceiling_factor = 0.0)))
        state = neel_state(chain_length)
        start_bonds = bond_dims(state)
        @test all(==(1), start_bonds)                         # fixture guard: a product state
        info = cbe_bug_step!(state, hamiltonian, time_step; step_settings..., order = 2, rule_keywords...)
        ceilings = bond_work_ceilings(start_bonds, site_dim, 64; factor = factor)
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

    # the older per-bond rule, 2*d*D_b
    twice_site = (work_ceiling_rule = :bond, bond_ceiling_factor = 0.0)
    for order in (2, 3)
        @testset "order $order: per-bond vs global ceiling" begin
            default_rule = ceiling_step(; order = order)
            explicit_default =
                ceiling_step(; order = order, work_ceiling_rule = :bond, bond_ceiling_factor = 1.2)
            global_rule = ceiling_step(; order = order, work_ceiling_rule = :maxdim)
            per_bond = ceiling_step(; order = order, twice_site...)
            # `:bond` at 1.2 is the default, bitwise
            @test norm(dense_state(explicit_default.state) - dense_state(default_rule.state)) == 0
            @test all(per_bond.widths .<= ceilings)
            @test all(bond_dims(per_bond.state) .<= ceilings)
            @test per_bond.stage <= global_rule.stage
        end
    end

    @testset "maxdim binds: every frame within min(2*d*D_b, maxdim)" begin
        small_maxdim = max(maximum(start_bonds), 2)
        capped_ceilings = bond_work_ceilings(start_bonds, site_dim, small_maxdim)
        @test any(capped_ceilings .< ceilings)                # fixture guard: the cap binds
        capped = ceiling_step(; order = 2, maxdim = small_maxdim, twice_site...)
        @test all(capped.widths .<= capped_ceilings)
        @test all(bond_dims(capped.state) .<= small_maxdim)
    end

    @testset "bond_ceiling_factor = 1.2: every frame within ceil(1.2*d*D_b)" begin
        tight_ceilings = bond_work_ceilings(start_bonds, site_dim, 64; factor = 1.2)
        @test all(tight_ceilings .< ceilings)                 # fixture guard: tighter everywhere
        for order in (2, 3)
            per_bond = ceiling_step(; order = order, twice_site...)
            tight =
                ceiling_step(; order = order, work_ceiling_rule = :bond, bond_ceiling_factor = 1.2)
            @test all(tight.widths .<= tight_ceilings)
            @test all(bond_dims(tight.state) .<= tight_ceilings)
            @test tight.stage <= per_bond.stage
        end
    end

    @testset "2D_b frame cap and ceil(1.2D_root) Galerkin input cap" begin
        @test bond_frame_root_ceilings([1,5,10,65], 1024, 2) == [2,6,20,130]
        @test bond_frame_root_ceilings([1,5,10,65], 1024, 3) == [2,10,12,130]
        @test bond_frame_root_ceilings([1,5,10,65], 64, 3) == [2,10,12,64]
        @test_throws ArgumentError bond_frame_root_ceilings([1,2,3], 64, 2;
            frame_factor=1.1, root_factor=1.2)
        @test_throws BoundsError bond_frame_root_ceilings([1,2,3], 64, 4)

        root_link = fld(chain_length, 2)
        split_caps = bond_frame_root_ceilings(start_bonds, 64, root_link)
        @test split_caps[root_link] ==
            min(64, ceil(Int, round(1.2*start_bonds[root_link];digits=9)))
        for b in eachindex(start_bonds)
            b == root_link && continue
            @test split_caps[b] == min(64,2*start_bonds[b])
        end
        for order in (2, 3)
            state = copy(start_state)
            info = cbe_bug_step!(state,hamiltonian,time_step;
                step_settings...,order=order,
                work_ceiling_rule=:frame_root,
                bond_ceiling_factor=2.0,
                root_ceiling_factor=1.2,
                closure_checks=true)
            @test info.root == root_link
            @test all(info.expanded .<= split_caps)
            @test info.expanded[info.root] <= split_caps[root_link]
            @test info.root_right <= split_caps[root_link]
            @test all(bond_dims(state) .<= split_caps)
            @test all(entry.width <= split_caps[entry.bond]
                      for entry in info.ledger.bond_log)
            # Root S tensor uses W and Z INPUT widths, not just the SVD rank.
            @test info.expanded[info.root]*info.root_right <= split_caps[root_link]^2
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
        # a factor needs a per-bond rule to scale, and must widen the bond
        @test_throws ArgumentError refused(; work_ceiling_rule = :maxdim, bond_ceiling_factor = 1.2)
        @test_throws ArgumentError refused(; work_ceiling_rule = :bond, bond_ceiling_factor = 1.0)
    end
end
