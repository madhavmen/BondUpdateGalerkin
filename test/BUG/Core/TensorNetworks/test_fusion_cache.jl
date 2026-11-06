using Test, LinearAlgebra, LurCGT, Telum
import BondUpdateGalerkin
using BondUpdateGalerkin.BUG, BondUpdateGalerkin.QMB
const BUGCORE = BondUpdateGalerkin.BUG

# ── WHY THIS FILE EXISTS ─────────────────────────────────────────────────────────────────────
#
# `fusion_basis` was 45% of a fused d=4 step, recomputing a PURE function. Flat profile of one
# truncating step (job 16342051 stage 3, L=20, chi=32, d=4), 57598 samples in the step:
#
#     svd_std 68.6% | cbe_expand 50.1% | sector_graded_sketch 48.1% | fusion_basis 44.8%
#     gesdd! (LAPACK, SELF) 40.7%      getIdentity 0.05%
#
# i.e. building the isometry is free and taking an SVD OF IT is nearly half the step. The result
# depends only on the legs' spaces, arrows, tags and prime levels, never on the tensor's data, so
# it is now memoised.
#
# The defect was invisible for a whole campaign, and that is what these tests guard. Nothing
# measured allocation or call counts, so a step that churned 28.5 GiB at bond dimension 32 looked
# exactly like a step that was simply "slow", and the L=40 production runs were written off as
# expensive rather than broken. Correctness gates alone would never have caught it -- every number
# the campaign printed was RIGHT, just unaffordable.

norm_diff(a, b) = sqrt(abs(tensor_inner(TLArray(a - b), TLArray(a - b))))

# Bond frames spanning several states, deliberately including links that carry a NON-ZERO charge.
# A vacuum link is a FALSE PASS here: its reachable set {+q,-q} is its own dual, so a wrong
# arrow or a key that drops the itag still agrees. That exact false pass hid a real sign bug
# (jobs 93413, 93458), and it would equally hide a colliding cache key.
function sample_frames()
    out = Any[]
    charged = Ref(false)
    for psi in
        (domain_wall_state(4), neel_state(6), product_state([:up, :down, :down, :up, :up, :down]))
        for i = 1:(length(psi)-1)
            canonical!(psi, i)
            t = bond_frame(psi, i).U0
            any(q != dual_charge(symm(t), q) for (q, _) in Telum.spaces(t)[1]) && (charged[] = true)
            push!(out, t)
        end
    end
    return out, charged[]
end

@testset "fusion_basis memo" begin
    set_symmetry!(:U1)

    @testset "a cache HIT returns exactly what a cold MISS returns" begin
        frames, saw_charged = sample_frames()
        @test saw_charged            # the test is vacuous on vacuum links only

        # Cold: every call is a miss, computed against an empty cache.
        cold = map(frames) do t
            clear_fusion_cache!()
            fusion_basis(t, 1, 2; tag = "cbeG")
        end

        # Warm: one shared cache across all of them, so later calls hit -- and a key that fails to
        # discriminate two different frames shows up here as a mismatch, not as a crash.
        clear_fusion_cache!()
        warm = map(t -> fusion_basis(t, 1, 2; tag = "cbeG"), frames)

        st = fusion_stats()
        # The cache must actually have hit, or everything below passes for the wrong reason.
        @test st.calls == length(frames)
        @test st.misses < st.calls

        for (c, w) in zip(cold, warm)
            @test Telum.spaces(c) == Telum.spaces(w)
            @test string.(Telum.inds(c)) == string.(Telum.inds(w))
            @test norm_diff(c, w) < 1e-12
        end
    end

    @testset "the tag is part of the key" begin
        frames, _ = sample_frames()
        t = first(frames)
        clear_fusion_cache!()
        a = fusion_basis(t, 1, 2; tag = "cbeF")
        b = fusion_basis(t, 1, 2; tag = "cbeG")
        # Two distinct tags must not share an entry: the fused leg is labelled with the tag, and
        # handing back the wrong one is how `cbe_bug` broke under U(1) before ("Duplicate TLIndex
        # with non-empty itag"). Two keys, and the labels genuinely differ.
        @test fusion_stats().misses == 2
        @test string(Telum.inds(a)[3]) != string(Telum.inds(b)[3])
    end

    @testset "the symmetry mode is part of the key" begin
        clear_fusion_cache!()
        set_symmetry!(:U1)
        fusion_basis(bond_frame(canonical!(neel_state(4), 1), 1).U0, 1, 2; tag = "cbeG")
        n1 = fusion_stats().keys
        set_symmetry!(:none)
        fusion_basis(bond_frame(canonical!(neel_state(4), 1), 1).U0, 1, 2; tag = "cbeG")
        # A :none tensor must never be served a :U1 entry -- the sector structure differs.
        @test fusion_stats().keys > n1
        set_symmetry!(:U1)
    end

    # The key holds the legs' exact spaces, so an adaptive run adds keys nearly every step; without
    # a bound the memo grows with the run. The budget empties it when an insert would pass it.
    @testset "the memo stays inside its element budget" begin
        frames, _ = sample_frames()
        nel(t) = sum(length(r) for r in t.RMTs; init = 0)
        cold = map(frames) do t
            clear_fusion_cache!()
            fusion_basis(t, 1, 2; tag = "cbeG")
        end
        # the stored count is exact: the distinct keys' elements
        clear_fusion_cache!()
        old = set_fusion_budget!(nothing)
        try
            seen = Dict{Any,Int}()
            for (t, c) in zip(frames, cold)
                fusion_basis(t, 1, 2; tag = "cbeG")
                seen[(BUGCORE._leg_key(t, 1), BUGCORE._leg_key(t, 2))] = nel(c)
            end
            st = fusion_stats()
            @test st.keys == length(seen) && st.elems == sum(values(seen))
            full = st.elems
            # a budget below the total: the memo is emptied and never holds more than the budget
            cap = maximum(nel, cold) + 1
            @test cap < full                               # the bound is exercised
            clear_fusion_cache!()
            @test set_fusion_budget!(cap) === nothing
            warm = map(frames) do t
                b = fusion_basis(t, 1, 2; tag = "cbeG")
                @test fusion_stats().elems <= cap
                b
            end
            @test fusion_stats().keys < length(seen)
            for (c, w) in zip(cold, warm)                  # eviction never changes a result
                @test Telum.spaces(c) == Telum.spaces(w)
                @test norm_diff(c, w) < 1e-12
            end
            # `nothing` is the unbounded memo
            set_fusion_budget!(nothing)
            clear_fusion_cache!()
            foreach(t -> fusion_basis(t, 1, 2; tag = "cbeG"), frames)
            @test fusion_stats().elems == full
            @test_throws ArgumentError set_fusion_budget!(0)
        finally
            set_fusion_budget!(old)
        end
    end
end
