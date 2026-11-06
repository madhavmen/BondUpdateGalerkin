# Reachable sectors, and the P_perp check that K1 lies inside U0's range.

using Test, LinearAlgebra, LurCGT, Telum
using BondUpdateGalerkin.BUG, BondUpdateGalerkin.QMB

function mps_sum(a::SymMPS, b::SymMPS)
    L = length(a)
    ts = Any[]
    for i = 1:L
        dims = i == 1 ? (3,) : (i == L ? (1,) : (1, 3))
        push!(ts, TLArray(oplus(a[i], b[i], dims)))
    end
    return SymMPS(ts, L)
end

@testset "sector enumeration" begin

    @testset "reachable sectors of link(x)site cover the fused charges" begin
        psi = domain_wall_state(4)
        canonical!(psi, 1)
        f = bond_frame(psi, 1)
        secs = reachable_sectors(f.U0, 1, 2)
        # `dim(f.link_l)` is impossible -- a TLIndex carries no dimension.
        @test sum(d for (_, d) in secs) == leg_dim(f.U0, 1) * leg_dim(f.U0, 2)
        @test length(unique(first.(secs))) == length(secs)   # no duplicate charges
    end

    @testset "hand-rolled charge arithmetic agrees with Telum's own fusion" begin
        # Must cover links carrying a NON-ZERO charge. With a vacuum link the
        # reachable set is {+q,-q}, which is its own dual, so a spurious extra
        # dual still passes -- that false pass hid a real sign bug (job 93413).
        saw_charged_link = Ref(false)
        for psi in (
            domain_wall_state(4),
            neel_state(6),
            mps_sum(
                product_state([:up, :down, :up, :down]),
                product_state([:down, :up, :up, :down]),
            ),
        )
            for i = 1:(length(psi)-1)
                canonical!(psi, i)
                f = bond_frame(psi, i)
                t = f.U0
                viaTelum = reachable_sectors(t, 1, 2)
                mine = fuse_spaces(symm(t), Telum.spaces(t)[1], Telum.inds(t)[1].dir, Telum.spaces(t)[2], Telum.inds(t)[2].dir)
                any(q != dual_charge(symm(t), q) for (q, _) in Telum.spaces(t)[1]) &&
                    (saw_charged_link[] = true)
                @test sort(collect(viaTelum), by = string) == sort(collect(mine), by = string)
            end
        end
        @test saw_charged_link[]
    end

    @testset "K1 inside U0's range has no component outside it" begin
        # K1 = U0*S0 lies entirely in U0's column space, so P_perp K1 == 0.
        psi = neel_state(6)
        for i = 1:5
            canonical!(psi, i)
            f = bond_frame(psi, i)
            K1 = TLArray(f.U0 * f.S0)
            @test norm(perp_component(f.U0, K1)) < 1e-12
        end
    end
end
