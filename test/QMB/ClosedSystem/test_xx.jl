# The two U(1) XX models
# (s = 1/2, d = 2 and s = 3, d = 7) on one exchange convention. d = 2s + 1 sectors, s = 1/2 is
# `xxz_mpo(delta = 0)`, <H> = 0 and the exact variance on product states (the normalisation of the
# exchange at both spins). The entangled comparison state is made by `cbe_bug_step!`: the check
# compares two operators on one state, so any entangled state does.

using Test, LinearAlgebra
import Random
using BondUpdateGalerkin.BUG, BondUpdateGalerkin.QMB

"""
`<H_s^2>` on the product state `2 m_i = twoms[i]` (`J = 1`): each bond's two flips give states
orthogonal to every other bond's, so the variance is the sum of the squared flip amplitudes,
`|<m+1|S^+|m>|^2 = s(s+1) - m(m+1)`.
"""
function _product_variance(twoms, twoS)
    s = twoS / 2
    r(m) = s * (s + 1) - m * (m + 1)
    return sum(
        r(twoms[b] / 2) * r(-twoms[b+1] / 2) + r(-twoms[b] / 2) * r(twoms[b+1] / 2) for
        b = 1:(length(twoms)-1)
    ) / 4
end

_wall(L, twoS) = [i <= L ÷ 2 ? twoS : -twoS for i = 1:L]

@testset "the two XX models" begin
    set_symmetry!(:U1)
    @test spin_space(1) === local_space(:U1)
    rng = Random.MersenneTwister(7)
    for twoS in (1, 6)
        q = spin_space(twoS)
        @test leg_dim(q.I, 1) == twoS + 1 && length(q.I.spaces[1]) == twoS + 1   # d = 2s + 1
        L = 5
        mpo = spin_xx_mpo(L, twoS)
        for twoms in (_wall(L, twoS), rand(rng, twoS:-2:-twoS, L), rand(rng, twoS:-2:-twoS, L))
            psi = sector_product_state(q, [((m,),) for m in twoms])
            @test spin_sz_profile(psi, twoS) ≈ twoms ./ 2 atol = 1e-14
            @test abs(mpo_energy(psi, mpo)) < 1e-13
            @test mpo_variance(psi, mpo) ≈ _product_variance(twoms, twoS) atol = 1e-12
        end
        # Var_0(H_s) = J^2 s^2 on the L = 16 domain wall: only the central bond acts
        @test mpo_variance(spin_domain_wall_state(16, twoS), spin_xx_mpo(16, twoS)) ≈ (twoS / 2)^2 atol =
            1e-12
        @test_throws ArgumentError sector_product_state(q, [((twoS + 2,),)])
    end
    # s = 1/2 is the package's own chain and state
    @test abs(overlap(spin_domain_wall_state(8, 1), domain_wall_state(8))) ≈ 1 atol = 1e-14
    chain_length = 8
    xx_reference = xxz_mpo(chain_length; J = 1.0, delta = 0.0)
    psi = neel_state(chain_length)
    for _ = 1:3
        cbe_bug_step!(psi, xx_reference, ComplexF64(-0.2im); maxdim = 64)
    end
    @test maximum(bond_dims(psi)) > 1                     # the comparison state is entangled
    @test mpo_energy(psi, spin_xx_mpo(chain_length, 1)) ≈ mpo_energy(psi, xx_reference) atol = 1e-12
    @test mpo_variance(psi, spin_xx_mpo(chain_length, 1)) ≈ mpo_variance(psi, xx_reference) atol =
        1e-12
end
