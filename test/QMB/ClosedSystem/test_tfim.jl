# The TFIM operator against a dense TFIM, the two symmetries, and the refusals. The ZZ coupling is
# pinned with no integrator, by the MPO's off-diagonal matrix elements between sigma^x product
# states (`mpo_matrix_element`).
#
#     H = -J Σ Z_i Z_{i+1} - h Σ X_i
#
# The reference is a dense `2^L` TFIM built from the Pauli algebra, sharing no code with the
# tensor stack.
#
# The basis is σˣ: `X` is DIAGONAL and `Z` is the FLIP. Reading it the usual way describes a
# different Hamiltonian -- see the header of `src/QMB/Models/ClosedSystem/tfim.jl`.

using Test, LinearAlgebra
using BondUpdateGalerkin.BUG, BondUpdateGalerkin.QMB

"Dense TFIM in the sigma^x product basis: site i <-> bit (i-1), bit CLEAR = |+>."
function _tfim_dense_ref(L::Int; J::Float64 = 1.0, h::Float64 = 1.0)
    D = 2^L
    H = zeros(ComplexF64, D, D)
    for s = 0:(D-1)
        d = 0.0
        for i = 0:(L-1)
            d += ((s >> i) & 1) == 0 ? 1.0 : -1.0
        end
        H[s+1, s+1] = -h * d
        for i = 0:(L-2)
            t = s ⊻ (1 << i) ⊻ (1 << (i + 1))
            H[t+1, s+1] += -J
        end
    end
    return H
end

"A sigma^x product state as a dense vector, in `_tfim_dense_ref`'s convention."
function _x_product_dense(dirs::Vector{Symbol})
    v = zeros(ComplexF64, 2^length(dirs))
    s = 0
    for (i, d) in enumerate(dirs)
        d === :minus && (s |= 1 << (i - 1))
    end
    v[s+1] = 1.0
    return v
end

"`dirs` with the spins on sites `i` and `i + 1` flipped: what one `Z_i Z_{i+1}` term does."
_flip_pair(dirs, i) =
    [j in (i, i + 1) ? (d === :plus ? :minus : :plus) : d for (j, d) in enumerate(dirs)]

@testset "tfim_mpo" begin
    L, J, h = 6, 1.0, 0.7
    HD = _tfim_dense_ref(L; J = J, h = h)

    # Asymmetric states are the point. A state symmetric under reversal is a fixed point of a
    # MIRRORED site convention, so it passes whether or not the two orderings agree.
    cases = [
        "polarised" => fill(:plus, L),
        "kink" => [i <= L ÷ 2 ? :plus : :minus for i = 1:L],
        "asymmetric" => [:plus, :minus, :minus, :plus, :minus, :plus],
        "one flip" => [i == 2 ? :minus : :plus for i = 1:L],
    ]

    @testset "the operator, under $sym" for sym in (:none, :Z2)
        set_symmetry!(sym)
        W = tfim_mpo(L; J = J, h = h)

        # Virtual dimension 3 on EVERY internal link, independent of L: `id`, `open`, `done`.
        @test length(W) == L
        @test mpo_virtual_dims(W) == [1; fill(3, L - 1); 1]

        for (name, dirs) in cases
            psi = ising_product_state(dirs)
            v = _x_product_dense(dirs)
            @test real(mpo_energy(copy(psi), W)) ≈ real(dot(v, HD * v)) atol = 1e-10
        end

        # The analytic pin: <+...+|H|+...+> = -h*L for ANY J, because `Z_i Z_{i+1}` is purely
        # off-diagonal in this basis. A dropped boundary field breaks it.
        @test real(mpo_energy(ising_polarised_state(L), W)) ≈ -h * L atol = 1e-10
        @test !isapprox(real(mpo_energy(ising_polarised_state(L), W)), -h * (L - 2); atol = 1e-6)

        # THE COUPLING. Every energy above is blind to it (`<s|Z_i Z_{i+1}|s> = 0` on a sigma^x
        # product state), so read it off the matrix elements between a state and the state with
        # one adjacent pair flipped: exactly `-J`, and zero for a pair the operator cannot reach.
        for (name, dirs) in cases, bond = 1:(L-1)
            flipped = _flip_pair(dirs, bond)
            element = mpo_matrix_element(ising_product_state(flipped), W, ising_product_state(dirs))
            @test element ≈ dot(_x_product_dense(flipped), HD * _x_product_dense(dirs)) atol = 1e-10
        end
        far_flip = [j in (1, 3) ? :minus : :plus for j = 1:L]       # sites 1 and 3: no ZZ term
        @test abs(mpo_matrix_element(ising_product_state(far_flip), W, ising_polarised_state(L))) <
              1e-12
    end

    @testset "the two symmetries describe one operator" begin
        # Same numbers, different block structure. A gap here is charge bookkeeping, not physics.
        e = Dict{Symbol,Vector{Float64}}()
        for sym in (:none, :Z2)
            set_symmetry!(sym)
            W = tfim_mpo(L; J = J, h = h)
            e[sym] = [real(mpo_energy(ising_product_state(d), W)) for (_, d) in cases]
        end
        @test maximum(abs.(e[:none] - e[:Z2])) < 1e-10
    end

    @testset "unsupported symmetries are refused, not approximated" begin
        # `Z_i Z_{i+1}` does not conserve S^z and the field picks an axis, so there is no U(1) or
        # SU(2) form.
        for sym in (:U1, :SU2)
            set_symmetry!(sym)
            @test_throws ArgumentError tfim_mpo(L; J = J, h = h)
        end
        set_symmetry!(:Z2)
        @test_throws ArgumentError tfim_mpo(1; J = J, h = h)
    end
    set_symmetry!(:U1)
end
