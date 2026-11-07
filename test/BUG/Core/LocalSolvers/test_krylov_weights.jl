# The Krylov stopping weights on plain matrices, graded against a dense exponential:
#   - the local stage's weight is the residual integral `|tau| beta_k int_0^1 |f(s)| ds`: Saad's
#     scalar cancels on a 2x2 with an O(1) error, the integral does not, and it bounds the error of
#     random Hermitian generators;
#   - the recycled stage's block weight reduces to it with nothing recycled, damped steps included;
#   - the root's tail is guarded over the step (`_tail_sup`): the endpoint entry vanishes on a
#     gapped 3-site chain, the guarded tail does not; imaginary time keeps the endpoint value;
#   - `_tridiag_eigen` is exact on finite input and names non-finite input.

using Test, LinearAlgebra
import Random
import BondUpdateGalerkin.BUG:
    _kry_weight, _block_weight, _arnoldi_chain, _tail, _tail_sup, _tridiag_eigen, _coeffs_from_eigen

# Plain Arnoldi with two Gram-Schmidt passes, `H basis[:, 1:k] = basis[:, 1:k+1] hessenberg[1:k+1,
# 1:k]`: the test's own decomposition, so the weights are graded against a recurrence they do not
# share.
function arnoldi_decomposition(generator::AbstractMatrix, seed::AbstractVector, depth::Int)
    basis = zeros(ComplexF64, length(seed), depth + 1)
    hessenberg = zeros(ComplexF64, depth + 1, depth + 1)
    basis[:, 1] = seed / norm(seed)
    for step = 1:depth
        image = generator * basis[:, step]
        for _ = 1:2
            projection = basis[:, 1:step]' * image
            hessenberg[1:step, step] .+= projection
            image -= basis[:, 1:step] * projection
        end
        hessenberg[step+1, step] = norm(image)
        basis[:, step+1] = image / hessenberg[step+1, step]
    end
    return basis, hessenberg
end

# `||exp(tau H) v - basis_k exp(tau H_k) e1||` for a unit `v`: the depth-`k` Krylov error
function krylov_error(generator, seed, tau, basis, hessenberg, depth::Int)
    exact = exp(tau * generator) * seed
    reduced = exp(tau * hessenberg[1:depth, 1:depth])[:, 1]
    return norm(exact - basis[:, 1:depth] * reduced)
end

@testset "Krylov stopping weights" begin
    @testset "the residual integral does not cancel on a 2x2" begin
        generator = ComplexF64[2pi 1; 1 0]
        tau = ComplexF64(-1im)
        seed = ComplexF64[1, 0]
        basis, hessenberg = arnoldi_decomposition(generator, seed, 1)
        true_error = krylov_error(generator, seed, tau, basis, hessenberg, 1)
        weight = _kry_weight(hessenberg, 1, tau)
        @test true_error > 0.1                   # one direction is far from enough
        @test weight ≈ 1.0 rtol = 1e-10          # |tau| beta int_0^1 |exp(-2 pi i s)| ds
        @test weight >= true_error
        # the chain offers e2 at its true weight and runs to the breakdown at depth 2
        offered = Float64[]
        record = (vector, offered_weight, first) -> (push!(offered, offered_weight); true)
        chain = _arnoldi_chain(seed, vector -> generator * vector, tau, 1e-8, record)
        @test chain.closed && !chain.capped
        @test chain.n_h1 == 2
        @test only(offered) ≈ 1.0 rtol = 1e-10
    end

    @testset "the residual integral bounds the error of random Hermitian generators" begin
        rng = Random.MersenneTwister(11)
        dimension, depth = 30, 16
        checked = 0
        for _ = 1:3, tau in ComplexF64.((-0.1im, -1im, -3im))
            raw = randn(rng, ComplexF64, dimension, dimension)
            generator = (raw + raw') / 2
            seed = normalize(randn(rng, ComplexF64, dimension))
            basis, hessenberg = arnoldi_decomposition(generator, seed, depth)
            for step = 1:(depth-1)
                true_error = krylov_error(generator, seed, tau, basis, hessenberg, step)
                true_error < 1e-13 && break                  # past roundoff: nothing to bound
                checked += 1
                # rigorous in real time up to the Simpson rule's own error (~1e-5 relative)
                @test true_error <= (1 + 1e-3) * _kry_weight(hessenberg, step, tau) + 1e-14
            end
        end
        @test checked > 30
    end

    @testset "the recycled block weight is the fresh chain's with nothing recycled" begin
        # one residual column beta_k v_{k+1}: Gram = beta_k^2 e_k e_k', so the block weight must
        # equal `_kry_weight` in every normalisation, real time damped or not and imaginary time
        rng = Random.MersenneTwister(5)
        dimension, depth = 20, 6
        raw = randn(rng, ComplexF64, dimension, dimension)
        generator = (raw + raw') / 2
        seed = normalize(randn(rng, ComplexF64, dimension))
        _, hessenberg = arnoldi_decomposition(generator, seed, depth)
        residual_gram = zeros(ComplexF64, depth, depth)
        residual_gram[depth, depth] = abs2(hessenberg[depth+1, depth])
        for damping in (0.0, 0.1, 1.0, 10.0), state_seed in (true, false)
            tau = ComplexF64(-damping * 0.1 - 0.1im)
            fresh = _kry_weight(hessenberg, depth, tau; state_seed = state_seed)
            block = _block_weight(
                hessenberg[1:depth, 1:depth],
                residual_gram,
                tau;
                state_seed = state_seed,
            )
            @test block ≈ fresh rtol = 1e-10
        end
        imaginary = ComplexF64(-0.5)
        @test _block_weight(hessenberg[1:depth, 1:depth], residual_gram, imaginary) ≈
              _kry_weight(hessenberg, depth, imaginary) rtol = 1e-10
    end

    @testset "the root's tail is guarded over the step" begin
        # Lanczos from e1 on [0 pi 0; pi 0 1; 0 1 0]: T_2 = [0 pi; pi 0] has
        # [exp(-i T_2)]_{2,1} = -i sin(pi) = 0, so the endpoint stops at depth 2 with an O(1) error
        chain = ComplexF64[0 pi 0; pi 0 1; 0 1 0]
        tau = ComplexF64(-1im)
        eigen_t2 = _tridiag_eigen([0.0, 0.0], [Float64(pi)])
        evalcoeff(step) = _coeffs_from_eigen(eigen_t2.values, eigen_t2.vectors, ComplexF64(step))
        residual_norm = 1.0                                      # beta_2 = H[3, 2]
        endpoint = _tail(first(evalcoeff(tau)), residual_norm)
        guarded = _tail_sup(evalcoeff, tau, residual_norm)
        true_error = norm(exp(tau * chain) * ComplexF64[1, 0, 0] - vcat(first(evalcoeff(tau)), 0))
        @test true_error > 0.1
        @test endpoint < 1e-12
        @test guarded >= 0.9                                     # |sin(pi s)| at s = 1/2
        @test abs(tau) * guarded >= true_error                   # the real-time bound it samples
        # never below the endpoint; imaginary time and loss keep the endpoint value exactly
        for step in ComplexF64.((-0.3im, -2.7im))
            @test _tail_sup(evalcoeff, step, residual_norm) >=
                  _tail(first(evalcoeff(step)), residual_norm)
        end
        for step in ComplexF64.((-0.5, -0.5 - 1im))
            @test _tail_sup(evalcoeff, step, residual_norm) ==
                  _tail(first(evalcoeff(step)), residual_norm)
        end
    end

    @testset "_tridiag_eigen: exact on finite input, names non-finite input" begin
        diagonal = [0.3, -1.2, 2.5, 0.0, 1e-9]
        off_diagonal = [1.0, 1e-12, 0.7, 2.0]
        decomposition = _tridiag_eigen(diagonal, off_diagonal)
        dense = Matrix(SymTridiagonal(diagonal, off_diagonal))
        eigenvectors = decomposition.vectors
        @test norm(dense * eigenvectors - eigenvectors * Diagonal(decomposition.values)) < 1e-12
        @test norm(eigenvectors' * eigenvectors - I) < 1e-12
        @test_throws ArgumentError _tridiag_eigen([0.0, NaN], [1.0])
        @test_throws ArgumentError _tridiag_eigen([0.0, 1.0], [Inf])
    end
end
