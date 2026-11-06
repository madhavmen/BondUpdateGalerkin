# `cbe_bug_step!`'s `frontier_tol`: each closure frontier is compressed to its own rank before its
# sweep instead of being swept at frame width. Sized to run in seconds (Heisenberg, L = 6, U(1)).
#
# With exact selection no random numbers are drawn, so:
#   - a step whose frontiers are compressed at 1e-12 returns the uncompressed step's state, and both
#     match exp(tau H) as closely;
#   - the compressed step with parallel half-sweeps equals the serial one to round-off: the two
#     halves sweep on private side caches, so neither reads the other's environment stacks;
#   - a negative tolerance is refused, not read as "off".

using Test, LinearAlgebra
using BondUpdateGalerkin.BUG, BondUpdateGalerkin.QMB

include(joinpath(@__DIR__, "..", "..", "..", "common", "dense_reference.jl"))

@testset "frontier compression (frontier_tol)" begin
    set_symmetry!(:U1)
    clear_fusion_cache!()
    chain_length = 6
    hamiltonian = xxz_mpo(chain_length; J = 1.0, delta = 1.0)
    time_step = ComplexF64(-0.2im)
    step_settings = (order = 3, tol_basis = 1e-8, tol_trunc = 1e-12, maxdim = 64, exact = true)

    # entangle the Neel state first, so the frames are wider than the frontiers' own rank
    start_state = neel_state(chain_length)
    for _ = 1:2
        cbe_bug_step!(start_state, hamiltonian, time_step; step_settings..., order = 2)
    end
    start_vector = dense_state(start_state)
    start_norm = norm(start_vector)
    exact_vector = exp(time_step * Matrix(dense_heisenberg(chain_length; J = 1.0, delta = 1.0))) *
                   start_vector

    plain_state, compressed_state = copy(start_state), copy(start_state)
    cbe_bug_step!(plain_state, hamiltonian, time_step; step_settings..., parallel = false)
    cbe_bug_step!(
        compressed_state,
        hamiltonian,
        time_step;
        step_settings...,
        parallel = false,
        frontier_tol = 1e-12,
    )
    plain_vector, compressed_vector = dense_state(plain_state), dense_state(compressed_state)
    @test norm(compressed_vector - plain_vector) < 1e-9 * start_norm
    plain_error = norm(plain_vector - exact_vector) / start_norm
    compressed_error = norm(compressed_vector - exact_vector) / start_norm
    @test compressed_error <= plain_error + 1e-9

    # parallel half-sweeps on the compressed frontiers: the same method, so the same state
    parallel_state = copy(start_state)
    cbe_bug_step!(
        parallel_state,
        hamiltonian,
        time_step;
        step_settings...,
        parallel = true,
        frontier_tol = 1e-12,
    )
    parallel_vector = dense_state(parallel_state)
    fidelity = abs(dot(parallel_vector, compressed_vector)) /
               (norm(parallel_vector) * norm(compressed_vector))
    @test 1 - fidelity < 1e-12
    @test bond_dims(parallel_state) == bond_dims(compressed_state)

    @test_throws ArgumentError cbe_bug_step!(
        copy(start_state),
        hamiltonian,
        time_step;
        step_settings...,
        frontier_tol = -1.0,
    )
end
