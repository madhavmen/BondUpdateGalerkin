# Parallel against serial half-sweeps, sized to run in seconds: L = 6, two steps, and the ordinary, midpoint and order-2 schemes
# (order 4 is the same frontier code as order 2).
#
# `parallel = true` runs the two half-sweeps as two tasks. It must be the SAME METHOD as the serial
# sweep:
#   exact = true  no random numbers are drawn, so parallel and serial must agree to round-off,
#                 ledger verdict included -- any difference is a race or a shared-state bug.
#   RSVD          the right half-sweep draws from its own seeded stream when parallel, so the
#                 sketches differ; the states must agree to the tolerance, and repeated parallel
#                 runs must agree EXACTLY (independent tasks, not "usually wins the race").

using Test, LinearAlgebra
using BondUpdateGalerkin.BUG, BondUpdateGalerkin.QMB

@testset "adaptive BUG: parallel half-sweeps == serial" begin
    set_symmetry!(:U1)
    chain_length = 6
    hamiltonian = xxz_mpo(chain_length; J = 1.0, delta = 1.0)
    time_step = ComplexF64(-0.05im)
    tolerances = (tol_basis = 1e-7, tol_trunc = 1e-9, tol_expv = 1e-10, maxdim = 64)
    fidelity(state_a, state_b) = abs(overlap(state_a, state_b)) / (norm(state_a) * norm(state_b))
    schemes = (
        (
            "ordinary",
            (state; keywords...) -> cbe_bug_step!(state, hamiltonian, time_step; keywords...),
        ),
        (
            "midpoint",
            (state; keywords...) ->
                cbe_bug_midpoint_step!(state, hamiltonian, time_step; keywords...),
        ),
        (
            "order 2",
            (state; keywords...) ->
                cbe_bug_step!(state, hamiltonian, time_step; order = 2, keywords...),
        ),
    )
    for (scheme_name, step!) in schemes
        @testset "$scheme_name" begin
            # Exact selection: agreement to round-off, step by step verdicts included.
            serial_state, parallel_state = neel_state(chain_length), neel_state(chain_length)
            for _ = 1:2
                serial_info = step!(serial_state; parallel = false, exact = true, tolerances...)
                parallel_info = step!(parallel_state; parallel = true, exact = true, tolerances...)
                @test budget_status(serial_info.ledger) === budget_status(parallel_info.ledger)
            end
            @test 1 - fidelity(serial_state, parallel_state) < 1e-12
            @test bond_dims(serial_state) == bond_dims(parallel_state)

            # RSVD selection: equal to tolerance, and two parallel runs identical.
            serial_state = neel_state(chain_length)
            parallel_states = [neel_state(chain_length) for _ = 1:2]
            for _ = 1:2
                step!(serial_state; parallel = false, tolerances...)
                for state in parallel_states
                    step!(state; parallel = true, tolerances...)
                end
            end
            @test 1 - fidelity(serial_state, parallel_states[1]) < 1e-8
            @test 1 - fidelity(parallel_states[1], parallel_states[2]) < 1e-13
        end
    end
end
