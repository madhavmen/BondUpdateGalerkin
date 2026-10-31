# Gross-Pitaevskii ground states: imaginary-time BUG and QTT gradient descent.
#
# Moved from BUG-Julia@b155550:src/RSVDCBEBondUpdate/qtt/qtt_imaginary.jl (`gp_ground_state!`,
# `gp_gradient_descent!`; the schedules are in `BUG/ITE/schedules.jl`). The `:tdvp` method of
# `gp_ground_state!` called the QTT TDVP2 baseline, which belongs to `Baselines` and is not reachable
# from QTTS; only the BUG method is kept here.
#
# Imaginary time d_tau psi = -(H0 + g|psi|^2) psi with the density rebuilt at every stage (or the
# normalised gradient flow with mu[psi], `normalized_flow`). Every method renormalises in the
# physical quadrature after each accepted step and rejects (halves) a step whose stage solve fails
# or whose energy rises.

"""
    gp_ground_state!(psi, m, schedule; order = 2, maxdim = 128, tol_trunc = 1e-10, tol_basis = 1e-8,
                     max_reject = 6, normalized_flow = false, kwargs...) -> Vector{NamedTuple}

Imaginary-time relaxation of `psi` along the step list `schedule` by [`qtt_bug_step!`](@ref),
renormalised after each step. A step is rejected and retried at half the step when its stage solve
does not converge or the normalised energy rises beyond roundoff (at most `max_reject` rejections
per schedule entry on average). The ledger records every attempt with `tau`, `h`, `accepted`,
`reason`, `E`, `mu`, `R_mu`, `chi` and the accumulated wall time.
"""
function gp_ground_state!(
    psi::SymMPS,
    m::GPModel,
    schedule::AbstractVector;
    order::Int = 2,
    maxdim::Int = 128,
    tol_trunc::Real = 1e-10,
    tol_basis::Real = 1e-8,
    max_reject::Int = 6,
    normalized_flow::Bool = false,
    kwargs...,
)
    fl = GPFlow(m; imaginary = true, normalized = normalized_flow)
    g = m.grid
    qtt_normalize!(psi, g)
    start = gp_observables(m, qtt_cores(psi), 0.0)
    history = NamedTuple[(
        tau = 0.0,
        h = 0.0,
        accepted = true,
        reason = :start,
        E = start.E,
        mu = start.mu,
        R_mu = start.R_mu,
        chi = start.chi,
        wall = 0.0,
    )]
    tau = 0.0
    wall = 0.0
    for scheduled in schedule
        h = Float64(scheduled)
        done = 0.0
        while done < scheduled - 1e-14
            h = min(h, scheduled - done)
            trial = copy(psi)
            attempt_start = time_ns()
            report = qtt_bug_step!(
                trial,
                fl,
                tau,
                h;
                order = order,
                maxdim = maxdim,
                tol_trunc = tol_trunc,
                tol_basis = tol_basis,
                kwargs...,
            )
            qtt_normalize!(trial, g)
            observed = gp_observables(m, qtt_cores(trial), 0.0)
            wall += (time_ns() - attempt_start) / 1e9
            energy_rose = observed.E > history[end].E + 1e-10 * abs(history[end].E)
            rejections = count(entry -> !entry.accepted, history)
            if (!report.converged || energy_rose) && rejections < max_reject * length(schedule)
                push!(
                    history,
                    (
                        tau = tau,
                        h = h,
                        accepted = false,
                        reason = report.converged ? :energy : :stage,
                        E = observed.E,
                        mu = observed.mu,
                        R_mu = observed.R_mu,
                        chi = observed.chi,
                        wall = wall,
                    ),
                )
                h /= 2
                continue
            end
            _replace_state!(psi, trial)
            tau += h
            done += h
            push!(
                history,
                (
                    tau = tau,
                    h = h,
                    accepted = true,
                    reason = :ok,
                    E = observed.E,
                    mu = observed.mu,
                    R_mu = observed.R_mu,
                    chi = observed.chi,
                    wall = wall,
                ),
            )
        end
    end
    return history
end

"""
    gp_gradient_descent!(psi, m; eta = 0.05, maxiter = 500, tol = 1e-8, maxdim = 128,
                         tol_trunc = 1e-10) -> Vector{NamedTuple}

QTT gradient descent on the normalised energy (the ground-state baseline of Chen et al.,
arXiv:2507.04279v5): `psi <- normalise(psi - eta (H[psi] - mu[psi]) psi)`, compressed, with
backtracking (`eta` halved while the energy rises, grown by 1.2 after an accepted step). Stops at
`R_mu < tol`.
"""
function gp_gradient_descent!(
    psi::SymMPS,
    m::GPModel;
    eta::Real = 0.05,
    maxiter::Int = 500,
    tol::Real = 1e-8,
    maxdim::Int = 128,
    tol_trunc::Real = 1e-10,
)
    g = m.grid
    qtt_normalize!(psi, g)
    A = qtt_cores(psi)
    observed = gp_observables(m, A, 0.0)
    history = NamedTuple[(
        it = 0,
        eta = 0.0,
        E = observed.E,
        mu = observed.mu,
        R_mu = observed.R_mu,
        chi = observed.chi,
        wall = 0.0,
    )]
    wall = 0.0
    step_size = Float64(eta)
    for iteration = 1:maxiter
        observed.R_mu < tol && break
        iteration_start = time_ns()
        H, _ = gp_hamiltonian_op(m, A, 0.0)
        HA, _ = op_zipup(H, A; tol = 0.1 * tol_trunc)
        mu = observed.mu - m.mu
        local candidate, candidate_observed
        # backtrack until the normalised energy does not rise
        while true
            candidate = tt_lincomb(
                [(1.0, A), (-step_size, HA), (step_size * mu, A)];
                tol = tol_trunc,
                maxdim = maxdim,
            )
            candidate = tt_scale(candidate, 1 / (sqrt(qtt_cell(g)) * tt_norm(candidate)))
            candidate_observed = gp_observables(m, candidate, 0.0)
            (candidate_observed.E <= observed.E + 1e-12 * abs(observed.E) || step_size < 1e-8) &&
                break
            step_size /= 2
        end
        A, observed = candidate, candidate_observed
        step_size *= 1.2
        wall += (time_ns() - iteration_start) / 1e9
        push!(
            history,
            (
                it = iteration,
                eta = step_size,
                E = observed.E,
                mu = observed.mu,
                R_mu = observed.R_mu,
                chi = observed.chi,
                wall = wall,
            ),
        )
    end
    _replace_state!(psi, qtt_mps(A))
    return history
end
