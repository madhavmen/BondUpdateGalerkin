# Gross-Pitaevskii ground states: imaginary-time BUG and QTT gradient descent.
#
# The schedules are in `BUG/ITE/schedules.jl`. Only the BUG method is here: the QTT TDVP2 baseline
# belongs to `Baselines`, which QTTS may not reference.
#
# Imaginary time d_tau psi = -(H0 + g|psi|^2) psi with the density rebuilt at every stage (or the
# normalised gradient flow with mu[psi], `normalized_flow`). Every method renormalises in the
# physical quadrature after each accepted step and rejects (halves) a step whose stage solve fails
# or whose energy rises.

"""
    gp_ground_state!(psi, m, schedule; order = 2, maxdim = 128, tol_trunc = 1e-10, tol_basis = 1e-8,
                     max_reject = 6, normalized_flow = false, N = 1, growth = 0, on_step = nothing,
                     tol_scaling = :fixed, kwargs...) -> Vector{NamedTuple}

Imaginary-time relaxation of `psi` along the step list `schedule` by [`qtt_bug_step!`](@ref),
renormalised to particle number `N` after each step (the nonlinear ground state depends on `N`:
relax at the physical one, not at 1 and rescale). A step is rejected and retried at half the step
when its stage solve does not converge or the normalised energy rises beyond roundoff above the
energy of the current state (at most `max_reject` rejections per schedule entry on average).

`growth = 0` integrates every scheduled step in full (halved substeps after a rejection: a
relaxation to a fixed time). `growth = q > 1` is the logarithmic policy with memory: entry `k` is
one accepted step of `min(schedule[k], q * h_accepted)`, so after a rejection the steps grow again
from the step that was accepted instead of restarting at the scheduled size.

`tol_scaling` is `qtt_bug_step!`'s, `:fixed` here: a relaxation is graded by the state it reaches,
not by an `O(h^p)` time error, so the real-time `:order` scaling would only tighten its small first
steps. `on_step(entry)` is called with every ledger entry as it is recorded.

The ledger records every attempt with `tau`, `h`, `accepted`, `reason`, `E`, `mu`, `R_mu`, `chi`,
the accumulated wall time `wall`, the attempt's `step_wall`, the cumulative Newton iterations and
GMRES products of all attempts so far (`newton`, `gmres`: rejected attempts included, their work
was spent) and the attempt's cost split (`frame`, the root-bond frame widths, `t_basis`, `t_root`).
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
    N::Real = 1.0,
    growth::Real = 0.0,
    on_step = nothing,
    tol_scaling::Symbol = :fixed,
    kwargs...,
)
    growth == 0 || growth > 1 || throw(ArgumentError("growth must be 0 or > 1, got $growth"))
    fl = GPFlow(m; imaginary = true, normalized = normalized_flow)
    g = m.grid
    qtt_normalize!(psi, g; N = N)
    start = gp_observables(m, qtt_cores(psi), 0.0)
    newton_total, gmres_total = 0, 0
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
        newton = 0,
        gmres = 0,
    )]
    report() = on_step === nothing || on_step(history[end])
    report()
    tau = 0.0
    wall = 0.0
    # the energy of the current state: a retry after a rejection is judged against it, not
    # against the rejected trial (the last ledger entry)
    current_energy = start.E
    h_accepted = Inf
    for scheduled in schedule
        # the span this entry covers: the scheduled step, or (growth) one step capped by the memory
        span = growth > 0 ? min(Float64(scheduled), growth * h_accepted) : Float64(scheduled)
        h = span
        done = 0.0
        while done < span - 1e-14
            h = min(h, span - done)
            trial = copy(psi)
            attempt_start = time_ns()
            step_report = qtt_bug_step!(
                trial,
                fl,
                tau,
                h;
                order = order,
                maxdim = maxdim,
                tol_trunc = tol_trunc,
                tol_basis = tol_basis,
                tol_scaling = tol_scaling,
                kwargs...,
            )
            newton_total += step_report.newton
            gmres_total += step_report.gmres
            qtt_normalize!(trial, g; N = N)
            observed = gp_observables(m, qtt_cores(trial), 0.0)
            step_wall = (time_ns() - attempt_start) / 1e9
            wall += step_wall
            energy_rose = observed.E > current_energy + 1e-10 * abs(current_energy)
            rejections = count(entry -> !entry.accepted, history)
            rejected =
                (!step_report.converged || energy_rose) &&
                rejections < max_reject * length(schedule)
            push!(
                history,
                (
                    tau = rejected ? tau : tau + h,
                    h = h,
                    accepted = !rejected,
                    reason = !rejected ? :ok : step_report.converged ? :energy : :stage,
                    E = observed.E,
                    mu = observed.mu,
                    R_mu = observed.R_mu,
                    chi = observed.chi,
                    wall = wall,
                    newton = newton_total,
                    gmres = gmres_total,
                    step_wall = step_wall,
                    frame = join(step_report.frame_width, "x"),
                    t_basis = step_report.t_basis,
                    t_root = step_report.t_root,
                ),
            )
            report()
            if rejected
                h /= 2
                continue
            end
            _replace_state!(psi, trial)
            current_energy = observed.E
            tau += h
            done += h
            if growth > 0
                h_accepted = h
                break
            end
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
`R_mu < tol`. `on_step(entry)` is called with every ledger entry as it is recorded.
"""
function gp_gradient_descent!(
    psi::SymMPS,
    m::GPModel;
    eta::Real = 0.05,
    maxiter::Int = 500,
    tol::Real = 1e-8,
    maxdim::Int = 128,
    tol_trunc::Real = 1e-10,
    on_step = nothing,
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
    on_step === nothing || on_step(history[end])
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
        on_step === nothing || on_step(history[end])
    end
    _replace_state!(psi, qtt_mps(A))
    return history
end
