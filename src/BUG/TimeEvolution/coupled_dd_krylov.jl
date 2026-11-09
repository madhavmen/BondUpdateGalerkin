# Coupled DD-RSVD + enriched Krylov global-closure retries.
#
# The global-closure residual is not available until AFTER both half-sweeps.
# A retry therefore restarts the entire candidate step on an untouched copy
# of the original state and raises the minimum DD retained rank. This is an
# expensive but physically meaningful baseline for a future targeted/local
# retry controller. We cache thin DD sketch actions across the candidate ranks
# of each attempt (dd_cbe_expand.jl), but NOT across different environment
# gauges/outer retries: that reuse would be mathematically invalid.
#
# This function is deliberately NON-mutating; rejected trials are not committed.
# Only the returned state after accepted=true is valid to advance time.
"""
    cbe_bug_coupled_step(psi, mpo, tau; tol_basis=1e-3, max_retries=4, ...)

Run order-2 CBE-BUG under `ceil(1.2*D_b_old)` working-rank ceilings, *without*
a physical-d factor. An independent DD holdout first chooses a minimal rank.
The local enriched Lanczos stages and the checked global Krylov frontiers then
validate the combined frame. If the global or DD budget fails, the entire
candidate step is retried on the same original MPS with a larger minimum DD
rank, while retaining the same exact cap. No failed trial is silently accepted.

The diagnostic `attempts` exposes the total DD sketch columns, DD holdouts,
K/L H calls, closure H calls, root H calls, and peak augmented-root sizes
INCLUDING discarded trial attempts. A small `max_retries` prevents unbounded
resampling when the rank cap makes the tolerance impossible.

Returns a NamedTuple `(state, info, accepted, status, attempts, totals)`.
If `accepted=false`, `state` is a fresh copy of the ORIGINAL input MPS.
The caller may reduce the physical timestep and retry; it must not advance
time using an unsuccessful candidate.
"""
function cbe_bug_coupled_step(
    psi::SymMPS,
    mpo::MPO,
    tau::Number;
    tol_basis::Real = 1e-3,
    tol_trunc::Real = 1e-7,
    tol_root::Real = 1e-10,
    maxdim::Int = 128,
    max_retries::Int = 4,
    rank_initial_width::Int = 4,
    holdout_width::Int = 4,
    acceptance_factor::Float64 = 0.7,
    krylov_mode::Symbol = :lanczos,
    rng::AbstractRNG = MersenneTwister(0x5EED),
)
    1 <= max_retries <= 12 || throw(ArgumentError("max_retries must be in 1:12"))
    krylov_mode in (:lanczos, :arnoldi) ||
        throw(ArgumentError("unknown local Krylov recurrence: $krylov_mode"))
    original = copy(psi)
    old = bond_dims(original)
    caps = bond_work_ceilings_strict(old, maxdim; factor = 1.2)
    maxroom = maximum(caps .- old; init = 0)
    base_seed = rand(rng, UInt64)
    history = Any[]
    cumulative = (
        DD_discovery = 0,
        DD_holdouts = 0,
        K_L_H = 0,
        closure_H = 0,
        root_H = 0,
        wall_s = 0.0,
    )
    last = nothing
    last_floor = -1
    for attempt = 1:max_retries
        # The first pass is the ordinary minimal-rank DD selection. Later
        # passes enforce rank floors 1,2,4,..., bounded by physical cap room.
        floor = attempt == 1 ? 0 : min(maxroom, 2^(attempt-2))
        floor == last_floor && break
        last_floor = floor
        candidate = copy(original)
        policy = DDQRExpansion(
            mode = :rank_adaptive,
            rank_initial_width = max(rank_initial_width, floor + 2),
            rank_floor = floor,
            holdout_width = holdout_width,
            acceptance_factor = acceptance_factor,
            krylov_mode = krylov_mode,
        )
        t0 = time_ns()
        info = cbe_bug_step!(
            candidate, mpo, ComplexF64(tau);
            order = 2,
            closure_checks = true,
            tol_basis = tol_basis,
            tol_trunc = tol_trunc,
            tol_root = tol_root,
            maxdim = maxdim,
            expansion = policy,
            work_ceiling_rule = :bond_strict,
            bond_ceiling_factor = 1.2,
            rng = MersenneTwister(base_seed),
        )
        wall_s = (time_ns()-t0)/1e9
        # Work caps are checked against the dimensions BEFORE any trial.
        all(info.expanded .<= caps) ||
            error("DD/KL frame exceeded strict 1.2*old bond cap")
        info.root_right <= caps[info.root] ||
            error("right root input exceeded strict 1.2*old bond cap")
        all(bond_dims(candidate) .<= caps) ||
            error("final bond exceeded strict 1.2*old bond cap")
        for bl in info.ledger.bond_log
            bl.width <= caps[bl.bond] ||
                error("one of the augmented K/L/closure frames violated cap")
        end
        lg = info.ledger
        # The two statuses MUST be kept distinct. In particular, closure
        # passing does not override a failed DD/sample or truncation check.
        accepted = lg.tolerance_met
        row = (
            trial = attempt,
            rank_floor = floor,
            accepted = accepted,
            unmet = copy(lg.unmet),
            DD_discovery = lg.n_probes,
            DD_holdouts = lg.n_holdout,
            K_L_H = lg.n_h1,
            closure_H = lg.n_h1_closure,
            root_H = info.record.counters.n_root,
            max_stage_frame = info.record.D_stage,
            root_left = info.expanded[info.root],
            root_right = info.root_right,
            root_core_elements = info.expanded[info.root]*info.root_right,
            final_bond = maximum(bond_dims(candidate)),
            wall_s = wall_s,
        )
        push!(history, row)
        cumulative = (
            DD_discovery = cumulative.DD_discovery + row.DD_discovery,
            DD_holdouts = cumulative.DD_holdouts + row.DD_holdouts,
            K_L_H = cumulative.K_L_H + row.K_L_H,
            closure_H = cumulative.closure_H + row.closure_H,
            root_H = cumulative.root_H + row.root_H,
            wall_s = cumulative.wall_s + row.wall_s,
        )
        last = info
        accepted && return (
            state = candidate,
            info = info,
            accepted = true,
            status = :accepted,
            attempts = history,
            totals = cumulative,
        )
        # Increasing DD rank cannot cure a failing root integration or
        # compression tolerance; stop rather than burning unnecessary probes.
        possible = all(reason in (:sample, :krylov, :closure, :tail, :projection)
                       for reason in lg.unmet)
        possible || break
        floor >= maxroom && break
    end
    return (
        state = copy(original),
        info = last,
        accepted = false,
        status = :cap_or_budget_unmet,
        attempts = history,
        totals = cumulative,
    )
end
