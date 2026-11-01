# bug_step_stages.jl: the stages of `cbe_bug_step!` that run between its half-sweeps and its
# compression -- keyword checks, the order-p frontier closure, and the root report with its
# outside-residual re-enrichment. Split out of `bug_step.jl` so that file stays readable; the step
# calls them in the order they appear here.

"""
    _resolve_step_options(psi, mpo, root, parallel, kls; kwargs...) -> NamedTuple

Check `cbe_bug_step!`'s keywords against each other (in the order the step always checked them, so
the first error raised is unchanged) and resolve their `nothing` defaults. Returns the site count,
the root link, the parallel mode as a `Symbol` and as a `Bool`, the K/L/S schedule, the
`AdaptiveBudget`, the leak-folding switch and the rank limit.
"""
function _resolve_step_options(
    psi,
    mpo,
    root,
    parallel,
    kls;
    tol_basis,
    tol_trunc,
    tol_expv,
    tol_root,
    order,
    fold_omega,
    exact,
    frame_state,
    closure_probe,
    outside_residual,
    maxdim,
)
    site_count = length(psi)
    length(mpo) == site_count ||
        throw(DimensionMismatch("MPO has $(length(mpo)) sites, state has $site_count"))
    site_count >= 2 || throw(ArgumentError("cbe_bug_step! needs at least two sites"))
    root_link = root === nothing ? _bug_root_link(site_count) : root
    1 <= root_link <= site_count - 1 ||
        throw(ArgumentError("root must be a bond in 1:$(site_count - 1), got $root_link"))
    # `parallel` is a `Bool` or a mode name; the step records the name and runs on the `Bool`.
    parallel_mode = parallel isa Symbol ? parallel : (parallel ? :half_sweeps : :serial)
    run_parallel = _resolve_parallel(parallel)
    parallel_mode === :full && (kls = something(kls, :concurrent))
    (tol_root !== nothing && tol_expv !== nothing) &&
        throw(ArgumentError("tol_root and tol_expv name the same tolerance; pass one"))
    tol_root === nothing || (tol_expv = tol_root)

    budget = AdaptiveBudget(
        something(tol_basis, TOL_DEFAULTS.basis),
        something(tol_trunc, TOL_DEFAULTS.trunc),
        something(tol_expv, TOL_DEFAULTS.expv),
    )
    order >= 1 || throw(ArgumentError("order must be >= 1, got $order"))
    fold_omega = something(fold_omega, order > 1 && !exact)
    order == 1 ||
        frame_state === nothing ||
        throw(
            ArgumentError(
                "order > 1 is a single-step method and does not combine with the midpoint " *
                "predictor",
            ),
        )
    closure_probe >= 0 ||
        throw(ArgumentError("closure_probe must be a non-negative count, got $closure_probe"))
    if kls !== nothing
        kls in KLS_SCHEDULES ||
            throw(ArgumentError("the K/L/S schedule must be one of $(KLS_SCHEDULES), got $kls"))
        frame_state === nothing && !outside_residual || throw(
            ArgumentError(
                "the parallel K/L/S step does not combine with the midpoint predictor or the " *
                "outside-residual re-enrichment",
            ),
        )
    end
    # Plan §1 signature: `maxdim = 128`.
    maxdim = something(maxdim, 128)
    return (
        site_count = site_count,
        root_link = root_link,
        parallel_mode = parallel_mode,
        run_parallel = run_parallel,
        kls = kls,
        budget = budget,
        fold_omega = fold_omega,
        maxdim = maxdim,
    )
end

"""
    _refresh_expanded!(expanded, left_frames, right_frames, root_link) -> expanded

Record the grown frame widths off the root bond: `left_frames[bond]` (bonds `1..root_link-1`) by
its right leg, `right_frames[bond]` (bonds `root_link+1..length(expanded)`) by its left leg.
"""
function _refresh_expanded!(expanded, left_frames, right_frames, root_link::Int)
    for bond = 1:(root_link-1)
        expanded[bond] = leg_dim(left_frames[bond], 3)
    end
    for bond = (root_link+1):length(expanded)
        expanded[bond] = leg_dim(right_frames[bond], 1)
    end
    return expanded
end

"""
    _close_root_frontiers!(frames, psi, mpo, root_start, tau, alloc, expansion_settings, expanded,
                           ledger; kwargs...) -> Union{Nothing,NamedTuple}

The general-order stage (plan: krylov_closure_cbe_bug_realtime_L16_plan.md). The half-sweeps
closed the frontier j = 0 (`psi` itself). For `order > 1` the frontiers j = 1..order-1 of the ROOT
generator are closed against the GLOBAL operator, growing `frames.left_frames` and `frames.right_frames` in place; the
root environments and the root start vector come back rebuilt in the final frames, and the single
root solve runs in them. Returns those (and the closure's operator applications), or `nothing`
when the frames are left as they were: for `order = 1`, and for a `closure_probe` measurement,
which only records the frontiers in `ledger`.
"""
function _close_root_frontiers!(
    frames::KLFrames,
    psi,
    mpo,
    root_start,
    tau,
    alloc,
    expansion_settings,
    expanded,
    ledger;
    order,
    closure_probe,
    closure_checks,
    expansion,
    krylov_reuse,
    recycle,
    recycle_report,
    krylov_depth,
    workcap,
    norm_scale,
    parallel,
    rng,
    fold,
)
    if order > 1
        # Plan T4.1: the frontier sweeps expand by the same DD-QR CBE as the half-sweeps.
        closure_settings = merge(
            expansion_settings,
            (
                expansion = expansion,
                krylov_reuse = krylov_reuse,
                recycle = recycle,
                krylov_depth = krylov_depth,
            ),
        )
        closure = _krylov_closure!(
            frames,
            psi,
            mpo,
            root_start,
            tau,
            alloc,
            closure_settings;
            order = order,
            workcap = workcap,
            nrm = norm_scale,
            parallel = parallel,
            rng = rng,
            ledger = ledger,
            fold = fold,
            checks = closure_checks,
        )
        _refresh_expanded!(expanded, frames.left_frames, frames.right_frames, frames.root_link)
        recycle_report === nothing || recycle === nothing || (recycle_report[] = recycle.log)
        return (
            lenv = frames.lenv,
            renv = frames.renv,
            root_start = closure.root_start,
            operator_applications = closure.n_h1 + closure.n_root,
        )
    elseif closure_probe > 0
        # A copied `rng` and untouched frames: the step itself stays bit-identical.
        probe = _measure_frontiers(
            frames,
            psi,
            mpo,
            root_start,
            tau,
            alloc,
            (expansion = expansion,),
            closure_probe;
            nrm = norm_scale,
            parallel = parallel,
            rng = copy(rng),
            fold = fold,
        )
        merge_ledger!(ledger, probe.ledger)
    end
    return nothing
end

"""
    _report_root!(root_report, psi, mpo, left_frames, right_frames, root_link, lenv_root,
                  renv_root, root_operator, root_start, root_result, solve_with; kwargs...)
        -> NamedTuple

The `root_report` instrumentation of `cbe_bug_step!`: the energies before and after the root solve,
the solve's report and, with `outside_residual = true`, the outside-Galerkin action of the evolved
state. When that residual fails `tol_basis`, the frames are re-enriched by the evolved state's own
frontier sweep and the root is re-solved (plan Task 1). `solve_with(operator, start)` is the step's
root solver; `root_seconds` (a `Ref`) accumulates the re-solves' time. Returns the possibly
re-solved `root_start`/`root_result` and the root environments they live in.
"""
function _report_root!(
    root_report::Base.RefValue,
    psi,
    mpo,
    left_frames,
    right_frames,
    root_link::Int,
    lenv_root,
    renv_root,
    root_operator,
    root_start,
    root_result,
    solve_with;
    tau,
    eshift,
    alloc,
    expansion_settings,
    expansion,
    krylov_depth,
    expanded,
    ledger,
    root_seconds,
    outside_residual,
    tol_basis,
    order,
    workcap,
    norm_scale,
    parallel,
    rng,
    fold,
    root_method,
    root_solve_report,
    root_applications,
)
    # Rayleigh quotients in the Galerkin space: `root_start` and `root_result` ARE the state
    # before the root and after it (before the compression), and `<S|H_0|S> = <psi|P H P|psi> =
    # <psi|H|psi>` for `psi` in the frames' span, so these are exact energies, not projections.
    rayleigh_quotient(state) = (norm_squared = norm(state)^2;
    norm_squared > 0 ?
    real(tensor_inner(state, apply_zero_site(root_operator, state))) / norm_squared +
    Float64(eshift) : NaN)
    energy_before = rayleigh_quotient(root_start)
    # The root frames as they stand now (the environments are rebound by a re-enrichment).
    current_frames() = KLFrames(
        left_frames,
        right_frames,
        root_link,
        lenv_root,
        renv_root,
        0,
        nothing,
        nothing,
        Float64(eshift),
    )
    # The outside-Galerkin action `(I - P) H state` in the current frames, from a copied `rng` so
    # the measurement leaves the step's random stream untouched.
    outside_of(state) = estimate_global_complement_action(
        psi,
        mpo,
        current_frames(),
        state;
        expansion = expansion,
        rng = copy(rng),
    )
    outside = outside_residual ? outside_of(root_result) : nothing
    # The step's defect `|tau| ||(I - P) H state|| / ||state||` (`NaN` when not measured).
    residual_of(action, state) =
        action === nothing ? NaN : abs(tau) * action.bound / max(norm(state), eps())
    initial_residual = residual_of(outside, root_result)
    # ── re-enrich and re-solve on the failing residual (plan Task 1) ──
    # The closure can meet its shares while the outside action fails: grow frames by the evolved
    # state's own sweep, re-solve the root until the defect fits, nothing is added, or the ceiling
    # binds.
    passes, added_directions = 0, 0
    if outside_residual
        while !(residual_of(outside, root_result) <= tol_basis) && passes < _closure_maxpass(order)
            frames = current_frames()
            passes += 1
            sweep = _frontier_sweep!(
                frames,
                psi,
                mpo,
                KrylovFrontier(root_result, 0, frames.gen),
                tau,
                alloc,
                merge(expansion_settings, (expansion = expansion, krylov_depth = krylov_depth));
                enrich = true,
                workcap = workcap,
                nrm = norm_scale,
                parallel = parallel,
                rng = rng,
                pass = passes,
                fold = fold,
            )
            merge_ledger!(ledger, sweep.ledger)
            sweep.added == 0 && break
            added_directions += sweep.added
            lenv_root, renv_root = frames.lenv, frames.renv
            root_start = _project_root(psi, frames)
            root_operator = shift_h(
                zero_site_h(
                    mpo,
                    root_link,
                    lenv_root,
                    renv_root,
                    left_frames[root_link],
                    right_frames[root_link],
                ),
                eshift,
            )
            solve_started = time_ns()
            root_result = solve_with(root_operator, root_start)
            root_seconds[] += (time_ns() - solve_started) / 1e9
            outside = outside_of(root_result)
            sweep.capped && break
        end
        if added_directions > 0
            _refresh_expanded!(expanded, left_frames, right_frames, root_link)
            ledger.r_aug =
                max(maximum(expanded; init = 0), leg_dim(right_frames[root_link], 1))
        end
    end
    energy_after = rayleigh_quotient(root_result)
    result_norm = max(norm(root_result), eps())
    root_report[] = (
        root_method = root_method,
        eshift = Float64(eshift),
        e_pre = energy_before,
        e_root = energy_after,
        solve = root_solve_report[],
        n_root = root_applications[],
        # the defect before the re-enrichment above, its passes and directions
        resid_initial = initial_residual,
        outside_passes = passes,
        outside_added = added_directions,
        outside = outside === nothing ? NaN : outside.bound / result_norm,
        outside_left = outside === nothing ? NaN : outside.left / result_norm,
        outside_right = outside === nothing ? NaN : outside.right / result_norm,
        # per bond (left frames 1..root_link, right frames root_link..end), scale-free as above
        per_left = outside === nothing ? Float64[] : outside.per_left ./ result_norm,
        per_right = outside === nothing ? Float64[] : outside.per_right ./ result_norm,
        # the step's outside-Galerkin defect, `|tau| ||(I-P) H psi_1|| / ||psi_1||`
        resid = outside === nothing ? NaN : abs(tau) * outside.bound / result_norm,
    )
    return (
        root_start = root_start,
        root_result = root_result,
        lenv = lenv_root,
        renv = renv_root,
    )
end
