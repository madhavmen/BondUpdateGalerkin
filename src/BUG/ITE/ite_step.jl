# Moved from BUG-Julia@b155550:src/RSVDCBEBondUpdate/bug/imaginary_time.jl.

"`<psi|H|psi> / <psi|psi>`, any gauge (`mpo_energy` is exact in every gauge)."
_ite_energy(psi::SymMPS, mpo::MPO) =
    real(mpo_energy(psi, mpo)) / max(real(overlap(psi, psi)), eps())

"Renormalise a canonical state in place through its centre tensor; returns the old norm."
function _ite_normalise!(psi::SymMPS)
    state_norm = norm(psi)
    (isfinite(state_norm) && state_norm > 0) || throw(
        ErrorException(
            "imaginary-time step produced a state of norm $state_norm -- not representable",
        ),
    )
    psi[psi.center] = to_concrete((1.0 / state_norm) * psi[psi.center])
    return state_norm
end

# The ledger reasons a retry can address; `:outside` is this file's own (the outside-Galerkin
# defect). `:projection` belongs to the midpoint step, which this driver does not use.
const _ITE_REASONS = (:sample, :tail, :krylov, :closure, :expv, :trunc, :outside)

# Largest number of sub-steps one rejection may split a step into.
const _ITE_MAXSPLIT = 64

"""
How many sub-steps a rejected `attempt` is split into. The outside defect of an order-`p` step is
`|tau| ||(I-P) H psi_1||` with `(I-P) H psi_1 = O(tau^p)` (the closure holds the frontiers below
`p`), i.e. `O(dbeta^(p+1))` -- MEASURED from Neel and the domain wall at L = 10 (fp1, job 17342386:
slopes ~3 for p = 2, ~2 for p = 1 from the domain wall). So the split that brings it within
`tol_basis` is `(resid / tol_basis)^(1/(p+1))`, with a 20% margin. Other reasons halve.

⛔ A FIXED HALVING WAS THE FIRST-STEP "FAILURE". From a product state the energy fluctuation is
maximal, so the defect's constant is 400-1000x the second step's (fp1): the panel's first step from
Neel (dbeta 0.0625, tol_basis 1e-7) needed ~dbeta/7, and two halvings (`retries = 2`) reach only
dbeta/4 -- the step was flagged although the method meets the tolerance at a smaller step.
"""
function _ite_nsplit(attempt, tol_basis::Float64, order::Int)
    (:outside in attempt.unmet && isfinite(attempt.resid) && attempt.resid > tol_basis) ||
        return 2
    return clamp(ceil(Int, 1.2 * (attempt.resid / tol_basis)^(1 / (order + 1))), 2, _ITE_MAXSPLIT)
end

"""
Whether a rejected `attempt` failed for a RESOURCE cap -- the rank cap `maxdim` cut a bond past its
`tol_trunc` share, or the work ceiling stopped an enrichment with weight left. A smaller step
cannot widen either, and splitting multiplies the binding truncations (IT-2 4x4 at maxdim 64, s38:
4 sub-steps per step, each cut at the cap, drove E - E(beta) to 6e-2), so such a step is flagged
at once.
"""
_ite_capped(attempt) = attempt.info.ledger.maxdim_reached || attempt.info.ledger.workcap_reached

"""
One attempt at one step: the state's energy, the step at `tau = -dbeta` with the shifted
generators and the root report, renormalisation, the energy after, and the verdicts.
"""
function _ite_attempt!(
    psi::SymMPS,
    mpo::MPO,
    dbeta::Float64;
    order::Int,
    root_method::Symbol,
    tol_basis::Float64,
    tol_trunc::Float64,
    tol_root::Float64,
    maxdim::Int,
    eshift,
    outside_residual::Bool,
    closure_checks::Bool,
    parallel::Bool,
    rng::AbstractRNG,
    kwargs...,
)
    start_time = time_ns()
    energy_before = _ite_energy(psi, mpo)
    es = eshift === :energy ? energy_before : Float64(eshift)
    rep = Ref{Any}(nothing)
    info = cbe_bug_step!(
        psi,
        mpo,
        ComplexF64(-dbeta);
        order = order,
        root_method = root_method,
        eshift = es,
        tol_basis = tol_basis,
        tol_trunc = tol_trunc,
        tol_expv = tol_root,
        maxdim = maxdim,
        root_report = rep,
        outside_residual = outside_residual,
        closure_checks = closure_checks,
        parallel = parallel,
        rng = rng,
        kwargs...,
    )
    growth = _ite_normalise!(psi)
    energy_after = _ite_energy(psi, mpo)
    report = rep[]          # the root step's energies and outside defect
    unmet = copy(info.ledger.unmet)
    # `!(resid <= tol)` and not `resid > tol`: a NaN defect is a failed measurement, not a pass.
    outside_residual && !(report.resid <= tol_basis) && push!(unmet, :outside)
    return (
        dbeta = dbeta,
        ok = isempty(unmet),
        unmet = unmet,
        eshift = es,
        e_before = energy_before,
        e_pre_root = report.e_pre,
        e_root = report.e_root,
        e_after = energy_after,
        de_root = report.e_root - report.e_pre,
        de_compression = energy_after - report.e_root,
        outside = report.outside,
        resid = report.resid,
        resid_initial = report.resid_initial,
        outside_passes = report.outside_passes,
        outside_added = report.outside_added,
        root_err = info.ledger.expv_err,
        root_iters = info.ledger.expv_iters,
        discarded = info.ledger.discarded,
        growth = growth,
        chi = maximum(bond_dims(psi); init = 0),
        r_aug = info.ledger.r_aug,
        closure_checks = closure_checks,
        wall = (time_ns() - start_time) / 1e9,
        info = info,
    )
end

"""
    ite_step!(psi, mpo, dbeta; order = 2, root_method = :exp, tol_basis = 1e-5,
              tol_trunc = 1e-6, tol_root = 1e-6, maxdim = 128, eshift = :energy,
              outside_residual = true, retries = 2, on_fail = :flag, parallel = false,
              rng, kwargs...) -> NamedTuple

One normalised imaginary-time step over `dbeta`, in place. Accepted when the ledger is `:met` and
the outside defect is within `tol_basis`. Otherwise the step is re-enriched, then redone from the
saved state with order-p closure and, while `retries > 0`, as sub-steps (their number from
`_ite_nsplit`). Resource-capped failures are not split. A step still failing returns
`status = :failed` (or throws with `on_fail = :error`).
Returns `status`, `unmet`, the energies, `resid`, `nsub` and `attempts`.
"""
function ite_step!(
    psi::SymMPS,
    mpo::MPO,
    dbeta::Real;
    order::Int = 2,
    root_method::Symbol = :exp,
    tol_basis::Float64 = 1e-5,
    tol_trunc::Float64 = 1e-6,
    tol_root::Float64 = 1e-6,
    maxdim::Int = 128,
    eshift = :energy,
    outside_residual::Bool = true,
    retries::Int = 2,
    on_fail::Symbol = :flag,
    parallel::Bool = false,
    rng::AbstractRNG = MersenneTwister(0x5EED),
    kwargs...,
)
    dbeta > 0 || throw(ArgumentError("dbeta must be positive, got $dbeta"))
    root_method in ROOT_METHODS ||
        throw(ArgumentError("root_method must be one of $(ROOT_METHODS), got :$root_method"))
    on_fail in (:flag, :error) || throw(ArgumentError("on_fail must be :flag or :error"))
    retries >= 0 || throw(ArgumentError("retries must be >= 0, got $retries"))
    start_time = time_ns()
    _ite_normalise!(psi)
    saved = copy(psi)         # every mutation replaces a slot, so this snapshot is stable
    akw = (
        order = order,
        root_method = root_method,
        tol_basis = tol_basis,
        tol_trunc = tol_trunc,
        tol_root = tol_root,
        maxdim = maxdim,
        eshift = eshift,
        outside_residual = outside_residual,
        parallel = parallel,
        rng = rng,
    )
    # Put the saved tensors and gauge centre back after a rejected attempt.
    function restore!()
        for site = 1:length(psi)
            psi[site] = saved[site]
        end
        psi.center = saved.center
        return nothing
    end

    attempts = Any[]
    attempt = _ite_attempt!(psi, mpo, Float64(dbeta); akw..., closure_checks = false, kwargs...)
    push!(attempts, attempt)
    # `:outside` is re-enriched inside the attempt (`cbe_bug_step!`); rerunning the closure adds
    # nothing for it (d51: identical defect), so only an unclosed closure is retried this way.
    if !attempt.ok && order > 1 && :closure in attempt.unmet
        restore!()
        attempt =
            _ite_attempt!(psi, mpo, Float64(dbeta); akw..., closure_checks = true, kwargs...)
        push!(attempts, attempt)
    end
    attempt.ok &&
        return _ite_result(length(attempts) == 1 ? :met : :retried, [attempt], attempts, start_time)

    if retries > 0 && !_ite_capped(attempt)
        restore!()
        substep_count = _ite_nsplit(attempt, tol_basis, order)
        halves = Any[]
        for _ = 1:substep_count
            substep = ite_step!(
                psi,
                mpo,
                dbeta / substep_count;
                akw...,
                retries = retries - 1,
                on_fail = :flag,
                kwargs...,
            )
            append!(attempts, substep.attempts)
            push!(halves, substep)
        end
        ok = all(substep.status !== :failed for substep in halves)
        res = _ite_result(ok ? :retried : :failed, halves, attempts, start_time)
        ok ||
            on_fail === :flag ||
            throw(
                ErrorException(
                    "imaginary-time step dbeta=$dbeta failed after retries: " *
                    "unmet $(res.unmet)",
                ),
            )
        return res
    end
    on_fail === :error && throw(
        ErrorException(
            "imaginary-time step dbeta=$dbeta failed: unmet $(attempt.unmet); " *
            budget_diagnostic(attempt.info.ledger),
        ),
    )
    return _ite_result(:failed, [attempt], attempts, start_time)
end

"""
Fold the accepted pieces of a step (one attempt, or the sub-steps of a split) into one record:
first energy before, last after, root and compression changes summed, worst defects. `start_time`
is the step's `time_ns()` at entry, for the wall time.
"""
function _ite_result(status::Symbol, pieces, attempts, start_time)
    # a piece is an attempt (one step) or a split's sub-step record (`nsub` steps)
    nsub(piece) = hasproperty(piece, :nsub) ? piece.nsub : 1
    unmet = unique(vcat(Symbol[], (piece.unmet for piece in pieces)...))
    return (
        status = status,
        unmet = unmet,
        dbeta = sum(piece.dbeta for piece in pieces),
        e_before = pieces[1].e_before,
        e_pre_root = pieces[1].e_pre_root,
        e_root = pieces[end].e_root,
        e_after = pieces[end].e_after,
        de_root = sum(piece.de_root for piece in pieces),
        de_compression = sum(piece.de_compression for piece in pieces),
        resid = maximum(piece.resid for piece in pieces),
        outside = maximum(piece.outside for piece in pieces),
        root_err = maximum(piece.root_err for piece in pieces),
        discarded = maximum(piece.discarded for piece in pieces),
        chi = pieces[end].chi,
        nsub = sum(nsub(piece) for piece in pieces),
        attempts = attempts,
        wall = (time_ns() - start_time) / 1e9,
    )
end

"""
    ite_evolve!(psi, mpo, steps; observe = nothing, kwargs...) -> Vector{NamedTuple}

Normalised imaginary-time evolution over the step list `steps` (`log_panel_steps`,
`uniform_beta_steps`, or any positive vector), one [`ite_step!`](@ref) per entry with `kwargs`.
Returns one record per step: `beta` reached plus the step's record, and `obs = observe(psi)` when
an observable is given. A `:failed` step is recorded and the evolution continues
(`on_fail = :error` stops it instead).
"""
function ite_evolve!(
    psi::SymMPS,
    mpo::MPO,
    steps::AbstractVector{<:Real};
    observe = nothing,
    kwargs...,
)
    log = Any[]
    beta = 0.0
    for db in steps
        step_record = ite_step!(psi, mpo, db; kwargs...)
        beta += db
        push!(
            log,
            merge(
                (beta = beta,),
                step_record,
                observe === nothing ? NamedTuple() : (obs = observe(psi),),
            ),
        )
    end
    return log
end
