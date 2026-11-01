# Moved from BUG-Julia@b155550:src/RSVDCBEBondUpdate/bug/cbe_bug.jl.

"""
    cbe_bug_step!(psi, mpo, tau; kwargs...) -> CBEBugSweepInfo

One step of the CBE-BUG sweep, in place. `tau = -im*dt` is real time, `tau = -dt` imaginary; the
caller renormalises. Only the root bond evolves; both half-sweeps just generate bases.

Keywords: `tol_basis` (its local Krylov share separately: `tol_krylov`), `tol_trunc`, `tol_expv` (alias `tol_root`) and `maxdim` (the resource limit,
`adaptive_budget.jl`); `order` (temporal order `p`, see `krylov_closure.jl`); `root` (the Galerkin
bond, default `L/2`); `reorth` (re-orthogonalisation in the root solve); `expansion` (the
basis-selection strategy of both half-sweeps and the frontier sweeps, a `BasisExpansion`;
default `DDQRExpansion()`, the DD-projected, MPO-streamed, QR-only CBE).
"""
function cbe_bug_step!(
    psi::SymMPS,
    mpo::MPO,
    tau::ComplexF64;
    # ── THE API: three tolerances and one resource limit ──────────────
    # `nothing` takes the default of `TOL_DEFAULTS` (`adaptive_budget.jl`).
    tol_basis::Union{Nothing,Float64} = nothing,
    tol_trunc::Union{Nothing,Float64} = nothing,
    tol_expv::Union{Nothing,Float64} = nothing,
    # The plan's name for `tol_expv` (cbe_bug_high_order_rsvd_parallel_tasks.md
    # invariant 1); naming both is refused.
    tol_root::Union{Nothing,Float64} = nothing,
    # The share of `tol_basis` the recycled local Krylov stage (Tests C/D, half-
    # and frontier sweeps) is held to, set on its own (`basis_allocation`);
    # `nothing` is `tol_basis`, the plan's one-knob default.
    tol_krylov::Union{Nothing,Float64} = nothing,
    # The temporal order `p` -- a METHOD choice, not a fourth tolerance.
    # `p > 1` closes the global Krylov frontiers `j = 1..p-1` of the root
    # generator before the one root solve (`krylov_closure.jl`); `1` is the
    # ordinary step, unchanged.
    order::Int = 1,
    # ⚠ INSTRUMENTATION, NOT A METHOD CHOICE. `m > 0` on an `order = 1`
    # step MEASURES (never enriches) the frontiers `j = 0..m-1` on the frames
    # the ordinary step solves in -- `ledger.leak_initial`/`rho_initial`,
    # what CBE + local Krylov already capture of the global Krylov sequence.
    # The evolved state is unchanged. `order > 1` always records them.
    closure_probe::Int = 0,
    # ⚠ VALIDATION, NOT A METHOD CHOICE (USER 2026-10-17: off by default).
    # `false`: an `order = p` step is the two half-sweeps plus ONE enriching
    # sweep per frontier j = 1..p-1. `true`: the closure as it was validated --
    # measure every frontier first, loop until a pass adds nothing, measure
    # j = 0 on the final frames (`_krylov_closure!(...; checks)`).
    closure_checks::Bool = false,
    # ⚠ ABLATION, NOT A METHOD CHOICE (plan T4.1 / P07). `true`: every local
    # Krylov stage (half-sweeps and frontier sweeps) grows from the CBE output
    # `[U0 | U_CBE]`, so it never regenerates a CBE direction. `false`: it grows
    # from `U0` alone and the CBE directions are merged afterwards
    # (`_merge_frame`) -- the reuse-off arm of P07.
    krylov_reuse::Bool = true,
    # ⚠ ABLATION, NOT A METHOD CHOICE (xx-efficiency plan Fig. 3; dissipative
    # plan §1.2). `true`: each order-p round's local Krylov stage reuses the
    # previous round's vectors at that bond, transported into its gauge
    # (`krylov_recycle.jl`); real-time and no-jump steps only. Off by default:
    # neither study found a gain (docs/dissipative_results_summary.md §6). A
    # `Ref` in `recycle_report` receives the per-bond log.
    krylov_recycle::Bool = false,
    recycle_report::Union{Nothing,Base.RefValue} = nothing,
    # `:images` refreshes and continues the cached recurrence; `:frame` only
    # seeds the next round's frame with the transported vectors (no images);
    # `:complement` reuses the previous block only where it certifies the old
    # input's evolution under this round's `H1`, and extends only the complement
    # of that input (`_rk_complement`).
    recycle_mode::Symbol = :images,
    # ⚠ ABLATION, NOT A METHOD CHOICE (xx-efficiency plan Fig. 4, conventional
    # controls). `dd_width = l > 0`: every DD-QR CBE takes ONE probe of width
    # `l` (no holdout, no adaptive growth). `krylov_depth = m > 0`: every local
    # Krylov stage runs exactly `m` steps with no weight stop.
    dd_width::Int = 0,
    krylov_depth::Int = 0,
    # ── IMAGINARY TIME (plan: cbe_bug_imaginary_open_benchmark, Task 1) ──
    # The root integrator (`ROOT_METHODS`): `:exp` is the Saad-converged
    # exponential (the reference); `:trapezoid`/`:radau2`/`:radau3` apply that
    # implicit method's stability function in one stage; `:radau2_gmres` solves
    # Radau IIA(2)'s coupled stages by GMRES to `tol_root`. A METHOD choice: the
    # CBE, local Krylov and frontier enrichment before the root are the same.
    root_method::Symbol = :exp,
    # The step integrates `H - eshift` (`shift_h`): the root and every Krylov
    # generator, so the basis weights measure the change of the NORMALISED
    # state rather than its `exp(-dbeta*E)` growth. `0.0` is bit-identical.
    eshift::Real = 0.0,
    # ⚠ INSTRUMENTATION. A `Ref` receives the root's account: energies of the
    # state before and after the root solve (`<S|H|S>/<S|S>`, exact in the
    # Galerkin space), the solve's report, and with `outside_residual = true`
    # the outside-Galerkin action `||(I - P) H psi_1|| / ||psi_1||` of the
    # evolved state (`estimate_global_complement_action`, one measuring sweep).
    root_report::Union{Nothing,Base.RefValue} = nothing,
    outside_residual::Bool = false,
    # ⚠ INSTRUMENTATION. A `Ref` receives the root's final Galerkin frames
    # (`KLFrames`), for small-system diagnostics of the product space the step
    # integrated in (plan T3.5 Taylor leak). Never read by the step.
    frames_report::Union{Nothing,Base.RefValue} = nothing,
    # ── TRIAL SPACE CONTAINS `keep_states` (plan Task 2) ──
    # Each half-sweep carries these MPS through the new frames, so the root space
    # contains each one exactly. Keeping `|I>>` makes an MPDO step
    # trace-preserving before the compression. The states must share `psi`'s site
    # legs; `[]` is the ordinary step.
    keep_states::AbstractVector = Any[],
    # ⚠ DEBUG REFERENCE (plan: exact DD references debug-only). `true` draws the
    # complete DD-projected probe instead of the randomised one.
    exact::Bool = false,
    root::Union{Nothing,Int} = nothing,
    truncate::Bool = true,
    maxdim::Union{Nothing,Int} = nothing,
    reorth::Bool = true,
    # Build the bases from THIS state instead of from `psi`, while still
    # evolving `psi` over the full `tau`. `nothing` is the ordinary step.
    # Exists for [`cbe_bug_midpoint_step!`](@ref); see the comment at `psiL`.
    frame_state = nothing,
    # ⛔ `hermitian = false` SWITCHES THE ROOT SOLVE TO ARNOLDI. Lanczos on a
    # non-Hermitian generator -- an open system's Lindbladian or a
    # non-Hermitian H_eff -- returns a plausible vector and no error at all.
    # The half-sweep basis needs no flag: it already builds the full
    # upper-Hessenberg projection, which is the tridiagonal when H = H'.
    hermitian::Bool = true,
    # ── `parallel`: the two half-sweeps run concurrently ──
    # They share no data; only the root solve needs both. Needs `julia -t 2` or
    # more, off by default. Roundoff-level differences from serial. `:full`
    # selects the K/L/S integrator.
    parallel::Union{Bool,Symbol} = false,
    # ⚠ A DIFFERENT INTEGRATOR, set by `parallel_kls_step!` (plan Task 6): the
    # parallel K/L/S step in the preliminary frames, with this K/L/S schedule
    # (`KLS_SCHEDULES`). `nothing` is the standard augmented root step.
    kls::Union{Nothing,Symbol} = nothing,
    # The closure's leak measurement (`krylov_closure.jl`): fold the probe into
    # the two-site contraction instead of building `H*Theta`. `nothing` folds
    # for `order > 1`, so the closure forms no dense two-site action (dissipative
    # plan §1.2), except under `exact`: the folded leak is sampled, and the debug
    # reference draws no random numbers. `false` keeps the exact `H*Theta` leak.
    fold_omega::Union{Nothing,Bool} = nothing,
    # Which DD-QR CBE directions survive when the work ceiling binds:
    # `:weight` the heaviest across all sectors, `:sector` the first sectors in
    # storage order (`dd_cbe_expand`). Below the ceiling both are identical.
    cap_order::Symbol = :sector,
    # ⚠ ABLATION, NOT A METHOD CHOICE (QTT plan §1.2A, arm D3). `:hybrid`: every
    # DD-QR CBE takes the predictor + one-shot + independent-holdout controller
    # (`_dd_hybrid`), its predictor carried across steps in `dd_memory` (a
    # `DDMemory`, which also traces every call). `:adaptive` is the existing D1.
    dd_policy::Symbol = :adaptive,
    dd_memory = nothing,
    # The sweeps' basis-selection strategy (Core/RankAdaptation/basis_expansion.jl).
    expansion::BasisExpansion = DDQRExpansion(),
    rng::AbstractRNG = MersenneTwister(0x5EED),
)
    # Check the keywords against each other and resolve their `nothing` defaults
    # (`bug_step_stages.jl`).
    options = _resolve_step_options(
        psi,
        mpo,
        root,
        parallel,
        kls;
        tol_basis = tol_basis,
        tol_trunc = tol_trunc,
        tol_expv = tol_expv,
        tol_root = tol_root,
        order = order,
        fold_omega = fold_omega,
        exact = exact,
        frame_state = frame_state,
        closure_probe = closure_probe,
        outside_residual = outside_residual,
        maxdim = maxdim,
    )
    L, c, budget = options.site_count, options.root_link, options.budget
    parallel_sym, parallel, kls = options.parallel_mode, options.run_parallel, options.kls
    fold_omega, maxdim = options.fold_omega, options.maxdim

    # The per-bond, per-test allowances `tol_basis` buys, and the `|tau|/||psi||` scale every
    # residual is measured in. The norm comes from the centre tensor, which carries the whole norm
    # only while the state is canonical (the sweeps gauge before reading). An out-of-range `center`
    # falls back to 1.0 rather than erroring in a diagnostic.
    _nrm = (1 <= psi.center <= L) ? norm(psi[psi.center]) : 1.0
    # Plan T3.2: DD-QR spends `tol_basis` as a linear sum of shares.
    alloc = basis_allocation(
        budget,
        L - 1;
        tau = tau,
        nrm = _nrm,
        nfront = order,
        rule = :linear,
        krylov = tol_krylov,
    )
    ledger = AdaptiveLedger()
    ledger.order = order
    bd0 = bond_dims(psi)                 # per-bond `r_old` for the bond log (plan §8)
    ledger.r_old = maximum(bd0; init = 0)
    bwts = zeros(L - 1)                  # per-bond discarded norm of the compression

    expanded = zeros(Int, L - 1)
    # ⚠ ONE ARRAY PER HALF-SWEEP: bond `c` is written by BOTH (`n_new_l` from the left sweep,
    # `n_new_r` from the right), so a single shared array is a genuine write conflict there --
    # benign when the sweeps ran in sequence and last-writer-wins, undefined when they run
    # concurrently. Merged after the join with the RIGHT sweep winning at `c`, which is exactly
    # what the sequential order used to produce.
    n_newL = zeros(Int, L - 1)
    n_newR = zeros(Int, L - 1)
    accL = _HalfAcc()
    accR = _HalfAcc()

    # The ONE work ceiling, shared by the CBE sketch and the Krylov enrichment (plan §4/§5), read
    # ONCE, globally, before anything widens.
    wcap_step = work_ceiling(maxdim, maximum(bond_dims(psi); init = 0), local_dim(psi))
    # The expansion settings every DD-QR CBE of the step takes, half-sweeps and frontier sweeps.
    exkw = (
        growth = ADAPTIVE_GROWTH,
        exact = exact,
        alloc = alloc,
        workcap = wcap_step,
        cap_order = cap_order,
        dd_width = dd_width,
        dd_policy = dd_policy,
        dd_memory = dd_memory,
    )

    # ⛔ THE LEDGER AND THE RNG ARE PER HALF-SWEEP, because under `parallel = true` the two run
    # concurrently: a shared `AdaptiveLedger` loses `+=` updates, and a shared `MersenneTwister`
    # is not thread-safe at all. The right sweep gets a stream SEEDED FROM `rng` only when
    # parallel, so the serial path draws exactly the sequence it always did. The two ledgers are
    # folded into the step's after the join.
    ledgerL, ledgerR = AdaptiveLedger(), AdaptiveLedger()
    rngR = parallel ? MersenneTwister(rand(rng, UInt64)) : rng
    rec = krylov_recycle && order > 1 ? KrylovRecycle(recycle_mode) : nothing
    cx = _HalfSweepCtx(
        mpo,
        tau,
        eshift,
        c,
        alloc,
        expansion,
        exkw,
        wcap_step,
        krylov_reuse,
        krylov_depth,
        keep_states,
        rec,
    )

    troot = Ref(0.0)
    ttrunc = Ref(0.0)
    _t0step = time_ns()
    _b0step = Base.gc_bytes()
    W = Vector{Any}(undef, c)          # augmented LEFT frames,  sites 1 … c
    Z = Vector{Any}(undef, L - 1)      # augmented RIGHT frames, Z[j] absorbs site j+1

    # ── `frame_state`: bases from one state, evolution from another ──
    # The midpoint BUG integrator (arXiv:2402.08607) takes its bases from `Y_{1/2}` but integrates
    # the Galerkin ODE over the full step from `Y_0`. The start vector `S0 = <frames|Y_0>` must not
    # come from the half-step state, or the second half would be integrated twice. `psi` stays
    # L-canonical so that all assembly reads one gauge; with no `frame_state`, `psiR` aliases `psi`.
    psiL = copy(frame_state === nothing ? psi : frame_state)
    psiR = frame_state === nothing ? psi : copy(frame_state)

    # ── RUN THEM ── (`parallel` needs `julia -t 2`; why it is safe: `half_sweeps.jl`)
    # `lenv_root` / `renv_root`: the root environments at links `c` / `c+2`, in the NEW basis;
    # the right sweep's final carry IS `AR` (`_half_right!`).
    run_left() = _half_left!(psiL, W, expanded, n_newL, accL, ledgerL, rng, cx)
    run_right() = _half_right!(psiR, Z, expanded, n_newR, accR, ledgerR, rngR, cx)
    if parallel
        tR = Threads.@spawn run_right()
        lenv_root = run_left()
        renv_root, AR = fetch(tR)
    else
        lenv_root = run_left()
        renv_root, AR = run_right()
    end
    merge_ledger!(ledger, ledgerL)
    merge_ledger!(ledger, ledgerR)

    # `n_new` at bond `c` is written by BOTH sweeps; the right one wins, which is what the old
    # sequential order produced.
    n_new = vcat(n_newL[1:(c-1)], n_newR[c:(L-1)])
    epre = max(accL.epre, accR.epre)
    efnl = max(accL.efnl, accR.efnl)

    # ── with a separate `frame_state`, the right sweep no longer touches `psi` ──
    # Gauge: `AL` and `AR` must be contracted in one gauge, since mixing them is silently wrong
    # rather than an error. `AR` must be `<Z|psi>`, not the right carry `<Z|frame_state>`: the
    # midpoint's Galerkin step starts from `Y_0`, and reusing the carry integrates the first
    # half-step twice, quietly making the method first order.
    if frame_state !== nothing
        canonical!(psi, L)
        AR = _overlap_right(psi, Z, c)           # mirrors `_half_right!`'s `D` recursion on `psi`
    end

    AL = _overlap_left(psi, W, c)                                     # (b_L, link_{c+1})
    S0 = to_concrete!(contract(AL, (2,), AR, (1,)))                   # (b_L, b_R)

    # ── GENERAL ORDER: for `order > 1` close the root generator's frontiers j = 1..order-1,
    # growing `W` and `Z` in place; `closure_probe` only measures them (`bug_step_stages.jl`).
    nclose = 0
    closure = _close_root_frontiers!(
        KLFrames(W, Z, c, lenv_root, renv_root, 0, nothing, nothing, Float64(eshift)),
        psi,
        mpo,
        S0,
        tau,
        alloc,
        exkw,
        expanded,
        ledger;
        order = order,
        closure_probe = closure_probe,
        closure_checks = closure_checks,
        expansion = expansion,
        krylov_reuse = krylov_reuse,
        recycle = rec,
        recycle_report = recycle_report,
        krylov_depth = krylov_depth,
        workcap = wcap_step,
        norm_scale = _nrm,
        parallel = parallel,
        rng = rng,
        fold = fold_omega,
    )
    if closure !== nothing
        lenv_root, renv_root, S0 = closure.lenv, closure.renv, closure.root_start
        nclose = closure.operator_applications
    end
    # PLAN §5 `state_projection_error`: how much of `psi` the frames miss. On the ordinary step the
    # frames contain `psi`, so this is roundoff; on the midpoint's second stage it is the genuine
    # error of projecting `Y_0` onto frames built from `Y_{1/2}`, charged against `tol_basis`. The
    # norm is taken from the state, not from `sqrt(||psi||^2 - ||S0||^2)`, which is pure
    # cancellation at ~1e-8 and would flag steps spuriously.
    npsi = norm(psi[psi.center])
    ledger.r_aug = max(maximum(expanded; init = 0), leg_dim(Z[c], 1))
    ledger.proj_err = npsi > 0 ? _projection_miss(psi, W, Z, c) / npsi : 0.0
    ledger.proj_err > budget.basis && _unmet!(ledger, :projection)
    peak = accL.peak + accR.peak + sum(tensor_elements(t) for t in (W[c], Z[c], S0); init = 0)

    # Plan Task 6: the K and L stages of the parallel K/L/S step start from `psi` in these frames
    # and run beside the root solve (the S stage), and `_kls_finish` merges the three. `tol_expv`
    # drives the root solve's substep split (`substeps = 0`), so the iteration count follows from
    # it; the basis cap and breakdown threshold are numerical settings only.
    kls_stage(apply, x, rep) = expv(
        apply,
        tau,
        x;
        hermitian = hermitian,
        maxiter = _ROOT_MAXITER,
        tol = 1e-15,
        reorth = reorth,
        conv_tol = budget.expv,
        substeps = 0,
        report = rep,
        root_method = root_method,
    )
    klsst =
        kls === nothing ? nothing :
        _kls_start(psi, mpo, W, Z, c, lenv_root, renv_root, kls_stage, eshift, kls)

    kroot = KernelCounters()
    H0 = RootOp(shift_h(zero_site_h(mpo, c, lenv_root, renv_root, W[c], Z[c]), eshift), c, kroot)
    nroot = Ref(0)
    rootrep = Ref{Any}(nothing)
    _t0root = time_ns()
    # The root solver for any root operator; it counts every application in `nroot`.
    solve_with(operator, S) =
        kls_stage(x -> (nroot[] += 1; apply_zero_site(operator, x)), S, rootrep)
    S1 = solve_with(H0, S0)
    troot[] = (time_ns() - _t0root) / 1e9
    if root_report !== nothing
        # Energies, the outside-Galerkin residual and its re-enrichment (`bug_step_stages.jl`).
        report = _report_root!(
            root_report,
            psi,
            mpo,
            W,
            Z,
            c,
            lenv_root,
            renv_root,
            H0,
            S0,
            S1,
            solve_with;
            tau = tau,
            eshift = eshift,
            alloc = alloc,
            expansion_settings = exkw,
            expansion = expansion,
            krylov_depth = krylov_depth,
            expanded = expanded,
            ledger = ledger,
            root_seconds = troot,
            outside_residual = outside_residual,
            tol_basis = budget.basis,
            order = order,
            workcap = wcap_step,
            norm_scale = _nrm,
            parallel = parallel,
            rng = rng,
            fold = fold_omega,
            root_method = root_method,
            root_solve_report = rootrep,
            root_applications = nroot,
        )
        S0, S1 = report.root_start, report.root_result
        lenv_root, renv_root = report.lenv, report.renv
    end
    nkls = 0
    if klsst !== nothing
        # The four-block core in the grown root frames, or -- when the omitted new-new block is
        # above `tol_basis` -- the standard augmented root solve in them (plan T6.3).
        solve_aug(h, S) = (op = RootOp(shift_h(h, eshift), c, kroot);
        kls_stage(x -> (nroot[] += 1; apply_zero_site(op, x)), S, rootrep))
        kr = _kls_finish(
            klsst,
            mpo,
            W,
            Z,
            c,
            lenv_root,
            renv_root,
            S0,
            S1,
            tau,
            solve_aug;
            nrm = _nrm,
            tol_missing = budget.basis,
        )
        W[c], Z[c], S1 = kr.W, kr.Z, kr.S1
        merge_counters!(kroot, kr.kc)
        nkls = kr.account.n_K + kr.account.n_L
        ledger.kls = kr.account
        ledger.r_aug = max(ledger.r_aug, leg_dim(W[c], 3), leg_dim(Z[c], 1))
        max(kr.account.err_K, kr.account.err_L) > budget.expv * (1 + 1e-8) && _unmet!(ledger, :expv)
    end
    if rootrep[] !== nothing
        ledger.expv_err = rootrep[].err
        ledger.expv_iters = rootrep[].iterations
        # Relative slack: each substep is accepted at `tol_expv`, so only a sum over several
        # substeps (or a step that could not advance) can exceed it.
        rootrep[].err > budget.expv * (1 + 1e-8) && _unmet!(ledger, :expv)
    end

    # ⛔ THE ROOT CORE SPLIT IS LOSSLESS (plan §5). This SVD only splits `S1` into the two site
    # tensors below: `cutoff = 0.0` keeps every singular value with `sv > 0` (Telum's selector is
    # relative to `sv_max`) and there is no `Nkeep`, so the closing pass is the step's one
    # compression -- and, with `truncate = false`, there is no rank control at the root at all.
    frames_report === nothing || (
        frames_report[] =
            KLFrames(W, Z, c, lenv_root, renv_root, 0, nothing, nothing, Float64(eshift))
    )
    res = svd(S1, (1,); cutoff = 0.0, get_lists = true)
    expanded[c] = leg_dim(W[c], 3)
    # bytes by hand, not `_measure`: a closure here would box the reassigned `maxdim`
    bcomp = Base.gc_bytes()
    root_rank, discarded, ttrunc[] = _assemble_and_compress!(
        psi,
        W,
        Z,
        c,
        res,
        ledger,
        bwts,
        bd0;
        maxdim = maxdim,
        tol_trunc = budget.trunc,
        truncate = truncate,
    )
    bcomp = Base.gc_bytes() - bcomp

    # ⚠ THE PHASE TIMERS ARE SUMS ACROSS BOTH HALF-SWEEPS, i.e. CPU time, while `t_step` is WALL
    # clock. Under `parallel = false` they are the same clock and the shares sum to ~1 as before.
    # Under `parallel = true` they DELIBERATELY DO NOT: two sweeps running at once spend more CPU
    # seconds than the step takes, and a share above 1 is the speedup showing up, not a bug. Keep
    # comparing phases arm-to-arm in seconds; only the serial arm's shares are fractions of a step.
    t_step = (time_ns() - _t0step) / 1e9
    kc =
        merge_counters!(merge_counters!(merge_counters!(KernelCounters(), accL.kc), accR.kc), kroot)
    kc.b_compress += bcomp
    rec = StepRecord(
        :dd_qr_streamed,
        _qr_backend(),
        order > 1 ? :dd_qr_streamed : :none,
        order,
        kls === nothing ? parallel_sym : :full,
        kc,
        maximum(bd0; init = 0),
        max(maximum(expanded; init = 0), leg_dim(Z[c], 1)),
        maximum(bond_dims(psi); init = 0),
        maxdim,
        wcap_step,
        nroot[],
        t_step,
        Base.gc_bytes() - _b0step,
        Int(Sys.maxrss()),
        _step_limits(ledger),
        alloc.rule,
    )
    return CBEBugSweepInfo(
        expanded,
        n_new,
        c,
        root_rank,
        epre,
        efnl,
        discarded,
        accL.nmv + accR.nmv + nroot[] + nclose + nkls,
        peak,
        accL.ngrow + accR.ngrow,
        leg_dim(Z[c], 1),
        accL.tcbe + accR.tcbe,
        accL.tkry + accR.tkry,
        troot[],
        ttrunc[],
        accL.tsplit + accR.tsplit,
        accL.tframe + accR.tframe,
        accL.tenv + accR.tenv,
        t_step,
        ledger,
        rec,
    )
end

cbe_bug_step!(psi::SymMPS, mpo::MPO, tau::Number; kwargs...) =
    cbe_bug_step!(psi, mpo, ComplexF64(tau); kwargs...)

# ── from BUG/Core/Sweeps/parallel_kls.jl: the step entry point of the K/L/S kernel there ──

"""
    parallel_kls_step!(psi, mpo, tau; schedule = :concurrent, kwargs...) -> CBEBugSweepInfo

ONE step of the fully parallel K/L/S integrator (plan Task 6): the preliminary half-sweeps (and,
for `order = p > 1`, the frontier closure) of [`cbe_bug_step!`](@ref), then the K, L and root S
equations from `psi` in those frames, the four-block reconstruction and the one compression.
Takes `cbe_bug_step!`'s normal-API keywords. `schedule` is `:sequential` (K, L, then S),
`:kl_then_s` (K and L concurrently, then S) or `:concurrent` (all three); the three give the
same state up to the order of floating-point reductions. `ledger.kls` reports the new directions,
the omitted block `missing` and whether the step fell back to the standard augmented root solve.
"""
parallel_kls_step!(psi::SymMPS, mpo, tau; schedule::Symbol = :concurrent, kwargs...) =
    cbe_bug_step!(psi, mpo, tau; kwargs..., kls = schedule)

# From BUG-Julia@b155550:src/RSVDCBEBondUpdate/qtt/qtt_flows.jl.
"Overwrite `psi` with the tensors and gauge centre of `src` (same length)."
function _replace_state!(psi::SymMPS, src::SymMPS)
    length(psi) == length(src) ||
        throw(DimensionMismatch("_replace_state!: $(length(psi)) vs $(length(src)) sites"))
    for i = 1:length(psi)
        psi[i] = src[i]
    end
    psi.center = src.center
    return psi
end
