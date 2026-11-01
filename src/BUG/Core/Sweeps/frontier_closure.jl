# Moved from BUG-Julia@b155550:src/RSVDCBEBondUpdate/bug/frontier_sweeps.jl.

# frontier_closure.jl: one frontier's sweep on both sides (`_frontier_sweep!`), the measurements
# built on it and the closure's termination loop (split from `frontier_sweeps.jl`).

# The widest frame of `fr` over all bonds of a `site_count`-site chain (closure log only).
_frame_width(fr::KLFrames, site_count::Int) = max(
    maximum(leg_dim(fr.left_frames[bond], 3) for bond = 1:fr.root_link),
    maximum(leg_dim(fr.right_frames[bond], 1) for bond = fr.root_link:(site_count-1)),
)

"""
    _frontier_sweep!(fr, psi, mpo, frontier, tau, alloc, ckw; enrich, workcap, nrm, parallel,
                     rng, pass, nsample) -> NamedTuple

Test one frontier's outside-space action on both half-sweeps and, when `enrich`, grow `fr` by the
validated directions (plan: `enrich_KL_from_frontiers!`). The frames are replaced -- and their
generation advanced -- only if something was added, so frontiers built in them stay valid
otherwise. `leak` is the time-weighted outside-space action `|tau| sqrt(left^2 + right^2) /
||psi||` of the frontier BEFORE this sweep enriched; `met` is whether every bond ended within
its share.
"""
function _frontier_sweep!(
    fr::KLFrames,
    psi::SymMPS,
    mpo,
    frontier::KrylovFrontier,
    tau::ComplexF64,
    alloc::BasisAllocation,
    ckw::NamedTuple;
    enrich::Bool = true,
    workcap::Int = 0,
    nrm::Float64 = 1.0,
    parallel::Bool = false,
    rng::AbstractRNG = MersenneTwister(0x5EED),
    pass::Int = 0,
    nsample::Int = 0,
    fold::Bool = false,
)
    frontier.gen == fr.gen || throw(
        ArgumentError(
            "stale frontier: built in frames generation $(frontier.gen), the frames are at " *
            "$(fr.gen) -- the root operator and frontiers must be rebuilt whenever the frames " *
            "change",
        ),
    )
    start_time = time_ns()
    site_count, root_link = length(psi), fr.root_link
    nF = norm(frontier.core)
    zero_leaks = (zeros(root_link), zeros(site_count - 1))
    # a negligible frontier has no action worth measuring: log it and report every share met
    if nF <= _FRONTIER_FLOOR * max(nrm, eps())
        lg = AdaptiveLedger()
        push!(
            lg.closure_log,
            (
                pass = pass,
                frontier = frontier.degree,
                enrich = enrich,
                norm = nF / max(nrm, eps()),
                leak = 0.0,
                added = 0,
                zero = true,
                t = 0.0,
            ),
        )
        return (
            added = 0,
            leak = 0.0,
            leak_left = 0.0,
            leak_right = 0.0,
            met = true,
            capped = false,
            ledger = lg,
            leaks_left = zero_leaks[1],
            leaks_right = zero_leaks[2],
            sleaks_left = zero_leaks[1],
            sleaks_right = zero_leaks[2],
        )
    end
    # each half-sweep regauges its own copy of the frontier MPS
    Fm = _frontier_mps(psi, fr, frontier.core)
    FmR = copy(Fm)
    gen = fr.gen + 1
    amp = nF / max(nrm, eps())
    rngR = parallel ? MersenneTwister(rand(rng, UInt64)) : rng
    # the keywords both half-sweeps share; only the RNG stream differs
    kw(sweep_rng) = (
        enrich = enrich,
        gen = gen,
        workcap = workcap,
        amp = amp,
        rng = sweep_rng,
        nsample = nsample,
        fold = fold,
        bonds = !enrich && _BOND_TASKS[] && _frontier_parallel(parallel),
    )
    rl, rr = if parallel
        tR = Threads.@spawn _frontier_right(FmR, mpo, fr, tau, alloc, ckw; kw(rngR)...)
        left_result = _frontier_left(Fm, mpo, fr, tau, alloc, ckw; kw(rng)...)
        (left_result, fetch(tR))
    else
        (
            _frontier_left(Fm, mpo, fr, tau, alloc, ckw; kw(rng)...),
            _frontier_right(FmR, mpo, fr, tau, alloc, ckw; kw(rngR)...),
        )
    end
    lg = AdaptiveLedger()
    for half in (rl.ledger, rr.ledger)
        half.bond_log =
            Any[merge(entry, (pass = pass, frontier = frontier.degree)) for entry in half.bond_log]
        merge_ledger!(lg, half)
    end
    lL, lR = sqrt(sum(abs2, rl.leak)), sqrt(sum(abs2, rr.leak))
    leak = indicator(alloc, sqrt(lL^2 + lR^2))
    met =
        enrich ? lg.tolerance_met :
        all(indicator(alloc, bond_leak) <= alloc.per_bond for bond_leak in vcat(rl.leak, rr.leak))
    added = rl.added + rr.added
    wbefore = _frame_width(fr, site_count)
    if enrich           # the sweep's recycle entries live in its frames: usable only if committed
        rec = get(ckw, :recycle, nothing)
        _recycle_commit!(rec, :left, added > 0)
        _recycle_commit!(rec, :right, added > 0)
    end
    if enrich && added > 0
        for bond = 1:root_link
            fr.left_frames[bond] = rl.frames[bond]
        end
        for bond = root_link:(site_count-1)
            fr.right_frames[bond] = rr.frames[bond]
        end
        fr.lenv, fr.renv, fr.gen = rl.env, rr.env, gen
        # the channels this sweep carried through the new frames ARE the new generation's chain
        fr.lcache = (gen = gen, chain = rl.chain, stack = nothing)
        fr.rcache = (gen = gen, chain = rr.chain, stack = nothing)
    end
    lg.n_closure_sweeps = 1
    lg.n_h2_leak = rl.n_h2 + rr.n_h2
    lg.n_h1_closure = rl.n_h1 + rr.n_h1
    push!(
        lg.closure_log,
        (
            pass = pass,
            frontier = frontier.degree,
            enrich = enrich,
            norm = amp,
            leak = leak,
            leak_left = indicator(alloc, lL),
            leak_right = indicator(alloc, lR),
            met = met,
            added = added,
            added_left = rl.added,
            added_right = rr.added,
            new_sectors = rl.new_sectors + rr.new_sectors,
            skipped = rl.skipped + rr.skipped,
            bonds = root_link + (site_count - root_link),
            width_before = wbefore,
            width_after = _frame_width(fr, site_count),
            n_h2 = lg.n_h2_leak,
            n_h1 = lg.n_h1_closure,
            n_probes = lg.n_probes,
            n_selected = lg.n_selected,
            n_krylov = lg.n_krylov,
            zero = false,
            capped = enrich && lg.workcap_reached,
            t = (time_ns() - start_time) / 1e9,
        ),
    )
    # `capped`: an enrichment of this sweep stopped at the work ceiling (`_closure_loop!`).
    return (
        added = added,
        leak = leak,
        leak_left = indicator(alloc, lL),
        leak_right = indicator(alloc, lR),
        met = met,
        capped = enrich && lg.workcap_reached,
        ledger = lg,
        leaks_left = rl.leak,
        leaks_right = rr.leak,
        sleaks_left = rl.sleak,
        sleaks_right = rr.sleak,
    )
end

"""
    estimate_global_complement_action(psi, mpo, fr, core; expansion, nsample = 0, rng)
        -> NamedTuple

The outside-space action `(I - P) H v` of the Galerkin vector `v = (W ⊗ Z) core`, from the per-bond
two-site misses of a non-enriching frontier sweep. For a nearest-neighbour `H` the per-bond pieces
are orthogonal, so

    max(left, right) <= ||(I - P) H v|| <= sqrt(left^2 + right^2)

with `left = ||(I - P_L ⊗ 1) H v||` and `right = ||(1 ⊗ (I - P_R)) H v||`. Longer-range terms make
these an estimate; `nsample > 0` adds randomised per-bond estimates (`sampled_*`). `expansion` is
the step's `BasisExpansion`, which measures the per-bond actions.
"""
function estimate_global_complement_action(
    psi::SymMPS,
    mpo,
    fr::KLFrames,
    core;
    expansion::BasisExpansion,
    nsample::Int = 0,
    rng::AbstractRNG = MersenneTwister(0x5EED),
)
    site_count, root_link = length(psi), fr.root_link
    alloc = basis_allocation(AdaptiveBudget(1.0, 0.0, 1.0), site_count - 1)    # scale 1: raw norms
    sweep_result = _frontier_sweep!(
        fr,
        psi,
        mpo,
        KrylovFrontier(core, 0, fr.gen),
        ComplexF64(0),
        alloc,
        (expansion = expansion,);
        enrich = false,
        nrm = max(norm(core), eps()),
        nsample = nsample,
        rng = rng,
    )
    # the right sweep's leaks are indexed by bond; keep the bonds right of the root
    pl, pr = sweep_result.leaks_left, sweep_result.leaks_right[root_link:(site_count-1)]
    sl, sr = sweep_result.sleaks_left, sweep_result.sleaks_right[root_link:(site_count-1)]
    left, right = sqrt(sum(abs2, pl)), sqrt(sum(abs2, pr))
    return (
        left = left,
        right = right,
        bound = sqrt(left^2 + right^2),
        per_left = pl,
        per_right = pr,
        sampled_left = sqrt(sum(abs2, sl)),
        sampled_right = sqrt(sum(abs2, sr)),
        per_sampled_left = sl,
        per_sampled_right = sr,
    )
end

"""
    _measure_frontiers(fr, psi, mpo, root_start, tau, alloc, ckw, frontier_count; nrm, parallel,
                       rng, pass = 0) -> NamedTuple

Measure, never enrich, the frontiers `j = 0..frontier_count-1` of `root_start` on the frames `fr`
(left untouched): what the frames already capture, `rho_j = ||(I - P) A F_j|| / ||F_j||`. Returns
the time-weighted `leak[j+1]`, scale-free `rho[j+1]`, `met` (all bonds of frontiers `j >= 1` within
share), the merged `ledger`, `n_root`, and `met0` (the `j = 0` verdict). With `parallel` the
frontiers run concurrently, each on its own RNG stream from `rng`; results merge in `j` order.
"""
function _measure_frontiers(
    fr::KLFrames,
    psi::SymMPS,
    mpo,
    root_start,
    tau::ComplexF64,
    alloc::BasisAllocation,
    ckw::NamedTuple,
    frontier_count::Int;
    nrm::Float64 = 1.0,
    parallel::Bool = false,
    rng::AbstractRNG = MersenneTwister(0x5EED),
    pass::Int = 0,
    workcap::Int = 0,
    fold::Bool = false,
)
    root_operator = _root_generator(mpo, fr)
    nroot = Ref(0)
    # frontiers 1..frontier_count-1 from the counted root action; frontier 0 is the root itself
    frontiers =
        frontier_count > 1 ?
        _projected_frontiers(
            vector -> (nroot[] += 1; apply_zero_site(root_operator, vector)),
            root_start,
            tau,
            frontier_count - 1,
        ) : Any[]
    lg = AdaptiveLedger()
    leak, rho, met, met0 = zeros(frontier_count), zeros(frontier_count), true, true
    cores =
        Any[frontier == 0 ? root_start : frontiers[frontier] for frontier = 0:(frontier_count-1)]
    # one non-enriching sweep of frontier `frontier` on the RNG stream `frontier_rng`
    measure(frontier, frontier_rng) = _frontier_sweep!(
        fr,
        psi,
        mpo,
        KrylovFrontier(cores[frontier+1], frontier, fr.gen),
        tau,
        alloc,
        ckw;
        enrich = false,
        workcap = workcap,
        nrm = nrm,
        parallel = parallel,
        rng = frontier_rng,
        pass = pass,
        fold = fold,
    )
    rs = if _frontier_parallel(parallel) && frontier_count > 1
        _prime_caches!(fr, psi, mpo, root_start)
        rngs = [MersenneTwister(rand(rng, UInt64)) for _ = 1:frontier_count]
        fetch.([
            Threads.@spawn(measure(frontier, rngs[frontier+1])) for frontier = 0:(frontier_count-1)
        ])
    else
        [measure(frontier, rng) for frontier = 0:(frontier_count-1)]
    end
    for frontier = 0:(frontier_count-1)
        core, result = cores[frontier+1], rs[frontier+1]
        merge_ledger!(lg, result.ledger)
        nF = norm(core)
        raw = sqrt(sum(abs2, result.leaks_left) + sum(abs2, result.leaks_right))
        leak[frontier+1] = result.leak
        rho[frontier+1] = nF > 0 ? raw / nF : 0.0
        frontier >= 1 ? (met &= result.met) : (met0 = result.met)
    end
    lg.leak_initial, lg.rho_initial = leak, rho
    lg.n_root_frontier = nroot[]
    return (leak = leak, rho = rho, met = met, met0 = met0, ledger = lg, n_root = nroot[])
end

"""
    _closure_loop!(frontier, sweep!, order; maxpass, include_zero = true) -> NamedTuple

Termination logic of the order-`p` closure, independent of storage. `frontier(degree)` builds
frontier `degree` in the current space; `sweep!(built, degree, pass)` tests and enriches the built
frontier, returning `(added, leak, met)`. Pass 1 sweeps `degree = 0..p-1` (`1..p-1` if `degree = 0`
is closed); later passes sweep `degree = 1..p-1`. Converged means a pass added nothing and met
every share. A pass with a share unmet and a sweep at the work ceiling stops with `capped = true`,
since those bonds cannot widen. `maxpass` is a safety bound whose hit is reported.
"""
function _closure_loop!(frontier, sweep!, order::Int; maxpass::Int, include_zero::Bool = true)
    order >= 1 || throw(ArgumentError("order must be >= 1, got $order"))
    log = Any[]
    later = 1:(order-1)
    initial = include_zero ? (0:(order-1)) : later
    isempty(initial) &&
        return (passes = 0, converged = true, exhausted = false, capped = false, log = log)
    ok0 = true
    for pass = 1:maxpass
        grew, allmet, capped = false, true, false
        for degree in (pass == 1 ? initial : later)
            result = sweep!(frontier(degree), degree, pass)
            push!(
                log,
                (
                    pass = pass,
                    frontier = degree,
                    added = result.added,
                    leak = result.leak,
                    met = result.met,
                ),
            )
            grew |= result.added > 0
            allmet &= result.met
            capped |= get(result, :capped, false)
            degree == 0 && (ok0 = result.met)
        end
        (!grew || isempty(later)) && return (
            passes = pass,
            converged = allmet && ok0,
            exhausted = false,
            capped = false,
            log = log,
        )
        capped &&
            !allmet &&
            return (passes = pass, converged = false, exhausted = false, capped = true, log = log)
    end
    return (passes = maxpass, converged = false, exhausted = true, capped = false, log = log)
end
