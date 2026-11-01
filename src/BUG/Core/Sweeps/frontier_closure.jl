# Moved from BUG-Julia@b155550:src/RSVDCBEBondUpdate/bug/frontier_sweeps.jl.

# frontier_closure.jl: one frontier's sweep on both sides (`_frontier_sweep!`), the measurements
# built on it and the closure's termination loop (split from `frontier_sweeps.jl`).

_frame_width(fr::KLFrames, L::Int) = max(
    maximum(leg_dim(fr.W[i], 3) for i = 1:fr.c),
    maximum(leg_dim(fr.Z[j], 1) for j = fr.c:(L-1)),
)

"""
    _frontier_sweep!(fr, psi, mpo, F, tau, alloc, ckw; enrich, workcap, nrm, parallel, rng,
                     pass, nsample) -> NamedTuple

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
    F::KrylovFrontier,
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
    F.gen == fr.gen || throw(
        ArgumentError(
            "stale frontier: built in frames generation $(F.gen), the frames are at $(fr.gen) -- " *
            "the root operator and frontiers must be rebuilt whenever the frames change",
        ),
    )
    t0 = time_ns()
    L, c = length(psi), fr.c
    nF = norm(F.core)
    zero_leaks = (zeros(c), zeros(L - 1))
    if nF <= _FRONTIER_FLOOR * max(nrm, eps())
        lg = AdaptiveLedger()
        push!(
            lg.closure_log,
            (
                pass = pass,
                frontier = F.degree,
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
    Fm = _frontier_mps(psi, fr, F.core)
    FmR = copy(Fm)
    gen = fr.gen + 1
    amp = nF / max(nrm, eps())
    rngR = parallel ? MersenneTwister(rand(rng, UInt64)) : rng
    kw(r) = (
        enrich = enrich,
        gen = gen,
        workcap = workcap,
        amp = amp,
        rng = r,
        nsample = nsample,
        fold = fold,
        bonds = !enrich && _BOND_TASKS[] && _frontier_parallel(parallel),
    )
    rl, rr = if parallel
        tR = Threads.@spawn _frontier_right(FmR, mpo, fr, tau, alloc, ckw; kw(rngR)...)
        l = _frontier_left(Fm, mpo, fr, tau, alloc, ckw; kw(rng)...)
        (l, fetch(tR))
    else
        (
            _frontier_left(Fm, mpo, fr, tau, alloc, ckw; kw(rng)...),
            _frontier_right(FmR, mpo, fr, tau, alloc, ckw; kw(rngR)...),
        )
    end
    lg = AdaptiveLedger()
    for half in (rl.ledger, rr.ledger)
        half.bond_log = Any[merge(e, (pass = pass, frontier = F.degree)) for e in half.bond_log]
        merge_ledger!(lg, half)
    end
    lL, lR = sqrt(sum(abs2, rl.leak)), sqrt(sum(abs2, rr.leak))
    leak = indicator(alloc, sqrt(lL^2 + lR^2))
    met =
        enrich ? lg.tolerance_met :
        all(indicator(alloc, x) <= alloc.per_bond for x in vcat(rl.leak, rr.leak))
    added = rl.added + rr.added
    wbefore = _frame_width(fr, L)
    if enrich           # the sweep's recycle entries live in its frames: usable only if committed
        rec = get(ckw, :recycle, nothing)
        _recycle_commit!(rec, :left, added > 0)
        _recycle_commit!(rec, :right, added > 0)
    end
    if enrich && added > 0
        for i = 1:c
            fr.W[i] = rl.frames[i]
        end
        for j = c:(L-1)
            fr.Z[j] = rr.frames[j]
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
            frontier = F.degree,
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
            bonds = c + (L - c),
            width_before = wbefore,
            width_after = _frame_width(fr, L),
            n_h2 = lg.n_h2_leak,
            n_h1 = lg.n_h1_closure,
            n_probes = lg.n_probes,
            n_selected = lg.n_selected,
            n_krylov = lg.n_krylov,
            zero = false,
            capped = enrich && lg.workcap_reached,
            t = (time_ns() - t0) / 1e9,
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
    L, c = length(psi), fr.c
    alloc = basis_allocation(AdaptiveBudget(1.0, 0.0, 1.0), L - 1)    # scale 1: raw norms
    r = _frontier_sweep!(
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
    pl, pr = r.leaks_left, r.leaks_right[c:(L-1)]
    sl, sr = r.sleaks_left, r.sleaks_right[c:(L-1)]
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
    _measure_frontiers(fr, psi, mpo, S0, tau, alloc, ckw, m; nrm, parallel, rng, pass = 0)
        -> NamedTuple

Measure, never enrich, the frontiers `j = 0..m-1` of `S0` on the frames `fr` (left untouched): what
the frames already capture, `rho_j = ||(I - P) A F_j|| / ||F_j||`. Returns the time-weighted
`leak[j+1]`, scale-free `rho[j+1]`, `met` (all bonds of frontiers `j >= 1` within share), the merged
`ledger`, `n_root`, and `met0` (the `j = 0` verdict). With `parallel` the frontiers run
concurrently, each on its own RNG stream from `rng`; results merge in `j` order.
"""
function _measure_frontiers(
    fr::KLFrames,
    psi::SymMPS,
    mpo,
    S0,
    tau::ComplexF64,
    alloc::BasisAllocation,
    ckw::NamedTuple,
    m::Int;
    nrm::Float64 = 1.0,
    parallel::Bool = false,
    rng::AbstractRNG = MersenneTwister(0x5EED),
    pass::Int = 0,
    workcap::Int = 0,
    fold::Bool = false,
)
    H0 = _root_generator(mpo, fr)
    nroot = Ref(0)
    F =
        m > 1 ? _projected_frontiers(x -> (nroot[] += 1; apply_zero_site(H0, x)), S0, tau, m - 1) :
        Any[]
    lg = AdaptiveLedger()
    leak, rho, met, met0 = zeros(m), zeros(m), true, true
    cores = Any[j == 0 ? S0 : F[j] for j = 0:(m-1)]
    measure(j, r) = _frontier_sweep!(
        fr,
        psi,
        mpo,
        KrylovFrontier(cores[j+1], j, fr.gen),
        tau,
        alloc,
        ckw;
        enrich = false,
        workcap = workcap,
        nrm = nrm,
        parallel = parallel,
        rng = r,
        pass = pass,
        fold = fold,
    )
    rs = if _frontier_parallel(parallel) && m > 1
        _prime_caches!(fr, psi, mpo, S0)
        rngs = [MersenneTwister(rand(rng, UInt64)) for _ = 1:m]
        fetch.([Threads.@spawn(measure(j, rngs[j+1])) for j = 0:(m-1)])
    else
        [measure(j, rng) for j = 0:(m-1)]
    end
    for j = 0:(m-1)
        core, r = cores[j+1], rs[j+1]
        merge_ledger!(lg, r.ledger)
        nF = norm(core)
        raw = sqrt(sum(abs2, r.leaks_left) + sum(abs2, r.leaks_right))
        leak[j+1] = r.leak
        rho[j+1] = nF > 0 ? raw / nF : 0.0
        j >= 1 ? (met &= r.met) : (met0 = r.met)
    end
    lg.leak_initial, lg.rho_initial = leak, rho
    lg.n_root_frontier = nroot[]
    return (leak = leak, rho = rho, met = met, met0 = met0, ledger = lg, n_root = nroot[])
end

"""
    _closure_loop!(frontier, sweep!, order; maxpass, include_zero = true) -> NamedTuple

Termination logic of the order-`p` closure, independent of storage. `frontier(j)` builds frontier
`j` in the current space; `sweep!(F, j, pass)` tests and enriches, returning `(added, leak, met)`.
Pass 1 sweeps `j = 0..p-1` (`j = 1..p-1` if `j = 0` is closed); later passes sweep `j = 1..p-1`.
Converged means a pass added nothing and met every share. A pass with a share unmet and a sweep at
the work ceiling stops with `capped = true`, since those bonds cannot widen. `maxpass` is a safety
bound whose hit is reported.
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
        for j in (pass == 1 ? initial : later)
            r = sweep!(frontier(j), j, pass)
            push!(log, (pass = pass, frontier = j, added = r.added, leak = r.leak, met = r.met))
            grew |= r.added > 0
            allmet &= r.met
            capped |= get(r, :capped, false)
            j == 0 && (ok0 = r.met)
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
