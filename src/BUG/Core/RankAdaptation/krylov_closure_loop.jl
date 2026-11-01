# Moved from BUG-Julia@b155550:src/RSVDCBEBondUpdate/bug/krylov_enrich.jl,
# src/RSVDCBEBondUpdate/bug/krylov_closure.jl, src/RSVDCBEBondUpdate/bug/krylov_closure_loop.jl.
# krylov_closure_loop.jl: the order-p closure drivers `_krylov_closure!` and
# `_single_pass_closure!`, on the frames of `krylov_closure_frames.jl` (included before it); the
# frontier sweeps they call live in `Sweeps/frontier_sweeps.jl` and `Sweeps/frontier_closure.jl`.

# ── from src/RSVDCBEBondUpdate/bug/krylov_closure_loop.jl ──

"""
    _krylov_closure!(fr, psi, mpo, root_start, tau, alloc, ckw; order, workcap, nrm, parallel,
                     rng, ledger, include_zero = false, maxpass, checks = false) -> NamedTuple

Close the frontiers of degree `0..order-1` of the frames `fr` in place, for `psi` with root
coefficients `root_start`. `checks = false` (default) runs one enriching sweep per frontier, each
rebuilt in the frames the previous one grew, and each sweep measures its own leaks.
`checks = true` first measures every frontier on the incoming frames
([`_measure_frontiers`](@ref)) and returns `fr` untouched if all close; otherwise it enriches and
then measures degree 0 on the final frames. Non-convergence is `:closure`, never a silent pass;
`capped` marks a stop at the working-rank limit ([`_closure_loop!`](@ref)). The returned `S0` is
the root coefficients in the final frames.
"""
function _krylov_closure!(
    fr::KLFrames,
    psi::SymMPS,
    mpo,
    root_start,
    tau::ComplexF64,
    alloc::BasisAllocation,
    ckw::NamedTuple;
    order::Int,
    workcap::Int = 0,
    nrm::Float64 = 1.0,
    parallel::Bool = false,
    rng::AbstractRNG = MersenneTwister(0x5EED),
    ledger::Union{Nothing,AdaptiveLedger} = nothing,
    include_zero::Bool = false,
    maxpass::Int = _closure_maxpass(order),
    fold::Bool = false,
    checks::Bool = false,
)
    start_time = time_ns()
    lg = AdaptiveLedger()
    lg.order = order
    sweepkw = (workcap = workcap, nrm = nrm, parallel = parallel, rng = rng, fold = fold)
    checks || return _single_pass_closure!(
        fr,
        psi,
        mpo,
        root_start,
        tau,
        alloc,
        ckw,
        lg,
        start_time;
        order = order,
        include_zero = include_zero,
        ledger = ledger,
        sweepkw...,
    )
    # CHECK FIRST: what the incoming frames (the ordinary CBE + local-Krylov half-sweeps)
    # already capture of the global Krylov sequence. If they close every frontier, no closure
    # sweep runs and the frames stay exactly as the half-sweeps built them.
    pre = _measure_frontiers(fr, psi, mpo, root_start, tau, alloc, ckw, order; sweepkw...)
    merge_ledger!(lg, pre.ledger)
    nroot = Ref(pre.n_root)
    # The root coefficients in the current frames, re-projected whenever a sweep grows them.
    current_root = Ref{Any}(root_start)
    # `pre.leak[slot]` is the leak of the frontier of degree `slot - 1`.
    final_leak = Dict{Int,Float64}(slot - 1 => pre.leak[slot] for slot = 1:order)
    gen0 = fr.gen
    res = if pre.met && (pre.met0 || !include_zero)
        (passes = 0, converged = true, exhausted = false, capped = false, log = Any[])
    else
        root_operator = Ref{Any}(_root_generator(mpo, fr))
        applyB = vector -> (nroot[] += 1; apply_zero_site(root_operator[], vector))
        # The frontier of `degree` in the current frames.
        frontier =
            degree -> KrylovFrontier(
                degree == 0 ? current_root[] :
                _projected_frontiers(applyB, current_root[], tau, degree)[degree],
                degree,
                fr.gen,
            )
        # ⚠ SPECULATIVE FINAL MEASUREMENT (>= _FRONTIER_THREADS threads). A later pass usually
        # adds nothing and is then the last; its last sweep cannot change the frames unless it
        # adds. So degree 0 is measured CONCURRENTLY on a snapshot of the frames, and kept only
        # if the loop ends in that generation -- otherwise it is discarded and redone below.
        spec = Ref{Any}(nothing)
        r0kw = (workcap = workcap, nrm = nrm, parallel = parallel, fold = fold, enrich = false)
        # One enriching sweep of `krylov_frontier` (of `degree`) in pass `pass`; re-projects the
        # root when it grew the frames.
        sweep! =
            (krylov_frontier, degree, pass) -> begin
                if _frontier_parallel(parallel) && pass >= 2 && degree == order - 1
                    spec[] === nothing || fetch(spec[].task)
                    snap = KLFrames(
                        copy(fr.left_frames),
                        copy(fr.right_frames),
                        fr.root_link,
                        fr.lenv,
                        fr.renv,
                        fr.gen,
                        fr.lcache,
                        fr.rcache,
                        fr.eshift,
                    )
                    zero_frontier = KrylovFrontier(current_root[], 0, snap.gen)
                    speculative_rng = MersenneTwister(rand(rng, UInt64))
                    spec[] = (
                        gen = snap.gen,
                        task = Threads.@spawn(
                            _frontier_sweep!(
                                snap,
                                psi,
                                mpo,
                                zero_frontier,
                                tau,
                                alloc,
                                ckw;
                                r0kw...,
                                rng = speculative_rng,
                                pass = pass + 1,
                            )
                        )
                    )
                end
                sweep_result = _frontier_sweep!(
                    fr,
                    psi,
                    mpo,
                    krylov_frontier,
                    tau,
                    alloc,
                    ckw;
                    sweepkw...,
                    pass = pass,
                )
                merge_ledger!(lg, sweep_result.ledger)
                final_leak[degree] = sweep_result.leak
                if sweep_result.added > 0
                    root_operator[] = _root_generator(mpo, fr)
                    current_root[] = _project_root(psi, fr)
                end
                sweep_result
            end
        loop = _closure_loop!(
            frontier,
            sweep!,
            order;
            maxpass = maxpass,
            include_zero = include_zero,
        )
        # The speculative measurement's result; its NamedTuple field `r` is the sweep result.
        sr = spec[] === nothing ? nothing : (gen = spec[].gen, sweep = fetch(spec[].task))
        if fr.gen == gen0
            # Nothing grew: the frames and `root_start` are the ones `pre` measured degree 0 on,
            # so its measurement IS the final one (`final_leak[0]` already holds it).
        elseif sr !== nothing && sr.gen == fr.gen
            merge_ledger!(lg, sr.sweep.ledger)
            final_leak[0] = sr.sweep.leak
        else
            final_sweep = _frontier_sweep!(
                fr,
                psi,
                mpo,
                frontier(0),
                tau,
                alloc,
                ckw;
                r0kw...,
                rng = rng,
                pass = loop.passes + 1,
            )
            merge_ledger!(lg, final_sweep.ledger)
            final_leak[0] = final_sweep.leak
        end
        loop
    end
    lg.n_passes = res.passes
    lg.leak = [final_leak[degree] for degree = 0:(order-1)]
    # The defect weights frontier `degree` by 1 / (degree + 1).
    lg.defect = sum(lg.leak[degree+1] / (degree + 1) for degree = 0:(order-1))
    lg.n_root_frontier = nroot[]
    res.converged || _unmet!(lg, :closure)
    lg.t_closure = (time_ns() - start_time) / 1e9
    ledger === nothing || merge_ledger!(ledger, lg)
    return (
        root_start = current_root[],
        passes = res.passes,
        converged = res.converged,
        exhausted = res.exhausted,
        capped = res.capped,
        log = res.log,
        ledger = lg,
        n_root = nroot[],
        n_h1 = lg.n_h1_closure,
    )
end

"""
    _single_pass_closure!(fr, psi, mpo, root_start, tau, alloc, ckw, lg, start_time; order,
                          include_zero, ledger, sweepkw...) -> NamedTuple

[`_krylov_closure!`](@ref) with `checks = false`: one enriching sweep per frontier, each rebuilt
in the frames the previous sweep grew, and nothing measured before or after. Same return fields.
"""
function _single_pass_closure!(
    fr::KLFrames,
    psi::SymMPS,
    mpo,
    root_start,
    tau::ComplexF64,
    alloc::BasisAllocation,
    ckw::NamedTuple,
    lg::AdaptiveLedger,
    start_time::UInt64;
    order::Int,
    include_zero::Bool,
    ledger::Union{Nothing,AdaptiveLedger},
    sweepkw...,
)
    nroot = Ref(0)
    current_root = Ref{Any}(root_start)
    root_operator = Ref{Any}(_root_generator(mpo, fr))
    applyB = vector -> (nroot[] += 1; apply_zero_site(root_operator[], vector))
    leak = fill(NaN, order)                      # NaN: not measured (degree 0 unless include_zero)
    log = Any[]
    met, capped = true, false
    for degree = (include_zero ? 0 : 1):(order-1)
        core =
            degree == 0 ? current_root[] :
            _projected_frontiers(applyB, current_root[], tau, degree)[degree]
        sweep_result = _frontier_sweep!(
            fr,
            psi,
            mpo,
            KrylovFrontier(core, degree, fr.gen),
            tau,
            alloc,
            ckw;
            sweepkw...,
            enrich = true,
            pass = 1,
        )
        merge_ledger!(lg, sweep_result.ledger)
        push!(
            log,
            (
                pass = 1,
                frontier = degree,
                added = sweep_result.added,
                leak = sweep_result.leak,
                met = sweep_result.met,
            ),
        )
        leak[degree+1] = sweep_result.leak     # before this sweep's enrichment
        met &= sweep_result.met
        capped |= sweep_result.capped
        if sweep_result.added > 0
            root_operator[] = _root_generator(mpo, fr)
            current_root[] = _project_root(psi, fr)
        end
    end
    lg.n_passes = 1
    lg.leak = leak
    lg.defect = sum(
        (leak[degree+1] / (degree + 1) for degree = 0:(order-1) if !isnan(leak[degree+1]));
        init = 0.0,
    )
    lg.n_root_frontier = nroot[]
    met || _unmet!(lg, :closure)
    lg.t_closure = (time_ns() - start_time) / 1e9
    ledger === nothing || merge_ledger!(ledger, lg)
    return (
        root_start = current_root[],
        passes = 1,
        converged = met,
        exhausted = false,
        capped = capped && !met,
        log = log,
        ledger = lg,
        n_root = nroot[],
        n_h1 = lg.n_h1_closure,
    )
end
