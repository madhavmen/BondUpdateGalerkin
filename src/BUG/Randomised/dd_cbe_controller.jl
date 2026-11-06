# DD-QR CBE controllers, which decide how many probe columns one side of a bond needs: the
# incremental D1 sketch controller (`_dd_controller`) and the D3 hybrid predictor controller
# (`_dd_hybrid`, with its per-bond memory `DDMemory`). The probe and QR kernels they call
# (`_dd_probe`, `_dd_select`) are in `dd_cbe.jl`; `dd_cbe_expand.jl` drives them for one side.

# ── the incremental DD sketch controller ──────────────────────────────────────────────────

"""
The default independent holdout width `holdout_width` (`s`; an
implementation constant; tests 2, 4, 8).
"""
const _DD_HOLDOUT = 4

"""
    _dd_controller(frame, side, draw, sketch, project, initial_width, wcap, ncap, alloc, ledger;
                   kc, holdout_width, audit, holdout, trace) -> (dd_basis, status)

Grows the DD basis `dd_basis` of one side block by block until an independent holdout says the
rest of `G = P^D X P^D` fits this side's share `alloc.sample`. Each probe block is streamed through
the MPO once and its action cached; a failed check draws a fresh holdout before promoting the old
one. The first block has `initial_width` columns; each holdout has `holdout_width` columns.

Statuses: `:met`; `:work_limited` (`wcap` or `ncap` reached); `:basis_limited` (a pass added no
direction); `:empty` (no discarded space). Every continuing pass adds a direction, so the loop ends.
"""
function _dd_controller(
    frame,
    side::Symbol,
    draw,
    sketch,
    project,
    initial_width::Int,
    wcap::Int,
    ncap::Int,
    alloc::BasisAllocation,
    ledger::Union{Nothing,AdaptiveLedger};
    kc::KernelCounters = KernelCounters(),
    holdout_width::Int = _DD_HOLDOUT,
    audit::Bool = false,
    holdout = width -> draw(width, nothing),
    trace::Union{Nothing,Vector} = nothing,
)
    left = side === :left
    # `fl` is the frame's new (growing) leg, `wleg` the probe-width leg of a sketch block.
    fl, wleg = left ? (3, 1) : (1, 3)
    tags = Telum.inds(frame)[fl].itags
    aug, dd_basis, direction_count, streamed = frame, nothing, 0, 0
    boost = Dict{Any,Int}()
    # QR the sketched block against the frame augmented so far, append the new directions to
    # `dd_basis` and `aug`, and return how many were added.
    function absorb!(sketched)
        new_basis, _ = _dd_select(aug, sketched, side, kc)
        new_basis === nothing && return 0
        new_basis = TLArray(setitag(new_basis, fl, tags))
        dd_basis =
            dd_basis === nothing ? new_basis : TLArray(oplus([dd_basis, new_basis], (fl,)))
        aug = TLArray(oplus([aug, new_basis], (fl,)))
        return leg_dim(new_basis, fl)
    end
    # Apply the MPO sketch to a probe block, counting its columns as streamed work (and in the
    # ledger field `field`). The assignment rebinds the enclosing `width`, a captured variable, so
    # after each call `width` is the width of the last block streamed (probe or holdout).
    function stream(Om, field::Symbol)
        width = leg_dim(Om, wleg)
        streamed += width
        ledger === nothing || setfield!(ledger, field, getfield(ledger, field) + width)
        return sketch(Om)
    end
    # A fresh holdout `(probe block, its sketched action)` of `columns` columns, or nothing.
    function fresh(columns)
        Om = holdout(columns)
        return Om === nothing ? nothing : (Om, stream(Om, :n_holdout))
    end
    # The sampled residual of `dd_basis` measured on one holdout pair.
    resid(check) = _sample_residual(dd_basis, project, _ -> check[2], _ -> check[1], 1, wleg, side)
    # Whether the basis reaches a charge sector the frame does not have yet.
    newsector() =
        dd_basis !== nothing &&
        any(((charge, _),) -> sector_dim(frame, fl, charge) == 0, Telum.spaces(dd_basis)[fl])

    width = clamp(initial_width, 1, max(wcap, 1))
    last = draw(width, nothing)
    last === nothing && return (nothing, :empty)
    direction_count += absorb!(stream(last, :n_probes))
    check = fresh(holdout_width)
    first_check = check
    escalated = false
    while true
        check === nothing && return (dd_basis, :met)            # nothing left to validate against
        est, per, hcols = resid(check)
        ind = indicator(alloc, est)
        trace === nothing ||
            push!(trace, (direction_count, ind, indicator(alloc, resid(first_check)[1])))
        ok = ind <= alloc.sample
        # A pass close to the line (or into a new sector) is confirmed on a twice wider holdout.
        if ok && !escalated && (audit || ind > alloc.sample / 2 || newsector())
            escalated = true
            wide_check = fresh(2 * holdout_width)
            wide_check === nothing || (check = wide_check; continue)
        end
        capped = streamed >= wcap || direction_count >= ncap
        if ok || capped
            ledger === nothing || _record_sample!(ledger, side, ind, per, hcols, last, wleg, alloc)
            ok && return (dd_basis, :met)
            if ledger !== nothing
                _unmet!(ledger, :sample)
                ledger.workcap_reached = true
            end
            return (dd_basis, :work_limited)
        end
        nt = _target_sectors!(boost, per, check[1], wleg, alloc)
        ledger === nothing || (ledger.n_targeted += nt)
        next_check = fresh(holdout_width)                  # new validation BEFORE the old one is promoted
        added = absorb!(check[2])
        nt == 0 && (width = min(2 * width, wcap))
        blk = draw(width, isempty(boost) ? nothing : boost)
        if blk !== nothing
            last = blk
            added += absorb!(stream(blk, :n_probes))
        end
        direction_count += added
        if added == 0
            ledger === nothing ||
                (_record_sample!(ledger, side, ind, per, hcols, last, wleg, alloc);
                _unmet!(ledger, :sample))
            return (dd_basis, :basis_limited)
        end
        check = next_check
    end
end

# ── arm D3: the hybrid predictor controller ───────────────────────────────────────────────

"Bounded oversampling of D3's first block over its predictor: `clamp(cld(pred, 4), 4, 8)` columns."
_dd_oversample(pred::Int) = clamp(cld(pred, 4), 4, 8)

"""
    DDMemory()

Per-bond memory of the D3 hybrid controller (`dd_policy = :hybrid`): the directions accepted last
time at `(side, bond)` -- the PREDICTOR of the next first block, never a correctness condition --
and one trace entry per call (`bond`, `side`, `branch`, `ell`, `s`, `retries`, `m`, `n`, `added`,
`streamed`, `status`; `m`, `n` are the discarded dimensions of this side and of the probe side)
for the first-impressions report (reported for every sampled bond). Locked, since the
two half-sweeps may run concurrently.
"""
mutable struct DDMemory
    accepted::Dict{Tuple{Symbol,Int},Int}
    trace::Vector{NamedTuple}
    lock::ReentrantLock
end
DDMemory() = DDMemory(Dict{Tuple{Symbol,Int},Int}(), NamedTuple[], ReentrantLock())

"The number of directions accepted last time at `(side, bond)`, or `nothing` on a first visit."
_dd_predict(mem::DDMemory, side::Symbol, bond::Int) =
    lock(() -> get(mem.accepted, (side, bond), nothing), mem.lock)

"Store `accepted_count` as the predictor of `(side, bond)` and append `rec` to the trace."
function _dd_remember!(mem::DDMemory, side::Symbol, bond::Int, accepted_count::Int, rec::NamedTuple)
    lock(mem.lock) do
        mem.accepted[(side, bond)] = accepted_count
        push!(mem.trace, rec)
    end
    return mem
end

"""
    _dd_hybrid(frame, opp, side, draw, sketch, sketch_adj, exhaust_opp, exhaust_self, project,
               initial_width, wcap, cap, self_discarded, probe_discarded, alloc, ledger;
               pred, rng, kc, holdout_width) -> (dd_basis, status, record)

Arm D3, on the same `X = mat(F(psi))`, frames and DD-QR kernels as D1:

1. PREFLIGHT. `self_discarded` (`m`) and `probe_discarded` (`n`) are the discarded dimensions of
   this side and of the probe side. With `s = holdout_width`, when
   `ell + s + ell_retry >= min(m, n)` sampling cannot
   undercut the exhaustive DD action (a cost heuristic, not a theorem), so that runs instead: the
   probe side's whole DD basis (`n` columns), or for `2m < n` the mirrored route (`m` adjoint
   columns give the row space, then its rank in forward columns). Exhaustive is exact up to the QR
   floor, so no holdout is drawn.
2. Otherwise ONE block of `ell = pred + _dd_oversample(pred)` columns (`pred` the rank accepted
   here last time, else `initial_width`), QR, and an INDEPENDENT holdout of `s` columns from its
   own RNG stream, accepted at `alloc.sample` (the time-weighted `eta_DD = h R_DD / ||psi||`).
3. On failure the holdout's actions are absorbed and a bounded block of `max(ell ÷ 2, s)` new
   columns appended, with a fresh holdout, until met, `wcap` streamed columns or `cap` directions
   (`:work_limited`), or a pass adds nothing (`:basis_limited`).
"""
function _dd_hybrid(
    frame,
    opp,
    side::Symbol,
    draw,
    sketch,
    sketch_adj,
    exhaust_opp,
    exhaust_self,
    project,
    initial_width::Int,
    wcap::Int,
    cap::Int,
    self_discarded::Int,
    probe_discarded::Int,
    alloc::BasisAllocation,
    ledger::Union{Nothing,AdaptiveLedger};
    pred::Union{Nothing,Int},
    rng::AbstractRNG,
    kc::KernelCounters = KernelCounters(),
    holdout_width::Int = _DD_HOLDOUT,
)
    left = side === :left
    # `fl` is the frame's new (growing) leg, `wleg` the probe-width leg of a sketch block.
    fl, wleg = left ? (3, 1) : (1, 3)
    oside = left ? :right : :left
    tags = Telum.inds(frame)[fl].itags
    aug, dd_basis, nq, streamed = frame, nothing, 0, 0
    # QR the sketched block against the frame augmented so far, append the new directions to
    # `dd_basis` and `aug`, and return how many were added.
    function absorb!(sketched)
        new_basis, _ = _dd_select(aug, sketched, side, kc)
        new_basis === nothing && return 0
        new_basis = TLArray(setitag(new_basis, fl, tags))
        dd_basis =
            dd_basis === nothing ? new_basis : TLArray(oplus([dd_basis, new_basis], (fl,)))
        aug = TLArray(oplus([aug, new_basis], (fl,)))
        return leg_dim(new_basis, fl)
    end
    # Apply the MPO sketch to a probe block, counting its columns as streamed work.
    function stream(Om, field::Symbol)
        streamed += leg_dim(Om, wleg)
        ledger === nothing || setfield!(ledger, field, getfield(ledger, field) + leg_dim(Om, wleg))
        return sketch(Om)
    end
    ell = clamp(pred === nothing ? initial_width : pred + _dd_oversample(pred), 1, max(wcap, 1))
    retry = max(ell ÷ 2, holdout_width)
    # The trace record of this call (the `DDMemory` trace fields).
    rec(branch, st, retries) = (
        side = side,
        branch = branch,
        ell = ell,
        s = holdout_width,
        retries = retries,
        m = self_discarded,
        n = probe_discarded,
        added = nq,
        streamed = streamed,
        status = st,
    )

    if ell + holdout_width + retry >= min(self_discarded, probe_discarded)
        if probe_discarded <= 2self_discarded
            exhaustive_probe = exhaust_opp()
            exhaustive_probe === nothing && return (nothing, :empty, rec(:exhaustive, :empty, 0))
            # orthonormal DD basis of the probe side, `probe_discarded` columns
            Bq, _ = _dd_select(opp, exhaustive_probe, oside, kc)
            Bq === nothing && return (nothing, :empty, rec(:exhaustive, :empty, 0))
            nq += absorb!(stream(Bq, :n_probes))
            return (dd_basis, :met, rec(:exhaustive, :met, 0))
        end
        Sx = exhaust_self()
        Sx === nothing && return (nothing, :empty, rec(:mirrored, :empty, 0))
        Sq, _ = _dd_select(frame, Sx, side, kc)          # this side's DD basis, `self_discarded`
        Sq === nothing && return (nothing, :empty, rec(:mirrored, :empty, 0))
        streamed += leg_dim(Sq, fl)
        ledger === nothing || (ledger.n_probes += leg_dim(Sq, fl))
        Wq, _ = _dd_select(opp, sketch_adj(Sq), oside, kc)   # the row space of A_DD
        Wq === nothing && return (nothing, :met, rec(:mirrored, :met, 0))
        nq += absorb!(stream(Wq, :n_probes))
        return (dd_basis, :met, rec(:mirrored, :met, 0))
    end

    hrng = MersenneTwister(rand(rng, UInt64))                 # the holdout's own stream
    blk = draw(ell, nothing)
    blk === nothing && return (nothing, :empty, rec(:sampled, :empty, 0))
    nq += absorb!(stream(blk, :n_probes))
    retries = 0
    while true
        Oh = _dd_probe(opp, oside, holdout_width; rng = hrng, kc = kc)
        Oh === nothing && return (dd_basis, :met, rec(:sampled, :met, retries))
        Zh = stream(Oh, :n_holdout)
        est, per, hcols = _sample_residual(dd_basis, project, _ -> Zh, _ -> Oh, 1, wleg, side)
        ind = indicator(alloc, est)
        if ind <= alloc.sample || streamed >= wcap || nq >= cap
            ledger === nothing || _record_sample!(ledger, side, ind, per, hcols, blk, wleg, alloc)
            ind <= alloc.sample && return (dd_basis, :met, rec(:sampled, :met, retries))
            if ledger !== nothing
                _unmet!(ledger, :sample)
                ledger.workcap_reached = true
            end
            return (dd_basis, :work_limited, rec(:sampled, :work_limited, retries))
        end
        added = absorb!(Zh)                                   # the failed holdout is kept
        width = min(retry, max(wcap - streamed, 0))
        if width > 0
            retry_block = draw(width, nothing)
            retry_block === nothing ||
                (blk = retry_block; added += absorb!(stream(retry_block, :n_probes)))
        end
        nq += added
        retries += 1
        if added == 0
            ledger === nothing || (_record_sample!(ledger, side, ind, per, hcols, blk, wleg, alloc);
            _unmet!(ledger, :sample))
            return (dd_basis, :basis_limited, rec(:sampled, :basis_limited, retries))
        end
    end
end
