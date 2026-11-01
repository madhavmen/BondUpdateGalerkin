# Moved from BUG-Julia@b155550:src/RSVDCBEBondUpdate/expansion/dd_cbe.jl, src/RSVDCBEBondUpdate/expansion/dd_cbe_controller.jl.

# DD-QR CBE controllers: the incremental D1 sketch controller and the D3 hybrid predictor (`DDMemory`).

# ── from src/RSVDCBEBondUpdate/expansion/dd_cbe_controller.jl ──

# dd_cbe_controller.jl: the incremental DD controller, `dd_cbe_expand` and the step record (from `dd_cbe.jl`, included right after it).

# ── T3.1-T3.4: the incremental DD sketch controller ───────────────────────────────────────

"Plan T3.3: the default independent holdout width `s` (an implementation constant; tests 2, 4, 8)."
const _DD_HOLDOUT = 4

"""
    _dd_controller(frame, side, draw, sketch, project, w0, wcap, ncap, alloc, ledger;
                   kc, s, audit, holdout, trace) -> (Q, status)

Grows the DD basis `Q` of one side block by block until an independent holdout says the rest of
`G = P^D X P^D` fits this side's share `alloc.sample`. Each probe block is streamed through the MPO
once and its action cached; a failed check draws a fresh holdout before promoting the old one.

Statuses: `:met`; `:work_limited` (`wcap` or `ncap` reached); `:basis_limited` (a pass added no
direction); `:empty` (no discarded space). Every continuing pass adds a direction, so the loop ends.
"""
function _dd_controller(
    frame,
    side::Symbol,
    draw,
    sketch,
    project,
    w0::Int,
    wcap::Int,
    ncap::Int,
    alloc::BasisAllocation,
    ledger::Union{Nothing,AdaptiveLedger};
    kc::KernelCounters = KernelCounters(),
    s::Int = _DD_HOLDOUT,
    audit::Bool = false,
    holdout = w -> draw(w, nothing),
    trace::Union{Nothing,Vector} = nothing,
)
    left = side === :left
    fl, wleg = left ? (3, 1) : (1, 3)
    tags = frame.inds[fl].itags
    aug, Q, n, streamed = frame, nothing, 0, 0
    boost = Dict{Any,Int}()
    function absorb!(Z)
        Qn, _ = _dd_select(aug, Z, side, kc)
        Qn === nothing && return 0
        Qn = to_concrete(setitag(Qn, fl, tags))
        Q = Q === nothing ? Qn : to_concrete(oplus([Q, Qn], (fl,)))
        aug = to_concrete(oplus([aug, Qn], (fl,)))
        return leg_dim(Qn, fl)
    end
    function stream(Om, field::Symbol)
        w = leg_dim(Om, wleg)
        streamed += w
        ledger === nothing || setfield!(ledger, field, getfield(ledger, field) + w)
        return sketch(Om)
    end
    function fresh(w)
        Om = holdout(w)
        return Om === nothing ? nothing : (Om, stream(Om, :n_holdout))
    end
    resid(h) = _sample_residual(Q, project, _ -> h[2], _ -> h[1], 1, wleg, side)
    newsector() = Q !== nothing && any(((q, _),) -> sector_dim(frame, fl, q) == 0, Q.spaces[fl])

    w = clamp(w0, 1, max(wcap, 1))
    last = draw(w, nothing)
    last === nothing && return (nothing, :empty)
    n += absorb!(stream(last, :n_probes))
    h = fresh(s)
    h1 = h
    escalated = false
    while true
        h === nothing && return (Q, :met)            # nothing left to validate against
        est, per, hcols = resid(h)
        ind = indicator(alloc, est)
        trace === nothing || push!(trace, (n, ind, indicator(alloc, resid(h1)[1])))
        ok = ind <= alloc.sample
        if ok && !escalated && (audit || ind > alloc.sample / 2 || newsector())
            escalated = true
            he = fresh(2s)
            he === nothing || (h = he; continue)
        end
        capped = streamed >= wcap || n >= ncap
        if ok || capped
            ledger === nothing || _record_sample!(ledger, side, ind, per, hcols, last, wleg, alloc)
            ok && return (Q, :met)
            if ledger !== nothing
                _unmet!(ledger, :sample)
                ledger.workcap_reached = true
            end
            return (Q, :work_limited)
        end
        nt = _target_sectors!(boost, per, h[1], wleg, alloc)
        ledger === nothing || (ledger.n_targeted += nt)
        hnew = fresh(s)                          # new validation BEFORE the old one is promoted
        added = absorb!(h[2])
        nt == 0 && (w = min(2 * w, wcap))
        blk = draw(w, isempty(boost) ? nothing : boost)
        if blk !== nothing
            last = blk
            added += absorb!(stream(blk, :n_probes))
        end
        n += added
        if added == 0
            ledger === nothing ||
                (_record_sample!(ledger, side, ind, per, hcols, last, wleg, alloc);
                _unmet!(ledger, :sample))
            return (Q, :basis_limited)
        end
        h = hnew
    end
end

# ── QTT plan §1.2 / §1.2A, arm D3: the hybrid predictor controller ────────────────────────

"Bounded oversampling of D3's first block over its predictor: `clamp(cld(pred, 4), 4, 8)` columns."
_dd_oversample(pred::Int) = clamp(cld(pred, 4), 4, 8)

"""
    DDMemory()

Per-bond memory of the D3 hybrid controller (`dd_policy = :hybrid`): the directions accepted last
time at `(side, bond)` -- the PREDICTOR of the next first block, never a correctness condition --
and one trace entry per call (`bond`, `side`, `branch`, `ell`, `s`, `retries`, `m`, `n`, `added`,
`streamed`, `status`) for the first-impressions report (plan §1.2A: reported for every sampled
bond). Locked, since the two half-sweeps may run concurrently.
"""
mutable struct DDMemory
    accepted::Dict{Tuple{Symbol,Int},Int}
    trace::Vector{NamedTuple}
    lock::ReentrantLock
end
DDMemory() = DDMemory(Dict{Tuple{Symbol,Int},Int}(), NamedTuple[], ReentrantLock())

_dd_predict(mem::DDMemory, side::Symbol, i::Int) =
    lock(() -> get(mem.accepted, (side, i), nothing), mem.lock)

function _dd_remember!(mem::DDMemory, side::Symbol, i::Int, n::Int, rec::NamedTuple)
    lock(mem.lock) do
        mem.accepted[(side, i)] = n
        push!(mem.trace, rec)
    end
    return mem
end

"""
    _dd_hybrid(frame, opp, side, draw, sketch, sketch_adj, exhaust_opp, exhaust_self, project,
               w0, wcap, cap, m, n, alloc, ledger; pred, rng, kc, s) -> (Q, status, record)

Arm D3 of plan §1.2A, on the same `X = mat(F(psi))`, frames and DD-QR kernels as D1:

1. PREFLIGHT. `m`, `n` are the discarded dimensions of this side and of the probe side. When
   `ell + s + ell_retry >= min(m, n)` sampling cannot undercut the exhaustive DD action (a cost
   heuristic, not a theorem), so that runs instead: the probe side's whole DD basis (`n` columns),
   or for `2m < n` the mirrored route (`m` adjoint columns give the row space, then its rank in
   forward columns). Exhaustive is exact up to the QR floor, so no holdout is drawn.
2. Otherwise ONE block of `ell = pred + _dd_oversample(pred)` columns (`pred` the rank accepted
   here last time, else `w0`), QR, and an INDEPENDENT holdout of `s` columns from its own RNG
   stream, accepted at `alloc.sample` (the time-weighted `eta_DD = h R_DD / ||psi||`).
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
    w0::Int,
    wcap::Int,
    cap::Int,
    m::Int,
    n::Int,
    alloc::BasisAllocation,
    ledger::Union{Nothing,AdaptiveLedger};
    pred::Union{Nothing,Int},
    rng::AbstractRNG,
    kc::KernelCounters = KernelCounters(),
    s::Int = _DD_HOLDOUT,
)
    left = side === :left
    fl, wleg = left ? (3, 1) : (1, 3)
    oside = left ? :right : :left
    tags = frame.inds[fl].itags
    aug, Q, nq, streamed = frame, nothing, 0, 0
    function absorb!(Z)
        Qn, _ = _dd_select(aug, Z, side, kc)
        Qn === nothing && return 0
        Qn = to_concrete(setitag(Qn, fl, tags))
        Q = Q === nothing ? Qn : to_concrete(oplus([Q, Qn], (fl,)))
        aug = to_concrete(oplus([aug, Qn], (fl,)))
        return leg_dim(Qn, fl)
    end
    function stream(Om, field::Symbol)
        streamed += leg_dim(Om, wleg)
        ledger === nothing || setfield!(ledger, field, getfield(ledger, field) + leg_dim(Om, wleg))
        return sketch(Om)
    end
    ell = clamp(pred === nothing ? w0 : pred + _dd_oversample(pred), 1, max(wcap, 1))
    retry = max(ell ÷ 2, s)
    rec(branch, st, retries) = (
        side = side,
        branch = branch,
        ell = ell,
        s = s,
        retries = retries,
        m = m,
        n = n,
        added = nq,
        streamed = streamed,
        status = st,
    )

    if ell + s + retry >= min(m, n)
        if n <= 2m
            B = exhaust_opp()
            B === nothing && return (nothing, :empty, rec(:exhaustive, :empty, 0))
            Bq, _ = _dd_select(opp, B, oside, kc)            # orthonormal DD basis, n columns
            Bq === nothing && return (nothing, :empty, rec(:exhaustive, :empty, 0))
            nq += absorb!(stream(Bq, :n_probes))
            return (Q, :met, rec(:exhaustive, :met, 0))
        end
        Sx = exhaust_self()
        Sx === nothing && return (nothing, :empty, rec(:mirrored, :empty, 0))
        Sq, _ = _dd_select(frame, Sx, side, kc)               # this side's DD basis, m columns
        Sq === nothing && return (nothing, :empty, rec(:mirrored, :empty, 0))
        streamed += leg_dim(Sq, fl)
        ledger === nothing || (ledger.n_probes += leg_dim(Sq, fl))
        Wq, _ = _dd_select(opp, sketch_adj(Sq), oside, kc)   # the row space of A_DD
        Wq === nothing && return (nothing, :met, rec(:mirrored, :met, 0))
        nq += absorb!(stream(Wq, :n_probes))
        return (Q, :met, rec(:mirrored, :met, 0))
    end

    hrng = MersenneTwister(rand(rng, UInt64))                 # the holdout's own stream
    blk = draw(ell, nothing)
    blk === nothing && return (nothing, :empty, rec(:sampled, :empty, 0))
    nq += absorb!(stream(blk, :n_probes))
    retries = 0
    while true
        Oh = _dd_probe(opp, oside, s; rng = hrng, kc = kc)
        Oh === nothing && return (Q, :met, rec(:sampled, :met, retries))
        Zh = stream(Oh, :n_holdout)
        est, per, hcols = _sample_residual(Q, project, _ -> Zh, _ -> Oh, 1, wleg, side)
        ind = indicator(alloc, est)
        if ind <= alloc.sample || streamed >= wcap || nq >= cap
            ledger === nothing || _record_sample!(ledger, side, ind, per, hcols, blk, wleg, alloc)
            ind <= alloc.sample && return (Q, :met, rec(:sampled, :met, retries))
            if ledger !== nothing
                _unmet!(ledger, :sample)
                ledger.workcap_reached = true
            end
            return (Q, :work_limited, rec(:sampled, :work_limited, retries))
        end
        added = absorb!(Zh)                                   # the failed holdout is kept
        w = min(retry, max(wcap - streamed, 0))
        if w > 0
            b2 = draw(w, nothing)
            b2 === nothing || (blk = b2; added += absorb!(stream(b2, :n_probes)))
        end
        nq += added
        retries += 1
        if added == 0
            ledger === nothing || (_record_sample!(ledger, side, ind, per, hcols, blk, wleg, alloc);
            _unmet!(ledger, :sample))
            return (Q, :basis_limited, rec(:sampled, :basis_limited, retries))
        end
    end
end
