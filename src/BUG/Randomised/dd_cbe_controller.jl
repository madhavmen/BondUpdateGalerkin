# The DD-QR CBE controller, which decides how many probe columns one side of a bond needs: the
# incremental sketch controller `_dd_controller`, stopped by an independent holdout. The probe and
# QR kernels it calls (`_dd_probe`, `_dd_select`) are in `dd_cbe.jl`; `dd_cbe_expand.jl` drives it
# for one side.

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
    batch_blocks::Int = 1,
    audit::Bool = false,
    holdout = width -> draw(width, nothing),
    trace::Union{Nothing,Vector} = nothing,
    acceptance_factor::Float64 = 1.0,
)
    0 < acceptance_factor <= 1 ||
        throw(ArgumentError("acceptance_factor must lie in (0,1]"))
    1 <= batch_blocks <= 8 ||
        throw(ArgumentError("batch_blocks must lie in 1:8"))
    left = side === :left
    # `fl` is the frame's new (growing) leg, `wleg` the probe-width leg of a sketch block.
    fl, wleg = left ? (3, 1) : (1, 3)
    tags = Telum.inds(frame)[fl].itags
    aug, dd_basis, direction_count, streamed, discovered = frame, nothing, 0, 0, 0
    boost = Dict{Any,Int}()
    # QR the sketched block against the frame augmented so far, append the new directions to
    # `dd_basis` and `aug`, and return how many were added.
    function absorb!(sketched)
        new_basis, _ = _dd_select(aug, sketched, side, kc)
        new_basis === nothing && return 0
        # The held-out error MUST be tested on the same basis that can actually
        # enter the CBE frame. Previously a probe could exceed ncap, pass Test B,
        # then be trimmed afterwards, thereby falsely certifying an untested
        # smaller basis under a tight 1.2*chi_b cap.
        current = dd_basis === nothing ? 0 : leg_dim(dd_basis, fl)
        room = ncap - current
        room <= 0 && return 0
        if leg_dim(new_basis, fl) > room
            new_basis = _trim_total(new_basis, fl, room)
            new_basis === nothing && return 0
        end
        new_basis = TLArray(setitag(new_basis, fl, tags))
        dd_basis = dd_basis === nothing ? new_basis : TLArray(oplus([dd_basis, new_basis], (fl,)))
        aug = TLArray(oplus([aug, new_basis], (fl,)))
        return leg_dim(new_basis, fl)
    end
    # Apply the MPO sketch to a probe block, counting its columns as streamed work (and in the
    # ledger field `field`). The assignment rebinds the enclosing `width`, a captured variable, so
    # after each call `width` is the width of the last block streamed (probe or holdout).
    function stream(Om, field::Symbol)
        width = leg_dim(Om, wleg)
        streamed += width
        field === :n_probes && (discovered += width)
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

    width = clamp(initial_width, 1, max(min(wcap, ncap), 1))
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
        ok = ind <= acceptance_factor * alloc.sample
        # A pass close to the line (or into a new sector) is confirmed on a twice wider holdout.
        if ok && !escalated && (audit || ind > alloc.sample / 2 || newsector())
            escalated = true
            wide_check = fresh(2 * holdout_width)
            wide_check === nothing || (check = wide_check; continue)
        end
        # Validation probes are not discovery columns. Keep their accounting in the
        # ledger, but do not let repeated statistical checks exhaust the discovery cap.
        capped = discovered >= wcap || direction_count >= ncap
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
        if batch_blocks == 1
            # Keep the legacy sequential stream ordering as an A/B control.
            next_check = fresh(holdout_width)
            added = absorb!(check[2])
            nt == 0 && (width = min(2 * width, wcap))
            # Once ncap is reached there is no reason to stream another block.
            remaining = ncap - (dd_basis === nothing ? 0 : leg_dim(dd_basis, fl))
            if remaining > 0
                blk = draw(min(width, max(remaining + 2, 1)),
                    isempty(boost) ? nothing : boost)
                if blk !== nothing
                    last = blk
                    added += absorb!(stream(blk, :n_probes))
                end
            end
            check = next_check
        else
            # Batched block discovery: reuse the failed holdout as a discovery
            # sketch, then grow charge-targeted QR directions for <=batch_blocks
            # independent blocks before paying for the NEXT holdout.
            # The validation probes are NEVER used for the candidate basis
            # they validate: a new holdout is drawn only after the batch ends.
            added = absorb!(check[2])
            for _ in 1:batch_blocks
                remaining = ncap - (dd_basis === nothing ? 0 : leg_dim(dd_basis, fl))
                remaining <= 0 && break
                nt == 0 && (width = min(2 * width, wcap))
                blk = draw(min(width, max(remaining + 2, 1)),
                    isempty(boost) ? nothing : boost)
                blk === nothing && break
                last = blk
                added += absorb!(stream(blk, :n_probes))
            end
            check = fresh(holdout_width)
        end
        direction_count += added
        if added == 0
            ledger === nothing ||
                (_record_sample!(ledger, side, ind, per, hcols, last, wleg, alloc);
                _unmet!(ledger, :sample))
            return (dd_basis, :basis_limited)
        end
    end
end
