# DD-QR CBE bond expansion: `dd_cbe_expand` grows one side of a bond on top of the controllers in
# `dd_cbe_controller.jl`; `_resolve_parallel` reads the `parallel` option of the step.

# ── one side of the bond expansion ───────────────────────────────────────────────────────

"""
    dd_cbe_expand(frame_pair, mpo, bond, lenv, renv, side; alloc, ledger, workcap, growth,
                  exact, rng, kc) -> CBEExpansion

Grows ONE frame of `frame_pair` at bond `(bond, bond+1)`: `U0` for `side = :left` (K-sweep), `V0`
for `:right` (L-sweep). The other frame is returned unchanged. Rank comes from the QR; the probe
width from [`_dd_controller`](@ref), which widens the probe until an independent holdout of
`holdout_width` columns finds the rest within the side's share of `tol_basis`. Without an
allocation (`alloc === nothing`), or for the deterministic reference `exact = true` (the complete
DD-projected probe), the numerical rank alone stops the search. Retained directions are bounded by
the local room and the work ceiling; past the ceiling the first sectors in storage order are kept
(`_trim_total`).
"""
function dd_cbe_expand(
    frame_pair::BondFrame,
    mpo::MPO,
    bond::Int,
    lenv::MPOLink,
    renv::MPOLink,
    side::Symbol;
    alloc::Union{Nothing,BasisAllocation} = nothing,
    ledger::Union{Nothing,AdaptiveLedger} = nothing,
    workcap::Int = 0,
    growth::Float64 = ADAPTIVE_GROWTH,
    exact::Bool = false,
    rng::AbstractRNG = MersenneTwister(0x5EED),
    kc::KernelCounters = KernelCounters(),
    version::Int = 0,
    holdout_width::Int = _DD_HOLDOUT,
    batch_blocks::Int = 1,
    cap_selection::Symbol = :storage,
    audit::Bool = false,
    mode::Symbol = :adaptive,
    fixed_width::Int = 0,
    fixed_rank::Int = 0,
    acceptance_factor::Float64 = 1.0,
    rank_initial_width::Int = 4,
    rank_floor::Int = 0,
)
    side in (:left, :right) || throw(ArgumentError("side must be :left or :right"))
    mode in (:adaptive, :fixed, :rank_adaptive) ||
        throw(ArgumentError("unknown DD mode: $mode"))
    rank_initial_width >= 1 || throw(ArgumentError("rank_initial_width must be positive"))
    rank_floor >= 0 || throw(ArgumentError("rank_floor must be nonnegative"))
    1 <= batch_blocks <= 8 || throw(ArgumentError("batch_blocks must be in 1:8"))
    cap_selection in (:storage, :svd) ||
        throw(ArgumentError("cap_selection must be :storage or :svd"))
    mode === :fixed && (fixed_width < 1 || fixed_rank < 1) &&
        throw(ArgumentError("fixed mode requires positive fixed_width and fixed_rank"))
    0 < acceptance_factor <= 1 ||
        throw(ArgumentError("acceptance_factor must be in (0,1]"))
    left = side === :left
    left_basis, right_basis = frame_pair.U0, frame_pair.V0
    old_rank = frame_pair.old_rank
    # `frame` grows; `opp` is the opposite frame, whose side the probes are drawn on.
    frame, opp = left ? (left_basis, right_basis) : (right_basis, left_basis)
    oside = left ? :right : :left
    # `fl` is the frame's new (growing) leg, `wleg` the probe-width leg of a sketch block.
    fl, wleg = left ? (3, 1) : (1, 3)
    # The unchanged frames: the result when nothing is added.
    none() = CBEExpansion(left_basis, right_basis, 0, 0, 0.0, 0.0)

    # The fused local dimension (SU(2) counts multiplets).
    fused(tensor, leg_a, leg_b) =
        symmetry_mode() === :SU2 ?
        sum(sector_size for (_, sector_size) in reachable_sectors(tensor, leg_a, leg_b); init = 0) :
        leg_dim(tensor, leg_a) * leg_dim(tensor, leg_b)
    full_self = left ? fused(left_basis, 1, 2) : fused(right_basis, 2, 3)
    full_opp = left ? fused(right_basis, 2, 3) : fused(left_basis, 1, 2)
    cap = full_self - old_rank
    wbind = false
    if alloc !== nothing && workcap > 0
        wk = max(workcap - old_rank, 0)
        wbind = wk < cap
        cap = min(cap, wk)
    end
    cap <= 0 && return none()

    dmax = max(leg_dim(left_basis, 1), old_rank, leg_dim(right_basis, 3))
    initial_width = ceil(Int, 1.2 * max(ceil(Int, growth * dmax) - old_rank, 1))
    wcap = exact ? initial_width : full_opp
    (alloc !== nothing && workcap > 0 && !exact) && (wcap = min(wcap, workcap))

    # A probe block of `width` columns on the opposite side (`boost` raises targeted sectors).
    draw(width, boost) =
        exact ? _dd_probe(opp, oside, width; rng = rng, exact = true, kc = kc) :
        _dd_probe(opp, oside, width; rng = rng, boost = boost, kc = kc)
    op = TwoSiteSketchOp(
        frame_pair,
        mpo,
        bond,
        lenv,
        renv;
        adjoint = !left,
        version = version,
        counters = kc,
    )
    out = Ref{Any}(nothing)
    sketch(Om) = (apply_two_site_sketch!(out, Om, op); out[])
    project =
        left ? (block -> perp_component(left_basis, block)) :
        (block -> perp_component_right(right_basis, block))
    select(Om) = (first(_dd_select(frame, sketch(Om), side, kc)), 0.0)

    # A separate rank-adaptive controller ranks the small U(1)-resolved DD sketch
    # by its singular vectors and uses *held-out MPO actions* to choose the MINIMUM
    # retained rank. The original QR controller only adapts discovery width; it
    # usually keeps every QR direction in the initial block.
    #
    # Candidate and validation sketches never share Gaussian draws. The first
    # validation can be used to choose k, so a second independent confirmation is
    # mandatory (otherwise rank search exploits holdout noise). This is an empirical
    # error monitor, NOT a rigorous confidence bound under repeated optional stops.
    function ranked_dd()
        maxprobe = min(full_opp, max(wcap, cap + 4))
        width = min(maxprobe, max(rank_initial_width, rank_floor + 2))
        width <= 0 && return (nothing, 0)
        best = nothing
        for _attempt in 1:12
            Om = draw(width, nothing)
            Om === nothing && return (best, best === nothing ? 0 : leg_dim(best, fl))
            ledger === nothing || (ledger.n_probes += leg_dim(Om, wleg))
            Y = TLArray(project(sketch(Om)))
            norm(Y) <= _CBE_EPS &&
                return (nothing, 0)
            # The full discarded-space matrix is never formed. Reuse this same
            # MPO sketch for EVERY candidate DD rank during binary search.
            legs = left ? (1, 2) : (2, 3)
            rmax = min(cap, width)
            if rmax < min(rank_floor, cap) && width < maxprobe
                width = min(maxprobe, 2 * width)
                continue
            end
            rank_cache = Dict{Int,Any}()
            function selected(k)
                k == 0 && return nothing
                haskey(rank_cache, k) && return rank_cache[k]
                fac = svd(Y, legs, _TAG_K, _TAG_KR;
                    Nkeep = k,
                    cutoff = _rank_cut(0.0, Y, norm(Y)),
                    get_lists = true)
                result = isempty(fac.kept_list) ? nothing :
                         left ? _reortho_left(frame, TLArray(fac.U)) :
                                _reortho_right(frame, TLArray(permutedims(fac.U, (3, 1, 2))))
                rank_cache[k] = result
                return result
            end
            # Cache one independent holdout for ALL rank candidates. A minimal
            # satisfactory k is found by logarithmically many small block SVDs.
            Ho = draw(holdout_width, nothing)
            Ho === nothing && return (selected(rmax), rmax)
            ledger === nothing || (ledger.n_holdout += leg_dim(Ho, wleg))
            HY = sketch(Ho)
            test_on = function (candidate, X, OM)
                est, _, _ = _sample_residual(candidate, project,
                    _ -> X, _ -> OM, 1, wleg, side)
                return indicator(alloc, est)
            end
            record_test = function (candidate, X, OM)
                ledger === nothing && return nothing
                est, per, hc = _sample_residual(candidate, project,
                    _ -> X, _ -> OM, 1, wleg, side)
                _record_sample!(ledger, side, indicator(alloc, est),
                    per, hc, OM, wleg, alloc)
                return nothing
            end
            threshold = acceptance_factor * alloc.sample
            if test_on(nothing, HY, Ho) <= threshold
                chosen_k = min(rank_floor, rmax)
                pick = selected(chosen_k)
            else
                best = selected(rmax)
                if test_on(best, HY, Ho) > threshold
                    if width < maxprobe
                        width = min(maxprobe, 2 * width)
                        continue
                    end
                    record_test(best, HY, Ho)
                    ledger === nothing || begin
                        _unmet!(ledger, :sample)
                        ledger.workcap_reached = true
                    end
                    return (best, best === nothing ? 0 : leg_dim(best, fl))
                end
                low, high = 0, rmax
                while high - low > 1
                    mid = (low + high) ÷ 2
                    trial = selected(mid)
                    if test_on(trial, HY, Ho) <= threshold
                        high = mid
                    else
                        low = mid
                    end
                end
                chosen_k = max(high, min(rank_floor, rmax))
                pick = selected(chosen_k)
            end
            # Fresh independent holdout: rank selection never sees this action.
            Ho2 = draw(2 * holdout_width, nothing)
            if Ho2 !== nothing
                ledger === nothing || (ledger.n_holdout += leg_dim(Ho2, wleg))
                Y2 = sketch(Ho2)
                if test_on(pick, Y2, Ho2) > threshold
                    if width < maxprobe
                        width = min(maxprobe, 2 * width)
                        continue
                    end
                    best = selected(rmax)
                    record_test(best, Y2, Ho2)
                    ledger === nothing || begin
                        _unmet!(ledger, :sample)
                        ledger.workcap_reached = true
                    end
                    return (best, best === nothing ? 0 : leg_dim(best, fl))
                end
                record_test(pick, Y2, Ho2)
            else
                record_test(pick, HY, Ho)
            end
            return (pick, chosen_k)
        end
        ledger === nothing || _unmet!(ledger, :sample)
        return (best, best === nothing ? 0 : leg_dim(best, fl))
    end

    # With an allocation the incremental controller decides the width (it is always
    # thin); without one, or for the exact reference, the numerical-rank stop of `_range_find`
    # alone.
    directions, direction_count = if mode === :rank_adaptive && alloc !== nothing && !exact
        ranked_dd()
    elseif mode === :fixed && !exact
        probe = draw(min(fixed_width, full_opp), nothing)
        if probe !== nothing && ledger !== nothing
            # The fixed-knob comparison is not validated, but its sketch work is real.
            ledger.n_probes += leg_dim(probe, wleg)
        end
        # A fair fixed-rank RSVD baseline: SVD of the *projected sketch*.
        # Unlike taking the first QR columns (sector/storage order), this keeps
        # the most significant sampled singular directions across charge sectors.
        fixed_basis = if probe === nothing
            nothing
        else
            Y = TLArray(project(sketch(probe)))
            if norm(Y) <= _CBE_EPS
                nothing
            else
                legs = left ? (1, 2) : (2, 3)
                out = svd(Y, legs, _TAG_K, _TAG_KR;
                    Nkeep = min(fixed_rank, cap),
                    cutoff = _rank_cut(0.0, Y, norm(Y)),
                    get_lists = true)
                if isempty(out.kept_list)
                    nothing
                elseif left
                    _reortho_left(frame, TLArray(out.U))
                else
                    _reortho_right(frame, TLArray(permutedims(out.U, (3, 1, 2))))
                end
            end
        end
        (fixed_basis, fixed_basis === nothing ? 0 : leg_dim(fixed_basis, fl))
    elseif alloc === nothing || exact
        Qr, _, nr = _certified_range(
            draw,
            wleg,
            select,
            fl,
            project,
            sketch,
            side,
            initial_width,
            wcap,
            cap,
            nothing,
            ledger,
        )
        (Qr, nr)
    else
        Qc, _ = _dd_controller(
            frame,
            side,
            draw,
            sketch,
            project,
            initial_width,
            wcap,
            cap,
            alloc,
            ledger;
            kc = kc,
            holdout_width = holdout_width,
            batch_blocks = batch_blocks,
            cap_selection = cap_selection,
            audit = audit,
            acceptance_factor = acceptance_factor,
        )
        (Qc, Qc === nothing ? 0 : leg_dim(Qc, fl))
    end
    directions === nothing && return none()
    if direction_count > cap
        directions = _trim_total(directions, fl, cap)
        directions === nothing && return none()
    end
    ledger === nothing || (wbind && direction_count >= cap && (ledger.workcap_reached = true))
    # Insurance only: the QR floor is per sector and absolute, so roundoff in a saturated sector
    # does not become a column the way a globally relative SVD cut lets it.
    directions = _cap_to_local_room(frame, directions, side)
    directions === nothing && return none()
    nq = leg_dim(directions, fl)
    kc.n_sector_new +=
        count(((charge, _),) -> sector_dim(frame, fl, charge) == 0, Telum.spaces(directions)[fl])
    ledger === nothing || (ledger.n_selected += nq)
    grown = TLArray(
        oplus([frame, TLArray(setitag(directions, fl, Telum.inds(frame)[fl].itags))], (fl,)),
    )
    # The grown frame replaces its side; the other frame passes through unchanged.
    if left
        return CBEExpansion(
            _check_frame(grown, :left, "dd_cbe_expand U_ex"),
            right_basis,
            nq,
            0,
            0.0,
            0.0,
        )
    end
    return CBEExpansion(
        left_basis,
        _check_frame(grown, :right, "dd_cbe_expand V_ex"),
        0,
        nq,
        0.0,
        0.0,
    )
end

"""
    _resolve_parallel(parallel) -> Bool

`parallel`: `:serial`/`false`, `:half_sweeps`/`true`. `:full` also runs the
half-sweeps concurrently; `cbe_bug_step!` then takes the separate parallel K/L/S integrator
(`parallel_kls_step!`).
"""
function _resolve_parallel(parallel)
    parallel isa Bool && return parallel
    parallel === :serial && return false
    parallel in (:half_sweeps, :full) && return true
    throw(ArgumentError("parallel must be :serial, :half_sweeps, :full or a Bool, got $parallel"))
end
