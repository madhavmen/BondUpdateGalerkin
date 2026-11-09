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
    audit::Bool = false,
    mode::Symbol = :adaptive,
    fixed_width::Int = 0,
    fixed_rank::Int = 0,
    acceptance_factor::Float64 = 1.0,
)
    side in (:left, :right) || throw(ArgumentError("side must be :left or :right"))
    mode in (:adaptive, :fixed) || throw(ArgumentError("unknown DD mode: $mode"))
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

    # With an allocation the incremental controller decides the width (it is always
    # thin); without one, or for the exact reference, the numerical-rank stop of `_range_find`
    # alone.
    directions, direction_count = if mode === :fixed && !exact
        probe = draw(min(fixed_width, full_opp), nothing)
        if probe !== nothing && ledger !== nothing
            # The fixed-knob comparison is not validated, but its sketch work is real.
            ledger.n_probes += leg_dim(probe, wleg)
        end
        selected = probe === nothing ? nothing : first(_dd_select(frame, sketch(probe), side, kc))
        # Fixed-rank is a cap, deliberately without a residual certification.
        fixed_basis = selected === nothing ? nothing : _trim_total(selected, fl, fixed_rank)
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
