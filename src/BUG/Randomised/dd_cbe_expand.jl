# DD-QR CBE bond expansion: `dd_cbe_expand` grows one side of a bond on top of the controllers in
# `dd_cbe_controller.jl`; `_resolve_parallel` reads the `parallel` option of the step.

# ── one side of the bond expansion ───────────────────────────────────────────────────────

const _TAG_W, _TAG_WR = "cbeW,L", "cbeW,R"

"""
    _trim_by_weight(directions, projected_action, side, keep_count) -> directions or nothing

Keep the `keep_count` directions of `directions` (orthonormal, outside the frame) that carry the
most of the DD action `M = P^D X P^D`, given `projected_action = directions' M` (rows
`(g, site_r, link_r)` for `:left`, columns `(link_l, site_l, g)` for `:right`, `g` = the
directions' leg): Telum's SVD ranks them across ALL sectors (`Nkeep`), as the final selection of
`cbe_expand` does. `_trim_total` instead fills sectors in storage order, so at a binding work
ceiling whole charge sectors lose their expansion. Measured (dissipative A1, L = 12, dt 0.1): frame
cut at 1024 against 1679 needed, obs error 2.2e-4 against 1.3e-12 uncut, the spurious components
then forcing the compressed rank to maxdim off the root.
"""
function _trim_by_weight(directions, projected_action, side::Symbol, keep_count::Int)
    keep_count <= 0 && return nothing
    left = side === :left
    leg_dim(directions, left ? 3 : 1) <= keep_count && return directions
    left_vectors =
        svd(
            projected_action,
            left ? (1,) : (3,),
            _TAG_W,
            _TAG_WR;
            cutoff = 0.0,
            Nkeep = keep_count,
        ).U                                                           # (g, k)
    return left ? TLArray(contract(directions, (3,), left_vectors, (1,))) :  # (link, site, k)
           TLArray(contract(left_vectors, (1,), directions, (1,)))         # (k, site, link)
end

"""
    dd_cbe_expand(frame_pair, mpo, bond, lenv, renv, side; alloc, ledger, workcap, growth,
                  exact, rng, kc) -> CBEExpansion

Grows ONE frame of `frame_pair` at bond `(bond, bond+1)`: `U0` for `side = :left` (K-sweep), `V0`
for `:right` (L-sweep). The other frame is returned unchanged. Rank comes from the QR; the probe
width from [`_dd_controller`](@ref), or from the numerical rank alone when `alloc === nothing` or
`exact`. Retained directions are bounded by the local room and the work ceiling; past the ceiling
`cap_order = :weight` keeps the heaviest ([`_trim_by_weight`](@ref)), `:sector` the first sectors
in storage order (`_trim_total`).

`dd_width = l > 0` is the fixed-oversampling ablation: one
probe of width `l` through the same DD projection, streamed kernel and QR, no holdout.
`dd_policy = :hybrid` is arm D3 ([`_dd_hybrid`](@ref)), its predictor read from and
written to `dd_memory` (a [`DDMemory`](@ref); `nothing` = no predictor); `:adaptive` is D1; `:off`
adds nothing (the no-CBE control).
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
    cap_order::Symbol = :sector,
    dd_width::Int = 0,
    dd_policy::Symbol = :adaptive,
    dd_memory = nothing,
)
    side in (:left, :right) || throw(ArgumentError("side must be :left or :right"))
    cap_order in (:sector, :weight) ||
        throw(ArgumentError("cap_order must be :sector or :weight, got $cap_order"))
    dd_policy in (:adaptive, :hybrid, :off) ||
        throw(ArgumentError("dd_policy must be :adaptive, :hybrid or :off, got $dd_policy"))
    # `:off` -- the no-CBE control (the frame then grows by local Krylov alone)
    dd_policy === :off && return CBEExpansion(frame_pair.U0, frame_pair.V0, 0, 0, 0.0, 0.0)
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

    # The fused local dimension, as `cbe_expand` reads it (SU(2) counts multiplets).
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
    directions, direction_count = if dd_width > 0 && !exact
        blk = draw(min(dd_width, full_opp), nothing)
        Qf = blk === nothing ? nothing : first(_dd_select(frame, sketch(blk), side, kc))
        blk === nothing || ledger === nothing || (ledger.n_probes += leg_dim(blk, wleg))
        (Qf, Qf === nothing ? 0 : leg_dim(Qf, fl))
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
    elseif dd_policy === :hybrid
        opA = TwoSiteSketchOp(
            frame_pair,
            mpo,
            bond,
            lenv,
            renv;
            adjoint = left,
            version = version,
            counters = kc,
        )
        outA = Ref{Any}(nothing)
        # anonymous: a named local function inside a branch is hoisted to the whole method
        sketch_adj = Om -> (apply_two_site_sketch!(outA, Om, opA); outA[])
        exhaust_opp = () -> _dd_probe(opp, oside, 0; rng = rng, exact = true, kc = kc)
        exhaust_self = () -> _dd_probe(frame, side, 0; rng = rng, exact = true, kc = kc)
        pred = dd_memory === nothing ? nothing : _dd_predict(dd_memory, side, bond)
        Qh, _, rec = _dd_hybrid(
            frame,
            opp,
            side,
            draw,
            sketch,
            sketch_adj,
            exhaust_opp,
            exhaust_self,
            project,
            initial_width,
            wcap,
            cap,
            full_self - old_rank,
            full_opp - old_rank,
            alloc,
            ledger;
            pred = pred,
            rng = rng,
            kc = kc,
            holdout_width = holdout_width,
        )
        nh = Qh === nothing ? 0 : leg_dim(Qh, fl)
        dd_memory === nothing ||
            _dd_remember!(dd_memory, side, bond, nh, merge((bond = bond,), rec))
        (Qh, nh)
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
        )
        (Qc, Qc === nothing ? 0 : leg_dim(Qc, fl))
    end
    directions === nothing && return none()
    if direction_count > cap
        directions = if cap_order === :weight
            # directions' M exactly: the opposite sketch applied to the directions themselves
            opT = TwoSiteSketchOp(
                frame_pair,
                mpo,
                bond,
                lenv,
                renv;
                adjoint = left,
                version = version,
                counters = kc,
            )
            projected_action = (apply_two_site_sketch!(out, directions, opT); out[])
            _trim_by_weight(
                directions,
                left ? perp_component_right(right_basis, projected_action) :
                perp_component(left_basis, projected_action),
                side,
                cap,
            )
        else
            _trim_total(directions, fl, cap)
        end
        directions === nothing && return none()
    end
    ledger === nothing || (wbind && direction_count >= cap && (ledger.workcap_reached = true))
    # Insurance only: the QR floor is per sector and absolute, so roundoff in a saturated sector
    # does not become a column the way a globally relative SVD cut lets it (`_guarded_block`).
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
