# Moved from BUG-Julia@b155550:src/RSVDCBEBondUpdate/expansion/dd_cbe.jl, src/RSVDCBEBondUpdate/expansion/dd_cbe_controller.jl.

# DD-QR CBE bond expansion: `dd_cbe_expand` (one side of a bond, on top of the controllers) and `_resolve_parallel`.

# ── T2.5: one side of the bond expansion ──────────────────────────────────────────────────

const _TAG_W, _TAG_WR = "cbeW,L", "cbeW,R"

"""
    _trim_by_weight(Q, B, side, n) -> Q or nothing

Keep the `n` directions of `Q` (orthonormal, outside the frame) that carry the most of the DD
action `M = P^D X P^D`, given `B = Q' M` (rows `(g, site_r, link_r)` for `:left`, columns
`(link_l, site_l, g)` for `:right`, `g` = `Q`'s leg): Telum's SVD ranks them across ALL sectors
(`Nkeep`), as the final selection of `cbe_expand` does. `_trim_total` instead fills sectors in
storage order, so at a binding work ceiling whole charge sectors lose their expansion (the latent
trap of PLAN-tolerance-control.md §1). MEASURED (dissipative A1 L = 12, dt 0.1, rank probe rp7f):
frame cut at 1024 against 1679 needed, obs error 2.2e-4 against 1.3e-12 uncut, the spurious
components then forcing the compressed rank to maxdim off the root.
"""
function _trim_by_weight(Q, B, side::Symbol, n::Int)
    n <= 0 && return nothing
    left = side === :left
    leg_dim(Q, left ? 3 : 1) <= n && return Q
    U = svd(B, left ? (1,) : (3,), _TAG_W, _TAG_WR; cutoff = 0.0, Nkeep = n).U    # (g, k)
    return left ? to_concrete(contract(Q, (3,), U, (1,))) :     # (link, site, k)
           to_concrete(contract(U, (1,), Q, (1,)))       # (k, site, link)
end

"""
    dd_cbe_expand(f, mpo, i, lenv, renv, side; alloc, ledger, workcap, growth,
                  exact, rng, kc) -> CBEExpansion

Grows ONE frame at bond `(i, i+1)`: `U0` for `side = :left` (K-sweep), `V0` for `:right` (L-sweep).
The other frame is returned unchanged. Rank comes from the QR; the probe width from
[`_dd_controller`](@ref), or from the numerical rank alone when `alloc === nothing` or `exact`.
Retained directions are bounded by the local room and the work ceiling; past the ceiling
`cap_order = :weight` keeps the heaviest ([`_trim_by_weight`](@ref)), `:sector` the first sectors
in storage order (`_trim_total`).

`dd_width = l > 0` is the fixed-oversampling ablation (xx-efficiency plan Figs. 1A and 4): one
probe of width `l` through the same DD projection, streamed kernel and QR, no holdout.
`dd_policy = :hybrid` is the QTT plan's arm D3 ([`_dd_hybrid`](@ref)), its predictor read from and
written to `dd_memory` (a [`DDMemory`](@ref); `nothing` = no predictor); `:adaptive` is D1; `:off`
adds nothing (the no-CBE control).
"""
function dd_cbe_expand(
    f::BondFrame,
    mpo::MPO,
    i::Int,
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
    # `:off` -- the no-CBE control of plan §1.2A (the frame then grows by local Krylov alone)
    dd_policy === :off && return CBEExpansion(f.U0, f.V0, 0, 0, 0.0, 0.0)
    left = side === :left
    U0, V0 = f.U0, f.V0
    r = f.old_rank
    frame, opp = left ? (U0, V0) : (V0, U0)
    oside = left ? :right : :left
    fl, wleg = left ? (3, 1) : (1, 3)
    none() = CBEExpansion(U0, V0, 0, 0, 0.0, 0.0)

    # The fused local dimension, as `cbe_expand` reads it (SU(2) counts multiplets).
    fused(t, a, b) =
        symmetry_mode() === :SU2 ? sum(d for (_, d) in reachable_sectors(t, a, b); init = 0) :
        leg_dim(t, a) * leg_dim(t, b)
    full_self = left ? fused(U0, 1, 2) : fused(V0, 2, 3)
    full_opp = left ? fused(V0, 2, 3) : fused(U0, 1, 2)
    cap = full_self - r
    wbind = false
    if alloc !== nothing && workcap > 0
        wk = max(workcap - r, 0)
        wbind = wk < cap
        cap = min(cap, wk)
    end
    cap <= 0 && return none()

    dmax = max(leg_dim(U0, 1), r, leg_dim(V0, 3))
    w0 = ceil(Int, 1.2 * max(ceil(Int, growth * dmax) - r, 1))
    wcap = exact ? w0 : full_opp
    (alloc !== nothing && workcap > 0 && !exact) && (wcap = min(wcap, workcap))

    draw(w, b) =
        exact ? _dd_probe(opp, oside, w; rng = rng, exact = true, kc = kc) :
        _dd_probe(opp, oside, w; rng = rng, boost = b, kc = kc)
    op = TwoSiteSketchOp(f, mpo, i, lenv, renv; adjoint = !left, version = version, counters = kc)
    out = Ref{Any}(nothing)
    sketch(Om) = (apply_two_site_sketch!(out, Om, op); out[])
    project = left ? (X -> perp_component(U0, X)) : (X -> perp_component_right(V0, X))
    select(Om) = (first(_dd_select(frame, sketch(Om), side, kc)), 0.0)

    # Plan T3.1: with an allocation the incremental controller decides the width (it is always
    # thin); without one, or for the exact reference, the numerical-rank stop of `_range_find`
    # alone.
    Q, n = if dd_width > 0 && !exact
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
            w0,
            wcap,
            cap,
            nothing,
            ledger,
        )
        (Qr, nr)
    elseif dd_policy === :hybrid
        opA = TwoSiteSketchOp(f, mpo, i, lenv, renv; adjoint = left, version = version, counters = kc)
        outA = Ref{Any}(nothing)
        # anonymous: a named local function inside a branch is hoisted to the whole method
        sketch_adj = Om -> (apply_two_site_sketch!(outA, Om, opA); outA[])
        exhaust_opp = () -> _dd_probe(opp, oside, 0; rng = rng, exact = true, kc = kc)
        exhaust_self = () -> _dd_probe(frame, side, 0; rng = rng, exact = true, kc = kc)
        pred = dd_memory === nothing ? nothing : _dd_predict(dd_memory, side, i)
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
            w0,
            wcap,
            cap,
            full_self - r,
            full_opp - r,
            alloc,
            ledger;
            pred = pred,
            rng = rng,
            kc = kc,
            s = holdout_width,
        )
        nh = Qh === nothing ? 0 : leg_dim(Qh, fl)
        dd_memory === nothing || _dd_remember!(dd_memory, side, i, nh, merge((bond = i,), rec))
        (Qh, nh)
    else
        Qc, _ = _dd_controller(
            frame,
            side,
            draw,
            sketch,
            project,
            w0,
            wcap,
            cap,
            alloc,
            ledger;
            kc = kc,
            s = holdout_width,
            audit = audit,
        )
        (Qc, Qc === nothing ? 0 : leg_dim(Qc, fl))
    end
    Q === nothing && return none()
    if n > cap
        Q = if cap_order === :weight
            # B = Q' M exactly: the opposite sketch applied to the `n` directions themselves
            opT = TwoSiteSketchOp(
                f,
                mpo,
                i,
                lenv,
                renv;
                adjoint = left,
                version = version,
                counters = kc,
            )
            B = (apply_two_site_sketch!(out, Q, opT); out[])
            _trim_by_weight(
                Q,
                left ? perp_component_right(V0, B) : perp_component(U0, B),
                side,
                cap,
            )
        else
            _trim_total(Q, fl, cap)
        end
        Q === nothing && return none()
    end
    ledger === nothing || (wbind && n >= cap && (ledger.workcap_reached = true))
    # Insurance only: the QR floor is per sector and absolute, so roundoff in a saturated sector
    # does not become a column the way a globally relative SVD cut lets it (`_guarded_block`).
    Q = _cap_to_local_room(frame, Q, side)
    Q === nothing && return none()
    nq = leg_dim(Q, fl)
    kc.n_sector_new += count(((q, _),) -> sector_dim(frame, fl, q) == 0, Q.spaces[fl])
    ledger === nothing || (ledger.n_selected += nq)
    grown = to_concrete(oplus([frame, to_concrete(setitag(Q, fl, frame.inds[fl].itags))], (fl,)))
    return left ?
           CBEExpansion(_check_frame(grown, :left, "dd_cbe_expand U_ex"), V0, nq, 0, 0.0, 0.0) :
           CBEExpansion(U0, _check_frame(grown, :right, "dd_cbe_expand V_ex"), 0, nq, 0.0, 0.0)
end

"""
    _resolve_parallel(p) -> Bool

`parallel` of the plan's API: `:serial`/`false`, `:half_sweeps`/`true`. `:full` also runs the
half-sweeps concurrently; `cbe_bug_step!` then takes the separate parallel K/L/S integrator of
plan Task 6 (`parallel_kls_step!`).
"""
function _resolve_parallel(p)
    p isa Bool && return p
    p === :serial && return false
    p in (:half_sweeps, :full) && return true
    throw(ArgumentError("parallel must be :serial, :half_sweeps, :full or a Bool, got $p"))
end
