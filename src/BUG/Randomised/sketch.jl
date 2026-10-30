# Moved from BUG-Julia@b155550:src/RSVDCBEBondUpdate/expansion/cbe_sketch.jl.

# cbe_sketch.jl: the sector-graded Gaussian sketch, preselection and final selection (from `cbe_core.jl`, included right after it).

"""
    sector_graded_sketch(frame, side, npre; comp_ratio=0.5, rng, tag) -> TLArray or nothing

The CBE probe for one side of a bond, in the frame's layout (`:left` like `U0`, `:right` like
`V0`), `npre` columns split between the frame's complement and the frame itself (`comp_ratio`;
`nothing` = an unprojected Gaussian). `boost` (sector => extra columns) is Test B's targeted
widening. A saturated frame still returns its isometry half: it probes the opposite side.
"""
function sector_graded_sketch(frame, side::Symbol, npre::Int;
                              comp_ratio::Union{Float64, Nothing} = 0.5,
                              rng::AbstractRNG = MersenneTwister(0x5EED),
                              tag::AbstractString = "cbeG",
                              boost = nothing)
    side in (:left, :right) || throw(ArgumentError("side must be :left or :right"))
    # The ENDPOINTS ARE ALLOWED, and they are the interesting A/B: `1.0` is
    # complement-only (the pure discarded space), `0.0` isometry-only. An earlier version
    # demanded strict inequality and also floored both halves at one column, so
    # "complement only" silently still drew one kept-space column and could not be measured.
    comp_ratio === nothing || 0.0 <= comp_ratio <= 1.0 || throw(ArgumentError(
        "comp_ratio must be in [0, 1] or nothing, got $comp_ratio"))
    npre >= 1 || throw(ArgumentError("npre must be at least 1, got $npre"))

    fl = side === :left ? 3 : 1                       # the frame's bond leg
    a, b = side === :left ? (1, 2) : (2, 3)           # the fused local pair
    sk = side === :left ? 3 : 1                       # the sketch leg, in frame layout
    perp_of = side === :left ? perp_component : perp_component_right
    # `fusion_basis` returns (a, b, fused); a right frame carries its bond leg first.
    tolayout(t) = side === :left ? t : to_concrete(permutedims(t, (3, 1, 2)))

    F = fusion_basis(frame, a, b; tag = tag)
    dfull, dT = _frame_sector_dims(frame, side, F, "sector_graded_sketch")
    dC    = Dict(q => max(dfull[q] - dT[q], 0) for q in keys(dfull))

    # ── comp_ratio = nothing: a plain sector-graded Gaussian, no projector in the probe ──
    # The preselection still applies the exact `P_perp`, so orthogonality is unaffected.
    if comp_ratio === nothing
        cols = _alloc_columns(dfull, npre)
        boost === nothing || _boost!(cols, boost, dfull)
        R = _trim_per_sector(_randomize_like(F, rng), 3, cols)
        R === nothing && return nothing
        Rl = to_concrete(tolayout(R))
        return norm(Rl) > 0 ? Rl : nothing
    end

    # `n_c + n_t == npre` is only the TARGET split, and either may be zero -- that is what
    # makes the pure cases reachable.
    n_c = clamp(round(Int, npre * comp_ratio), 0, npre)
    n_t = npre - n_c

    # A half that cannot fill its columns hands the shortfall to the other half, so the probe stays
    # `npre` wide near a chain edge instead of collapsing.
    cc = _alloc_columns(dC, n_c)
    n_t += n_c - sum(values(cc); init = 0)        # complement short -> widen the isometry half
    ct = _alloc_columns(dT, n_t)
    short = n_t - sum(values(ct); init = 0)
    short > 0 && (cc = _alloc_columns(dC, n_c + short))  # and back, if that half is short too
    if boost !== nothing
        # A sector whose complement is already fully drawn can still hide range behind the
        # frame's own span, so the remainder goes to the isometry half rather than vanishing.
        _boost!(ct, _boost!(cc, boost, dC), dT)
    end

    # NEVER RETURN AN EMPTY PROBE while the space has dimensions. The sketch probes the frame
    # on the OPPOSITE side of the bond, so a saturated frame is still a perfectly good probe --
    # `Om = frame` is the EXACT probe. Declining here refuses to expand one side of a bond
    # because the other was full, the edge blockage above in its purest form.
    # (`comp_ratio = 1.0` on a saturated frame hits it: `dC` is all zeros and `n_t` is 0.)
    if sum(values(cc); init = 0) + sum(values(ct); init = 0) == 0
        ct = _alloc_columns(dfull, npre)
    end

    blocks = Any[]

    # Complement half: random, then the frame projected OUT.
    RC = _trim_per_sector(_randomize_like(F, rng), 3, cc)
    if RC !== nothing
        C = perp_of(frame, tolayout(RC))
        norm(C) > 0 && push!(blocks, C)
    end

    # Isometry half: random, then the frame projected IN. `R - perp(R)` is the projection
    # onto the frame's span, built from the same primitive so the two halves can never
    # disagree about what "the frame's span" is.
    RT = _trim_per_sector(_randomize_like(F, rng), 3, ct)
    if RT !== nothing
        R = tolayout(RT)
        T = to_concrete(R - perp_of(frame, R))
        norm(T) > 0 && push!(blocks, T)
    end

    isempty(blocks) && return nothing
    length(blocks) == 1 && return blocks[1]
    return to_concrete(oplus(blocks, (sk,)))
end

# ── preselection and final selection (`RSVDpreBE0SiQS.m`) ──
# Preselect: sketch, project off the frame (twice, the reference's `RepairOrtho`), orthonormalise.
# Final: rank the candidates by their weight in the small `U0L' (H Theta) V0R'`.

"""
    _range_find(draw, wleg, select, qleg, w0, wcap, ncap) -> (Q, err, n)

Randomised range finder for one side: double the probe width until the preselection SVD cuts
something (`n < w`), the admitted count reaches `ncap`, the width reaches `wcap`, or the probe
runs out of columns. Returns the basis, the discarded weight and `n`. At most `log2(wcap)` passes.
"""
function _range_find(draw, wleg::Int, select, qleg::Int, w0::Int, wcap::Int, ncap::Int)
    w = clamp(w0, 1, max(wcap, 1))
    while true
        Om = draw(w)
        Om === nothing && return (nothing, 0.0, 0)
        wd = leg_dim(Om, wleg)
        Q, err = select(Om)
        Q === nothing && return (nothing, 0.0, 0)
        n = leg_dim(Q, qleg)
        (n < wd || n >= ncap || wd < w || w >= wcap) && return (Q, err, n)
        w = min(2 * w, wcap)
    end
end

"""
    _sample_residual(Q, project, sketch, draw, nh, wleg, side) -> (est, per_sector, cols)

Test B: estimate `||(I - QQ†) P_perp (H Theta)||_F` from independent holdout probes, normalised
per sector (`sum_q ||A Om_q||^2 / n_q`). Returns the estimate, each sector's squared share and the
holdout columns per sector.
"""
function _sample_residual(Q, project, sketch, draw, nh::Int, wleg::Int, side::Symbol)
    side in (:left, :right) || throw(ArgumentError("side must be :left or :right"))
    per = Dict{Any, Float64}()
    cols = Dict{Any, Int}()
    nh >= 1 || return (0.0, per, cols)
    Om = draw(nh)
    Om === nothing && return (0.0, per, cols)
    for (q, d) in Om.spaces[wleg]
        d > 0 && (cols[q] = d)
    end
    isempty(cols) && return (0.0, per, cols)
    Y = sketch(Om)
    Y === nothing && return (0.0, per, cols)
    R = to_concrete(project(Y))
    norm(R) <= _CBE_EPS && return (0.0, per, cols)
    if Q !== nothing
        off = side === :left ? perp_component(Q, R) : perp_component_right(Q, R)
        # `project` is reapplied so the estimate cannot be polluted by a component along the
        # frame, which `Q` is not meant to cover.
        R = to_concrete(project(to_concrete(off)))
    end
    # The sketch contracts the probe's local legs and leaves its width leg free, so the
    # residual's width sectors ARE the probe's sectors, label for label.
    rleg = side === :left ? 3 : 1
    for (q, d) in R.spaces[rleg]
        n_q = get(cols, q, 0)
        n_q > 0 || error("Test B: residual sector $q has no holdout column -- the sketch " *
                         "did not preserve the probe's width labels, so the per-sector " *
                         "normalisation would be wrong")
        Rq = length(R.spaces[rleg]) == 1 ? R :
             to_concrete(getsub(R, rleg, s -> s == q ? (1:d) : nothing))
        per[q] = norm(Rq)^2 / n_q
    end
    return (sqrt(sum(values(per); init = 0.0)), per, cols)
end

"""
    _target_sectors!(boost, per, Om, wleg, alloc) -> Int

Where Test B widens next: double the columns of every sector whose share of the missed residual
exceeds a fair `1/S` split of the allowance. Returns the number of sectors targeted.
"""
function _target_sectors!(boost::Dict, per::Dict, Om, wleg::Int, alloc::BasisAllocation)
    isempty(per) && return 0
    have = Om === nothing ? Dict{Any, Int}() : Dict{Any, Int}(q => d for (q, d) in Om.spaces[wleg])
    share = alloc.sample^2 / length(per)
    n = 0
    for (q, s) in per
        alloc.scale^2 * s > share || continue
        boost[q] = get(boost, q, 0) + max(get(have, q, 0), 1)
        n += 1
    end
    return n
end

"""
    _one_sided_tail(M, alloc, kmax, stol, ledger) -> U or nothing

Test A on one side's candidates. `M` carries the candidate leg first (the conjugate of the
side's bond leg) and the opposite local space second; its singular values are the candidates'
weights. Returns the left singular vectors of the smallest rank the tail test allows (the
coefficients that rotate the candidates onto their ranked directions), or `nothing` if no
direction is needed. Weight cut by `kmax` counts toward the tail, as in the joint branch.
"""
function _one_sided_tail(M, alloc::BasisAllocation, kmax::Int, stol::Float64,
                         ledger::Union{Nothing, AdaptiveLedger})
    (length(M.qlabels) == 0 || norm(M) == 0) && return nothing
    res = svd(M, (1,); Nkeep = kmax, cutoff = stol, get_lists = true)
    nk = length(res.kept_list)
    k, tail = _tail_rank(res.kept_list, alloc.scale, alloc.tail, nk;
                         base = _sq_weight(res.trunc_list))
    if ledger !== nothing
        ledger.max_tail = max(ledger.max_tail, tail)
        tail <= alloc.tail || _unmet!(ledger, :tail)
    end
    k == 0 && return nothing
    k < nk && (res = svd(M, (1,); Nkeep = k, cutoff = stol))
    return to_concrete(res.U)
end

"""
    _record_sample!(ledger, side, ind, per, hcols, sketch, wleg, alloc)

Write one Test B verdict into the ledger: the per-side maximum, and a `sample_log` entry with
the per-sector indicators and a bound on the estimate's relative sampling error.

THE UNCERTAINTY BOUND. For a complex Gaussian column `||A w||^2 = sum_i s_i^2 |z_i|^2` with
`|z_i|^2 ~ Exp(1)`, so its variance is `sum_i s_i^4 <= ||A||_F^4`. Averaging `n_q` columns gives
`var(per_q) <= per_q^2 / n_q`, and the relative standard deviation of `est = sqrt(sum per_q)` is
half that of its square: `0.5 * sqrt(sum_q per_q^2 / n_q) / sum_q per_q`. An upper bound, tight
for a rank-1 miss -- the case that matters, a single missed direction.
"""
function _record_sample!(ledger::AdaptiveLedger, side::Symbol, ind::Float64, per::Dict,
                         hcols::Dict, sketch, wleg::Int, alloc::BasisAllocation)
    ledger.max_sample = max(ledger.max_sample, ind)
    if side === :left
        ledger.max_sample_l = max(ledger.max_sample_l, ind)
    else
        ledger.max_sample_r = max(ledger.max_sample_r, ind)
    end
    tot = sum(values(per); init = 0.0)
    relunc = tot > 0 ? 0.5 * sqrt(sum(s^2 / hcols[q] for (q, s) in per; init = 0.0)) / tot : 0.0
    push!(ledger.sample_log,
          (side = side, est = ind, relunc = relunc,
           per = Dict{Any, Float64}(q => alloc.scale * sqrt(s) for (q, s) in per),
           holdout = copy(hcols),
           sketch = sketch === nothing ? Dict{Any, Int}() :
                    Dict{Any, Int}(q => d for (q, d) in sketch.spaces[wleg])))
    return ledger
end
