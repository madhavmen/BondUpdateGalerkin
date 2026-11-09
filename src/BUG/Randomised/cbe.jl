# cbe.jl: the sketch closures (a built `H Theta`, or the probe folded into the MPO contraction)
# and the sector-graded probe helpers of controlled bond expansion, and the `DDQRExpansion`
# strategy the Core sweeps call. The probe and the residual test (`sketch.jl`) and the range
# finder (`range_finder.jl`) are included before it; the DD-QR kernels, controller and bond
# expansion (`dd_cbe*.jl`) after it.

# ── project-first sketches: `Y = (P_perp (H Theta)) Om'` ──
# `P_perp` and `Om` act on disjoint legs, so the order is exact; `_preselect_*` projects again
# (idempotent) because it also orthonormalises and returns the residual.

# `H*Theta` is built once per bond visit and shared by every sketch call on it; the arithmetic is
# the same as building it per call.

"""
    _sketch_closures(frames, buildHT) -> (skl, skr)

The `(skl, skr)` sketch pair of one bond, sharing one lazily built `H*Theta` and one projection per
side. Lazy because a saturated frame returns before it sketches.
"""
function _sketch_closures(frames::BondFrame, buildHT)
    HT = Ref{Any}(nothing)
    AL = Ref{Any}(nothing)          # P_perp^L (H Theta), shared by every `skl` call
    AR = Ref{Any}(nothing)          # (H Theta) P_perp^R
    _ht() = (HT[] === nothing && (HT[] = buildHT()); HT[])
    # P_perp^L (H Theta) = H Theta - U0 (U0' H Theta), legs (link_l, site_l, site_r, link_r).
    function _al()
        AL[] !== nothing && return AL[]
        h_theta = _ht()
        frame_coeffs = contract(frames.U0', (1, 2), h_theta, (1, 2))  # (bond, site_r, link_r)
        AL[] = TLArray(h_theta - contract(frames.U0, (3,), frame_coeffs, (1,)))
        return AL[]
    end
    # (H Theta) P_perp^R = H Theta - (H Theta V0') V0, same leg order.
    function _ar()
        AR[] !== nothing && return AR[]
        h_theta = _ht()
        frame_coeffs = contract(h_theta, (3, 4), frames.V0', (2, 3))  # (link_l, site_l, bond)
        AR[] = TLArray(h_theta - contract(frame_coeffs, (3,), frames.V0, (1,)))
        return AR[]
    end
    skl = Om -> contract(_al(), (3, 4), Om', (2, 3))       # (link_l, site_l, g)
    skr = Om -> TLArray(permutedims(contract(_ar(), (1, 2), Om', (1, 2)), (3, 1, 2)))
    return skl, skr
end

# ── fold-omega-first: the sketch that never forms H*Theta ──
# Contracting `Om` in first makes the cost scale with the sketch width g, not d*chi. It is exact,
# since P_perp and Om act on disjoint legs and commute.
# Each probe needs its own folded chain, so it wins only for a narrow probe (g < chi).

"""
    _fold_sketch_closures(frame, mpo, bond, lenv, renv) -> (skl, skr)

The `(skl, skr)` sketch pair of [`_sketch_closures`](@ref), with `Om` folded into the contraction chain
instead of applied to a finished `H*Theta`.

`skl(Om)` takes `Om` in `V0`'s layout `(g, site_r, link_r)` and returns `(link_l, site_l, g)`;
`skr(Om)` takes `U0`'s layout `(link_l, site_l, g)` and returns `(g, site_r, link_r)`. Both
return the DISCARDED-space component, i.e. with `P_perp` already applied -- identical to
[`_sketch_closures`](@ref)'s output.
"""
function _fold_sketch_closures(frame::BondFrame, mpo::MPO, bond::Int, lenv::MPOLink, renv::MPOLink)
    left_mpo, right_mpo = mpo[bond], mpo[bond+1]
    # `Theta` is shared, `H*Theta` is not. `frame_theta` is `U0*S0*V0` -- an `O(chi^2 d^2)`
    # object the frame already implies, with no `H` in it. Rebuilding it per call would be pure
    # waste; caching the thing this function exists NOT to build would defeat the purpose.
    Theta = frame_theta(frame)                       # (ℓ_l, s_l, s_r, ℓ_r)

    # ── LEFT: fold `Om` through renv and right_mpo, then Theta, then left_mpo and lenv ─────
    function skl(Om)
        Omd = Om'                                # (g, s_bra_r, bra_r)
        # 1. the right cap. At the boundary `renv.E` is absent and `right_mpo`'s `w_r` is the
        #    dim-1 leg left by trimming, dropped rather than contracted -- the same asymmetry
        #    `apply_h_two_site` handles, and the reason this is not one expression.
        right_cap = if renv.E === nothing
            # (g, bra_r, w_mid, s_ket_r, w_r)
            partial = contract(Omd, (2,), right_mpo, (3,))
            TLArray(deleteSingleton(partial, 5))         # (g, bra_r, w_mid, s_ket_r)
        else
            partial = contract(Omd, (3,), renv.E, (1,))  # (g, s_bra_r, w, ket_r)
            # right_mpo = (w_mid, s_ket_r, s_bra_r, w_r); close `s_bra_r` and `w_r`.
            # -> (g, ket_r, w_mid, s_ket_r)
            contract(partial, (2, 3), right_mpo, (3, 4))
        end
        # 2. into Theta: close the ket site and ket link on the right. -> (ℓ_l, s_l, g, w_mid)
        theta_capped = contract(Theta, (3, 4), right_cap, (4, 2))
        # 3. left_mpo = (w_l, s_ket_l, s_bra_l, w_mid); close the ket site and the shared MPO leg.
        #    -> (ℓ_l, g, w_l, s_bra_l)
        with_mpo = contract(theta_capped, (2, 4), left_mpo, (2, 4))
        sketch = if lenv.E === nothing
            partial = TLArray(deleteSingleton(with_mpo, 3))   # (ℓ_l, g, s_bra_l)
            TLArray(permutedims(partial, (1, 3, 2)))          # (ℓ_l, s_bra_l, g)
        else
            # lenv.E = (bra, w, ket); close the ket link and `w_l`. -> (g, s_bra_l, bra)
            partial = contract(with_mpo, (1, 3), lenv.E, (3, 2))
            _unprime(TLArray(permutedims(partial, (3, 2, 1))), 1)   # (bra, s_bra_l, g)
        end
        return _project_left(frame, sketch)
    end

    # ── RIGHT: the mirror -- fold `Om` through lenv and left_mpo first ──────────────────
    function skr(Om)
        Omd = Om'                                # (bra_l, s_bra_l, g)
        left_cap = if lenv.E === nothing
            # (bra_l, g, w_l, s_ket_l, w_mid)
            partial = contract(Omd, (2,), left_mpo, (3,))
            TLArray(deleteSingleton(partial, 3))         # (bra_l, g, s_ket_l, w_mid)
        else
            partial = contract(Omd, (1,), lenv.E, (1,))  # (s_bra_l, g, w, ket_l)
            # left_mpo = (w_l, s_ket_l, s_bra_l, w_mid); close `s_bra_l` and `w_l`.
            # -> (g, ket_l, s_ket_l, w_mid)
            capped = contract(partial, (1, 3), left_mpo, (3, 1))
            TLArray(permutedims(capped, (2, 1, 3, 4)))  # (ket_l, g, s_ket_l, w_mid)
        end
        # into Theta: close the ket link and ket site on the left. -> (s_r, ℓ_r, g, w_mid)
        theta_capped = contract(Theta, (1, 2), left_cap, (1, 3))
        # right_mpo = (w_mid, s_ket_r, s_bra_r, w_r); close the ket site and the shared MPO leg.
        # -> (ℓ_r, g, s_bra_r, w_r)
        with_mpo = contract(theta_capped, (1, 4), right_mpo, (2, 1))
        sketch = if renv.E === nothing
            partial = TLArray(deleteSingleton(with_mpo, 4))   # (ℓ_r, g, s_bra_r)
            TLArray(permutedims(partial, (2, 3, 1)))          # (g, s_bra_r, ℓ_r)
        else
            # -> (g, s_bra_r, bra_r)
            partial = contract(with_mpo, (1, 4), renv.E, (3, 2))
            _unprime(partial, 3)
        end
        return _project_right(frame, sketch)
    end

    return skl, skr
end

"`P_perp^L sketch` with `P_perp^L = I - U0 U0'`, for `sketch` of layout `(link_l, site_l, g)`."
function _project_left(frame::BondFrame, sketch)
    kept_coeffs = contract(frame.U0', (1, 2), sketch, (1, 2))      # (bond, g)
    return TLArray(sketch - contract(frame.U0, (3,), kept_coeffs, (1,)))
end

"`sketch P_perp^R` with `P_perp^R = I - V0' V0`, for `sketch` of layout `(g, site_r, link_r)`."
function _project_right(frame::BondFrame, sketch)
    kept_coeffs = contract(sketch, (2, 3), frame.V0', (2, 3))      # (g, bond)
    return TLArray(sketch - contract(kept_coeffs, (2,), frame.V0, (1,)))
end

# ── the sector-graded Gaussian sketch (`GaussRandMat`, `RSVDpreBE0SiQS.m:557`) ──
# `CompRatio·Dpre` columns in the frame's complement (per sector, by complement dimension) and the
# rest inside the frame, so empty-but-reachable sectors are sampled through `H` by construction.

"""
    _randomize_like(fused_basis, rng) -> TLArray

`fused_basis`'s sparsity pattern with Gaussian payloads. Only the blocks the symmetry allows are
ever touched.
"""
function _randomize_like(fused_basis, rng::AbstractRNG)
    scalar_type = eltype(fused_basis) <: Complex ? eltype(fused_basis) : ComplexF64
    stored_blocks = fused_basis.RMTs
    block_rank = isempty(stored_blocks) ? 0 : ndims(eltype(stored_blocks))
    # Gaussian payloads in the stored layout; a slot without a block (a known-zero sector) stays so
    blocks = Vector{Array{scalar_type,block_rank}}(undef, length(stored_blocks))
    for slot in eachindex(stored_blocks)
        isassigned(stored_blocks, slot) &&
            (blocks[slot] = randn(rng, scalar_type, size(stored_blocks[slot])))
    end
    stored = TLArray(
        symm(fused_basis),
        copy(Telum.stored_qlabels(fused_basis)),
        copy(fused_basis.wmatdata),
        copy(fused_basis.wmatinfo),
        blocks,
        Telum.stored_inds(fused_basis),
        Telum.stored_spaces(fused_basis),
    )
    # the same deferred leg permutation and conjugation as `fused_basis` (Telum view state)
    permuted = permutedims(stored, Telum.stored_perm(fused_basis))
    return Telum.stored_conj(fused_basis) ? conj(permuted) : permuted
end

"""
    _alloc_columns(dims, budget) -> Dict

Split `budget` columns across sectors in proportion to `dims`, capped per sector at its
own dimension. `ceil` per sector as in `GaussRandMat`, so a small budget still reaches
every sector that has room rather than rounding the small ones to zero -- which is the
whole point when the sector we most need is the empty one.
"""
function _alloc_columns(dims::Dict, budget::Int)
    tot = sum(values(dims); init = 0)
    (tot == 0 || budget <= 0) && return Dict(sector => 0 for sector in keys(dims))
    return Dict(
        sector => min(sector_size, ceil(Int, sector_size / tot * budget)) for
        (sector, sector_size) in dims
    )
end

"""
    _boost!(cols, boost, cap) -> Dict

Add `boost[q]` columns to `cols[q]`, capped at `cap[q]`, and return what did not fit.

A sector absent from `cap` is not reachable from this side, so it is skipped rather than
created: Test B targets a sector by its residual, and a residual can never name a charge the
fusion forbids, but the guard keeps that true by construction instead of by argument.
"""
function _boost!(cols::Dict, boost, cap::Dict)
    left = Dict{Any,Int}()
    for (sector, extra) in boost
        haskey(cap, sector) || continue
        have = get(cols, sector, 0)
        add = clamp(extra, 0, max(cap[sector] - have, 0))
        cols[sector] = have + add
        extra > add && (left[sector] = extra - add)
    end
    return left
end

"Keep at most `column_count` columns of `basis` on `leg`, walking sectors in Telum's order."
function _trim_total(basis, leg::Int, column_count::Int)
    column_count <= 0 && return nothing
    leg_dim(basis, leg) <= column_count && return basis
    left = column_count
    keep = Dict{Any,Int}()
    for (sector, sector_size) in Telum.spaces(basis)[leg]
        keep[sector] = min(sector_size, left)
        left -= keep[sector]
    end
    any(kept > 0 for kept in values(keep)) || return nothing
    return TLArray(
        getsub(
            basis,
            leg,
            label -> (kept_count = get(keep, label, 0); kept_count == 0 ? nothing : 1:kept_count),
        ),
    )
end

# ── the strategy the Core sweeps call (Core/RankAdaptation/basis_expansion.jl) ──

"""
    DDQRExpansion()

The default [`BasisExpansion`](@ref) of `cbe_bug_step!`: each frame grows by the DD-projected,
MPO-streamed, QR-only CBE ([`dd_cbe_expand`](@ref)), and the frontier sweeps sketch and estimate
their outside-space action with the sector-graded Gaussian probes of this file.
"""
Base.@kwdef struct DDQRExpansion <: BasisExpansion
    # :adaptive uses residual-certified incremental sketches. :fixed deliberately
    # skips validation to provide the fixed oversampling/retention baseline.
    mode::Symbol = :adaptive
    fixed_width::Int = 0
    fixed_rank::Int = 0
    holdout_width::Int = 4
    # 1 = existing sequential controller; >1 adds multiple charge-targeted
    # discovery blocks per independent holdout checkpoint.
    batch_blocks::Int = 1
    # :storage = historical first-sector trimming; :svd = dominant singular
    # vectors of the THIN projected DD sketch if the 1.2*chi_b rank cap binds.
    cap_selection::Symbol = :storage
    acceptance_factor::Float64 = 1.0
    # The K/L one-site generator must be Hermitian to use :lanczos.
    krylov_mode::Symbol = :arnoldi
    # 0: residual-controlled stop; >0: fixed depth (experimental baseline).
    krylov_fixed_depth::Int = 0
end

krylov_controls(policy::DDQRExpansion) =
    (recurrence = policy.krylov_mode, fixed_depth = policy.krylov_fixed_depth)

# Grow one frame of the bond `(site, site+1)` by the DD-QR CBE; returns `dd_cbe_expand`'s result.
expand_frame(
    strategy::DDQRExpansion,
    frames::BondFrame,
    mpo,
    site::Int,
    lch,
    rch,
    side::Symbol;
    kwargs...,
) = dd_cbe_expand(
    frames, mpo, site, lch, rch, side;
    kwargs...,
    mode = strategy.mode,
    fixed_width = strategy.fixed_width,
    fixed_rank = strategy.fixed_rank,
    holdout_width = strategy.holdout_width,
    batch_blocks = strategy.batch_blocks,
    cap_selection = strategy.cap_selection,
    acceptance_factor = strategy.acceptance_factor,
)

# The shared lazily built `(skl, skr)` pair for a caller-supplied `H Theta` builder.
sketch_closures(::DDQRExpansion, frames::BondFrame, build_action) =
    _sketch_closures(frames, build_action)

# The `(skl, skr)` pair with the MPO folded into the sketch (no `H Theta` is formed).
folded_sketch_closures(::DDQRExpansion, frames::BondFrame, mpo, site::Int, lch, rch) =
    _fold_sketch_closures(frames, mpo, site, lch, rch)

"""
    sampled_leak(::DDQRExpansion, frames, skl, skr, side, column_count, rng) -> Float64

The randomised estimate of the frame's miss: `column_count` sector-graded Gaussian columns on the
FULL opposite local space (no projector in the probe), through the production sketch closures
`(skl, skr)` -- shared or folded -- and Test B's per-sector unbiased normalisation
(`_sample_residual`). An estimate, not a bound.
"""
function sampled_leak(
    ::DDQRExpansion,
    frames::BondFrame,
    skl,
    skr,
    side::Symbol,
    column_count::Int,
    rng::AbstractRNG,
)
    if side === :left
        draw =
            width -> sector_graded_sketch(frames.V0, :right, width; comp_ratio = nothing, rng = rng)
        return first(
            _sample_residual(
                nothing,
                sketched -> perp_component(frames.U0, sketched),
                skl,
                draw,
                column_count,
                1,
                :left,
            ),
        )
    end
    draw = width -> sector_graded_sketch(frames.U0, :left, width; comp_ratio = nothing, rng = rng)
    return first(
        _sample_residual(
            nothing,
            sketched -> perp_component_right(frames.V0, sketched),
            skr,
            draw,
            column_count,
            3,
            :right,
        ),
    )
end
