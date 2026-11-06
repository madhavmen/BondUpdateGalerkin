# cbe.jl: the shared sketch closures and sector-graded probe helpers of controlled bond
# expansion, and the `DDQRExpansion` strategy the Core sweeps call. The probe and the residual
# tests (`sketch.jl`) and the range finder (`range_finder.jl`) are included before it; the
# selection (`cbe_expand.jl`), the gate-based bond update (`cbe_bond_update.jl`) and the MPO
# paths (`cbe_mpo.jl`) right after it.

# ── project-first sketches: `Y = (P_perp (H Theta)) Om'` ──
# `P_perp` and `Om` act on disjoint legs, so the order is exact; `_preselect_*` projects again
# (idempotent) because it also orthonormalises and returns the residual.

# `H*Theta` is built once per `cbe_expand` and shared by its three sketch calls; the arithmetic is
# the same as building it per call.

"""
    _sketch_closures(frames, buildHT) -> (skl, skr)

The `(skl, skr)` pair [`cbe_expand`](@ref) wants, sharing one lazily built `H*Theta` and one
projection per side. Lazy because `cbe_expand` often returns before sketching (saturated frames).
"""
function _sketch_closures(frames::BondFrame, buildHT; share::Bool = true)
    HT = Ref{Any}(nothing)
    AL = Ref{Any}(nothing)          # P_perp^L (H Theta) -- wanted by BOTH `skl` calls
    AR = Ref{Any}(nothing)          # (H Theta) P_perp^R
    # `share = false` is the old behaviour, kept as an A/B and nothing else. A performance claim
    # that can only be reproduced by checking out an old commit is not reproducible in practice;
    # this makes "how much did sharing buy?" a switch that `benchmarks/rsvd_parallel.jl` can flip
    # inside ONE process -- which matters here, because a cold Julia process spends ~10 minutes
    # compiling before it can time anything.
    _ht() = share ? (HT[] === nothing && (HT[] = buildHT()); HT[]) : buildHT()
    # P_perp^L (H Theta) = H Theta - U0 (U0' H Theta), legs (link_l, site_l, site_r, link_r).
    function _al()
        share && AL[] !== nothing && return AL[]
        h_theta = _ht()
        frame_coeffs = contract(frames.U0', (1, 2), h_theta, (1, 2))  # (bond, site_r, link_r)
        perp_action =
            TLArray(h_theta - contract(frames.U0, (3,), frame_coeffs, (1,)))
        share && (AL[] = perp_action)
        return perp_action
    end
    # (H Theta) P_perp^R = H Theta - (H Theta V0') V0, same leg order.
    function _ar()
        share && AR[] !== nothing && return AR[]
        h_theta = _ht()
        frame_coeffs = contract(h_theta, (3, 4), frames.V0', (2, 3))  # (link_l, site_l, bond)
        perp_action =
            TLArray(h_theta - contract(frame_coeffs, (3,), frames.V0, (1,)))
        share && (AR[] = perp_action)
        return perp_action
    end
    skl = Om -> contract(_al(), (3, 4), Om', (2, 3))       # (link_l, site_l, g)
    skr = Om -> TLArray(permutedims(contract(_ar(), (1, 2), Om', (1, 2)), (3, 1, 2)))
    return skl, skr
end

"""
    sketch_h_left(frames, hamiltonian, site, lch, rch, Om) -> TLArray

`Y = (P_perp (H Theta)) Om†`, legs `(link_l, site_l, g)`: the discarded-space component of
`H Theta` sketched from the right, with `Om` in `V0`'s layout. A standalone call rebuilds
`H*Theta`; sweeps use the cached closures.
"""
sketch_h_left(
    frames::BondFrame,
    hamiltonian::XXZChain,
    site::Int,
    lch::ChannelSet,
    rch::ChannelSet,
    Om,
) = first(
    _sketch_closures(
        frames,
        () -> apply_h_two_site(frame_theta(frames), hamiltonian, site, lch, rch),
    ),
)(
    Om,
)

"""
    sketch_h_right(frames, hamiltonian, site, lch, rch, Om) -> TLArray

Mirror of [`sketch_h_left`](@ref): the right matricization of `(H Theta) P_perp^R` with
`P_perp^R = I - V0' V0`, sketched from the left. `Om` carries `U0`'s layout
`(link_l, site_l, g)` and the result is `(g, site_r, link_r)`.
"""
sketch_h_right(
    frames::BondFrame,
    hamiltonian::XXZChain,
    site::Int,
    lch::ChannelSet,
    rch::ChannelSet,
    Om,
) = last(
    _sketch_closures(
        frames,
        () -> apply_h_two_site(frame_theta(frames), hamiltonian, site, lch, rch),
    ),
)(
    Om,
)

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

"""
    full_local_basis(frame, side; tag="cbeF") -> TLArray

The complete local basis for one side of a bond, in the frame's layout, with `g` the full fused
dimension. Fed to the same sketch closures it turns RSVD-CBE into plain CBE (the A/B reference).
"""
function full_local_basis(frame, side::Symbol; tag::AbstractString = "cbeF")
    side in (:left, :right) || throw(ArgumentError("side must be :left or :right"))
    first_leg, second_leg = side === :left ? (1, 2) : (2, 3)
    fused_basis = fusion_basis(frame, first_leg, second_leg; tag = tag)
    # `fusion_basis` returns (first_leg, second_leg, fused); a right frame carries its bond leg
    # first.
    return side === :left ? fused_basis : TLArray(permutedims(fused_basis, (3, 1, 2)))
end

# ── the strategy the Core sweeps call (Core/RankAdaptation/basis_expansion.jl) ──

"""
    DDQRExpansion()

The default [`BasisExpansion`](@ref) of `cbe_bug_step!`: each frame grows by the DD-projected,
MPO-streamed, QR-only CBE ([`dd_cbe_expand`](@ref)), and the frontier sweeps sketch and estimate
their outside-space action with the sector-graded Gaussian probes of this file.
"""
struct DDQRExpansion <: BasisExpansion end

# Grow one frame of the bond `(site, site+1)` by the DD-QR CBE; returns `dd_cbe_expand`'s result.
expand_frame(
    ::DDQRExpansion,
    frames::BondFrame,
    mpo,
    site::Int,
    lch,
    rch,
    side::Symbol;
    kwargs...,
) = dd_cbe_expand(frames, mpo, site, lch, rch, side; kwargs...)

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
