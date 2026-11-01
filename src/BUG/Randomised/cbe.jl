# Moved from BUG-Julia@b155550:src/RSVDCBEBondUpdate/expansion/cbe_core.jl, src/RSVDCBEBondUpdate/expansion/cbe_expand.jl, src/RSVDCBEBondUpdate/expansion/cbe_options.jl, src/RSVDCBEBondUpdate/expansion/cbe_bond_update.jl, src/RSVDCBEBondUpdate/operators/mpo_env.jl, src/RSVDCBEBondUpdate/operators/mpo_fold.jl, src/RSVDCBEBondUpdate/expansion/bond_expansion.jl.

# cbe.jl: the shared sketch closures and sector-graded probe helpers of controlled bond
# expansion, and the `DDQRExpansion` strategy the Core sweeps call. The selection
# (`cbe_expand.jl`), the gate-based bond update (`cbe_bond_update.jl`) and the MPO paths
# (`cbe_mpo.jl`) are included right after it.

# ── from src/RSVDCBEBondUpdate/expansion/cbe_core.jl ──

# ── project-first sketches: `Y = (P_perp (H Theta)) Om'` ──
# `P_perp` and `Om` act on disjoint legs, so the order is exact; `_preselect_*` projects again
# (idempotent) because it also orthonormalises and returns the residual.

# `H*Theta` is built once per `cbe_expand` and shared by its three sketch calls; the arithmetic is
# the same as building it per call.

"""
    _sketch_closures(f, buildHT) -> (skl, skr)

The `(skl, skr)` pair [`cbe_expand`](@ref) wants, sharing one lazily built `H*Theta` and one
projection per side. Lazy because `cbe_expand` often returns before sketching (saturated frames).
"""
function _sketch_closures(f::BondFrame, buildHT; share::Bool = true)
    HT = Ref{Any}(nothing)
    AL = Ref{Any}(nothing)          # P_perp^L (H Theta) -- wanted by BOTH `skl` calls
    AR = Ref{Any}(nothing)          # (H Theta) P_perp^R
    # ⚠ `share = false` IS THE OLD BEHAVIOUR, KEPT AS AN A/B AND NOTHING ELSE. A performance claim
    # that can only be reproduced by checking out an old commit is not reproducible in practice;
    # this makes "how much did sharing buy?" a switch that `benchmarks/rsvd_parallel.jl` can flip
    # inside ONE process -- which matters here, because a cold Julia process spends ~10 minutes
    # compiling before it can time anything.
    _ht() = share ? (HT[] === nothing && (HT[] = buildHT()); HT[]) : buildHT()
    function _al()
        share && AL[] !== nothing && return AL[]
        H = _ht()
        c = contract(f.U0', (1, 2), H, (1, 2))             # (bond, site_r, link_r)
        A = to_concrete(H - to_concrete!(contract(f.U0, (3,), c, (1,))))
        share && (AL[] = A)
        return A
    end
    function _ar()
        share && AR[] !== nothing && return AR[]
        H = _ht()
        c = contract(H, (3, 4), f.V0', (2, 3))             # (link_l, site_l, bond)
        A = to_concrete(H - to_concrete!(contract(c, (3,), f.V0, (1,))))
        share && (AR[] = A)
        return A
    end
    skl = Om -> to_concrete!(contract(_al(), (3, 4), Om', (2, 3)))       # (link_l, site_l, g)
    skr = Om -> to_concrete(permutedims(contract(_ar(), (1, 2), Om', (1, 2)), (3, 1, 2)))
    return skl, skr
end

"""
    sketch_h_left(f, h, i, lenv, renv, Om) -> TLArray

`Y = (P_perp (H Theta)) Om†`, legs `(link_l, site_l, g)`: the discarded-space component of
`H Theta` sketched from the right, with `Om` in `V0`'s layout. A standalone call rebuilds
`H*Theta`; sweeps use the cached closures.
"""
sketch_h_left(f::BondFrame, h::XXZChain, i::Int, lch::ChannelSet, rch::ChannelSet, Om) =
    first(_sketch_closures(f, () -> apply_h_two_site(frame_theta(f), h, i, lch, rch)))(Om)

"""
    sketch_h_right(f, h, i, lenv, renv, Om) -> TLArray

Mirror of [`sketch_h_left`](@ref): the right matricization of `(H Theta) P_perp^R` with
`P_perp^R = I - V0' V0`, sketched from the left. `Om` carries `U0`'s layout
`(link_l, site_l, g)` and the result is `(g, site_r, link_r)`.
"""
sketch_h_right(f::BondFrame, h::XXZChain, i::Int, lch::ChannelSet, rch::ChannelSet, Om) =
    last(_sketch_closures(f, () -> apply_h_two_site(frame_theta(f), h, i, lch, rch)))(Om)

# ── the sector-graded Gaussian sketch (`GaussRandMat`, `RSVDpreBE0SiQS.m:557`) ──
# `CompRatio·Dpre` columns in the frame's complement (per sector, by complement dimension) and the
# rest inside the frame, so empty-but-reachable sectors are sampled through `H` by construction.

"""
    _randomize_like(F, rng) -> TLArray

`F`'s sparsity pattern with Gaussian payloads. Only the blocks the symmetry allows are
ever touched.
"""
function _randomize_like(F, rng::AbstractRNG)
    T = eltype(F) <: Complex ? eltype(F) : ComplexF64
    return TLArray(
        symm(F),
        copy(F.qlabels),
        copy(F.wmatdata),
        copy(F.wmatinfo),
        [randn(rng, T, size(r)) for r in F.RMTs],
        F.inds,
        F.spaces,
    )
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
    (tot == 0 || budget <= 0) && return Dict(q => 0 for q in keys(dims))
    return Dict(q => min(d, ceil(Int, d / tot * budget)) for (q, d) in dims)
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
    for (q, b) in boost
        haskey(cap, q) || continue
        have = get(cols, q, 0)
        add = clamp(b, 0, max(cap[q] - have, 0))
        cols[q] = have + add
        b > add && (left[q] = b - add)
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
    a, b = side === :left ? (1, 2) : (2, 3)
    F = fusion_basis(frame, a, b; tag = tag)
    # `fusion_basis` returns (a, b, fused); a right frame carries its bond leg first.
    return side === :left ? F : to_concrete(permutedims(F, (3, 1, 2)))
end

# ── the strategy the Core sweeps call (Core/RankAdaptation/basis_expansion.jl) ──

"""
    DDQRExpansion()

The default [`BasisExpansion`](@ref) of `cbe_bug_step!`: each frame grows by the DD-projected,
MPO-streamed, QR-only CBE ([`dd_cbe_expand`](@ref)), and the frontier sweeps sketch and estimate
their outside-space action with the sector-graded Gaussian probes of this file.
"""
struct DDQRExpansion <: BasisExpansion end

expand_frame(::DDQRExpansion, f::BondFrame, mpo, i::Int, lch, rch, side::Symbol; kwargs...) =
    dd_cbe_expand(f, mpo, i, lch, rch, side; kwargs...)

sketch_closures(::DDQRExpansion, f::BondFrame, build_action) = _sketch_closures(f, build_action)

folded_sketch_closures(::DDQRExpansion, f::BondFrame, mpo, i::Int, lch, rch) =
    _fold_sketch_closures(f, mpo, i, lch, rch)

"""
    sampled_leak(::DDQRExpansion, f, skl, skr, side, n, rng) -> Float64

The randomised estimate of the frame's miss: `n` sector-graded Gaussian columns on the FULL opposite
local space (no projector in the probe), through the production sketch closures `(skl, skr)` --
shared or folded -- and Test B's per-sector unbiased normalisation (`_sample_residual`). An
estimate, not a bound.
"""
function sampled_leak(
    ::DDQRExpansion,
    f::BondFrame,
    skl,
    skr,
    side::Symbol,
    n::Int,
    rng::AbstractRNG,
)
    if side === :left
        draw = w -> sector_graded_sketch(f.V0, :right, w; comp_ratio = nothing, rng = rng)
        return first(
            _sample_residual(nothing, Y -> perp_component(f.U0, Y), skl, draw, n, 1, :left),
        )
    end
    draw = w -> sector_graded_sketch(f.U0, :left, w; comp_ratio = nothing, rng = rng)
    return first(
        _sample_residual(nothing, Y -> perp_component_right(f.V0, Y), skr, draw, n, 3, :right),
    )
end
