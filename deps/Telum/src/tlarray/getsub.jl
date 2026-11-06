"""
    getsub(q::TLArray, selector) -> TLArray

Construct a subselected concrete `TLArray` using the legacy selector form.

`selector` is passed to the `TLArray(q, selector)` constructor, which performs
the actual interpretation. Predicate-based methods below are preferred when
the selection depends on qlabels and per-sector block dimensions.
"""
getsub(q::TLArray, selector) = TLArray(q, selector)

"""
    _normalize_getsub_index(i::Int, dim::Int, sector, leg::Int) -> Int

Normalize one user index for a sector-local leg slice.

`i` is the raw index returned by a `getsub` predicate, `dim` is the dimension of
that qlabel sector on `leg`, and `sector`/`leg` are used for diagnostics.
Negative indices count back from the end of the sector; zero is rejected.
"""
function _normalize_getsub_index(i::Int, dim::Int, sector, leg::Int)
    i == 0 && throw(ArgumentError(
        "selector for sector $sector on leg $leg cannot contain 0"))
    return i < 0 ? dim + i + 1 : i
end

"""
    _normalize_getsub_indices(raw, dim::Int, sector, leg::Int) -> Vector{Int}

Normalize one predicate return value into explicit positive sector indices.

`raw` may be an integer, tuple of integers, integer range, or integer vector.
The result is non-empty, duplicate-free, one-based, and bounds-checked against
`dim`. `sector` and `leg` identify the failing qlabel sector in error messages.
"""
function _normalize_getsub_indices(raw, dim::Int, sector, leg::Int)
    inds = if raw isa Integer
        [Int(raw)]
    elseif raw isa Tuple && all(i -> i isa Integer, raw)
        Int[Int(i) for i in raw]
    elseif raw isa AbstractRange{<:Integer}
        Int[i for i in raw]
    elseif raw isa AbstractVector{<:Integer}
        Int[i for i in raw]
    else
        throw(ArgumentError(
            "selector for sector $sector on leg $leg must be :, Int, integer tuple, AbstractRange, or AbstractVector{<:Integer}"))
    end

    isempty(inds) && throw(ArgumentError(
        "selector for sector $sector on leg $leg must not be empty"))
    inds = [_normalize_getsub_index(i, dim, sector, leg) for i in inds]
    length(unique(inds)) == length(inds) || throw(ArgumentError(
        "selector for sector $sector on leg $leg contains duplicate indices"))
    all(1 <= i <= dim for i in inds) || throw(ArgumentError(
        "selector for sector $sector on leg $leg contains out-of-bounds indices for dimension $dim"))
    return inds
end

"""
    _normalize_getsub_predicate_pick(raw, dim::Int, sector, leg::Int)

Interpret one `getsub` predicate result for a sector on a selected leg.

`raw === nothing` drops the sector, `raw isa Colon` keeps the whole sector, and
integer-like values select sector-local indices. Boolean returns are rejected
because they are ambiguous between filtering and integer indexing.
"""
function _normalize_getsub_predicate_pick(raw, dim::Int, sector, leg::Int)
    raw isa Bool && throw(ArgumentError("getsub predicate for sector $sector on leg $leg must not return Bool; use Colon() or nothing explicitly"))
    raw === nothing && return nothing
    raw isa Colon && return Colon()
    return _normalize_getsub_indices(raw, dim, sector, leg)
end

"""
    _collect_getsub_predicate_picks(q, positions, pred::Function)

Evaluate a sector-selection predicate for every selected leg space.

`q` supplies cached space lists, `positions` are visible leg numbers, and
`pred` is called as `pred(sector)` for each qlabel sector on each selected leg.
Returns a dictionary indexed first by leg and then by qlabel sector; sectors
whose predicate returns `nothing` are omitted.
"""
function _collect_getsub_predicate_picks(q::AbstractTLArray, positions, pred::Function)
    selected_picks = Dict{Int, Dict{Any, Any}}()
    for leg in positions
        picks = Dict{Any, Any}()
        for (sector, dim) in spaces(q)[leg]
            pick = _normalize_getsub_predicate_pick(pred(sector), dim, sector, leg)
            isnothing(pick) && continue
            picks[sector] = pick
        end
        selected_picks[leg] = picks
    end
    return selected_picks
end

"""
    _apply_getsub_picks(q::TLArray, positions, selected_picks; preserve_space=false)

Materialize predicate-based subsetting for an identity-state concrete `TLArray`.

`positions` are visible legs selected by the caller. `selected_picks` maps each
selected leg and qlabel sector to either `Colon()` or explicit local indices.
When `preserve_space=false`, selected leg space dimensions are reduced to the
kept sub-block sizes; when `true`, only whole-sector keeps are allowed and the
original space lists are copied unchanged.
"""
function _apply_getsub_picks(q::TLArray{T, QD, N, RD, QT},
                             positions,
                             selected_picks::Dict{Int, Dict{Any, Any}};
                             preserve_space::Bool=false) where {T, QD, N, RD, QT}
    if !_is_identity_view_state(stored_conj(q), stored_scale(q), stored_perm(q))
        return _apply_getsub_picks_preserve_state(q, positions, selected_picks;
                                                  preserve_space=preserve_space)
    end

    if preserve_space
        for leg in positions
            for (sector, pick) in selected_picks[leg]
                pick isa Colon && continue
                throw(ArgumentError(
                    "preserve_space=true is incompatible with slicing sector $sector on leg $leg; return Colon() or nothing instead"))
            end
        end
    end

    selected_leg_set = Set(positions)
    qlabels_out = NTuple{QD, QT}[]
    source_wmat_sectors = Int[]
    RMTs_out = Array{T, RD}[]
    for sector_index in sector_slots(q)
        q.iszero[sector_index] && continue
        picks_by_leg = Dict{Int, Any}()
        keep = true
        for leg in positions
            sector = sector_qlabel(q, sector_index, leg)
            picks = selected_picks[leg]
            haskey(picks, sector) || (keep = false; break)
            picks_by_leg[leg] = picks[sector]
        end
        keep || continue
        selectors = ntuple(d -> get(picks_by_leg, d, Colon()), RD)
        selected_rmt = _getsub_is_whole_sector(selectors) ?
                       q.RMTs[sector_index] : q.RMTs[sector_index][selectors...]
        iszero(sum(abs2, selected_rmt)) && continue
        push!(qlabels_out, ntuple(l -> sector_qlabel(q, sector_index, l), Val(QD)))
        push!(source_wmat_sectors, sector_index)
        push!(RMTs_out, selected_rmt)
    end

    spaces_out = if preserve_space
        _copy_spaces_tuple(spaces(q))
    else
        ntuple(l -> begin
            if l in selected_leg_set
                out = eltype(spaces(q)[l])[]
                picks = selected_picks[l]
                for (sector, dim) in spaces(q)[l]
                    haskey(picks, sector) || continue
                    pick = picks[sector]
                    push!(out, (sector, pick isa Colon ? dim : length(pick)))
                end
                out
            else
                copy(spaces(q)[l])
            end
        end, QD)
    end

    wmatdata, wmatinfo = _copy_wmat_storage(q, source_wmat_sectors; deep=true)
    return TLArray(symm(q), qlabels_out, wmatdata, wmatinfo, RMTs_out, inds(q), spaces_out)
end

"""
    _apply_getsub_picks_preserve_state(q::TLArray, positions, selected_picks; preserve_space=false)

Materialize `getsub` for a concrete tensor that carries non-identity view state.

`positions` are visible legs, but RMT axes are addressed through
`stored_perm(q)`. The result keeps the stored conjugation, scale, and
permutation state instead of canonicalizing the full tensor, while the selected
payload slices are copied into new owned RMT blocks.
"""
function _apply_getsub_picks_preserve_state(q::TLArray{T, QD, N, RD, QT, PS, M, RMT},
                                            positions,
                                            selected_picks::Dict{Int, Dict{Any, Any}};
                                            preserve_space::Bool=false) where {T, QD, N, RD, QT, PS, M, RMT}
    if preserve_space
        for leg in positions
            for (sector, pick) in selected_picks[leg]
                pick isa Colon && continue
                throw(ArgumentError(
                    "preserve_space=true is incompatible with slicing sector $sector on leg $leg; return Colon() or nothing instead"))
            end
        end
    end

    selected_leg_set = Set(positions)
    stored_positions = Dict{Int, Int}(leg => stored_perm(q)[leg] for leg in positions)
    qlabels_out = NTuple{QD, QT}[]
    source_wmat_sectors = Int[]
    TR = eltype(RMT)
    RMTs_out = Array{TR, RD}[]
    for sector_index in sector_slots(q)
        q.iszero[sector_index] && continue
        selectors = ntuple(_ -> Colon(), Val(RD))
        keep = true
        for leg in positions
            sector = sector_qlabel(q, sector_index, leg)
            picks = selected_picks[leg]
            haskey(picks, sector) || (keep = false; break)
            selectors = Base.setindex(selectors, picks[sector], stored_positions[leg])
        end
        keep || continue
        selected_rmt = q.RMTs[sector_index][selectors...]
        iszero(sum(abs2, selected_rmt)) && continue
        push!(qlabels_out, ntuple(l -> stored_sector_qlabel(q, sector_index, l), Val(QD)))
        push!(source_wmat_sectors, sector_index)
        push!(RMTs_out, selected_rmt)
    end

    spaces_out = if preserve_space
        _copy_spaces_tuple(stored_spaces(q))
    else
        ntuple(stored_leg -> begin
            logical_leg = findfirst(==(stored_leg), stored_perm(q))
            if logical_leg in selected_leg_set
                out = eltype(stored_spaces(q)[stored_leg])[]
                picks = selected_picks[logical_leg]
                for (sector, dim) in stored_spaces(q)[stored_leg]
                    haskey(picks, sector) || continue
                    pick = picks[sector]
                    push!(out, (sector, pick isa Colon ? dim : length(pick)))
                end
                out
            else
                copy(stored_spaces(q)[stored_leg])
            end
        end, QD)
    end

    wmatdata, wmatinfo = _copy_wmat_storage(q, source_wmat_sectors; deep=true)
    isdefined_bits = trues(length(qlabels_out))
    iszero_bits = falses(length(qlabels_out))
    return TLArray(Val(:alias_storage_view_state), symm(q), qlabels_out,
                   wmatdata, wmatinfo, RMTs_out, isdefined_bits, iszero_bits,
                   stored_inds(q), spaces_out, stored_conj(q), stored_scale(q),
                   stored_perm(q))
end

"""
    _normalize_getsub_predicate_legs(q, legs) -> Vector{Int}

Normalize leg arguments shared by predicate-based `getsub` methods.

`legs` may be one integer or an iterable of integers. The result preserves the
given order, is non-empty, unique, and bounds-checked against the visible rank
of `q`.
"""
function _normalize_getsub_predicate_legs(q::AbstractTLArray{T, QD}, legs) where {T, QD}
    positions = legs isa Integer ? [Int(legs)] : Int[leg for leg in legs]
    isempty(positions) && throw(ArgumentError("getsub requires at least one leg"))
    all(1 <= leg <= QD for leg in positions) || throw(ArgumentError(
        "getsub legs must lie in 1:$QD, got $positions"))
    length(unique(positions)) == length(positions) || throw(ArgumentError(
        "getsub legs must be unique, got $positions"))
    return positions
end

"""
    _getsub_preserve_space_check(positions, selected_picks; preserve_space)

Validate `preserve_space=true` for predicate-based subsetting.

Preserving cached space lists is only valid when every kept selected sector is
kept whole. `positions` and `selected_picks` identify the offending leg/sector
when a predicate returned explicit indices instead of `Colon()`.
"""
function _getsub_preserve_space_check(positions, selected_picks; preserve_space::Bool)
    preserve_space || return nothing
    for leg in positions
        for (sector, pick) in selected_picks[leg]
            pick isa Colon && continue
            throw(ArgumentError(
                "preserve_space=true is incompatible with slicing sector $sector on leg $leg; return Colon() or nothing instead"))
        end
    end
    return nothing
end

"""
    _getsub_result_spaces(q, positions, selected_picks; preserve_space)

Build output space lists for a lazy `getsub` wrapper.

`q` supplies the source spaces, `positions` are selected visible legs, and
`selected_picks` describes which sectors and local indices survive. If
`preserve_space=true`, source space lists are copied exactly; otherwise selected
legs are filtered and their dimensions shrink to the selected index count.
"""
function _getsub_result_spaces(q::AbstractTLArray{T, QD, N, RD, QT},
                               positions,
                               selected_picks;
                               preserve_space::Bool) where {T, QD, N, RD, QT}
    source_spaces = spaces(q)
    preserve_space && return _copy_spaces_tuple(source_spaces)
    selected_leg_set = Set(positions)
    return ntuple(l -> begin
        if l in selected_leg_set
            out = eltype(source_spaces[l])[]
            picks = selected_picks[l]
            for (sector, dim) in source_spaces[l]
                haskey(picks, sector) || continue
                pick = picks[sector]
                push!(out, (sector, pick isa Colon ? dim : length(pick)))
            end
            out
        else
            copy(source_spaces[l])
        end
    end, QD)
end

"""
    _getsub_sector_selectors(::Val{RD}, positions, selected_picks, q, sector_index)

Return RMT selectors for one source sector, or `nothing` if it is dropped.

`RD` is the stored RMT rank. `positions` names selected physical legs,
`selected_picks` maps leg/qlabel to per-axis selections, and `q`/`sector_index`
provide the sector qlabels. Unselected physical axes and all w-matrix axes are
kept with `Colon()`.
"""
function _getsub_sector_selectors(::Val{RD}, positions, selected_picks, q, sector_index) where {RD}
    picks_by_leg = Dict{Int, GetSubSelector}()
    keep = true
    for leg in positions
        sector = sector_qlabel(q, sector_index, leg)
        picks = selected_picks[leg]
        haskey(picks, sector) || (keep = false; break)
        picks_by_leg[leg] = picks[sector]
    end
    keep || return nothing
    return ntuple(d -> get(picks_by_leg, d, Colon()), Val(RD))
end

@inline _getsub_selector_axis_size(dim::Int, selector::Colon) = dim
@inline _getsub_selector_axis_size(dim::Int, selector::Vector{Int}) = length(selector)

function _getsub_rmt_size(source_size::NTuple{RD, Int},
                          selectors::NTuple{RD, GetSubSelector}) where {RD}
    return ntuple(d -> _getsub_selector_axis_size(source_size[d], selectors[d]), Val(RD))
end

@inline _getsub_is_whole_sector(selectors::NTuple{RD, GetSubSelector}) where {RD} =
    all(selector -> selector isa Colon, selectors)

"""
    _getsub_result_rmt_type(RMTsrc, saved_indices, T, Val(RD)) -> Type

Choose the RMT storage type for a lazy `SubTLArray`.

`RMTsrc` is the source RMT type. `saved_indices` holds the selector tuple for
each retained sector. If every retained sector is a whole-sector keep, the lazy
wrapper may preserve the source RMT type; any actual slice forces dense
`Array{T,RD}` output storage.
"""
function _getsub_result_rmt_type(::Type{RMTsrc},
                                 saved_indices::Vector{NTuple{RD, GetSubSelector}},
                                 ::Type{T},
                                 ::Val{RD}) where {RMTsrc, T, RD}
    return all(_getsub_is_whole_sector, saved_indices) ? RMTsrc : Array{T, RD}
end

"""
    _getsub_lazy(q, legs, pred; preserve_space=false) -> SubTLArray

Create a lazy predicate-based subarray wrapper.

`q` is any lazy or abstract source tensor, `legs` selects visible legs, and
`pred(sector)` returns `nothing`, `Colon()`, or sector-local indices. The
wrapper records retained source sector slots, RMT selectors, expected RMT
sizes, and copied w-matrix metadata, but defers payload extraction until
`compute_sectors`.
"""
function _getsub_lazy(q::AbstractTLArray{T, QD, N, RD, QT, PS, M, RMTsrc},
                      legs,
                      pred::Function;
                      preserve_space::Bool=false) where {T, QD, N, RD, QT, PS, M, RMTsrc}
    positions = _normalize_getsub_predicate_legs(q, legs)
    selected_picks = _collect_getsub_predicate_picks(q, positions, pred)
    _getsub_preserve_space_check(positions, selected_picks; preserve_space=preserve_space)

    qlabels_out = NTuple{QD, QT}[]
    source_sectors = Int[]
    saved_indices = NTuple{RD, GetSubSelector}[]
    rmt_sizes = NTuple{RD, Int}[]
    source_wmat_sectors = Int[]
    iszero_out = Bool[]
    for sector_index in sector_slots(q)
        selectors = _getsub_sector_selectors(Val(RD), positions, selected_picks, q, sector_index)
        isnothing(selectors) && continue
        push!(qlabels_out, ntuple(l -> sector_qlabel(q, sector_index, l), Val(QD)))
        push!(source_sectors, sector_index)
        push!(saved_indices, selectors)
        source_size = is_sector_zero(q, sector_index) ?
            ntuple(_ -> 0, Val(RD)) : sector_rmt_dim(q, sector_index)
        push!(rmt_sizes, _getsub_rmt_size(source_size, selectors))
        push!(source_wmat_sectors, sector_index)
        push!(iszero_out, is_sector_zero(q, sector_index))
    end

    spaces_out = _getsub_result_spaces(q, positions, selected_picks; preserve_space=preserve_space)
    wmatdata, wmatinfo = _copy_wmat_storage(q, source_wmat_sectors; deep=true)
    RMT = _getsub_result_rmt_type(RMTsrc, saved_indices, T, Val(RD))
    RMTs = Vector{RMT}(undef, length(qlabels_out))
    return SubTLArray{T, QD, N, RD, QT, PS, M, RMT}(
        qlabels_out,
        wmatdata,
        wmatinfo,
        RMTs,
        falses(length(qlabels_out)),
        BitVector(iszero_out),
        stored_inds(q),
        spaces_out,
        stored_conj(q),
        stored_scale(q),
        stored_perm(q),
        q,
        source_sectors,
        saved_indices,
        positions,
        rmt_sizes,
        ReentrantLock())
end

"""
    _getsub_materialized_rmt(RMT, rmt, selectors)

Materialize one selected RMT payload for `SubTLArray`.

`RMT` is the wrapper's output RMT type, `rmt` is the source sector payload, and
`selectors` are the stored-axis selectors saved by `_getsub_lazy`. Whole-sector
keeps may share or convert the source payload; sliced sectors are copied into
dense arrays.
"""
function _getsub_materialized_rmt(::Type{RMT},
                                  rmt::RMT,
                                  selectors::NTuple{RD, GetSubSelector}) where {RMT, RD}
    _getsub_is_whole_sector(selectors) || throw(ArgumentError(
        "non-Array whole-sector sharing requires all selectors to be Colon()"))
    return rmt
end

function _getsub_materialized_rmt(::Type{Array{T, RD}},
                                  rmt::Array{T, RD},
                                  selectors::NTuple{RD, GetSubSelector}) where {T, RD}
    _getsub_is_whole_sector(selectors) && return rmt
    return rmt[selectors...]
end

function _getsub_materialized_rmt(::Type{Array{T, RD}},
                                  rmt::AbstractArray,
                                  selectors::NTuple{RD, GetSubSelector}) where {T, RD}
    _getsub_is_whole_sector(selectors) && return Array{T, RD}(rmt)
    selected = rmt[selectors...]
    selected isa Array{T, RD} && return selected
    out = Array{T, RD}(undef, size(selected))
    copyto!(out, selected)
    return out
end

"""
    compute_sectors(q::SubTLArray, sector_inds) -> SubTLArray

Compute selected lazy-subarray sectors in place.

`sector_inds` are output sector slots. The method maps them to source sector
slots, computes each needed source sector once, applies saved RMT selectors,
combines source scale with the wrapper scale when needed, and marks output
sector state. Sector numbering is preserved; no sectors are reordered or
compacted during lazy evaluation.
"""
function compute_sectors(q::SubTLArray{T, QD, N, RD, QT, PS, M, RMT},
                         sector_inds::AbstractVector{<:Integer}) where {T, QD, N, RD, QT, PS, M, RMT}
    requested = Int[]
    source_requested = Int[]
    source_seen = falses(sector_count(q.arr))
    for sector_raw in sector_inds
        sector = Int(sector_raw)
        1 <= sector <= sector_count(q) || throw(BoundsError(q, sector))
        (q.isdefined[sector] || q.iszero[sector]) && continue
        push!(requested, sector)
        source_sector = q.source_sectors[sector]
        if !source_seen[source_sector]
            source_seen[source_sector] = true
            push!(source_requested, source_sector)
        end
    end
    isempty(requested) && return q

    compute_sectors(q.arr, source_requested)
    lock(q.lock) do
        for sector in requested
            (q.isdefined[sector] || q.iszero[sector]) && continue
            source_sector = q.source_sectors[sector]
            rmt, scale = sector_rmt(q.arr, source_sector)
            selectors = q.saved_indices[sector]
            data = _getsub_materialized_rmt(RMT, rmt, selectors)
            if scale != stored_scale(q) && scale != one(typeof(scale))
                data = data * scale
            end
            q.RMTs[sector] = data
            q.isdefined[sector] = true
            q.iszero[sector] = _rmt_iszero(data)
        end
    end
    return q
end

"""
    getsub(q::TLArray, leg::Integer, pred::Function; preserve_space::Bool=false) -> TLArray

Return a new `TLArray` containing only sectors whose sector on `leg` satisfies
`pred`.

`pred(sector)` may return `nothing` to drop that sector, `Colon()` to keep the
full sector, or an integer / integer range / integer tuple / integer vector to
keep sector-local basis positions. Negative indices count from the end of the
sector dimension, and zero is invalid.

If `preserve_space=false` (the default), only `spaces(q)[leg]` is truncated to
the retained sectors and all other leg-space lists are copied unchanged. If
`preserve_space=true`, all cached leg-space lists are preserved exactly and only
the sectors are filtered. This requires `pred` to keep whole sectors, so any
index-selection return value is rejected when `preserve_space=true`.
"""
function getsub(q::TLArray{T, QD, N, RD}, leg::Integer, pred::Function; preserve_space::Bool=false) where {T, QD, N, RD}
    return getsub(q, (Int(leg),), pred; preserve_space=preserve_space)
end

"""
    getsub(q::TLArray, legs, pred::Function; preserve_space::Bool=false) -> TLArray

Return a new `TLArray` containing only sectors whose sectors on every selected leg
satisfy `pred`.

`pred(sector)` may return `nothing` to drop that sector, `Colon()` to keep the
full sector, or an integer / integer range / integer tuple / integer vector to
keep sector-local basis positions. Negative indices count from the end of the
sector dimension, and zero is invalid.

If `preserve_space=false` (the default), each selected leg keeps only the
matching entries in its space list, while unselected legs keep copies of their
original space lists. If `preserve_space=true`, all cached leg-space lists are
preserved exactly and only the sectors are filtered. This requires `pred` to keep
whole sectors, so any index-selection return value is rejected when
`preserve_space=true`.
"""
function getsub(q::TLArray{T, QD, N, RD}, legs::LegList, pred::Function; preserve_space::Bool=false) where {T, QD, N, RD}
    positions = _normalize_getsub_predicate_legs(q, legs)
    selected_picks = _collect_getsub_predicate_picks(q, positions, pred)
    return _apply_getsub_picks(q, positions, selected_picks; preserve_space=preserve_space)
end

"""
    getsub(q::Union{TLArrayContraction, SubTLArray, SingletonTLArray}, leg::Integer,
           pred::Function; preserve_space=false) -> SubTLArray

Lazily subset one visible leg of a lazy tensor.

`leg` is interpreted in visible leg order. `pred` has the same return contract
as the concrete `TLArray` method. The returned wrapper keeps source sector
slots and computes only requested payload slices.
"""
function getsub(q::Union{TLArrayContraction, SubTLArray, SingletonTLArray}, leg::Integer,
                pred::Function; preserve_space::Bool=false)
    return _getsub_lazy(q, (Int(leg),), pred; preserve_space=preserve_space)
end

"""
    getsub(q::Union{TLArrayContraction, SubTLArray, SingletonTLArray}, legs::LegList,
           pred::Function; preserve_space=false) -> SubTLArray

Lazily subset multiple visible legs of a lazy tensor.

`legs` may be a tuple or vector of visible leg numbers. A source sector is kept
only when `pred` keeps the qlabel sector on every selected leg.
"""
function getsub(q::Union{TLArrayContraction, SubTLArray, SingletonTLArray}, legs::LegList,
                pred::Function; preserve_space::Bool=false)
    return _getsub_lazy(q, legs, pred; preserve_space=preserve_space)
end

"""
    getsub(q::TLArray, pred::Function; preserve_space::Bool=false, dir=nothing,
           itag=nothing, plev=nothing, lock=nothing, rev=false) -> TLArray

Apply predicate-based `getsub` to every leg selected by the keyword criteria.

`dir`, `itag`, `plev`, and `lock` are passed to the same leg-selection machinery
as `findlegs`; `rev=true` reverses the selected order before applying the
predicate. `pred` uses the same return convention as the explicit-leg methods.
"""
function getsub(q::TLArray{T, QD, N, RD}, pred::Function; preserve_space::Bool=false,
                dir=nothing, itag=nothing, plev=nothing, lock=nothing,
                rev::Bool=false) where {T, QD, N, RD}
    legs = _resolve_matching_legs(q; dir=dir, itag=itag, plev=plev, lock=lock,
                                  rev=rev, opname="getsub")
    return getsub(q, legs, pred; preserve_space=preserve_space)
end

"""
    getsub(q::Union{TLArrayContraction, SubTLArray, SingletonTLArray}, pred::Function;
           preserve_space=false, dir=nothing, itag=nothing, plev=nothing,
           lock=nothing, rev=false) -> SubTLArray

Lazily apply predicate-based subsetting to legs selected by metadata.

The keyword selectors choose visible legs with `findlegs` semantics. The
returned `SubTLArray` defers all payload slicing while preserving source sector
slot numbering.
"""
function getsub(q::Union{TLArrayContraction, SubTLArray, SingletonTLArray}, pred::Function;
                preserve_space::Bool=false, dir=nothing, itag=nothing,
                plev=nothing, lock=nothing, rev::Bool=false)
    legs = _resolve_matching_legs(q; dir=dir, itag=itag, plev=plev, lock=lock,
                                  rev=rev, opname="getsub")
    return _getsub_lazy(q, legs, pred; preserve_space=preserve_space)
end
