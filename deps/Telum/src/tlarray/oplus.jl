"""
    _normalize_oplus_dims(dimensions, QD::Int; sort_dims=true) -> Tuple{Vararg{Int}}

Normalize the physical legs used as direct-sum axes.

`dimensions` may be one integer or an iterable of integers. `QD` is the visible
rank used for bounds checking. The result is non-empty, unique, and sorted
unless `sort_dims=false` is requested by a caller that needs to preserve input
axis order.
"""
function _normalize_oplus_dims(dimensions, QD::Int; sort_dims::Bool=true)
    dims = dimensions isa Integer ? (Int(dimensions),) : Tuple(Int(d) for d in dimensions)
    isempty(dims) && throw(ArgumentError("oplus requires at least one dimension"))
    all(d -> 1 <= d <= QD, dims) || throw(ArgumentError(
        "oplus dimensions must lie in 1:$QD, got $(collect(dims))"))
    length(unique(dims)) == length(dims) || throw(ArgumentError(
        "oplus dimensions must be unique, got $(collect(dims))"))
    return sort_dims ? Tuple(sort(collect(dims))) : dims
end

"""
    _normalize_oplus_matrix_dims(dimensions, QD::Int)

Normalize the row-axis and column-axis leg groups for matrix `oplus`.

`dimensions` must be a two-entry tuple. Each entry is normalized by
`_normalize_oplus_dims`, and the two groups must be disjoint because they
represent independent direct-sum axes of the block matrix.
"""
function _normalize_oplus_matrix_dims(dimensions, QD::Int)
    dimensions isa Tuple && length(dimensions) == 2 || throw(ArgumentError(
        "matrix oplus requires exactly two axis-dimension specifications"))

    dims1 = _normalize_oplus_dims(dimensions[1], QD)
    dims2 = _normalize_oplus_dims(dimensions[2], QD)
    isempty(intersect(dims1, dims2)) || throw(ArgumentError(
        "matrix-axis oplus legs must be disjoint, got $(collect(dims1)) and $(collect(dims2))"))
    return dims1, dims2
end

"""
    _splist_dim_map(splist::Vector) -> Dict

Convert a leg space list to a qlabel-to-dimension map.

`splist` contains `(qlabels, dim)` entries. Duplicate qlabels are rejected
because direct-sum padding relies on each sector having a single unambiguous
target dimension on that leg.
"""
function _splist_dim_map(splist::Vector)
    dims = Dict{Any, Int}()
    for (qlabels, dim) in splist
        if haskey(dims, qlabels)
            throw(ArgumentError("duplicate qlabel sector encountered in space list: $qlabels"))
        end
        dims[qlabels] = dim
    end
    return dims
end

"""
    _sum_splists_many(splists) -> Vector

Merge several leg space lists for a direct-sum leg.

For each qlabel sector, dimensions are added across all input lists. The output
preserves first-seen qlabel order across `splists`, which keeps sector metadata
stable while accumulating the block-space sizes needed for padding.
"""
function _sum_splists_many(splists)
    isempty(splists) && throw(ArgumentError("cannot sum an empty collection of space lists"))

    dims = Dict{Any, Int}()
    seen = Set{Any}()
    result = copy(first(splists))
    empty!(result)

    for splist in splists
        for (qlabels, dim) in splist
            dims[qlabels] = get(dims, qlabels, 0) + dim
            if qlabels ∉ seen
                push!(result, (qlabels, 0))
                push!(seen, qlabels)
            end
        end
    end

    for i in eachindex(result)
        qlabels, _ = result[i]
        result[i] = (qlabels, dims[qlabels])
    end

    return result
end

_copy_spaces_tuple(spaces::NTuple{QD, Vector}) where {QD} = ntuple(l -> copy(spaces[l]), QD)
_tlarray_eltype(::AbstractTLArray{T}) where {T} = T

"""
    _oplus_pad_tlarray(q, result_spaces, dims_tuple, start_dim_maps, result_dim_maps)

Embed one direct-sum input tensor into the full output block spaces.

`q` supplies sector qlabels, w-matrices, and RMT payloads. `result_spaces` are
the final leg spaces shared by all summands. `dims_tuple` names direct-sum
physical legs. `start_dim_maps[leg][qlabel]` gives this summand's first local
offset within the output sector, while `result_dim_maps[leg][qlabel]` gives the
full output dimension for that qlabel. Non-direct-sum legs are copied
unchanged.
"""
function _oplus_pad_tlarray(q::AbstractTLArray{T, QD, N, RD, QT},
                           result_spaces,
                           dims_tuple,
                           start_dim_maps,
                           result_dim_maps) where {T, QD, N, RD, QT}
    materialize(q)
    dims_set = Set(dims_tuple)
    qlabels = [ntuple(leg -> sector_qlabel(q, sector, leg), Val(QD))::NTuple{QD, QT}
               for sector in sector_slots(q)]
    wmatdata, wmatinfo = _copy_wmat_storage(q; deep=true)
    # Padding a direct-sum leg generally destroys diagonal RMT structure.
    RMTs = Vector{Array{T, RD}}(undef, sector_count(q))
    for sector_index in sector_slots(q)
        q.iszero[sector_index] && continue
        rmt, scale = sector_rmt_permuted(q, sector_index, _identity_rmt_perm(Val(RD)))
        old_sizes = size(rmt)
        new_phys_sizes = collect(old_sizes[1:QD])
        starts = ones(Int, QD)

        for leg in 1:QD
            leg ∈ dims_set || continue
            qlabel = sector_qlabel(q, sector_index, leg)
            new_phys_sizes[leg] = result_dim_maps[leg][qlabel]
            starts[leg] = get(start_dim_maps[leg], qlabel, 1)
        end

        new_sizes = Tuple(vcat(new_phys_sizes, collect(old_sizes[QD+1:end])))
        new_data = zeros(T, new_sizes)
        fill_inds = ntuple(axis -> begin
            if axis <= QD && axis ∈ dims_set
                start = starts[axis]
                stop = start + old_sizes[axis] - 1
                start:stop
            else
                Colon()
            end
        end, RD)
        new_data[fill_inds...] = scale .* rmt
        RMTs[sector_index] = new_data
    end
    return TLArray(symm(q), qlabels, wmatdata, wmatinfo, RMTs, inds(q), _copy_spaces_tuple(result_spaces))
end

"""
    _zero_tlarray_with_spaces(symm, inds, spaces; T=Float64) -> TLArray

Create a zero-sector concrete TLArray with explicit leg spaces.

`symm` and `inds` define the tensor identity and visible leg metadata.
`spaces` is copied into the result so the zero tensor can participate in
direct-sum matrix completion. `T` chooses the element type of any later payloads.
"""
function _zero_tlarray_with_spaces(symm::NTuple{N, Any},
                                  inds::NTuple{QD, TLIndex},
                                  spaces::NTuple{QD, Vector};
                                  T::Type=Float64) where {N, QD}
    QT = qlabeltype(symm)
    qlabels = Matrix{QT}(undef, QD, 0)
    M = n_nonabelian_symmetries(productsymm(symm))
    wmatdata = Float64[]
    wmatinfo = WMatInfo{M}[]
    RMTs = Array{T, QD + N}[]
    return TLArray(symm, qlabels, wmatdata, wmatinfo, RMTs, inds, _copy_spaces_tuple(spaces))
end

"""
    _accumulate_oplus_starts(qs, dims_tuple, QD::Int)

Compute per-summand offsets for direct-sum legs.

`qs` is the aligned list of summand tensors. `dims_tuple` names the physical
legs being summed, and `QD` is the tensor rank. The returned nested vector gives
`starts[qi][leg][qlabel]`, the one-based starting coordinate for summand `qi`
inside the final sector on that leg.
"""
function _accumulate_oplus_starts(qs, dims_tuple, QD::Int)
    dims_set = Set(dims_tuple)
    running = [Dict{Any, Int}() for _ in 1:QD]
    starts = Vector{Vector{Dict{Any, Int}}}(undef, length(qs))

    for (qi, q) in enumerate(qs)
        starts[qi] = [Dict{Any, Int}() for _ in 1:QD]
        for leg in 1:QD
            leg ∈ dims_set || continue
            for (qlabels, dim) in spaces(q)[leg]
                start = get(running[leg], qlabels, 1)
                starts[qi][leg][qlabels] = start
                running[leg][qlabels] = start + dim
            end
        end
    end

    return starts
end

_qindex_match_for_oplus(a::TLIndex, b::TLIndex) =
    a.itags == b.itags && a.dir == b.dir && a.plev == b.plev && a.lock == b.lock

"""
    _inds_match_for_oplus(inds1, inds2) -> Bool

Check whether two visible index tuples are compatible for direct-sum assembly.

Indices must have the same length and match in tags, direction, prime level,
and lock. The check deliberately ignores `dual`, following the local direct-sum
alignment convention.
"""
function _inds_match_for_oplus(inds1, inds2)
    length(inds1) == length(inds2) || return false
    return all(_qindex_match_for_oplus(idx1, idx2) for (idx1, idx2) in zip(inds1, inds2))
end

"""
    _find_oplus_leg_permutation(ref_inds, inds, entry::Int)

Find how an input tensor's legs should be permuted to match reference indices.

`ref_inds` are the target index descriptors, `inds` are one entry's current
descriptors, and `entry` is used in error messages. The function matches legs
by tag, direction, prime level, and lock, requires a unique bijection, and
returns the permutation that reorders the input to reference order.
"""
function _find_oplus_leg_permutation(ref_inds, inds, entry::Int)
    length(inds) == length(ref_inds) || throw(ArgumentError(
        "TLArray entry $entry has rank $(length(inds)), expected $(length(ref_inds))"))

    QD = length(ref_inds)
    candidates = Vector{Vector{Int}}(undef, QD)
    for i in 1:QD
        candidates[i] = [j for j in 1:QD if _qindex_match_for_oplus(inds[j], ref_inds[i])]
        isempty(candidates[i]) && throw(ArgumentError(
            "TLArray entry $entry has no leg matching reference leg $i " *
            "(same itag, direction, prime level, and lock required; dual ignored)"))
    end

    results = Vector{NTuple{QD, Int}}()
    _enum_leg_perms!(results, candidates, Int[], Set{Int}(), QD)
    isempty(results) && throw(ArgumentError(
        "TLArray entry $entry has no bijective leg permutation matching the reference indices"))
    length(results) > 1 && throw(ArgumentError(
        "TLArray entry $entry has ambiguous leg permutation matching the reference indices"))
    return results[1]
end

"""
    _align_oplus_inputs(qs) -> Vector{AbstractTLArray}

Materialize and align vector direct-sum inputs.

`qs` may contain concrete or lazy `AbstractTLArray` values. Each entry is
materialized in place, checked for the same symmetry tuple, and wrapped with a
permutation when its indices uniquely match the first tensor's indices in a
different order.
"""
function _align_oplus_inputs(qs)
    isempty(qs) && throw(ArgumentError("oplus requires at least one TLArray"))
    first(qs) isa AbstractTLArray || throw(ArgumentError("oplus entry 1 is not a TLArray"))

    ref = first(qs)
    materialize(ref)
    aligned = Vector{AbstractTLArray}(undef, length(qs))
    aligned[1] = ref
    for i in 2:length(qs)
        q = qs[i]
        q isa AbstractTLArray || throw(ArgumentError("oplus entry $i is not a TLArray"))
        symm(q) == symm(ref) || throw(ArgumentError(
            "TLArray entry $i has a different symmetry tuple"))
        materialize(q)
        qinds = inds(q)
        perm = _find_oplus_leg_permutation(inds(ref), qinds, i)
        aligned[i] = perm == ntuple(identity, ndims(ref)) ? q : permutedims(q, perm)
    end
    return aligned
end

"""
    _validate_oplus_common(qs) -> AbstractTLArray

Validate common metadata for already aligned direct-sum inputs.

All entries in `qs` must be `AbstractTLArray` values with the same symmetry
tuple, rank, and direct-sum-compatible visible indices. The first tensor is
returned as the reference for output symmetry, indices, and rank.
"""
function _validate_oplus_common(qs)
    isempty(qs) && throw(ArgumentError("oplus requires at least one TLArray"))
    first(qs) isa AbstractTLArray || throw(ArgumentError("oplus entry 1 is not a TLArray"))

    ref = first(qs)
    for (i, q) in enumerate(qs)
        q isa AbstractTLArray || throw(ArgumentError("oplus entry $i is not a TLArray"))
        symm(q) == symm(ref) || throw(ArgumentError(
            "TLArray entry $i has a different symmetry tuple"))
        length(inds(q)) == length(inds(ref)) || throw(ArgumentError(
            "TLArray entry $i has rank $(length(inds(q))), expected $(length(inds(ref)))"))
        _inds_match_for_oplus(inds(q), inds(ref)) || throw(ArgumentError(
            "TLArray entry $i has different indices " *
            "(same itag, direction, prime level, and lock required; dual ignored)"))
    end

    return ref
end

"""
    _oplus_dims_from_keywords(ref::TLArray; dir=nothing, itag=nothing, plev=nothing, lock=nothing, rev=false)

Resolve direct-sum legs from index metadata selectors.

`ref` is the reference tensor after input alignment. At least one selector must
be supplied. `dir`, `itag`, `plev`, `lock`, and `rev` are forwarded to
`findlegs`, and the matched visible leg numbers become the direct-sum axes.
"""
function _oplus_dims_from_keywords(ref::TLArray; dir=nothing, itag=nothing,
                                   plev=nothing, lock=nothing, rev::Bool=false)
    isnothing(dir) && isnothing(itag) && isnothing(plev) && isnothing(lock) &&
        throw(ArgumentError("oplus keyword selection requires at least one selector"))
    dims = Tuple(findlegs(ref; dir=dir, itag=itag, plev=plev, lock=lock, rev=rev))
    isempty(dims) && throw(ArgumentError("oplus keyword selectors did not match any legs"))
    return dims
end

"""
    _build_vector_oplus_spaces(qs, dims_tuple)

Build output leg spaces for vector `oplus`.

`qs` are aligned concrete inputs and `dims_tuple` names direct-sum legs. On
direct-sum legs, qlabel dimensions are added across inputs. On all other legs,
space lists must match exactly and are copied from the reference tensor.
"""
function _build_vector_oplus_spaces(qs, dims_tuple)
    ref = _validate_oplus_common(qs)
    QD = length(inds(ref))
    dims_set = Set(dims_tuple)

    return ntuple(leg -> begin
        if leg ∈ dims_set
            _sum_splists_many([spaces(q)[leg] for q in qs])
        else
            target = spaces(ref)[leg]
            for i in 2:length(qs)
                spaces(qs[i])[leg] == target || throw(ArgumentError(
                    "space lists must match on non-oplus leg $leg"))
            end
            copy(target)
        end
    end, QD)
end

"""
    _materialize_vector_oplus(qs, dims_tuple) -> TLArray

Assemble the direct sum of a vector of aligned concrete tensors.

`qs` supplies the summands, and `dims_tuple` selects physical legs whose sector
dimensions are concatenated. Each summand is padded into the shared output
spaces and then added. Element type is promoted across all inputs.
"""
function _materialize_vector_oplus(qs, dims_tuple)
    ref = _validate_oplus_common(qs)
    result_spaces = _build_vector_oplus_spaces(qs, dims_tuple)
    QD = length(inds(ref))
    result_dim_maps = ntuple(leg -> _splist_dim_map(result_spaces[leg]), QD)
    start_maps = _accumulate_oplus_starts(qs, dims_tuple, QD)
    T = promote_type((_tlarray_eltype(q) for q in qs)...)

    acc = _zero_tlarray_with_spaces(symm(ref), inds(ref), result_spaces; T=T)
    for (q, qstarts) in zip(qs, start_maps)
        padded = _oplus_pad_tlarray(q, result_spaces, dims_tuple, qstarts, result_dim_maps)
        padded = inds(q) == inds(ref) ? padded : TLArray(padded, inds(ref))
        acc = acc + padded
    end
    return acc
end

"""
    _oplus_matrix_entry(mat, i::Int, j::Int)

Read and validate one matrix entry for matrix `oplus`.

Unassigned entries, `nothing`, and `missing` are interpreted as absent blocks
that may be inferred as zero tensors later. Any present value must be an
`AbstractTLArray`; it is materialized in place but not converted to owned
concrete storage.
"""
function _oplus_matrix_entry(mat, i::Int, j::Int)
    if applicable(isassigned, mat, i, j) && !isassigned(mat, i, j)
        return nothing
    end

    val = mat[i, j]
    if val === nothing || val === missing
        return nothing
    end
    val isa AbstractTLArray || throw(ArgumentError(
        "matrix oplus entry ($i, $j) is neither an AbstractTLArray nor an undefined entry"))
    materialize(val)
    return val
end

"""
    _infer_zero_matrix_spaces(first_axis_sources, second_axis_sources, i::Int, j::Int, QD::Int)

Infer leg spaces for an absent matrix-oplus entry.

`first_axis_sources` collects row-wise space constraints and
`second_axis_sources` collects column-wise constraints. `i` and `j` identify the
missing matrix position. For each leg, the available row/column constraints
must agree; otherwise a zero tensor at that block cannot be constructed
unambiguously.
"""
function _infer_zero_matrix_spaces(first_axis_sources, second_axis_sources, i::Int, j::Int, QD::Int)
    return ntuple(leg -> begin
        have_first_axis = haskey(first_axis_sources[i], leg)
        have_second_axis = haskey(second_axis_sources[j], leg)
        if have_first_axis && have_second_axis
            first_axis_sources[i][leg] == second_axis_sources[j][leg] || throw(ArgumentError(
                "cannot infer zero TLArray at ($i, $j): inconsistent spaces on leg $leg"))
            copy(first_axis_sources[i][leg])
        elseif have_first_axis
            copy(first_axis_sources[i][leg])
        elseif have_second_axis
            copy(second_axis_sources[j][leg])
        else
            throw(ArgumentError(
                "cannot infer zero TLArray at ($i, $j): missing space information on leg $leg"))
        end
    end, QD)
end

"""
    oplus(qs::AbstractVector, dimensions)

Direct sum of a vector of `TLArray` objects along one or more physical legs.

`qs` must be non-empty and all entries must be defined tensors. `dimensions`
selects the physical legs whose qlabel-sector dimensions are concatenated
across entries. Inputs may have permuted but uniquely matching leg metadata;
they are aligned before assembly.
"""
function oplus(qs::AbstractVector, dimensions)
    isempty(qs) && throw(ArgumentError("oplus requires at least one TLArray"))
    any(q -> q === nothing || q === missing, qs) && throw(ArgumentError(
        "vector oplus requires every entry to be well defined"))

    aligned = _align_oplus_inputs(collect(qs))
    ref = _validate_oplus_common(aligned)
    dims_tuple = _normalize_oplus_dims(dimensions, length(inds(ref)))
    return _materialize_vector_oplus(aligned, dims_tuple)
end

"""
    oplus(qs::AbstractVector; dir=nothing, itag=nothing, plev=nothing, lock=nothing, rev=false)

Direct sum a vector of tensors along legs selected by index metadata.

`qs` is the non-empty list of tensors. `dir`, `itag`, `plev`, `lock`, and `rev`
select direct-sum legs on the aligned reference tensor using `findlegs`
semantics. All other rules match `oplus(qs, dimensions)`.
"""
function oplus(qs::AbstractVector; dir=nothing, itag=nothing, plev=nothing,
               lock=nothing, rev::Bool=false)
    isempty(qs) && throw(ArgumentError("oplus requires at least one TLArray"))
    any(q -> q === nothing || q === missing, qs) && throw(ArgumentError(
        "vector oplus requires every entry to be well defined"))

    aligned = _align_oplus_inputs(collect(qs))
    ref = _validate_oplus_common(aligned)
    dims_tuple = _oplus_dims_from_keywords(ref; dir=dir, itag=itag, plev=plev,
                                           lock=lock, rev=rev)
    return _materialize_vector_oplus(aligned, dims_tuple)
end

"""
    oplus(q1::AbstractTLArray, q2::AbstractTLArray, dimensions)

Direct sum two tensors along explicit physical legs.

`q1` and `q2` are wrapped into a two-entry vector and processed by the vector
method. `dimensions` has the same meaning as in `oplus(qs, dimensions)`.
"""
function oplus(q1::AbstractTLArray, q2::AbstractTLArray, dimensions)
    return oplus(AbstractTLArray[q1, q2], dimensions)
end

"""
    oplus(q1::AbstractTLArray, q2::AbstractTLArray; dir=nothing, itag=nothing, plev=nothing, lock=nothing, rev=false)

Direct sum two tensors along legs selected by index metadata.

The keyword selectors are resolved on the aligned reference tensor and then
forwarded to the vector direct-sum implementation.
"""
function oplus(q1::AbstractTLArray, q2::AbstractTLArray; dir=nothing, itag=nothing,
               plev=nothing, lock=nothing, rev::Bool=false)
    return oplus(AbstractTLArray[q1, q2]; dir=dir, itag=itag, plev=plev,
                 lock=lock, rev=rev)
end

"""
    _complete_oplus_matrix(mat::AbstractMatrix, dimensions)

Validate and complete the block matrix used by matrix `oplus`.

`mat` must contain at least one defined `AbstractTLArray`. Missing, unassigned,
`nothing`, or `missing` entries are replaced by zero tensors whose spaces are
inferred from row and column constraints. `dimensions` is split into row-axis
and column-axis direct-sum leg groups. Returns the completed matrix and the two
normalized leg groups.
"""
function _complete_oplus_matrix(mat::AbstractMatrix, dimensions)
    size(mat, 1) > 0 && size(mat, 2) > 0 || throw(ArgumentError(
        "matrix oplus requires a non-empty matrix"))

    defined_positions = Tuple{Int, Int}[]
    defined_qs = AbstractTLArray[]
    for j in axes(mat, 2), i in axes(mat, 1)
        q = _oplus_matrix_entry(mat, i, j)
        q === nothing && continue
        push!(defined_positions, (i, j))
        push!(defined_qs, q)
    end
    isempty(defined_qs) && throw(ArgumentError(
        "matrix oplus requires at least one defined TLArray to infer spaces"))

    aligned_qs = _align_oplus_inputs(defined_qs)
    aligned_by_position = Dict(pos => q for (pos, q) in zip(defined_positions, aligned_qs))
    ref = _validate_oplus_common(aligned_qs)
    first_axis_dims, second_axis_dims = _normalize_oplus_matrix_dims(dimensions, length(inds(ref)))
    first_axis_dims_set = Set(first_axis_dims)
    second_axis_dims_set = Set(second_axis_dims)

    first_axis_sources = [Dict{Int, Any}() for _ in axes(mat, 1)]
    second_axis_sources = [Dict{Int, Any}() for _ in axes(mat, 2)]

    for ((i, j), q) in zip(defined_positions, aligned_qs)
        for leg in 1:length(inds(ref))
            if leg ∉ second_axis_dims_set
                if haskey(first_axis_sources[i], leg)
                    first_axis_sources[i][leg] == spaces(q)[leg] || throw(ArgumentError(
                        "sector $i has incompatible spaces on leg $leg"))
                else
                    first_axis_sources[i][leg] = copy(spaces(q)[leg])
                end
            end
            if leg ∉ first_axis_dims_set
                if haskey(second_axis_sources[j], leg)
                    second_axis_sources[j][leg] == spaces(q)[leg] || throw(ArgumentError(
                        "column $j has incompatible spaces on leg $leg"))
                else
                    second_axis_sources[j][leg] = copy(spaces(q)[leg])
                end
            end
        end
    end

    T = promote_type((_tlarray_eltype(q) for q in defined_qs)...)
    filled = Matrix{AbstractTLArray}(undef, size(mat, 1), size(mat, 2))
    for j in axes(mat, 2), i in axes(mat, 1)
        q = _oplus_matrix_entry(mat, i, j)
        if q === nothing
            inferred_spaces = _infer_zero_matrix_spaces(first_axis_sources, second_axis_sources, i, j, length(inds(ref)))
            filled[i, j] = _zero_tlarray_with_spaces(symm(ref), inds(ref), inferred_spaces; T=T)
        else
            filled[i, j] = aligned_by_position[(i, j)]
        end
    end

    return filled, first_axis_dims, second_axis_dims
end

"""
    complete_oplus_matrix(mat::AbstractMatrix, dimensions)

Validate a matrix input for `oplus`, infer zero `TLArray` objects for undefined
entries, and return the completed `Matrix{AbstractTLArray}`.

`dimensions` must be a two-entry tuple naming row-axis and column-axis
direct-sum legs. This helper performs the same validation and zero-entry
inference as matrix `oplus` but stops before assembling the final tensor.
"""
function complete_oplus_matrix(mat::AbstractMatrix, dimensions)
    filled, _, _ = _complete_oplus_matrix(mat, dimensions)
    return filled
end

"""
    oplus(mat::AbstractMatrix, dimensions) -> TLArray

Assemble a block-matrix direct sum of tensors.

`mat` is a non-empty matrix whose defined entries are `AbstractTLArray` values;
undefined entries may be unassigned, `nothing`, or `missing` and are inferred as
compatible zero tensors. `dimensions` is a two-entry tuple: the first entry
selects legs direct-summed down matrix rows, and the second selects legs
direct-summed across columns. Assembly first sums each column over the row-axis
legs and then sums the column aggregates over the column-axis legs.
"""
function oplus(mat::AbstractMatrix, dimensions)
    filled, first_axis_dims, second_axis_dims = _complete_oplus_matrix(mat, dimensions)

    col_aggregates = Vector{TLArray}(undef, size(filled, 2))
    for j in axes(filled, 2)
        col_aggregates[j] = _materialize_vector_oplus(vec(filled[:, j]), first_axis_dims)
    end

    return _materialize_vector_oplus(col_aggregates, second_axis_dims)
end
