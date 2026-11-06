"""
    fuselegs(q, legs; itag="", dir='-', plev=0, lock=0) -> (tensor, isometry)

Fuse at least two distinct logical legs of `q`. The order in `legs` defines
the fused basis order; do not sort it. The fused leg occupies the earliest
selected logical position, and the unselected legs retain their relative order.

Unsymmetric tensors are fused by permuting and reshaping their RMT payloads.
Symmetric tensors are fused by contracting with the isometry returned by
`getIdentity`. The result is `(fused_tensor, isometry)` for symmetric inputs
and `(fused_tensor, nothing)` for unsymmetric inputs.
"""
function fuselegs(q::AbstractTLArray{T, QD, N}, legs;
                  itag::AbstractString="", dir::Char='-',
                  plev::Integer=0, lock::Integer=0) where {T, QD, N}
    dir in ('+', '-') || throw(ArgumentError("dir must be '+' or '-', got '$dir'"))
    positions = _normalize_fuselegs(q, legs)
    if N == 0
        return _fuselegs_unsymmetric(q, positions, String(itag), dir, Int(plev), Int(lock))
    end

    # getIdentity keeps the selected-leg order supplied by the caller. Its
    # selected legs are paired with q's selected legs; its final leg is fused.
    isometry = getIdentity(q, positions; itag=itag, plev=Int(plev), lock=Int(lock))
    isometry_legs = ntuple(identity, Val(length(positions)))
    fused = contract(q, positions, isometry, isometry_legs; reduce_lock=false)

    free_legs = [leg for leg in 1:QD if !(leg in positions)]
    first_selected = minimum(positions)
    n_prefix = count(leg -> leg < first_selected, free_legs)
    fused_position = n_prefix + 1
    final_order = (collect(1:n_prefix)...,
                   QD - length(positions) + 1,
                   collect(n_prefix+1:QD-length(positions))...)
    fused = permutedims(fused, final_order)
    dir == '+' && (fused = legflip(fused, fused_position))
    return fused, isometry
end

function _normalize_fuselegs(q::AbstractTLArray{T, QD}, legs) where {T, QD}
    (legs isa Tuple || legs isa AbstractVector) || throw(ArgumentError(
        "legs must be an ordered tuple or vector of logical leg numbers"))
    positions = Int[]
    for leg in legs
        leg isa Integer || throw(ArgumentError("legs must contain only integers"))
        push!(positions, Int(leg))
    end
    length(positions) >= 2 || throw(ArgumentError("fuselegs requires at least two legs"))
    all(1 <= leg <= QD for leg in positions) || throw(BoundsError(q, positions))
    length(unique(positions)) == length(positions) || throw(ArgumentError(
        "fuselegs legs must be distinct, got $positions"))
    return Tuple(positions)
end

function _fuselegs_unsymmetric(q::AbstractTLArray{T, QD, 0}, legs::Tuple,
                               itag::String, dir::Char, plev::Int,
                               lock::Int) where {T, QD}
    # Convert lazy wrappers to a concrete source once; concrete TLArrays keep
    # their payloads and deferred view information untouched.
    src = q isa TLArray ? q : TLArray(q)
    free = [leg for leg in 1:QD if !(leg in legs)]
    first_selected = minimum(legs)
    n_prefix = count(leg -> leg < first_selected, free)
    axis_order = (free[1:n_prefix]..., legs..., free[n_prefix+1:end]...)
    out_rank = QD - length(legs) + 1
    fused_dim = prod(size(src, leg) for leg in legs)
    out_inds = TLIndex[inds(src)[leg] for leg in free[1:n_prefix]]
    push!(out_inds, TLIndex(itag, dir, plev, lock))
    append!(out_inds, TLIndex[inds(src)[leg] for leg in free[n_prefix+1:end]])

    QT = qlabeltype(src)
    out_spaces = Vector{Tuple{QT, Int}}[
        Tuple{QT, Int}[((), size(src, leg))] for leg in free[1:n_prefix]
    ]
    push!(out_spaces, Tuple{QT, Int}[((), fused_dim)])
    append!(out_spaces, Vector{Tuple{QT, Int}}[
        Tuple{QT, Int}[((), size(src, leg))] for leg in free[n_prefix+1:end]
    ])

    out_type = promote_type(T, typeof(stored_scale(src)))
    identity_view = !stored_conj(src) &&
                    stored_scale(src) == one(typeof(stored_scale(src))) &&
                    stored_perm(src) == _identity_phys_perm(Val(QD))
    adjacent_ordered = all(legs[i+1] == legs[i] + 1 for i in 1:length(legs)-1)
    can_share = identity_view && adjacent_ordered &&
                all(sector -> !is_sector_defined(src, sector) ||
                    (src.RMTs[sector] isa Array && eltype(src.RMTs[sector]) === out_type),
                    sector_slots(src))

    example_sector = nothing
    if can_share
        for sector in sector_slots(src)
            if is_sector_defined(src, sector)
                example_sector = sector
                break
            end
        end
    end
    RMT = if isnothing(example_sector)
        Array{out_type, out_rank}
    else
        example_rmt = src.RMTs[example_sector]
        example_dims = _fuselegs_rmt_shape(example_rmt, n_prefix, length(legs),
                                            fused_dim, QD)
        typeof(reshape(example_rmt, example_dims...))
    end
    RMTs = Vector{RMT}(undef, sector_count(src))

    for sector in sector_slots(src)
        is_sector_defined(src, sector) || continue
        if can_share
            dims = _fuselegs_rmt_shape(src.RMTs[sector], n_prefix,
                                       length(legs), fused_dim, QD)
            RMTs[sector] = reshape(src.RMTs[sector], dims...)
        else
            payload, scale = sector_rmt_permuted(src, sector, axis_order)
            out_dims = _fuselegs_rmt_shape(payload, n_prefix, length(legs),
                                           fused_dim, QD)
            dest = Array{out_type, out_rank}(undef, out_dims)
            dest .= convert(out_type, scale) .* reshape(payload, out_dims)
            RMTs[sector] = dest
        end
    end

    QTout = qlabeltype(())
    trivial = ()
    qlabels = Vector{NTuple{out_rank, QTout}}(undef, sector_count(src))
    fill!(qlabels, ntuple(_ -> trivial, Val(out_rank)))
    wmatinfo = [_empty_wmat_info(Val(0)) for _ in 1:sector_count(src)]
    spaces_tuple = Tuple(out_spaces)
    result = TLArray((), qlabels, Float64[], wmatinfo, RMTs,
                     Tuple(out_inds), spaces_tuple)
    return result, nothing
end

function _fuselegs_rmt_shape(rmt::AbstractArray, n_prefix::Int,
                             n_selected::Int, fused_dim::Int, qd::Int)
    return (ntuple(i -> size(rmt, i), Val(n_prefix))...,
            fused_dim,
            ntuple(i -> size(rmt, n_prefix + n_selected + i),
                   Val(qd - n_prefix - n_selected))...)
end

"""
    splitleg(q, leg, dims; itag="", dir=nothing, plev=0, lock=0) -> tensor

Split one leg of an unsymmetric tensor into two or more legs. `dims` is an
ordered tuple/vector of positive integers whose product equals the selected
leg dimension. At most one entry may be `-1`, in which case it is inferred.
The metadata keywords accept a scalar broadcast to all output legs or a
tuple/vector with one value per output leg. By default, `dir` inherits the
selected leg's logical direction; `itag`, `plev`, and `lock` default to
`""`, `0`, and `0` respectively.

The RMT payloads are reshaped views and may share storage with the input. The
input's `conj`, `scale`, and `perm` view state is preserved. Symmetric inputs
are not supported yet and raise `ArgumentError`.
"""
function splitleg(q::AbstractTLArray{T, QD, N}, leg::Integer, dims;
                  itag="", dir=nothing, plev=0, lock=0) where {T, QD, N}
    N == 0 || throw(ArgumentError("splitleg currently supports unsymmetric tensors only"))
    1 <= leg <= QD || throw(BoundsError(q, leg))
    split_dims = _normalize_split_dims(dims, size(q, leg))
    k = length(split_dims)

    source = q isa TLArray ? q : TLArray(q)
    itag_values = _split_leg_values(itag, k, "itag")
    dir_values = isnothing(dir) ? ntuple(_ -> inds(source)[leg].dir, Val(k)) :
                                  _split_leg_values(dir, k, "dirs")
    plev_values = _split_leg_values(plev, k, "plevs")
    lock_values = _split_leg_values(lock, k, "locks")
    all(x -> x isa AbstractString, itag_values) || throw(ArgumentError("itags values must be strings"))
    all(x -> x isa Char && (x == '+' || x == '-'), dir_values) ||
        throw(ArgumentError("dir values must be '+' or '-'"))
    all(x -> x isa Integer, plev_values) || throw(ArgumentError("plev values must be integers"))
    all(x -> x isa Integer, lock_values) || throw(ArgumentError("lock values must be integers"))

    old_perm = stored_perm(source)
    stored_leg = old_perm[leg]
    out_rank = QD + k - 1

    new_perm = Int[]
    for logical_leg in 1:QD
        old_stored = old_perm[logical_leg]
        if logical_leg == leg
            append!(new_perm, stored_leg:(stored_leg + k - 1))
        else
            push!(new_perm, old_stored < stored_leg ? old_stored : old_stored + k - 1)
        end
    end

    stored_inds_new = TLIndex[]
    for axis in 1:QD
        if axis == stored_leg
            for j in 1:k
                output_dir = dir_values[j]
                stored_dir = stored_conj(source) ? (output_dir == '+' ? '-' : '+') : output_dir
                push!(stored_inds_new, TLIndex(String(itag_values[j]), stored_dir,
                                               Int(plev_values[j]), Int(lock_values[j])))
            end
        else
            push!(stored_inds_new, stored_inds(source)[axis])
        end
    end

    stored_spaces_new = Vector{Tuple{qlabeltype(source), Int}}[]
    for axis in 1:QD
        if axis == stored_leg
            append!(stored_spaces_new,
                    [Tuple{qlabeltype(source), Int}[((), d)] for d in split_dims])
        else
            push!(stored_spaces_new, stored_spaces(source)[axis])
        end
    end

    example_sector = nothing
    for sector in sector_slots(source)
        if is_sector_defined(source, sector)
            example_sector = sector
            break
        end
    end
    RMT = if isnothing(example_sector)
        Array{T, out_rank}
    else
        old_rmt = source.RMTs[example_sector]
        example_shape = _splitleg_rmt_shape(old_rmt, stored_leg, split_dims, QD)
        typeof(reshape(old_rmt, example_shape...))
    end
    RMTs = Vector{RMT}(undef, sector_count(source))
    for sector in sector_slots(source)
        is_sector_defined(source, sector) || continue
        old_rmt = source.RMTs[sector]
        shape = _splitleg_rmt_shape(old_rmt, stored_leg, split_dims, QD)
        RMTs[sector] = reshape(old_rmt, shape...)
    end

    QT = qlabeltype(source)
    qlabels = Vector{NTuple{out_rank, QT}}(undef, sector_count(source))
    fill!(qlabels, ntuple(_ -> (), Val(out_rank)))
    perm_tuple = Tuple(new_perm)
    result = TLArray(Val(:alias_storage_view_state), symm(source), qlabels,
                     copy(source.wmatdata), copy(source.wmatinfo), RMTs,
                     copy(source.isdefined), copy(source.iszero),
                     Tuple(stored_inds_new), Tuple(stored_spaces_new),
                     stored_conj(source), stored_scale(source), perm_tuple)
    return result
end

function _splitleg_rmt_shape(rmt::AbstractArray, split_leg::Int,
                             split_dims::Tuple, qd::Int)
    shape = Int[]
    for axis in 1:qd
        if axis == split_leg
            append!(shape, split_dims)
        else
            push!(shape, size(rmt, axis))
        end
    end
    return shape
end

function _normalize_split_dims(dims, original_dim::Int)
    (dims isa Tuple || dims isa AbstractVector) || throw(ArgumentError(
        "dims must be a tuple or vector of integers"))
    values = Int[]
    for d in dims
        d isa Integer || throw(ArgumentError("dims must contain only integers"))
        push!(values, Int(d))
    end
    length(values) >= 2 || throw(ArgumentError("splitleg requires at least two output dimensions"))
    count(==(-1), values) <= 1 || throw(ArgumentError("dims may contain at most one -1"))
    all(d -> d > 0 || d == -1, values) || throw(ArgumentError(
        "dims entries must be positive integers, except for at most one -1"))
    inferred = findfirst(==(-1), values)
    if !isnothing(inferred)
        known = prod(values[i] for i in eachindex(values) if i != inferred; init=1)
        known > 0 && original_dim % known == 0 || throw(ArgumentError(
            "cannot infer a positive split dimension from $values for dimension $original_dim"))
        values[inferred] = div(original_dim, known)
        values[inferred] > 0 || throw(ArgumentError("inferred split dimensions must be positive"))
    end
    prod(values) == original_dim || throw(ArgumentError(
        "split dimensions $values have product $(prod(values)); expected $original_dim"))
    return Tuple(values)
end

function _split_leg_values(value, n::Int, name::AbstractString)
    values = if value isa Tuple || value isa AbstractVector
        length(value) == n || throw(ArgumentError(
            "$name must have one value per output leg ($n), got $(length(value))"))
        Tuple(value)
    else
        ntuple(_ -> value, Val(n))
    end
    return values
end
