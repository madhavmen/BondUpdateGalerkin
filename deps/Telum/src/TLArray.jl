using Printf
using LinearAlgebra
include("utils.jl")
include("localspaces/localspaces.jl")

"""
    ProductSymm(S1, S2, ...)

Compile-time representation of a direct product of symmetry families.  Telum
stores it in tensor type metadata; use `product_symms`, `nsymms`, or `symm` to
inspect its component symmetry types.
"""
abstract type ProductSymm{Syms<:Tuple{Vararg{Symmetry}}} <: Symmetry end

ProductSymm(syms::Type{<:Symmetry}...) = ProductSymm{Tuple{syms...}}

"""
    product_symms(ps_or_q) -> Tuple{Vararg{DataType}}

Return the ordered component symmetry types of a `ProductSymm`, symmetry tuple,
`leginfo`, or `AbstractTLArray`. The order is the q-label tuple order and must
not be changed when interpreting sector metadata.
"""
product_symms(::Type{<:ProductSymm{Syms}}) where {Syms} = Tuple(Syms.parameters)
"""
    nsymms(ps_or_q) -> Int

Return the number of symmetry components carried by a product symmetry, tuple,
leg-info value, or tensor. This is metadata-only and is independent of tensor
rank and the number of non-Abelian w-matrix slots.
"""
nsymms(::Type{<:ProductSymm{Syms}}) where {Syms} = length(Syms.parameters)

"""
    productsymm(symm) -> Type{<:ProductSymm}

Construct the compile-time product-symmetry type for tuple `symm`. Every entry
must be a `Type{<:Symmetry}`; otherwise an `ArgumentError` is thrown. Passing
an existing `ProductSymm` type returns it unchanged.
"""
function productsymm(symm::Tuple)
    all(s -> s isa Type{<:Symmetry}, symm) || throw(ArgumentError(
        "all entries of a product symmetry must be symmetry types, got $symm"))
    return ProductSymm(symm...)
end

productsymm(::Type{PS}) where {PS<:ProductSymm} = PS
product_symms(symm::Tuple) = product_symms(productsymm(symm))
nsymms(symm::Tuple) = nsymms(productsymm(symm))

@generated function nonabelian_symmetry_indices(::Type{PS}) where {PS<:ProductSymm}
    syms = product_symms(PS)
    inds = Tuple(i for i in eachindex(syms) if !isabelian(syms[i]))
    return :($inds)
end

@inline n_nonabelian_symmetries(::Type{PS}) where {PS<:ProductSymm} =
    length(nonabelian_symmetry_indices(PS))

@inline is_stored_wmat_symmetry(::Type{PS}, ::Val{n}) where {PS<:ProductSymm, n} =
    wmat_tuple_slot(PS, Val(n)) !== nothing

@generated function nonabelian_wmat_slot(::Type{PS}, ::Val{n}) where {PS<:ProductSymm, n}
    syms = product_symms(PS)
    slot = 0
    for i in eachindex(syms)
        isabelian(syms[i]) || (slot += 1)
        i == n && break
    end
    1 <= n <= length(syms) && !isabelian(syms[n]) ||
        return :(throw(ArgumentError("symmetry index $n is Abelian and has no stored w-matrix slot")))
    return :($slot)
end

function nonabelian_wmat_slot(::Type{PS}, n::Int) where {PS<:ProductSymm}
    syms = product_symms(PS)
    1 <= n <= length(syms) || throw(BoundsError(syms, n))
    slot = 0
    for i in eachindex(syms)
        isabelian(syms[i]) || (slot += 1)
        i == n && break
    end
    isabelian(syms[n]) &&
        throw(ArgumentError("symmetry index $n is Abelian and has no stored w-matrix slot"))
    return slot
end

@generated function wmat_tuple_slot(::Type{PS}, ::Val{n}) where {PS<:ProductSymm, n}
    syms = product_symms(PS)
    1 <= n <= length(syms) || return :(throw(BoundsError(product_symms(PS), $n)))
    isabelian(syms[n]) && return :(nothing)
    slot = count(i -> !isabelian(syms[i]), 1:n)
    return :($slot)
end

@generated function product_symmetry_index_from_wmat_slot(::Type{PS}, ::Val{slot}) where {PS<:ProductSymm, slot}
    syms = product_symms(PS)
    count = 0
    for i in eachindex(syms)
        isabelian(syms[i]) && continue
        count += 1
        count == slot && return :($i)
    end
    return :(throw(BoundsError(nonabelian_symmetry_indices(PS), $slot)))
end

@inline _trivial_wmat() = ones(1, 1)

const WMatInfo{M} = NTuple{M, NTuple{3, Int}}

@inline _wmat_tuple_width(::Type{PS}) where {PS<:ProductSymm} =
    n_nonabelian_symmetries(PS)

@inline _empty_wmat_info(::Val{M}) where {M} =
    ntuple(_ -> (0, 0, 0), Val(M))

@inline function _wmat_info_present(info::NTuple{3, Int})
    offset, nrow, ncol = info
    return offset != 0 || nrow != 0 || ncol != 0
end

"""
    _validate_wmat_storage(wmatdata, wmatinfo, sector_count)

Validate packed non-Abelian w-matrix storage before constructing a tensor.
`wmatdata` is the contiguous numeric arena; `wmatinfo` gives each stable
sector's offsets/dimensions; and `sector_count` is the required slot count.
Throws `ArgumentError` when offsets, dimensions, or arena bounds are
inconsistent. It performs no normalization or allocation.
"""
function _validate_wmat_storage(wmatdata::Vector{Float64},
                                wmatinfo::Vector{WMatInfo{M}},
                                nslots::Int) where {M}
    length(wmatinfo) == nslots ||
        throw(ArgumentError("w-matrix info length must equal number of sector slots"))
    ranges = Tuple{Int, Int}[]
    for sector in eachindex(wmatinfo)
        infos = wmatinfo[sector]
        for slot in 1:M
            offset, nrow, ncol = infos[slot]
            if offset == 0 && nrow == 0 && ncol == 0
                continue
            end
            offset >= 1 || throw(ArgumentError("w-matrix sector $sector slot $slot has invalid offset $offset"))
            nrow > 0 && ncol > 0 ||
                throw(ArgumentError("w-matrix sector $sector slot $slot has invalid size ($nrow, $ncol)"))
            last = offset + nrow * ncol - 1
            last <= length(wmatdata) ||
                throw(ArgumentError("w-matrix sector $sector slot $slot range exceeds wmatdata length"))
            push!(ranges, (offset, last))
        end
    end
    sort!(ranges; by=first)
    for i in 2:length(ranges)
        ranges[i][1] <= ranges[i - 1][2] &&
            throw(ArgumentError("w-matrix arena ranges must not overlap"))
    end
    return nothing
end

function _empty_wmat_storage(::Type{PS}) where {PS<:ProductSymm}
    M = _wmat_tuple_width(PS)
    return Float64[], WMatInfo{M}[]
end

@inline _empty_wmat_storage(symm::NTuple) =
    _empty_wmat_storage(productsymm(symm))

function _append_wmat_tuple!(wmatdata::Vector{Float64},
                             wmatinfo::Vector{WMatInfo{M}},
                             wmats::NTuple{M, <:AbstractMatrix}) where {M}
    info = ntuple(Val(M)) do slot
        wmat = wmats[slot]
        len = length(wmat)
        len == 0 && return (0, 0, 0)
        offset = length(wmatdata) + 1
        append!(wmatdata, vec(wmat))
        (offset, size(wmat, 1), size(wmat, 2))
    end
    push!(wmatinfo, info)
    return wmatdata, wmatinfo
end

function _store_wmat_tuple!(wmatdata::Vector{Float64},
                            wmatinfo::Vector{WMatInfo{M}},
                            sector::Int,
                            next_offset::Int,
                            wmats::NTuple{M, <:AbstractMatrix}) where {M}
    wmatinfo[sector] = ntuple(Val(M)) do slot
        wmat = wmats[slot]
        len = length(wmat)
        len == 0 && return (0, 0, 0)
        offset = next_offset
        copyto!(view(wmatdata, offset:offset + len - 1), vec(wmat))
        next_offset += len
        (offset, size(wmat, 1), size(wmat, 2))
    end
    return next_offset
end

function _unit_wmat_storage(::Type{PS}, nslots::Int) where {PS<:ProductSymm}
    M = _wmat_tuple_width(PS)
    total = M * nslots
    data = ones(Float64, total)
    info = Vector{WMatInfo{M}}(undef, nslots)
    offset = 1
    for sector in 1:nslots
        info[sector] = ntuple(Val(M)) do _
            current = offset
            offset += 1
            (current, 1, 1)
        end
    end
    return data, info
end

@inline _unit_wmat_storage(symm::NTuple, nslots::Int) =
    _unit_wmat_storage(productsymm(symm), nslots)

@inline function _wmat_from_storage(wmatdata::Vector{Float64},
                                    wmatinfo::Vector{WMatInfo{M}},
                                    sector::Int,
                                    slot::Int) where {M}
    offset, nrow, ncol = wmatinfo[sector][slot]
    if offset == 0 && nrow == 0 && ncol == 0
        throw(ArgumentError("w-matrix sector $sector slot $slot is not assigned"))
    end
    return reshape(@view(wmatdata[offset:offset + nrow * ncol - 1]), nrow, ncol)
end

@inline function _copy_wmat_to_storage!(wmatdata::Vector{Float64},
                                        wmatinfo::Vector{WMatInfo{M}},
                                        sector::Int,
                                        slot::Int,
                                        wmat::AbstractMatrix) where {M}
    offset, nrow, ncol = wmatinfo[sector][slot]
    if offset == 0 && nrow == 0 && ncol == 0
        throw(ArgumentError("w-matrix sector $sector slot $slot is not assigned"))
    end
    size(wmat) == (nrow, ncol) ||
        throw(DimensionMismatch("w-matrix sector $sector slot $slot expected size ($nrow, $ncol), got $(size(wmat))"))
    copyto!(view(wmatdata, offset:offset + nrow * ncol - 1), vec(wmat))
    return wmatdata
end

# ─── Tag helpers ──────────────────────────────────────────────────────────────
#
# Tags are stored internally as an Itag wrapper around a canonical
# comma-separated String sorted alphabetically, e.g. "bond,site,u1"
# (ITensor convention). Empty string means no tags. Individual tags contain no
# commas or whitespace.
#
# ─────────────────────────────────────────────────────────────────────────────

_itag_text(tags::AbstractString) = String(tags)

# Split tag text into sorted vector of non-empty individual tags.
_parse_itag(tags::AbstractString) = sort!(filter!(!isempty, strip.(split(_itag_text(tags), ','))))

# Canonical form: sorted, comma-joined, no spaces.
_normalize_itag_text(tags::AbstractString) = join(_parse_itag(tags), ',')

"""
    Itag(tags)

Immutable normalized tag string used by `TLIndex`. `tags` is parsed as
comma-separated tag names, deduplicated, sorted, and stored without whitespace;
an empty string represents no tags. Construct it directly only when index-tag
normalization is needed outside the higher-level tag operations.

# Fields

- `value`: canonical comma-separated tag text; this is the sole stored field.
"""
struct Itag <: AbstractString
    value::String

    Itag(tags::AbstractString) = new(_normalize_itag_text(tags))
end

Itag(tags::Itag) = tags
_itag_text(tags::Itag) = tags.value
_normalize_itag(tags::AbstractString) = Itag(tags)

Base.String(tags::Itag) = tags.value
Base.convert(::Type{String}, tags::Itag) = tags.value
Base.ncodeunits(tags::Itag) = ncodeunits(tags.value)
Base.codeunit(::Type{Itag}) = codeunit(String)
Base.codeunit(tags::Itag, i::Int) = codeunit(tags.value, i)
Base.isvalid(tags::Itag, i::Int) = isvalid(tags.value, i)
Base.iterate(tags::Itag) = iterate(tags.value)
Base.iterate(tags::Itag, i::Int) = iterate(tags.value, i)
Base.length(tags::Itag) = length(tags.value)
Base.lastindex(tags::Itag) = lastindex(tags.value)
Base.thisind(tags::Itag, i::Int) = thisind(tags.value, i)
Base.nextind(tags::Itag, i::Int) = nextind(tags.value, i)
Base.nextind(tags::Itag, i::Int, n::Int) = nextind(tags.value, i, n)
Base.prevind(tags::Itag, i::Int) = prevind(tags.value, i)
Base.prevind(tags::Itag, i::Int, n::Int) = prevind(tags.value, i, n)
Base.show(io::IO, tags::Itag) = print(io, tags.value)
Base.isempty(tags::Itag) = isempty(tags.value)
Base.hash(tags::Itag, h::UInt) = hash(tags.value, h)
Base.:(==)(a::Itag, b::Itag) = a.value == b.value
Base.:(==)(a::Itag, b::AbstractString) = a.value == _normalize_itag_text(b)
Base.:(==)(a::AbstractString, b::Itag) = _normalize_itag_text(a) == b.value
Base.isequal(a::Itag, b::Itag) = (a == b)
Base.isequal(a::Itag, b::AbstractString) = (a == b)
Base.isequal(a::AbstractString, b::Itag) = (a == b)

# True iff all comma-separated tags in `query` are present in `base`.
function _has_itag(base::AbstractString, query::AbstractString)
    bset = Set(_parse_itag(base))
    return all(t -> t ∈ bset, _parse_itag(query))
end

_matches_itag_selector(base::AbstractString, query::AbstractString) = _has_itag(base, query)
_matches_itag_selector(base::AbstractString, queries::Tuple{Vararg{AbstractString}}) =
    any(query -> _has_itag(base, query), queries)
_matches_itag_selector(base::AbstractString, queries::AbstractVector{<:AbstractString}) =
    any(query -> _has_itag(base, query), queries)

# Add tags from `newtags` to `base`; result is sorted and deduplicated.
_add_itag(base::AbstractString, newtags::AbstractString) =
    Itag(join(sort!(unique!(vcat(_parse_itag(base), _parse_itag(newtags)))), ','))

# Remove every tag listed in `rmtags` from `base`.
function _remove_itag(base::AbstractString, rmtags::AbstractString)
    rm = Set(_parse_itag(rmtags))
    return Itag(join(filter(t -> t ∉ rm, _parse_itag(base)), ','))
end

# Remove tags from `base` using one or more ITensor-style tag queries.
# Each query is applied independently, and its tags are removed only if all
# tags in that query are present in `base`.
function _remove_itag(base::AbstractString,
                      rmtags::Union{Tuple{Vararg{AbstractString}},
                                    AbstractVector{<:AbstractString}})
    base_tags = _parse_itag(base)
    base_set = Set(base_tags)
    rm = Set{String}()
    for query in rmtags
        parsed = _parse_itag(query)
        all(tag -> tag ∈ base_set, parsed) || continue
        union!(rm, parsed)
    end
    return Itag(join(filter(tag -> tag ∉ rm, base_tags), ','))
end

function _replace_itag(base::AbstractString, replacements)
    base_tags = _parse_itag(base)
    base_set = Set(base_tags)
    replaced = Set{String}()
    new_tags = String[]

    for (from, to) in replacements
        from_tags = _parse_itag(from)
        all(tag -> tag ∈ base_set, from_tags) || continue
        union!(replaced, from_tags)
        append!(new_tags, _parse_itag(to))
    end

    append!(new_tags, (tag for tag in base_tags if tag ∉ replaced))
    return Itag(join(sort!(unique!(new_tags)), ','))
end

"""
    TLIndex(itag=""; dir=1, plev=0, lock=false, dual=false)

Metadata for one logical tensor leg. `itag` is a tag string or `Itag`; `dir`
is the leg direction (`+1` incoming or `-1` outgoing in Telum conventions);
`plev` is the nonnegative prime level; `lock` controls automatic matching; and
`dual` records an explicit dual space. Index-transforming operations return
new metadata while preserving sector storage when possible.
"""
struct TLIndex
    # Tags associated with the leg, similar to ITensor
    itags::Itag
    # Direction of the leg, '+' for incoming, '-' for outgoing
    dir::Char
    # Prime level similar to ITensor
    plev::Int
    # Lock level. Cannot cntracted if lock > 0. 
    # Decrease lock level by 1 after contraction.
    lock::Int
    # If true, this leg represents the dual space.
    dual::Bool

    TLIndex(itags::AbstractString, dir::Char, plev::Int=0, lock::Int=0, dual::Bool=false) = new(_normalize_itag(itags), dir, plev, lock, dual)
end

TLIndex(dir::Char, plev::Int=0, lock::Int=0) = TLIndex("", dir, plev, lock)

# Two TLIndex objects are equal if they share the same itags, dir, plev, and dual
# (lock is intentionally ignored — it is a transient contraction counter).
Base.:(==)(a::TLIndex, b::TLIndex) = a.itags == b.itags && a.dir == b.dir && a.plev == b.plev && a.dual == b.dual
Base.isequal(a::TLIndex, b::TLIndex) = (a == b)
Base.hash(a::TLIndex, h::UInt) = hash((a.itags, a.dir, a.plev, a.dual), h)

to_incoming(idx::TLIndex) = TLIndex(idx.itags, '+', idx.plev, idx.lock, idx.dual)
change_dir(idx::TLIndex)  = TLIndex(idx.itags, idx.dir == '+' ? '-' : '+', idx.plev, idx.lock, idx.dual)
dual(idx::TLIndex) = TLIndex(idx.itags, idx.dir, idx.plev, idx.lock, true)
change_dual(idx::TLIndex) = TLIndex(idx.itags, idx.dir, idx.plev, idx.lock, !idx.dual)

# Format a scalar RMT value as a string with consistent width:
# integers print without decimal point; floats use %#.7g which always
# shows 7 significant digits including trailing zeros (e.g. 3.46410 not 3.4641).
function _fmt_scalar_str(v::Real)
    return @sprintf("%#.7g", v)
end

function _fmt_scalar_str(v::Complex)
    re = _fmt_scalar_str(real(v))
    im = _fmt_scalar_str(abs(imag(v)))
    sign = signbit(imag(v)) ? " - " : " + "
    return string(re, sign, im, "im")
end

"""
    _localspace_cgt_fields(data, symm, spaces)

Convert decomposed local-operator blocks into the `qlabels`, packed w-matrix
arena, w-matrix descriptors, and RMT slots required by `TLArray`. `data`
contains `(sector_qlabels, reduced_block)` pairs; `symm` fixes q-label meaning;
and `spaces` fixes logical leg dimensions. The helper establishes canonical
sector metadata for local-space construction and returns a named field bundle.
"""
function _localspace_cgt_fields(data::Vector{Tuple{NTuple{QD, NTuple{N, Tuple{Vararg{Int}}}}, Array{T, RD}}},
                                symm::NTuple{N, Any},
                                spaces::Tuple{Vararg{AbstractVector, QD}}) where {T, QD, N, RD}

    @assert RD == QD + N; @assert QD == 2 || QD == 3
    QT = qlabeltype(symm)
    qlabels = Vector{NTuple{QD, QT}}(undef, length(data))
    PS = productsymm(symm)
    wmatdata, wmatinfo = _empty_wmat_storage(PS)
    nonabelian_indices = nonabelian_symmetry_indices(PS)
    RMTs = Array{T, RD}[]

    for (sector_index, (sector_qlabels, block0)) in pairs(data)
        block = block0
        sector_wmats = Vector{Matrix{Float64}}(undef, N)
        for i in 1:N
            wmat, block, _ = _localspace_wmat(block, QD + i)
            sector_wmats[i] = wmat
        end
        RMT = block
        qlabels[sector_index] = sector_qlabels
        cgt_metadata = ntuple(N) do i
            qforsymm = Tuple(sector_qlabels[j][i] for j in 1:QD)
            if QD == 2
                cgp = (1, 2); legdir = (1, 1)
            elseif QD == 3
                cgp = qforsymm[2] <= qforsymm[3] ? (1, 2, 3) : (1, 3, 2); legdir = (1, 2)
            else error("Invalid TLArray tensor dimension for getLocalSpace: $QD") end

            if QD == 3 && qforsymm[2] > qforsymm[3]
                qforsymm = (qforsymm[1], qforsymm[3], qforsymm[2])
            end
            (qforsymm, cgp, legdir)
        end

        for i in 1:N
            if length(sector_wmats[i]) == 1 && sector_wmats[i][1] < 0
                sector_wmats[i][:] .*= -1
                RMT[:] .*= -one(T)
            end
        end

        _append_wmat_tuple!(wmatdata, wmatinfo,
                            ntuple(slot -> sector_wmats[nonabelian_indices[slot]],
                                   Val(length(nonabelian_indices))))
        push!(RMTs, RMT)
    end

    return (qlabels = qlabels, wmatdata = wmatdata, wmatinfo = wmatinfo, RMTs = RMTs)
end

# ─── TLArray invariant checkers ─────────────────────────────────────────────

# No two TLIndex objects with non-empty itags in the inds tuple
# may be equal (as determined by ==, which compares itags, dir, plev, dual).
function _check_unique_inds(inds::NTuple{QD, TLIndex}) where QD
    tagged = [idx for idx in inds if !isempty(idx.itags)]
    for i in 1:length(tagged), j in i+1:length(tagged)
        @assert tagged[i] != tagged[j] begin
            "Duplicate TLIndex with non-empty itag in TLArray.inds: $(tagged[i])"
        end
    end
end

@inline _rmt_iszero(rmt::AbstractArray) = iszero(sum(abs2, rmt))
@inline _rmt_iszero(rmt::DiagRMT) = iszero(sum(abs2, rmt.diag))

@inline _check_tlarray_rmt_storage(rmt::AbstractArray, ::Val{QD}) where {QD} = nothing

function _check_tlarray_rmt_storage(rmt::DiagRMT, ::Val{QD}) where {QD}
    rmt.axis[2] == 0 && throw(ArgumentError(
        "axis[2] == 0 DiagRMT is only valid for contraction temporaries, not TLArray sector storage"))
    rmt.axis[2] <= QD || throw(ArgumentError(
        "DiagRMT sector axes must refer to TLArray tensor legs, got axis=$(rmt.axis) for tensor rank $QD"))
    return nothing
end

# T: type of element in the RMT array, can be Float64, ComplexF64, etc.
# QD: The rank of tensor (# of legs), N: The number of symmetries
# RD: The rank of RMT array, which is equal to QD + N
# QT: The qlabel type for one leg sector, inferred from the symmetries
"""
    AbstractTLArray

Abstract symmetry-aware tensor array.  Logical legs are described by `TLIndex`
and q-label spaces; concrete and lazy subtypes share sector metadata accessors.
`T` is the scalar type, `QD` rank, `N` symmetry count, and `RD` reduced-matrix
rank. Lazy subtypes preserve sector-slot numbering until materialization and
must expose the common sector accessors rather than inventing array semantics.
"""
abstract type AbstractTLArray{T, QD, N, RD, QT, PS<:ProductSymm, M, RMT<:AbstractArray{<:Any, RD}} <: AbstractArray{T, QD} end

"""
    TLArray

Concrete symmetry-adapted tensor whose Clebsch-Gordan structure and reduced
matrix-tensor (RMT) blocks are stored by sector. `qlabels`, w-matrices, index
metadata, and spaces define sector structure; `RMTs` hold dense or `DiagRMT`
payloads. Defined sectors share one concrete RMT type; known zero sectors
normally have no payload. Use public constructors/operations rather than
mutating fields, so the sector-state and logical-leg invariants are retained.

# Fields

- `qlabels`: q-label tuple for every stable sector slot in stored-leg order.
- `wmatdata`: packed numerical storage for non-Abelian CGT w-matrices.
- `wmatinfo`: per-sector offsets/shapes into `wmatdata`.
- `RMTs`: per-sector reduced-matrix payload slots of one concrete RMT type.
- `isdefined`: whether an RMT slot has been assigned a payload.
- `iszero`: whether a sector is known zero; eager tensors satisfy undefined ⇒ zero.
- `inds`: stored-leg `TLIndex` metadata.
- `spaces`: stored-leg q-label/multiplicity lists.
- `conj`, `scale`, `perm`: deferred logical view state; `perm` maps logical to stored legs.
"""
struct TLArray{T, QD, N, RD, QT, PS<:ProductSymm, M, RMT<:AbstractArray{<:Any, RD}} <:
       AbstractTLArray{T, QD, N, RD, QT, PS, M, RMT}
    qlabels::Vector{NTuple{QD, QT}}
    wmatdata::Vector{Float64}
    wmatinfo::Vector{WMatInfo{M}}
    RMTs::Vector{RMT}
    isdefined::BitVector
    iszero::BitVector
    inds::NTuple{QD, TLIndex}
    # Space list for each leg: vector of (qlabels, RMT_dim) pairs
    # Similar to leginfo.splist but precomputed for all legs
    spaces::NTuple{QD, Vector{Tuple{QT, Int}}}
    # Embedded view state maps logical tensor legs to stored RMT axes.
    conj::Bool
    scale::T
    perm::NTuple{QD, Int}

    function TLArray(symm::NTuple{N, Any},
        qlabels::Vector{QL},
        wmatdata::Vector{Float64},
        wmatinfo::Vector{WMatInfo{M}},
        RMTs::Vector{RMT},
        inds::NTuple{QD, TLIndex},
        spaces::Tuple{Vararg{AbstractVector, QD}}) where {
            N, QD, QL, M, T, RD, RMT<:AbstractArray{T, RD}}

        RD == QD + N ||
            throw(ArgumentError("RMT rank must equal tensor rank plus number of symmetries"))

        PS = productsymm(symm)
        QT = qlabeltype(symm)
        typed_qlabels = Vector{NTuple{QD, QT}}(qlabels)
        M == n_nonabelian_symmetries(PS) ||
            throw(ArgumentError("w-matrix tuple width must equal the number of non-Abelian symmetries"))
        length(typed_qlabels) == length(RMTs) ||
            throw(ArgumentError("qlabels length must equal number of sector slots"))
        _validate_wmat_storage(wmatdata, wmatinfo, length(RMTs))

        sector_isdefined = falses(length(RMTs))
        sector_iszero = falses(length(RMTs))
        for sector_index in eachindex(RMTs)
            rmt_defined = isassigned(RMTs, sector_index)
            if rmt_defined
                _check_tlarray_rmt_storage(RMTs[sector_index], Val(QD))
                sector_isdefined[sector_index] = true
                sector_iszero[sector_index] = _rmt_iszero(RMTs[sector_index])
            else
                sector_isdefined[sector_index] = false
                sector_iszero[sector_index] = true
            end
        end

        typed_spaces = ntuple(l -> convert(Vector{Tuple{QT, Int}}, spaces[l]), QD)
        q = new{T, QD, N, RD, QT, PS, M, RMT}(
            typed_qlabels, wmatdata, wmatinfo, RMTs, sector_isdefined, sector_iszero,
            inds, typed_spaces, false, one(T), ntuple(identity, Val(QD)))
        _check_unique_inds(getfield(q, :inds))
        return q
    end

    function TLArray(::Val{:alias_storage},
        symm::NTuple{N, Any},
        qlabels::Vector{NTuple{QD, QT}},
        wmatdata::Vector{Float64},
        wmatinfo::Vector{WMatInfo{M}},
        RMTs::Vector{RMT},
        isdefined::BitVector,
        iszero::BitVector,
        inds::NTuple{QD, TLIndex},
        spaces::NTuple{QD, Vector{Tuple{QT, Int}}}) where {
            N, QD, QT, M, T, RD, RMT<:AbstractArray{T, RD}}

        RD == QD + N ||
            throw(ArgumentError("RMT rank must equal tensor rank plus number of symmetries"))

        PS = productsymm(symm)
        QT == qlabeltype(symm) ||
            throw(ArgumentError("qlabel type does not match symmetry tuple"))
        M == n_nonabelian_symmetries(PS) ||
            throw(ArgumentError("w-matrix tuple width must equal the number of non-Abelian symmetries"))
        length(qlabels) == length(RMTs) ||
            throw(ArgumentError("qlabels length must equal number of sector slots"))
        length(isdefined) == length(RMTs) && length(iszero) == length(RMTs) ||
            throw(ArgumentError("sector-state lengths must equal number of sector slots"))
        _validate_wmat_storage(wmatdata, wmatinfo, length(RMTs))

        for sector_index in eachindex(RMTs)
            if isdefined[sector_index]
                isassigned(RMTs, sector_index) ||
                    throw(ArgumentError("defined sector $sector_index has no RMT payload"))
                _check_tlarray_rmt_storage(RMTs[sector_index], Val(QD))
            elseif !iszero[sector_index]
                throw(ArgumentError("nonzero sector $sector_index must be defined before aliasing"))
            end
        end

        q = new{T, QD, N, RD, QT, PS, M, RMT}(
            qlabels, wmatdata, wmatinfo, RMTs, isdefined, iszero, inds, spaces,
            false, one(T), ntuple(identity, Val(QD)))
        _check_unique_inds(getfield(q, :inds))
        return q
    end

    function TLArray(::Val{:alias_storage_view_state},
        symm::NTuple{N, Any},
        qlabels::Vector{NTuple{QD, QT}},
        wmatdata::Vector{Float64},
        wmatinfo::Vector{WMatInfo{M}},
        RMTs::Vector{RMT},
        isdefined::BitVector,
        iszero::BitVector,
        inds::NTuple{QD, TLIndex},
        spaces::NTuple{QD, Vector{Tuple{QT, Int}}},
        conj_flag::Bool,
        scale::T,
        perm::NTuple{QD, Int}) where {
            N, QD, QT, M, T, RD, RMT<:AbstractArray{<:Any, RD}}

        RD == QD + N ||
            throw(ArgumentError("RMT rank must equal tensor rank plus number of symmetries"))
        _is_valid_perm(perm) || throw(ArgumentError("perm must be a valid permutation of 1:$QD"))

        PS = productsymm(symm)
        QT == qlabeltype(symm) ||
            throw(ArgumentError("qlabel type does not match symmetry tuple"))
        M == n_nonabelian_symmetries(PS) ||
            throw(ArgumentError("w-matrix tuple width must equal the number of non-Abelian symmetries"))
        length(qlabels) == length(RMTs) ||
            throw(ArgumentError("qlabels length must equal number of sector slots"))
        length(isdefined) == length(RMTs) && length(iszero) == length(RMTs) ||
            throw(ArgumentError("sector-state lengths must equal number of sector slots"))
        _validate_wmat_storage(wmatdata, wmatinfo, length(RMTs))

        for sector_index in eachindex(RMTs)
            if isdefined[sector_index]
                isassigned(RMTs, sector_index) ||
                    throw(ArgumentError("defined sector $sector_index has no RMT payload"))
                _check_tlarray_rmt_storage(RMTs[sector_index], Val(QD))
            elseif !iszero[sector_index]
                throw(ArgumentError("nonzero sector $sector_index must be defined before aliasing"))
            end
        end

        q = new{T, QD, N, RD, QT, PS, M, RMT}(
            qlabels, wmatdata, wmatinfo, RMTs, isdefined, iszero, inds, spaces,
            conj_flag, scale, perm)
        _check_unique_inds(getfield(q, :inds))
        return q
    end
end

"""
    TLArrayContraction <: AbstractTLArray

Lazy contraction result. It stores source tensors, precomputed sector work
items/factors, output metadata, and the same deferred conjugation/scale/
permutation state as concrete tensors. `isdefined` and `iszero` are indexed by
the final result's stable sector slots; evaluating one slot must never reorder
or compact the others.

# Fields

- `qlabels`, `wmatdata`, `wmatinfo`, `RMTs`, `isdefined`, `iszero`, `inds`, `spaces`, `conj`, `scale`, `perm`: result metadata and deferred view state, with the same meanings as `TLArray`.
- `arr1`, `arr2`: source tensors retained behind abstract fields to avoid specialization on every input type.
- `work_items`: source-sector/result-sector/factor references describing independent contraction contributions.
- `factors`: precomputed symmetry contraction factors indexed by work item.
- `perm1`, `perm2`: source RMT-axis layouts used while preparing operands.
- `rmt_sizes`: result RMT dimensions per sector slot.
- `lock`: serializes lazy sector materialization and metadata updates.
"""
struct TLArrayContraction{T, QD, N, RD, QT, PS<:ProductSymm, M, RMT<:AbstractArray{<:Any, RD}} <:
       AbstractTLArray{T, QD, N, RD, QT, PS, M, RMT}
    qlabels::Vector{NTuple{QD, QT}}
    wmatdata::Vector{Float64}
    wmatinfo::Vector{WMatInfo{M}}
    RMTs::Vector{RMT}
    isdefined::BitVector
    iszero::BitVector
    inds::NTuple{QD, TLIndex}
    spaces::NTuple{QD, Vector{Tuple{QT, Int}}}
    conj::Bool
    scale::T
    perm::NTuple{QD, Int}

    arr1::AbstractTLArray
    arr2::AbstractTLArray
    work_items::Vector{NTuple{4, Int}}
    factors::Vector{Vector{Array{Float64, 3}}}
    perm1::Vector{Int}
    perm2::Vector{Int}
    rmt_sizes::Vector{NTuple{RD, Int}}
    lock::ReentrantLock
end

const GetSubSelector = Union{Colon, Vector{Int}}

"""
    SubTLArray <: AbstractTLArray

Lazy sector/element sub-selection of a source tensor. `source_sectors` maps
each result slot back to its unchanged source slot; saved selection metadata
identifies affected legs. It owns cheap index/space/view metadata but preserves
source RMT storage until a payload is requested.

# Fields

- `qlabels`, `wmatdata`, `wmatinfo`, `RMTs`, `isdefined`, `iszero`, `inds`, `spaces`, `conj`, `scale`, `perm`: result metadata/view state.
- `arr`: source tensor whose RMT payloads remain authoritative.
- `source_sectors`: result-slot to source-slot mapping.
- `saved_indices`: per-sector RMT-axis selectors (`:` or selected positions).
- `affected_legs`: logical legs on which selection changes payload shape.
- `rmt_sizes`: selected RMT shape for each result slot.
- `lock`: guards lazy source/result sector evaluation.
"""
struct SubTLArray{T, QD, N, RD, QT, PS<:ProductSymm, M, RMT<:AbstractArray{<:Any, RD}} <:
       AbstractTLArray{T, QD, N, RD, QT, PS, M, RMT}
    qlabels::Vector{NTuple{QD, QT}}
    wmatdata::Vector{Float64}
    wmatinfo::Vector{WMatInfo{M}}
    RMTs::Vector{RMT}
    isdefined::BitVector
    iszero::BitVector
    inds::NTuple{QD, TLIndex}
    spaces::NTuple{QD, Vector{Tuple{QT, Int}}}
    conj::Bool
    scale::T
    perm::NTuple{QD, Int}

    arr::AbstractTLArray{T, QD, N, RD, QT, PS, M}
    source_sectors::Vector{Int}
    saved_indices::Vector{NTuple{RD, GetSubSelector}}
    affected_legs::Vector{Int}
    rmt_sizes::Vector{NTuple{RD, Int}}
    lock::ReentrantLock
end

"""
Lazy structural wrapper that inserts singleton legs. Its leg maps translate
logical result legs to source legs while all source sector slots and RMT
payloads are retained.

# Fields

- `qlabels`, `inds`, `spaces`: result structural metadata.
- `conj`, `scale`, `perm`: deferred logical view state.
- `arr`: source tensor retaining all sector/w-matrix/RMT storage.
- `inserted_legs`: sorted result-leg positions created as singleton legs.
- `source_to_result_legs`, `result_to_source_legs`: inverse maps between noninserted result legs and source legs.
"""
struct AddSingletonTLArray{T, QD, N, RD, QT, PS<:ProductSymm, M, RMT<:AbstractArray{<:Any, RD}} <:
       AbstractTLArray{T, QD, N, RD, QT, PS, M, RMT}
    qlabels::Vector{NTuple{QD, QT}}
    inds::NTuple{QD, TLIndex}
    spaces::NTuple{QD, Vector{Tuple{QT, Int}}}
    conj::Bool
    scale::T
    perm::NTuple{QD, Int}

    arr::AbstractTLArray
    inserted_legs::Vector{Int}
    source_to_result_legs::Vector{Int}
    result_to_source_legs::Vector{Int}
end

"""
Lazy structural wrapper that removes singleton legs. Its leg maps retain source
storage and sector slots; only logical index/space metadata changes.

# Fields

- `qlabels`, `inds`, `spaces`: result structural metadata.
- `conj`, `scale`, `perm`: deferred logical view state.
- `arr`: source tensor retaining all sector/w-matrix/RMT storage.
- `deleted_legs`: source-leg positions removed from the logical result.
- `source_to_result_legs`, `result_to_source_legs`: inverse maps between retained source legs and result legs.
"""
struct DeleteSingletonTLArray{T, QD, N, RD, QT, PS<:ProductSymm, M, RMT<:AbstractArray{<:Any, RD}} <:
       AbstractTLArray{T, QD, N, RD, QT, PS, M, RMT}
    qlabels::Vector{NTuple{QD, QT}}
    inds::NTuple{QD, TLIndex}
    spaces::NTuple{QD, Vector{Tuple{QT, Int}}}
    conj::Bool
    scale::T
    perm::NTuple{QD, Int}

    arr::AbstractTLArray
    deleted_legs::Vector{Int}
    source_to_result_legs::Vector{Int}
    result_to_source_legs::Vector{Int}
end

_identity_view_state(::Type{T}, ::Val{QD}) where {T, QD} =
    (false, one(T), ntuple(identity, Val(QD)))

function TLArrayContraction{T, QD, N, RD, QT, PS, M, RMT}(
    qlabels, wmatdata, wmatinfo, RMTs, isdefined, iszero, inds, spaces,
    arr1, arr2, work_items, factors, perm1, perm2, rmt_sizes, lock) where {T, QD, N, RD, QT, PS, M, RMT}
    return TLArrayContraction{T, QD, N, RD, QT, PS, M, RMT}(
        qlabels, wmatdata, wmatinfo, RMTs, isdefined, iszero, inds, spaces,
        _identity_view_state(T, Val(QD))..., arr1, arr2, work_items, factors,
        perm1, perm2, rmt_sizes, lock)
end

function SubTLArray{T, QD, N, RD, QT, PS, M, RMT}(
    qlabels, wmatdata, wmatinfo, RMTs, isdefined, iszero, inds, spaces,
    arr, source_sectors, saved_indices, affected_legs, rmt_sizes, lock) where {T, QD, N, RD, QT, PS, M, RMT}
    return SubTLArray{T, QD, N, RD, QT, PS, M, RMT}(
        qlabels, wmatdata, wmatinfo, RMTs, isdefined, iszero, inds, spaces,
        _identity_view_state(T, Val(QD))..., arr, source_sectors,
        saved_indices, affected_legs, rmt_sizes, lock)
end

function AddSingletonTLArray{T, QD, N, RD, QT, PS, M, RMT}(
    qlabels, inds, spaces, arr, inserted_legs, source_to_result_legs,
    result_to_source_legs) where {T, QD, N, RD, QT, PS, M, RMT}
    return AddSingletonTLArray{T, QD, N, RD, QT, PS, M, RMT}(
        qlabels, inds, spaces, _identity_view_state(T, Val(QD))...,
        arr, inserted_legs, source_to_result_legs, result_to_source_legs)
end

function DeleteSingletonTLArray{T, QD, N, RD, QT, PS, M, RMT}(
    qlabels, inds, spaces, arr, deleted_legs, source_to_result_legs,
    result_to_source_legs) where {T, QD, N, RD, QT, PS, M, RMT}
    return DeleteSingletonTLArray{T, QD, N, RD, QT, PS, M, RMT}(
        qlabels, inds, spaces, _identity_view_state(T, Val(QD))...,
        arr, deleted_legs, source_to_result_legs, result_to_source_legs)
end

function _qlabel_vector(qlabels::AbstractMatrix{QT}, ::Val{QD}) where {QT, QD}
    size(qlabels, 1) == QD ||
        throw(ArgumentError("qlabels row count must equal number of TLArray legs"))
    return [ntuple(leg -> qlabels[leg, sector], Val(QD)) for sector in axes(qlabels, 2)]
end

function TLArray(symm::NTuple{N, Any},
                 qlabels::AbstractMatrix{QT},
                 wmatdata::Vector{Float64},
                 wmatinfo::Vector{WMatInfo{M}},
                 RMTs::Vector{RMT},
                 inds::NTuple{QD, TLIndex},
                 spaces::Tuple{Vararg{AbstractVector, QD}}) where {
        N, QD, QT, M, T, RD, RMT<:AbstractArray{T, RD}}
    return TLArray(symm, _qlabel_vector(qlabels, Val(QD)), wmatdata, wmatinfo, RMTs, inds, spaces)
end

function _normalize_wmats!(q::TLArray{T, QD, N}) where {T, QD, N}
    if QD == 0
        for sector_index in sector_slots(q)
            q.iszero[sector_index] && continue
            for i in 1:N
                isabelian(symm(q)[i]) && continue
                wmat = sector_wmat(q, sector_index, i)
                @assert size(wmat) == (1, 1) "0D TLArray must have 1x1 w-matrices"
                w_val = wmat[1]
                if w_val != 1.0
                    wmat[:] .= 1.0
                    q.RMTs[sector_index][:] .*= w_val
                end
            end
        end
    end
    return q
end

productsymm(::AbstractTLArray{T, QD, N, RD, QT, PS, M, RMT}) where {T, QD, N, RD, QT, PS, M, RMT} = PS
product_symms(q::AbstractTLArray) = product_symms(productsymm(q))
"""Return the ordered component symmetry tuple carried by tensor `q`; its positions match every q-label tuple."""
@inline symm(::AbstractTLArray{T, QD, N, RD, QT, PS, M, RMT}) where {T, QD, N, RD, QT, PS, M, RMT} =
    product_symms(PS)
nsymms(q::AbstractTLArray) = nsymms(productsymm(q))

Base.propertynames(q::AbstractTLArray, private::Bool=false) = fieldnames(typeof(q))

@inline stored_inds(q::AbstractTLArray) = getfield(q, :inds)
@inline stored_spaces(q::AbstractTLArray) = getfield(q, :spaces)
@inline stored_qlabels(q::AbstractTLArray) = getfield(q, :qlabels)
@inline stored_conj(q::AbstractTLArray) = getfield(q, :conj)
@inline stored_scale(q::AbstractTLArray) = getfield(q, :scale)
@inline stored_perm(q::AbstractTLArray) = getfield(q, :perm)
@inline stored_wmat_storage(q::AbstractTLArray) = (getfield(q, :wmatdata), getfield(q, :wmatinfo))
@inline stored_sector_qlabel(q::AbstractTLArray, sector::Int, leg::Int) =
    stored_qlabels(q)[sector][leg]
@inline stored_sector_qlabel(::Type{QT}, q::AbstractTLArray, sector::Int, leg::Int) where {QT} =
    stored_qlabels(q)[sector][leg]::QT

@inline function _logical_sector_qlabel_tuple(q::AbstractTLArray{T, QD}, sector::Int) where {T, QD}
    return ntuple(leg -> sector_qlabel(q, sector, leg), Val(QD))
end

"""Return all sector q-label tuples in logical-leg order. For an identity permutation it reuses stored metadata; otherwise it rebuilds tuples by logical leg while retaining original sector-slot order."""
function logical_qlabels(q::AbstractTLArray)
    if stored_perm(q) == _identity_phys_perm(Val(ndims(q)))
        return stored_qlabels(q)
    end
    return [_logical_sector_qlabel_tuple(q, sector) for sector in sector_slots(q)]
end

@inline _logical_leg(q::AbstractTLArray, leg::Int) = stored_perm(q)[leg]

@inline function _logical_index(q::AbstractTLArray, leg::Int)
    idx = stored_inds(q)[_logical_leg(q, leg)]
    return stored_conj(q) ? change_dir(idx) : idx
end

@inline Base.eltype(::Type{<:AbstractTLArray{T}}) where {T} = T
@inline Base.ndims(::Type{<:AbstractTLArray{T, QD}}) where {T, QD} = QD
@inline Base.ndims(q::AbstractTLArray{T, QD}) where {T, QD} = QD
@inline Base.size(q::AbstractTLArray{T, QD}) where {T, QD} =
    ntuple(l -> sum(last, spaces(q)[l]; init=0), Val(QD))
@inline Base.size(q::AbstractTLArray, d::Integer) = d <= ndims(q) ? size(q)[d] : 1
@inline Base.length(q::AbstractTLArray) = sector_count(q)
function Base.getproperty(q::AbstractTLArray, name::Symbol)
    return getfield(q, name)
end
"""Return `NTuple{ndims(q),TLIndex}` in logical-leg order, applying deferred permutation and conjugation direction state."""
@inline inds(q::TLArray{T, QD}) where {T, QD} =
    ntuple(l -> _logical_index(q, l), Val(QD))
"""Return one `(q-label, multiplicity)` space list per logical tensor leg, applying deferred permutation but never reordering a list's sectors."""
@inline spaces(q::TLArray{T, QD}) where {T, QD} =
    ntuple(l -> stored_spaces(q)[_logical_leg(q, l)], Val(QD))
"""Return the stable count of sector storage slots, including known-zero and unevaluated lazy slots."""
@inline sector_count(q::TLArray) = length(stored_qlabels(q))
"""Return an allocation-free iterable over stable sector-slot indices; use this rather than `1:nsectors(q)` in sector loops."""
@inline sector_slots(q::TLArray) = eachindex(stored_qlabels(q))
"""Return the number of nonzero sectors. This scans zero flags, so use `sector_slots` plus `is_sector_zero` in performance-sensitive loops."""
@inline nsectors(q::TLArray) = count(!, q.iszero)
"""Return whether sector `sector` has a defined RMT payload. For eager tensors undefined implies known zero; lazy tensors may be unevaluated instead."""
@inline is_sector_defined(q::TLArray, sector::Int) = q.isdefined[sector]
"""Return whether sector `sector` is known to have zero value, independently of whether lazy payload work has run."""
@inline is_sector_zero(q::TLArray, sector::Int) = q.iszero[sector]
"""Return whether sector `sector` is not known zero and therefore has potential payload work or data."""
@inline is_sector_active(q::TLArray, sector::Int) = !q.iszero[sector]
"""Return the q-label of logical `leg` in stable slot `sector`, applying deferred logical-leg permutation without renumbering sectors."""
@inline sector_qlabel(q::TLArray, sector::Int, leg::Int) =
    stored_sector_qlabel(q, sector, _logical_leg(q, leg))
@inline sector_qlabel(::Type{QT}, q::TLArray, sector::Int, leg::Int) where {QT} =
    stored_sector_qlabel(QT, q, sector, _logical_leg(q, leg))
"""Return the Clebsch-Gordan w-matrix for `sector` and product-symmetry component `n`; Abelian components return the shared trivial matrix."""
@inline function sector_wmat(q::TLArray{T, QD, N, RD, QT, PS, M, RMT}, sector::Int, n::Int) where {T, QD, N, RD, QT, PS, M, RMT}
    isabelian(symm(q)[n]) && return _trivial_wmat()
    return _wmat_from_storage(q.wmatdata, q.wmatinfo, sector, nonabelian_wmat_slot(PS, n))
end
@inline function sector_wmat(q::TLArray{T, QD, N, RD, QT, PS, M, RMT}, sector::Int, ::Val{n}) where {T, QD, N, RD, QT, PS, M, RMT, n}
    is_stored_wmat_symmetry(PS, Val(n)) || return _trivial_wmat()
    return _wmat_from_storage(q.wmatdata, q.wmatinfo, sector, nonabelian_wmat_slot(PS, Val(n)))
end
"""Return the stored non-Abelian w-matrix for `sector` by compact non-Abelian `slot`, bypassing the full product-symmetry index."""
@inline function sector_wmat_slot(q::TLArray, sector::Int, slot::Int)
    return _wmat_from_storage(q.wmatdata, q.wmatinfo, sector, slot)
end
@inline function sector_wmat_slot(q::AbstractTLArray, sector::Int, slot::Int)
    n = nonabelian_symmetry_indices(productsymm(q))[slot]
    return sector_wmat(q, sector, n)
end
"""Return the concrete RMT payload in defined `sector`; throw `ArgumentError` when the sector has not been materialized."""
@inline function _defined_sector_rmt(q::TLArray, sector::Int)
    q.isdefined[sector] || throw(ArgumentError("sector $sector is not evaluated"))
    return q.RMTs[sector]
end
@inline stored_sector_rmt_dim(q::TLArray, sector::Int) = size(_defined_sector_rmt(q, sector))
"""Return the `RD` logical RMT axis dimensions for `sector`, including deferred physical-leg permutation and without copying payload data."""
@inline sector_rmt_dim(q::TLArray{T, QD, N, RD}, sector::Int) where {T, QD, N, RD} =
    ntuple(i -> i <= QD ? stored_sector_rmt_dim(q, sector)[_logical_leg(q, i)] :
                 stored_sector_rmt_dim(q, sector)[i], Val(RD))
@inline function sector_rmt_axis_dim(q::TLArray, sector::Int, leg::Int)
    return sector_rmt_dim(q, sector)[leg]
end
"""Return `(rmt, scale)` for `sector`; callers must apply `scale` to `rmt` when logical numerical values are required."""
@inline sector_rmt(q::TLArray{T}, sector::Int) where {T} =
    (_defined_sector_rmt(q, sector), stored_scale(q))

@inline inds(q::TLArrayContraction{T, QD}) where {T, QD} =
    ntuple(l -> _logical_index(q, l), Val(QD))
@inline spaces(q::TLArrayContraction{T, QD}) where {T, QD} =
    ntuple(l -> stored_spaces(q)[_logical_leg(q, l)], Val(QD))
@inline sector_count(q::TLArrayContraction) = length(stored_qlabels(q))
@inline sector_slots(q::TLArrayContraction) = eachindex(stored_qlabels(q))
@inline nsectors(q::TLArrayContraction) = count(!, q.iszero)
@inline is_sector_defined(q::TLArrayContraction, sector::Int) = q.isdefined[sector]
@inline is_sector_zero(q::TLArrayContraction, sector::Int) = q.iszero[sector]
@inline is_sector_active(q::TLArrayContraction, sector::Int) = !q.iszero[sector]
@inline sector_qlabel(q::TLArrayContraction, sector::Int, leg::Int) =
    stored_sector_qlabel(q, sector, _logical_leg(q, leg))
@inline sector_qlabel(::Type{QT}, q::TLArrayContraction, sector::Int, leg::Int) where {QT} =
    stored_sector_qlabel(QT, q, sector, _logical_leg(q, leg))
@inline function sector_wmat(q::TLArrayContraction{T, QD, N, RD, QT, PS, M, RMT}, sector::Int, n::Int) where {T, QD, N, RD, QT, PS, M, RMT}
    isabelian(symm(q)[n]) && return _trivial_wmat()
    return _wmat_from_storage(q.wmatdata, q.wmatinfo, sector, nonabelian_wmat_slot(PS, n))
end
@inline function sector_wmat(q::TLArrayContraction{T, QD, N, RD, QT, PS, M, RMT}, sector::Int, ::Val{n}) where {T, QD, N, RD, QT, PS, M, RMT, n}
    is_stored_wmat_symmetry(PS, Val(n)) || return _trivial_wmat()
    return _wmat_from_storage(q.wmatdata, q.wmatinfo, sector, nonabelian_wmat_slot(PS, Val(n)))
end
@inline function sector_wmat_slot(q::TLArrayContraction, sector::Int, slot::Int)
    return _wmat_from_storage(q.wmatdata, q.wmatinfo, sector, slot)
end
@inline function sector_rmt_axis_dim(q::TLArrayContraction, sector::Int, leg::Int)
    return q.rmt_sizes[sector][leg]
end
@inline stored_sector_rmt_dim(q::TLArrayContraction, sector::Int) = q.rmt_sizes[sector]
@inline function sector_rmt_dim(q::TLArrayContraction{T, QD, N, RD}, sector::Int) where {T, QD, N, RD}
    source_dims = stored_sector_rmt_dim(q, sector)
    return ntuple(i -> i <= QD ? source_dims[_logical_leg(q, i)] : source_dims[i], Val(RD))
end
@inline function _defined_sector_rmt(q::TLArrayContraction, sector::Int)
    q.isdefined[sector] || throw(ArgumentError("sector $sector is not evaluated"))
    return q.RMTs[sector]
end
@inline sector_rmt(q::TLArrayContraction{T}, sector::Int) where {T} =
    (_defined_sector_rmt(q, sector), stored_scale(q))

@inline inds(q::SubTLArray{T, QD}) where {T, QD} =
    ntuple(l -> _logical_index(q, l), Val(QD))
@inline spaces(q::SubTLArray{T, QD}) where {T, QD} =
    ntuple(l -> stored_spaces(q)[_logical_leg(q, l)], Val(QD))
@inline sector_count(q::SubTLArray) = length(stored_qlabels(q))
@inline sector_slots(q::SubTLArray) = eachindex(stored_qlabels(q))
@inline source_sector(q::SubTLArray, sector::Int) = q.source_sectors[sector]
@inline nsectors(q::SubTLArray) = count(!, q.iszero)
@inline is_sector_defined(q::SubTLArray, sector::Int) = q.isdefined[sector]
@inline is_sector_zero(q::SubTLArray, sector::Int) = q.iszero[sector]
@inline is_sector_active(q::SubTLArray, sector::Int) = !q.iszero[sector]
@inline sector_qlabel(q::SubTLArray, sector::Int, leg::Int) =
    stored_sector_qlabel(q, sector, _logical_leg(q, leg))
@inline sector_qlabel(::Type{QT}, q::SubTLArray, sector::Int, leg::Int) where {QT} =
    stored_sector_qlabel(QT, q, sector, _logical_leg(q, leg))
@inline function sector_wmat(q::SubTLArray{T, QD, N, RD, QT, PS, M, RMT}, sector::Int, n::Int) where {T, QD, N, RD, QT, PS, M, RMT}
    isabelian(symm(q)[n]) && return _trivial_wmat()
    return _wmat_from_storage(q.wmatdata, q.wmatinfo, sector, nonabelian_wmat_slot(PS, n))
end
@inline function sector_wmat(q::SubTLArray{T, QD, N, RD, QT, PS, M, RMT}, sector::Int, ::Val{n}) where {T, QD, N, RD, QT, PS, M, RMT, n}
    is_stored_wmat_symmetry(PS, Val(n)) || return _trivial_wmat()
    return _wmat_from_storage(q.wmatdata, q.wmatinfo, sector, nonabelian_wmat_slot(PS, Val(n)))
end
@inline function sector_wmat_slot(q::SubTLArray, sector::Int, slot::Int)
    return _wmat_from_storage(q.wmatdata, q.wmatinfo, sector, slot)
end
@inline function sector_rmt_axis_dim(q::SubTLArray, sector::Int, leg::Int)
    return q.rmt_sizes[sector][leg]
end
@inline stored_sector_rmt_dim(q::SubTLArray, sector::Int) = q.rmt_sizes[sector]
@inline function sector_rmt_dim(q::SubTLArray{T, QD, N, RD}, sector::Int) where {T, QD, N, RD}
    source_dims = stored_sector_rmt_dim(q, sector)
    return ntuple(i -> i <= QD ? source_dims[_logical_leg(q, i)] : source_dims[i], Val(RD))
end
@inline function _defined_sector_rmt(q::SubTLArray, sector::Int)
    q.isdefined[sector] || throw(ArgumentError("sector $sector is not evaluated"))
    return q.RMTs[sector]
end
@inline sector_rmt(q::SubTLArray{T}, sector::Int) where {T} =
    (_defined_sector_rmt(q, sector), stored_scale(q))

const SingletonTLArray = Union{AddSingletonTLArray, DeleteSingletonTLArray}

@inline stored_wmat_storage(q::SingletonTLArray) = stored_wmat_storage(getfield(q, :arr))

@inline inds(q::Union{AddSingletonTLArray{T, QD}, DeleteSingletonTLArray{T, QD}}) where {T, QD} =
    ntuple(l -> _logical_index(q, l), Val(QD))
@inline spaces(q::Union{AddSingletonTLArray{T, QD}, DeleteSingletonTLArray{T, QD}}) where {T, QD} =
    ntuple(l -> stored_spaces(q)[_logical_leg(q, l)], Val(QD))
@inline sector_count(q::SingletonTLArray) = sector_count(q.arr)
@inline sector_slots(q::SingletonTLArray) = sector_slots(q.arr)
@inline nsectors(q::SingletonTLArray) = nsectors(q.arr)
@inline is_sector_defined(q::SingletonTLArray, sector::Int) = is_sector_defined(q.arr, sector)
@inline is_sector_zero(q::SingletonTLArray, sector::Int) = is_sector_zero(q.arr, sector)
@inline is_sector_active(q::SingletonTLArray, sector::Int) = is_sector_active(q.arr, sector)
@inline sector_qlabel(q::SingletonTLArray, sector::Int, leg::Int) =
    stored_sector_qlabel(q, sector, _logical_leg(q, leg))
@inline sector_qlabel(::Type{QT}, q::SingletonTLArray, sector::Int, leg::Int) where {QT} =
    stored_sector_qlabel(QT, q, sector, _logical_leg(q, leg))
@inline sector_wmat(q::SingletonTLArray, sector::Int, n::Int) =
    sector_wmat(q.arr, sector, n)
@inline sector_wmat(q::SingletonTLArray, sector::Int, ::Val{n}) where {n} =
    sector_wmat(q.arr, sector, Val(n))
@inline sector_wmat_slot(q::SingletonTLArray, sector::Int, slot::Int) =
    sector_wmat_slot(q.arr, sector, slot)

@inline function sector_rmt_dim(q::AddSingletonTLArray{T, QD, N, RD}, sector::Int) where {T, QD, N, RD}
    source_dims = sector_rmt_dim(q.arr, sector)
    ninsert = length(q.inserted_legs)
    return ntuple(Val(RD)) do axis
        if axis <= QD
            source_axis = q.result_to_source_legs[axis]
            source_axis == 0 ? 1 : source_dims[source_axis]
        else
            source_dims[axis - ninsert]
        end
    end::NTuple{RD, Int}
end

@inline function sector_rmt_dim(q::DeleteSingletonTLArray{T, QD, N, RD}, sector::Int) where {T, QD, N, RD}
    source_dims = sector_rmt_dim(q.arr, sector)
    ndelete = length(q.deleted_legs)
    return ntuple(Val(RD)) do axis
        axis <= QD ? source_dims[q.result_to_source_legs[axis]] :
                     source_dims[axis + ndelete]
    end::NTuple{RD, Int}
end

@inline sector_rmt_axis_dim(q::SingletonTLArray, sector::Int, leg::Int) =
    sector_rmt_dim(q, sector)[leg]

@inline function sector_rmt(q::AddSingletonTLArray{T, QD, N, RD, QT, PS, M, RMT}, sector::Int) where {T, QD, N, RD, QT, PS, M, RMT}
    rmt, scale = sector_rmt(q.arr, sector)
    data = _insert_singleton_rmt(rmt, q.inserted_legs, QD - length(q.inserted_legs), N)
    return data::RMT, (scale * stored_scale(q))::T
end

@inline function sector_rmt(q::DeleteSingletonTLArray{T, QD, N, RD, QT, PS, M, RMT}, sector::Int) where {T, QD, N, RD, QT, PS, M, RMT}
    rmt, scale = sector_rmt(q.arr, sector)
    data = _delete_singleton_rmt(rmt, q.deleted_legs, QD + length(q.deleted_legs), N)
    return data::RMT, (scale * stored_scale(q))::T
end

@inline _identity_phys_perm(::Val{QD}) where {QD} = ntuple(identity, Val(QD))
@inline _identity_rmt_perm(::Val{RD}) where {RD} = ntuple(identity, Val(RD))

function _normalize_phys_perm(perm, ::Val{QD}) where {QD}
    p = Tuple(Int(i) for i in perm)
    length(p) == QD || throw(ArgumentError("permutation length $(length(p)) != TLArray rank $QD"))
    p = p::NTuple{QD, Int}
    _is_valid_perm(p) || throw(ArgumentError("perm must be a valid permutation of 1:$QD"))
    return p
end

@inline function _is_valid_perm(perm::NTuple{QD, Int}) where {QD}
    seen = ntuple(_ -> false, Val(QD))
    for p in perm
        1 <= p <= QD || return false
        seen = Base.setindex(seen, true, p)
    end
    return all(seen)
end

function Base.hvcat(rows::Tuple{Vararg{Int}}, qs::AbstractTLArray...)
    nrows = length(rows)
    ncols = rows[1]
    all(==(ncols), rows) || throw(ArgumentError("TLArray matrix literal requires rectangular rows"))
    length(qs) == nrows * ncols || throw(ArgumentError("TLArray matrix literal has inconsistent dimensions"))
    out = Matrix{AbstractTLArray}(undef, nrows, ncols)
    pos = 1
    for i in 1:nrows, j in 1:ncols
        out[i, j] = qs[pos]
        pos += 1
    end
    return out
end

function _conj_sector_wmat_after_perm(q::AbstractTLArray{T, QD}, sector::Int, n::Int,
                                      wmat, perm::NTuple{QD, Int}) where {T, QD}
    S = symm(q)[n]
    isabelian(S) && return wmat

    ordered_qlabels, legdir = _sector_cgt_metadata_after_perm(q, sector, n, perm)
    return _conj_sector_wmat_from_metadata(S, ordered_qlabels, legdir, wmat)
end

function _sector_cgt_metadata_after_perm(q::AbstractTLArray{T, QD}, sector::Int, n::Int,
                                         perm::NTuple{QD, Int}) where {T, QD}
    qinds = inds(q)
    incoming = Int[l for l in 1:QD if qinds[perm[l]].dir == '+']
    outgoing = Int[l for l in 1:QD if qinds[perm[l]].dir == '-']

    sort!(incoming; by = l -> sector_qlabel(q, sector, perm[l])[n], alg = MergeSort)
    sort!(outgoing; by = l -> sector_qlabel(q, sector, perm[l])[n], alg = MergeSort)

    stored_to_view = (incoming..., outgoing...)
    ordered_qlabels = ntuple(i -> sector_qlabel(q, sector, perm[stored_to_view[i]])[n], Val(QD))
    return ordered_qlabels, (length(incoming), length(outgoing))
end

function _conj_sector_wmat_from_metadata(S, ordered_qlabels, legdir, wmat)
    m, k = legdir
    ins, outs = ordered_qlabels[1:m], ordered_qlabels[m+1:m+k]
    ins_, _ = remove_zeros(S, ins)
    outs_, _ = remove_zeros(S, outs)
    if ins_ == outs_
        conjperm = getNsave_Conjperm(S, ins_)
        return wmat[conjperm.perm, :]
    end
    return wmat
end

function _view_sector_wmat_slot(arr::AbstractTLArray{T, QD, N, RD, QT, PS}, sector::Int, ::Val{slot},
                                conj_flag::Bool, scale,
                                perm::NTuple{QD, Int}) where {T, QD, N, RD, QT, PS, slot}
    n = product_symmetry_index_from_wmat_slot(PS, Val(slot))
    wmat = if perm == _identity_phys_perm(Val(QD))
        sector_wmat_slot(arr, sector, slot)
    else
        _permute_sector_wmat(arr, sector, perm, Val(n))
    end
    conj_flag || return wmat
    return _conj_sector_wmat_after_perm(arr, sector, n, wmat, perm)
end

function _view_wmat_storage(arr::AbstractTLArray{T, QD, N, RD, QT, PS, M, RMT},
                            conj_flag::Bool, scale,
                            perm::NTuple{QD, Int}) where {T, QD, N, RD, QT, PS, M, RMT}
    old_perm = stored_perm(arr)
    relative_perm = ntuple(Val(QD)) do leg
        findfirst(==(perm[leg]), old_perm)::Int
    end
    relative_conj = stored_conj(arr) != conj_flag
    if !relative_conj && relative_perm == _identity_phys_perm(Val(QD))
        return stored_wmat_storage(arr)
    end

    source_wmatdata, source_wmatinfo = stored_wmat_storage(arr)
    wmatdata = copy(source_wmatdata)
    wmatinfo = copy(source_wmatinfo)
    for sector in sector_slots(arr), slot in 1:M
        _wmat_info_present(source_wmatinfo[sector][slot]) || continue
        _copy_wmat_to_storage!(wmatdata, wmatinfo, sector, slot,
                               _view_sector_wmat_slot(arr, sector, Val(slot),
                                                      relative_conj, scale, relative_perm))
    end
    return wmatdata, wmatinfo
end

@inline function _is_identity_view_state(conj_flag::Bool, scale::T,
                                         perm::NTuple{QD, Int}) where {T, QD}
    return !conj_flag && scale == one(T) && perm == _identity_phys_perm(Val(QD))
end

@inline function _view_state_matches(q::AbstractTLArray, conj_flag::Bool, scale,
                                     perm::NTuple{QD, Int}) where {QD}
    return stored_conj(q) == conj_flag &&
           stored_scale(q) == scale &&
           stored_perm(q) == perm
end

"""Return a `TLArray` value sharing storage with `q` but replacing deferred `conj_flag`, scalar `scale`, and logical-to-stored `perm`. Internal callers must validate/compose state before use."""
function _with_view_state(q::TLArray{T, QD, N, RD, QT, PS, M, RMT},
                          conj_flag::Bool, scale,
                          perm::NTuple{QD, Int}) where {T, QD, N, RD, QT, PS, M, RMT}
    _is_valid_perm(perm) || throw(ArgumentError("perm must be a valid permutation of 1:$QD"))
    RT = promote_type(eltype(RMT), typeof(scale))
    cscale = convert(RT, scale)
    _view_state_matches(q, conj_flag, cscale, perm) && T === RT && return q
    wmatdata, wmatinfo = stored_conj(q) == conj_flag && stored_perm(q) == perm ?
                         stored_wmat_storage(q) :
                         _view_wmat_storage(q, conj_flag, scale, perm)
    return TLArray(Val(:alias_storage_view_state), symm(q), stored_qlabels(q), wmatdata, wmatinfo,
                   q.RMTs, q.isdefined, q.iszero, stored_inds(q), stored_spaces(q),
                   conj_flag, cscale, perm)
end

function _with_view_state(q::TLArrayContraction{T, QD, N, RD, QT, PS, M, RMT},
                          conj_flag::Bool, scale,
                          perm::NTuple{QD, Int}) where {T, QD, N, RD, QT, PS, M, RMT}
    _is_valid_perm(perm) || throw(ArgumentError("perm must be a valid permutation of 1:$QD"))
    RT = promote_type(eltype(RMT), typeof(scale))
    cscale = convert(RT, scale)
    _view_state_matches(q, conj_flag, cscale, perm) && T === RT && return q
    wmatdata, wmatinfo = stored_conj(q) == conj_flag && stored_perm(q) == perm ?
                         stored_wmat_storage(q) :
                         _view_wmat_storage(q, conj_flag, scale, perm)
    return TLArrayContraction{RT, QD, N, RD, QT, PS, M, RMT}(
        stored_qlabels(q), wmatdata, wmatinfo, q.RMTs, q.isdefined, q.iszero,
        stored_inds(q), stored_spaces(q), conj_flag, cscale, perm, q.arr1, q.arr2,
        q.work_items, q.factors, q.perm1, q.perm2, q.rmt_sizes, q.lock)
end

function _with_view_state(q::SubTLArray{T, QD, N, RD, QT, PS, M, RMT},
                          conj_flag::Bool, scale,
                          perm::NTuple{QD, Int}) where {T, QD, N, RD, QT, PS, M, RMT}
    _is_valid_perm(perm) || throw(ArgumentError("perm must be a valid permutation of 1:$QD"))
    RT = promote_type(eltype(RMT), typeof(scale))
    cscale = convert(RT, scale)
    _view_state_matches(q, conj_flag, cscale, perm) && T === RT && return q
    wmatdata, wmatinfo = stored_conj(q) == conj_flag && stored_perm(q) == perm ?
                         stored_wmat_storage(q) :
                         _view_wmat_storage(q, conj_flag, scale, perm)
    return SubTLArray{RT, QD, N, RD, QT, PS, M, RMT}(
        stored_qlabels(q), wmatdata, wmatinfo, q.RMTs, q.isdefined, q.iszero,
        stored_inds(q), stored_spaces(q), conj_flag, cscale, perm, q.arr, q.source_sectors,
        q.saved_indices, q.affected_legs, q.rmt_sizes, q.lock)
end

function _with_view_state(q::AddSingletonTLArray{T, QD, N, RD, QT, PS, M, RMT},
                          conj_flag::Bool, scale,
                          perm::NTuple{QD, Int}) where {T, QD, N, RD, QT, PS, M, RMT}
    _is_valid_perm(perm) || throw(ArgumentError("perm must be a valid permutation of 1:$QD"))
    RT = promote_type(eltype(RMT), typeof(scale))
    cscale = convert(RT, scale)
    _view_state_matches(q, conj_flag, cscale, perm) && T === RT && return q
    return AddSingletonTLArray{RT, QD, N, RD, QT, PS, M, RMT}(
        stored_qlabels(q), stored_inds(q), stored_spaces(q), conj_flag, cscale, perm, q.arr,
        q.inserted_legs, q.source_to_result_legs, q.result_to_source_legs)
end

function _with_view_state(q::DeleteSingletonTLArray{T, QD, N, RD, QT, PS, M, RMT},
                          conj_flag::Bool, scale,
                          perm::NTuple{QD, Int}) where {T, QD, N, RD, QT, PS, M, RMT}
    _is_valid_perm(perm) || throw(ArgumentError("perm must be a valid permutation of 1:$QD"))
    RT = promote_type(eltype(RMT), typeof(scale))
    cscale = convert(RT, scale)
    _view_state_matches(q, conj_flag, cscale, perm) && T === RT && return q
    return DeleteSingletonTLArray{RT, QD, N, RD, QT, PS, M, RMT}(
        stored_qlabels(q), stored_inds(q), stored_spaces(q), conj_flag, cscale, perm, q.arr,
        q.deleted_legs, q.source_to_result_legs, q.result_to_source_legs)
end

"""
    sector_rmt_permuted(q, sector, perm) -> (rmt, scale)

Return sector `sector` in RMT-axis order `perm`, composing it with `q`'s
deferred view permutation. The payload is reused when no permutation or complex
conjugation copy is needed; otherwise a copied/permuted payload is returned.
`scale` remains separate so callers can defer multiplication.
"""
function sector_rmt_permuted(q::Union{TLArray{T, QD}, TLArrayContraction{T, QD},
                                      SubTLArray{T, QD},
                                      AddSingletonTLArray{T, QD},
                                      DeleteSingletonTLArray{T, QD}},
                             sector::Int,
                             perm::NTuple{RD, Int}) where {T, QD, RD}
    rmt, scale = sector_rmt(q, sector)
    source_perm = ntuple(i -> perm[i] <= QD ? stored_perm(q)[perm[i]] : perm[i], Val(RD))
    no_conj_copy = !stored_conj(q) || T <: Real
    if source_perm == _identity_rmt_perm(Val(RD)) && no_conj_copy
        return rmt, scale
    end

    payload = source_perm == _identity_rmt_perm(Val(RD)) ? copy(rmt) :
              _hptt_permutedims(rmt, source_perm)
    if stored_conj(q) && !(T <: Real)
        payload .= conj.(payload)
    end
    return payload, scale
end

"""Write sector `sector` of `q` in RMT-axis order `perm` into preallocated `buffer`, applying the deferred scalar scale. `buffer` must have the resulting shape."""
function sector_rmt_permuted!(buffer, q::AbstractTLArray{T},
                              sector::Int,
                              perm::NTuple{RD, Int}) where {T, RD}
    rmt, scale = sector_rmt_permuted(q, sector, perm)
    copyto!(buffer, rmt)
    scale != one(typeof(scale)) && (buffer .*= scale)
    return buffer
end

"""
    materialize(q)

Evaluate all pending sectors of lazy `q` and return `q`. Sector slots retain
their original numbers, including zero slots. Concrete `TLArray`s are returned
unchanged; this does not canonicalize deferred permutation, scale, or
conjugation state.
"""
materialize(q::TLArray) = q

function materialize(q::TLArrayContraction)
    compute_sectors(q, sector_slots(q))
    return q
end

function materialize(q::SubTLArray)
    compute_sectors(q, sector_slots(q))
    return q
end

function materialize(q::SingletonTLArray)
    materialize(q.arr)
    return q
end

@inline function _normalize_tlarray_view(arr::AbstractTLArray{T, QD, N, RD, QT, PS, M, RMT},
                                         conj_flag::Bool,
                                         scale,
                                         perm::NTuple{QD, Int}) where {T, QD, N, RD, QT, PS, M, RMT}
    _is_valid_perm(perm) || throw(ArgumentError("perm must be a valid permutation of 1:$QD"))
    return _with_view_state(arr, conj_flag, scale, perm)
end

"""Create a lazy logical-leg permutation view of `q`. `perm` must be a valid 1-based permutation of `1:QD`; no RMT payload is permuted here."""
function _view_permutedims(q::AbstractTLArray{T, QD}, perm) where {T, QD}
    p = _normalize_phys_perm(perm, Val(QD))
    new_perm = ntuple(l -> stored_perm(q)[p[l]], Val(QD))
    return _normalize_tlarray_view(q, stored_conj(q), stored_scale(q), new_perm)
end

"""Create a lazy conjugation view of `q` by toggling stored conjugation state; complex RMT entries are not copied or conjugated eagerly."""
function _view_conj(q::AbstractTLArray{T, QD}) where {T, QD}
    return _normalize_tlarray_view(q, !stored_conj(q), stored_scale(q), stored_perm(q))
end

"""Create a lazy scaled view of `q` by composing scalar `fac` with deferred scale metadata; no RMT payload is multiplied eagerly."""
function _view_scale(q::AbstractTLArray{T, QD, N, RD, QT}, fac::Number) where {T, QD, N, RD, QT}
    RT = promote_type(T, typeof(fac))
    if iszero(fac)
        qlabels = [ntuple(leg -> sector_qlabel(q, sector, leg), Val(QD))::NTuple{QD, QT}
                   for sector in sector_slots(q)]
        M = n_nonabelian_symmetries(productsymm(q))
        wmatinfo = [_empty_wmat_info(Val(M)) for _ in 1:sector_count(q)]
        RMTs = Vector{Array{RT, RD}}(undef, sector_count(q))
        return TLArray(symm(q), qlabels, Float64[], wmatinfo, RMTs, inds(q), spaces(q))
    end
    cfac = fac
    new_scale = stored_conj(q) ? stored_scale(q) * conj(cfac) : stored_scale(q) * cfac
    return _normalize_tlarray_view(q, stored_conj(q), new_scale, stored_perm(q))
end

@inline function _stored_position(stored_to_phys::NTuple{QD, Int}, phys_leg::Int) where {QD}
    @inbounds for stored_pos in 1:QD
        stored_to_phys[stored_pos] == phys_leg && return stored_pos
    end
    throw(BoundsError(stored_to_phys, phys_leg))
end

@inline _phys_to_stored_order(stored_to_phys::NTuple{QD, Int}) where {QD} =
    ntuple(phys_leg -> _stored_position(stored_to_phys, phys_leg), Val(QD))

function _stored_leg_order(q::AbstractTLArray{T, QD, N}, sector::Int, n::Int) where {T, QD, N}
    qinds = inds(q)
    incoming = Int[l for l in 1:QD if qinds[l].dir == '+']
    outgoing = Int[l for l in 1:QD if qinds[l].dir == '-']

    sort!(incoming; by = l -> sector_qlabel(q, sector, l)[n], alg = MergeSort)
    sort!(outgoing; by = l -> sector_qlabel(q, sector, l)[n], alg = MergeSort)

    n_in = length(incoming)
    return ntuple(i -> i <= n_in ? incoming[i] : outgoing[i - n_in], Val(QD))
end

function _stored_leg_order(qlabels::AbstractVector{<:Tuple},
                           inds::NTuple{QD, TLIndex},
                           sector::Int,
                           n::Int) where {QD}
    incoming = Int[l for l in 1:QD if inds[l].dir == '+']
    outgoing = Int[l for l in 1:QD if inds[l].dir == '-']

    sort!(incoming; by = l -> qlabels[sector][l][n], alg = MergeSort)
    sort!(outgoing; by = l -> qlabels[sector][l][n], alg = MergeSort)

    n_in = length(incoming)
    return ntuple(i -> i <= n_in ? incoming[i] : outgoing[i - n_in], Val(QD))
end

function _sector_cgt_metadata(q::AbstractTLArray{T, QD, N}, sector::Int, n::Int) where {T, QD, N}
    stored_to_phys = _stored_leg_order(q, sector, n)
    qlabels = ntuple(i -> sector_qlabel(q, sector, stored_to_phys[i])[n], Val(QD))

    qinds = inds(q)
    n_in = count(l -> qinds[l].dir == '+', 1:QD)
    return qlabels, _phys_to_stored_order(stored_to_phys), (n_in, QD - n_in)
end

function _sector_cgt_metadata(qlabels::AbstractVector{<:Tuple},
                           inds::NTuple{QD, TLIndex},
                           sector::Int,
                           n::Int) where {QD}
    stored_to_phys = _stored_leg_order(qlabels, inds, sector, n)
    stored_qlabels = ntuple(i -> qlabels[sector][stored_to_phys[i]][n], Val(QD))

    n_in = count(l -> inds[l].dir == '+', 1:QD)
    return stored_qlabels, _phys_to_stored_order(stored_to_phys), (n_in, QD - n_in)
end

function _sector_label_widths(q::TLArray{T, QD, N}, sector_index::Int) where {T, QD, N}
    map(1:N) do n
        vals = (v for l in 1:QD for v in sector_qlabel(q, sector_index, l)[n])
        mxabs = maximum(abs, vals, init=0)
        needs_sign = symm(q)[n] <: U1
        ndigits(max(mxabs, 1)) + (needs_sign ? 1 : 0)
    end
end

function _print_sector_qlabels(io::IO, q::TLArray{T, QD, N}, sector_index::Int, widths) where {T, QD, N}
    print(io, "[")
    for l in 1:QD
        l > 1 && print(io, " ;")
        for n in 1:N
            print(io, " ")
            for v in sector_qlabel(q, sector_index, l)[n]
                print(io, lpad(v, widths[n]))
            end
        end
    end
    print(io, " ]")
end

function _print_sector_cgt_dims(io::IO, q::TLArray{T, QD, N}, sector_index::Int) where {T, QD, N}
    first = true
    for n in 1:N
        S = symm(q)[n]
        isabelian(S) && continue
        dims = [dimension(S, sector_qlabel(q, sector_index, l)[n]) for l in 1:QD]
        first && print(io, "| ")
        first = false
        print(io, join(dims, "x"), "\t")
    end
end

function _sector_cgt_size_2d(q::TLArray{T, 2, N}, sector_index::Int) where {T, N}
    total = 1
    for n in 1:N
        S = symm(q)[n]
        isabelian(S) && continue
        total *= dimension(S, sector_qlabel(q, sector_index, 1)[n])
    end
    return total
end

Base.ndims(q::TLArray{T, QD}) where {T, QD} = QD

function _sector_cgt_size_2d(symm::NTuple{N, Any},
                             qlabels::Vector{NTuple{QD, QT}},
                             sector_index::Int) where {N, QD, QT}
    total = 1
    for n in 1:N
        S = symm[n]
        isabelian(S) && continue
        total *= dimension(S, qlabels[sector_index][1][n])
    end
    return total
end

function _orient_wmats!(q::TLArray{T, QD, N, RD, QT, PS, M, RMT}) where {T, QD, N, RD, QT, PS, M, RMT}
    for sector_index in sector_slots(q)
        q.iszero[sector_index] && continue
        for m in 1:M
            wmat = sector_wmat_slot(q, sector_index, m)
            if length(wmat) == 1 && wmat[1] < 0
                wmat[:] .*= -1
                if q.RMTs[sector_index] isa DiagRMT
                    q.RMTs[sector_index] = -q.RMTs[sector_index]
                else
                    q.RMTs[sector_index][:] .*= -one(T)
                end
            end
        end
    end
    return q
end

function _display_scalar_rmt(q::TLArray, sector_index::Int)
    rmt, scale = sector_rmt(q, sector_index)
    return scale * only(rmt) * _scalar_wmat_product(q, sector_index)
end

function _display_scalar_rmt(q::TLArray{T, 2}, sector_index::Int) where {T}
    rmt, scale = sector_rmt(q, sector_index)
    return scale * only(rmt) * _scalar_wmat_product(q, sector_index) /
           sqrt(_sector_cgt_size_2d(q, sector_index))
end

function _copy_assigned_vector(v::Vector, inds; deep::Bool=false)
    out = similar(v, length(inds))
    for (out_index, in_index) in pairs(collect(inds))
        if isassigned(v, in_index)
            out[out_index] = deep ? deepcopy(v[in_index]) : v[in_index]
        end
    end
    return out
end

function _copy_wmat_storage(q::AbstractTLArray; deep::Bool=false)
    wmatdata, wmatinfo = stored_wmat_storage(q)
    return (deep ? copy(wmatdata) : wmatdata,
            deep ? copy(wmatinfo) : wmatinfo)
end

function _copy_wmat_storage(q::AbstractTLArray{T, QD, N, RD, QT, PS, M},
                            inds; deep::Bool=false) where {T, QD, N, RD, QT, PS, M}
    collected = collect(inds)
    source_wmatdata, source_wmatinfo = stored_wmat_storage(q)
    total = 0
    for sector in collected, slot in 1:M
        offset, nrow, ncol = source_wmatinfo[sector][slot]
        offset == 0 && continue
        total += nrow * ncol
    end

    wmatdata = Vector{Float64}(undef, total)
    wmatinfo = Vector{WMatInfo{M}}(undef, length(collected))
    next_offset = 1
    for (out_index, sector) in pairs(collected)
        wmatinfo[out_index] = ntuple(Val(M)) do slot
            offset, nrow, ncol = source_wmatinfo[sector][slot]
            offset == 0 && return (0, 0, 0)
            len = nrow * ncol
            copyto!(view(wmatdata, next_offset:next_offset + len - 1),
                    view(source_wmatdata, offset:offset + len - 1))
            new_offset = next_offset
            next_offset += len
            (new_offset, nrow, ncol)
        end
    end
    return wmatdata, wmatinfo
end

_copy_sector_RMTs(q::TLArray; deep::Bool=false) =
    _copy_assigned_vector(q.RMTs, eachindex(q.RMTs); deep=deep)
_copy_sector_RMTs(q::TLArray, inds; deep::Bool=false) =
    _copy_assigned_vector(q.RMTs, inds; deep=deep)

# Construct a TLArray with the same sectors but with itags replaced.
# itags: Tuple{Vararg{AbstractString, QD}} — one tag per leg; all other TLIndex
# fields are preserved.
function TLArray(q::TLArray{T, QD, N, RD}, itags::Tuple{Vararg{AbstractString, QD}}) where {T, QD, N, RD}
    logical_inds = inds(q)
    new_inds = ntuple(l -> TLIndex(itags[l], logical_inds[l].dir, logical_inds[l].plev,
                                  logical_inds[l].lock, logical_inds[l].dual), QD)
    return TLArray(q, new_inds)
end

# Construct a TLArray with the same sectors but with all TLIndex fields replaced.
# inds: NTuple{QD, TLIndex} — one full TLIndex per leg.
# Arrow directions must match the original TLArray (only itags/lock/plev/dual may differ).
function TLArray(q::TLArray{T, QD, N, RD}, new_inds::NTuple{QD, TLIndex}) where {T, QD, N, RD}
    @assert ntuple(l -> new_inds[l].dir, QD) == ntuple(l -> inds(q)[l].dir, QD) "TLArray(q, inds): arrow directions must match the original TLArray on all legs"
    stored_new_inds = ntuple(Val(QD)) do stored_leg
        logical_leg = findfirst(==(stored_leg), stored_perm(q))
        idx = new_inds[logical_leg]
        stored_conj(q) ? change_dir(idx) : idx
    end
    return TLArray(Val(:alias_storage_view_state), symm(q), copy(stored_qlabels(q)),
                   _copy_wmat_storage(q; deep=true)...,
                   _copy_sector_RMTs(q; deep=true),
                   copy(q.isdefined), copy(q.iszero),
                   stored_new_inds, _copy_spaces_tuple(stored_spaces(q)),
                   stored_conj(q), stored_scale(q), stored_perm(q))
end

TLArray(q::TLArrayContraction{T, QD, N, RD}, inds::NTuple{QD, TLIndex}) where {T, QD, N, RD} =
    TLArray(TLArray(q), inds)

TLArray(q::TLArrayContraction{T, QD, N, RD}, itags::Tuple{Vararg{AbstractString, QD}}) where {T, QD, N, RD} =
    TLArray(TLArray(q), itags)

TLArray(q::TLArray) = q

function _tlarray_alias_materialized_storage(q::Union{TLArrayContraction{T, QD, N, RD, QT, PS, M, RMT},
                                                      SubTLArray{T, QD, N, RD, QT, PS, M, RMT}}) where {T, QD, N, RD, QT, PS, M, RMT}
    materialize(q)
    return TLArray(Val(:alias_storage_view_state), symm(q), stored_qlabels(q), q.wmatdata, q.wmatinfo,
                   q.RMTs, q.isdefined, q.iszero, stored_inds(q), stored_spaces(q),
                   stored_conj(q), stored_scale(q), stored_perm(q))
end

TLArray(q::Union{TLArrayContraction, SubTLArray}) = _tlarray_alias_materialized_storage(q)

"""
    _tlarray_materialize_singleton(q::SingletonTLArray) -> TLArray

Materialize a singleton insertion/deletion wrapper into explicit owned dense
RMT storage.

The result uses logical indices, spaces, and q-labels with identity embedded
view state. Sector slot numbers are preserved. Defined sectors copy the
singleton-adjusted payload returned by the common RMT interface and apply its
deferred scalar scale; undefined zero sectors remain unassigned.
"""
function _tlarray_materialize_singleton(q::Union{AddSingletonTLArray{T, QD, N, RD, QT, PS, M},
                                                DeleteSingletonTLArray{T, QD, N, RD, QT, PS, M}}) where {T, QD, N, RD, QT, PS, M}
    materialize(q)
    qlabels = Vector{NTuple{QD, QT}}(undef, sector_count(q))
    RMTs = Vector{Array{T, RD}}(undef, sector_count(q))
    for sector in sector_slots(q)
        qlabels[sector] = ntuple(leg -> sector_qlabel(QT, q, sector, leg), Val(QD))
        is_sector_defined(q, sector) || continue
        rmt, scale = sector_rmt(q, sector)
        dims = sector_rmt_dim(q, sector)
        dest = Array{T, RD}(undef, dims)
        dest .= convert(T, scale) .* rmt
        RMTs[sector] = dest
    end
    return TLArray(symm(q), qlabels, _copy_wmat_storage(q; deep=true)...,
                   RMTs, inds(q), spaces(q))
end

TLArray(q::SingletonTLArray) = _tlarray_materialize_singleton(q)

Base.convert(::Type{TLArray}, q::AbstractTLArray) = TLArray(q)
function Base.convert(::Type{TLA}, q::AbstractTLArray) where {TLA<:TLArray}
    q isa TLA && return q
    result = TLArray(q)
    result isa TLA && return result
    throw(ArgumentError("cannot convert $(typeof(q)) to requested TLArray type $TLA without changing storage type"))
end

Base.getindex(q::TLArray, i::Int) = TLArray(q, i)
Base.firstindex(q::AbstractTLArray) = 1
Base.lastindex(q::AbstractTLArray) = sector_count(q)

function _normalize_tlarray_sector_index(i::Int, nsectors::Int)
    i == 0 && throw(BoundsError(1:nsectors, i))
    idx = i < 0 ? nsectors + i + 1 : i
    1 <= idx <= nsectors || throw(BoundsError(1:nsectors, i))
    return idx
end

function _normalize_tlarray_sector_selector(selector, nsectors::Int)
    if selector isa Colon
        return collect(1:nsectors)
    elseif selector isa Integer
        return Int[_normalize_tlarray_sector_index(Int(selector), nsectors)]
    elseif selector isa AbstractVector{Bool}
        length(selector) == nsectors || throw(DimensionMismatch(
            "sector selector of length $(length(selector)) does not match number of sectors $nsectors"))
        return findall(selector)
    elseif selector isa AbstractRange{<:Integer}
        return _normalize_tlarray_sector_selector(collect(selector), nsectors)
    elseif selector isa AbstractVector{<:Integer}
        inds = Int[_normalize_tlarray_sector_index(Int(i), nsectors) for i in selector]
        length(unique(inds)) == length(inds) || throw(ArgumentError(
            "sector selector must not contain duplicate indices"))
        return inds
    else
        throw(ArgumentError(
            "sector selector must be :, Int, AbstractRange{<:Integer}, AbstractVector{<:Integer}, or AbstractVector{Bool}"))
    end
end

"""
    TLArray(q::TLArray, selector) -> TLArray

Construct a concrete tensor containing a selected subset of sector slots.

`selector` may be `:`, an `Int`, an integer range, an integer vector, or a
Boolean mask. Negative integer indices count from the end. The selector is
normalized over `sector_count(q)`. Selected qlabels, w-matrix storage, RMT
payloads, indices, and cached leg-space metadata are copied into a new owned
`TLArray`; view state on `q` is not preserved by this sector-copy constructor.
"""
function TLArray(q::TLArray{T, QD, N, RD}, selector) where {T, QD, N, RD}
    sector_indices = _normalize_tlarray_sector_selector(selector, sector_count(q))
    qlabels = copy(stored_qlabels(q)[sector_indices])
    wmatdata, wmatinfo = _copy_wmat_storage(q, sector_indices; deep=true)
    RMTs = _copy_sector_RMTs(q, sector_indices; deep=true)
    return TLArray(symm(q), qlabels, wmatdata, wmatinfo, RMTs, inds(q), _copy_spaces_tuple(spaces(q)))
end

"""
    getindex(q::TLArray, selector) -> TLArray

Select sector slots from a concrete TLArray.

`selector` may be `:`, an integer range, an integer vector, or a Boolean vector.
The returned tensor owns copies of the selected sector metadata and payloads.
"""
Base.getindex(q::TLArray,
              selector::Union{Colon, AbstractRange{<:Integer},
                              AbstractVector{<:Integer}, AbstractVector{Bool}}) = TLArray(q, selector)

function _unsymmetric_index_sector(q::TLArray{T, QD, 0}) where {T, QD}
    active = findall(!, q.iszero)
    length(active) <= 1 || throw(ArgumentError(
        "an unsymmetric TLArray must have at most one active sector"))
    return isempty(active) ? 0 : only(active)
end

function _unsymmetric_dense_data(q::TLArray{T, QD, 0}) where {T, QD}
    sector = _unsymmetric_index_sector(q)
    sector == 0 && return zeros(T, size(q))
    q.isdefined[sector] || throw(ArgumentError(
        "cannot index an unevaluated unsymmetric TLArray sector"))
    rmt, scale = sector_rmt_permuted(q, sector, _identity_rmt_perm(Val(QD)))
    data = Array{T, QD}(undef, size(q))
    data .= scale .* rmt
    return data
end

Base.Array(q::TLArray{T, QD, 0}) where {T, QD} = _unsymmetric_dense_data(q)

@inline _inverse_phys_perm(perm::NTuple{QD, Int}) where {QD} =
    ntuple(i -> findfirst(==(i), perm)::Int, Val(QD))

@inline _unsymmetric_trivial_view(q::TLArray{T, QD, 0}) where {T, QD} =
    !stored_conj(q) && stored_scale(q) == one(T) &&
    stored_perm(q) == _identity_phys_perm(Val(QD))

function _unsymmetric_getindex(q::TLArray{T, QD, 0}, I...) where {T, QD}
    indices = Base.to_indices(q, axes(q), I)
    result = getindex(_unsymmetric_dense_data(q), indices...)
    result isa Number && return result

    retained_legs = Int[]
    for (leg, index) in enumerate(indices)
        index isa Integer || push!(retained_legs, leg)
    end
    ndims(result) == length(retained_legs) || return result
    result_inds = Tuple(inds(q)[leg] for leg in retained_legs)
    return TLArray(Array(result), result_inds; copy=false)
end

"""Read elements of an unsymmetric `TLArray` using ordinary Julia array indexing.

Integer indexing on symmetric tensors selects sector slots; for unsymmetric
tensors it indexes the tensor elements. Use `TLArray(q, selector)` to select
sector slots explicitly.
"""
Base.getindex(q::TLArray{T, QD, 0}, I...) where {T, QD} =
    _unsymmetric_getindex(q, I...)
Base.getindex(q::TLArray{T, QD, 0}, I::Union{Colon,
              AbstractRange{<:Integer}, AbstractVector{<:Integer},
              AbstractVector{Bool}}) where {T, QD} =
    _unsymmetric_getindex(q, I)
Base.length(q::TLArray{T, QD, 0}) where {T, QD} = prod(size(q); init=1)
Base.firstindex(q::TLArray{T, QD, 0}) where {T, QD} = 1
Base.lastindex(q::TLArray{T, QD, 0}) where {T, QD} = length(q)
Base.IndexStyle(::Type{<:TLArray{T, QD, 0}}) where {T, QD} = IndexLinear()

function Base.setindex!(q::TLArray{T, QD, 0}, value, I...) where {T, QD}
    sector = _unsymmetric_index_sector(q)
    if sector == 0
        defined = findall(q.isdefined)
        length(defined) == 1 && (sector = only(defined))
    end
    sector == 0 && throw(ArgumentError(
        "cannot assign into an unsymmetric TLArray with no defined sector"))
    q.isdefined[sector] || throw(ArgumentError(
        "cannot assign into an unevaluated unsymmetric TLArray sector"))
    scale = stored_scale(q)
    iszero(scale) && throw(ArgumentError(
        "cannot assign through an unsymmetric TLArray view with zero scale"))

    raw = q.RMTs[sector]
    if _unsymmetric_trivial_view(q)
        setindex!(raw, value, I...)
        q.iszero[sector] = _rmt_iszero(raw)
        return q
    end

    data = _unsymmetric_dense_data(q)
    setindex!(data, value, I...)
    logical = data ./ scale
    if stored_conj(q) && !(T <: Real)
        logical .= conj.(logical)
    end
    stored_data = stored_perm(q) == _identity_phys_perm(Val(QD)) ? logical :
                  permutedims(logical, _inverse_phys_perm(stored_perm(q)))
    copyto!(raw, stored_data)
    q.iszero[sector] = _rmt_iszero(raw)
    return q
end

"""
    _scalar_wmat_product(q, sector_index::Int) -> Float64

Return the product of scalar w-matrix slots for a scalar sector.

`q` supplies the sector w-matrix storage and `sector_index` selects the sector.
Only one-element w-matrix slots contribute. This factor is needed when reading
0D TLArray values, where the scalar RMT element alone does not include scalar
CGT/OM factors.
"""
function _scalar_wmat_product(q::AbstractTLArray{T, QD, N, RD, QT, PS, M}, sector_index::Int) where {T, QD, N, RD, QT, PS, M}
    factor = 1.0
    for slot in 1:M
        wmat = sector_wmat_slot(q, sector_index, slot)
        length(wmat) == 1 && (factor *= only(wmat))
    end
    return factor
end

# For 0-dimensional TLArray (scalar), q[] returns the unique RMT element.
"""
    getindex(q::TLArray{T,0,N,N}) -> Number

Read the scalar value of a fully contracted concrete TLArray.

The tensor must have zero or one active sector, and the active RMT must contain
one element. The returned value combines the RMT element, stored scale, and
scalar w-matrix factors. An empty scalar tensor returns `zero(T)`.
"""
function Base.getindex(q::TLArray{T, 0, N, N}) where {T, N}
    @assert nsectors(q) <= 1 "0D TLArray must have zero or one sector"
    if nsectors(q) == 1
        sector_index = findfirst(!, q.iszero)
        rmt, scale = sector_rmt(q, sector_index)
        @assert length(rmt) == 1 "0D TLArray RMT must be a scalar"
        return scale * only(rmt) * _scalar_wmat_product(q, sector_index)
    else return zero(T) end
end

"""
    getindex(q::TLArrayContraction{T,0,N,N}) -> Number

Compute and read the scalar value of a lazy full contraction.

The single active sector is computed on demand, then read with the same
RMT/scale/w-matrix factor convention as the concrete scalar `TLArray` method.
"""
function Base.getindex(q::TLArrayContraction{T, 0, N, N}) where {T, N}
    @assert nsectors(q) <= 1 "0D TLArrayContraction must have zero or one sector"
    if nsectors(q) == 1
        sector_index = first(sector for sector in sector_slots(q) if is_sector_active(q, sector))
        compute_sectors(q, [sector_index])
        rmt, scale = sector_rmt(q, sector_index)
        @assert length(rmt) == 1 "0D TLArrayContraction RMT must be a scalar"
        return scale * only(rmt) * _scalar_wmat_product(q, sector_index)
    else return zero(T) end
end

# ─── Leg selection utilities ─────────────────────────────────────────────────

"""
    findlegs(q::TLArray, pred::Function) -> Vector{Int}

Find all leg indices where `pred(qindex)` returns true.

# Examples
```julia
findlegs(q, idx -> idx.dir == '-')                    # all outgoing legs
findlegs(q, idx -> occursin("site", idx.itags))       # legs with "site" in tag
findlegs(q, idx -> idx.dir == '-' && idx.plev == 0)   # outgoing, unprimed
```
"""
findlegs(q::AbstractTLArray{T, QD}, pred::Function) where {T, QD} =
    [i for i in 1:QD if pred(inds(q)[i])]

"""
    findleg(q::TLArray, pred::Function) -> Int

Find the first leg index where `pred(qindex)` returns true.
Throws an error if no leg matches.

# Examples
```julia
findleg(q, idx -> idx.itags == "bond")   # first leg with exact tag "bond"
findleg(q, idx -> idx.dir == '+')        # first incoming leg
```
"""
function findleg(q::AbstractTLArray{T, QD}, pred::Function) where {T, QD}
    qinds = inds(q)
    for i in 1:QD pred(qinds[i]) && return i end
    return nothing
end

# Internal: check if a TLIndex matches all specified criteria
"""
    _matches_criteria(idx::TLIndex; dir=nothing, itag=nothing, plev=nothing, lock=nothing) -> Bool

Return whether one index satisfies every supplied leg-selection criterion.

`dir`, `plev`, and `lock` require exact equality when provided. `itag` uses the
same tag-query semantics as `findlegs`: comma-separated tags are matched as a
set and tuple/vector selectors are treated as alternatives.
"""
function _matches_criteria(idx::TLIndex; dir=nothing, itag=nothing, plev=nothing, lock=nothing)
    (!isnothing(dir)  && idx.dir != dir)                                 && return false
    (!isnothing(itag) && !_matches_itag_selector(idx.itags, itag))       && return false
    (!isnothing(plev) && idx.plev != plev)                               && return false
    (!isnothing(lock) && idx.lock != lock)                               && return false
    return true
end

"""
    findlegs(q::TLArray; dir=nothing, itag=nothing, plev=nothing, lock=nothing, rev=false) -> Vector{Int}

Find all leg indices matching the specified criteria. Unspecified criteria match any value.
If `rev=true`, the selection is reversed: legs that do *not* match are returned.

# Arguments
- `dir`: Match direction ('+' for incoming, '-' for outgoing)
- `itag`: Match exact tags. A single string like `"a,b"` means both `a` and `b`
  must be present; a tuple/vector of strings means any one of those tag-sets may match
- `plev`: Match exact prime level
- `lock`: Match exact lock level
- `rev`: If `true`, return legs that do *not* satisfy the criteria (default `false`)

# Examples
```julia
findlegs(q; dir='-')                    # all outgoing legs
findlegs(q; itag="site")                # legs carrying tag "site"
findlegs(q; itag=("a,b", "a,c"))        # legs with tags a+b, or a+c
findlegs(q; dir='+', plev=0)            # incoming, unprimed legs
findlegs(q; lock=0)                     # non-locked legs
findlegs(q; dir='-', rev=true)          # all legs that are NOT outgoing
```
"""
function findlegs(q::AbstractTLArray{T, QD}; dir=nothing, itag=nothing, plev=nothing, lock=nothing, rev::Bool=false) where {T, QD}
    qinds = inds(q)
    return [i for i in 1:QD if _matches_criteria(qinds[i]; dir=dir, itag=itag, plev=plev, lock=lock) ⊻ rev]
end

"""
    findleg(q::TLArray; dir=nothing, itag=nothing, plev=nothing, lock=nothing, rev=false) -> Union{Int, Nothing}

Find the first leg index matching the specified criteria.
Returns `nothing` if no leg matches.
If `rev=true`, returns the first leg that does *not* match the criteria.

# Examples
```julia
findleg(q; itag="bond")           # first leg with "bond" in tag
findleg(q; dir='+')               # first incoming leg
findleg(q; lock=0)                # first non-locked leg
findleg(q; dir='+', rev=true)     # first leg that is NOT incoming
```
"""
function findleg(q::AbstractTLArray{T, QD}; dir=nothing, itag=nothing, plev=nothing, lock=nothing, rev::Bool=false) where {T, QD}
    qinds = inds(q)
    for i in 1:QD
        _matches_criteria(qinds[i]; dir=dir, itag=itag, plev=plev, lock=lock) ⊻ rev && return i
    end
    return nothing
end

"""
    _matching_targets(q::AbstractTLArray; require_unlocked=false) -> Set{TLIndex}

Build the set of index signatures in `q` that could match the opposite tensor.

Each target index has direction flipped with `change_dir`, because cross-tensor
matches require opposite arrows. When `require_unlocked=true`, locked indices in
`q` are excluded from the target set.
"""
function _matching_targets(q::AbstractTLArray; require_unlocked::Bool=false)
    return Set(change_dir(idx) for idx in inds(q)
               if !require_unlocked || idx.lock == 0)
end

_has_target_match(idx::TLIndex, targets; require_unlocked::Bool=false) =
    (!require_unlocked || idx.lock == 0) && (idx in targets)

"""
    _find_matching_legs(a, b; ..., matched=true, require_unlocked=false) -> Vector{Int}

Shared implementation for matching/unmatching and contractable leg queries.

`a` supplies returned visible leg indices and `b` supplies target signatures.
`matched=true` returns legs of `a` that have a target match; `matched=false`
returns those without one. `require_unlocked=true` additionally requires
`lock == 0` on both sides. Other keyword filters are applied to candidate legs
of `a` with `_matches_criteria`.
"""
function _find_matching_legs(a::AbstractTLArray{T, QD}, b::AbstractTLArray;
                             dir=nothing, itag=nothing, plev=nothing,
                             lock=nothing, rev::Bool=false,
                             matched::Bool=true,
                             require_unlocked::Bool=false) where {T, QD}
    targets = _matching_targets(b; require_unlocked=require_unlocked)
    ainds = inds(a)
    return [i for i in 1:QD
            if (_has_target_match(ainds[i], targets; require_unlocked=require_unlocked) == matched) &&
               (_matches_criteria(ainds[i]; dir=dir, itag=itag,
                                  plev=plev, lock=lock) ⊻ rev)]
end

"""
    _find_matching_leg(a, b; ..., matched=true, require_unlocked=false)

Return the first leg selected by `_find_matching_legs`, or `nothing`.

This helper exists so public `matching`, `unmatching`, `contractable`, and
`uncontractable` methods do not allocate a full vector when only the first match
is needed.
"""
function _find_matching_leg(a::AbstractTLArray{T, QD}, b::AbstractTLArray;
                            dir=nothing, itag=nothing, plev=nothing,
                            lock=nothing, rev::Bool=false,
                            matched::Bool=true,
                            require_unlocked::Bool=false) where {T, QD}
    targets = _matching_targets(b; require_unlocked=require_unlocked)
    ainds = inds(a)
    for i in 1:QD
        (_has_target_match(ainds[i], targets; require_unlocked=require_unlocked) == matched) || continue
        _matches_criteria(ainds[i]; dir=dir, itag=itag, plev=plev, lock=lock) ⊻ rev && return i
    end
    return nothing
end

"""
    matchings(a::TLArray, b::TLArray; dir=nothing, itag=nothing, plev=nothing, lock=nothing, rev=false) -> Vector{Int}

Return all leg indices of `a` that have at least one matching leg in `b`.

A leg matches when it has the same `itags`, `plev`, and `dual` flag as a leg
of `b`, but with opposite direction. Lock is ignored for the cross-tensor
match test. Keyword arguments filter the returned legs of `a` using the same
rules as `findlegs`.
"""
function matchings(a::AbstractTLArray{T, QD}, b::AbstractTLArray;
                   dir=nothing, itag=nothing, plev=nothing,
                   lock=nothing, rev::Bool=false) where {T, QD}
    return _find_matching_legs(a, b; dir=dir, itag=itag, plev=plev, lock=lock,
                               rev=rev, matched=true)
end

"""
    matching(a::TLArray, b::TLArray; dir=nothing, itag=nothing, plev=nothing, lock=nothing, rev=false) -> Union{Int, Nothing}

Return the first leg index of `a` that has a matching leg in `b`, or `nothing`
if no such leg exists.

Matching ignores lock between tensors; keyword arguments filter the returned
leg of `a` using the same rules as `findleg`.
"""
function matching(a::AbstractTLArray{T, QD}, b::AbstractTLArray;
                  dir=nothing, itag=nothing, plev=nothing,
                  lock=nothing, rev::Bool=false) where {T, QD}
    return _find_matching_leg(a, b; dir=dir, itag=itag, plev=plev, lock=lock,
                              rev=rev, matched=true)
end

"""
    unmatchings(a::TLArray, b::TLArray; dir=nothing, itag=nothing, plev=nothing, lock=nothing, rev=false) -> Vector{Int}

Return all leg indices of `a` that do not have any matching leg in `b`.

The match test uses the same rule as `matchings`: same `itags`, `plev`, and
`dual`, opposite direction, and lock ignored between tensors. Keyword
arguments filter the returned legs of `a` using the same rules as `findlegs`.
"""
function unmatchings(a::AbstractTLArray{T, QD}, b::AbstractTLArray;
                     dir=nothing, itag=nothing, plev=nothing,
                     lock=nothing, rev::Bool=false) where {T, QD}
    return _find_matching_legs(a, b; dir=dir, itag=itag, plev=plev, lock=lock,
                               rev=rev, matched=false)
end

"""
    unmatching(a::TLArray, b::TLArray; dir=nothing, itag=nothing, plev=nothing, lock=nothing, rev=false) -> Union{Int, Nothing}

Return the first leg index of `a` that does not have any matching leg in `b`,
or `nothing` if every leg is matched.

The match test uses the same rule as `matchings`: same `itags`, `plev`, and
`dual`, opposite direction, and lock ignored between tensors. Keyword
arguments filter the returned leg of `a` using the same rules as `findleg`.
"""
function unmatching(a::AbstractTLArray{T, QD}, b::AbstractTLArray;
                    dir=nothing, itag=nothing, plev=nothing,
                    lock=nothing, rev::Bool=false) where {T, QD}
    return _find_matching_leg(a, b; dir=dir, itag=itag, plev=plev, lock=lock,
                              rev=rev, matched=false)
end

"""
    contractables(a::TLArray, b::TLArray; dir=nothing, itag=nothing, plev=nothing, lock=nothing, rev=false) -> Vector{Int}

Return all leg indices of `a` that have at least one contractable leg in `b`.

Two legs are contractable when they satisfy the same cross-tensor match rule as
`matchings` and both legs have `lock == 0`. Keyword arguments filter the
returned legs of `a` using the same rules as `findlegs`.
"""
function contractables(a::AbstractTLArray{T, QD}, b::AbstractTLArray;
                       dir=nothing, itag=nothing, plev=nothing,
                       lock=nothing, rev::Bool=false) where {T, QD}
    return _find_matching_legs(a, b; dir=dir, itag=itag, plev=plev, lock=lock,
                               rev=rev, matched=true, require_unlocked=true)
end

"""
    contractable(a::TLArray, b::TLArray; dir=nothing, itag=nothing, plev=nothing, lock=nothing, rev=false) -> Union{Int, Nothing}

Return the first leg index of `a` that has a contractable leg in `b`, or
`nothing` if no such leg exists.

Contractable legs satisfy the same cross-tensor match rule as `matchings`, but
both tensors must have `lock == 0` on the matched legs.
"""
function contractable(a::AbstractTLArray{T, QD}, b::AbstractTLArray;
                      dir=nothing, itag=nothing, plev=nothing,
                      lock=nothing, rev::Bool=false) where {T, QD}
    return _find_matching_leg(a, b; dir=dir, itag=itag, plev=plev, lock=lock,
                              rev=rev, matched=true, require_unlocked=true)
end

"""
    uncontractables(a::TLArray, b::TLArray; dir=nothing, itag=nothing, plev=nothing, lock=nothing, rev=false) -> Vector{Int}

Return all leg indices of `a` that do not have any contractable leg in `b`.

Contractability requires the same cross-tensor match rule as `matchings`, plus
`lock == 0` on both matched legs. Keyword arguments filter the returned legs of
`a` using the same rules as `findlegs`.
"""
function uncontractables(a::AbstractTLArray{T, QD}, b::AbstractTLArray;
                         dir=nothing, itag=nothing, plev=nothing,
                         lock=nothing, rev::Bool=false) where {T, QD}
    return _find_matching_legs(a, b; dir=dir, itag=itag, plev=plev, lock=lock,
                               rev=rev, matched=false, require_unlocked=true)
end

"""
    uncontractable(a::TLArray, b::TLArray; dir=nothing, itag=nothing, plev=nothing, lock=nothing, rev=false) -> Union{Int, Nothing}

Return the first leg index of `a` that does not have any contractable leg in
`b`, or `nothing` if every eligible leg is contractable.

Contractability requires the same cross-tensor match rule as `matchings`, plus
`lock == 0` on both matched legs.
"""
function uncontractable(a::AbstractTLArray{T, QD}, b::AbstractTLArray;
                        dir=nothing, itag=nothing, plev=nothing,
                        lock=nothing, rev::Bool=false) where {T, QD}
    return _find_matching_leg(a, b; dir=dir, itag=itag, plev=plev, lock=lock,
                              rev=rev, matched=false, require_unlocked=true)
end

# ─── Lock utilities ──────────────────────────────────────────────────────────
#
# Lock level semantics:
#   lock > 0  : Cannot be contracted. Decreases by 1 after each contraction.
#   lock == 0 : Normal leg, can be contracted.
#   lock == -1: Permanently locked, never changes and cannot be contracted.
#
# ─────────────────────────────────────────────────────────────────────────────

const LegList = Union{AbstractVector{<:Integer}, Tuple{Vararg{Integer}}}

# Internal: apply a lock modification function to selected leg indices.
# legs can be any iterable of integers (Int, Vector, UnitRange, Tuple, etc.)
"""
    _modify_lock(q::TLArray, legs, modify_fn) -> TLArray

Return a tensor with modified lock metadata on selected visible legs.

`legs` is any iterable of leg indices. `modify_fn(old_lock)` computes the new
lock value for each selected leg. RMT, qlabel, w-matrix, spaces, and view state
are preserved by reconstructing only the index tuple.
"""
function _modify_lock(q::TLArray{T, QD, N, RD}, legs, modify_fn::Function) where {T, QD, N, RD}
    new_inds = collect(inds(q))
    for i in legs
        idx = new_inds[i]
        new_lock = modify_fn(idx.lock)
        new_inds[i] = TLIndex(idx.itags, idx.dir, idx.plev, new_lock, idx.dual)
    end
    return TLArray(q, Tuple(new_inds))
end

# Lock increase function (respects permanent lock)
_lock_inc(current_lock, inc) = current_lock == -1 ? -1 : current_lock + inc

"""
    lock(q::TLArray, leg::Integer; inc::Int=1)

Increase lock level of a single specified leg by `inc` (default 1).
Permanently locked legs (lock=-1) are unchanged.
"""
function Base.lock(q::TLArray, leg::Integer; inc::Int=1)
    return _modify_lock(q, (leg,), lk -> _lock_inc(lk, inc))
end

"""
    lock(q::TLArray, legs::LegList; inc::Int=1)

Increase lock level of the specified legs by `inc` (default 1).
`legs` can be any vector, range, or tuple of integers, e.g. `[1, 3]`, `1:3`, or `(1, 3)`.
Permanently locked legs (lock=-1) are unchanged.
"""
function Base.lock(q::TLArray, legs::LegList; inc::Int=1)
    return _modify_lock(q, legs, lk -> _lock_inc(lk, inc))
end

"""
    lock(q::TLArray, pred::Function; inc::Int=1)

Increase lock level of legs satisfying predicate by `inc` (default 1).
"""
function Base.lock(q::TLArray, pred::Function; inc::Int=1)
    legs = findlegs(q, pred)
    return _modify_lock(q, legs, lk -> _lock_inc(lk, inc))
end

"""
    lock(q::TLArray; inc::Int=1, dir=nothing, itag=nothing, plev=nothing, lock=nothing, rev=false)

Increase lock level of legs matching criteria by `inc`.
Use `rev=true` to act on legs that do *not* match the criteria.

# Examples
```julia
lock(q; dir='-')                  # lock all outgoing legs by 1
lock(q; inc=2, itag="bond")       # lock legs with "bond" in tag by 2
lock(q; lock=0)                   # lock all currently-unlocked legs by 1
lock(q; dir='-', rev=true)        # lock all legs that are NOT outgoing
```
"""
function Base.lock(q::TLArray; inc::Int=1, dir=nothing, itag=nothing, plev=nothing, lock=nothing, rev::Bool=false)
    legs = findlegs(q; dir=dir, itag=itag, plev=plev, lock=lock, rev=rev)
    return _modify_lock(q, legs, lk -> _lock_inc(lk, inc))
end

"""
    lockp(q::TLArray, leg::Integer)

Permanently lock a single specified leg (set lock=-1).
"""
function lockp(q::TLArray, leg::Integer)
    return _modify_lock(q, (leg,), _ -> -1)
end

"""
    lockp(q::TLArray, legs::LegList)

Permanently lock the specified legs (set lock=-1).
`legs` can be any vector, range, or tuple of integers.
"""
function lockp(q::TLArray, legs::LegList)
    return _modify_lock(q, legs, _ -> -1)
end

"""
    lockp(q::TLArray, pred::Function)

Permanently lock legs satisfying predicate.
"""
function lockp(q::TLArray, pred::Function)
    legs = findlegs(q, pred)
    return _modify_lock(q, legs, _ -> -1)
end

"""
    lockp(q::TLArray; dir=nothing, itag=nothing, plev=nothing, lock=nothing, rev=false)

Permanently lock legs matching criteria.
Use `rev=true` to act on legs that do *not* match the criteria.

# Examples
```julia
lockp(q; itag="phys")          # permanently lock physical legs
lockp(q; lock=0)               # permanently lock all currently-unlocked legs
lockp(q; itag="phys", rev=true)   # permanently lock all legs except "phys"
```
"""
function lockp(q::TLArray; dir=nothing, itag=nothing, plev=nothing, lock=nothing, rev::Bool=false)
    legs = findlegs(q; dir=dir, itag=itag, plev=plev, lock=lock, rev=rev)
    return _modify_lock(q, legs, _ -> -1)
end

"""
    unlock(q::TLArray, leg::Integer)

Unlock a single specified leg (set lock=0). Also removes permanent lock.
"""
function Base.unlock(q::TLArray, leg::Integer)
    return _modify_lock(q, (leg,), _ -> 0)
end

"""
    unlock(q::TLArray, legs::LegList)

Unlock the specified legs (set lock=0).
`legs` can be any vector, range, or tuple of integers.
"""
function Base.unlock(q::TLArray, legs::LegList)
    return _modify_lock(q, legs, _ -> 0)
end

"""
    unlock(q::TLArray, pred::Function)

Unlock legs satisfying predicate.
"""
function Base.unlock(q::TLArray, pred::Function)
    legs = findlegs(q, pred)
    return _modify_lock(q, legs, _ -> 0)
end

"""
    unlock(q::TLArray; dir=nothing, itag=nothing, plev=nothing, lock=nothing, rev=false)

Unlock legs matching criteria.
Use `rev=true` to act on legs that do *not* match the criteria.

# Examples
```julia
unlock(q; dir='-')             # unlock all outgoing legs
unlock(q; lock=1)              # unlock all legs currently at lock=1
unlock(q; dir='-', rev=true)   # unlock all legs that are NOT outgoing
```
"""
function Base.unlock(q::TLArray; dir=nothing, itag=nothing, plev=nothing, lock=nothing, rev::Bool=false)
    legs = findlegs(q; dir=dir, itag=itag, plev=plev, lock=lock, rev=rev)
    return _modify_lock(q, legs, _ -> 0)
end

# ─── Prime level modification ────────────────────────────────────────────────
#
# ITensor-style prime level modification. Prime level is always non-negative.
# All increment forms use `inc` as an optional keyword argument (default 1).
#
#   prime(q; inc=1, kw...)         – increase plev of selected legs by inc
#   prime(q, leg; inc=1)           – increase plev of a single leg by inc
#   prime(q, legs; inc=1)          – increase plev of specified legs by inc
#   prime(q, pred; inc=1)          – increase plev of legs matching predicate
#   setprime(q, n; kw...)          – set plev of selected legs to n (≥0)
#   setprime(q, legs, n)           – set plev of specified legs to n
#   noprime(q; kw...)              – set plev of selected legs to 0
#   noprime(q, leg)                – set plev of a single leg to 0
#   noprime(q, legs)               – set plev of specified legs to 0
#   noprime(q, pred)               – set plev of legs matching predicate to 0
#
# Keyword selectors for criteria forms (all optional): dir, itag, plev, lock
# ─────────────────────────────────────────────────────────────────────────────

# legs can be any iterable of integers
"""
    _modify_plev(q::TLArray, legs, modify_fn) -> TLArray

Return a tensor with modified prime levels on selected visible legs.

`modify_fn(old_plev)` computes the replacement prime level. Callers enforce
non-negativity or clamping. Only index metadata changes; sector storage is
preserved.
"""
function _modify_plev(q::TLArray{T, QD, N, RD}, legs, modify_fn::Function) where {T, QD, N, RD}
    new_inds = collect(inds(q))
    for i in legs
        idx = new_inds[i]
        new_inds[i] = TLIndex(idx.itags, idx.dir, modify_fn(idx.plev), idx.lock, idx.dual)
    end
    return TLArray(q, Tuple(new_inds))
end

"""
    prime(q::TLArray; inc::Int=1, dir, itag, plev, lock, rev=false)

Increase the prime level of matching legs by `inc` (default 1).
Prime level is clamped to 0 from below.
With no keyword arguments, all legs are affected.
Use `rev=true` to act on legs that do *not* match the criteria.

# Examples
```julia
prime(q)                        # all legs: plev += 1
prime(q; inc=2)                 # all legs: plev += 2
prime(q; inc=-1)                # all legs: plev -= 1, clamped to 0
prime(q; dir='+')               # incoming legs only
prime(q; itag="site")           # legs whose tag contains "site"
prime(q; plev=0)                # only currently unprimed legs
prime(q; dir='+', rev=true)     # all legs that are NOT incoming
```
"""
function prime(q::TLArray{T, QD}; inc::Int=1, dir=nothing, itag=nothing, plev=nothing, lock=nothing, rev::Bool=false) where {T, QD}
    legs = findlegs(q; dir=dir, itag=itag, plev=plev, lock=lock, rev=rev)
    return _modify_plev(q, legs, p -> max(0, p + inc))
end

"""
    prime(q::TLArray, leg::Integer; inc::Int=1)

Increase the prime level of a single specified leg by `inc` (default 1).
Prime level is clamped to 0 from below.

# Examples
```julia
prime(q, 2)                     # leg 2: plev += 1
prime(q, 2; inc=3)              # leg 2: plev += 3
```
"""
function prime(q::TLArray{T, QD}, leg::Integer; inc::Int=1) where {T, QD}
    return _modify_plev(q, (leg,), p -> max(0, p + inc))
end

"""
    prime(q::TLArray, legs::LegList; inc::Int=1)

Increase the prime level of the specified legs by `inc` (default 1).
Prime level is clamped to 0 from below.
`legs` can be any vector, range, or tuple, e.g. `[1, 3]`, `1:3`, or `(1, 3)`.

# Examples
```julia
prime(q, [1, 3])                # legs 1 and 3: plev += 1
prime(q, 1:2; inc=2)            # legs 1–2: plev += 2
prime(q, (1, 3))                # same as prime(q, [1, 3])
```
"""
function prime(q::TLArray{T, QD}, legs::LegList; inc::Int=1) where {T, QD}
    return _modify_plev(q, legs, p -> max(0, p + inc))
end

"""
    prime(q::TLArray, pred::Function; inc::Int=1)

Increase the prime level of legs satisfying predicate by `inc` (default 1).
Prime level is clamped to 0 from below.

# Examples
```julia
prime(q, idx -> idx.dir == '+')          # incoming legs: plev += 1
prime(q, idx -> idx.plev == 0; inc=2)   # unprimed legs: plev += 2
```
"""
function prime(q::TLArray{T, QD}, pred::Function; inc::Int=1) where {T, QD}
    legs = findlegs(q, pred)
    return _modify_plev(q, legs, p -> max(0, p + inc))
end

"""
    setprime(q::TLArray, n::Int; dir, itag, plev, lock, rev=false)

Set the prime level of matching legs to `n`. `n` must be non-negative.
Use `rev=true` to act on legs that do *not* match the criteria.

# Examples
```julia
setprime(q, 2)                  # set all legs to plev=2
setprime(q, 1; dir='-')         # set outgoing legs to plev=1
setprime(q, 0; itag="bond")     # same as noprime(q; itag="bond")
setprime(q, 0; dir='-', rev=true)  # clear prime on all non-outgoing legs
```
"""
function setprime(q::TLArray{T, QD}, n::Int; dir=nothing, itag=nothing, plev=nothing, lock=nothing, rev::Bool=false) where {T, QD}
    n >= 0 || throw(ArgumentError("prime level must be non-negative, got $n"))
    legs = findlegs(q; dir=dir, itag=itag, plev=plev, lock=lock, rev=rev)
    return _modify_plev(q, legs, _ -> n)
end

"""
    setprime(q::TLArray, legs, n::Int)

Set the prime level of the specified legs to `n`. `n` must be non-negative.
`legs` can be any iterable of leg indices.

# Examples
```julia
setprime(q, [1, 3], 2)          # set legs 1 and 3 to plev=2
setprime(q, 1:2, 0)             # same as noprime(q, 1:2)
```
"""
function setprime(q::TLArray{T, QD}, legs, n::Int) where {T, QD}
    n >= 0 || throw(ArgumentError("prime level must be non-negative, got $n"))
    return _modify_plev(q, legs, _ -> n)
end

"""
    noprime(q::TLArray; dir, itag, plev, lock, rev=false)

Set the prime level of matching legs to 0.
Use `rev=true` to act on legs that do *not* match the criteria.

# Examples
```julia
noprime(q)                      # clear prime on all legs
noprime(q; dir='+')             # clear prime on incoming legs only
noprime(q; plev=1)              # clear all legs currently at plev=1
noprime(q; dir='+', rev=true)   # clear prime on all non-incoming legs
```
"""
function noprime(q::TLArray{T, QD}; dir=nothing, itag=nothing, plev=nothing, lock=nothing, rev::Bool=false) where {T, QD}
    legs = findlegs(q; dir=dir, itag=itag, plev=plev, lock=lock, rev=rev)
    return _modify_plev(q, legs, _ -> 0)
end

"""
    noprime(q::TLArray, leg::Integer)

Set the prime level of a single specified leg to 0.
"""
function noprime(q::TLArray{T, QD}, leg::Integer) where {T, QD}
    return _modify_plev(q, (leg,), _ -> 0)
end

"""
    noprime(q::TLArray, legs::LegList)

Set the prime level of the specified legs to 0.
`legs` can be any vector, range, or tuple, e.g. `[1, 3]`, `1:3`, or `(1, 3)`.
"""
function noprime(q::TLArray{T, QD}, legs::LegList) where {T, QD}
    return _modify_plev(q, legs, _ -> 0)
end

"""
    noprime(q::TLArray, pred::Function)

Set the prime level of legs satisfying predicate to 0.
"""
function noprime(q::TLArray{T, QD}, pred::Function) where {T, QD}
    legs = findlegs(q, pred)
    return _modify_plev(q, legs, _ -> 0)
end

# ─── Tag modification ────────────────────────────────────────────────────────
#
# ITensor-style tag manipulation on TLArray legs.
# Tags are stored as sorted, comma-separated strings (e.g. "bond,site").
#
#   additag(q, newtags; kw...)     – add tag(s) to matching legs
#   removeitag(q, tags; kw...)     – remove tag(s) from matching legs
#   replaceitag(q, old=>new...)    – replace tag queries on matching legs
#   setitag(q, tags; kw...)        – replace entire tag string of matching legs
#
# Keyword selectors (all optional): dir, itag, plev, lock
# ─────────────────────────────────────────────────────────────────────────────

const ITagQuerySpec = Union{AbstractString,
                            Tuple{Vararg{AbstractString}},
                            AbstractVector{<:AbstractString}}
const ITagReplacementPair = Pair{<:AbstractString, <:AbstractString}
const ITagReplacementDict = AbstractDict{<:AbstractString, <:AbstractString}

"""
    _modify_itag(q::TLArray, legs, modify_fn) -> TLArray

Return a tensor with modified tag strings on selected visible legs.

`modify_fn(old_itags)` returns the normalized replacement tag string used to
construct each updated `TLIndex`. Numerical and sector metadata are unchanged.
"""
function _modify_itag(q::TLArray{T, QD, N, RD}, legs, modify_fn::Function) where {T, QD, N, RD}
    new_inds = collect(inds(q))
    for i in legs
        idx = new_inds[i]
        new_inds[i] = TLIndex(modify_fn(idx.itags), idx.dir, idx.plev, idx.lock, idx.dual)
    end
    return TLArray(q, Tuple(new_inds))
end

"""
    additag(q::TLArray, newtags; dir, itag, plev, lock, rev=false)

Add one or more tags to matching legs. `newtags` may be a comma-separated
string of tags (e.g. `"bond,u1"`). The result is always sorted.
Use `rev=true` to act on legs that do *not* match the criteria.

# Examples
```julia
additag(q, "site")              # add "site" to all legs
additag(q, "phys"; dir='+')     # add "phys" to incoming legs only
additag(q, "u1"; itag="bond")   # add "u1" to legs that already have "bond"
additag(q, "aux"; dir='+', rev=true)  # add "aux" to all non-incoming legs
```
"""
function additag(q::TLArray{T, QD}, newtags::AbstractString; dir=nothing, itag=nothing, plev=nothing, lock=nothing, rev::Bool=false) where {T, QD}
    legs = findlegs(q; dir=dir, itag=itag, plev=plev, lock=lock, rev=rev)
    return _modify_itag(q, legs, base -> _add_itag(base, newtags))
end

"""    additag(q::TLArray, leg::Integer, newtags)

Add tags to a single specified leg.
"""
function additag(q::TLArray{T, QD}, leg::Integer, newtags::AbstractString) where {T, QD}
    return _modify_itag(q, (leg,), base -> _add_itag(base, newtags))
end

"""    additag(q::TLArray, legs::LegList, newtags)

Add tags to the specified legs. `legs` can be any vector, range, or tuple.
"""
function additag(q::TLArray{T, QD}, legs::LegList, newtags::AbstractString) where {T, QD}
    return _modify_itag(q, legs, base -> _add_itag(base, newtags))
end

"""    additag(q::TLArray, pred::Function, newtags)

Add tags to legs satisfying predicate.
"""
function additag(q::TLArray{T, QD}, pred::Function, newtags::AbstractString) where {T, QD}
    return _modify_itag(q, findlegs(q, pred), base -> _add_itag(base, newtags))
end

"""
    removeitag(q::TLArray, tags; dir, itag, plev, lock, rev=false)

Remove one or more tags from matching legs.

`tags` may be either a single comma-separated tag string or a tuple/vector of
such strings. For tuple/vector input, each query is applied independently and
its tags are removed only from legs that contain all tags in that query.
Use `rev=true` to act on legs that do *not* match the criteria.

# Examples
```julia
removeitag(q, "site")                     # remove "site" from all legs
removeitag(q, ("aaa,bbb", "ccc"))       # remove aaa+bbb together, and ccc independently
removeitag(q, "phys"; dir='-')           # remove "phys" from outgoing legs
removeitag(q, "aux"; dir='-', rev=true)  # remove "aux" from all non-outgoing legs
```
"""
function removeitag(q::TLArray{T, QD}, tags::ITagQuerySpec; dir=nothing, itag=nothing, plev=nothing, lock=nothing, rev::Bool=false) where {T, QD}
    legs = findlegs(q; dir=dir, itag=itag, plev=plev, lock=lock, rev=rev)
    return _modify_itag(q, legs, base -> _remove_itag(base, tags))
end

"""    removeitag(q::TLArray, leg::Integer, tags)

Remove tags from a single specified leg.
"""
function removeitag(q::TLArray{T, QD}, leg::Integer, tags::ITagQuerySpec) where {T, QD}
    return _modify_itag(q, (leg,), base -> _remove_itag(base, tags))
end

"""    removeitag(q::TLArray, legs::LegList, tags)

Remove tags from the specified legs. `legs` can be any vector, range, or tuple.
"""
function removeitag(q::TLArray{T, QD}, legs::LegList, tags::ITagQuerySpec) where {T, QD}
    return _modify_itag(q, legs, base -> _remove_itag(base, tags))
end

"""    removeitag(q::TLArray, pred::Function, tags)

Remove tags from legs satisfying predicate.
"""
function removeitag(q::TLArray{T, QD}, pred::Function, tags::ITagQuerySpec) where {T, QD}
    return _modify_itag(q, findlegs(q, pred), base -> _remove_itag(base, tags))
end

"""
    replaceitag(q::TLArray, replacements::Pair...; dir, itag, plev, lock, rev=false)
    replaceitag(q::TLArray, replacements::AbstractDict; dir, itag, plev, lock, rev=false)

Replace tags on matching legs. Each replacement removes the tags in the source
query and adds the destination tags when the leg contains all tags in the source
query. For example, `"aaa,bbb"=>"ccc"` removes both `aaa` and `bbb` and adds
`ccc` only on legs containing both source tags.

Dictionary input applies each `key => value` replacement in iteration order.
Use `rev=true` to act on legs that do *not* match the criteria.

# Examples
```julia
replaceitag(q, "site"=>"phys")
replaceitag(q, "aaa,bbb"=>"ccc")
replaceitag(q, "site1"=>"left", "site2"=>"right"; dir='-')
replaceitag(q, Dict("site"=>"phys"))
```
"""
function replaceitag(q::TLArray{T, QD}, replacements::ITagReplacementPair...;
                     dir=nothing, itag=nothing, plev=nothing, lock=nothing,
                     rev::Bool=false) where {T, QD}
    isempty(replacements) &&
        throw(ArgumentError("replaceitag requires at least one replacement pair"))
    legs = findlegs(q; dir=dir, itag=itag, plev=plev, lock=lock, rev=rev)
    return _modify_itag(q, legs, base -> _replace_itag(base, replacements))
end

function replaceitag(q::TLArray{T, QD}, replacements::ITagReplacementDict;
                     dir=nothing, itag=nothing, plev=nothing, lock=nothing,
                     rev::Bool=false) where {T, QD}
    legs = findlegs(q; dir=dir, itag=itag, plev=plev, lock=lock, rev=rev)
    return _modify_itag(q, legs, base -> _replace_itag(base, replacements))
end

"""    replaceitag(q::TLArray, leg::Integer, replacements)

Replace tags on a single specified leg.
"""
function replaceitag(q::TLArray{T, QD}, leg::Integer,
                     replacements::ITagReplacementPair...) where {T, QD}
    isempty(replacements) &&
        throw(ArgumentError("replaceitag requires at least one replacement pair"))
    return _modify_itag(q, (leg,), base -> _replace_itag(base, replacements))
end

function replaceitag(q::TLArray{T, QD}, leg::Integer,
                     replacements::ITagReplacementDict) where {T, QD}
    return _modify_itag(q, (leg,), base -> _replace_itag(base, replacements))
end

"""    replaceitag(q::TLArray, legs::LegList, replacements)

Replace tags on the specified legs. `legs` can be any vector, range, or tuple.
"""
function replaceitag(q::TLArray{T, QD}, legs::LegList,
                     replacements::ITagReplacementPair...) where {T, QD}
    isempty(replacements) &&
        throw(ArgumentError("replaceitag requires at least one replacement pair"))
    return _modify_itag(q, legs, base -> _replace_itag(base, replacements))
end

function replaceitag(q::TLArray{T, QD}, legs::LegList,
                     replacements::ITagReplacementDict) where {T, QD}
    return _modify_itag(q, legs, base -> _replace_itag(base, replacements))
end

"""    replaceitag(q::TLArray, pred::Function, replacements)

Replace tags on legs satisfying predicate.
"""
function replaceitag(q::TLArray{T, QD}, pred::Function,
                     replacements::ITagReplacementPair...) where {T, QD}
    isempty(replacements) &&
        throw(ArgumentError("replaceitag requires at least one replacement pair"))
    return _modify_itag(q, findlegs(q, pred), base -> _replace_itag(base, replacements))
end

function replaceitag(q::TLArray{T, QD}, pred::Function,
                     replacements::ITagReplacementDict) where {T, QD}
    return _modify_itag(q, findlegs(q, pred), base -> _replace_itag(base, replacements))
end

"""
    setitag(q::TLArray, tags; dir, itag, plev, lock, rev=false)

Replace the entire tag string of matching legs with `tags`.
Use `rev=true` to act on legs that do *not* match the criteria.

# Examples
```julia
setitag(q, "bond")              # set all legs to tag "bond"
setitag(q, ""; dir='+')         # clear tags on incoming legs
setitag(q, "phys"; itag="site") # rename "site" → "phys" (full replacement)
setitag(q, "aux"; dir='+', rev=true)  # set tag on all non-incoming legs
```
"""
function setitag(q::TLArray{T, QD}, tags::AbstractString; dir=nothing, itag=nothing, plev=nothing, lock=nothing, rev::Bool=false) where {T, QD}
    legs = findlegs(q; dir=dir, itag=itag, plev=plev, lock=lock, rev=rev)
    norm = _normalize_itag(tags)
    return _modify_itag(q, legs, _ -> norm)
end

"""    setitag(q::TLArray, leg::Integer, tags)

Set the entire tag string of a single specified leg.
"""
function setitag(q::TLArray{T, QD}, leg::Integer, tags::AbstractString) where {T, QD}
    norm = _normalize_itag(tags)
    return _modify_itag(q, (leg,), _ -> norm)
end

"""    setitag(q::TLArray, legs::LegList, tags)

Set the entire tag string of the specified legs. `legs` can be any vector, range, or tuple.
"""
function setitag(q::TLArray{T, QD}, legs::LegList, tags::AbstractString) where {T, QD}
    norm = _normalize_itag(tags)
    return _modify_itag(q, legs, _ -> norm)
end

"""    setitag(q::TLArray, pred::Function, tags)

Set the entire tag string of legs satisfying predicate.
"""
function setitag(q::TLArray{T, QD}, pred::Function, tags::AbstractString) where {T, QD}
    norm = _normalize_itag(tags)
    return _modify_itag(q, findlegs(q, pred), _ -> norm)
end

"""
    _contraction_rewrap(q::TLArrayContraction, inds) -> TLArrayContraction

Rebuild a lazy contraction with a replacement visible index tuple.

`inds` must have the same rank as `q`. All contraction work items, source
tensors, sector storage, view state, buffers, and locks are reused. This is the
metadata-only path used by index mutation methods on lazy contractions.
"""
function _contraction_rewrap(q::TLArrayContraction{T, QD, N, RD, QT, PS, M, RMT},
                             inds::NTuple{QD, TLIndex}) where {T, QD, N, RD, QT, PS, M, RMT}
    return TLArrayContraction{T, QD, N, RD, QT, PS, M, RMT}(
        stored_qlabels(q), q.wmatdata, q.wmatinfo, q.RMTs, q.isdefined, q.iszero,
        inds, spaces(q), stored_conj(q), stored_scale(q), stored_perm(q),
        q.arr1, q.arr2, q.work_items, q.factors, q.perm1, q.perm2,
        q.rmt_sizes, q.lock)
end

_contraction_rewrap(q::TLArrayContraction, ref::AbstractTLArray) =
    _contraction_rewrap(q, inds(ref))

"""
    _modify_lock(q::TLArrayContraction, legs, modify_fn) -> TLArrayContraction
    _modify_plev(q::TLArrayContraction, legs, modify_fn) -> TLArrayContraction
    _modify_itag(q::TLArrayContraction, legs, modify_fn) -> TLArrayContraction

Metadata-only index mutation helpers for lazy contractions.

`legs` selects visible output legs of the contraction. `modify_fn` is applied to
the selected lock level, prime level, or tag string. The returned contraction
reuses all work items, source tensors, sector metadata, RMT buffers, and lazy
state, changing only the visible index tuple.
"""
function _modify_lock(q::TLArrayContraction{T, QD}, legs, modify_fn::Function) where {T, QD}
    new_inds = collect(inds(q))
    for i in legs
        idx = new_inds[i]
        new_inds[i] = TLIndex(idx.itags, idx.dir, idx.plev, modify_fn(idx.lock), idx.dual)
    end
    return _contraction_rewrap(q, Tuple(new_inds))
end

function _modify_plev(q::TLArrayContraction{T, QD}, legs, modify_fn::Function) where {T, QD}
    new_inds = collect(inds(q))
    for i in legs
        idx = new_inds[i]
        new_inds[i] = TLIndex(idx.itags, idx.dir, modify_fn(idx.plev), idx.lock, idx.dual)
    end
    return _contraction_rewrap(q, Tuple(new_inds))
end

function _modify_itag(q::TLArrayContraction{T, QD}, legs, modify_fn::Function) where {T, QD}
    new_inds = collect(inds(q))
    for i in legs
        idx = new_inds[i]
        new_inds[i] = TLIndex(modify_fn(idx.itags), idx.dir, idx.plev, idx.lock, idx.dual)
    end
    return _contraction_rewrap(q, Tuple(new_inds))
end

"""
    lock/lockp/unlock/prime/setprime/noprime/additag/removeitag/replaceitag/setitag(q::TLArrayContraction, ...)

Apply the ordinary TLArray index-metadata operations to a lazy contraction.

These methods use the same argument forms and keyword selectors as the concrete
`TLArray` methods. They do not compute contraction sectors or alter numerical
storage; only the contraction's visible index metadata is rewrapped.
"""
Base.lock(q::TLArrayContraction, leg::Integer; inc::Int=1) =
    _modify_lock(q, (leg,), lk -> _lock_inc(lk, inc))
Base.lock(q::TLArrayContraction, legs::LegList; inc::Int=1) =
    _modify_lock(q, legs, lk -> _lock_inc(lk, inc))
Base.lock(q::TLArrayContraction, pred::Function; inc::Int=1) =
    _modify_lock(q, findlegs(q, pred), lk -> _lock_inc(lk, inc))
function Base.lock(q::TLArrayContraction; inc::Int=1, dir=nothing, itag=nothing,
                   plev=nothing, lock=nothing, rev::Bool=false)
    legs = findlegs(q; dir=dir, itag=itag, plev=plev, lock=lock, rev=rev)
    return _modify_lock(q, legs, lk -> _lock_inc(lk, inc))
end

lockp(q::TLArrayContraction, leg::Integer) = _modify_lock(q, (leg,), _ -> -1)
lockp(q::TLArrayContraction, legs::LegList) = _modify_lock(q, legs, _ -> -1)
lockp(q::TLArrayContraction, pred::Function) =
    _modify_lock(q, findlegs(q, pred), _ -> -1)
function lockp(q::TLArrayContraction; dir=nothing, itag=nothing, plev=nothing,
               lock=nothing, rev::Bool=false)
    legs = findlegs(q; dir=dir, itag=itag, plev=plev, lock=lock, rev=rev)
    return _modify_lock(q, legs, _ -> -1)
end

Base.unlock(q::TLArrayContraction, leg::Integer) = _modify_lock(q, (leg,), _ -> 0)
Base.unlock(q::TLArrayContraction, legs::LegList) = _modify_lock(q, legs, _ -> 0)
Base.unlock(q::TLArrayContraction, pred::Function) =
    _modify_lock(q, findlegs(q, pred), _ -> 0)
function Base.unlock(q::TLArrayContraction; dir=nothing, itag=nothing, plev=nothing,
                     lock=nothing, rev::Bool=false)
    legs = findlegs(q; dir=dir, itag=itag, plev=plev, lock=lock, rev=rev)
    return _modify_lock(q, legs, _ -> 0)
end

function prime(q::TLArrayContraction; inc::Int=1, dir=nothing, itag=nothing,
               plev=nothing, lock=nothing, rev::Bool=false)
    legs = findlegs(q; dir=dir, itag=itag, plev=plev, lock=lock, rev=rev)
    return _modify_plev(q, legs, p -> max(0, p + inc))
end
prime(q::TLArrayContraction, leg::Integer; inc::Int=1) =
    _modify_plev(q, (leg,), p -> max(0, p + inc))
prime(q::TLArrayContraction, legs::LegList; inc::Int=1) =
    _modify_plev(q, legs, p -> max(0, p + inc))
prime(q::TLArrayContraction, pred::Function; inc::Int=1) =
    _modify_plev(q, findlegs(q, pred), p -> max(0, p + inc))

function setprime(q::TLArrayContraction, n::Int; dir=nothing, itag=nothing,
                  plev=nothing, lock=nothing, rev::Bool=false)
    n >= 0 || throw(ArgumentError("prime level must be non-negative, got $n"))
    legs = findlegs(q; dir=dir, itag=itag, plev=plev, lock=lock, rev=rev)
    return _modify_plev(q, legs, _ -> n)
end
function setprime(q::TLArrayContraction, leg::Integer, n::Int)
    n >= 0 || throw(ArgumentError("prime level must be non-negative, got $n"))
    return _modify_plev(q, (leg,), _ -> n)
end
function setprime(q::TLArrayContraction, legs, n::Int)
    n >= 0 || throw(ArgumentError("prime level must be non-negative, got $n"))
    return _modify_plev(q, legs, _ -> n)
end

function noprime(q::TLArrayContraction; dir=nothing, itag=nothing, plev=nothing,
                 lock=nothing, rev::Bool=false)
    legs = findlegs(q; dir=dir, itag=itag, plev=plev, lock=lock, rev=rev)
    return _modify_plev(q, legs, _ -> 0)
end
noprime(q::TLArrayContraction, leg::Integer) = _modify_plev(q, (leg,), _ -> 0)
noprime(q::TLArrayContraction, legs::LegList) = _modify_plev(q, legs, _ -> 0)
noprime(q::TLArrayContraction, pred::Function) =
    _modify_plev(q, findlegs(q, pred), _ -> 0)

function additag(q::TLArrayContraction, newtags::AbstractString; dir=nothing,
                 itag=nothing, plev=nothing, lock=nothing, rev::Bool=false)
    legs = findlegs(q; dir=dir, itag=itag, plev=plev, lock=lock, rev=rev)
    return _modify_itag(q, legs, base -> _add_itag(base, newtags))
end
additag(q::TLArrayContraction, leg::Integer, newtags::AbstractString) =
    _modify_itag(q, (leg,), base -> _add_itag(base, newtags))
additag(q::TLArrayContraction, legs::LegList, newtags::AbstractString) =
    _modify_itag(q, legs, base -> _add_itag(base, newtags))
additag(q::TLArrayContraction, pred::Function, newtags::AbstractString) =
    _modify_itag(q, findlegs(q, pred), base -> _add_itag(base, newtags))

function removeitag(q::TLArrayContraction, tags::ITagQuerySpec; dir=nothing,
                    itag=nothing, plev=nothing, lock=nothing, rev::Bool=false)
    legs = findlegs(q; dir=dir, itag=itag, plev=plev, lock=lock, rev=rev)
    return _modify_itag(q, legs, base -> _remove_itag(base, tags))
end
removeitag(q::TLArrayContraction, leg::Integer, tags::ITagQuerySpec) =
    _modify_itag(q, (leg,), base -> _remove_itag(base, tags))
removeitag(q::TLArrayContraction, legs::LegList, tags::ITagQuerySpec) =
    _modify_itag(q, legs, base -> _remove_itag(base, tags))
removeitag(q::TLArrayContraction, pred::Function, tags::ITagQuerySpec) =
    _modify_itag(q, findlegs(q, pred), base -> _remove_itag(base, tags))

function replaceitag(q::TLArrayContraction, replacements::ITagReplacementPair...;
                     dir=nothing, itag=nothing, plev=nothing, lock=nothing,
                     rev::Bool=false)
    isempty(replacements) &&
        throw(ArgumentError("replaceitag requires at least one replacement pair"))
    legs = findlegs(q; dir=dir, itag=itag, plev=plev, lock=lock, rev=rev)
    return _modify_itag(q, legs, base -> _replace_itag(base, replacements))
end
function replaceitag(q::TLArrayContraction, replacements::ITagReplacementDict;
                     dir=nothing, itag=nothing, plev=nothing, lock=nothing,
                     rev::Bool=false)
    legs = findlegs(q; dir=dir, itag=itag, plev=plev, lock=lock, rev=rev)
    return _modify_itag(q, legs, base -> _replace_itag(base, replacements))
end
function replaceitag(q::TLArrayContraction, leg::Integer,
                     replacements::ITagReplacementPair...)
    isempty(replacements) &&
        throw(ArgumentError("replaceitag requires at least one replacement pair"))
    return _modify_itag(q, (leg,), base -> _replace_itag(base, replacements))
end
replaceitag(q::TLArrayContraction, leg::Integer,
            replacements::ITagReplacementDict) =
    _modify_itag(q, (leg,), base -> _replace_itag(base, replacements))
function replaceitag(q::TLArrayContraction, legs::LegList,
                     replacements::ITagReplacementPair...)
    isempty(replacements) &&
        throw(ArgumentError("replaceitag requires at least one replacement pair"))
    return _modify_itag(q, legs, base -> _replace_itag(base, replacements))
end
replaceitag(q::TLArrayContraction, legs::LegList,
            replacements::ITagReplacementDict) =
    _modify_itag(q, legs, base -> _replace_itag(base, replacements))
function replaceitag(q::TLArrayContraction, pred::Function,
                     replacements::ITagReplacementPair...)
    isempty(replacements) &&
        throw(ArgumentError("replaceitag requires at least one replacement pair"))
    return _modify_itag(q, findlegs(q, pred), base -> _replace_itag(base, replacements))
end
replaceitag(q::TLArrayContraction, pred::Function,
            replacements::ITagReplacementDict) =
    _modify_itag(q, findlegs(q, pred), base -> _replace_itag(base, replacements))

function setitag(q::TLArrayContraction, tags::AbstractString; dir=nothing,
                 itag=nothing, plev=nothing, lock=nothing, rev::Bool=false)
    legs = findlegs(q; dir=dir, itag=itag, plev=plev, lock=lock, rev=rev)
    norm = _normalize_itag(tags)
    return _modify_itag(q, legs, _ -> norm)
end
function setitag(q::TLArrayContraction, leg::Integer, tags::AbstractString)
    norm = _normalize_itag(tags)
    return _modify_itag(q, (leg,), _ -> norm)
end
function setitag(q::TLArrayContraction, legs::LegList, tags::AbstractString)
    norm = _normalize_itag(tags)
    return _modify_itag(q, legs, _ -> norm)
end
function setitag(q::TLArrayContraction, pred::Function, tags::AbstractString)
    norm = _normalize_itag(tags)
    return _modify_itag(q, findlegs(q, pred), _ -> norm)
end

"""
    _singleton_rewrap(q::AddSingletonTLArray, inds) -> AddSingletonTLArray
    _singleton_rewrap(q::DeleteSingletonTLArray, inds) -> DeleteSingletonTLArray

Rebuild a lazy singleton wrapper with replacement visible indices.

`inds` must match the wrapper rank. All source references, inserted/deleted leg
maps, qlabels, spaces, and view state are preserved. This is used by
metadata-only lock, prime-level, and tag operations on singleton lazy wrappers.
"""
function _singleton_rewrap(q::AddSingletonTLArray{T, QD, N, RD, QT, PS, M, RMT},
                           inds::NTuple{QD, TLIndex}) where {T, QD, N, RD, QT, PS, M, RMT}
    return AddSingletonTLArray{T, QD, N, RD, QT, PS, M, RMT}(
        stored_qlabels(q), inds, stored_spaces(q), stored_conj(q), stored_scale(q),
        stored_perm(q), q.arr, q.inserted_legs, q.source_to_result_legs,
        q.result_to_source_legs)
end

function _singleton_rewrap(q::DeleteSingletonTLArray{T, QD, N, RD, QT, PS, M, RMT},
                           inds::NTuple{QD, TLIndex}) where {T, QD, N, RD, QT, PS, M, RMT}
    return DeleteSingletonTLArray{T, QD, N, RD, QT, PS, M, RMT}(
        stored_qlabels(q), inds, stored_spaces(q), stored_conj(q), stored_scale(q),
        stored_perm(q), q.arr, q.deleted_legs, q.source_to_result_legs,
        q.result_to_source_legs)
end

"""
    _modify_lock(q::SingletonTLArray, legs, modify_fn) -> SingletonTLArray
    _modify_plev(q::SingletonTLArray, legs, modify_fn) -> SingletonTLArray
    _modify_itag(q::SingletonTLArray, legs, modify_fn) -> SingletonTLArray

Metadata-only index mutation helpers for lazy singleton insertion/deletion.

`legs` selects visible wrapper legs. The helper rewrites only the visible
`TLIndex` tuple through `_singleton_rewrap`; source tensor storage and
source/result leg maps are left untouched.
"""
function _modify_lock(q::SingletonTLArray, legs, modify_fn::Function)
    new_inds = collect(inds(q))
    for i in legs
        idx = new_inds[i]
        new_inds[i] = TLIndex(idx.itags, idx.dir, idx.plev, modify_fn(idx.lock), idx.dual)
    end
    return _singleton_rewrap(q, Tuple(new_inds))
end

function _modify_plev(q::SingletonTLArray, legs, modify_fn::Function)
    new_inds = collect(inds(q))
    for i in legs
        idx = new_inds[i]
        new_inds[i] = TLIndex(idx.itags, idx.dir, modify_fn(idx.plev), idx.lock, idx.dual)
    end
    return _singleton_rewrap(q, Tuple(new_inds))
end

function _modify_itag(q::SingletonTLArray, legs, modify_fn::Function)
    new_inds = collect(inds(q))
    for i in legs
        idx = new_inds[i]
        new_inds[i] = TLIndex(modify_fn(idx.itags), idx.dir, idx.plev, idx.lock, idx.dual)
    end
    return _singleton_rewrap(q, Tuple(new_inds))
end

"""
    lock/lockp/unlock/prime/setprime/noprime/additag/removeitag/replaceitag/setitag(q::SingletonTLArray, ...)

Apply TLArray index-metadata operations to lazy singleton wrappers.

The argument forms mirror the concrete `TLArray` methods. These methods do not
materialize the singleton operation or touch RMT payloads; they only update the
wrapper's visible `TLIndex` metadata.
"""
Base.lock(q::SingletonTLArray, leg::Integer; inc::Int=1) =
    _modify_lock(q, (leg,), lk -> _lock_inc(lk, inc))
Base.lock(q::SingletonTLArray, legs::LegList; inc::Int=1) =
    _modify_lock(q, legs, lk -> _lock_inc(lk, inc))
Base.lock(q::SingletonTLArray, pred::Function; inc::Int=1) =
    _modify_lock(q, findlegs(q, pred), lk -> _lock_inc(lk, inc))
function Base.lock(q::SingletonTLArray; inc::Int=1, dir=nothing, itag=nothing,
                   plev=nothing, lock=nothing, rev::Bool=false)
    legs = findlegs(q; dir=dir, itag=itag, plev=plev, lock=lock, rev=rev)
    return _modify_lock(q, legs, lk -> _lock_inc(lk, inc))
end

lockp(q::SingletonTLArray, leg::Integer) = _modify_lock(q, (leg,), _ -> -1)
lockp(q::SingletonTLArray, legs::LegList) = _modify_lock(q, legs, _ -> -1)
lockp(q::SingletonTLArray, pred::Function) =
    _modify_lock(q, findlegs(q, pred), _ -> -1)
function lockp(q::SingletonTLArray; dir=nothing, itag=nothing, plev=nothing,
               lock=nothing, rev::Bool=false)
    legs = findlegs(q; dir=dir, itag=itag, plev=plev, lock=lock, rev=rev)
    return _modify_lock(q, legs, _ -> -1)
end

Base.unlock(q::SingletonTLArray, leg::Integer) = _modify_lock(q, (leg,), _ -> 0)
Base.unlock(q::SingletonTLArray, legs::LegList) = _modify_lock(q, legs, _ -> 0)
Base.unlock(q::SingletonTLArray, pred::Function) =
    _modify_lock(q, findlegs(q, pred), _ -> 0)
function Base.unlock(q::SingletonTLArray; dir=nothing, itag=nothing, plev=nothing,
                     lock=nothing, rev::Bool=false)
    legs = findlegs(q; dir=dir, itag=itag, plev=plev, lock=lock, rev=rev)
    return _modify_lock(q, legs, _ -> 0)
end

function prime(q::SingletonTLArray; inc::Int=1, dir=nothing, itag=nothing,
               plev=nothing, lock=nothing, rev::Bool=false)
    legs = findlegs(q; dir=dir, itag=itag, plev=plev, lock=lock, rev=rev)
    return _modify_plev(q, legs, p -> max(0, p + inc))
end
prime(q::SingletonTLArray, leg::Integer; inc::Int=1) =
    _modify_plev(q, (leg,), p -> max(0, p + inc))
prime(q::SingletonTLArray, legs::LegList; inc::Int=1) =
    _modify_plev(q, legs, p -> max(0, p + inc))
prime(q::SingletonTLArray, pred::Function; inc::Int=1) =
    _modify_plev(q, findlegs(q, pred), p -> max(0, p + inc))

function setprime(q::SingletonTLArray, n::Int; dir=nothing, itag=nothing,
                  plev=nothing, lock=nothing, rev::Bool=false)
    n >= 0 || throw(ArgumentError("prime level must be non-negative, got $n"))
    legs = findlegs(q; dir=dir, itag=itag, plev=plev, lock=lock, rev=rev)
    return _modify_plev(q, legs, _ -> n)
end
function setprime(q::SingletonTLArray, leg::Integer, n::Int)
    n >= 0 || throw(ArgumentError("prime level must be non-negative, got $n"))
    return _modify_plev(q, (leg,), _ -> n)
end
function setprime(q::SingletonTLArray, legs, n::Int)
    n >= 0 || throw(ArgumentError("prime level must be non-negative, got $n"))
    return _modify_plev(q, legs, _ -> n)
end

function noprime(q::SingletonTLArray; dir=nothing, itag=nothing, plev=nothing,
                 lock=nothing, rev::Bool=false)
    legs = findlegs(q; dir=dir, itag=itag, plev=plev, lock=lock, rev=rev)
    return _modify_plev(q, legs, _ -> 0)
end
noprime(q::SingletonTLArray, leg::Integer) = _modify_plev(q, (leg,), _ -> 0)
noprime(q::SingletonTLArray, legs::LegList) = _modify_plev(q, legs, _ -> 0)
noprime(q::SingletonTLArray, pred::Function) =
    _modify_plev(q, findlegs(q, pred), _ -> 0)

function additag(q::SingletonTLArray, newtags::AbstractString; dir=nothing,
                 itag=nothing, plev=nothing, lock=nothing, rev::Bool=false)
    legs = findlegs(q; dir=dir, itag=itag, plev=plev, lock=lock, rev=rev)
    return _modify_itag(q, legs, base -> _add_itag(base, newtags))
end
additag(q::SingletonTLArray, leg::Integer, newtags::AbstractString) =
    _modify_itag(q, (leg,), base -> _add_itag(base, newtags))
additag(q::SingletonTLArray, legs::LegList, newtags::AbstractString) =
    _modify_itag(q, legs, base -> _add_itag(base, newtags))
additag(q::SingletonTLArray, pred::Function, newtags::AbstractString) =
    _modify_itag(q, findlegs(q, pred), base -> _add_itag(base, newtags))

function removeitag(q::SingletonTLArray, tags::ITagQuerySpec; dir=nothing,
                    itag=nothing, plev=nothing, lock=nothing, rev::Bool=false)
    legs = findlegs(q; dir=dir, itag=itag, plev=plev, lock=lock, rev=rev)
    return _modify_itag(q, legs, base -> _remove_itag(base, tags))
end
removeitag(q::SingletonTLArray, leg::Integer, tags::ITagQuerySpec) =
    _modify_itag(q, (leg,), base -> _remove_itag(base, tags))
removeitag(q::SingletonTLArray, legs::LegList, tags::ITagQuerySpec) =
    _modify_itag(q, legs, base -> _remove_itag(base, tags))
removeitag(q::SingletonTLArray, pred::Function, tags::ITagQuerySpec) =
    _modify_itag(q, findlegs(q, pred), base -> _remove_itag(base, tags))

function replaceitag(q::SingletonTLArray, replacements::ITagReplacementPair...;
                     dir=nothing, itag=nothing, plev=nothing, lock=nothing,
                     rev::Bool=false)
    isempty(replacements) &&
        throw(ArgumentError("replaceitag requires at least one replacement pair"))
    legs = findlegs(q; dir=dir, itag=itag, plev=plev, lock=lock, rev=rev)
    return _modify_itag(q, legs, base -> _replace_itag(base, replacements))
end
function replaceitag(q::SingletonTLArray, replacements::ITagReplacementDict;
                     dir=nothing, itag=nothing, plev=nothing, lock=nothing,
                     rev::Bool=false)
    legs = findlegs(q; dir=dir, itag=itag, plev=plev, lock=lock, rev=rev)
    return _modify_itag(q, legs, base -> _replace_itag(base, replacements))
end
function replaceitag(q::SingletonTLArray, leg::Integer,
                     replacements::ITagReplacementPair...)
    isempty(replacements) &&
        throw(ArgumentError("replaceitag requires at least one replacement pair"))
    return _modify_itag(q, (leg,), base -> _replace_itag(base, replacements))
end
replaceitag(q::SingletonTLArray, leg::Integer,
            replacements::ITagReplacementDict) =
    _modify_itag(q, (leg,), base -> _replace_itag(base, replacements))
function replaceitag(q::SingletonTLArray, legs::LegList,
                     replacements::ITagReplacementPair...)
    isempty(replacements) &&
        throw(ArgumentError("replaceitag requires at least one replacement pair"))
    return _modify_itag(q, legs, base -> _replace_itag(base, replacements))
end
replaceitag(q::SingletonTLArray, legs::LegList,
            replacements::ITagReplacementDict) =
    _modify_itag(q, legs, base -> _replace_itag(base, replacements))
function replaceitag(q::SingletonTLArray, pred::Function,
                     replacements::ITagReplacementPair...)
    isempty(replacements) &&
        throw(ArgumentError("replaceitag requires at least one replacement pair"))
    return _modify_itag(q, findlegs(q, pred), base -> _replace_itag(base, replacements))
end
replaceitag(q::SingletonTLArray, pred::Function,
            replacements::ITagReplacementDict) =
    _modify_itag(q, findlegs(q, pred), base -> _replace_itag(base, replacements))

function setitag(q::SingletonTLArray, tags::AbstractString; dir=nothing,
                 itag=nothing, plev=nothing, lock=nothing, rev::Bool=false)
    legs = findlegs(q; dir=dir, itag=itag, plev=plev, lock=lock, rev=rev)
    norm = _normalize_itag(tags)
    return _modify_itag(q, legs, _ -> norm)
end
function setitag(q::SingletonTLArray, leg::Integer, tags::AbstractString)
    norm = _normalize_itag(tags)
    return _modify_itag(q, (leg,), _ -> norm)
end
function setitag(q::SingletonTLArray, legs::LegList, tags::AbstractString)
    norm = _normalize_itag(tags)
    return _modify_itag(q, legs, _ -> norm)
end
function setitag(q::SingletonTLArray, pred::Function, tags::AbstractString)
    norm = _normalize_itag(tags)
    return _modify_itag(q, findlegs(q, pred), _ -> norm)
end

include("tlarray/display.jl")
include("tlarray/algebra.jl")
include("tlarray/oplus.jl")
include("tlarray/getsub.jl")
include("tlarray/constructors.jl")

include("getLocalSpace.jl")
include("getIdentity.jl")
include("contract.jl")
include("sum_tlarray.jl")
include("get1jtensor.jl")
include("svd.jl")
include("qr.jl")
include("eig.jl")
include("permute.jl")
include("lazy.jl")
include("fuselegs.jl")
