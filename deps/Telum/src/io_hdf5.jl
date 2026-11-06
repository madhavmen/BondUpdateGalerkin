const _TLARRAY_HDF5_SCHEMA_VERSION = "2.0"

_h5_attrs(obj) = HDF5.attrs(obj)

function _h5_set_attr!(obj, name::AbstractString, value)
    _h5_attrs(obj)[name] = value
    return obj
end

_h5_attr(obj, name::AbstractString) = _h5_attrs(obj)[name]
_h5_attr_string(obj, name::AbstractString) = String(_h5_attr(obj, name))

"""
    _package_version_string(mod::Module) -> String

Return the package version string recorded in TLArray HDF5 metadata.

If `Base.pkgversion(mod)` is unavailable or throws, `"unknown"` is returned.
This keeps file writing robust for checked-out development packages.
"""
function _package_version_string(mod::Module)
    try
        v = Base.pkgversion(mod)
        return v === nothing ? "unknown" : string(v)
    catch
        return "unknown"
    end
end

"""
    _eltype_tag(T) -> String
    _eltype_from_tag(tag::AbstractString) -> Type

Map supported RMT element types to stable HDF5 schema tags and back.

Only `Float64`, `Float32`, `ComplexF64`, and `ComplexF32` are supported. Other
types are rejected explicitly so files do not silently encode unserializable
Julia-specific element types.
"""
_eltype_tag(::Type{Float64}) = "Float64"
_eltype_tag(::Type{Float32}) = "Float32"
_eltype_tag(::Type{ComplexF64}) = "ComplexF64"
_eltype_tag(::Type{ComplexF32}) = "ComplexF32"
_eltype_tag(::Type{T}) where {T} =
    throw(ArgumentError("unsupported TLArray HDF5 RMT element type $T"))

function _eltype_from_tag(tag::AbstractString)
    tag == "Float64" && return Float64
    tag == "Float32" && return Float32
    tag == "ComplexF64" && return ComplexF64
    tag == "ComplexF32" && return ComplexF32
    throw(ArgumentError("unsupported TLArray HDF5 RMT element type $tag"))
end

"""
    _symmetry_family_parameter(::Type{S}) -> (String, Int)
    _symmetry_from_family_parameter(family, parameter) -> Type{<:Symmetry}

Convert symmetry types to stable HDF5 family/parameter metadata and back.

Parameterized families such as `SU{N}`, `SO{N}`, `Sp{N}`, and `Z{N}` store `N`
as `parameter`. Unparameterized families such as `U1` and `G2` store parameter
zero.
"""
function _symmetry_family_parameter(::Type{S}) where {S<:Symmetry}
    S === U1 && return ("U1", 0)
    S === LurCGT.G2 && return ("G2", 0)
    S <: Z && return ("Z", Int(S.parameters[1]))
    S <: SU && return ("SU", Int(S.parameters[1]))
    S <: SO && return ("SO", Int(S.parameters[1]))
    S <: Sp && return ("Sp", Int(S.parameters[1]))
    throw(ArgumentError("unsupported symmetry type $S for TLArray HDF5 I/O"))
end

function _symmetry_from_family_parameter(family::AbstractString, parameter::Integer)
    family == "U1" && return U1
    family == "G2" && return LurCGT.G2
    family == "Z" && return Z{Int(parameter)}
    family == "SU" && return SU{Int(parameter)}
    family == "SO" && return SO{Int(parameter)}
    family == "Sp" && return Sp{Int(parameter)}
    throw(ArgumentError("unsupported symmetry family $family in TLArray HDF5 file"))
end

"""
    _write_symmetries!(parent, symm) -> HDF5.Group
    _read_symmetries(parent) -> Tuple

Write and read the product-symmetry tuple in TLArray HDF5 files.

The stored group records the number of symmetry factors and one child group per
factor with family, parameter, and Abelian metadata. Reading reconstructs the
symmetry type tuple from those stable tags.
"""
function _write_symmetries!(parent, symm::NTuple{N, Any}) where {N}
    g = HDF5.create_group(parent, "symmetries")
    g["count"] = N
    for n in 1:N
        sg = HDF5.create_group(g, string(n))
        family, parameter = _symmetry_family_parameter(symm[n])
        _h5_set_attr!(sg, "family", family)
        _h5_set_attr!(sg, "parameter", parameter)
        _h5_set_attr!(sg, "abelian", isabelian(symm[n]))
    end
    return g
end

function _read_symmetries(parent)
    g = parent["symmetries"]
    N = Int(read(g["count"]))
    return ntuple(Val(N)) do n
        sg = g[string(n)]
        family = _h5_attr_string(sg, "family")
        parameter = Int(_h5_attr(sg, "parameter"))
        _symmetry_from_family_parameter(family, parameter)
    end
end

"""
    _qlabel_width(symm) -> Int

Return the flattened integer width of one product-symmetry qlabel.

The width is the sum of `nzops` over every symmetry factor and determines row
counts in HDF5 qlabel matrices.
"""
@inline _qlabel_width(symm) = sum(n -> nzops(symm[n]), eachindex(symm); init=0)

"""
    _encode_oneleg_qlabels(qlabels, symm) -> Matrix{Int}
    _decode_oneleg_qlabels(data, symm) -> Vector

Flatten and reconstruct qlabels for one leg space list.

Columns correspond to space-list entries. Rows concatenate the qlabel
coordinates for every symmetry factor. Decoding validates the expected row
count and reconstructs the concrete `qlabeltype(symm)`.
"""
function _encode_oneleg_qlabels(qlabels::AbstractVector, symm)
    width = _qlabel_width(symm)
    data = Matrix{Int}(undef, width, length(qlabels))
    for sector in eachindex(qlabels)
        row = 1
        q = qlabels[sector]
        for n in eachindex(symm)
            qn = q[n]
            for k in 1:nzops(symm[n])
                data[row, sector] = Int(qn[k])
                row += 1
            end
        end
    end
    return data
end

function _decode_oneleg_qlabels(data::AbstractMatrix{<:Integer}, symm)
    width = _qlabel_width(symm)
    size(data, 1) == width ||
        throw(ArgumentError("qlabel matrix row count $(size(data, 1)) does not match expected width $width"))
    QT = qlabeltype(symm)
    result = Vector{QT}(undef, size(data, 2))
    for sector in axes(data, 2)
        row = 1
        q = ntuple(Val(length(symm))) do n
            qn = ntuple(k -> Int(data[row + k - 1, sector]), Val(nzops(symm[n])))
            row += nzops(symm[n])
            qn
        end
        result[sector] = q
    end
    return result
end

"""
    _encode_sector_qlabels(qlabels, symm) -> Matrix{Int}
    _decode_sector_qlabels(data, symm, ::Val{QD}) -> Vector

Flatten and reconstruct full per-sector qlabel tuples.

Rows are grouped by visible leg, and within each leg by symmetry-factor qlabel
coordinates. Columns correspond to sector slots. Decoding validates
`QD * _qlabel_width(symm)` rows before rebuilding `NTuple{QD,QT}` sector labels.
"""
function _encode_sector_qlabels(qlabels::AbstractVector{<:NTuple{QD}}, symm) where {QD}
    width = _qlabel_width(symm)
    data = Matrix{Int}(undef, QD * width, length(qlabels))
    for sector in eachindex(qlabels)
        for leg in 1:QD
            row0 = (leg - 1) * width
            row = row0 + 1
            q = qlabels[sector][leg]
            for n in eachindex(symm)
                qn = q[n]
                for k in 1:nzops(symm[n])
                    data[row, sector] = Int(qn[k])
                    row += 1
                end
            end
        end
    end
    return data
end

function _decode_sector_qlabels(data::AbstractMatrix{<:Integer}, symm, ::Val{QD}) where {QD}
    width = _qlabel_width(symm)
    size(data, 1) == QD * width ||
        throw(ArgumentError("sector qlabel matrix row count $(size(data, 1)) does not match expected $(QD * width)"))
    QT = qlabeltype(symm)
    result = Vector{NTuple{QD, QT}}(undef, size(data, 2))
    for sector in axes(data, 2)
        result[sector] = ntuple(Val(QD)) do leg
            row = (leg - 1) * width + 1
            q = ntuple(Val(length(symm))) do n
                qn = ntuple(k -> Int(data[row + k - 1, sector]), Val(nzops(symm[n])))
                row += nzops(symm[n])
                qn
            end
            q
        end
    end
    return result
end

"""
    _write_indices!(parent, inds) -> HDF5.Group
    _read_indices(parent, ::Val{QD}) -> NTuple{QD,TLIndex}

Write and read visible TLIndex metadata.

Stored arrays contain normalized tags, direction bytes, prime levels, lock
levels, and dual flags. Reading checks that every metadata array has length
`QD` before reconstructing `TLIndex` values.
"""
function _write_indices!(parent, inds::NTuple{QD, TLIndex}) where {QD}
    g = HDF5.create_group(parent, "indices")
    g["itags"] = String[String(idx.itags) for idx in inds]
    g["dir"] = UInt8[UInt8(idx.dir) for idx in inds]
    g["plev"] = Int[idx.plev for idx in inds]
    g["lock"] = Int[idx.lock for idx in inds]
    g["dual"] = Bool[idx.dual for idx in inds]
    return g
end

function _read_indices(parent, ::Val{QD}) where {QD}
    g = parent["indices"]
    itags = Vector{String}(read(g["itags"]))
    dirs = Vector{UInt8}(read(g["dir"]))
    plev = Vector{Int}(read(g["plev"]))
    lock = Vector{Int}(read(g["lock"]))
    duals = Vector{Bool}(read(g["dual"]))
    length(itags) == QD && length(dirs) == QD && length(plev) == QD &&
        length(lock) == QD && length(duals) == QD ||
        throw(ArgumentError("index metadata length does not match tensor rank $QD"))
    return ntuple(i -> TLIndex(itags[i], Char(dirs[i]), plev[i], lock[i], duals[i]), Val(QD))
end

"""
    _write_spaces!(parent, spaces, symm) -> HDF5.Group
    _read_spaces(parent, symm, ::Val{QD}) -> NTuple

Write and read TLArray visible leg space lists.

For each leg, qlabels are stored through `_encode_oneleg_qlabels` and RMT
dimensions are stored as an integer vector. Reading reconstructs
`Vector{Tuple{QT,Int}}` entries for each leg and checks qlabel/dimension length
agreement.
"""
function _write_spaces!(parent, spaces::NTuple{QD, <:AbstractVector}, symm) where {QD}
    g = HDF5.create_group(parent, "spaces")
    for leg in 1:QD
        lg = HDF5.create_group(g, string(leg))
        entries = spaces[leg]
        lg["qlabels"] = _encode_oneleg_qlabels([entry[1] for entry in entries], symm)
        lg["dims"] = Int[entry[2] for entry in entries]
    end
    return g
end

function _read_spaces(parent, symm, ::Val{QD}) where {QD}
    g = parent["spaces"]
    QT = qlabeltype(symm)
    return ntuple(Val(QD)) do leg
        lg = g[string(leg)]
        qlabels = _decode_oneleg_qlabels(read(lg["qlabels"]), symm)
        dims = Vector{Int}(read(lg["dims"]))
        length(qlabels) == length(dims) ||
            throw(ArgumentError("space qlabel/dimension length mismatch on leg $leg"))
        Tuple{QT, Int}[(qlabels[i], dims[i]) for i in eachindex(dims)]
    end
end

"""
    _encode_wmatinfo(wmatinfo) -> Vector{Int}
    _decode_wmatinfo(data, sector_count::Int, ::Val{M}) -> Vector{WMatInfo{M}}

Flatten and reconstruct w-matrix arena descriptors.

Each descriptor stores `(offset, nrows, ncols)` for each non-Abelian symmetry
slot in each sector. The encoded vector has length `3 * M * sector_count`, which
is checked during decoding.
"""
function _encode_wmatinfo(wmatinfo::Vector{WMatInfo{M}}) where {M}
    data = Vector{Int}(undef, 3 * M * length(wmatinfo))
    p = 1
    for sector in eachindex(wmatinfo), slot in 1:M, item in 1:3
        data[p] = wmatinfo[sector][slot][item]
        p += 1
    end
    return data
end

function _decode_wmatinfo(data::AbstractVector{<:Integer}, sector_count::Int, ::Val{M}) where {M}
    expected = 3 * M * sector_count
    length(data) == expected ||
        throw(ArgumentError("wmat info length $(length(data)) does not match expected $expected"))
    wmatinfo = Vector{WMatInfo{M}}(undef, sector_count)
    p = 1
    for sector in 1:sector_count
        wmatinfo[sector] = ntuple(Val(M)) do _
            info = (Int(data[p]), Int(data[p + 1]), Int(data[p + 2]))
            p += 3
            info
        end
    end
    return wmatinfo
end

"""
    _write_wmat_storage!(parent, q::TLArray) -> HDF5.Group
    _read_wmat_storage(parent, sector_count::Int, ::Val{M})

Write and read contiguous w-matrix arena storage.

The `data` dataset stores the raw `Float64` arena and `info` stores flattened
`WMatInfo` descriptors. Attributes record tuple width and sector count so schema
mismatches can be detected while reading.
"""
function _write_wmat_storage!(parent, q::TLArray{T, QD, N, RD, QT, PS, M}) where {T, QD, N, RD, QT, PS, M}
    g = HDF5.create_group(parent, "wmat")
    g["data"] = q.wmatdata
    g["info"] = _encode_wmatinfo(q.wmatinfo)
    _h5_set_attr!(g, "tuple_width", M)
    _h5_set_attr!(g, "sector_count", sector_count(q))
    return g
end

"""
    _read_wmat_storage(parent, sector_count::Int, ::Val{M}) -> (data, info)

Read the w-matrix arena for a concrete `TLArray` from an HDF5 parent object.

`parent` is the group that owns the `"wmat"` subgroup. `sector_count` is the
number of sector slots already decoded from sector metadata and is checked
against the stored attribute before allocation. `Val{M}` carries the number of
non-Abelian symmetry slots per sector so that decoded `WMatInfo{M}` entries have
the same tuple width as the tensor type being reconstructed.

Returns the contiguous `Vector{Float64}` arena and the decoded per-sector
descriptor vector. This routine deliberately does not inspect RMT state; sector
slot numbering is validated separately by sector metadata and RMT readers.
"""
function _read_wmat_storage(parent, sector_count::Int, ::Val{M}) where {M}
    g = parent["wmat"]
    Int(_h5_attr(g, "tuple_width")) == M ||
        throw(ArgumentError("wmat tuple width does not match symmetry metadata"))
    Int(_h5_attr(g, "sector_count")) == sector_count ||
        throw(ArgumentError("wmat sector count does not match sector metadata"))
    return Vector{Float64}(read(g["data"])), _decode_wmatinfo(read(g["info"]), sector_count, Val(M))
end

"""
    _validate_concrete_tlarray_for_hdf5(q::TLArray) -> Union{Type,Nothing}

Check that an eager concrete tensor can be serialized without changing its
sector semantics.

`q` is the source tensor. The function verifies that `RMTs`, `isdefined`, and
`iszero` have one entry per sector slot; that every sector marked defined has an
assigned RMT payload; that every undefined sector is explicitly marked zero; and
that all defined RMT payloads share one concrete storage type. The returned
value is that concrete RMT type, or `nothing` when no sectors are defined.

The check mirrors Telum's eager-sector invariant and prevents writing ambiguous
lazy/view-like state into the compact HDF5 arena format.
"""
function _validate_concrete_tlarray_for_hdf5(q::TLArray)
    nslots = sector_count(q)
    length(q.RMTs) == nslots && length(q.isdefined) == nslots &&
        length(q.iszero) == nslots ||
        throw(ArgumentError("inconsistent TLArray sector storage lengths"))
    rmt_type = nothing
    for sector in sector_slots(q)
        if q.isdefined[sector]
            isassigned(q.RMTs, sector) ||
                throw(ArgumentError("sector $sector is marked defined but has no assigned RMT"))
            current = typeof(q.RMTs[sector])
            rmt_type === nothing || current === rmt_type ||
                throw(ArgumentError("all defined RMTs in one TLArray must have the same concrete type"))
            rmt_type = current
        else
            q.iszero[sector] ||
                throw(ArgumentError("undefined concrete TLArray sector $sector must be marked zero"))
        end
    end
    return rmt_type
end

"""
    _rmt_storage_tag(::Type{RMT}) -> String

Map a concrete RMT payload type to the HDF5 storage backend tag.

Dense Julia arrays use `"dense_arena"` and `DiagRMT` values use `"diag_arena"`.
Any other `RMT` type is rejected because the reader only knows how to rebuild
these two arena layouts. The tag is stored both on the tensor group and on the
RMT subgroup so schema mismatches can be diagnosed locally while reading.
"""
@inline _rmt_storage_tag(::Type{<:Array}) = "dense_arena"
@inline _rmt_storage_tag(::Type{<:DiagRMT}) = "diag_arena"
@inline _rmt_storage_tag(::Type{RMT}) where {RMT} =
    throw(ArgumentError("unsupported RMT storage type $RMT for TLArray HDF5 I/O"))

"""
    _write_dense_rmt_arena!(g, q::TLArray) -> HDF5.Group

Write dense RMT payloads into a single flat arena.

`g` is the already-created `"rmt"` HDF5 group. `q` is a concrete `TLArray` whose
defined sector payloads are dense `Array{T,RD}` values. Defined sectors append
`vec(rmt)` to the `data` dataset and store `(offset, length, axis sizes...)` in
the `info` matrix at the same sector slot. Undefined sectors leave an all-zero
metadata column, preserving sector slot numbering rather than compacting active
sectors.
"""
function _write_dense_rmt_arena!(g, q::TLArray{T, QD, N, RD, QT, PS, M, RMT}) where {T, QD, N, RD, QT, PS, M, RMT}
    data = eltype(RMT)[]
    info = zeros(Int, 2 + RD, sector_count(q))
    for sector in sector_slots(q)
        q.isdefined[sector] || continue
        rmt = q.RMTs[sector]
        offset = length(data) + 1
        append!(data, vec(rmt))
        len = length(rmt)
        info[1, sector] = offset
        info[2, sector] = len
        for axis in 1:RD
            info[2 + axis, sector] = size(rmt, axis)
        end
    end
    g["data"] = data
    g["info"] = info
    return g
end

"""
    _write_diag_rmt_arena!(g, q::TLArray) -> HDF5.Group

Write diagonal RMT payloads into a compact arena.

`g` is the `"rmt"` HDF5 group. `q` is a concrete tensor whose defined sector
payloads are `DiagRMT` values. For each defined sector, `diag` entries are
appended to `data`, while `info[:, sector]` records the arena offset, diagonal
length, and the two dense axes represented by the diagonal storage. Undefined
sectors keep zero metadata columns so that HDF5 sector indices match TLArray
sector slots exactly.
"""
function _write_diag_rmt_arena!(g, q::TLArray{T, QD, N, RD, QT, PS, M, RMT}) where {T, QD, N, RD, QT, PS, M, RMT}
    data = eltype(RMT)[]
    info = zeros(Int, 4, sector_count(q))
    for sector in sector_slots(q)
        q.isdefined[sector] || continue
        rmt = q.RMTs[sector]
        offset = length(data) + 1
        append!(data, rmt.diag)
        info[1, sector] = offset
        info[2, sector] = length(rmt.diag)
        info[3, sector] = rmt.axis[1]
        info[4, sector] = rmt.axis[2]
    end
    g["data"] = data
    g["info"] = info
    return g
end

"""
    _write_rmt_storage!(parent, q::TLArray) -> HDF5.Group

Create and populate the `"rmt"` subgroup for a concrete tensor.

`parent` is the tensor group being written. `q` supplies the element type,
rank-dependent RMT dimensionality, sector state bits, and concrete RMT payloads.
The method records the selected storage backend, one-based arena alignment, and
sector count, then delegates to either the dense-array or diagonal-RMT arena
writer. The selected backend is based only on the concrete RMT type parameter,
not on individual sector contents.
"""
function _write_rmt_storage!(parent, q::TLArray{T, QD, N, RD, QT, PS, M, RMT}) where {T, QD, N, RD, QT, PS, M, RMT}
    g = HDF5.create_group(parent, "rmt")
    storage = _rmt_storage_tag(RMT)
    _h5_set_attr!(g, "storage", storage)
    _h5_set_attr!(g, "alignment", 1)
    _h5_set_attr!(g, "sector_count", sector_count(q))
    if storage == "dense_arena"
        _write_dense_rmt_arena!(g, q)
    else
        _write_diag_rmt_arena!(g, q)
    end
    return g
end

"""
    _check_rmt_range(data_len, offset, len, sector) -> nothing

Validate one defined sector's slice into an RMT arena.

`data_len` is the total length of the flat arena. `offset` is the one-based
starting position stored in HDF5. `len` is the number of scalar entries assigned
to this sector. `sector` is used only for precise error messages. The function
throws if the range is empty, non-positive, or extends past `data_len`.
"""
function _check_rmt_range(data_len::Int, offset::Int, len::Int, sector::Int)
    offset > 0 || throw(ArgumentError("defined sector $sector has non-positive RMT arena offset"))
    len > 0 || throw(ArgumentError("defined sector $sector has non-positive RMT arena length"))
    offset + len - 1 <= data_len ||
        throw(ArgumentError("RMT arena range for sector $sector exceeds data length"))
    return nothing
end

"""
    _read_dense_rmts(parent, ::Type{T}, ::Val{RD}, isdefined_bits, iszero_bits)

Reconstruct dense `Array{T,RD}` RMT payload storage from an HDF5 arena.

`parent` owns the `"rmt"` group. `T` is the element type decoded from the tensor
schema. `Val{RD}` is the dense RMT rank. `isdefined_bits` selects sectors that
must be materialized from the arena. `iszero_bits` is accepted for the common
reader interface and documents that sector state is controlled externally by
metadata, not inferred from nonzero payloads.

The returned vector has one slot per sector. Only defined slots are assigned;
undefined slots remain unassigned and must also have zero arena metadata.
"""
function _read_dense_rmts(parent, ::Type{T}, ::Val{RD}, isdefined_bits::BitVector, iszero_bits::BitVector) where {T, RD}
    g = parent["rmt"]
    data = Vector{T}(read(g["data"]))
    info = Matrix{Int}(read(g["info"]))
    sector_count = length(iszero_bits)
    size(info) == (2 + RD, sector_count) ||
        throw(ArgumentError("dense RMT info shape $(size(info)) does not match expected $((2 + RD, sector_count))"))
    RMTs = Vector{Array{T, RD}}(undef, sector_count)
    for sector in 1:sector_count
        offset, len = info[1, sector], info[2, sector]
        if isdefined_bits[sector]
            _check_rmt_range(length(data), offset, len, sector)
            shape = ntuple(axis -> info[2 + axis, sector], Val(RD))
            prod(shape) == len ||
                throw(ArgumentError("dense RMT shape product does not match arena length for sector $sector"))
            RMTs[sector] = Array(reshape(copy(view(data, offset:offset + len - 1)), shape))
        else
            offset == 0 && len == 0 ||
                throw(ArgumentError("undefined sector $sector must have zero RMT arena range"))
        end
    end
    return RMTs
end

"""
    _read_diag_rmts(parent, ::Type{T}, ::Val{RD}, isdefined_bits, iszero_bits)

Reconstruct `DiagRMT{T,RD}` payload storage from an HDF5 arena.

`parent` owns the `"rmt"` group. `T` is the scalar type. `Val{RD}` carries the
rank of the dense RMT that the diagonal object represents. `isdefined_bits`
determines which sector columns contain live payloads; `iszero_bits` is part of
the shared RMT-reader signature and keeps the call aligned with sector metadata.

For each defined sector, the method reads the diagonal vector and the pair of
axes stored in `info`. Undefined sectors are required to have all-zero metadata
so stale arena ranges cannot accidentally revive a sector.
"""
function _read_diag_rmts(parent, ::Type{T}, ::Val{RD}, isdefined_bits::BitVector, iszero_bits::BitVector) where {T, RD}
    g = parent["rmt"]
    data = Vector{T}(read(g["data"]))
    info = Matrix{Int}(read(g["info"]))
    sector_count = length(iszero_bits)
    size(info) == (4, sector_count) ||
        throw(ArgumentError("diagonal RMT info shape $(size(info)) does not match expected $((4, sector_count))"))
    RMTs = Vector{DiagRMT{T, RD}}(undef, sector_count)
    for sector in 1:sector_count
        offset, len = info[1, sector], info[2, sector]
        if isdefined_bits[sector]
            _check_rmt_range(length(data), offset, len, sector)
            axis = (info[3, sector], info[4, sector])
            RMTs[sector] = DiagRMT(Vector{T}(view(data, offset:offset + len - 1)), Val(RD), axis)
        else
            all(iszero, view(info, :, sector)) ||
                throw(ArgumentError("undefined sector $sector must have zero diagonal RMT metadata"))
        end
    end
    return RMTs
end

"""
    _read_rmt_storage(parent, ::Type{T}, ::Val{RD}, isdefined_bits, iszero_bits)

Dispatch HDF5 RMT reconstruction based on the stored backend tag.

`parent` is the tensor group, `T` is the decoded raw RMT scalar type, and
`Val{RD}` is the dense RMT rank. `isdefined_bits` and `iszero_bits` come from
sector metadata and define the exact sector slot state for the returned RMT
vector. The function checks arena alignment and sector count before delegating
to the dense or diagonal reader.
"""
function _read_rmt_storage(parent, ::Type{T}, ::Val{RD}, isdefined_bits::BitVector, iszero_bits::BitVector) where {T, RD}
    g = parent["rmt"]
    storage = _h5_attr_string(g, "storage")
    Int(_h5_attr(g, "alignment")) == 1 ||
        throw(ArgumentError("unsupported RMT arena alignment in TLArray HDF5 file"))
    Int(_h5_attr(g, "sector_count")) == length(iszero_bits) ||
        throw(ArgumentError("RMT sector count does not match sector metadata"))
    storage == "dense_arena" && return _read_dense_rmts(parent, T, Val(RD), isdefined_bits, iszero_bits)
    storage == "diag_arena" && return _read_diag_rmts(parent, T, Val(RD), isdefined_bits, iszero_bits)
    throw(ArgumentError("unsupported RMT storage $storage in TLArray HDF5 file"))
end

"""
    _write_sector_metadata!(parent, q::TLArray) -> HDF5.Group
    _read_sector_metadata(parent, symmetries, ::Val{QD}) -> (qlabels, isdefined, iszero)

Write and read the sector table for a concrete tensor.

`parent` is the tensor group. `q` supplies ordered stored q-labels and sector
state bits. `symmetries` is the decoded product-symmetry tuple used to decode
serialized q-labels. `Val{QD}` is the visible tensor rank and therefore the
number of q-label entries per sector.

The q-label list, `isdefined`, and `iszero` arrays are required to have identical
lengths. During read, every undefined sector must be marked zero, preserving the
eager `TLArray` invariant that undefined concrete slots are known zero sectors.
"""
function _write_sector_metadata!(parent, q::TLArray)
    g = HDF5.create_group(parent, "sectors")
    g["qlabels"] = _encode_sector_qlabels(stored_qlabels(q), symm(q))
    g["isdefined"] = collect(Bool, q.isdefined)
    g["iszero"] = collect(Bool, q.iszero)
    return g
end

function _read_sector_metadata(parent, symmetries, ::Val{QD}) where {QD}
    g = parent["sectors"]
    qlabels = _decode_sector_qlabels(read(g["qlabels"]), symmetries, Val(QD))
    isdefined_bits = BitVector(Vector{Bool}(read(g["isdefined"])))
    iszero_bits = BitVector(Vector{Bool}(read(g["iszero"])))
    sector_count = length(qlabels)
    length(isdefined_bits) == sector_count && length(iszero_bits) == sector_count ||
        throw(ArgumentError("sector qlabel/state lengths do not match"))
    for sector in 1:sector_count
        (!isdefined_bits[sector] && !iszero_bits[sector]) &&
            throw(ArgumentError("undefined sector $sector is not marked zero"))
    end
    return qlabels, isdefined_bits, iszero_bits
end

"""
    _write_tlarray_group!(h5, name, q::TLArray) -> TLArray

Write a complete concrete `TLArray` object below `h5[name]`.

`h5` is an open HDF5 file or group handle. `name` is the child group name to
create; existing objects with the same name are rejected. `q` is the concrete
tensor to serialize. Type parameters determine the scalar element tag, visible
rank `QD`, number of symmetry factors `N`, dense RMT rank `RD`, q-label type,
number of non-Abelian w-matrix slots `M`, and RMT storage backend.

The method writes top-level writer/schema attributes, then stores symmetries,
indices, spaces, sector state, w-matrix arena data, and RMT arena data. It
returns `q` so callers can use it in fluent save paths without re-reading.
"""
function _write_tlarray_group!(h5, name::AbstractString, q::TLArray{T, QD, N, RD, QT, PS, M, RMT}) where {T, QD, N, RD, QT, PS, M, RMT}
    _validate_concrete_tlarray_for_hdf5(q)
    haskey(h5, name) && throw(ArgumentError("HDF5 object $name already exists"))

    _h5_set_attr!(h5, "telum_hdf5_schema_version", _TLARRAY_HDF5_SCHEMA_VERSION)
    _h5_set_attr!(h5, "writer_package", "Telum")
    _h5_set_attr!(h5, "writer_telum_version", _package_version_string(@__MODULE__))
    _h5_set_attr!(h5, "writer_lurcgt_version", _package_version_string(LurCGT))
    _h5_set_attr!(h5, "writer_julia_version", string(VERSION))

    g = HDF5.create_group(h5, name)
    _h5_set_attr!(g, "object_type", "TLArray")
    _h5_set_attr!(g, "schema_version", _TLARRAY_HDF5_SCHEMA_VERSION)
    _h5_set_attr!(g, "eltype", _eltype_tag(T))
    _h5_set_attr!(g, "rmt_eltype", _eltype_tag(eltype(RMT)))
    _h5_set_attr!(g, "rmt_storage", _rmt_storage_tag(RMT))
    _h5_set_attr!(g, "conj", stored_conj(q))
    _h5_set_attr!(g, "scale", stored_scale(q))
    _h5_set_attr!(g, "perm", collect(stored_perm(q)))
    _h5_set_attr!(g, "qd", QD)
    _h5_set_attr!(g, "nsymms", N)
    _h5_set_attr!(g, "rd", RD)
    _h5_set_attr!(g, "sector_count", sector_count(q))
    _h5_set_attr!(g, "nonzero_sector_count", nsectors(q))

    _write_symmetries!(g, symm(q))
    _write_indices!(g, stored_inds(q))
    _write_spaces!(g, stored_spaces(q), symm(q))
    _write_sector_metadata!(g, q)
    _write_wmat_storage!(g, q)
    _write_rmt_storage!(g, q)
    return q
end

"""
    save_tlarray(path, q; name="tensor")
    save_tlarray(h5, name, q)

Write concrete tensor `q` to HDF5 group `name`. In the path form, `path` is
opened for writing and the path is returned; in the handle form, `h5` is an
existing HDF5 file/group handle and the written group is returned. Lazy
evaluation objects are not serializable through this API.
Existing data at `name` may be replaced, so callers should choose paths/groups
carefully. The source tensor is never mutated.
"""
function save_tlarray(path::AbstractString, q::TLArray; name::AbstractString="tensor")
    HDF5.h5open(path, "w") do h5
        save_tlarray(h5, name, q)
    end
    return path
end

function save_tlarray(h5, name::AbstractString, q::TLArray)
    return _write_tlarray_group!(h5, name, q)
end

function _read_tlarray_group(parent)
    _h5_attr_string(parent, "object_type") == "TLArray" ||
        throw(ArgumentError("HDF5 group does not contain a TLArray"))
    schema_version = _h5_attr_string(parent, "schema_version")
    schema_version in ("1.0", _TLARRAY_HDF5_SCHEMA_VERSION) ||
        throw(ArgumentError("unsupported TLArray HDF5 schema version $schema_version"))

    QD = Int(_h5_attr(parent, "qd"))
    N = Int(_h5_attr(parent, "nsymms"))
    RD = Int(_h5_attr(parent, "rd"))
    RD == QD + N ||
        throw(ArgumentError("stored RMT rank $RD does not equal QD + N = $(QD + N)"))

    symmetries = _read_symmetries(parent)
    length(symmetries) == N ||
        throw(ArgumentError("symmetry count does not match tensor metadata"))
    M = n_nonabelian_symmetries(productsymm(symmetries))
    T = _eltype_from_tag(_h5_attr_string(parent, "eltype"))
    RT = schema_version == "1.0" ? T : _eltype_from_tag(_h5_attr_string(parent, "rmt_eltype"))

    qlabels, isdefined_bits, iszero_bits = _read_sector_metadata(parent, symmetries, Val(QD))
    sector_count = length(qlabels)
    Int(_h5_attr(parent, "sector_count")) == sector_count ||
        throw(ArgumentError("stored sector_count does not match qlabel metadata"))
    Int(_h5_attr(parent, "nonzero_sector_count")) == count(!, iszero_bits) ||
        throw(ArgumentError("stored nonzero sector count does not match sector state"))

    inds = _read_indices(parent, Val(QD))
    spaces = _read_spaces(parent, symmetries, Val(QD))
    wmatdata, wmatinfo = _read_wmat_storage(parent, sector_count, Val(M))
    RMTs = _read_rmt_storage(parent, RT, Val(RD), isdefined_bits, iszero_bits)
    conj_flag = schema_version == "1.0" ? false : Bool(_h5_attr(parent, "conj"))
    scale = schema_version == "1.0" ? one(T) : convert(T, _h5_attr(parent, "scale"))
    perm = schema_version == "1.0" ? ntuple(identity, Val(QD)) :
           ntuple(i -> Int(_h5_attr(parent, "perm")[i]), Val(QD))
    _is_valid_perm(perm) || throw(ArgumentError("stored permutation is invalid"))

    q = TLArray(Val(:alias_storage_view_state), symmetries, qlabels, wmatdata, wmatinfo,
                RMTs, isdefined_bits, iszero_bits, inds, spaces, conj_flag, scale, perm)
    q.isdefined == isdefined_bits ||
        throw(ArgumentError("loaded TLArray isdefined bits do not match stored bits"))
    q.iszero == iszero_bits ||
        throw(ArgumentError("loaded TLArray iszero bits do not match stored bits"))
    return q
end

"""
    load_tlarray(path; name="tensor") -> TLArray
    load_tlarray(h5, name) -> TLArray

Read concrete tensor metadata, w-matrices, sector flags, and RMT storage from
HDF5 group `name`. `path` is opened read-only by the path overload; `h5` is an
already open handle in the handle overload. Returns a newly owned concrete
`TLArray` and validates the serialized shape/type/symmetry metadata, throwing
an error for incompatible or corrupted files.
"""
function load_tlarray(path::AbstractString; name::AbstractString="tensor")
    HDF5.h5open(path, "r") do h5
        return load_tlarray(h5, name)
    end
end

load_tlarray(h5, name::AbstractString) = _read_tlarray_group(h5[name])
