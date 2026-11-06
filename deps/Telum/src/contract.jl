# ─── RMT contraction helpers ─────────────────────────────────────────────────
"""
    _identity_perm(::Val{D}) -> NTuple{D,Int}

Return the identity permutation for a rank-`D` object.

`Val{D}` carries the rank at compile time so callers can compare axis
permutations without allocating a `1:D` vector in hot RMT preparation paths.
"""
@inline _identity_perm(::Val{D}) where {D} = ntuple(identity, Val(D))

"""
    _hptt_permutedims(A, perm) -> AbstractArray
    _hptt_permutedims!(dest, A, perm) -> dest

Permute dense RMT axes, using HPTT for common BLAS scalar types.

`A` is the source rank-`D` array and `perm` is the one-based output-axis order
in Julia's `permutedims` convention. The allocating form returns `A` unchanged
for identity permutations, otherwise allocates the permuted result. The in-place
form writes into `dest`, whose shape must match the permuted source shape.

Specialized `Float32`, `Float64`, `ComplexF32`, and `ComplexF64` methods call
HPTT with zero-based `Int32` permutations and preserve array pointers across
the `ccall`. Generic array methods fall back to Julia `permutedims`/`copyto!`.
"""
function _hptt_permutedims(A::Array{Float64, D},
                           perm::NTuple{D, Int}) where {D}
    perm == _identity_perm(Val(D)) && return A

    out = Array{Float64}(undef, ntuple(i -> size(A, perm[i]), Val(D)))
    perm0 = Int32[perm[i] - 1 for i in 1:D]
    sizeA = Int32[size(A, i) for i in 1:D]
    ccall((:dTensorTranspose, HPTT_jll.libhptt), Cvoid,
          (Ptr{Int32}, Cint, Float64, Ptr{Float64}, Ptr{Int32}, Ptr{Int32},
           Float64, Ptr{Float64}, Ptr{Int32}, Cint, Cint),
          perm0, D, 1.0, A, sizeA, C_NULL,
          0.0, out, C_NULL, max(1, Threads.nthreads()), 0)
    return out
end

function _hptt_permutedims!(dest::StridedArray{Float64, D},
                            A::Array{Float64, D},
                            perm::NTuple{D, Int}) where {D}
    if perm == _identity_perm(Val(D))
        copyto!(dest, A)
        return dest
    end

    @assert size(dest) == ntuple(i -> size(A, perm[i]), Val(D))
    perm0 = Int32[perm[i] - 1 for i in 1:D]
    sizeA = Int32[size(A, i) for i in 1:D]
    GC.@preserve dest A begin
        ccall((:dTensorTranspose, HPTT_jll.libhptt), Cvoid,
              (Ptr{Int32}, Cint, Float64, Ptr{Float64}, Ptr{Int32}, Ptr{Int32},
               Float64, Ptr{Float64}, Ptr{Int32}, Cint, Cint),
              perm0, D, 1.0, pointer(A), sizeA, C_NULL,
              0.0, pointer(dest), C_NULL, max(1, Threads.nthreads()), 0)
    end
    return dest
end

function _hptt_permutedims(A::Array{Float32, D},
                           perm::NTuple{D, Int}) where {D}
    perm == _identity_perm(Val(D)) && return A

    out = Array{Float32}(undef, ntuple(i -> size(A, perm[i]), Val(D)))
    perm0 = Int32[perm[i] - 1 for i in 1:D]
    sizeA = Int32[size(A, i) for i in 1:D]
    ccall((:sTensorTranspose, HPTT_jll.libhptt), Cvoid,
          (Ptr{Int32}, Cint, Float32, Ptr{Float32}, Ptr{Int32}, Ptr{Int32},
           Float32, Ptr{Float32}, Ptr{Int32}, Cint, Cint),
          perm0, D, Float32(1), A, sizeA, C_NULL,
          Float32(0), out, C_NULL, max(1, Threads.nthreads()), 0)
    return out
end

function _hptt_permutedims!(dest::StridedArray{Float32, D},
                            A::Array{Float32, D},
                            perm::NTuple{D, Int}) where {D}
    if perm == _identity_perm(Val(D))
        copyto!(dest, A)
        return dest
    end

    @assert size(dest) == ntuple(i -> size(A, perm[i]), Val(D))
    perm0 = Int32[perm[i] - 1 for i in 1:D]
    sizeA = Int32[size(A, i) for i in 1:D]
    GC.@preserve dest A begin
        ccall((:sTensorTranspose, HPTT_jll.libhptt), Cvoid,
              (Ptr{Int32}, Cint, Float32, Ptr{Float32}, Ptr{Int32}, Ptr{Int32},
               Float32, Ptr{Float32}, Ptr{Int32}, Cint, Cint),
              perm0, D, Float32(1), pointer(A), sizeA, C_NULL,
              Float32(0), pointer(dest), C_NULL, max(1, Threads.nthreads()), 0)
    end
    return dest
end

function _hptt_permutedims(A::Array{ComplexF64, D},
                           perm::NTuple{D, Int}) where {D}
    perm == _identity_perm(Val(D)) && return A

    out = Array{ComplexF64}(undef, ntuple(i -> size(A, perm[i]), Val(D)))
    perm0 = Int32[perm[i] - 1 for i in 1:D]
    sizeA = Int32[size(A, i) for i in 1:D]
    ccall((:zTensorTranspose, HPTT_jll.libhptt), Cvoid,
          (Ptr{Int32}, Cint, ComplexF64, Cuchar, Ptr{ComplexF64}, Ptr{Int32}, Ptr{Int32},
           ComplexF64, Ptr{ComplexF64}, Ptr{Int32}, Cint, Cint),
          perm0, D, one(ComplexF64), false, A, sizeA, C_NULL,
          zero(ComplexF64), out, C_NULL, max(1, Threads.nthreads()), 0)
    return out
end

function _hptt_permutedims!(dest::StridedArray{ComplexF64, D},
                            A::Array{ComplexF64, D},
                            perm::NTuple{D, Int}) where {D}
    if perm == _identity_perm(Val(D))
        copyto!(dest, A)
        return dest
    end

    @assert size(dest) == ntuple(i -> size(A, perm[i]), Val(D))
    perm0 = Int32[perm[i] - 1 for i in 1:D]
    sizeA = Int32[size(A, i) for i in 1:D]
    GC.@preserve dest A begin
        ccall((:zTensorTranspose, HPTT_jll.libhptt), Cvoid,
              (Ptr{Int32}, Cint, ComplexF64, Cuchar, Ptr{ComplexF64}, Ptr{Int32}, Ptr{Int32},
               ComplexF64, Ptr{ComplexF64}, Ptr{Int32}, Cint, Cint),
              perm0, D, one(ComplexF64), false, pointer(A), sizeA, C_NULL,
              zero(ComplexF64), pointer(dest), C_NULL, max(1, Threads.nthreads()), 0)
    end
    return dest
end

function _hptt_permutedims(A::Array{ComplexF32, D},
                           perm::NTuple{D, Int}) where {D}
    perm == _identity_perm(Val(D)) && return A

    out = Array{ComplexF32}(undef, ntuple(i -> size(A, perm[i]), Val(D)))
    perm0 = Int32[perm[i] - 1 for i in 1:D]
    sizeA = Int32[size(A, i) for i in 1:D]
    ccall((:cTensorTranspose, HPTT_jll.libhptt), Cvoid,
          (Ptr{Int32}, Cint, ComplexF32, Cuchar, Ptr{ComplexF32}, Ptr{Int32}, Ptr{Int32},
           ComplexF32, Ptr{ComplexF32}, Ptr{Int32}, Cint, Cint),
          perm0, D, one(ComplexF32), false, A, sizeA, C_NULL,
          zero(ComplexF32), out, C_NULL, max(1, Threads.nthreads()), 0)
    return out
end

function _hptt_permutedims!(dest::StridedArray{ComplexF32, D},
                            A::Array{ComplexF32, D},
                            perm::NTuple{D, Int}) where {D}
    if perm == _identity_perm(Val(D))
        copyto!(dest, A)
        return dest
    end

    @assert size(dest) == ntuple(i -> size(A, perm[i]), Val(D))
    perm0 = Int32[perm[i] - 1 for i in 1:D]
    sizeA = Int32[size(A, i) for i in 1:D]
    GC.@preserve dest A begin
        ccall((:cTensorTranspose, HPTT_jll.libhptt), Cvoid,
              (Ptr{Int32}, Cint, ComplexF32, Cuchar, Ptr{ComplexF32}, Ptr{Int32}, Ptr{Int32},
               ComplexF32, Ptr{ComplexF32}, Ptr{Int32}, Cint, Cint),
              perm0, D, one(ComplexF32), false, pointer(A), sizeA, C_NULL,
              zero(ComplexF32), pointer(dest), C_NULL, max(1, Threads.nthreads()), 0)
    end
    return dest
end

@inline function _hptt_permutedims(A::AbstractArray{T, D},
                                   perm::NTuple{D, Int}) where {T, D}
    return Tuple(perm) == _identity_perm(Val(D)) ? A : permutedims(A, perm)
end

function _hptt_permutedims!(dest::AbstractArray{T, D},
                            A::AbstractArray{T, D},
                            perm::NTuple{D, Int}) where {T, D}
    if Tuple(perm) == _identity_perm(Val(D))
        copyto!(dest, A)
    else
        copyto!(dest, permutedims(A, perm))
    end
    return dest
end

function _hptt_permutedims!(dest::AbstractArray{T1, D},
                            A::AbstractArray{T2, D},
                            perm::NTuple{D, Int}) where {T1, T2, D}
    if Tuple(perm) == _identity_perm(Val(D))
        copyto!(dest, A)
    else
        copyto!(dest, permutedims(A, perm))
    end
    return dest
end

"""
    _contract_om_axis_data(A, M, axis) -> AbstractArray

Multiply one RMT outer-multiplicity axis by a dense transform matrix.

`A` is the source RMT-like array. `M` is a real matrix whose columns match the
length of axis `axis`; its rows become the output dimension on that axis.
`axis` is one-based in the axes of `A`. The helper reshapes `A` into matrix
multiplication form, choosing a cheaper two-dimensional path when the trailing
product is one, then reshapes the result back to the original rank with the
transformed axis size.
"""
function _contract_om_axis_data(A::AbstractArray{T1, D}, M::AbstractMatrix{T2}, axis::Int) where {T1, T2<:Real, D}
    dims  = size(A)
    k     = dims[axis]
    r     = size(M, 1)
    @assert size(M, 2) == k "axis size $(k) != M columns $(size(M, 2))"

    findim = Base.setindex(dims, r, axis)
    if r == 1 && k == 1
        return reshape(A .* M[1, 1], findim)
    end

    prod_before = 1
    for i in 1:axis-1
        prod_before *= dims[i]
    end
    prod_after = 1
    for i in axis+1:D
        prod_after *= dims[i]
    end

    A_3  = reshape(A, prod_before, k, prod_after)
    if prod_after == 1
        A_mk = reshape(A_3, prod_before, k)
        R_mr = A_mk * transpose(M)
        return reshape(R_mr, findim)
    end

    A_mk = reshape(permutedims(A_3, (1, 3, 2)), prod_before * prod_after, k)
    R_mr = A_mk * transpose(M)
    R_3  = permutedims(reshape(R_mr, prod_before, prod_after, r), (1, 3, 2))
    return reshape(R_3, findim)
end

"""
    _contract_om_axis(A, M, axis) -> AbstractArray

Public internal wrapper for applying one outer-multiplicity transform to an RMT.

`A` is the dense RMT block, `M` is the transform matrix, and `axis` selects the
RMT axis whose dimension is replaced by `size(M, 1)`. This wrapper exists so
call sites can express the operation in contraction terminology while sharing
the implementation in `_contract_om_axis_data`.
"""
function _contract_om_axis(A::AbstractArray{T1, D}, M::AbstractMatrix{T2}, axis::Int) where {T1, T2<:Real, D}
    return _contract_om_axis_data(A, M, axis)
end

"""
    _rmt_layout_sizes(rmt, perm, nf, cn, nsyms)
    _rmt_layout_sizes(rmt, perm, ::Val{NF}, ::Val{CN}, ::Val{N})
        -> (kept_sizes, contracted_sizes, om_sizes)

Split RMT dimensions into free, contracted, and outer-multiplicity groups.

`rmt` supplies the original axis sizes. `perm` is the source-axis order used to
prepare the contraction layout. `NF`/`nf` is the number of free physical axes,
`CN`/`cn` is the number of contracted physical axes, and `N`/`nsyms` is the
number of symmetry/OM axes. The returned tuples preserve the grouped axis order
and are later multiplied into three dimensions `(free, contracted, om)`.
"""
function _rmt_layout_sizes(rmt::AbstractArray{T, RD},
                           perm::NTuple{RD, Int},
                           nf::Int,
                           cn::Int,
                           nsyms::Int) where {T, RD}
    return _rmt_layout_sizes(rmt, perm, Val(nf), Val(cn), Val(nsyms))
end

function _rmt_layout_sizes(rmt::AbstractArray{T, RD},
                           perm::NTuple{RD, Int},
                           ::Val{NF},
                           ::Val{CN},
                           ::Val{N}) where {T, RD, NF, CN, N}
    dims = size(rmt)
    kept_sizes = ntuple(i -> dims[perm[i]], Val(NF))
    contracted_sizes = ntuple(i -> dims[perm[NF + i]], Val(CN))
    om_sizes = ntuple(i -> dims[perm[NF + CN + i]], Val(N))
    return kept_sizes, contracted_sizes, om_sizes
end

"""
    _rmt_array_data(rmt) -> AbstractArray

Return dense array storage for an RMT object that can be passed to permutation
kernels.

The dense `Array` method returns the array itself. Other RMT-like storage types
may define specialized accessors when they can expose compatible dense storage
without changing the contraction preparation interface.
"""
@inline _rmt_array_data(rmt::Array{T, RD}) where {T, RD} = rmt
@inline _rmt_array_data(rmt::AbstractArray) = Array(rmt)

"""
    _permuted_rmt_type(::Type{RMT}) -> Type

Return the prepared grouped rank-three RMT type for a source RMT container.

Dense source arrays become `Array{T,3}` after axes are permuted and grouped.
`DiagRMT{T,RD}` sources become `DiagRMT{T,3}` when their diagonal axes can be
represented in the grouped `(free, contracted, om)` layout.
"""
@inline _permuted_rmt_type(::Type{<:Array{T,RD}}) where {T,RD} = Array{T,3}
@inline _permuted_rmt_type(::Type{<:DiagRMT{T,RD}}) where {T,RD} = DiagRMT{T,3}
@inline _permuted_rmt_type(::Type{<:AbstractArray{T,RD}}) where {T,RD} = Array{T,3}

"""
    _prepared_axis_group(axis, perm, ::Val{NF}, ::Val{CN}) -> Int

Map an original RMT axis to its grouped prepared-RMT axis.

`axis` is an original one-based RMT axis. `perm` is the preparation permutation.
Positions `1:NF` in `perm` are grouped into prepared axis `1` (free), positions
`NF+1:NF+CN` into prepared axis `2` (contracted), and remaining positions into
prepared axis `3` (outer multiplicity). A missing axis raises `BoundsError`.
"""
@inline function _prepared_axis_group(axis::Int,
                                      perm::NTuple{RD, Int},
                                      ::Val{NF},
                                      ::Val{CN}) where {RD, NF, CN}
    @inbounds for pos in 1:RD
        if perm[pos] == axis
            pos <= NF && return 1
            pos <= NF + CN && return 2
            return 3
        end
    end
    throw(BoundsError(perm, axis))
end

"""
    _permuted_rmt_data(rmt, perm, nf, cn, nsyms)
    _permuted_rmt_data(rmt, perm, ::Val{NF}, ::Val{CN}, ::Val{N})

Prepare an RMT payload for pair contraction.

`rmt` is either dense array storage or a diagonal RMT. `perm` orders source axes
as `(free physical axes..., contracted physical axes..., OM axes...)`. The
free, contracted, and symmetry counts are supplied either as integers or as
`Val` parameters. Dense arrays are permuted with `_hptt_permutedims` and
reshaped to rank three. Diagonal RMTs keep diagonal storage when the diagonal
axes fall in the free/contracted groups and the OM group is singleton.
"""
function _permuted_rmt_data(rmt::AbstractArray{T, RD},
                            perm::NTuple{RD, Int},
                            nf::Int,
                            cn::Int,
                            nsyms::Int) where {T, RD}
    return _permuted_rmt_data(rmt, perm, Val(nf), Val(cn), Val(nsyms))
end

function _permuted_rmt_data(rmt::AbstractArray{T, RD},
                            perm::NTuple{RD, Int},
                            ::Val{NF},
                            ::Val{CN},
                            ::Val{N}) where {T, RD, NF, CN, N}
    kept_sizes, contracted_sizes, om_sizes =
        _rmt_layout_sizes(rmt, perm, Val(NF), Val(CN), Val(N))
    data = reshape(_hptt_permutedims(_rmt_array_data(rmt), perm),
                   prod(kept_sizes; init=1),
                   prod(contracted_sizes; init=1),
                   prod(om_sizes; init=1))
    return data
end

function _permuted_rmt_data(rmt::DiagRMT{T, RD},
                            perm::NTuple{RD, Int},
                            ::Val{NF},
                            ::Val{CN},
                            ::Val{N}) where {T, RD, NF, CN, N}
    kept_sizes, contracted_sizes, om_sizes =
        _rmt_layout_sizes(rmt, perm, Val(NF), Val(CN), Val(N))
    prod(om_sizes; init=1) == 1 || throw(ArgumentError(
        "DiagRMT prepared contraction data requires singleton OM dimension"))

    if rmt.axis[2] == 0
        prepared_axis = _prepared_axis_group(rmt.axis[1], perm, Val(NF), Val(CN))
        prepared_axis == 3 && throw(ArgumentError(
            "DiagRMT vectorized axis cannot be prepared on the OM dimension"))
        return DiagRMT{T,3}(rmt.diag, (prepared_axis, 0))
    end

    axis1 = _prepared_axis_group(rmt.axis[1], perm, Val(NF), Val(CN))
    axis2 = _prepared_axis_group(rmt.axis[2], perm, Val(NF), Val(CN))
    (axis1 == 3 || axis2 == 3) && throw(ArgumentError(
        "DiagRMT diagonal axes cannot be prepared on the OM dimension"))
    axis = axis1 == axis2 ? (axis1, 0) : minmax(axis1, axis2)
    return DiagRMT{T,3}(rmt.diag, axis)
end

"""
    prepared_sector_rmt(q, sector, perm, Val(NF), Val(CN), Val(N)) -> (prepared, scale)

Internal contraction boundary: obtain RMT data for stable source `sector` in
the grouped free/contracted/outer-multiplicity layout. `perm` maps the source
RMT axes into this layout; `Val(NF)`, `Val(CN)`, and `Val(N)` encode the number
of free legs, contracted legs, and symmetries for type-stable dispatch. The
returned `(prepared_rmt, scale)` respects deferred permutation and conjugation
without changing source storage; callers apply `scale` during accumulation.
"""
function prepared_sector_rmt(q::AbstractTLArray{T, QD},
                             idx::Int,
                             perm::NTuple{RD, Int},
                             ::Val{NF},
                             ::Val{CN},
                             ::Val{N}) where {T, QD, RD, NF, CN, N}
    source_perm = _effective_rmt_perm(q, perm, Val(RD))
    rmt, alpha = sector_rmt(q, idx)
    prepared = _permuted_rmt_data(rmt, source_perm, Val(NF), Val(CN), Val(N))
    stored_conj(q) && !(T <: Real) && return conj(prepared), conj(alpha)
    return prepared, alpha
end

"""
    _effective_rmt_perm(q, perm, ::Val{RD}) -> NTuple{RD,Int}

Translate a visible-axis RMT preparation permutation through stored view state.

`q` may carry a deferred physical-leg permutation. `perm` is expressed in
visible RMT axes; physical axes `<= QD` are mapped through `stored_perm(q)`,
while trailing OM axes are left unchanged. `Val{RD}` fixes the full RMT rank.
"""
@inline function _effective_rmt_perm(q::AbstractTLArray{T, QD},
                                     perm::NTuple{RD, Int},
                                     ::Val{RD}) where {T, QD, RD}
    return ntuple(pos -> perm[pos] <= QD ? stored_perm(q)[perm[pos]] : perm[pos], Val(RD))
end

"""
    _layout_source_rmt(q, sector) -> AbstractArray

Return the raw source RMT payload used only for dimension/layout inspection.

`q` is the source tensor and `sector` is a stable sector slot. The scalar scale
component returned by `sector_rmt` is intentionally ignored; this helper is used
only to determine prepared dimensions and storage-size estimates.
"""
@inline function _layout_source_rmt(q::AbstractTLArray, sector::Int)
    rmt, _ = sector_rmt(q, sector)
    return rmt
end

"""
    _stream_conj_requires_buffer(q) -> Bool

Return whether streaming preparation needs a writable buffer for conjugation.

Complex tensors with deferred conjugation cannot expose source storage directly
when streaming prepared RMT data because entries must be conjugated. Real
tensors or non-conjugated tensors can avoid this extra buffer requirement.
"""
@inline _stream_conj_requires_buffer(q::AbstractTLArray{T}) where {T} =
    stored_conj(q) && !(T <: Real)

"""
    _can_stream_prepared_rmt(q, perm, ::Val{NF}, ::Val{CN}, ::Val{N}) -> Bool

Decide whether prepared RMT data for `q` should be streamed through a reusable
buffer instead of cached as allocated prepared objects.

Dense RMT sources can stream when effective preparation requires a physical
axis permutation or complex conjugation, because a buffer is needed anyway.
Diagonal RMT sources return false because their compact representation is cheap
to cache and has specialized contraction paths. `perm`, `NF`, `CN`, and `N`
describe the same grouped layout used by `prepared_sector_rmt`.
"""
function _can_stream_prepared_rmt(q::AbstractTLArray{T, QD, N, RD, QT, PS, M, RMT},
                                  perm::NTuple{RD, Int},
                                  ::Val{NF},
                                  ::Val{CN},
                                  ::Val{N}) where {T, QD, N, RD, QT, PS, M, RMT, NF, CN}
    RMT <: DiagRMT && return false
    eff_perm = _effective_rmt_perm(q, perm, Val(RD))
    return eff_perm != _identity_perm(Val(RD)) || _stream_conj_requires_buffer(q)
end

"""
    _prepared_rmt_storage_size(q, perm, ::Val{NF}, ::Val{CN}, ::Val{N})
    _prepared_rmt_storage_size(q, sectors, perm, ::Val{NF}, ::Val{CN}, ::Val{N})
        -> (total_len, max_len)

Estimate prepared dense storage needed for source sectors.

`q` is the source tensor. The first form scans all active/defined sectors; the
second scans only the provided sector slots. `perm`, `NF`, `CN`, and `N` define
the grouped prepared RMT layout. `total_len` is the sum of prepared element
counts, and `max_len` is the largest single-sector prepared buffer length. Lazy
contraction uses these values to choose which side, if any, should be streamed.
"""
function _prepared_rmt_storage_size(q::AbstractTLArray{T, QD, N, RD},
                                    perm::NTuple{RD, Int},
                                    ::Val{NF},
                                    ::Val{CN},
                                    ::Val{N}) where {T, QD, N, RD, NF, CN}
    eff_perm = _effective_rmt_perm(q, perm, Val(RD))
    total_len = 0
    max_len = 0
    for sector in sector_slots(q)
        is_sector_zero(q, sector) && !is_sector_defined(q, sector) && continue
        fdim, cdim, odim =
            _rmt_contract_dims(_layout_source_rmt(q, sector), eff_perm, Val(NF), Val(CN), Val(N))
        len = fdim * cdim * odim
        total_len += len
        max_len = max(max_len, len)
    end
    return total_len, max_len
end

function _prepared_rmt_storage_size(q::AbstractTLArray{T, QD, N, RD},
                                    sectors::AbstractVector{<:Integer},
                                    perm::NTuple{RD, Int},
                                    ::Val{NF},
                                    ::Val{CN},
                                    ::Val{N}) where {T, QD, N, RD, NF, CN}
    eff_perm = _effective_rmt_perm(q, perm, Val(RD))
    total_len = 0
    max_len = 0
    for sector_raw in sectors
        sector = Int(sector_raw)
        is_sector_zero(q, sector) && !is_sector_defined(q, sector) && continue
        fdim, cdim, odim = _prepared_rmt_dims_from_size(
            _sector_rmt_size(q, sector), eff_perm, Val(NF), Val(CN), Val(N))
        len = fdim * cdim * odim
        total_len += len
        max_len = max(max_len, len)
    end
    return total_len, max_len
end

"""
    _sector_rmt_size(q, sector) -> NTuple

Return the dense RMT shape for `sector` without necessarily materializing a
payload.

Concrete tensors delegate to `sector_rmt_dim`. Lazy contractions and subarrays
use cached `rmt_sizes`. Singleton wrappers delegate through their common
metadata interface. The result is used for planning prepared storage and
scratch sizes.
"""
@inline _sector_rmt_size(q::TLArray{T, QD, N, RD}, sector::Int) where {T, QD, N, RD} =
    sector_rmt_dim(q, sector)::NTuple{RD, Int}

@inline _sector_rmt_size(q::TLArrayContraction{T, QD, N, RD}, sector::Int) where {T, QD, N, RD} =
    q.rmt_sizes[sector]
@inline _sector_rmt_size(q::SubTLArray{T, QD, N, RD}, sector::Int) where {T, QD, N, RD} =
    q.rmt_sizes[sector]
@inline _sector_rmt_size(q::Union{AddSingletonTLArray{T, QD, N, RD}, DeleteSingletonTLArray{T, QD, N, RD}}, sector::Int) where {T, QD, N, RD} =
    sector_rmt_dim(q, sector)::NTuple{RD, Int}

"""
    _prepared_rmt_dims_from_size(dims, perm, ::Val{NF}, ::Val{CN}, ::Val{N})
        -> (free_dim, contracted_dim, om_dim)

Collapse an RMT shape into the three grouped contraction dimensions.

`dims` is the original RMT shape. `perm` orders axes as free, contracted, then
OM. `NF`, `CN`, and `N` specify how many axes belong to each group. The result
is the product of axis sizes in each group, with empty products equal to one.
"""
function _prepared_rmt_dims_from_size(dims::NTuple{RD, Int},
                                      perm::NTuple{RD, Int},
                                      ::Val{NF},
                                      ::Val{CN},
                                      ::Val{N}) where {RD, NF, CN, N}
    kept_dim = 1
    contracted_dim = 1
    om_dim = 1
    for i in 1:NF
        kept_dim *= dims[perm[i]]
    end
    for i in 1:CN
        contracted_dim *= dims[perm[NF + i]]
    end
    for i in 1:N
        om_dim *= dims[perm[NF + CN + i]]
    end
    return kept_dim, contracted_dim, om_dim
end

"""
    _prepared_rmt_dims(q, sector, perm, ::Val{NF}, ::Val{CN}, ::Val{N})
        -> (free_dim, contracted_dim, om_dim)

Return grouped prepared dimensions for one source sector.

`q` supplies the source sector shape and any deferred physical permutation.
`sector` is the stable source slot. `_sector_rmt_size` returns dimensions in
visible logical-axis order, so `perm` is applied directly when collapsing
dimensions with `_prepared_rmt_dims_from_size`.
"""
function _prepared_rmt_dims(q::AbstractTLArray{T, QD, N, RD},
                            sector::Int,
                            perm::NTuple{RD, Int},
                            ::Val{NF},
                            ::Val{CN},
                            ::Val{N}) where {T, QD, N, RD, NF, CN}
    # `_sector_rmt_size` already reports logical-axis dimensions, including
    # any deferred physical permutation. Apply the contraction layout in that
    # same logical axis space; translating through stored_perm here would
    # permute the dimensions a second time.
    return _prepared_rmt_dims_from_size(_sector_rmt_size(q, sector), perm,
                                        Val(NF), Val(CN), Val(N))
end

"""
    _prepare_sector_rmt_into!(buffer, q, idx, perm, ::Val{NF}, ::Val{CN}, ::Val{N})
        -> (prepared_view, scale)

Prepare one dense source sector into caller-owned scratch storage.

`buffer` is a reusable flat vector large enough for the permuted source payload.
`q` is the source tensor, `idx` is the stable sector slot, and `perm` defines
the visible-axis grouping. `NF`, `CN`, and `N` specify free, contracted, and OM
axis counts. The function writes the effective-permuted source RMT into
`buffer`, applies deferred complex conjugation in place when required, and
returns a rank-three reshaped view plus the sector scale factor.
"""
function _prepare_sector_rmt_into!(buffer::Vector{T},
                                   q::AbstractTLArray{T, QD},
                                   idx::Int,
                                   perm::NTuple{RD, Int},
                                   ::Val{NF},
                                   ::Val{CN},
                                   ::Val{N}) where {T, QD, RD, NF, CN, N}
    source_perm = _effective_rmt_perm(q, perm, Val(RD))
    rmt, alpha = sector_rmt(q, idx)
    kept_sizes, contracted_sizes, om_sizes =
        _rmt_layout_sizes(rmt, source_perm, Val(NF), Val(CN), Val(N))
    out_dims = (kept_sizes..., contracted_sizes..., om_sizes...)
    len = prod(out_dims; init=1)
    @assert length(buffer) >= len
    dest = reshape(view(buffer, 1:len), out_dims)
    _hptt_permutedims!(dest, _rmt_array_data(rmt), source_perm)

    if stored_conj(q) && !(T <: Real)
        dest .= conj.(dest)
        alpha = conj(alpha)
    end

    return reshape(view(buffer, 1:len),
                   prod(kept_sizes; init=1),
                   prod(contracted_sizes; init=1),
                   prod(om_sizes; init=1)), alpha
end

"""
    _cached_prepared_sector_rmt!(cache, q, idx, perm, ::Val{NF}, ::Val{CN}, ::Val{N})

Return prepared RMT data from a full sector-indexed cache.

`cache` has one slot per source sector. `q`, `idx`, `perm`, `NF`, `CN`, and `N`
are forwarded to `prepared_sector_rmt` on a cache miss. Unlike the compact cache
used by lazy requested-sector evaluation, this helper indexes `cache` directly
by the original sector slot.
"""
function _cached_prepared_sector_rmt!(cache::Vector{Tuple{PRMT, T}},
                                      q::AbstractTLArray{T},
                                      idx::Int,
                                      perm::NTuple{RD, Int},
                                      ::Val{NF},
                                      ::Val{CN},
                                      ::Val{N}) where {PRMT, T, RD, NF, CN, N}
    if !isassigned(cache, idx)
        cache[idx] = prepared_sector_rmt(q, idx, perm, Val(NF), Val(CN), Val(N))
    end
    return cache[idx]
end

"""
    _combined_om_factor_info(factors) -> (rank_sizes, rank_dim, om1_dim, om2_dim, scale)

Inspect per-symmetry outer-multiplicity factor blocks before combining them.

`factors` is an `N`-tuple of three-axis factor arrays with dimensions
`(rank_dim, om1_dim, om2_dim)` for each symmetry. Scalar `1×1×1` factors are
folded into `scale` instead of increasing the explicit Kronecker-like factor
array. `rank_sizes` records the rank dimension of each symmetry, while
`rank_dim`, `om1_dim`, and `om2_dim` are the products over non-scalar factors.
"""
function _combined_om_factor_info(factors::NTuple{N, LT}) where {N, LT<:AbstractArray{Float64, 3}}
    rank_sizes = ntuple(n -> size(factors[n], 1), Val(N))
    scale = 1.0

    rank_dim = 1
    om1_dim = 1
    om2_dim = 1
    for n in 1:N
        factor = factors[n]
        r_n, om1_n, om2_n = size(factor)

        if r_n == 1 && om1_n == 1 && om2_n == 1
            scale *= factor[1, 1, 1]
            continue
        end

        rank_dim *= r_n
        om1_dim *= om1_n
        om2_dim *= om2_n
    end

    return rank_sizes, rank_dim, om1_dim, om2_dim, scale
end

"""
    _combined_om_factor_array(factors, rank_dim_total, om1_dim_total, om2_dim_total)
        -> Array{Float64,3}

Build the explicit combined non-Abelian factor block for a sector-pair
contribution.

`factors` contains one three-axis block per symmetry. The total dimensions are
precomputed by `_combined_om_factor_info` and passed as
`rank_dim_total`, `om1_dim_total`, and `om2_dim_total`. The method multiplies
non-scalar factors in tensor-product order, with prior dimensions varying
fastest, producing a block indexed by combined representation rank, left OM
axis, and right OM axis.
"""
function _combined_om_factor_array(factors::NTuple{N, LT},
                                   rank_dim_total::Int,
                                   om1_dim_total::Int,
                                   om2_dim_total::Int) where {N, LT<:AbstractArray{Float64, 3}}
    K = ones(Float64, 1, 1, 1)
    rank_dim = 1
    om1_dim = 1
    om2_dim = 1

    for n in 1:N
        factor = factors[n]
        r_n, om1_n, om2_n = size(factor)

        if r_n == 1 && om1_n == 1 && om2_n == 1
            continue
        end

        F = reshape(factor, r_n, om1_n, om2_n)
        K_new = Array{Float64}(undef, rank_dim * r_n,
                               om1_dim * om1_n,
                               om2_dim * om2_n)
        @inbounds for o2n in 1:om2_n, o2p in 1:om2_dim,
                      o1n in 1:om1_n, o1p in 1:om1_dim,
                      rn in 1:r_n, rp in 1:rank_dim
            K_new[rp + (rn - 1) * rank_dim,
                  o1p + (o1n - 1) * om1_dim,
                  o2p + (o2n - 1) * om2_dim] = K[rp, o1p, o2p] * F[rn, o1n, o2n]
        end
        K = K_new
        rank_dim *= r_n
        om1_dim *= om1_n
        om2_dim *= om2_n
    end

    @assert size(K) == (rank_dim_total, om1_dim_total, om2_dim_total)
    return K
end

"""
    _rmt_factor_array_and_rank_sizes(factors) -> (K, rank_sizes)

Combine per-symmetry factor blocks into the `K` tensor used by RMT contraction.

`factors` is the tuple of symmetry-local `(rank, om1, om2)` arrays. Scalar
factors are folded into the final array by multiplication after explicit
combination. `K` is the dense combined factor block; `rank_sizes` records the
per-symmetry rank dimensions used later to shape the result RMT axes.
"""
function _rmt_factor_array_and_rank_sizes(factors::NTuple{N, LT}) where {N, LT<:AbstractArray{Float64, 3}}
    rank_sizes, rank_dim, om1_dim, om2_dim, scale =
        _combined_om_factor_info(factors)
    K = _combined_om_factor_array(factors, rank_dim, om1_dim, om2_dim)
    scale == 1.0 || (K .*= scale)
    return K, rank_sizes
end

"""
    _rmt_contract_order(fdim, gdim, cdim, o1dim, o2dim, rdim) -> (order, cost)

Choose the multiplication order for a dense prepared-RMT pair contraction.

`fdim` and `gdim` are the free dimensions of the left and right prepared RMTs.
`cdim` is the shared contracted dimension. `o1dim` and `o2dim` are the left and
right outer-multiplicity dimensions. `rdim` is the combined non-Abelian rank
dimension of the factor tensor `K`.

The returned `order` is `:AB`, `:AK`, or `:BK`, indicating whether the first
intermediate contraction combines the two RMTs, the left RMT with `K`, or the
right RMT with `K`. `cost` is the estimated scalar multiply/add count for that
order and is used for cost accounting.
"""
function _rmt_contract_order(fdim::Int, gdim::Int, cdim::Int,
                             o1dim::Int, o2dim::Int, rdim::Int)
    ab_cost = fdim * gdim * o1dim * o2dim * cdim +
              fdim * gdim * rdim * o1dim * o2dim
    ak_cost = fdim * cdim * rdim * o1dim * o2dim +
              fdim * gdim * rdim * cdim * o2dim
    bk_cost = gdim * cdim * rdim * o1dim * o2dim +
              fdim * gdim * rdim * cdim * o1dim

    if ab_cost <= ak_cost && ab_cost <= bk_cost
        return :AB, ab_cost
    elseif ak_cost <= bk_cost
        return :AK, ak_cost
    else
        return :BK, bk_cost
    end
end

"""
    _rmt_contract_temp_len(...)

Return the temporary buffer length required for an RMT pair contraction.

Dimension arguments have the same meaning as `_rmt_contract_order`.
Order-specific methods compute the exact intermediate array size for `:AB`,
`:AK`, or `:BK`. Type-dispatch methods specialize diagonal/dense cases so lazy
contraction can allocate one reusable `temp` vector large enough for the whole
requested sector batch without over-materializing diagonal paths.
"""
function _rmt_contract_temp_len(fdim::Int, gdim::Int, cdim::Int,
                                o1dim::Int, o2dim::Int, rdim::Int,
                                order::Symbol)
    if order === :AB
        return fdim * o1dim * gdim * o2dim
    elseif order === :AK
        return fdim * cdim * o2dim * rdim
    else
        return gdim * cdim * o1dim * rdim
    end
end

function _rmt_contract_temp_len(fdim::Int, gdim::Int, cdim::Int,
                                o1dim::Int, o2dim::Int,
                                K::AbstractArray)
    rank_dim, om1_dim, om2_dim = size(K)
    rank_dim == 1 && om1_dim == 1 && om2_dim == 1 && return 0

    order, _ = _rmt_contract_order(fdim, gdim, cdim, o1dim, o2dim, rank_dim)
    return _rmt_contract_temp_len(fdim, gdim, cdim, o1dim, o2dim, rank_dim, order)
end

function _rmt_contract_temp_len(::Type{A}, ::Type{B},
                                fdim::Int, gdim::Int, cdim::Int,
                                o1dim::Int, o2dim::Int,
                                K::AbstractArray
) where {A<:AbstractArray, B<:AbstractArray}
    return _rmt_contract_temp_len(fdim, gdim, cdim, o1dim, o2dim, K)
end

function _rmt_contract_temp_len(::Type{A}, ::Type{B},
                                fdim::Int, gdim::Int, cdim::Int,
                                o1dim::Int, o2dim::Int,
                                K::AbstractArray
) where {A<:DiagRMT, B<:Array}
    return gdim * cdim * size(K, 1)
end

function _rmt_contract_temp_len(::Type{A}, ::Type{B},
                                fdim::Int, gdim::Int, cdim::Int,
                                o1dim::Int, o2dim::Int,
                                K::AbstractArray
) where {A<:Array, B<:DiagRMT}
    return 0
end

function _rmt_contract_temp_len(::Type{A}, ::Type{B},
                                fdim::Int, gdim::Int, cdim::Int,
                                o1dim::Int, o2dim::Int,
                                K::AbstractArray
) where {A<:DiagRMT, B<:DiagRMT}
    return 0
end

function _rmt_contract_temp_len(A::AbstractArray,
                                B::AbstractArray,
                                K::AbstractArray)
    fdim, cdim, o1dim = size(A)
    gdim, _, o2dim = size(B)
    return _rmt_contract_temp_len(typeof(A), typeof(B),
                                  fdim, gdim, cdim, o1dim, o2dim,
                                  K)
end

"""
    _rmt_contract_dims(rmt, perm, nf, cn, nsyms)
    _rmt_contract_dims(rmt, perm, ::Val{NF}, ::Val{CN}, ::Val{N})
        -> (free_dim, contracted_dim, om_dim)

Return the three grouped dimensions used by prepared RMT contraction.

`rmt` supplies source axis sizes. `perm` orders axes into free, contracted, and
OM groups. The free/contracted/symmetry counts may be runtime integers or
compile-time `Val`s. This is the shape-only analogue of `_permuted_rmt_data`.
"""
function _rmt_contract_dims(rmt::AbstractArray{T, RD},
                            perm::NTuple{RD, Int},
                            nf::Int,
                            cn::Int,
                            nsyms::Int) where {T, RD}
    return _rmt_contract_dims(rmt, perm, Val(nf), Val(cn), Val(nsyms))
end

function _rmt_contract_dims(rmt::AbstractArray{T, RD},
                            perm::NTuple{RD, Int},
                            ::Val{NF},
                            ::Val{CN},
                            ::Val{N}) where {T, RD, NF, CN, N}
    kept_sizes, contracted_sizes, om_sizes =
        _rmt_layout_sizes(rmt, perm, Val(NF), Val(CN), Val(N))
    return (prod(kept_sizes; init=1),
            prod(contracted_sizes; init=1),
            prod(om_sizes; init=1))
end

"""
    _rmt_temp_view(temp, dims) -> Array

Create an unowned array view over a reusable contraction scratch vector.

`temp` is a flat vector whose length must be at least `prod(dims)`. `dims`
defines the temporary array shape. The returned `Array` shares memory with
`temp` through `unsafe_wrap` and must not escape the contraction call that owns
the scratch buffer.
"""
@inline function _rmt_temp_view(temp::Vector{T},
                               dims::NTuple{N, Int}) where {T, N}
    @assert length(temp) >= prod(dims; init=1)
    return unsafe_wrap(Array, pointer(temp), dims; own=false)
end

"""
    _accumulate_mkl!(out, A, B, K, order, temp, beta=one(T)) -> out

Accumulate one dense prepared-RMT pair contribution with a non-scalar factor
block.

`out` has shape `(fdim, gdim, rank_dim)`. `A` and `B` are grouped prepared RMTs
with shapes `(fdim, cdim, om1dim)` and `(gdim, cdim, om2dim)`. `K` has shape
`(rank_dim, om1dim, om2dim)`. `order` is the strategy selected by
`_rmt_contract_order`. `temp` is reusable scratch storage sized by
`_rmt_contract_temp_len`. `beta` controls whether the contribution overwrites
or accumulates into existing `out` values.
"""
function _accumulate_mkl!(out::AbstractArray{T, 3},
                          A::AbstractArray,
                          B::AbstractArray,
                          K::AbstractArray,
                          order::Symbol,
                          temp::Vector{T},
                          beta::T = one(T)) where {T}
    fdim, cdim, o1dim = size(A)
    gdim = size(B, 1)
    o2dim = size(B, 3)
    rdim = size(K, 1)

    if order === :AB
        tmp = _rmt_temp_view(temp, (fdim, gdim, o1dim, o2dim))
        for i in 1:o1dim
            for j in 1:o2dim
                tmp_view = @view tmp[:, :, i, j]
                A_, B_ = @view(A[:, :, i]), @view B[:, :, j]
                mul!(tmp_view, A_, transpose(B_), one(T), zero(T))
            end
        end
        tmp_ = reshape(tmp, fdim * gdim, o1dim * o2dim)
        K_ = reshape(K, rdim, o1dim * o2dim)
        out_ = reshape(out, fdim * gdim, rdim)
        mul!(out_, tmp_, transpose(K_), one(T), beta)

    elseif order === :AK
        tmp = _rmt_temp_view(temp, (fdim, cdim, o2dim, rdim))
        Kt = permutedims(K, (2, 3, 1))
        A_ = reshape(A, fdim * cdim, o1dim)
        Kt_ = reshape(Kt, o1dim, o2dim * rdim)
        tmp_r1 = reshape(tmp, fdim * cdim, o2dim * rdim)
        mul!(tmp_r1, A_, Kt_, one(T), zero(T))

        B_mat_t = transpose(reshape(B, gdim, cdim * o2dim))
        for r in 1:rdim
            tmp_r2 = reshape(@view(tmp[:, :, :, r]), (fdim, cdim * o2dim))
            out_r = @view(out[:, :, r])
            mul!(out_r, tmp_r2, B_mat_t, one(T), beta)
        end

    else
        tmp = _rmt_temp_view(temp, (gdim, cdim, o1dim, rdim))
        Kt = permutedims(K, (3, 2, 1))
        B_ = reshape(B, gdim * cdim, o2dim)
        Kt_ = reshape(Kt, o2dim, o1dim * rdim)
        tmp_r1 = reshape(tmp, gdim * cdim, o1dim * rdim)
        mul!(tmp_r1, B_, Kt_, one(T), zero(T))

        A_mat = reshape(A, fdim, cdim * o1dim)
        for r in 1:rdim
            tmp_r2 = reshape(@view(tmp[:, :, :, r]), (gdim, cdim * o1dim))
            out_r = @view(out[:, :, r])
            mul!(out_r, A_mat, transpose(tmp_r2), one(T), beta)
        end
    end

    return out
end

"""
    _accumulate_scalar_om!(out, A, B, scale, beta=one(T)) -> out

Accumulate a prepared-RMT pair when all OM/factor dimensions are scalar.

`A` and `B` have singleton OM axes and a shared contracted dimension. `scale` is
the scalar factor block value. `beta` controls overwrite versus accumulation
into `out[:, :, 1]`. This path avoids building the general `K` contraction
temporary when the non-Abelian factor is effectively a scalar.
"""
function _accumulate_scalar_om!(out::AbstractArray{T, 3},
                                A::AbstractArray,
                                B::AbstractArray,
                                scale,
                                beta::T = one(T)) where {T}
    fdim, cdim, o1dim = size(A)
    gdim, cdim2, o2dim = size(B)
    @assert cdim == cdim2
    @assert o1dim == 1 && o2dim == 1 && size(out, 3) == 1

    A_mat = reshape(A, fdim, cdim)
    B_mat = reshape(B, gdim, cdim)
    out_mat = @view out[:, :, 1]
    mul!(out_mat, A_mat, transpose(B_mat), scale, beta)
    return out
end

"""
    _contract_RMT_pair_into_generic_no_count!(out, A, B, K, beta=one(T)) -> out

Reference loop implementation for prepared-RMT pair contraction.

`out`, `A`, `B`, and `K` use the same grouped dimensions as
`_accumulate_mkl!`. `beta` controls accumulation. This method does not update
the contraction-cost counter; callers use it for diagonal fallback paths whose
cost is counted by the dispatching wrapper before entry.
"""
function _contract_RMT_pair_into_generic_no_count!(out::AbstractArray{T, 3},
                                                   A::AbstractArray{T1, 3},
                                                   B::AbstractArray{T2, 3},
                                                   K::AbstractArray,
                                                   beta::T = one(T)) where {T, T1, T2}
    fdim, cdim, o1dim = size(A)
    gdim, cdim2, o2dim = size(B)
    rdim = size(K, 1)
    @assert cdim == cdim2

    @inbounds for r in 1:rdim
        for g in 1:gdim
            for f in 1:fdim
                acc = zero(T)
                for o2 in 1:o2dim
                    for o1 in 1:o1dim
                        for c in 1:cdim
                            acc += A[f, c, o1] * B[g, c, o2] * K[r, o1, o2]
                        end
                    end
                end
                if iszero(beta)
                    out[f, g, r] = acc
                else
                    out[f, g, r] = acc + beta * out[f, g, r]
                end
            end
        end
    end
    return out
end

"""
    _contract_RMT_pair_into!(out, A, B, K, temp, beta=one(T)) -> out

Contract one pair of prepared RMT blocks into a result-sector contribution.

`out` is the result buffer for one output sector. `A` and `B` are prepared
rank-three source RMTs or grouped `DiagRMT`s. `K` is the combined factor tensor.
`temp` is reusable scratch storage. `beta` is zero for the first contribution to
a result sector and one for later accumulations.

Dense/dense inputs use scalar-OM and optimized matrix-multiplication paths.
Diagonal combinations use specialized fast paths when the diagonal axes align
with free/contracted groups, and otherwise fall back to the generic exact loop.
"""
function _contract_RMT_pair_into!(out::AbstractArray{T, 3},
                                  A::AbstractArray{T1, 3},
                                  B::AbstractArray{T2, 3},
                                  K::AbstractArray,
                                  temp::Vector{T},
                                  beta::T = one(T)) where {T, T1, T2}
    rank_dim, om1_dim, om2_dim = size(K)

    if rank_dim == 1 && om1_dim == 1 && om2_dim == 1
        fdim, cdim, _ = size(A)
        gdim = size(B, 1)
        _add_contraction_cost!(fdim * gdim * cdim)
        return _accumulate_scalar_om!(out, A, B, K[1, 1, 1], beta)
    end

    fdim, cdim, o1dim = size(A)
    gdim, _, o2dim = size(B)
    order, cost = _rmt_contract_order(fdim, gdim, cdim, o1dim, o2dim, rank_dim)
    _add_contraction_cost!(cost)

    return _accumulate_mkl!(out, A, B, K, order, temp, beta)
end

function _contract_RMT_pair_into!(out::AbstractArray{T, 3},
                                  A::DiagRMT{T1, 3},
                                  B::DiagRMT{T2, 3},
                                  K::AbstractArray,
                                  temp::Vector{T},
                                  beta::T = one(T)) where {T, T1, T2}
    return _contract_RMT_pair_into_generic_no_count!(out, A, B, K, beta)
end

function _contract_RMT_pair_into!(out::AbstractArray{T, 3},
                                  A::DiagRMT{T1, 3},
                                  B::AbstractArray{T2, 3},
                                  K::AbstractArray,
                                  temp::Vector{T},
                                  beta::T = one(T)) where {T, T1, T2}
                                  
    if A.axis == (1, 2) && size(A, 3) == 1 && size(K, 2) == 1 && iszero(beta)
        fdim, cdim, o1dim = size(A)
        gdim, cdim2, o2dim = size(B)
        rdim = size(K, 1)
        @assert cdim == cdim2
        @assert o1dim == 1
        @assert size(out) == (fdim, gdim, rdim)

        # Diagonal-left common path: first contract the dense right RMT with K
        # into tmp[g, c, r], then transpose c into the output free-left axis.
        _add_contraction_cost!(gdim * cdim * rdim * o2dim)
        tmp = _rmt_temp_view(temp, (gdim, cdim, rdim))
        B_mat = reshape(B, gdim * cdim, o2dim)
        K_mat = reshape(K, rdim, o2dim)
        tmp_mat = reshape(tmp, gdim * cdim, rdim)
        mul!(tmp_mat, B_mat, transpose(K_mat), one(T), zero(T))

        @inbounds for r in 1:rdim
            for g in 1:gdim
                for f in 1:fdim
                    out[f, g, r] = A.diag[f] * tmp[g, f, r]
                end
            end
        end
        return out
    end

    fdim, cdim, o1dim = size(A)
    gdim, cdim2, o2dim = size(B)
    rdim = size(K, 1)
    @assert cdim == cdim2
    _add_contraction_cost!(gdim * cdim * rdim * o1dim * o2dim)
    return _contract_RMT_pair_into_generic_no_count!(out, A, B, K, beta)
end

function _contract_RMT_pair_into!(out::AbstractArray{T, 3},
                                  A::AbstractArray{T1, 3},
                                  B::DiagRMT{T2, 3},
                                  K::AbstractArray,
                                  temp::Vector{T},
                                  beta::T = one(T)) where {T, T1, T2}
    if B.axis == (1, 2) && size(B, 3) == 1 && size(K, 3) == 1 && iszero(beta)
        fdim, cdim, o1dim = size(A)
        gdim, cdim2, o2dim = size(B)
        rdim = size(K, 1)
        @assert cdim == cdim2
        @assert o2dim == 1
        @assert size(out) == (fdim, gdim, rdim)

        A_mat = reshape(A, fdim * cdim, o1dim)
        K_mat = reshape(K, rdim, o1dim)
        out_mat = reshape(out, fdim * gdim, rdim)
        _add_contraction_cost!(fdim * cdim * rdim * o1dim)
        mul!(out_mat, A_mat, transpose(K_mat), one(T), zero(T))

        @inbounds for r in 1:rdim
            for g in 1:gdim
                scale = B.diag[g]
                for f in 1:fdim
                    out[f, g, r] *= scale
                end
            end
        end
        return out
    end

    fdim, cdim, o1dim = size(A)
    gdim, cdim2, o2dim = size(B)
    rdim = size(K, 1)
    @assert cdim == cdim2
    _add_contraction_cost!(fdim * cdim * rdim * o1dim * o2dim)
    return _contract_RMT_pair_into_generic_no_count!(out, A, B, K, beta)
end

"""
    _prepared_cache_rmt(x)
    _prepared_cache_scale(x)

Accessors for prepared RMT cache entries.

Prepared entries are either `(prepared_rmt, scale)` tuples or bare arrays when
no separate scale is carried. `_prepared_cache_rmt` extracts the grouped RMT
payload. `_prepared_cache_scale` extracts the scalar multiplier, defaulting to
`one(T)` for bare arrays.
"""
@inline _prepared_cache_rmt(x::Tuple) = x[1]
@inline _prepared_cache_rmt(x::AbstractArray) = x
@inline _prepared_cache_scale(x::Tuple) = x[2]
@inline _prepared_cache_scale(x::AbstractArray{T}) where {T} = one(T)

"""
    _allocate_contract_result_rmts!(result_RMTs, prepared_sectors, out_keys,
                                    out_to_result, ::Val{NF1}, ::Val{NF2})
        -> result_RMTs

Allocate concrete result RMT arrays for output sectors participating in an
eager contraction.

`result_RMTs` is indexed by final result sector slot and is filled in place.
`prepared_sectors[out_pos]` contains the combined rank-size tuple for one
prepared output key. `out_keys[out_pos]` stores the free left/right q-label and
dense-dimension metadata used to recover physical free-axis sizes.
`out_to_result[out_pos]` maps each prepared output key to its final sector slot,
with zero meaning that the prepared key does not produce a stored result sector.
`NF1` and `NF2` are the numbers of free physical axes from the left and right
source tensors.
"""
function _allocate_contract_result_rmts!(result_RMTs::Vector{Array{RT, RD_out}},
                                         prepared_sectors,
                                         out_keys,
                                         out_to_result::AbstractVector{Int},
                                         ::Val{NF1},
                                         ::Val{NF2}) where {RT, RD_out, NF1, NF2}
    for out_pos in eachindex(out_keys)
        result_pos = out_to_result[out_pos]
        result_pos == 0 && continue
        rank_sizes = prepared_sectors[out_pos][3]
        kept_sizes1 = ntuple(i -> out_keys[out_pos][1][i][2], Val(NF1))
        kept_sizes2 = ntuple(i -> out_keys[out_pos][2][i][2], Val(NF2))
        result_shape::NTuple{RD_out, Int} = (kept_sizes1..., kept_sizes2..., rank_sizes...)
        result_RMTs[result_pos] = Array{RT, RD_out}(undef, result_shape)
    end
    return result_RMTs
end

"""
    _accumulate_contract_work_item!(result_RMTs, result_has_contribution,
                                    prepared_sectors, out_to_result, item,
                                    entry1, entry2, temp) -> result_RMTs

Accumulate one prepared eager-contraction work item.

`result_RMTs` contains already allocated result-sector RMT arrays.
`result_has_contribution` tracks whether each result slot has been written at
least once. `prepared_sectors` provides factor tensors for each prepared output
key. `out_to_result` maps the work item's prepared output position to the final
result sector slot. `item` is `(sector1, sector2, out_pos, factor_pos)`.
`entry1` and `entry2` are prepared RMT/cache-scale entries for the two source
sectors. `temp` is reusable scratch for `_contract_RMT_pair_into!`.

The source scales are folded into the selected factor block before contraction,
and `beta` is chosen from `result_has_contribution` so the first contribution
overwrites uninitialized storage while later contributions accumulate.
"""
function _accumulate_contract_work_item!(result_RMTs::Vector{Array{RT, RD_out}},
                                         result_has_contribution::BitVector,
                                         prepared_sectors,
                                         out_to_result::AbstractVector{Int},
                                         item::NTuple{4, Int},
                                         entry1,
                                         entry2,
                                         temp::Vector{RT}) where {RT, RD_out}
    _, _, out_pos, factor_pos = item
    result_pos = out_to_result[out_pos]
    data1 = _prepared_cache_rmt(entry1)
    data2 = _prepared_cache_rmt(entry2)
    pair_alpha = _prepared_cache_scale(entry1) * _prepared_cache_scale(entry2)
    factors = prepared_sectors[out_pos][2][factor_pos]
    factors_scaled = pair_alpha == one(typeof(pair_alpha)) ? factors : factors .* pair_alpha
    result_view = reshape(result_RMTs[result_pos],
                          size(data1, 1),
                          size(data2, 1),
                          size(factors, 1))
    beta = result_has_contribution[result_pos] ? one(RT) : zero(RT)
    _contract_RMT_pair_into!(result_view, data1, data2, factors_scaled, temp, beta)
    result_has_contribution[result_pos] = true
    return result_RMTs
end

"""
    _fill_cached_contract_side!(cache, q, perm, work_items,
                                ::Val{NF}, ::Val{CN}, ::Val{N}, ::Val{side})
        -> cache

Precompute prepared RMT cache entries for one side of an eager contraction.

`cache` is indexed by original source sector slot. `q` is the source tensor for
the selected side. `perm`, `NF`, `CN`, and `N` describe the grouped preparation
layout. `work_items` is the full contribution list. `Val{side}` chooses whether
`item[1]` or `item[2]` is the sector slot for `q`. The helper leaves unrelated
cache slots unassigned.
"""
function _fill_cached_contract_side!(cache::Vector{Tuple{PRMT, T}},
                                     q::AbstractTLArray{T},
                                     perm::NTuple{RD, Int},
                                     work_items::AbstractVector{NTuple{4, Int}},
                                     ::Val{NF},
                                     ::Val{CN},
                                     ::Val{N},
                                     ::Val{side}) where {PRMT, T, RD, NF, CN, N, side}
    for item in work_items
        idx = side == 1 ? item[1] : item[2]
        _cached_prepared_sector_rmt!(cache, q, idx, perm, Val(NF), Val(CN), Val(N))
    end
    return cache
end

"""
    _contract_with_streaming_side!(result_RMTs, prepared_sectors, work_items,
                                   out_keys, out_to_result, q1, q2, perm1, perm2,
                                   max_stream_len, temp,
                                   ::Val{NF1}, ::Val{NF2}, ::Val{CN},
                                   ::Val{streaming_side})
        -> result_RMTs

Execute eager contraction work using either full prepared caches or one streamed
source side.

`result_RMTs` is filled with allocated and accumulated result payloads.
`prepared_sectors`, `out_keys`, and `out_to_result` describe prepared output
sector metadata and factor blocks. `work_items` is the contribution list sorted
by the streamed side when `streaming_side` is `1` or `2`. `q1` and `q2` are the
source tensors. `perm1` and `perm2` prepare source RMT axes into grouped layout.
`max_stream_len` sizes the reusable streaming buffer. `temp` is the reusable
RMT-pair contraction scratch vector. `NF1`, `NF2`, and `CN` are compile-time
free/contracted leg counts.

When `streaming_side == 1`, source 1 is prepared into a reusable buffer per
sector while source 2 is cached. `streaming_side == 2` does the opposite.
`streaming_side == 0` caches both sides. In every mode, sector slot numbering
is preserved and only sectors represented by `out_to_result` are allocated.
"""
function _contract_with_streaming_side!(
    result_RMTs::Vector{Array{RT, RD_out}},
    prepared_sectors,
    work_items::Vector{NTuple{4, Int}},
    out_keys,
    out_to_result::AbstractVector{Int},
    q1::AbstractTLArray{T1, QD1, N, RD1, QT, PS, M, RMT1},
    q2::AbstractTLArray{T2, QD2, N, RD2, QT, PS, M, RMT2},
    perm1::NTuple{RD1, Int},
    perm2::NTuple{RD2, Int},
    max_stream_len::Int,
    temp::Vector{RT},
    ::Val{NF1},
    ::Val{NF2},
    ::Val{CN},
    ::Val{streaming_side},
) where {RT, RD_out, T1, QD1, N, RD1, QT, PS, M, RMT1,
         T2, QD2, RD2, RMT2, NF1, NF2, CN, streaming_side}
    _allocate_contract_result_rmts!(result_RMTs, prepared_sectors, out_keys,
                                    out_to_result, Val(NF1), Val(NF2))
    result_has_contribution = falses(length(result_RMTs))

    PermutedRMT1 = _permuted_rmt_type(RMT1)
    PermutedRMT2 = _permuted_rmt_type(RMT2)

    if streaming_side == 1
        cache2 = Vector{Tuple{PermutedRMT2, T2}}(undef, sector_count(q2))
        _fill_cached_contract_side!(cache2, q2, perm2, work_items,
                                    Val(NF2), Val(CN), Val(N), Val(2))
        buffer = Vector{T1}(undef, max_stream_len)
        pos = firstindex(work_items)
        while pos <= lastindex(work_items)
            idx1 = work_items[pos][1]
            streamed_entry = _prepare_sector_rmt_into!(buffer, q1, idx1, perm1,
                                                       Val(NF1), Val(CN), Val(N))
            next_pos = pos
            while next_pos <= lastindex(work_items) && work_items[next_pos][1] == idx1
                item = work_items[next_pos]
                entry2 = cache2[item[2]]
                _accumulate_contract_work_item!(result_RMTs, result_has_contribution,
                                                prepared_sectors, out_to_result,
                                                item, streamed_entry, entry2, temp)
                next_pos += 1
            end
            pos = next_pos
        end
    elseif streaming_side == 2
        cache1 = Vector{Tuple{PermutedRMT1, T1}}(undef, sector_count(q1))
        _fill_cached_contract_side!(cache1, q1, perm1, work_items,
                                    Val(NF1), Val(CN), Val(N), Val(1))
        buffer = Vector{T2}(undef, max_stream_len)
        pos = firstindex(work_items)
        while pos <= lastindex(work_items)
            idx2 = work_items[pos][2]
            streamed_entry = _prepare_sector_rmt_into!(buffer, q2, idx2, perm2,
                                                       Val(NF2), Val(CN), Val(N))
            next_pos = pos
            while next_pos <= lastindex(work_items) && work_items[next_pos][2] == idx2
                item = work_items[next_pos]
                entry1 = cache1[item[1]]
                _accumulate_contract_work_item!(result_RMTs, result_has_contribution,
                                                prepared_sectors, out_to_result,
                                                item, entry1, streamed_entry, temp)
                next_pos += 1
            end
            pos = next_pos
        end
    else
        cache1 = Vector{Tuple{PermutedRMT1, T1}}(undef, sector_count(q1))
        cache2 = Vector{Tuple{PermutedRMT2, T2}}(undef, sector_count(q2))
        for item in work_items
            entry1 = _cached_prepared_sector_rmt!(cache1, q1, item[1], perm1,
                                                  Val(NF1), Val(CN), Val(N))
            entry2 = _cached_prepared_sector_rmt!(cache2, q2, item[2], perm2,
                                                  Val(NF2), Val(CN), Val(N))
            _accumulate_contract_work_item!(result_RMTs, result_has_contribution,
                                            prepared_sectors, out_to_result,
                                            item, entry1, entry2, temp)
        end
    end

    return result_RMTs
end

"""
    compute_sectors(q, sector_inds) -> q

Evaluate selected sector slots of a possibly lazy tensor.

`q` is either a concrete `TLArray` or a lazy wrapper such as
`SingletonTLArray`/`TLArrayContraction`. `sector_inds` is a collection of stable
sector slot numbers in the result tensor's own metadata. Repeated indices are
ignored by lazy implementations, and unrelated slots must remain unevaluated.

Concrete tensors are already materialized and are returned unchanged. Lazy
implementations must preserve sector slot numbering exactly: they may fill
requested sectors and mark known-zero sectors, but they must not reorder,
compact, filter, or renumber sector storage.
"""
compute_sectors(q::TLArray, sector_inds::AbstractVector{<:Integer}) = q
function compute_sectors(q::SingletonTLArray, sector_inds::AbstractVector{<:Integer})
    compute_sectors(q.arr, sector_inds)
    return q
end

"""
    _unique_requested_sectors!(out, seen, sector_inds) -> out

Append unique requested sector slots to `out` while preserving first occurrence
order.

`out` is the destination vector and is not cleared by this helper. `seen` is a
scratch `BitVector` with one entry per valid sector slot; it is reset to all
false at the start of the call. `sector_inds` supplies the requested sector
numbers. The helper does not perform bounds checks, so callers must use it only
after choosing a `seen` vector sized for the relevant tensor.
"""
function _unique_requested_sectors!(out::Vector{Int},
                                    seen::BitVector,
                                    sector_inds::AbstractVector{<:Integer})
    fill!(seen, false)
    for sector_raw in sector_inds
        sector = Int(sector_raw)
        seen[sector] && continue
        seen[sector] = true
        push!(out, sector)
    end
    return out
end

"""
    _pending_contraction_work(q, sector_inds)
        -> (requested, work_items, used1, used2)

Filter lazy contraction work to the result sectors requested by the caller.

`q` is the lazy contraction object. `sector_inds` contains result sector slot
numbers in `q`'s metadata. The method validates bounds, removes duplicate
requests, skips sectors already defined or already known zero, and then selects
only `q.work_items` whose result sector is still pending.

The returned `requested` vector contains pending result sector slots in
first-requested order. `work_items` contains matching `(sector1, sector2,
result_sector, factor_slot)` tuples in the original work-list order. `used1`
and `used2` list the unique source sector slots that must be evaluated in the
left and right operands before payload contraction can run.
"""
function _pending_contraction_work(q::TLArrayContraction,
                                   sector_inds::AbstractVector{<:Integer})
    n = sector_count(q)
    requested_seen = falses(n)
    requested = Int[]
    sizehint!(requested, length(sector_inds))
    for sector_raw in sector_inds
        sector = Int(sector_raw)
        1 <= sector <= n || throw(BoundsError(q, sector))
        requested_seen[sector] && continue
        requested_seen[sector] = true
        (q.isdefined[sector] || q.iszero[sector]) && continue
        push!(requested, sector)
    end
    isempty(requested) && return requested, NTuple{4, Int}[], Int[], Int[]

    pending = falses(n)
    for sector in requested
        pending[sector] = true
    end

    filtered = Vector{NTuple{4, Int}}()
    sizehint!(filtered, length(q.work_items))
    used1_seen = falses(sector_count(q.arr1))
    used2_seen = falses(sector_count(q.arr2))
    used1 = Int[]
    used2 = Int[]
    for item in q.work_items
        result_sector = item[3]
        pending[result_sector] || continue
        push!(filtered, item)
        sector1 = item[1]
        if !used1_seen[sector1]
            used1_seen[sector1] = true
            push!(used1, sector1)
        end
        sector2 = item[2]
        if !used2_seen[sector2]
            used2_seen[sector2] = true
            push!(used2, sector2)
        end
    end
    return requested, filtered, used1, used2
end

"""
    compute_sectors(q::TLArrayContraction, sector_inds) -> q

Evaluate selected result sectors of a lazy binary contraction.

`q` owns the result metadata, sector state bits, work-item table, and
precomputed non-Abelian factor blocks. `sector_inds` identifies result slots to
materialize. The method filters pending work, recursively computes only the
needed source sectors on `q.arr1` and `q.arr2`, and then fills the requested
result payloads with `_compute_contraction_sectors_from_sources!`.
"""
function compute_sectors(q::TLArrayContraction, sector_inds::AbstractVector{<:Integer})
    requested, filtered, used1, used2 = _pending_contraction_work(q, sector_inds)
    isempty(requested) && return q
    isempty(filtered) && return q

    arr1 = q.arr1
    arr2 = q.arr2
    compute_sectors(arr1, used1)
    compute_sectors(arr2, used2)
    return _compute_contraction_sectors_from_sources!(q, arr1, arr2, requested,
                                                      filtered, used1, used2)
end

"""
    _compact_cache_index(needed, nsectors) -> Vector{Int}

Build a sector-slot to compact-cache-position lookup table.

`needed` is the list of source sector slots that may be accessed during a lazy
payload fill. `nsectors` is the total number of sectors in that source tensor.
The returned vector has length `nsectors`; entries are zero for sectors not in
`needed`, and otherwise store the one-based position in a compact cache vector.
"""
function _compact_cache_index(needed::AbstractVector{<:Integer}, nsectors::Int)
    slot_to_cache = zeros(Int, nsectors)
    for (cache_pos, sector_raw) in pairs(needed)
        slot_to_cache[Int(sector_raw)] = cache_pos
    end
    return slot_to_cache
end

"""
    _cached_prepared_sector_rmt_compact!(cache, slot_to_cache, q, idx, perm,
                                         ::Val{NF}, ::Val{CN}, ::Val{N})

Return a prepared RMT/cache-scale entry for one source sector using compact
cache storage.

`cache` stores prepared entries only for sectors listed in the corresponding
`needed` vector. `slot_to_cache` maps original sector slot `idx` to the compact
cache position. `q` is the source tensor. `perm` is the stored-leg permutation
that groups free, contracted, and OM axes. `NF`, `CN`, and `N` are compile-time
counts for free legs, contracted legs, and symmetry factors. Missing cache
entries are computed with `prepared_sector_rmt` and stored in place.
"""
function _cached_prepared_sector_rmt_compact!(cache::Vector{Tuple{PRMT, T}},
                                              slot_to_cache::Vector{Int},
                                              q::AbstractTLArray{T},
                                              idx::Int,
                                              perm::NTuple{RD, Int},
                                              ::Val{NF},
                                              ::Val{CN},
                                              ::Val{N}) where {PRMT, T, RD, NF, CN, N}
    cache_pos = slot_to_cache[idx]
    cache_pos > 0 || throw(ArgumentError("sector $idx is not in the compact cache"))
    if !isassigned(cache, cache_pos)
        cache[cache_pos] = prepared_sector_rmt(q, idx, perm, Val(NF), Val(CN), Val(N))
    end
    return cache[cache_pos]
end

"""
    _scale_factor_into!(scratch, factor, scale) -> AbstractArray

Write a scaled copy of a non-Abelian factor block into reusable scratch space.

`scratch` is a flat vector with at least `length(factor)` entries. `factor` is a
three-axis `Float64` block from the contraction factor table. `scale` is the
combined scalar/conjugation scale from the two prepared source RMT entries. The
returned array is a reshaped view into `scratch` with the same dimensions as
`factor`, avoiding allocation when a work item needs a scaled factor block.
"""
function _scale_factor_into!(scratch::Vector{RT},
                             factor::AbstractArray{Float64, 3},
                             scale) where {RT}
    dims = size(factor)
    len = length(factor)
    @assert length(scratch) >= len
    out = reshape(view(scratch, 1:len), dims)
    @inbounds for i in eachindex(factor)
        out[i] = factor[i] * scale
    end
    return out
end

"""
    _lazy_accumulate_contract_work_item!(result_RMTs, result_has_contribution,
                                         factors_by_sector, item, entry1, entry2,
                                         temp, factor_scratch)

Accumulate one lazy contraction work item into its result-sector RMT buffer.

`result_RMTs` contains allocated output RMT arrays indexed by result sector.
`result_has_contribution` records whether each result sector has already
received a contribution and therefore whether the BLAS-like accumulation beta
should be one or zero. `factors_by_sector[result_sector][factor_slot]` is the
non-Abelian factor block selected by `item`. `entry1` and `entry2` are prepared
source RMT/cache-scale tuples. `temp` is contraction scratch space and
`factor_scratch` is used only when the factor block must be scaled by the
prepared source scale product.
"""
function _lazy_accumulate_contract_work_item!(result_RMTs::Vector{Array{RT, RD_out}},
                                              result_has_contribution::BitVector,
                                              factors_by_sector::Vector{Vector{Array{Float64, 3}}},
                                              item::NTuple{4, Int},
                                              entry1,
                                              entry2,
                                              temp::Vector{RT},
                                              factor_scratch::Vector{RT}) where {RT, RD_out}
    _, _, result_pos, factor_pos = item
    data1 = _prepared_cache_rmt(entry1)
    data2 = _prepared_cache_rmt(entry2)
    pair_alpha = _prepared_cache_scale(entry1) * _prepared_cache_scale(entry2)
    factor = factors_by_sector[result_pos][factor_pos]
    K = pair_alpha == one(typeof(pair_alpha)) ?
        factor : _scale_factor_into!(factor_scratch, factor, pair_alpha)
    result_view = reshape(result_RMTs[result_pos],
                          size(data1, 1),
                          size(data2, 1),
                          size(factor, 1))
    beta = result_has_contribution[result_pos] ? one(RT) : zero(RT)
    _contract_RMT_pair_into!(result_view, data1, data2, K, temp, beta)
    result_has_contribution[result_pos] = true
    return result_RMTs
end

"""
    _max_lazy_temp_and_factor_len(q, arr1, arr2, work_items, perm1, perm2,
                                  ::Val{NF1}, ::Val{NF2}, ::Val{CN})
        -> (max_temp_len, max_factor_len)

Compute reusable scratch sizes for a batch of lazy contraction work items.

`q` supplies result factor blocks. `arr1` and `arr2` are the source tensors.
`work_items` are the filtered `(sector1, sector2, result_sector, factor_slot)`
tuples that will be evaluated. `perm1` and `perm2` are the source RMT
preparation permutations. `NF1`, `NF2`, and `CN` carry free-leg and
contracted-leg counts at compile time.

The first result is the maximum temporary vector length needed by
`_contract_RMT_pair_into!` over all selected sector pairs. The second is the
largest factor-block length and sizes `factor_scratch` for scaled factor views.
"""
function _max_lazy_temp_and_factor_len(q::TLArrayContraction{T, QD, N, RD_out},
                                       arr1::AbstractTLArray{T1, QD1, N, RD1, QT, PS, M, RMT1},
                                       arr2::AbstractTLArray{T2, QD2, N, RD2, QT, PS, M, RMT2},
                                       work_items::AbstractVector{NTuple{4, Int}},
                                       perm1::NTuple{RD1, Int},
                                       perm2::NTuple{RD2, Int},
                                       ::Val{NF1},
                                       ::Val{NF2},
                                       ::Val{CN}) where {T, QD, N, RD_out,
                                                         T1, QD1, RD1, QT, PS, M, RMT1,
                                                         T2, QD2, RD2, RMT2, NF1, NF2, CN}
    max_temp_len = 0
    max_factor_len = 0
    PermutedRMT1 = _permuted_rmt_type(RMT1)
    PermutedRMT2 = _permuted_rmt_type(RMT2)
    for item in work_items
        idx1, idx2, result_pos, factor_pos = item
        fdim, cdim1, o1dim =
            _prepared_rmt_dims(arr1, idx1, perm1, Val(NF1), Val(CN), Val(N))
        gdim, cdim2, o2dim =
            _prepared_rmt_dims(arr2, idx2, perm2, Val(NF2), Val(CN), Val(N))
        @assert cdim1 == cdim2
        factor = q.factors[result_pos][factor_pos]
        max_temp_len = max(max_temp_len,
                           _rmt_contract_temp_len(PermutedRMT1, PermutedRMT2,
                                                  fdim, gdim, cdim1,
                                                  o1dim, o2dim, factor))
        max_factor_len = max(max_factor_len, length(factor))
    end
    return max_temp_len, max_factor_len
end

"""
    _compute_contraction_sectors_from_sources!(q, arr1, arr2, requested,
                                               work_items, used1, used2) -> q

Fill selected lazy-contraction result sectors after source sectors are ready.

`q` is the lazy result object whose RMT slots and sector state bits will be
updated. `arr1` and `arr2` are the source tensors; their sector slots listed in
`used1` and `used2` have already been recursively computed. `requested`
contains result sector slots to allocate/fill. `work_items` contains only the
contribution tuples relevant to those result slots.

The function chooses whether to stream prepared RMT data from one side or cache
both sides based on estimated prepared-storage sizes. It allocates output RMTs
only for requested pending sectors, accumulates all independent sector-pair
contributions, and finally publishes completed sectors under `q.lock`. Sector
slot numbering is preserved throughout; no compaction or renumbering occurs.
"""
function _compute_contraction_sectors_from_sources!(
    q::TLArrayContraction{RT, QD_out, N, RD_out},
    arr1::AbstractTLArray{T1, QD1, N, RD1, QT, PS, M, RMT1},
    arr2::AbstractTLArray{T2, QD2, N, RD2, QT, PS, M, RMT2},
    requested::Vector{Int},
    work_items::Vector{NTuple{4, Int}},
    used1::Vector{Int},
    used2::Vector{Int},
) where {RT, QD_out, N, RD_out, T1, QD1, RD1, QT, PS, M, RMT1,
         T2, QD2, RD2, RMT2}
    perm1 = Tuple(q.perm1)::NTuple{RD1, Int}
    perm2 = Tuple(q.perm2)::NTuple{RD2, Int}
    # The stored permutation is (free..., contracted..., om...), so CN is
    # recoverable from the output rank: QD_out = QD1 + QD2 - 2CN.
    CN_val = (QD1 + QD2 - QD_out) ÷ 2
    NF1_val = QD1 - CN_val
    NF2_val = QD2 - CN_val

    can_stream1 = _can_stream_prepared_rmt(arr1, perm1, Val(NF1_val), Val(CN_val), Val(N))
    can_stream2 = _can_stream_prepared_rmt(arr2, perm2, Val(NF2_val), Val(CN_val), Val(N))
    total1, max1 = can_stream1 ?
        _prepared_rmt_storage_size(arr1, used1, perm1, Val(NF1_val), Val(CN_val), Val(N)) : (0, 0)
    total2, max2 = can_stream2 ?
        _prepared_rmt_storage_size(arr2, used2, perm2, Val(NF2_val), Val(CN_val), Val(N)) : (0, 0)
    streaming_side =
        can_stream1 && !can_stream2 ? 1 :
        can_stream2 && !can_stream1 ? 2 :
        !can_stream1 && !can_stream2 ? 0 :
        total1 > total2 || (total1 == total2 && max1 >= max2) ? 1 : 2
    max_stream_len = streaming_side == 1 ? max1 : streaming_side == 2 ? max2 : 0

    if streaming_side == 1
        sort!(work_items; by = item -> item[1], alg = MergeSort)
    elseif streaming_side == 2
        sort!(work_items; by = item -> item[2], alg = MergeSort)
    end

    max_temp_len, max_factor_len =
        _max_lazy_temp_and_factor_len(q, arr1, arr2, work_items, perm1, perm2,
                                      Val(NF1_val), Val(NF2_val), Val(CN_val))
    temp = Vector{RT}(undef, max_temp_len)
    factor_scratch = Vector{RT}(undef, max_factor_len)
    result_RMTs = Vector{Array{RT, RD_out}}(undef, sector_count(q))
    result_has_contribution = falses(sector_count(q))
    for sector in requested
        q.isdefined[sector] && continue
        q.iszero[sector] && continue
        result_RMTs[sector] = Array{RT, RD_out}(undef, q.rmt_sizes[sector])
    end

    PermutedRMT1 = _permuted_rmt_type(RMT1)
    PermutedRMT2 = _permuted_rmt_type(RMT2)
    if streaming_side == 1
        slot_to_cache2 = _compact_cache_index(used2, sector_count(arr2))
        cache2 = Vector{Tuple{PermutedRMT2, T2}}(undef, length(used2))
        buffer = Vector{T1}(undef, max_stream_len)
        pos = firstindex(work_items)
        while pos <= lastindex(work_items)
            idx1 = work_items[pos][1]
            streamed_entry = _prepare_sector_rmt_into!(buffer, arr1, idx1, perm1,
                                                       Val(NF1_val), Val(CN_val), Val(N))
            next_pos = pos
            while next_pos <= lastindex(work_items) && work_items[next_pos][1] == idx1
                item = work_items[next_pos]
                entry2 = _cached_prepared_sector_rmt_compact!(
                    cache2, slot_to_cache2, arr2, item[2], perm2,
                    Val(NF2_val), Val(CN_val), Val(N))
                _lazy_accumulate_contract_work_item!(
                    result_RMTs, result_has_contribution, q.factors,
                    item, streamed_entry, entry2, temp, factor_scratch)
                next_pos += 1
            end
            pos = next_pos
        end
    elseif streaming_side == 2
        slot_to_cache1 = _compact_cache_index(used1, sector_count(arr1))
        cache1 = Vector{Tuple{PermutedRMT1, T1}}(undef, length(used1))
        buffer = Vector{T2}(undef, max_stream_len)
        pos = firstindex(work_items)
        while pos <= lastindex(work_items)
            idx2 = work_items[pos][2]
            streamed_entry = _prepare_sector_rmt_into!(buffer, arr2, idx2, perm2,
                                                       Val(NF2_val), Val(CN_val), Val(N))
            next_pos = pos
            while next_pos <= lastindex(work_items) && work_items[next_pos][2] == idx2
                item = work_items[next_pos]
                entry1 = _cached_prepared_sector_rmt_compact!(
                    cache1, slot_to_cache1, arr1, item[1], perm1,
                    Val(NF1_val), Val(CN_val), Val(N))
                _lazy_accumulate_contract_work_item!(
                    result_RMTs, result_has_contribution, q.factors,
                    item, entry1, streamed_entry, temp, factor_scratch)
                next_pos += 1
            end
            pos = next_pos
        end
    else
        slot_to_cache1 = _compact_cache_index(used1, sector_count(arr1))
        slot_to_cache2 = _compact_cache_index(used2, sector_count(arr2))
        cache1 = Vector{Tuple{PermutedRMT1, T1}}(undef, length(used1))
        cache2 = Vector{Tuple{PermutedRMT2, T2}}(undef, length(used2))
        for item in work_items
            entry1 = _cached_prepared_sector_rmt_compact!(
                cache1, slot_to_cache1, arr1, item[1], perm1,
                Val(NF1_val), Val(CN_val), Val(N))
            entry2 = _cached_prepared_sector_rmt_compact!(
                cache2, slot_to_cache2, arr2, item[2], perm2,
                Val(NF2_val), Val(CN_val), Val(N))
            _lazy_accumulate_contract_work_item!(
                result_RMTs, result_has_contribution, q.factors,
                item, entry1, entry2, temp, factor_scratch)
        end
    end

    lock(q.lock) do
        for sector in requested
            result_has_contribution[sector] || continue
            if !q.isdefined[sector]
                rmt = result_RMTs[sector]
                q.RMTs[sector] = rmt
                q.isdefined[sector] = true
                q.iszero[sector] = _rmt_iszero(rmt)
            end
        end
    end
    return q
end

# ─── * operator ──────────────────────────────────────────────────────────────
# Automatically contract two TLArray objects by matching their tagged, unlocked
# indices.  An index on q1 is "contractible" when it has a nonempty tag AND
# lock == 0; same criterion applies to q2.  Two contractible indices are matched
# when they compare equal under TLIndex == (same itags, dir, plev, dual) and
# their precomputed leg spaces are equal. The collected matching pairs define
# legs1 / legs2 passed to `contract`.
#function _contract_matched_legs(q1::TLArray, q2::TLArray, legs1::Vector{Int}, legs2::Vector{Int})
#    length(legs1) == length(legs2) || throw(DimensionMismatch("matched leg vectors must have the same length"))
#    if length(legs1) == 1
#        return contract(q1, (legs1[1],), q2, (legs2[1],); verify_legs=false)
#    elseif length(legs1) == 2
#        return contract(q1, (legs1[1], legs1[2]), q2, (legs2[1], legs2[2]); verify_legs=false)
#    elseif length(legs1) == 3
#        return contract(q1, (legs1[1], legs1[2], legs1[3]), q2, (legs2[1], legs2[2], legs2[3]); verify_legs=false)
#    else
#        return contract(q1, Tuple(legs1), q2, Tuple(legs2); verify_legs=false)
#    end
#end

"""
    _matched_contract_legs(q1, q2) -> (legs1, legs2)

Find automatically contractible leg pairs between two TLArray-like tensors.

`q1` and `q2` provide `TLIndex` metadata and leg spaces. A candidate leg must
have a nonempty tag string and `lock == 0`. A leg from `q1` matches a leg from
`q2` when the first index equals `change_dir` of the second index and the
visible leg spaces are identical. Each candidate on `q2` can be consumed once.

The method returns tuples of leg positions such that `legs1[k]` contracts with
`legs2[k]`. It throws on ambiguous matches and asserts that at least one pair
exists.
"""
function _matched_contract_legs(q1::AbstractTLArray, q2::AbstractTLArray)
    inds1 = inds(q1)
    inds2 = inds(q2)
    spaces1 = spaces(q1)
    spaces2 = spaces(q2)
    # Collect candidate indices from each TLArray.
    cands1 = [(i, inds1[i]) for i in 1:length(inds1)
              if !isempty(inds1[i].itags) && inds1[i].lock == 0]
    cands2 = [(j, inds2[j]) for j in 1:length(inds2)
              if !isempty(inds2[j].itags) && inds2[j].lock == 0]

    # Match candidates: for each index in cands1, find the unique equal index
    # in cands2.  Raise an error if a tag appears more than once on either side.
    legs1 = Int[]
    legs2 = Int[]
    matched2 = Set{Int}()   # positions in cands2 already consumed

    for (i::Int, idx1) in cands1
        hits = [(pos, j, idx2) for (pos, (j, idx2)) in enumerate(cands2)
                if idx1 == change_dir(idx2) &&
                   spaces1[i] == spaces2[j] &&
                   pos ∉ matched2]::Vector{Tuple{Int, Int, TLIndex}}
        if length(hits) > 1
            error("Ambiguous contraction: tag \"$(idx1.itags)\" matches more than one index in q2")
        end
        if length(hits) == 1
            pos::Int, j::Int, _ = hits[1]
            push!(legs1, i)::Vector{Int}
            push!(legs2, j)::Vector{Int}
            push!(matched2, pos)::Set{Int}
        end
    end

    @assert length(legs1) > 0 "No matching contractible indices found between the two TLArray objects"

    return Tuple(legs1), Tuple(legs2)
end

"""
    contract(q1, q2; reduce_lock=true)
    contract(q1, legs1, q2, legs2; reduce_lock=true)

Contract compatible legs of `q1` and `q2`. The first form discovers matching
unlocked indices; the second explicitly pairs `legs1[i]` with `legs2[i]`, which
must have equal length and compatible spaces/directions. `reduce_lock` decides
whether lock metadata is reduced on the result. The result has uncontracted
logical legs, preserves symmetry-sector conventions, and remains lazy when
inputs or the calling context permit lazy evaluation. Throws `ArgumentError`
for invalid pairings or incompatible indices.
"""
function contract(q1::AbstractTLArray, q2::AbstractTLArray; reduce_lock::Bool=true)
    legs1, legs2 = _matched_contract_legs(q1, q2)
    return contract(q1, legs1, q2, legs2; reduce_lock=reduce_lock)
end

Base.:*(q1::AbstractTLArray, q2::AbstractTLArray) = contract(q1, q2)

# `@lazy` rewrites every `*`. The generic fallback preserves non-tensor products.
@inline _lazy_contract(a, b) = a * b

"""
    _lazy_contract(q1, q2; reduce_lock=true)
    _lazy_contract(q1, legs1, q2, legs2; reduce_lock=true)

Construct lazy contraction metadata without evaluating result RMT payloads.

The two-argument form discovers matching unlocked tagged legs with
`_matched_contract_legs`. The explicit-leg form, defined with the eager
contraction setup below, builds result indices, sector metadata, work items, and
factor tables for the requested leg pairs. `reduce_lock` controls whether
result-index locks are reduced in the same way as eager `contract`.
"""
function _lazy_contract(q1::AbstractTLArray, q2::AbstractTLArray; reduce_lock::Bool=true)
    legs1, legs2 = _matched_contract_legs(q1, q2)
    return _lazy_contract(q1, legs1, q2, legs2; reduce_lock=reduce_lock)
end

# ─── contract ────────────────────────────────────────────────────────────────
# Contraction sorted by *contracted* qlabels.  QR/factor data is prepared per
# output sector, then dense RMT work is executed from a flat sector-pair list.
#
# Algorithm sketch:
#   1. Build sorted sector-info vectors for each TLArray. Each entry carries the
#      sector index, all physical-leg qlabels, and the contracted-qlabel key.
#   2. Count matching contracted-label runs, fill one possible-pair table, and
#      sort it in-place by output free qlabels to form output-sector intervals.
#   3. Build concrete X-symbol caches and reject impossible output sectors by
#      swap-and-pop while collecting w-matrices and source RMT pairs.
#   4. Merge each output sector (QR compression → final RMT contraction).
#   5. Lock reduction / build result TLArray.

# ── Contracted-label helpers ─────────────────────────────────────────────────
"""
    _free_legs(Val(QD), legs) -> NTuple

Return the physical leg positions in `1:QD` that are absent from the contracted
leg tuple `legs`, preserving their original order.
"""
function _free_legs(::Val{QD}, legs::NTuple{CN, Int}) where {QD, CN}
    return ntuple(Val(QD - CN)) do k
        nfree = 0
        for l in 1:QD
            if l ∉ legs
                nfree += 1
                nfree == k && return l
            end
        end
        throw(BoundsError())
    end
end

_contracted_qlabel_type(::Type{QT}, ::Val{CN}) where {QT, CN} = NTuple{CN, Tuple{QT, Int}}

@inline _sector_leg_dim(q::AbstractTLArray, sector::Int, leg::Int) =
    sector_rmt_axis_dim(q, sector, leg)

"""Precomputed sortable descriptor for one contraction sector. It records free/contracted q-label keys and slot information so matching work can be planned without repeatedly inspecting RMT payloads."""
struct ContractSectorInfo{NF, QT, CQT}
    sector_index::Int
    free_qlabels::NTuple{NF, Tuple{QT, Int}}
    contracted_qlabels::CQT
end

"""
    _contract_sector_infos(QT, q, free_legs, legs) -> Vector{ContractSectorInfo}

Build and stable-sort descriptors for the nonzero sectors of `q` by their
contracted q-label/dimension key. The descriptors retain free-leg keys and the
original stable sector slot for subsequent pair planning.
"""
function _contract_sector_infos(::Type{QT}, q::AbstractTLArray{T, QD, N},
                             free_legs::NTuple{NF, Int},
                             legs::NTuple{CN, Int}) where {QT, T, QD, N, NF, CN}
    CQT = _contracted_qlabel_type(QT, Val(CN))
    Info = ContractSectorInfo{NF, QT, CQT}
    infos = Vector{Info}(undef, nsectors(q))
    out_pos = 1
    for i in sector_slots(q)
        is_sector_zero(q, i) && continue
        infos[out_pos] = Info(
            i,
            ntuple(k -> (sector_qlabel(QT, q, i, free_legs[k]), _sector_leg_dim(q, i, free_legs[k])), Val(NF)),
            ntuple(k -> (sector_qlabel(QT, q, i, legs[k]), _sector_leg_dim(q, i, legs[k])), Val(CN))::CQT,
        )
        out_pos += 1
    end
    return sort!(infos; by = info -> info.contracted_qlabels, alg=MergeSort)
end

"""
    _contracted_qlabel_run(infos, first_pos) -> (key, run, next_pos)

Return the contiguous contracted-key run starting at `first_pos` in a sorted
sector-info vector, together with the position immediately after that run.
"""
function _contracted_qlabel_run(infos::AbstractVector{<:ContractSectorInfo{NF, QT, CQT}},
                                first_pos::Int) where {NF, QT, CQT}
    key = infos[first_pos].contracted_qlabels
    next_pos = first_pos + 1
    while next_pos <= lastindex(infos) && infos[next_pos].contracted_qlabels == key
        next_pos += 1
    end
    return key, first_pos:(next_pos - 1), next_pos
end

"""
    _contracted_qlabel_overlaps(infos1, infos2) -> (overlaps, pair_count)

Merge two contracted-key-sorted sector-info vectors. `overlaps` records equal
key runs from both inputs, and `pair_count` is the exact Cartesian-pair count
needed to preallocate the possible-pair table.
"""
function _contracted_qlabel_overlaps(infos1::AbstractVector{ContractSectorInfo{NF1, QT, CQT}},
                                     infos2::AbstractVector{ContractSectorInfo{NF2, QT, CQT}}) where {NF1, NF2, QT, CQT}
    OverlapT = Tuple{CQT, UnitRange{Int}, UnitRange{Int}}
    overlaps = Vector{OverlapT}()
    sizehint!(overlaps, min(length(infos1), length(infos2)))
    total = 0
    pos1 = firstindex(infos1)
    pos2 = firstindex(infos2)
    while pos1 <= lastindex(infos1) && pos2 <= lastindex(infos2)
        ckey1 = infos1[pos1].contracted_qlabels
        ckey2 = infos2[pos2].contracted_qlabels

        if isless(ckey1, ckey2)
            _, _, pos1 = _contracted_qlabel_run(infos1, pos1)
        elseif isless(ckey2, ckey1)
            _, _, pos2 = _contracted_qlabel_run(infos2, pos2)
        else
            _, run1, next_pos1 = _contracted_qlabel_run(infos1, pos1)
            _, run2, next_pos2 = _contracted_qlabel_run(infos2, pos2)
            push!(overlaps, (ckey1, run1, run2))
            total += length(run1) * length(run2)
            pos1 = next_pos1
            pos2 = next_pos2
        end
    end
    return overlaps, total
end

"""
    _fill_possible_pairs!(pairs, infos1, infos2, overlaps) -> pairs

Fill preallocated `pairs` with every source-sector pair sharing a contracted
q-label key. Each entry carries its output free-leg key and both source slots.
"""
function _fill_possible_pairs!(pairs::Vector{PairT},
                               infos1::AbstractVector{<:ContractSectorInfo},
                               infos2::AbstractVector{<:ContractSectorInfo},
                               overlaps::AbstractVector{<:Tuple{CQT, UnitRange{Int}, UnitRange{Int}}}) where {PairT, CQT}
    out_pos = 1
    for (ckey, run1, run2) in overlaps
        for p1 in run1
            info1 = infos1[p1]
            for p2 in run2
                info2 = infos2[p2]
                pairs[out_pos] = (ckey, (info1.free_qlabels, info2.free_qlabels),
                                  info1.sector_index, info2.sector_index)::PairT
                out_pos += 1
            end
        end
    end
    @assert out_pos == length(pairs) + 1
    return pairs
end

"""
    _possible_pair_table(infos1, infos2) -> pairs

Return all compatible source-sector pairs, sorted stably by output free-leg
key so each output sector occupies one contiguous interval.
"""
function _possible_pair_table(infos1::AbstractVector{ContractSectorInfo{NF1, QT, CQT}},
                              infos2::AbstractVector{ContractSectorInfo{NF2, QT, CQT}}) where {NF1, NF2, QT, CQT}
    FreeKey1 = NTuple{NF1, Tuple{QT, Int}}
    FreeKey2 = NTuple{NF2, Tuple{QT, Int}}
    OutKey = Tuple{FreeKey1, FreeKey2}
    PairT = Tuple{CQT, OutKey, Int, Int}
    overlaps, n_possible = _contracted_qlabel_overlaps(infos1, infos2)
    pairs = Vector{PairT}(undef, n_possible)
    _fill_possible_pairs!(pairs, infos1, infos2, overlaps)
    return sort!(pairs; by = p -> p[2], alg = MergeSort)
end

"""
    _out_key_intervals(pairs) -> (out_keys, intervals)

Group an output-key-sorted possible-pair table into unique output keys and
their contiguous pair-table intervals.
"""
function _out_key_intervals(pairs::AbstractVector{Tuple{CQT, OutKey, Int, Int}}) where {CQT, OutKey}
    nout = 0
    pos = firstindex(pairs)
    while pos <= lastindex(pairs)
        key = pairs[pos][2]
        nout += 1
        next_pos = pos + 1
        while next_pos <= lastindex(pairs) && pairs[next_pos][2] == key
            next_pos += 1
        end
        pos = next_pos
    end

    out_keys = Vector{OutKey}(undef, nout)
    intervals = Vector{UnitRange{Int}}(undef, nout)
    out_pos = 1
    pos = firstindex(pairs)
    while pos <= lastindex(pairs)
        key = pairs[pos][2]
        next_pos = pos + 1
        while next_pos <= lastindex(pairs) && pairs[next_pos][2] == key
            next_pos += 1
        end
        out_keys[out_pos] = key
        intervals[out_pos] = pos:(next_pos - 1)
        out_pos += 1
        pos = next_pos
    end

    return out_keys, intervals
end

"""
    _out_key_qlabels(out_key, Val(QD_out)) -> NTuple

Extract output-leg q-labels from a free-leg output key. The dimension-carrying
key variant discards the RMT dimensions while retaining leg order.
"""
function _out_key_qlabels(out_key::Tuple{NTuple{NF1, QT}, NTuple{NF2, QT}},
                          ::Val{QD_out}) where {NF1, NF2, QT, QD_out}
    return ntuple(Val(QD_out)) do leg
        leg <= NF1 ? out_key[1][leg] : out_key[2][leg - NF1]
    end
end

function _out_key_qlabels(out_key::Tuple{NTuple{NF1, Tuple{QT, Int}}, NTuple{NF2, Tuple{QT, Int}}},
                          ::Val{QD_out}) where {NF1, NF2, QT, QD_out}
    return ntuple(Val(QD_out)) do leg
        leg <= NF1 ? out_key[1][leg][1] : out_key[2][leg - NF1][1]
    end
end

"""
    _sector_pairs_from_interval(pairs, interval) -> Vector{Tuple{Int,Int}}

Extract the source sector-slot pairs for one contiguous output interval.
"""
function _sector_pairs_from_interval(pairs::AbstractVector,
                                     interval::UnitRange{Int})
    sector_pairs = Vector{Tuple{Int, Int}}(undef, length(interval))
    out_pos = 1
    for pair_pos in interval
        pair = pairs[pair_pos]
        sector_pairs[out_pos] = (pair[3], pair[4])
        out_pos += 1
    end
    return sector_pairs
end

"""
    _fill_contract_wmat_concat!(concat, xarr, wm1, wm2, col_offset) -> concat

Write one non-Abelian source-pair contribution into the QR input matrix used to
form an output w-matrix. The contribution contracts the X-symbol's two source
outer-multiplicity axes with `wm1` and `wm2`.
"""
function _fill_contract_wmat_concat!(concat::Matrix{Float64},
                                     xarr::AbstractArray{Float64, 3},
                                     wm1::AbstractMatrix,
                                     wm2::AbstractMatrix,
                                     col_offset::Int)
    OM1, OM2, OM3 = size(xarr)
    size(wm1, 1) == OM1 ||
        throw(DimensionMismatch("wm1 first axis must match x-symbol OM1 axis"))
    size(wm2, 1) == OM2 ||
        throw(DimensionMismatch("wm2 first axis must match x-symbol OM2 axis"))

    d1 = size(wm1, 2)
    d2 = size(wm2, 2)
    @inbounds for c in 1:d2, b in 1:d1, a in 1:OM3
        val = 0.0
        for cc in 1:OM2, bb in 1:OM1
            val += xarr[bb, cc, a] * wm1[bb, b] * wm2[cc, c]
        end
        concat[a, col_offset + b + (c - 1) * d1] = val
    end
    return concat
end

@inline _symmetry_qlabels(qlabels::NTuple{NF, QT}, ::Val{n}) where {NF, QT, n} =
    ntuple(k -> qlabels[k][n], Val(NF))

@inline _symmetry_qlabels(qlabels::NTuple{NF, Tuple{QT, Int}}, ::Val{n}) where {NF, QT, n} =
    ntuple(k -> qlabels[k][1][n], Val(NF))

@inline _symmetry_contracted_qlabel(qlabels::NTuple{CN, Tuple{QT, Int}}, ::Val{n}, ::Val{CN}) where {CN, QT, n} =
    ntuple(k -> qlabels[k][1][n], Val(CN))

_xsym_cache_key_type(::Type{QT}, ::Val{n}, ::Val{NF1}, ::Val{NF2}, ::Val{CN}) where {QT, n, NF1, NF2, CN} =
    Tuple{NTuple{CN, fieldtype(QT, n)},
          NTuple{NF1, fieldtype(QT, n)},
          NTuple{NF2, fieldtype(QT, n)}}

const _ABELIAN_WMAT_3D = reshape([1.0], 1, 1, 1)

@generated function _xsym_cache_slot(::Type{PS}, ::Val{n}) where {PS<:ProductSymm, n}
    syms = product_symms(PS)
    1 <= n <= length(syms) || return :(throw(BoundsError(product_symms(PS), $n)))
    syms[n] <: AbelianSymm && return :(nothing)

    target = syms[n]
    slot = 0
    seen = Type[]
    for S in syms
        S <: AbelianSymm && continue
        if !(S in seen)
            push!(seen, S)
            slot += 1
        end
        S == target && return :($slot)
    end
    return :(throw(ArgumentError("symmetry index $n has no X-symbol cache")))
end

@inline _xsym_cache_for(::Type{S}, caches, ::Type{PS}, ::Val{n}) where {S<:AbelianSymm, PS<:ProductSymm, n} = nothing
@inline function _xsym_cache_for(::Type{S}, caches, ::Type{PS}, ::Val{n}) where {S<:NonabelianSymm, PS<:ProductSymm, n}
    slot = _xsym_cache_slot(PS, Val(n))
    slot === nothing &&
        throw(ArgumentError("symmetry index $n is Abelian and has no X-symbol cache"))
    return caches[slot]
end

@inline _xsym_cache_for(caches::CT, ::Type{S}, ::Type{PS}, ::Val{n}) where {CT<:Tuple, S, PS<:ProductSymm, n} =
    _xsym_cache_for(S, caches, PS, Val(n))

@generated function _new_xsym_caches(::Type{PS}, ::Type{QT},
                                     ::Val{NF1}, ::Val{NF2}, ::Val{CN}) where {PS<:ProductSymm, QT, NF1, NF2, CN}
    syms = product_symms(PS)
    seen = Type[]
    exprs = Expr[]
    for (n, S) in pairs(syms)
        S <: AbelianSymm && continue
        S in seen && continue
        push!(seen, S)
        Key = _xsym_cache_key_type(QT, Val(n), Val(NF1), Val(NF2), Val(CN))
        Cache = Dict{Key, Array{Float64, 3}}
        push!(exprs, :($Cache()))
    end
    return Expr(:tuple, exprs...)
end

@inline _impossible_key(free_qlabels1::NTuple{NF1, QT},
                        free_qlabels2::NTuple{NF2, QT},
                        ::Val{n}) where {NF1, NF2, QT, n} =
    (_symmetry_qlabels(free_qlabels1, Val(n)),
     _symmetry_qlabels(free_qlabels2, Val(n)))

@inline _impossible_key(free_qlabels1::NTuple{NF1, Tuple{QT, Int}},
                        free_qlabels2::NTuple{NF2, Tuple{QT, Int}},
                        ::Val{n}) where {NF1, NF2, QT, n} =
    (_symmetry_qlabels(free_qlabels1, Val(n)),
     _symmetry_qlabels(free_qlabels2, Val(n)))

@generated function _new_impossible_sets(::Type{PS}, ::Type{QT},
                                         ::Val{NF1}, ::Val{NF2}) where {PS<:ProductSymm, QT, NF1, NF2}
    syms = product_symms(PS)
    exprs = Vector{Union{Expr, Symbol}}()
    for (n, S) in pairs(syms)
        if S <: AbelianSymm
            push!(exprs, :(nothing))
        else
            K = Tuple{NTuple{NF1, fieldtype(QT, n)}, NTuple{NF2, fieldtype(QT, n)}}
            push!(exprs, :(Set{$K}()))
        end
    end
    return Expr(:tuple, exprs...)
end

@generated function _out_key_is_impossible(::Type{ProductSymm{Syms}},
                                           impossible_sets,
                                           out_key::Tuple{NTuple{NF1, Tuple{QT, Int}}, NTuple{NF2, Tuple{QT, Int}}}) where {Syms, NF1, NF2, QT}
    stmts = Expr[]
    for n in 1:length(Syms.parameters)
        S = Syms.parameters[n]
        S <: AbelianSymm && continue
        push!(stmts, quote
            (_impossible_key(out_key[1], out_key[2], Val($n)) in impossible_sets[$n]) && return true
        end)
    end
    return quote
        $(stmts...)
        return false
    end
end

@inline function _xsym_cache_key(::Type{QT},
                                 contracted_qlabels::NTuple{CN, Tuple{QT, Int}},
                                 free_qlabels1::NTuple{NF1, Tuple{QT, Int}},
                                 free_qlabels2::NTuple{NF2, Tuple{QT, Int}},
                                 ::Val{n}) where {QT, NF1, NF2, n, CN}
    return (_symmetry_contracted_qlabel(contracted_qlabels, Val(n), Val(CN)),
            _symmetry_qlabels(free_qlabels1, Val(n)),
            _symmetry_qlabels(free_qlabels2, Val(n)))
end

@inline _symmetry_sector_qlabels(q::AbstractTLArray{T, QD, N}, sector::Int, ::Val{n}) where {T, QD, N, n} =
    ntuple(l -> sector_qlabel(q, sector, l)[n], Val(QD))

"""
    _stored_symmetry_leg_order(qlabels, inds) -> (stored_to_physical, n_incoming)

Return CGT stored-leg order for one symmetry component: incoming legs sorted by
q-label followed by outgoing legs sorted by q-label. Stable sorting preserves
physical-leg order for equal-direction, equal-q-label ties.
"""
function _stored_symmetry_leg_order(qlabels::NTuple{QD, NTuple{NZ, Int}},
                                    inds::NTuple{QD, TLIndex}) where {QD, NZ}
    incoming = Int[]
    outgoing = Int[]
    sizehint!(incoming, QD)
    sizehint!(outgoing, QD)
    for l in 1:QD
        if inds[l].dir == '+'
            push!(incoming, l)
        else
            push!(outgoing, l)
        end
    end
    sort!(incoming; by = l -> qlabels[l], alg = MergeSort)
    sort!(outgoing; by = l -> qlabels[l], alg = MergeSort)
    n_in = length(incoming)
    return ntuple(i -> i <= n_in ? incoming[i] : outgoing[i - n_in], Val(QD)), n_in
end

@inline _stored_contract_legs(phys_to_stored::NTuple{QD, Int}, legs::NTuple{CN, Int}) where {QD, CN} =
    ntuple(k -> phys_to_stored[legs[k]], Val(CN))

"""
    _xsymbol_sector_args(q, sector, legs, Val(n)) -> (upsp, dnsp, ctlegs)

Translate one tensor sector to the q-label and stored-leg convention required
by the symmetry library's X-symbol lookup for component symmetry `n`.
"""
function _xsymbol_sector_args(q::AbstractTLArray{T, QD, N},
                              sector::Int,
                              legs::NTuple{CN, Int},
                              ::Val{n}) where {T, QD, N, CN, n}
    qlabels_by_phys = _symmetry_sector_qlabels(q, sector, Val(n))
    stored_to_phys, n_in = _stored_symmetry_leg_order(qlabels_by_phys, inds(q))
    qlabels = ntuple(i -> qlabels_by_phys[stored_to_phys[i]], Val(QD))
    phys_to_stored = _phys_to_stored_order(stored_to_phys)
    upsp = Tuple(qlabels[i] for i in 1:n_in)
    dnsp = Tuple(qlabels[i] for i in (n_in + 1):QD)
    ctlegs = _stored_contract_legs(phys_to_stored, legs)
    return upsp, dnsp, ctlegs
end

"""
    _load_nonabelian_xarr(S, q1, i1, q2, i2, legs1, legs2, Val(n)) -> Union{Array{Float64,3},Nothing}

Look up the non-Abelian X-symbol factor for one source-sector pair after
converting both inputs to CGT stored-leg order. Returns `nothing` when the
symmetry coupling is impossible.
"""
function _load_nonabelian_xarr(::Type{S},
                               q1::AbstractTLArray,
                               i1::Int,
                               q2::AbstractTLArray,
                               i2::Int,
                               legs1::NTuple{CN, Int},
                               legs2::NTuple{CN, Int},
                               ::Val{n}) where {S<:NonabelianSymm, CN, n}
    up1sp, dn1sp, ctlegs1 = _xsymbol_sector_args(q1, i1, legs1, Val(n))
    up2sp, dn2sp, ctlegs2 = _xsymbol_sector_args(q2, i2, legs2, Val(n))
    xsym_obj = getNsave_Xsymbol(S, up1sp, dn1sp, up2sp, dn2sp, ctlegs1, ctlegs2)
    isnothing(xsym_obj) && return nothing
    return xsym_obj.xsym_arr::Array{Float64, 3}
end

"""
    _xarr_for_symmetry(S, q1, i1, q2, i2, ..., cache, impossible_sets, ..., Val(n))
        -> Union{Array{Float64,3},Nothing}

Fetch or cache one non-Abelian X-symbol factor. Impossible free-leg keys are
memoized in `impossible_sets` so later source pairs can be rejected without a
symmetry-library lookup.
"""
@inline function _xarr_for_symmetry(::Type{S},
                                    q1::AbstractTLArray,
                                    i1::Int,
                                    q2::AbstractTLArray,
                                    i2::Int,
                                    contracted_qlabels::NTuple{CN, Tuple{QT, Int}},
                                    free_qlabels1::NTuple{NF1, Tuple{QT, Int}},
                                    free_qlabels2::NTuple{NF2, Tuple{QT, Int}},
                                    xsym_cache::Dict{K, Array{Float64, 3}},
                                    impossible_sets::Tuple,
                                    legs1::NTuple{CN, Int},
                                    legs2::NTuple{CN, Int},
                                    ::Val{n}) where {S<:NonabelianSymm, QT, NF1, NF2, CN, n, K}
    xkey = _xsym_cache_key(QT, contracted_qlabels, free_qlabels1, free_qlabels2, Val(n))
    if haskey(xsym_cache, xkey)
        return xsym_cache[xkey]
    end

    loaded = _load_nonabelian_xarr(S, q1, i1, q2, i2, legs1, legs2, Val(n))
    if loaded === nothing
        push!(impossible_sets[n], _impossible_key(free_qlabels1, free_qlabels2, Val(n)))
        return nothing
    end
    xsym_cache[xkey] = loaded
    return loaded
end

"""
    _qr_contract_concat(concat, pair_shapes) -> Union{Tuple{Matrix,Vector},Nothing}

Factor a concatenated non-Abelian contraction matrix into an isometric output
w-matrix and one reduced factor block per source pair. Rank-zero input returns
`nothing`, marking the output sector as impossible.
"""
function _qr_contract_concat(concat::Matrix{Float64},
                             pair_shapes::Vector{Tuple{Int, Int}})
    nrows = size(concat, 1)
    if nrows == 1
        Q = ones(1, 1)
        factors = Vector{Array{Float64, 3}}(undef, length(pair_shapes))
        col = 0
        for i in eachindex(pair_shapes)
            d1, d2 = pair_shapes[i]
            width = d1 * d2
            factors[i] = reshape(copy(concat[:, col+1:col+width]), 1, d1, d2)
            col += width
        end
        return Q, factors
    end

    F = qr(concat)
    Qfull = Matrix(F.Q)
    Rfull = Matrix(F.R)

    used = 0
    @inbounds for i in axes(Rfull, 1)
        nrm_sq = 0.0
        for j in axes(Rfull, 2)
            nrm_sq += abs2(Rfull[i, j])
        end
        !iszero(nrm_sq) && (used = i)
    end
    used == 0 && return nothing

    Q = Qfull[:, 1:used]
    R = Rfull[1:used, :]
    factors = Vector{Array{Float64, 3}}(undef, length(pair_shapes))
    col = 0
    for i in eachindex(pair_shapes)
        d1, d2 = pair_shapes[i]
        width = d1 * d2
        factors[i] = reshape(copy(R[:, col+1:col+width]), used, d1, d2)
        col += width
    end
    @assert col == size(R, 2)

    return Q, factors
end

"""
    _prepare_contract_interval_for_symmetry(S, q1, q2, possible_pairs, interval, ...)
        -> Union{Tuple{Matrix,Vector},Nothing}

Prepare one component symmetry over an output-sector pair interval. Abelian
components return the trivial isometry/factors; non-Abelian components combine
X-symbols and source w-matrices, then QR-compress the result.
"""
function _prepare_contract_interval_for_symmetry(::Type{S},
                                                 q1::AbstractTLArray,
                                                 q2::AbstractTLArray,
                                                 possible_pairs::AbstractVector,
                                                 interval::UnitRange{Int},
                                                 xsym_caches::Tuple,
                                                 impossible_sets::Tuple,
                                                 legs1::NTuple{CN, Int},
                                                 legs2::NTuple{CN, Int},
                                                 ::Val{n}) where {S<:AbelianSymm, CN, n}
    K = length(interval)
    Q = ones(1, 1)
    factors = Vector{Array{Float64, 3}}(undef, K)
    for i in 1:K
        factors[i] = _ABELIAN_WMAT_3D
    end
    return Q, factors
end

function _prepare_contract_interval_for_symmetry(::Type{S},
                                                 q1::AbstractTLArray,
                                                 q2::AbstractTLArray,
                                                 possible_pairs::AbstractVector,
                                                 interval::UnitRange{Int},
                                                 xsym_caches::Tuple,
                                                 impossible_sets::Tuple,
                                                 legs1::NTuple{CN, Int},
                                                 legs2::NTuple{CN, Int},
                                                 ::Val{n}) where {S<:NonabelianSymm, CN, n}
    K = length(interval)
    xarrs = Vector{Array{Float64, 3}}(undef, K)
    pair_shapes = Vector{Tuple{Int, Int}}(undef, K)
    nrows = 0
    total_width = 0

    i = 1
    for pair_pos in interval
        pair = possible_pairs[pair_pos]
        contracted_qlabels = pair[1]
        free_qlabels1, free_qlabels2 = pair[2]
        i1 = pair[3]
        i2 = pair[4]
        xarr = _xarr_for_symmetry(S, q1, i1, q2, i2,
                                  contracted_qlabels, free_qlabels1, free_qlabels2,
                                  _xsym_cache_for(xsym_caches, S, productsymm(q1), Val(n)),
                                  impossible_sets, legs1, legs2, Val(n))
        xarr === nothing && return nothing
        xarrs[i] = xarr

        wm1 = sector_wmat(q1, i1, Val(n))
        wm2 = sector_wmat(q2, i2, Val(n))
        d1 = size(wm1, 2)
        d2 = size(wm2, 2)
        pair_shapes[i] = (d1, d2)
        nrows == 0 && (nrows = size(xarr, 3))
        @assert nrows == size(xarr, 3)
        total_width += d1 * d2
        i += 1
    end

    concat = Matrix{Float64}(undef, nrows, total_width)
    col = 0
    i = 1
    for pair_pos in interval
        pair = possible_pairs[pair_pos]
        i1 = pair[3]
        i2 = pair[4]
        wm1 = sector_wmat(q1, i1, Val(n))
        wm2 = sector_wmat(q2, i2, Val(n))
        _fill_contract_wmat_concat!(concat, xarrs[i], wm1, wm2, col)
        col += pair_shapes[i][1] * pair_shapes[i][2]
        i += 1
    end
    @assert col == total_width

    return _qr_contract_concat(concat, pair_shapes)
end

@generated function _prepare_contract_interval(::Type{ProductSymm{Syms}},
                                               q1::AbstractTLArray{T1, QD1, N, RD1, QT, PS, M, RMT1},
                                               q2::AbstractTLArray{T2, QD2, N, RD2, QT, PS, M, RMT2},
                                               possible_pairs::AbstractVector,
                                               interval::UnitRange{Int},
                                               xsym_caches::CT,
                                               impossible_sets::Tuple,
                                               legs1::NTuple{CN, Int},
                                               legs2::NTuple{CN, Int}) where {Syms, T1, QD1, N, RD1, QT, PS, M, RMT1,
                                                                              T2, QD2, RD2, RMT2, CT<:Tuple, CN}
    stmts = Expr[]
    qnames = Symbol[]
    fnames = Symbol[]
    for n in 1:N
        S = Syms.parameters[n]
        res = Symbol(:prepared_sym_, n)
        q = Symbol(:Q_, n)
        f = Symbol(:factors_, n)
        push!(qnames, q)
        push!(fnames, f)
        push!(stmts, quote
            local $res = _prepare_contract_interval_for_symmetry(
                $S, q1, q2, possible_pairs, interval, xsym_caches,
                impossible_sets, legs1, legs2, Val($n))
            $res === nothing && return nothing
            local $q = $res[1]::Matrix{Float64}
            local $f = $res[2]::Vector{Array{Float64, 3}}
        end)
    end
    factor_tuple = Expr(:tuple, (:(SV_split[$n][i]) for n in 1:N)...)
    zero_rank_sizes = Expr(:tuple, (0 for _ in 1:N)...)

    return quote
        local K = length(interval)
        K > 0 || throw(ArgumentError("contract interval must contain at least one sector pair"))
        $(stmts...)
        local U_mats = Matrix{Float64}[$(qnames...)]
        local SV_split = ($(fnames...),)
        local factor_arrays = Vector{Array{Float64, 3}}(undef, K)
        local first_rank_sizes = $zero_rank_sizes
        for i in 1:K
            factor_arrays[i], rank_sizes =
                _rmt_factor_array_and_rank_sizes($factor_tuple)
            i == 1 && (first_rank_sizes = rank_sizes)
        end
        return U_mats, factor_arrays, first_rank_sizes
    end
end

# ── Convenience overloads ─────────────────────────────────────────────────────
contract(q1, l1::Int, q2, l2::Int) = contract(q1, (l1,), q2, (l2,))

"""
    contract(q1, legs1, q2, legs2; reduce_lock=true, verify_legs=true) -> TLArray

Eagerly contract matching physical-leg tuples from `q1` and `q2`. The result
has the uncontracted legs of `q1` followed by those of `q2`; payload sectors are
fully materialized before return. `reduce_lock` updates matched index locks and
`verify_legs` checks directions, tags, dual flags, and spaces before planning.
"""
function contract(q1::AbstractTLArray, legs1::AbstractVector{<:Integer},
                  q2::AbstractTLArray, legs2::AbstractVector{<:Integer}; kwargs...)
    return contract(q1, Tuple(legs1), q2, Tuple(legs2); kwargs...)
end

function contract(q1::AbstractTLArray{T1, QD1, N, RD1, QT, PS, M, RMT1},
                  legs1::NTuple{CN, Int},
                  q2::AbstractTLArray{T2, QD2, N, RD2, QT, PS, M, RMT2},
                  legs2::NTuple{CN, Int};
                  reduce_lock::Bool=true,
                  verify_legs::Bool=true) where {T1, T2, QD1, QD2, N, RD1, RD2, QT, PS, M, RMT1, RMT2, CN}
    q = _lazy_contract(q1, legs1, q2, legs2;
                       reduce_lock=reduce_lock,
                       verify_legs=verify_legs)
    compute_sectors(q, sector_slots(q))
    return TLArray(symm(q), stored_qlabels(q), q.wmatdata, q.wmatinfo, q.RMTs,
                   stored_inds(q), stored_spaces(q))
end

# ── Main lazy-contraction constructor ─────────────────────────────────────────
"""
    _lazy_contract(q1, legs1, q2, legs2; reduce_lock=true, verify_legs=true)
        -> TLArrayContraction

Plan an explicit-leg contraction without evaluating RMT payloads. It constructs
stable output-sector metadata, QR-normalized w-matrices, source work items, and
symmetry factors; `compute_sectors` later materializes selected result slots.
"""
function _lazy_contract(q1::AbstractTLArray{T1, QD1, N, RD1, QT, PS, M, RMT1},
                        legs1::NTuple{CN, Int},
                        q2::AbstractTLArray{T2, QD2, N, RD2, QT, PS, M, RMT2},
                        legs2::NTuple{CN, Int};
                        reduce_lock::Bool=true,
                        verify_legs::Bool=true) where {T1, T2, QD1, QD2, N, RD1, RD2, QT, PS, M, RMT1, RMT2, CN}

    symmetries = product_symms(PS)
    inds1 = inds(q1)
    inds2 = inds(q2)
    spaces1 = spaces(q1)
    spaces2 = spaces(q2)

    if verify_legs
        for i in 1:CN
            idx1::TLIndex = inds1[legs1[i]::Int]
            idx2::TLIndex = inds2[legs2[i]::Int]
            @assert idx1.dir != idx2.dir "Contracted legs must have opposite arrow directions: " *
                "q1 leg $(legs1[i]) has dir='$(idx1.dir)', q2 leg $(legs2[i]) has dir='$(idx2.dir)'"
            @assert idx1.itags == idx2.itags "Contracted legs must have matching itags: " *
                "q1 leg $(legs1[i]) has itag='$(idx1.itags)', q2 leg $(legs2[i]) has itag='$(idx2.itags)'"
            @assert idx1.dual == idx2.dual "Contracted legs must have matching dual flags: " *
                "q1 leg $(legs1[i]) has dual=$(idx1.dual), q2 leg $(legs2[i]) has dual=$(idx2.dual)"
            @assert spaces1[legs1[i]] == spaces2[legs2[i]] "Contracted legs must have matching space info: " *
                "q1 leg $(legs1[i]) spaces != q2 leg $(legs2[i]) spaces"
        end
    end

    T    = promote_type(T1, T2)

    nf1 = QD1 - CN
    nf2 = QD2 - CN
    free1::NTuple{nf1, Int}  = _free_legs(Val(QD1), legs1)
    free2::NTuple{nf2, Int}  = _free_legs(Val(QD2), legs2)
    QD_out = QD1 + QD2 - 2CN
    RD_out = QD_out + N

    inds_out = (ntuple(i -> inds1[free1[i]], Val(QD1 - CN))...,
                ntuple(i -> inds2[free2[i]], Val(QD2 - CN))...)

    # Fixed permutations: (free..., contracted..., om...).
    perm1 = (free1..., legs1..., ntuple(n -> QD1 + n, Val(N))...)
    perm2 = (free2..., legs2..., ntuple(n -> QD2 + n, Val(N))...)
    #println(perm1)
    #println(perm2)

    # ── 1. Build sorted sector-info vectors keyed by contracted qlabels ─────────
    sector_infos1 = _contract_sector_infos(QT, q1, free1, legs1)
    sector_infos2 = _contract_sector_infos(QT, q2, free2, legs2)

    # ── 2. Output-sector accumulator ────────────────────────────────────────
    FreeKey1  = NTuple{nf1, Tuple{QT, Int}}
    FreeKey2  = NTuple{nf2, Tuple{QT, Int}}
    OutKey    = Tuple{FreeKey1, FreeKey2}

    xsym_caches = _new_xsym_caches(PS, QT, Val(nf1), Val(nf2), Val(CN))
    impossible_sets = _new_impossible_sets(PS, QT, Val(nf1), Val(nf2))

    # ── 3. Collect possible pairs and output intervals ───────────────────────
    possible_pairs = _possible_pair_table(sector_infos1, sector_infos2)
    out_keys, out_intervals = _out_key_intervals(possible_pairs)
    valid_output = trues(length(out_keys))

    # ── 4. Prepare QR/K data and lazy contraction work items ─────────────────
    PreparedSectorT = Tuple{Vector{Matrix{Float64}},
                            Vector{Array{Float64, 3}},
                            NTuple{N, Int}}
    prepared_sectors = Vector{PreparedSectorT}(undef, length(out_keys))
    work_items = Vector{NTuple{4, Int}}()
    sizehint!(work_items, length(possible_pairs))
    for out_pos in eachindex(out_keys)
        valid_output[out_pos] || continue

        if _out_key_is_impossible(PS, impossible_sets, out_keys[out_pos])
            valid_output[out_pos] = false
            continue
        end

        prepared = _prepare_contract_interval(
            PS, q1, q2, possible_pairs, out_intervals[out_pos],
            xsym_caches, impossible_sets, legs1, legs2)
        if isnothing(prepared)
            valid_output[out_pos] = false
            continue
        end
        _, factor_arrays, _ = prepared

        sector_pairs = _sector_pairs_from_interval(possible_pairs, out_intervals[out_pos])
        @assert length(sector_pairs) == length(factor_arrays)
        for i in eachindex(sector_pairs)
            idx1, idx2 = sector_pairs[i]
            push!(work_items, (idx1, idx2, out_pos, i))
        end

        prepared_sectors[out_pos] = prepared
    end

    # ── 5. Merge each output sector ──────────────────────────────────────────
    RT = promote_type(T1, T2, Float64)
    res_nsectors = count(valid_output)
    out_to_result = zeros(Int, length(out_keys))
    result_pos = 1
    for out_pos in eachindex(out_keys)
        valid_output[out_pos] || continue
        out_to_result[out_pos] = result_pos
        result_pos += 1
    end
    result_qlabels = Matrix{QT}(undef, QD_out, res_nsectors)
    result_RMTs = Vector{Array{RT, RD_out}}(undef, res_nsectors)
    nonabelian_indices = nonabelian_symmetry_indices(PS)
    result_wmatinfo = Vector{WMatInfo{M}}(undef, res_nsectors)
    total_wmat_len = 0
    for out_pos in eachindex(out_keys)
        valid_output[out_pos] || continue
        result_pos = out_to_result[out_pos]
        U_mats = prepared_sectors[out_pos][1]
        result_wmatinfo[result_pos] = ntuple(Val(M)) do m
            n = nonabelian_indices[m]
            U = U_mats[n]
            len = length(U)
            offset = total_wmat_len + 1
            total_wmat_len += len
            (offset, size(U, 1), size(U, 2))
        end
    end
    result_wmatdata = Vector{Float64}(undef, total_wmat_len)
    result_rmt_sizes = Vector{NTuple{RD_out, Int}}(undef, res_nsectors)
    result_factors = Vector{Vector{Array{Float64, 3}}}(undef, res_nsectors)
    for out_pos in eachindex(out_keys)
        valid_output[out_pos] || continue
        result_pos = out_to_result[out_pos]
        out_qlabels = _out_key_qlabels(out_keys[out_pos], Val(QD_out))
        for leg in 1:QD_out
            result_qlabels[leg, result_pos] = out_qlabels[leg]
        end

        prepared = prepared_sectors[out_pos]
        result_factors[result_pos] = prepared[2]
        rank_sizes = prepared[3]
        kept_sizes1 = ntuple(i -> out_keys[out_pos][1][i][2], Val(nf1))
        kept_sizes2 = ntuple(i -> out_keys[out_pos][2][i][2], Val(nf2))
        result_rmt_sizes[result_pos] = (kept_sizes1..., kept_sizes2..., rank_sizes...)
        U_mats = prepared[1]
        for m in 1:M
            n = nonabelian_indices[m]
            U = U_mats[n]
            offset, nrow, ncol = result_wmatinfo[result_pos][m]
            copyto!(view(result_wmatdata, offset:offset + nrow * ncol - 1), vec(U))
        end
    end

    # ── 6. Lock reduction ────────────────────────────────────────────────────
    changed_inds2 = Set(change_dir(inds2[l]) for l in 1:QD2)
    changed_inds1 = Set(change_dir(inds1[l]) for l in 1:QD1)
    final_inds = if reduce_lock
        ntuple(Val(QD_out)) do l
            idx = inds_out[l]
            has_match = l <= nf1 ? (idx ∈ changed_inds2) : (idx ∈ changed_inds1)
            (idx.lock > 0 && has_match) ?
                TLIndex(idx.itags, idx.dir, idx.plev, idx.lock - 1, idx.dual) : idx
        end
    else
        inds_out
    end

    spaces_out = (ntuple(i -> spaces1[free1[i]], Val(QD1 - CN))...,
                  ntuple(i -> spaces2[free2[i]], Val(QD2 - CN))...)

    result_qlabel_vec = _qlabel_vector(result_qlabels, Val(QD_out))
    lazy_work_items = Vector{NTuple{4, Int}}()
    sizehint!(lazy_work_items, length(work_items))
    for item in work_items
        result_pos = out_to_result[item[3]]
        result_pos == 0 && continue
        push!(lazy_work_items, (item[1], item[2], result_pos, item[4]))
    end
    result_isdefined = falses(res_nsectors)
    result_iszero = falses(res_nsectors)
    return TLArrayContraction{promote_type(T1, T2, Float64), QD_out, N, RD_out,
                              QT, PS, M,
                              Array{promote_type(T1, T2, Float64), RD_out}}(
        result_qlabel_vec,
        result_wmatdata,
        result_wmatinfo,
        result_RMTs,
        result_isdefined,
        result_iszero,
        final_inds,
        spaces_out,
        q1,
        q2,
        lazy_work_items,
        result_factors,
        collect(Int, perm1),
        collect(Int, perm2),
        result_rmt_sizes,
        ReentrantLock())
end
