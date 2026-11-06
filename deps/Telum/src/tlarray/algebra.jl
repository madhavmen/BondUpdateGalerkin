"""
    *(q::AbstractTLArray, fac::Number)
    *(fac::Number, q::AbstractTLArray)
    /(q::AbstractTLArray, fac::Number)
    -(q::AbstractTLArray)

Return a scaled lazy view of `q`.

The scalar factor is stored in the tensor view state through `_view_scale`
rather than eagerly multiplying RMT payloads. This keeps qlabel, w-matrix, and
RMT storage unchanged.
"""
Base.:*(qs::AbstractTLArray, fac::Number) = _view_scale(qs, fac)
Base.:*(fac::Number, qs::AbstractTLArray) = qs * fac
Base.:/(qs::AbstractTLArray, fac::Number) = qs * (1 / fac)
Base.:-(qs::AbstractTLArray) = qs * -1

# Return a deep copy of a TLArray (CGT metadata, RMTs, indices, spaces all copied).
"""
    copy(q::TLArray) -> TLArray

Copy a concrete tensor.

All owned metadata and payload storage are deep-copied while the deferred
logical `conj`, `scale`, and `perm` state is retained. Lazy evaluation objects
intentionally have no `copy` method.
"""
Base.copy(q::TLArray) = deepcopy(q)

"""
    _identity_on_tlarray(q::AbstractTLArray) -> TLArray

Construct an identity tensor compatible with a rank-2 tensor `q`.

`q` must have exactly one incoming and one outgoing leg, and the two visible
space lists must match. The result uses `getIdentity` on the outgoing leg and is
then reordered to the incoming/outgoing index order of `q`. This helper supports
scalar add/subtract on matrix-like TLArrays.
"""
function _identity_on_tlarray(q::AbstractTLArray{T, QD, N, RD}) where {T, QD, N, RD}
    @assert QD == 2 "Scalar add/subtract is only defined for rank-2 TLArray objects"

    in_legs  = findlegs(q; dir='+')
    out_legs = findlegs(q; dir='-')
    @assert length(in_legs) == 1 && length(out_legs) == 1 "Scalar add/subtract requires exactly one incoming and one outgoing leg"

    in_leg  = only(in_legs)
    out_leg = only(out_legs)
    legspaces = spaces(q)
    qinds = inds(q)
    @assert legspaces[in_leg] == legspaces[out_leg] "Scalar add/subtract requires matching incoming and outgoing spaces"

    id_q = getIdentity((q, out_leg); itag=qinds[out_leg].itags)
    return TLArray(id_q, (qinds[in_leg], qinds[out_leg]))
end

# ─── TLArray norm ─────────────────────────────────────────────────────────────
"""
    norm(q::TLArray) -> Float64

Compute the Frobenius norm from TLArray sector RMTs.

The method uses Wigner-Eckart orthogonality: different qlabel sectors are
orthogonal, CGT canonical-basis columns are orthonormal after the left-orthogonal
w-matrix, and therefore cross-terms vanish. The norm is
`sqrt(sum(abs2(scale) * sum(abs2, rmt)))` over nonzero sectors, without
constructing the full dense tensor.
"""
function LinearAlgebra.norm(q::TLArray{T, QD, N}) where {T, QD, N}
    s = zero(Float64)
    for sector_index in sector_slots(q)
        q.iszero[sector_index] && continue
        rmt, scale = sector_rmt(q, sector_index)
        s += abs2(scale) * sum(abs2, rmt)
    end
    return sqrt(s)
end

function _unsymmetric_matrix(q::TLArray{T, 2, 0}) where {T}
    size(q, 1) == size(q, 2) || throw(ArgumentError("linear algebra operation requires a square rank-2 TLArray"))
    return reshape(_unsymmetric_dense_data(q), size(q, 1), size(q, 2))
end

function LinearAlgebra.diag(q::TLArray{T, 2, 0}) where {T}
    return collect(LinearAlgebra.diag(
        reshape(_unsymmetric_dense_data(q), size(q, 1), size(q, 2))))
end

LinearAlgebra.tr(q::TLArray{T, 2, 0}) where {T} = LinearAlgebra.tr(_unsymmetric_matrix(q))
LinearAlgebra.det(q::TLArray{T, 2, 0}) where {T} = LinearAlgebra.det(_unsymmetric_matrix(q))

function LinearAlgebra.inv(q::TLArray{T, 2, 0}) where {T}
    inv_inds = (change_dir(inds(q)[2]), change_dir(inds(q)[1]))
    return TLArray(LinearAlgebra.inv(_unsymmetric_matrix(q)), inv_inds)
end

"""
    norm(q::TLArrayContraction) -> Float64

Compute the Frobenius norm of a lazy contraction.

All contraction sectors are first computed in place. The final accumulation is
the same RMT-only orthogonality formula used for concrete `TLArray` values.
"""
function LinearAlgebra.norm(q::TLArrayContraction)
    compute_sectors(q, sector_slots(q))
    s = zero(Float64)
    for sector_index in sector_slots(q)
        q.iszero[sector_index] && continue
        rmt, scale = sector_rmt(q, sector_index)
        s += abs2(scale) * sum(abs2, rmt)
    end
    return sqrt(s)
end

Base.conj(q::AbstractTLArray) = _view_conj(q)
Base.adjoint(q::AbstractTLArray) = conj(q)
