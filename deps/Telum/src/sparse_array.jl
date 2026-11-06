"""
    to_sparse_array(q::AbstractTLArray[, FT]) -> SparseArray

Expand a symmetry-reduced tensor into a `SparseArrayKit.SparseArray`.

`q` supplies TLArray sector metadata, CGT/w-matrix data, and RMT payloads.
`FT` chooses the floating/scalar type used for the expanded entries; when it is
omitted, `eltype(q)` is used. The output uses logical visible leg order, not
stored RMT order. Deferred conjugation, scale, and physical permutation are
applied while expanding sector payloads.

This function is intended for validation and interoperability. It explicitly
constructs CGT canonical bases and dense sector blocks, so it should not be used
in performance-sensitive tensor-network contractions.
"""
function to_sparse_array(q::AbstractTLArray{T, QD, N, RD}, ::Type{FT}) where {T, QD, N, RD, FT}
    materialize(q)
    symmetries = symm(q)
    leg_info = [_sparse_leg_offsets(symmetries, spaces(q)[leg]) for leg in 1:QD]
    leg_offsets = first.(leg_info)
    result = SparseArray(zeros(FT, last.(leg_info)...))

    for sector in sector_slots(q)
        is_sector_zero(q, sector) && continue
        cgt_wmats = Vector{Array{FT}}(undef, N)
        for n in 1:N
            symmetry = symmetries[n]
            qlabels, cgp, legdir = _sector_cgt_metadata(q, sector, n)
            wmat = sector_wmat(q, sector, n)
            om = size(wmat, 1)
            bond_dim = size(wmat, 2)

            if isabelian(symmetry)
                @assert bond_dim == 1 "unexpected bond dimension for Abelian symmetry $n"
                cgt_wmats[n] = reshape(FT.(wmat), ones(Int, QD)..., bond_dim)
                continue
            end

            nin, nout = legdir
            incoming = Tuple(qlabels[leg] for leg in 1:nin)
            outgoing = Tuple(qlabels[leg] for leg in nin + 1:QD)
            incoming, _ = remove_zeros(symmetry, incoming)
            outgoing, _ = remove_zeros(symmetry, outgoing)
            cgtom = get_CGTom(symmetry, incoming, outgoing)
            @assert cgtom.totalOM == om "outer multiplicity mismatch for symmetry $n"

            canonical_basis = LurCGT.get_canonical_basis(symmetry,
                                                          Tuple(qlabels[leg] for leg in 1:nin),
                                                          Tuple(qlabels[leg] for leg in nin + 1:QD),
                                                          cgtom)
            cgt_shape = size(Array(canonical_basis[1]))
            cgt = zeros(FT, cgt_shape..., om)
            selectors = ntuple(_ -> Colon(), Val(QD))
            for i in 1:om
                cgt[selectors..., i] .= Array(canonical_basis[i])
            end
            cgt_wmat = reshape(reshape(cgt, :, om) * wmat, cgt_shape..., bond_dim)
            cgt_wmats[n] = permutedims(cgt_wmat, (cgp..., QD + 1))
        end

        cgt_block = if N == 0
            ones(FT, ones(Int, QD)..., 1)
        else
            block = cgt_wmats[1]
            for n in 2:N
                block = _sparse_leg_kron(block, cgt_wmats[n], QD)
            end
            block
        end
        bond_dim = size(cgt_block, QD + 1)
        rmt, scale = sector_rmt_permuted(q, sector, _identity_rmt_perm(Val(RD)))
        rmt_merged = reshape(Array(scale .* rmt), size(rmt)[1:QD]..., bond_dim)
        block = zeros(FT, ntuple(leg -> size(cgt_block, leg) * size(rmt_merged, leg), Val(QD)))
        selectors = ntuple(_ -> Colon(), Val(QD))
        for bond in 1:bond_dim
            block .+= _sparse_legwise_kron(cgt_block[selectors..., bond],
                                            rmt_merged[selectors..., bond])
        end
        ranges = ntuple(leg -> leg_offsets[leg][sector_qlabel(q, sector, leg)], Val(QD))
        result[ranges...] .+= block
    end
    return result
end

"""
    to_sparse_array(q::AbstractTLArray) -> SparseArray
    to_sparse_array(q::AbstractTLArray, ::Type{FT}) -> SparseArray

Convenience overloads for sparse expansion.

The two-argument form forces the expanded scalar type `FT`.
"""
to_sparse_array(q::AbstractTLArray) = to_sparse_array(q, eltype(q))

"""
    _sparse_leg_offsets(symmetries, splist) -> (offsets, total_dim)

Compute dense basis ranges for each qlabel sector on one physical leg.

`symmetries` is the product symmetry tuple and `splist` contains
`(qlabel, rmt_dim)` entries for one TLArray leg. Each sector's expanded size is
`rmt_dim` times the product of non-Abelian irrep dimensions. Abelian symmetries
contribute dimension one. The returned `offsets[qlabel]` ranges partition the
expanded leg dimension in sorted qlabel order.
"""
function _sparse_leg_offsets(symmetries, splist)
    QT = typeof(first(splist)[1])
    sector_sizes = Dict{QT, Int}()
    for (qlabel, rmt_dim) in splist
        irrep_dim = prod(isabelian(symmetries[n]) ? 1 :
                         dimension(LurCGT.getNsave_irep(symmetries[n], BigInt, qlabel[n]))
                         for n in eachindex(symmetries); init=1)
        sector_sizes[qlabel] = irrep_dim * rmt_dim
    end
    offsets = Dict{QT, UnitRange{Int}}()
    next_offset = 1
    for qlabel in sort!(collect(keys(sector_sizes)))
        size = sector_sizes[qlabel]
        offsets[qlabel] = next_offset:(next_offset + size - 1)
        next_offset += size
    end
    return offsets, next_offset - 1
end

"""
    _sparse_leg_kron(a, b, qd::Int)

Take a legwise Kronecker product of two CGT blocks with bond axes.

`a` and `b` have `qd` physical axes followed by one bond/outer-multiplicity
axis. The result interleaves the physical axes leg by leg and merges the two
bond axes into a single trailing bond axis. This combines CGT factors from
different symmetries within one TLArray sector.
"""
function _sparse_leg_kron(a::AbstractArray, b::AbstractArray, qd::Int)
    dims_a = ntuple(leg -> size(a, leg), qd)
    dims_b = ntuple(leg -> size(b, leg), qd)
    bond_a, bond_b = size(a, qd + 1), size(b, qd + 1)
    expanded_a = reshape(a, dims_a..., ones(Int, qd)..., bond_a, 1)
    expanded_b = reshape(b, ones(Int, qd)..., dims_b..., 1, bond_b)
    perm = Tuple(vcat([[leg, qd + leg] for leg in 1:qd]..., 2qd + 1, 2qd + 2))
    return reshape(permutedims(expanded_a .* expanded_b, perm),
                   ntuple(leg -> dims_a[leg] * dims_b[leg], qd)..., bond_a * bond_b)
end

"""
    _sparse_legwise_kron(a, b)

Take a legwise Kronecker product of two same-rank physical blocks.

`a` and `b` have the same number of axes. For each leg, the output dimension is
`size(a, leg) * size(b, leg)`. This is used to combine an expanded CGT block
with the corresponding RMT block for one sector and one OM-bond column.
"""
function _sparse_legwise_kron(a::AbstractArray, b::AbstractArray)
    qd = ndims(a)
    dims_a = ntuple(leg -> size(a, leg), qd)
    dims_b = ntuple(leg -> size(b, leg), qd)
    expanded_a = reshape(a, Tuple(Iterators.flatten((dims_a[leg], 1) for leg in 1:qd)))
    expanded_b = reshape(b, Tuple(Iterators.flatten((1, dims_b[leg]) for leg in 1:qd)))
    return reshape(expanded_a .* expanded_b,
                   ntuple(leg -> dims_a[leg] * dims_b[leg], qd)...)
end
