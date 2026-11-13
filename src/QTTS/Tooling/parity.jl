# Reflection-parity quantics tensors for the native Telum / BUG engine.
#
# Binary reflection j -> 2^n - 1 - j acts as X on every coordinate bit.
# Sitewise Hadamards transform it to diagonal Z2 charge. Two independently
# reflected axes carry the product group Z2 x Z2. Virtual sectors track the
# XOR of the charge bits to their left, without constructing 2^L samples.
#
# All inputs are ordinary QTT cores in the spatial grid-point basis. This
# separate adapter does not change the existing, dense :none QTT API.
# A coordinate reflection about zero corresponds to bitwise complement for
# half-cell-centered grids, not the default left-endpoint periodic grid.

using LurCGT: Z

const _QTT_PARITY_H = [1.0 1.0; 1.0 -1.0] / sqrt(2)

_qp_label(mask::Integer, ::Val{N}) where {N} =
    ntuple(a -> (((Int(mask) >> (a - 1)) & 1),), N)

_qp_tuple(mask::Integer, ::Val{N}) where {N} =
    ntuple(a -> (Int(mask) >> (a - 1)) & 1, N)

function _qp_mask(q, ::Val{N}) where {N}
    length(q) == N || throw(ArgumentError("expected $N reflection charges, got $q"))
    all(v -> v == 0 || v == 1, q) ||
        throw(ArgumentError("reflection charges must be 0 or 1, got $q"))
    return sum(Int(q[a]) << (a - 1) for a = 1:N)
end

function _qp_owners(o::QTTOrdering{N}) where {N}
    owners = zeros(Int, qtt_length(o))
    for a = 1:N, site in o.sites[a]
        owners[site] = a
    end
    all(!iszero, owners) || error("invalid QTT site-to-axis map")
    return owners
end

function _qp_check_chain(C, rank, L)
    length(C) == L || throw(DimensionMismatch("expected $L QTT cores"))
    all(ndims(c) == rank for c in C) ||
        throw(DimensionMismatch("expected rank-$rank cores"))
    size(first(C), 1) == 1 && size(last(C), rank) == 1 ||
        throw(DimensionMismatch("boundary QTT ranks must be one"))
    for k = 1:(L-1)
        size(C[k], rank) == size(C[k+1], 1) ||
            throw(DimensionMismatch("incompatible QTT bond $k"))
    end
end

"Apply a local Hadamard on every physical index of a QTT state."
function _qp_walsh(C)
    return [begin
        A = ComplexF64.(c)
        size(A, 2) == 2 || throw(DimensionMismatch("QTT local size must be two"))
        B = similar(A)
        B[:, 1, :] .= (A[:, 1, :] .+ A[:, 2, :]) ./ sqrt(2)
        B[:, 2, :] .= (A[:, 1, :] .- A[:, 2, :]) ./ sqrt(2)
        B
    end for c in C]
end

"Conjugate the input/output legs of an operator core by Hadamards."
function _qp_walsh_op(W)
    out = Vector{Array{ComplexF64,4}}(undef, length(W))
    for k in eachindex(W)
        A = ComplexF64.(W[k])
        size(A, 2) == size(A, 3) == 2 ||
            throw(DimensionMismatch("operator physical legs must both have size two"))
        B = zeros(ComplexF64, size(A))
        for o = 1:2, i = 1:2, a = 1:2, b = 1:2
            B[:, o, i, :] .+= _QTT_PARITY_H[o, a] * _QTT_PARITY_H[i, b] .* A[:, a, b, :]
        end
        out[k] = B
    end
    return out
end

_qp_character(q::Int, g::Int) = isodd(count_ones(q & g)) ? -1 : 1

"""
    qtt_parity_weights(cores, ordering; operator=false)

Squared Frobenius norms in the 2^N independent reflection-charge sectors.
For an operator, a charge describes the change in the state parity.
This contracts QTT cores, without forming the exponentially large field.
"""
function qtt_parity_weights(C::AbstractVector, o::QTTOrdering{N};
                            operator::Bool = false) where {N}
    N in (1, 2) || throw(ArgumentError("reflection charges require one or two axes"))
    L = qtt_length(o)
    _qp_check_chain(C, operator ? 4 : 3, L)
    owners = _qp_owners(o)
    A = operator ? _qp_walsh_op(C) : _qp_walsh(C)
    # Operator index: (output,input), output fastest when flattened.
    terms = operator ? [reshape(c, size(c, 1), 4, size(c, 4)) for c in A] : A
    G = 1 << N
    tracechars = zeros(Float64, G)
    for g = 0:(G-1)
        tagged = [copy(c) for c in terms]
        for (k, core) in enumerate(tagged)
            (g & (1 << (owners[k] - 1))) == 0 && continue
            if operator
                core[:, 2, :] .*= -1  # out=1, in=0
                core[:, 3, :] .*= -1  # out=0, in=1
            else
                core[:, 2, :] .*= -1
            end
        end
        tracechars[g+1] = real(tt_dot(terms, tagged))
    end
    return Dict(_qp_tuple(q, Val(N)) =>
                max(0.0, sum(_qp_character(q, g) * tracechars[g+1]
                              for g = 0:(G-1)) / G)
                for q = 0:(G-1))
end

function _qp_check_sector(weights, wanted, tol::Real, strict::Bool)
    total = sum(values(weights))
    total > 0 || throw(ArgumentError("zero-norm QTT input"))
    weight = get(weights, wanted, 0.0)
    weight > total * 1e-24 ||
        throw(ArgumentError("requested charge $wanted has negligible weight"))
    leakage = max(0.0, 1.0 - weight / total)
    # Character sums have a roundoff floor in squared norms.
    if strict && leakage > max(tol^2, 1e-12)
        throw(ArgumentError(
            "input mixes reflection charges: outside-sector squared weight $leakage; " *
            "use strict=false to project intentionally",
        ))
    end
    return leakage
end

"At a cut, include only charges reachable from both boundaries."
function _qp_bond_charges(owners::Vector{Int}, N::Int, target::Int)
    L = length(owners)
    G = 1 << N
    charges = Vector{Vector{Int}}(undef, L + 1)
    for k = 0:L
        prefix, suffix = 0, 0
        for i = 1:k
            prefix |= 1 << (owners[i] - 1)
        end
        for i = (k+1):L
            suffix |= 1 << (owners[i] - 1)
        end
        charges[k+1] = [q for q = 0:(G-1) if
                        (q & ~prefix) == 0 && ((q xor target) & ~suffix) == 0]
    end
    return charges
end

function _qp_space(charges, rank, ::Val{N}) where {N}
    sp = [(_qp_label(q, Val(N)), rank) for q in charges]
    sort!(sp; by = first)
    return sp
end

"Wrap reduced blocks in actual Z2 / Z2 x Z2 Telum TLArrays."
function _qp_tensor(labels::Vector{NTuple{D,QT}}, blocks::Vector{Array{T,RD}},
                    inds::NTuple{D,TLIndex}, spaces::NTuple{D,Any},
                    ::Val{N}) where {D,QT,T,RD,N}
    RD == D + N || throw(DimensionMismatch("invalid parity reduced-block rank"))
    p = sortperm(labels)
    return TLArray(ntuple(_ -> Z{2}, N), labels[p], Float64[],
                   [() for _ in p], blocks[p], inds, spaces)
end

"""
    qtt_parity_mps(C, ordering; charge=(0,...), strict=true, tol=1e-10, center=1)

A native charge-blocked SymMPS from grid-basis QTT cores. Every bond tracks
the XOR of preceding Walsh bits; the final bond is fixed to the requested
Z2 (1D) or Z2 x Z2 (2D) total charge. The original virtual ranks multiply
the charge sectors, but absent blocks are never allocated. No full-grid
dense array is formed. A mixed-charge field is rejected unless strict=false.
"""
function qtt_parity_mps(C::AbstractVector, o::QTTOrdering{N};
                        charge = ntuple(_ -> 0, N), strict::Bool = true,
                        tol::Real = 1e-10, center::Int = 1) where {N}
    N in (1, 2) || throw(ArgumentError("reflection charges require one or two axes"))
    L = qtt_length(o)
    _qp_check_chain(C, 3, L)
    1 <= center <= L || throw(ArgumentError("invalid orthogonality center"))
    target = _qp_mask(charge, Val(N))
    wanted = _qp_tuple(target, Val(N))
    _qp_check_sector(qtt_parity_weights(C, o), wanted, tol, strict)
    A = _qp_walsh(C)
    owners = _qp_owners(o)
    bonds = _qp_bond_charges(owners, N, target)
    tensors = Any[]
    for k = 1:L
        sl = _qp_space(bonds[k], size(A[k], 1), Val(N))
        sr = _qp_space(bonds[k+1], size(A[k], 3), Val(N))
        p = 1 << (owners[k] - 1)
        sp = [(_qp_label(0, Val(N)), 1), (_qp_label(p, Val(N)), 1)]
        labels = NTuple{3,typeof(sp[1][1])}[]
        blocks = Array{ComplexF64,3+N}[]
        allowed = Set(bonds[k+1])
        for ql in bonds[k], bit = 0:1
            qr = ql xor (bit == 0 ? 0 : p)
            qr in allowed || continue
            push!(labels, (_qp_label(ql, Val(N)),
                           _qp_label(bit == 0 ? 0 : p, Val(N)),
                           _qp_label(qr, Val(N))))
            push!(blocks, reshape(copy(A[k][:, bit+1, :]), size(A[k], 1),
                                  1, size(A[k], 3), ntuple(_ -> 1, N)...))
        end
        inds = (TLIndex("L,$k", '+'), TLIndex("S,$k", '+'),
                TLIndex("L,$(k+1)", '-'))
        push!(tensors, _qp_tensor(labels, blocks, inds, (sl, sp, sr), Val(N)))
    end
    psi = SymMPS(tensors, L)
    canonical!(psi, 1)
    canonical!(psi, center)
    return psi
end

"""
    qtt_parity_mpo(W, ordering; charge=(0,...), strict=true, tol=1e-10)

A native parity-charge MPO for original QTT operator cores (bond,out,in,bond).
Zero operator charge is appropriate for reflection-symmetric Laplacians.
An odd derivative can be represented by a nonzero operator charge.
The BUG MPO convention stores the *transpose* on its site legs.
"""
function qtt_parity_mpo(W::AbstractVector, o::QTTOrdering{N};
                        charge = ntuple(_ -> 0, N), strict::Bool = true,
                        tol::Real = 1e-10) where {N}
    N in (1, 2) || throw(ArgumentError("reflection charges require one or two axes"))
    L = qtt_length(o)
    L >= 2 || throw(ArgumentError("MPO needs at least two sites"))
    _qp_check_chain(W, 4, L)
    target = _qp_mask(charge, Val(N))
    wanted = _qp_tuple(target, Val(N))
    _qp_check_sector(qtt_parity_weights(W, o; operator = true), wanted, tol, strict)
    A = _qp_walsh_op(W)
    owners = _qp_owners(o)
    bonds = _qp_bond_charges(owners, N, target)
    tensors = Any[]
    for k = 1:L
        sl = _qp_space(bonds[k], size(A[k], 1), Val(N))
        sr = _qp_space(bonds[k+1], size(A[k], 4), Val(N))
        p = 1 << (owners[k] - 1)
        sp = [(_qp_label(0, Val(N)), 1), (_qp_label(p, Val(N)), 1)]
        labels = NTuple{4,typeof(sp[1][1])}[]
        blocks = Array{ComplexF64,4+N}[]
        allowed = Set(bonds[k+1])
        for ql in bonds[k], out = 0:1, input = 0:1
            dq = (out xor input) == 0 ? 0 : p
            qr = ql xor dq
            qr in allowed || continue
            push!(labels, (_qp_label(ql, Val(N)),
                           _qp_label(input == 0 ? 0 : p, Val(N)),
                           _qp_label(out == 0 ? 0 : p, Val(N)),
                           _qp_label(qr, Val(N))))
            push!(blocks, reshape(copy(A[k][:, out+1, input+1, :]),
                                  size(A[k], 1), 1, 1, size(A[k], 4),
                                  ntuple(_ -> 1, N)...))
        end
        inds = (TLIndex("W,$(k-1)", '+'), TLIndex("S,$k", '+'),
                TLIndex("S,$k", '-'), TLIndex("W,$k", '-'))
        push!(tensors, _qp_tensor(labels, blocks, inds,
                                  (sl, sp, sp, sr), Val(N)))
    end
    return MPO(tensors)
end

"Reassemble one full QTT core (not the full tensor) from the Telum sectors."
function _qp_unblock(t)
    out = zeros(eltype(t), size(t))
    offsets = [begin
        firstidx = 1
        ranges = Dict{Any,UnitRange{Int}}()
        for (q, d) in space
            ranges[q] = firstidx:(firstidx+d-1)
            firstidx += d
        end
        ranges
    end for space in Telum.spaces(t)]
    for (sector, labels) in enumerate(Telum.logical_qlabels(t))
        t.isdefined[sector] || continue
        ranges = ntuple(a -> offsets[a][labels[a]], ndims(t))
        out[ranges...] .= reshape(_sector_block(t, sector), map(length, ranges)...)
    end
    return out
end

"Decode a parity-charged SymMPS into ordinary spatial-basis QTT cores."
function qtt_parity_cores(psi::SymMPS)
    syms = Telum.symm(psi[1])
    length(syms) in (1, 2) && all(s -> s === Z{2}, syms) ||
        throw(ArgumentError("expected a Z2 or Z2 x Z2 parity QTT state"))
    return _qp_walsh([_qp_unblock(psi[k]) for k = 1:length(psi)])
end

"Decode a parity MPO into original (bond,out,in,bond) spatial-basis cores."
function qtt_parity_mpo_cores(mpo::MPO)
    syms = Telum.symm(mpo[1])
    length(syms) in (1, 2) && all(s -> s === Z{2}, syms) ||
        throw(ArgumentError("expected a Z2 or Z2 x Z2 parity MPO"))
    W = [permutedims(_qp_unblock(mpo[k]), (1, 3, 2, 4))
         for k = 1:length(mpo)]
    return _qp_walsh_op(W)
end
