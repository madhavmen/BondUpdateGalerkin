# ─── qr ─────────────────────────────────────────────────────────────────────
#
# Symmetry-adapted QR decomposition of a TLArray across a left/right leg split.

"""
Container for a symmetry-aware QR factorization.

# Fields

- `Q`: left/isometric TLArray factor with the generated QR bond leg.
- `R`: right/triangular TLArray factor sharing that bond leg.

Destructuring iterates as `(Q, R)`.
"""
struct QRResult{TQ, TR}
    Q::TQ
    R::TR
end

function QRResult(Q::TLArray, R::TLArray)
    return QRResult{typeof(Q), typeof(R)}(Q, R)
end

"""Internal dense QR result for one CGT equivalence class.

# Fields

- `sector`: shared intermediate q-label of the class.
- `left_infos`, `right_infos`: packed-side descriptors used to scatter factors.
- `Q`, `R`: dense matrix QR factors before TLArray sector assembly.
"""
struct _QRCGTClassResult{QT, LI, RI, Q, R}
    sector::QT
    left_infos::LI
    right_infos::RI
    Q::Q
    R::R
end

"""Internal QR row-to-symmetry split descriptor.

# Fields

- `sector_index`: source TLArray sector slot.
- `q`: product-symmetry q-label tuple for that slot.
- `left_signature`, `right_signature`: canonical q-label tuples defining each split side.
"""
struct _QRSplitRow{L, R, QT}
    sector_index::Int
    q::QT
    left_signature::NTuple{L, QT}
    right_signature::NTuple{R, QT}
end

"""Internal QR decomposition of one sector into symmetry isometries and core.

# Fields

- `sector_index`, `q`: source slot and its q-label tuple.
- `left_signature`, `right_signature`: canonical side signatures.
- `left_iso`, `right_iso`: symmetry basis isometries.
- `core`: reduced dense coefficient core.
"""
struct _QRSymmetrySplit{L, R, QTN}
    sector_index::Int
    q::QTN
    left_signature::NTuple{L, QTN}
    right_signature::NTuple{R, QTN}
    left_iso::Matrix{Float64}
    right_iso::Matrix{Float64}
    core::Array{Float64, 3}
end

"""Internal QR CGT block descriptor.

# Fields

- `q`: intermediate q-label.
- `omL`, `omR`: left/right outer-multiplicity dimensions.
- `coeffs`: dense CGT coefficient block coupling those OM axes.
"""
struct _QRCGTBlockInfo{NZ}
    q::NTuple{NZ, Int}
    omL::Int
    omR::Int
    coeffs::Array{Float64, 3}
end

"""Internal packed-side layout descriptor for a QR class.

# Fields

- `signature`: canonical q-label signature of the side.
- `row_index`, `sector_index`: source split-row and sector identifiers.
- `phys_dims`, `om_dims`: physical and non-Abelian OM extents.
- `range`: contiguous packed-matrix range assigned to the side.
"""
struct _QRClassSideInfo{Sig, L, N}
    signature::Sig
    row_index::Int
    sector_index::Int
    phys_dims::NTuple{L, Int}
    om_dims::NTuple{N, Int}
    range::UnitRange{Int}
end

"""Internal grouping metadata used to assemble one QR CGT class.

# Fields

- `sector`: class intermediate q-label.
- `rows`: split-row range belonging to the class.
- `left_infos`, `right_infos`: packed-side descriptors.
- `left_ranges`, `right_ranges`: signature-to-packed-range maps.
- `total_left`, `total_right`: packed matrix dimensions.
"""
struct _QRCGTClassMetadata{QT, LI, LR, RI, RR}
    sector::QT
    rows::UnitRange{Int}
    left_infos::LI
    left_ranges::LR
    total_left::Int
    right_infos::RI
    right_ranges::RR
    total_right::Int
end

_qr_class_side_info_type(::Type{QT}, ::Val{L}, ::Val{N}) where {QT, L, N} =
    _QRClassSideInfo{NTuple{L, QT}, L, N}

function _qr_cgt_class_metadata_type(::Type{QT}, ::Val{NL}, ::Val{NR}, ::Val{N}) where {QT, NL, NR, N}
    LeftSig = NTuple{NL, QT}
    RightSig = NTuple{NR, QT}
    LI = Vector{_qr_class_side_info_type(QT, Val(NL), Val(N))}
    RI = Vector{_qr_class_side_info_type(QT, Val(NR), Val(N))}
    LR = Dict{LeftSig, UnitRange{Int}}
    RR = Dict{RightSig, UnitRange{Int}}
    return _QRCGTClassMetadata{QT, LI, LR, RI, RR}
end

function _qr_class_result_type(
    ::Type{_QRCGTClassMetadata{QT, LI, LR, RI, RR}},
    ::Type{Tmat}) where {QT, LI, LR, RI, RR, Tmat}
    return _QRCGTClassResult{QT, LI, RI, Matrix{Tmat}, Matrix{Tmat}}
end

_qr_stable_sort_tuple(spaces) = Tuple(sort!(collect(spaces); alg=MergeSort))

@inline function _qr_physical_qlabels(::Type{QT},
                                      qlabels_by_symm::Tuple{Vararg{Tuple, N}},
                                      cgp_by_symm::Tuple{Vararg{NTuple{QD, Int}, N}},
                                      ::Val{QD}) where {QT, N, QD}
    return ntuple(Val(QD)) do leg
        ntuple(n -> qlabels_by_symm[n][cgp_by_symm[n][leg]], Val(N))::QT
    end
end

function _qr_symmetry_stored_leg_order(
    ::Type{QT},
    q::AbstractTLArray{T, QD, N, RD, QT, PS, M, RMT},
    sector_index::Int,
    ::Val{n},
) where {T, QD, N, RD, QT, PS, M, RMT, n}
    qlabels_by_phys = ntuple(leg -> sector_qlabel(QT, q, sector_index, leg)[n], Val(QD))

    incoming = Int[]
    outgoing = Int[]
    sizehint!(incoming, QD)
    sizehint!(outgoing, QD)
    for leg in 1:QD
        if inds(q)[leg].dir == '+'
            push!(incoming, leg)
        else
            push!(outgoing, leg)
        end
    end

    sort!(incoming; by = leg -> qlabels_by_phys[leg], alg=MergeSort)
    sort!(outgoing; by = leg -> qlabels_by_phys[leg], alg=MergeSort)

    nin = length(incoming)
    stored_to_phys = ntuple(i -> i <= nin ? incoming[i] : outgoing[i - nin], Val(QD))
    phys_to_stored = _phys_to_stored_order(stored_to_phys)
    stored_qlabels = ntuple(i -> qlabels_by_phys[stored_to_phys[i]], Val(QD))
    legdir = (nin, QD - nin)
    return stored_qlabels, phys_to_stored, stored_to_phys, legdir
end

@inline _qr_physical_qlabel_key(::Type{QT},
                                q::AbstractTLArray{T, QD, N, RD, QT, PS, M, RMT},
                                sector_index::Int,
                                ::Val{n}) where {T, QD, N, RD, QT, PS, M, RMT, n} =
    ntuple(leg -> sector_qlabel(QT, q, sector_index, leg)[n], Val(QD))

@inline _qr_physical_product_key(::Type{QT},
                                 q::AbstractTLArray{T, QD, N, RD, QT, PS, M, RMT},
                                 sector_index::Int) where {T, QD, N, RD, QT, PS, M, RMT} =
    ntuple(leg -> sector_qlabel(QT, q, sector_index, leg), Val(QD))

function _qr_physical_key_stored_order(physical_key::NTuple{QD},
                                       inds::NTuple{QD, TLIndex}) where {QD}
    incoming = Int[]
    outgoing = Int[]
    sizehint!(incoming, QD)
    sizehint!(outgoing, QD)
    for leg in 1:QD
        if inds[leg].dir == '+'
            push!(incoming, leg)
        else
            push!(outgoing, leg)
        end
    end

    sort!(incoming; by = leg -> physical_key[leg], alg=MergeSort)
    sort!(outgoing; by = leg -> physical_key[leg], alg=MergeSort)

    nin = length(incoming)
    stored_to_phys = ntuple(i -> i <= nin ? incoming[i] : outgoing[i - nin], Val(QD))
    phys_to_stored = _phys_to_stored_order(stored_to_phys)
    stored_qlabels = ntuple(i -> physical_key[stored_to_phys[i]], Val(QD))
    legdir = (nin, QD - nin)
    return stored_qlabels, phys_to_stored, stored_to_phys, legdir
end

function _qr_cgt_updn(qlabels::NTuple{QD}, legdir::Tuple{Int, Int}) where {QD}
    nin = legdir[1]
    upsp = Tuple(qlabels[i] for i in 1:nin)
    dnsp = Tuple(qlabels[i] for i in nin+1:QD)
    return upsp, dnsp
end

function _qr_to_cgtidx(cgp::NTuple{QD, Int}, lidxs) where {QD}
    return Tuple(Int[cgp[l] for l in lidxs])
end

function _qr_physical_key_split_args(::Type{S},
                                     physical_key::NTuple{QD, NTuple{NZ, Int}},
                                     inds::NTuple{QD, TLIndex},
                                     left_legs::NTuple{L, Int}) where {S<:NonabelianSymm, QD, NZ, L}
    @assert NZ == nzops(S)
    stored_qlabels, phys_to_stored, _, legdir =
        _qr_physical_key_stored_order(physical_key, inds)
    upsp, dnsp = _qr_cgt_updn(stored_qlabels, legdir)
    left_legs_canon = _qr_to_cgtidx(phys_to_stored, left_legs)::NTuple{L, Int}
    return upsp, dnsp, left_legs_canon
end

@inline function _qr_physical_key_side_signatures(physical_key::NTuple{QD, QT},
                                                  left_legs::NTuple{L, Int},
                                                  right_legs::NTuple{R, Int}) where {QD, QT, L, R}
    left_signature = ntuple(i -> physical_key[left_legs[i]], Val(L))
    right_signature = ntuple(i -> physical_key[right_legs[i]], Val(R))
    return left_signature, right_signature
end

function _qr_abelian_intermediate_q(::Type{S},
    qlabels::NTuple{QD},
    legdir::Tuple{Int, Int},
    left_legs_canon::NTuple{L, Int}) where {S<:AbelianSymm, QD, L}
    nin = legdir[1]
    leftset = Set(left_legs_canon)

    merged = _qr_stable_sort_tuple((
        Tuple(qlabels[i] for i in 1:nin if i in leftset)...,
        Tuple(get_dualq(S, qlabels[i]) for i in nin+1:QD if i in leftset)...,
    ))
    outcomes = combine_qlabels(S, merged)
    @assert length(outcomes) == 1
    return outcomes[1][1]
end

@inline _qr_unit_isometry() = [1.0;;]
@inline _qr_trivial_iso() = ones(Float64, 1, 1)
@inline _qr_trivial_core() = reshape(ones(Float64, 1), 1, 1, 1)

function _reduce_qr_cgt_split(block,
                              sector_index::Int,
                              left_signature::NTuple{L, QTN},
                              right_signature::NTuple{R, QTN};
                              tol::Float64 = 1e-12) where {L, R, QTN}
    coeffs = block.coeffs
    maximum(abs, coeffs) <= tol && return nothing
    left_iso = Matrix{Float64}(I, block.omL, block.omL)
    right_iso = Matrix{Float64}(I, block.omR, block.omR)
    return _QRSymmetrySplit{L, R, QTN}(
        sector_index, block.q, left_signature, right_signature,
        left_iso, right_iso, coeffs)
end

_qr_cgtqr_cache_key_type(::Type{QT}, ::Val{n}, ::Val{QD}) where {QT, n, QD} =
    NTuple{QD, fieldtype(QT, n)}

@generated function _qr_cgtqr_key_tuple_type(::Type{PS}, ::Type{QT}, ::Val{QD}) where {PS<:ProductSymm, QT, QD}
    M = _wmat_tuple_width(PS)
    key_types = Type[]
    for slot in 1:M
        n = product_symmetry_index_from_wmat_slot(PS, Val(slot))
        push!(key_types, _qr_cgtqr_cache_key_type(QT, Val(n), Val(QD)))
    end
    return :(Tuple{$(key_types...)})
end

@generated function _qr_cgtqr_row_keys(::Type{PS},
                                       ::Type{QT},
                                       q::AbstractTLArray{T, QD, N, RD, QT, PS, M, RMT},
                                       sector_index::Int,
                                       ::Val{M}) where {T, QD, N, RD, QT, PS<:ProductSymm, M, RMT}
    exprs = Expr[]
    for slot in 1:M
        n = product_symmetry_index_from_wmat_slot(PS, Val(slot))
        push!(exprs, :(_qr_physical_qlabel_key(QT, q, sector_index, Val($n))))
    end
    return Expr(:tuple, exprs...)
end

@generated function _qr_cgtqr_cache_slot(::Type{PS}, ::Val{n}) where {PS<:ProductSymm, n}
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
    return :(throw(ArgumentError("symmetry index $n has no CGTQR cache")))
end

@inline _qr_cgtqr_cache_for(::Type{S}, caches, ::Type{PS}, ::Val{n}) where {S<:AbelianSymm, PS<:ProductSymm, n} = nothing
@inline function _qr_cgtqr_cache_for(::Type{S}, caches, ::Type{PS}, ::Val{n}) where {S<:NonabelianSymm, PS<:ProductSymm, n}
    slot = _qr_cgtqr_cache_slot(PS, Val(n))
    slot === nothing &&
        throw(ArgumentError("symmetry index $n is Abelian and has no CGTQR cache"))
    return caches[slot]
end

@inline _qr_cgtqr_cache_for(caches::CT, ::Type{S}, ::Type{PS}, ::Val{n}) where {CT<:Tuple, S, PS<:ProductSymm, n} =
    _qr_cgtqr_cache_for(S, caches, PS, Val(n))

@inline _qr_irrepdim_cache_for(::Type{S}, caches, ::Type{PS}, ::Val{n}) where {S<:AbelianSymm, PS<:ProductSymm, n} = nothing
@inline function _qr_irrepdim_cache_for(::Type{S}, caches, ::Type{PS}, ::Val{n}) where {S<:NonabelianSymm, PS<:ProductSymm, n}
    slot = _qr_cgtqr_cache_slot(PS, Val(n))
    slot === nothing &&
        throw(ArgumentError("symmetry index $n is Abelian and has no irrep-dimension cache"))
    return caches[slot]
end

@inline _qr_irrepdim_cache_for(caches::CT, ::Type{S}, ::Type{PS}, ::Val{n}) where {CT<:Tuple, S, PS<:ProductSymm, n} =
    _qr_irrepdim_cache_for(S, caches, PS, Val(n))

@generated function _new_qr_irrepdim_caches(::Type{PS}, ::Type{QT}) where {PS<:ProductSymm, QT}
    syms = product_symms(PS)
    seen = Type[]
    exprs = Expr[]
    for (n, S) in pairs(syms)
        S <: AbelianSymm && continue
        S in seen && continue
        push!(seen, S)
        QTN = fieldtype(QT, n)
        Cache = Dict{QTN, Int}
        push!(exprs, :($Cache()))
    end
    return Expr(:tuple, exprs...)
end

_new_qr_irrepdim_caches(::AbstractTLArray{T, QD, N, RD, QT, PS}) where {T, QD, N, RD, QT, PS} =
    _new_qr_irrepdim_caches(PS, QT)

function _qr_irrep_dimension(::Type{S}, qlabel, cache) where {S<:NonabelianSymm}
    dim = get(cache, qlabel, 0)
    dim != 0 && return dim
    dim = dimension(S, qlabel)
    cache[qlabel] = dim
    return dim
end

@generated function _fill_qr_irrepdim_caches_for_sector!(irrepdim_caches::CT,
                                                         sector::QT,
                                                         ::Type{PS}) where {CT<:Tuple, QT, PS<:ProductSymm}
    syms = product_symms(PS)
    exprs = Expr[]
    for n in eachindex(syms)
        S = syms[n]
        S <: AbelianSymm && continue
        push!(exprs, quote
            qlabel = sector[$n]
            cache = _qr_irrepdim_cache_for(irrepdim_caches, $S, PS, Val($n))
            if !haskey(cache, qlabel)
                cache[qlabel] = dimension($S, qlabel)
            end
        end)
    end
    push!(exprs, :(return irrepdim_caches))
    return Expr(:block, exprs...)
end

function _fill_qr_irrepdim_caches!(irrepdim_caches::CT,
                                   class_metadata::AbstractVector,
                                   ::Type{PS}) where {CT<:Tuple, PS<:ProductSymm}
    for metadata in class_metadata
        _fill_qr_irrepdim_caches_for_sector!(irrepdim_caches, metadata.sector, PS)
    end
    return irrepdim_caches
end

@generated function _qr_sector_degeneracy(::Type{PS},
                                          sector::QT,
                                          irrepdim_caches::CT,
                                          ::Val{N}) where {PS<:ProductSymm, QT, CT<:Tuple, N}
    expr = :(1)
    syms = product_symms(PS)
    for n in 1:N
        S = syms[n]
        if S <: AbelianSymm
            term = :(dimension($S, sector[$n]))
        else
            term = :(_qr_irrep_dimension($S, sector[$n],
                _qr_irrepdim_cache_for(irrepdim_caches, $S, PS, Val($n))))
        end
        expr = :($expr * $term)
    end
    return expr
end

function _qr_right_legs(::Val{QD}, left_legs::NTuple{L, Int}) where {QD, L}
    return ntuple(Val(QD - L)) do k
        nright = 0
        for leg in 1:QD
            if leg ∉ left_legs
                nright += 1
                nright == k && return leg
            end
        end
        throw(BoundsError())
    end
end

@inline _qr_get_cgtqr(::Type{S}, physical_key, upsp, dnsp, left_legs_canon, cache) where {S<:AbelianSymm} =
    getNsave_CGTQR(S, upsp, dnsp, left_legs_canon; save=true)

function _qr_get_cgtqr(::Type{S}, physical_key, upsp, dnsp, left_legs_canon, cache) where {S<:NonabelianSymm}
    cache === nothing && return getNsave_CGTQR(S, upsp, dnsp, left_legs_canon; save=true)
    if haskey(cache, physical_key)
        return cache[physical_key]
    end
    cgtqr = getNsave_CGTQR(S, upsp, dnsp, left_legs_canon; save=true)
    cache[physical_key] = cgtqr
    return cgtqr
end

@generated function _new_cgtqr_caches(::Type{PS}, ::Type{QT}, ::Val{QD}) where {PS<:ProductSymm, QT, QD}
    syms = product_symms(PS)
    seen = Type[]
    exprs = Expr[]
    for (n, S) in pairs(syms)
        S <: AbelianSymm && continue
        S in seen && continue
        push!(seen, S)
        Key = _qr_cgtqr_cache_key_type(QT, Val(n), Val(QD))
        Value = Union{Bool, LurCGT.CGTQR{S}}
        Cache = Dict{Key, Value}
        push!(exprs, :($Cache()))
    end
    return Expr(:tuple, exprs...)
end

_new_cgtqr_caches(::AbstractTLArray{T, QD, N, RD, QT, PS}) where {T, QD, N, RD, QT, PS} =
    _new_cgtqr_caches(PS, QT, Val(QD))

@inline _qr_cgtqr_split_count(cgtqr::LurCGT.CGTQR) = length(cgtqr.bond_sps)
@inline _qr_cgtqr_split_count(::Bool) = 1
@inline _qr_cgtqr_split_count(::Nothing) = 1

function _get_qr_cgt_split_blocks(S, qlabels::NTuple{QD}, wmat::AbstractMatrix{Float64},
                                  cgp::NTuple{QD, Int}, legdir::Tuple{Int, Int},
                                  left_legs, right_legs, physical_key,
                                  cgtqr_cache = nothing) where {QD}
    left_legs_canon = _qr_to_cgtidx(cgp, left_legs)
    right_legs_canon = _qr_to_cgtidx(cgp, right_legs)
    NZ = length(qlabels[1])
    BlockInfo = _QRCGTBlockInfo{NZ}

    if isabelian(S)
        q = _qr_abelian_intermediate_q(S, qlabels, legdir, left_legs_canon)
        coeffs = reshape(copy(wmat), 1, 1, size(wmat, 2))
        _is_zero_array(coeffs) && return BlockInfo[]
        return [BlockInfo(q, 1, 1, coeffs)]
    end

    upsp, dnsp = _qr_cgt_updn(qlabels, legdir)
    cgtqr = _qr_get_cgtqr(S, physical_key, upsp, dnsp, left_legs_canon, cgtqr_cache)

    blocks = BlockInfo[]
    if cgtqr isa LurCGT.CGTQR
        coeff_split = cgtqr.qr_arr * wmat
        offset = 1
        for (q, omL, omR) in cgtqr.bond_sps
            width = omL * omR
            coeffs = reshape(coeff_split[offset:offset+width-1, :], omL, omR, size(coeff_split, 2))
            !_is_zero_array(coeffs) && push!(blocks, BlockInfo(q, omL, omR, coeffs))
            offset += width
        end
        @assert offset == size(coeff_split, 1) + 1
    else
        q = zero_qlabels((S,))[1]
        om = size(wmat, 1)
        omL, omR = cgtqr ? (1, om) : (om, 1)
        coeffs = reshape(wmat, omL, omR, size(wmat, 2))
        push!(blocks, BlockInfo(q, omL, omR, coeffs))
    end
    return blocks
end

function _count_qr_symmetry_splits(q::AbstractTLArray{T, QD, N, RD, QT, PS, M, RMT},
                                   sector_index::Int,
                                   left_legs::NTuple{L, Int},
                                   cgtqr_caches::CT,
                                   ::Val{n}) where {T, QD, N, RD, QT, PS, M, RMT, L, CT<:Tuple, n}
    S = nth_symm(PS, Val(n))
    isabelian(S) && return 1
    physical_key = _qr_physical_qlabel_key(QT, q, sector_index, Val(n))
    upsp, dnsp, left_legs_canon =
        _qr_physical_key_split_args(S, physical_key, inds(q), left_legs)
    cache = _qr_cgtqr_cache_for(cgtqr_caches, S, PS, Val(n))
    cgtqr = _qr_get_cgtqr(S, physical_key, upsp, dnsp, left_legs_canon, cache)
    return _qr_cgtqr_split_count(cgtqr)
end

function _count_qr_sector_product_splits(q::AbstractTLArray{T, QD, N, RD, QT, PS, M, RMT},
                                         sector_index::Int,
                                         left_legs::NTuple{L, Int},
                                         cgtqr_caches::CT,
                                         ::Val{N}) where {T, QD, N, RD, QT, PS, M, RMT, L, CT<:Tuple}
    count = 1
    for n in 1:N
        count *= _count_qr_symmetry_splits(
            q, sector_index, left_legs, cgtqr_caches, Val(n))
    end
    return count
end

function _prepare_qr_counted_preassembly(q::AbstractTLArray{T, QD, N, RD, QT, PS, M, RMT},
                                         left_legs::NTuple{L, Int},
                                         ::Val{N}) where {T, QD, N, RD, QT, PS, M, RMT, L}
    cgtqr_caches = _new_cgtqr_caches(q)
    KeyTuple = _qr_cgtqr_key_tuple_type(PS, QT, Val(QD))
    nactive = count(sector_index -> !is_sector_zero(q, sector_index), sector_slots(q))
    active_sector_indices = Vector{Int}(undef, nactive)
    keys_by_row = Vector{KeyTuple}(undef, nactive)
    sector_row_ranges = Vector{UnitRange{Int}}(undef, nactive)

    total_rows = 0
    active_position = 0
    for sector_index in sector_slots(q)
        is_sector_zero(q, sector_index) && continue
        active_position += 1
        active_sector_indices[active_position] = sector_index
        keys_by_row[active_position] = _qr_cgtqr_row_keys(PS, QT, q, sector_index, Val(M))
        nrows = _count_qr_sector_product_splits(
            q, sector_index, left_legs, cgtqr_caches, Val(N))
        sector_row_ranges[active_position] = total_rows + 1:total_rows + nrows
        total_rows += nrows
    end
    @assert active_position == nactive
    return cgtqr_caches, active_sector_indices, keys_by_row, sector_row_ranges, total_rows
end

function _get_qr_symmetry_splits(q::AbstractTLArray{T, QD, N, RD, QT, PS, M, RMT},
                                 sector_index::Int,
                                 active_position::Int,
                                 keys_by_row,
                                 cgtqr_caches::CT,
                                 left_legs::NTuple{L, Int},
                                 right_legs::NTuple{R, Int},
                                 ::Val{n};
                                 tol::Float64 = 1e-12) where {T, QD, N, RD, QT, PS, M, RMT, L, R, CT<:Tuple, n}
    S = nth_symm(PS, Val(n))
    physical_key = if isabelian(S)
        _qr_physical_qlabel_key(QT, q, sector_index, Val(n))
    else
        keys_by_row[active_position][nonabelian_wmat_slot(PS, Val(n))]
    end
    QTN = typeof(physical_key[1])
    splits = Vector{_QRSymmetrySplit{L, R, QTN}}()
    left_signature, right_signature =
        _qr_physical_key_side_signatures(physical_key, left_legs, right_legs)
    qlabels, cgp, _, legdir =
        _qr_symmetry_stored_leg_order(QT, q, sector_index, Val(n))
    cache = _qr_cgtqr_cache_for(cgtqr_caches, S, productsymm(PS), Val(n))
    for block in _get_qr_cgt_split_blocks(S, qlabels, sector_wmat(q, sector_index, Val(n)),
                                          cgp, legdir, left_legs, right_legs, physical_key, cache)
        split = _reduce_qr_cgt_split(
            block, sector_index, left_signature, right_signature; tol=tol)
        isnothing(split) || push!(splits, split)
    end
    return splits
end

function _get_qr_cgt_split_sectors(q::AbstractTLArray{T, QD, N, RD, QT, PS, M, RMT},
                                   active_sector_indices::Vector{Int},
                                   keys_by_row,
                                   cgtqr_caches::CT,
                                   left_legs::NTuple{L, Int},
                                   right_legs::NTuple{R, Int};
                                   tol::Float64 = 1e-12) where {T, QD, N, RD, QT, PS, M, RMT, L, R, CT<:Tuple}
    return ntuple(Val(N)) do n
        splits = Vector{_QRSymmetrySplit{L, R, fieldtype(QT, n)}}()
        for (active_position, ri) in pairs(active_sector_indices)
            append!(splits, _get_qr_symmetry_splits(
                q, ri, active_position, keys_by_row, cgtqr_caches,
                left_legs, right_legs, Val(n); tol=tol))
        end
        splits
    end
end

function _sort_qr_splits_by_sector!(splits_by_symm)
    for splits in splits_by_symm
        sort!(splits; by = split -> (split.sector_index, split.q), alg=MergeSort)
    end
    return splits_by_symm
end

@generated function _qr_payload_tuples(::Type{PS}, sector_splits::ST) where {PS<:ProductSymm, ST<:Tuple}
    M = _wmat_tuple_width(PS)
    N = fieldcount(ST)
    left = Expr[]
    right = Expr[]
    cores = Expr[]
    for slot in 1:M
        n = product_symmetry_index_from_wmat_slot(PS, Val(slot))
        1 <= n <= N || return :(throw(BoundsError(sector_splits, $n)))
        push!(left, :(sector_splits[$n].left_iso))
        push!(right, :(sector_splits[$n].right_iso))
        push!(cores, :(sector_splits[$n].core))
    end
    return :(($(left...),), ($(right...),), ($(cores...),))
end

function _compact_qr_rows_by_valid!(rows::Vector{Row},
                                    left_isos::Vector{NTuple{M, Matrix{Float64}}},
                                    right_isos::Vector{NTuple{M, Matrix{Float64}}},
                                    cores::Vector{NTuple{M, Array{Float64, 3}}},
                                    valid_rows::BitVector) where {Row, M}
    write = 1
    for read in eachindex(rows)
        valid_rows[read] || continue
        if write != read
            rows[write] = rows[read]
            left_isos[write] = left_isos[read]
            right_isos[write] = right_isos[read]
            cores[write] = cores[read]
        end
        write += 1
    end
    last = write - 1
    resize!(rows, last)
    resize!(left_isos, last)
    resize!(right_isos, last)
    resize!(cores, last)
    return rows, left_isos, right_isos, cores
end

function _get_qr_split_rows(q::AbstractTLArray{T, QD, N, RD, QT, PS, M, RMT},
                            splits_by_symm::Tuple{Vararg{AbstractVector, N}},
                            sector_indices::Vector{Int},
                            sector_row_ranges::Vector{UnitRange{Int}},
                            left_legs::NTuple{L, Int},
                            right_legs::NTuple{R, Int},
                            ::Val{N},
                            total_rows::Int = 0) where {T, QD, N, RD, QT, PS, M, RMT, L, R}
    _sort_qr_splits_by_sector!(splits_by_symm)
    Row = _QRSplitRow{L, R, QT}
    rows = Vector{Row}(undef, total_rows)
    left_isos = Vector{NTuple{M, Matrix{Float64}}}(undef, total_rows)
    right_isos = Vector{NTuple{M, Matrix{Float64}}}(undef, total_rows)
    cores = Vector{NTuple{M, Array{Float64, 3}}}(undef, total_rows)
    (isempty(sector_indices) || any(isempty, splits_by_symm)) &&
        return Row[], NTuple{M, Matrix{Float64}}[],
               NTuple{M, Matrix{Float64}}[], NTuple{M, Array{Float64, 3}}[]
    valid_rows = falses(total_rows)

    nslots = maximum(sector_indices; init=0)
    choices_by_sector = ntuple(n -> [eltype(splits_by_symm[n])[] for _ in 1:nslots], N)
    for n in 1:N
        for split in splits_by_symm[n]
            push!(choices_by_sector[n][split.sector_index], split)
        end
    end

    for (active_position, ri) in pairs(sector_indices)
        next_row = first(sector_row_ranges[active_position])
        if !any(n -> isempty(choices_by_sector[n][ri]), 1:N)
            physical_key = _qr_physical_product_key(QT, q, ri)
            left_signature, right_signature =
                _qr_physical_key_side_signatures(physical_key, left_legs, right_legs)
            choices = ntuple(n -> choices_by_sector[n][ri], N)
            for splits in Iterators.product(choices...)
                sector_splits = Tuple(splits)
                sector = ntuple(n -> sector_splits[n].q, Val(N))::QT
                liso, riso, core = _qr_payload_tuples(PS, sector_splits)
                @assert next_row in sector_row_ranges[active_position]
                rows[next_row] = Row(ri, sector, left_signature, right_signature)
                left_isos[next_row] = liso
                right_isos[next_row] = riso
                cores[next_row] = core
                valid_rows[next_row] = true
                next_row += 1
            end
        end
    end

    _compact_qr_rows_by_valid!(rows, left_isos, right_isos, cores, valid_rows)

    perm = sortperm(eachindex(rows);
        by = i -> (rows[i].q, rows[i].left_signature, rows[i].right_signature,
                   rows[i].sector_index),
        alg=MergeSort)
    return rows[perm], left_isos[perm], right_isos[perm], cores[perm]
end

@inline _qr_row_signature(row, ::Val{:left}) = row.left_signature
@inline _qr_row_signature(row, ::Val{:right}) = row.right_signature
@inline _qr_signature_field(::Val{:left}) = :left_signature
@inline _qr_signature_field(::Val{:right}) = :right_signature

function _sort_qr_split_rows_by_class!(::Type{QT},
    rows::Vector{Row},
    left_payloads::Vector{NTuple{M, Matrix{Float64}}},
    right_payloads::Vector{NTuple{M, Matrix{Float64}}},
    core_payloads::Vector{NTuple{M, Array{Float64, 3}}},
    ::Val{L},
    ::Val{R}) where {QT, Row, M, L, R}
    LeftSig = NTuple{L, QT}
    RightSig = NTuple{R, QT}
    isempty(rows) && return UnitRange{Int}[]

    perm = Int[]
    class_ranges = UnitRange{Int}[]

    pos = 1
    while pos <= length(rows)
        q = rows[pos].q
        nextpos = pos
        while nextpos <= length(rows) && rows[nextpos].q == q
            nextpos += 1
        end

        row_range = pos:nextpos-1
        sectors = sort!(unique([rows[i].sector_index for i in row_range]); alg=MergeSort)
        rows_by_sector = Dict{Int, Vector{Int}}()
        left_groups = Dict{LeftSig, Vector{Int}}()
        right_groups = Dict{RightSig, Vector{Int}}()
        for i in row_range
            row = rows[i]
            push!(get!(rows_by_sector, row.sector_index, Int[]), i)
            push!(get!(left_groups, row.left_signature, Int[]), row.sector_index)
            push!(get!(right_groups, row.right_signature, Int[]), row.sector_index)
        end
        for group in values(left_groups)
            sort!(group; alg=MergeSort)
            unique!(group)
        end
        for group in values(right_groups)
            sort!(group; alg=MergeSort)
            unique!(group)
        end

        unassigned = Set(sectors)
        classes = Vector{Vector{Int}}()
        for seed in sectors
            seed in unassigned || continue
            component = Int[seed]
            frontier = Int[seed]
            delete!(unassigned, seed)
            seen_left = Set{LeftSig}()
            seen_right = Set{RightSig}()

            while !isempty(frontier)
                ri = pop!(frontier)
                for row_index in rows_by_sector[ri]
                    row = rows[row_index]
                    if row.left_signature ∉ seen_left
                        push!(seen_left, row.left_signature)
                        for rj in left_groups[row.left_signature]
                            rj in unassigned || continue
                            delete!(unassigned, rj)
                            push!(frontier, rj)
                            push!(component, rj)
                        end
                    end
                    if row.right_signature ∉ seen_right
                        push!(seen_right, row.right_signature)
                        for rj in right_groups[row.right_signature]
                            rj in unassigned || continue
                            delete!(unassigned, rj)
                            push!(frontier, rj)
                            push!(component, rj)
                        end
                    end
                end
            end

            sort!(component; alg=MergeSort)
            class_rows = Int[]
            for ri in component
                append!(class_rows, rows_by_sector[ri])
            end
            sort!(class_rows; by = i -> (rows[i].sector_index, rows[i].left_signature,
                                         rows[i].right_signature), alg=MergeSort)
            push!(classes, class_rows)
        end
        sort!(classes; by = cls -> rows[cls[1]].sector_index, alg=MergeSort)
        for class_rows in classes
            start = length(perm) + 1
            append!(perm, class_rows)
            push!(class_ranges, start:length(perm))
        end
        pos = nextpos
    end

    @assert length(perm) == length(rows)
    rows[:] = rows[perm]
    left_payloads[:] = left_payloads[perm]
    right_payloads[:] = right_payloads[perm]
    core_payloads[:] = core_payloads[perm]
    return class_ranges
end

function _normalize_qr_left_legs(left_legs, rank::Int)
    legs = collect(Int, left_legs)
    isempty(legs) && throw(ArgumentError("qr requires at least one left leg"))
    length(legs) < rank || throw(ArgumentError("qr requires at most $(rank - 1) left legs for rank-$rank input"))
    all(1 .<= legs .<= rank) || throw(ArgumentError("qr left_legs must lie between 1 and $rank"))
    length(unique(legs)) == length(legs) || throw(ArgumentError("qr left_legs must not contain duplicates"))
    return collect(sort(legs))
end

function _select_qr_left_legs(q::AbstractTLArray; dir=nothing, itag=nothing, plev=nothing,
                              lock=nothing, rev::Bool=false)
    if isnothing(dir) && isnothing(itag) && isnothing(plev) && isnothing(lock)
        throw(ArgumentError("keyword-based qr requires at least one of dir, itag, plev, or lock"))
    end

    legs = findlegs(q; dir=dir, itag=itag, plev=plev, lock=lock, rev=rev)
    rank = length(inds(q))
    1 <= length(legs) <= rank - 1 || throw(ArgumentError("keyword-based qr selected legs $legs, but the number of selected legs must be between 1 and $(rank - 1)"))
    return sort(legs)
end

function _qr_core_kron_matrix(cores::NTuple{0, Array{Float64, 3}})
    return reshape(Float64[1.0], 1, 1)
end

function _qr_core_kron_matrix(cores::NTuple{N, Array{Float64, 3}}) where {N}
    core = cores[1]
    mat = reshape(core, size(core, 1) * size(core, 2), size(core, 3))
    for n in 2:N
        core = cores[n]
        mat_n = reshape(core, size(core, 1) * size(core, 2), size(core, 3))
        mat = kron(mat_n, mat)
    end
    return mat
end

@inline _qr_left_iso(payload::NTuple{M, Matrix{Float64}}, ::Type{PS}, ::Val{n}) where {M, PS<:ProductSymm, n} =
    isabelian(product_symms(PS)[n]) ? _qr_trivial_iso() : payload[nonabelian_wmat_slot(PS, Val(n))]

@inline _qr_right_iso(payload::NTuple{M, Matrix{Float64}}, ::Type{PS}, ::Val{n}) where {M, PS<:ProductSymm, n} =
    isabelian(product_symms(PS)[n]) ? _qr_trivial_iso() : payload[nonabelian_wmat_slot(PS, Val(n))]

@inline _qr_core(payload::NTuple{M, Array{Float64, 3}}, ::Type{PS}, ::Val{n}) where {M, PS<:ProductSymm, n} =
    isabelian(product_symms(PS)[n]) ? _qr_trivial_core() : payload[nonabelian_wmat_slot(PS, Val(n))]

@generated function _qr_core_tuple(core_payload::CP,
                                   ::Type{PS},
                                   ::Val{N}) where {CP, PS<:ProductSymm, N}
    return Expr(:tuple, [:( _qr_core(core_payload, PS, Val($n)) ) for n in 1:N]...)
end

@generated function _qr_expanded_core_dims(rmt_size,
                                           cores::CT,
                                           ::Val{QD},
                                           ::Val{N}) where {CT<:Tuple, QD, N}
    exprs = Any[:(rmt_size[$i]) for i in 1:QD]
    for n in 1:N
        push!(exprs, :(size(cores[$n], 1)))
        push!(exprs, :(size(cores[$n], 2)))
    end
    return Expr(:tuple, exprs...)
end

@generated function _qr_expanded_class_perm(left_legs::NTuple{NL, Int},
                                            right_legs::NTuple{NR, Int},
                                            ::Val{QD},
                                            ::Val{N}) where {NL, NR, QD, N}
    exprs = Any[:(left_legs[$i]) for i in 1:NL]
    append!(exprs, [:(QD + 2 * $n - 1) for n in 1:N])
    append!(exprs, [:(right_legs[$i]) for i in 1:NR])
    append!(exprs, [:(QD + 2 * $n) for n in 1:N])
    return Expr(:tuple, exprs...)
end

function _qr_permutedims_to_buffer!(permute_buffer::Matrix{T},
                                    storage::Matrix{T},
                                    dims::NTuple{D, Int},
                                    perm::NTuple{D, Int}) where {T, D}
    data = reshape(view(storage, 1:prod(dims; init=1)), dims)
    output_size = prod(dims; init=1)
    @assert length(permute_buffer) >= output_size
    permute_view = reshape(view(permute_buffer, 1:output_size), ntuple(i -> dims[perm[i]], Val(D)))
    permute_view .= permutedims(data, perm)
    return permute_view
end

function _qr_permutedims_to_buffer!(permute_buffer::Matrix{Float64},
                                    storage::Matrix{Float64},
                                    dims::NTuple{D, Int},
                                    perm::NTuple{D, Int}) where {D}
    output_size = prod(dims; init=1)
    @assert length(permute_buffer) >= output_size
    permute_mat = reshape(view(permute_buffer, 1:output_size), output_size, 1)
    perm0 = Int32[perm[i] - 1 for i in 1:D]
    sizeA = Int32[dims[i] for i in 1:D]
    ccall((:dTensorTranspose, HPTT_jll.libhptt), Cvoid,
          (Ptr{Int32}, Cint, Float64, Ptr{Float64}, Ptr{Int32}, Ptr{Int32},
           Float64, Ptr{Float64}, Ptr{Int32}, Cint, Cint),
          perm0, D, 1.0, storage, sizeA, C_NULL,
          0.0, permute_mat, C_NULL, max(1, Threads.nthreads()), 0)
    return reshape(view(permute_buffer, 1:output_size), ntuple(i -> dims[perm[i]], Val(D)))
end

function _qr_permutedims_to_buffer!(permute_buffer::Matrix{ComplexF64},
                                    storage::Matrix{ComplexF64},
                                    dims::NTuple{D, Int},
                                    perm::NTuple{D, Int}) where {D}
    output_size = prod(dims; init=1)
    @assert length(permute_buffer) >= output_size
    permute_mat = reshape(view(permute_buffer, 1:output_size), output_size, 1)
    perm0 = Int32[perm[i] - 1 for i in 1:D]
    sizeA = Int32[dims[i] for i in 1:D]
    ccall((:zTensorTranspose, HPTT_jll.libhptt), Cvoid,
          (Ptr{Int32}, Cint, ComplexF64, Cuchar, Ptr{ComplexF64}, Ptr{Int32}, Ptr{Int32},
           ComplexF64, Ptr{ComplexF64}, Ptr{Int32}, Cint, Cint),
          perm0, D, one(ComplexF64), false, storage, sizeA, C_NULL,
          zero(ComplexF64), permute_mat, C_NULL, max(1, Threads.nthreads()), 0)
    return reshape(view(permute_buffer, 1:output_size), ntuple(i -> dims[perm[i]], Val(D)))
end

function _qr_ordered_unique_split_rows(rows::Vector{Row},
                                       class_row_indices::UnitRange{Int},
                                       side) where {Row}
    Sig = fieldtype(Row, _qr_signature_field(side))
    ordered = Int[]
    seen = Set{Sig}()
    for row_index in class_row_indices
        row = rows[row_index]
        sig = _qr_row_signature(row, side)
        sig in seen && continue
        push!(seen, sig)
        push!(ordered, row_index)
    end
    return ordered
end

function _qr_class_side_infos(q::AbstractTLArray{T, QD, N, RD, QT, PS},
                              rows::Vector{Row},
                              class_row_indices::UnitRange{Int},
                              left_payloads,
                              right_payloads,
                              legs::NTuple{L, Int},
                              side) where {T, QD, N, RD, QT, PS, L, Row}
    ordered = _qr_ordered_unique_split_rows(rows, class_row_indices, side)
    Sig = fieldtype(Row, _qr_signature_field(side))
    Info = _QRClassSideInfo{Sig, L, N}
    infos = Info[]
    ranges = Dict{Sig, UnitRange{Int}}()
    offset = 0

    for row_index in ordered
        row = rows[row_index]
        sig = _qr_row_signature(row, side)
        ri = row.sector_index
        rmt_size = sector_rmt_dim(q, ri)
        phys_dims = ntuple(i -> rmt_size[legs[i]], L)
        om_dims = ntuple(Val(N)) do n
            side === Val(:left) ?
                size(_qr_left_iso(left_payloads[row_index], PS, Val(n)), 2) :
                size(_qr_right_iso(right_payloads[row_index], PS, Val(n)), 2)
        end
        block_size = prod(phys_dims; init=1) * prod(om_dims; init=1)
        block_range = offset + 1:offset + block_size
        push!(infos, Info(sig, row_index, ri, phys_dims, om_dims, block_range))
        ranges[sig] = block_range
        offset += block_size
    end

    return infos, ranges, offset
end

function _qr_cgt_class_metadata(q::AbstractTLArray{T, QD, N, RD, QT, PS, M, RMT},
                                rows::Vector{Row},
                                class_row_indices::UnitRange{Int},
                                left_payloads,
                                right_payloads,
                                left_legs::NTuple{NL, Int},
                                right_legs::NTuple{NR, Int}) where {T, QD, N, RD, QT, PS, M, RMT, NL, NR, Row}
    left_infos, left_ranges, total_left = _qr_class_side_infos(
        q, rows, class_row_indices, left_payloads, right_payloads, left_legs, Val(:left))

    right_infos, right_ranges, total_right = _qr_class_side_infos(
        q, rows, class_row_indices, left_payloads, right_payloads, right_legs, Val(:right))

    return _QRCGTClassMetadata(
        rows[first(class_row_indices)].q, class_row_indices,
        left_infos, left_ranges, total_left,
        right_infos, right_ranges, total_right)
end

function _qr_cgt_class_metadata(
    q::AbstractTLArray{T, QD, N, RD, QT, PS, M, RMT},
    rows::Vector{Row},
    class_ranges::Vector{UnitRange{Int}},
    left_payloads,
    right_payloads,
    left_legs::NTuple{NL, Int},
    right_legs::NTuple{NR, Int}) where {T, QD, N, RD, QT, PS, M, RMT, NL, NR, Row}
    isempty(class_ranges) && throw(ArgumentError("qr produced no CGT classes"))

    Meta = _qr_cgt_class_metadata_type(QT, Val(NL), Val(NR), Val(N))
    metadata = Vector{Meta}(undef, length(class_ranges))
    max_left = 0
    max_right = 0

    for ci in eachindex(class_ranges)
        item = _qr_cgt_class_metadata(
            q, rows, class_ranges[ci], left_payloads, right_payloads,
            left_legs, right_legs)
        metadata[ci] = item
        max_left = max(max_left, item.total_left)
        max_right = max(max_right, item.total_right)
    end
    return metadata, max_left, max_right
end

function _qr_sector_class_matrix!(dest::AbstractMatrix{Tbuf},
                                  q::AbstractTLArray{T, QD, N, RD, QT, PS},
                                  sector_index::Int,
                                  left_payload,
                                  right_payload,
                                  core_payload,
                                  left_legs::NTuple{NL, Int},
                                  right_legs::NTuple{NR, Int},
                                  product_buffer::Matrix{Tbuf},
                                  permute_buffer::Matrix{Tbuf}) where {T, QD, N, RD, QT, PS, NL, NR, Tbuf}
    rmt, rmt_scale = sector_rmt_permuted(q, sector_index, _identity_rmt_perm(Val(RD)))
    rmt_size = sector_rmt_dim(q, sector_index)
    phys_dim = prod(rmt_size[i] for i in 1:QD)
    om_dim = prod(rmt_size[QD + n] for n in 1:N; init=1)
    cores = _qr_core_tuple(core_payload, PS, Val(N))
    core_mat = _qr_core_kron_matrix(cores)
    @assert size(core_mat, 2) == om_dim

    expanded_om_dim = size(core_mat, 1)
    product_size = phys_dim * expanded_om_dim
    @assert length(product_buffer) >= product_size
    product_mat = reshape(view(product_buffer, 1:product_size), phys_dim, expanded_om_dim)
    mul!(product_mat, reshape(rmt, phys_dim, om_dim), transpose(core_mat), Tbuf(rmt_scale), zero(Tbuf))

    expanded_dims = _qr_expanded_core_dims(rmt_size, cores, Val(QD), Val(N))
    perm = _qr_expanded_class_perm(left_legs, right_legs, Val(QD), Val(N))
    left_dim = prod(rmt_size[leg] for leg in left_legs; init=1) *
               prod(size(_qr_left_iso(left_payload, PS, Val(n)), 2) for n in 1:N; init=1)
    right_dim = prod(rmt_size[leg] for leg in right_legs; init=1) *
                prod(size(_qr_right_iso(right_payload, PS, Val(n)), 2) for n in 1:N; init=1)
    @assert size(dest) == (left_dim, right_dim)
    permed = _qr_permutedims_to_buffer!(permute_buffer, product_buffer, expanded_dims, perm)
    dest .+= reshape(permed, left_dim, right_dim)
    return dest
end

function _build_qr_cgt_class(q::AbstractTLArray{T, QD, N, RD, QT, PS, M, RMT},
                                class_metadata::Meta,
                                rows::Vector{Row},
                                left_payloads,
                                right_payloads,
                                core_payloads,
                                left_legs::NTuple{NL, Int},
                                right_legs::NTuple{NR, Int},
                                irrepdim_caches::ICT,
                                buffer::AbstractMatrix{Tbuf},
                                product_buffer::Matrix{Tbuf},
                                permute_buffer::Matrix{Tbuf}) where {T, QD, N, RD, QT, PS, M, RMT, NL, NR, Row, Meta, ICT<:Tuple, Tbuf}
    sector = class_metadata.sector
    class_row_indices = class_metadata.rows
    left_ranges = class_metadata.left_ranges
    right_ranges = class_metadata.right_ranges
    total_left = class_metadata.total_left
    total_right = class_metadata.total_right

    @assert size(buffer, 1) >= total_left && size(buffer, 2) >= total_right
    mat = @view buffer[1:total_left, 1:total_right]
    fill!(mat, zero(Tbuf))
    for row_index in class_row_indices
        row = rows[row_index]
        lrange = left_ranges[row.left_signature]
        rrange = right_ranges[row.right_signature]
        block = @view mat[lrange, rrange]
        _qr_sector_class_matrix!(
            block, q, row.sector_index, left_payloads[row_index], right_payloads[row_index],
            core_payloads[row_index], left_legs, right_legs, product_buffer, permute_buffer)
    end

    m, n = size(mat)
    k = min(m, n)
    F = LinearAlgebra.qr(mat)
    dimq = Float64(_qr_sector_degeneracy(PS, sector, irrepdim_caches, Val(N)))
    scale = sqrt(dimq)
    Result = _qr_class_result_type(Meta, Tbuf)
    return Result(
        sector, class_metadata.left_infos, class_metadata.right_infos,
        Matrix(F.Q[:, 1:k]) .* scale, Matrix(F.R[1:k, :]))
end

function _build_qr_cgt_classes(q::AbstractTLArray{T, QD, N, RD, QT, PS, M, RMT},
                                  class_metadata::Vector{Meta},
                                  max_left::Int,
                                  max_right::Int,
                                  rows::Vector{Row},
                                  left_payloads,
                                  right_payloads,
                                  core_payloads,
                                  left_legs::NTuple{NL, Int},
                                  right_legs::NTuple{NR, Int},
                                  irrepdim_caches::ICT) where {T, QD, N, RD, QT, PS, M, RMT, Meta, Row, NL, NR, ICT<:Tuple}
    isempty(class_metadata) && throw(ArgumentError("qr produced no CGT classes"))
    Tmat = promote_type(T, Float64)
    buffer = Matrix{Tmat}(undef, max_left, max_right)
    product_buffer = similar(buffer)
    permute_buffer = similar(buffer)
    Result = _qr_class_result_type(Meta, Tmat)
    class_results = Vector{Result}(undef, length(class_metadata))
    for ci in eachindex(class_metadata)
        class_results[ci] = _build_qr_cgt_class(
            q, class_metadata[ci], rows, left_payloads, right_payloads, core_payloads,
            left_legs, right_legs, irrepdim_caches, buffer, product_buffer, permute_buffer)
    end
    return class_results
end

function _qr_class_ranges(class_results::AbstractVector, ::Val{N}) where {N}
    class_ranges = Vector{UnitRange{Int}}(undef, length(class_results))
    Sector = NTuple{N, Tuple{Vararg{Int}}}
    sector_counts = Dict{Sector, Int}()
    sector_order = Sector[]

    for (ci, result) in enumerate(class_results)
        k = size(result.Q, 2)
        if k == 0
            class_ranges[ci] = 1:0
            continue
        end
        sector = result.sector::Sector
        if !haskey(sector_counts, sector)
            sector_counts[sector] = 0
            push!(sector_order, sector)
        end
        start = sector_counts[sector] + 1
        stop = sector_counts[sector] + k
        class_ranges[ci] = start:stop
        sector_counts[sector] = stop
    end

    return class_ranges, sector_counts, sector_order
end

function _qr_build_side_cgt_metadata(source_qlabels::NTuple{QD},
                                     source_cgp::NTuple{QD, Int},
                                     source_legdir::Tuple{Int, Int},
                                     phys_legs::NTuple{L, Int},
                                     bond_q,
                                     bond_first::Bool,
                                     bond_dir::Char,
                                     wmat::AbstractMatrix{Float64}) where {QD, L}
    stored_phys = sort!([(source_cgp[leg], leg) for leg in phys_legs]; by = first, alg=MergeSort)
    nin = source_legdir[1]

    Entry = Tuple{typeof(bond_q), Int}
    incoming = Entry[]
    outgoing = Entry[]
    bond_source = bond_first ? 0 : length(source_qlabels) + 1
    if bond_first
        bond_dir == '+' ? push!(incoming, (bond_q, bond_source)) :
                          push!(outgoing, (bond_q, bond_source))
    end
    for (stored_pos, leg) in stored_phys
        entry = (source_qlabels[stored_pos], leg)
        if stored_pos <= nin
            push!(incoming, entry)
        else
            push!(outgoing, entry)
        end
    end
    if !bond_first
        bond_dir == '+' ? push!(incoming, (bond_q, bond_source)) :
                          push!(outgoing, (bond_q, bond_source))
    end
    sort!(incoming; by = first, alg=MergeSort)
    sort!(outgoing; by = first, alg=MergeSort)

    qlabels = (Tuple(entry[1] for entry in incoming)...,
               Tuple(entry[1] for entry in outgoing)...)
    legdir = (length(incoming), length(outgoing))

    source_to_stored = Dict{Int, Int}()
    for (stored_pos, (_, source)) in enumerate((incoming..., outgoing...))
        source_to_stored[source] = stored_pos
    end

    final_sources = bond_first ? (bond_source, phys_legs...) : (phys_legs..., bond_source)
    final_cgp = Tuple(source_to_stored[source] for source in final_sources)

    return (qlabels = qlabels, wmat = wmat, cgp = final_cgp, legdir = legdir)
end

"""
    qr_std(q, left_legs, bond_tag="qr") -> QRResult

Internal implementation of Telum's symmetry-aware QR. `left_legs` is a sorted,
unique tuple of logical legs placed on the Q side; all other legs form the R
side. `bond_tag` labels the generated bond index. The routine materializes lazy
input, constructs CGT split classes, performs dense QR independently per class,
and assembles `Q`/`R` TLArrays without changing `q`.
"""
function qr_std(q::AbstractTLArray{T, QD, N, RD, QT, PS, M, RMT},
                left_legs::NTuple{L, Int},
                bond_tag::AbstractString = "qr") where {T, QD, N, RD, QT, PS, M, RMT, L}
    materialize(q)
    @assert issorted(left_legs)
    @assert length(unique(left_legs)) == length(left_legs)
    N == 0 && return _qr_unsymmetric(q, left_legs, bond_tag)

    R = QD - L
    right_legs::NTuple{R, Int} = _qr_right_legs(Val(QD), left_legs)

    symmetries = product_symms(PS)

    cgtqr_caches, active_sector_indices, keys_by_row, sector_row_ranges, total_rows =
        _prepare_qr_counted_preassembly(q, left_legs, Val(N))
    splits_by_symm = _get_qr_cgt_split_sectors(
        q, active_sector_indices, keys_by_row, cgtqr_caches,
        left_legs, right_legs; tol=1e-12)
    split_rows, left_payloads, right_payloads, core_payloads =
        _get_qr_split_rows(q, splits_by_symm, active_sector_indices, sector_row_ranges,
                            left_legs, right_legs, Val(N), total_rows)
        split_row_classes = _sort_qr_split_rows_by_class!(
        QT, split_rows, left_payloads, right_payloads, core_payloads, Val(L), Val(R))
    class_metadata, max_left, max_right = _qr_cgt_class_metadata(
        q, split_rows, split_row_classes, left_payloads, right_payloads,
        left_legs, right_legs)
    irrepdim_caches = _fill_qr_irrepdim_caches!(_new_qr_irrepdim_caches(q), class_metadata, PS)
    class_results = _build_qr_cgt_classes(
        q, class_metadata, max_left, max_right, split_rows,
        left_payloads, right_payloads, core_payloads, left_legs, right_legs,
        irrepdim_caches)
    class_ranges, sector_counts, sector_order = _qr_class_ranges(class_results, Val(N))

    Tout = promote_type(T, Float64)
    qlabels_Q = NTuple{L + 1, QT}[]
    qlabels_R = NTuple{R + 1, QT}[]
    RMTs_Q = Array{Tout, L + 1 + N}[]
    RMTs_R = Array{Tout, R + 1 + N}[]
    nonabelian_indices = nonabelian_symmetry_indices(PS)

    n_wmat_Q = 0
    n_wmat_R = 0
    total_wmat_Q = 0
    total_wmat_R = 0
    for result in class_results
        for info in result.left_infos
            n_wmat_Q += 1
            payload = left_payloads[info.row_index]
            for slot in 1:M
                total_wmat_Q += length(payload[slot])
            end
        end
        for info in result.right_infos
            n_wmat_R += 1
            payload = right_payloads[info.row_index]
            for slot in 1:M
                total_wmat_R += length(payload[slot])
            end
        end
    end
    wmatdata_Q = Vector{Float64}(undef, total_wmat_Q)
    wmatinfo_Q = Vector{WMatInfo{M}}(undef, n_wmat_Q)
    wmatdata_R = Vector{Float64}(undef, total_wmat_R)
    wmatinfo_R = Vector{WMatInfo{M}}(undef, n_wmat_R)
    next_wmat_offset_Q = 1
    next_wmat_sector_Q = 1
    next_wmat_offset_R = 1
    next_wmat_sector_R = 1

    for (ci, result) in enumerate(class_results)
        sector = result.sector
        sector_count = sector_counts[sector]
        class_range = class_ranges[ci]

        for info in result.left_infos
            left_payload = left_payloads[info.row_index]
            part = result.Q[info.range, :]
            full = zeros(eltype(result.Q), length(info.range), sector_count)
            full[:, class_range] = part
            tmp = reshape(full, info.phys_dims..., info.om_dims..., sector_count)
            perm = Tuple(vcat(collect(1:L), L + N + 1, collect(L+1:L+N)))
            rmt_Q = permutedims(tmp, perm)
            rmt_Q_iszero = _rmt_iszero(rmt_Q)

            cgts_Q = ntuple(N) do n
                source_qlabels, source_cgp, _, source_legdir =
                    _qr_symmetry_stored_leg_order(QT, q, info.sector_index, Val(n))
                wmat = isabelian(nth_symm(PS, Val(n))) ? _qr_trivial_iso() :
                    _qr_left_iso(left_payload, PS, Val(n))
                _qr_build_side_cgt_metadata(
                    source_qlabels, source_cgp, source_legdir, left_legs,
                    sector[n], false, '-', wmat)
            end
            push!(qlabels_Q,
                  _qr_physical_qlabels(QT,
                                         ntuple(n -> cgts_Q[n].qlabels, Val(N)),
                                         ntuple(n -> cgts_Q[n].cgp, Val(N)),
                                         Val(L + 1)))
            resize!(RMTs_Q, length(qlabels_Q))
            if rmt_Q_iszero
                wmatinfo_Q[next_wmat_sector_Q] = _empty_wmat_info(Val(M))
            else
                next_wmat_offset_Q = _store_wmat_tuple!(
                    wmatdata_Q, wmatinfo_Q, next_wmat_sector_Q, next_wmat_offset_Q,
                    ntuple(slot -> cgts_Q[nonabelian_indices[slot]].wmat, Val(M)))
                RMTs_Q[end] = rmt_Q
            end
            next_wmat_sector_Q += 1
        end

        for info in result.right_infos
            right_payload = right_payloads[info.row_index]
            part = result.R[:, info.range]
            full = zeros(eltype(result.R), sector_count, length(info.range))
            full[class_range, :] = part
            rmt_R = reshape(full, sector_count, info.phys_dims..., info.om_dims...)
            rmt_R_iszero = _rmt_iszero(rmt_R)

            cgts_R = ntuple(N) do n
                source_qlabels, source_cgp, _, source_legdir =
                    _qr_symmetry_stored_leg_order(QT, q, info.sector_index, Val(n))
                wmat = isabelian(nth_symm(PS, Val(n))) ? _qr_trivial_iso() :
                    _qr_right_iso(right_payload, PS, Val(n))
                _qr_build_side_cgt_metadata(
                    source_qlabels, source_cgp, source_legdir, right_legs,
                    sector[n], true, '+', wmat)
            end
            push!(qlabels_R,
                  _qr_physical_qlabels(QT,
                                         ntuple(n -> cgts_R[n].qlabels, Val(N)),
                                         ntuple(n -> cgts_R[n].cgp, Val(N)),
                                         Val(R + 1)))
            resize!(RMTs_R, length(qlabels_R))
            if rmt_R_iszero
                wmatinfo_R[next_wmat_sector_R] = _empty_wmat_info(Val(M))
            else
                next_wmat_offset_R = _store_wmat_tuple!(
                    wmatdata_R, wmatinfo_R, next_wmat_sector_R, next_wmat_offset_R,
                    ntuple(slot -> cgts_R[nonabelian_indices[slot]].wmat, Val(M)))
                RMTs_R[end] = rmt_R
            end
            next_wmat_sector_R += 1
        end
    end

    bond_splist = Tuple{NTuple{N, Tuple{Vararg{Int}}}, Int}[
        (sector, sector_counts[sector]) for sector in sector_order
    ]

    inds_Q = (ntuple(i -> inds(q)[left_legs[i]], Val(L))..., TLIndex(bond_tag, '-'))
    inds_R = (TLIndex(bond_tag, '+'), ntuple(i -> inds(q)[right_legs[i]], Val(R))...)

    spaces_Q = (ntuple(i -> spaces(q)[left_legs[i]], Val(L))..., bond_splist)
    spaces_R = (bond_splist, ntuple(i -> spaces(q)[right_legs[i]], Val(R))...)

    Q_qlabels = Matrix{QT}(undef, L + 1, length(qlabels_Q))
    for sector_index in eachindex(qlabels_Q), leg in 1:(L + 1)
        Q_qlabels[leg, sector_index] = qlabels_Q[sector_index][leg]
    end
    R_qlabels = Matrix{QT}(undef, R + 1, length(qlabels_R))
    for sector_index in eachindex(qlabels_R), leg in 1:(R + 1)
        R_qlabels[leg, sector_index] = qlabels_R[sector_index][leg]
    end

    Q = TLArray(symmetries, Q_qlabels, wmatdata_Q, wmatinfo_Q, RMTs_Q, inds_Q, spaces_Q)
    Rfac = TLArray(symmetries, R_qlabels, wmatdata_R, wmatinfo_R, RMTs_R, inds_R, spaces_R)
    return QRResult(Q, Rfac)
end

function _qr_unsymmetric(q::AbstractTLArray{T, QD, 0},
                         left_legs::NTuple{L, Int},
                         bond_tag::AbstractString) where {T, QD, L}
    right_legs = _qr_right_legs(Val(QD), left_legs)
    rmt, scale = sector_rmt_permuted(q, 1, _identity_rmt_perm(Val(QD)))
    dense = Array(scale .* rmt)
    perm = (left_legs..., right_legs...)
    left_dims = Tuple(size(q, leg) for leg in left_legs)
    right_dims = Tuple(size(q, leg) for leg in right_legs)
    m = prod(left_dims; init=1)
    n = prod(right_dims; init=1)
    matrix = reshape(permutedims(dense, perm), m, n)

    k = min(m, n)
    factorization = LinearAlgebra.qr(matrix)
    Qmatrix = Matrix(factorization.Q)[:, 1:k]
    Rmatrix = factorization.R[1:k, :]
    Qdata = reshape(Qmatrix, (left_dims..., k))
    Rdata = reshape(Rmatrix, (k, right_dims...))

    Qinds = (ntuple(i -> inds(q)[left_legs[i]], Val(L))..., TLIndex(bond_tag, '-'))
    Rinds = (TLIndex(bond_tag, '+'), ntuple(i -> inds(q)[right_legs[i]], Val(QD - L))...)
    return QRResult(TLArray(Qdata, Qinds), TLArray(Rdata, Rinds))
end

function LinearAlgebra.qr(q::AbstractTLArray{T, QD, N, RD},
                          left_legs,
                          bond_tag::AbstractString = "qr") where {T, QD, N, RD}
    left_legs_ = Tuple(_normalize_qr_left_legs(left_legs, QD))
    return qr_std(q, left_legs_, bond_tag)
end

function LinearAlgebra.qr(q::AbstractTLArray{T, QD, N, RD},
                          bond_tag::AbstractString = "qr";
                          dir=nothing,
                          itag=nothing,
                          plev=nothing,
                          lock=nothing,
                          rev::Bool=false) where {T, QD, N, RD}
    left_legs = _select_qr_left_legs(q; dir=dir, itag=itag, plev=plev, lock=lock, rev=rev)
    return qr(q, left_legs, bond_tag)
end
