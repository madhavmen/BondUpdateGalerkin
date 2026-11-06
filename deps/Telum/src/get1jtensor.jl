get1jtensor(q::AbstractTLArray{T, QD, N, RD}, leg::Int) where {T, QD, N, RD} =
get1jtensor(leginfo(q, leg))

function _resolve_unique_leg(q::AbstractTLArray; dir=nothing, itag=nothing, plev=nothing,
                             lock=nothing, rev::Bool=false,
                             opname::AbstractString="operation")
    legs = findlegs(q; dir=dir, itag=itag, plev=plev, lock=lock, rev=rev)
    length(legs) == 1 && return only(legs)
    isempty(legs) && throw(ArgumentError("$opname requires a uniquely specified leg, but no legs matched"))
    throw(ArgumentError("$opname requires a uniquely specified leg, but matched legs $legs"))
end

function _resolve_matching_legs(q::AbstractTLArray; dir=nothing, itag=nothing, plev=nothing,
                                lock=nothing, rev::Bool=false,
                                opname::AbstractString="operation")
    legs = findlegs(q; dir=dir, itag=itag, plev=plev, lock=lock, rev=rev)
    isempty(legs) && throw(ArgumentError("$opname requires at least one matching leg, but no legs matched"))
    return legs
end

function _normalize_legflip_legs(q::AbstractTLArray{T, QD}, legs) where {T, QD}
    positions = legs isa Integer ? [Int(legs)] : Int[leg for leg in legs]
    isempty(positions) && throw(ArgumentError("legflip requires at least one leg"))

    for leg in positions
        1 <= leg <= QD || throw(BoundsError(q, leg))
    end

    length(unique(positions)) == length(positions) || throw(ArgumentError(
        "legflip legs must be unique, got $positions"))
    return positions
end

@inline _dual_product_qlabel(symmetries::Tuple, qlabel::Tuple) =
    ntuple(n -> get_dualq(symmetries[n], qlabel[n]), Val(length(symmetries)))

function _legflip_abelian(q::TLArray{T, QD}, leg::Int) where {T, QD}
    stored_leg = _logical_leg(q, leg)
    symmetries = symm(q)
    qlabels = copy(stored_qlabels(q))
    for sector in eachindex(qlabels)
        sector_labels = qlabels[sector]
        dual_label = _dual_product_qlabel(symmetries, sector_labels[stored_leg])
        qlabels[sector] = ntuple(l -> l == stored_leg ? dual_label : sector_labels[l], Val(QD))
    end

    source_spaces = stored_spaces(q)
    spaces = ntuple(l -> begin
        if l == stored_leg
            sort!([(_dual_product_qlabel(symmetries, qlabel), dim)
                   for (qlabel, dim) in source_spaces[l]], by=first)
        else
            copy(source_spaces[l])
        end
    end, Val(QD))
    inds = ntuple(l -> l == stored_leg ? change_dual(change_dir(stored_inds(q)[l])) :
                  stored_inds(q)[l], Val(QD))

    wmatdata, wmatinfo = stored_wmat_storage(q)
    return TLArray(Val(:alias_storage_view_state), symmetries, qlabels,
                   copy(wmatdata), copy(wmatinfo), copy(q.RMTs),
                   copy(q.isdefined), copy(q.iszero), inds, spaces,
                   stored_conj(q), stored_scale(q), stored_perm(q))
end

"""
    get1jtensor(q; dir, itag, plev, lock, rev=false)

Construct the one-leg identity tensor for exactly one leg of `q`. `dir`,
`itag`, `plev`, and `lock` filter candidates with `findleg`; `rev=true` searches
from the final leg. The selection must be unique or an `ArgumentError` is
thrown. The returned tensor carries the selected space with the appropriate
identity/duality structure and is independent of `q`'s RMT payloads.
"""
function get1jtensor(q::AbstractTLArray; dir=nothing, itag=nothing, plev=nothing,
                     lock=nothing, rev::Bool=false)
    leg = _resolve_unique_leg(q; dir=dir, itag=itag, plev=plev, lock=lock,
                              rev=rev, opname="get1jtensor")
    return get1jtensor(q, leg)
end

function get1jtensor(info::leginfo{N, QT, PS}) where {N, QT, PS}
    symm, ind = product_symms(PS), info.ind
    inds1 = (change_dir(ind), change_dir(change_dual(ind)))
    dir1 = inds1[1].dir

    nr = length(info.splist)
    qlabels1 = Matrix{QT}(undef, 2, nr)
    wmatdata1, wmatinfo1 = _unit_wmat_storage(PS, nr)

    # leg 1 = original space, leg 2 = dual space
    ET = eltype(info.splist)
    dual_splist = sort!(ET[(Tuple(get_dualq(symm[n], qlabels[n]) for n in 1:N), RMTd)
                           for (qlabels, RMTd) in info.splist], by=x->x[1])
    spaces1 = (info.splist, dual_splist)
    RMTs1 = Vector{DiagRMT{Float64, 2 + N}}(undef, nr)

    for (sector_index, (qlabels, RMTd)) in enumerate(info.splist)
        dual_qlabels = Tuple(get_dualq(symm[n], qlabels[n]) for n in 1:N)
        qlabels1[1, sector_index] = qlabels
        qlabels1[2, sector_index] = dual_qlabels

        cgt_dim = 1.0
        for n in 1:N
            cgt_dim *= dimension(symm[n], qlabels[n])
        end
        scale = sqrt(cgt_dim)
        RMTs1[sector_index] = _diag_rmt_from_values(ones(Float64, RMTd), Val(2 + N), (1, 2), scale)
    end

    q1 = TLArray(symm, qlabels1, wmatdata1, wmatinfo1, RMTs1, inds1, spaces1)
    return q1
end

"""
    legflip(q, leg)
    legflip(q, legs)

Reverse the direction of each selected leg and apply the associated symmetry
duality map. `leg` is one 1-based index; `legs` is a list of unique indices;
the keyword-selector form uses `findlegs`. The operation returns a new tensor
or lazy wrapper, preserves stable sector-slot numbering, and leaves `q`
unchanged. It throws when a requested index is invalid.
"""
function legflip(q::TLArray{T, QD, N, RD}, leg::Int) where {T, QD, N, RD}
    1 <= leg <= QD || throw(BoundsError(q, leg))
    if all(isabelian, symm(q))
        return _legflip_abelian(q, leg)
    end
    j = get1jtensor(q, leg)
    q_flip = contract(q, (leg,), j, (1,); reduce_lock=false)
    perm = (ntuple(i -> i, leg - 1)..., QD, ntuple(i -> leg - 1 + i, QD - leg)...)
    return permutedims(q_flip, perm)
end

function legflip(q::AbstractTLArray{T, QD, N, RD}, leg::Int) where {T, QD, N, RD}
    1 <= leg <= QD || throw(BoundsError(q, leg))
    j = get1jtensor(q, leg)
    q_flip = contract(q, (leg,), j, (1,); reduce_lock=false)
    perm = (ntuple(i -> i, leg - 1)..., QD, ntuple(i -> leg - 1 + i, QD - leg)...)
    return permutedims(q_flip, perm)
end

function legflip(q::AbstractTLArray, legs::LegList)
    q_flip = q
    for leg in _normalize_legflip_legs(q, legs)
        q_flip = legflip(q_flip, leg)
    end
    return q_flip
end

function legflip(q::AbstractTLArray; dir=nothing, itag=nothing, plev=nothing,
                 lock=nothing, rev::Bool=false)
    legs = _resolve_matching_legs(q; dir=dir, itag=itag, plev=plev, lock=lock,
                                  rev=rev, opname="legflip")
    return legflip(q, legs)
end
