# ─── eigen ────────────────────────────────────────────────────────────────────
#
# Perform symmetry-adapted eigendecomposition of a rank-2 TLArray object.
#
# Arguments:
#   q          : TLArray{T, 2, N, RD} to decompose (rank 2, one incoming and one outgoing leg)
#   eig_tag    : itag for the new eigenvector bond leg (default "eig")
#   hermitian  : if true, assume each RMT block is Hermitian and skip the
#                Hermiticity check. If false (default), compute
#                norm(q - q') / norm(q) and dispatch to the Hermitian or
#                general eigensolver automatically.
#
# Returns `EigenResult(V, D, V_inv, eig_list)` where:
#   D        : diagonal TLArray with legs normalized to ('+', '-')
#   V        : eigenvector TLArray with legs normalized to ('+', '-')
#   V_inv    : `nothing` on the Hermitian path, inverse eigenvector TLArray on
#              the general path
#   eig_list : Vector{Tuple{eigenval_type, Int, sector_qlabels, sector_index}}
#              — (eigenvalue, degeneracy, sector, in-sector index) entries
#              sorted by ascending eigenvalue (real for Hermitian, abs for non-Hermitian).
#              degeneracy = ∏_n dim(irep at leg 1, symm n) for the sector.
#
# Convention:
#   A = V * D * V'   (Hermitian; eigenvectors are orthonormal columns of V)
#
# ─────────────────────────────────────────────────────────────────────────────

_eig_sector_qlabels(q::AbstractTLArray{T, 2, N}, sector_index::Int) where {T, N} =
    ntuple(n -> sector_qlabel(q, sector_index, 1)[n], Val(N))

_eig_sort_value(ev::Real, hermitian::Bool) = hermitian ? real(ev) : abs(ev)
_eig_sort_value(ev::Complex, hermitian::Bool) = hermitian ? real(ev) : abs(ev)
const _EIG_HERMITIAN_RTOL = sqrt(eps(Float64))

"""
Container for a symmetry-aware eigendecomposition.

# Fields

- `V`: right-eigenvector TLArray.
- `D`: diagonal eigenvalue TLArray.
- `V_inv`: inverse/left-eigenvector factor, or `nothing` when not constructed.
- `eig_list`: globally ordered eigenvalue/sector/local-index bookkeeping used by `discard_eigen`.
"""
struct EigenResult{TV, TD, TVI, TL}
    V::TV
    D::TD
    V_inv::TVI
    eig_list::TL
end

function EigenResult(V::TLArray, D::TLArray, V_inv, eig_list)
    return EigenResult{typeof(V), typeof(D), typeof(V_inv), typeof(eig_list)}(
        V, D, V_inv, eig_list)
end

function _append_missing_eig_sectors!(symm::NTuple{N, Any},
                                      spaces,
                                      covered,
                                      qlabels_D,
                                      RMTs_D,
                                      qlabels_V,
                                      RMTs_V,
                                      qlabels_Vinv,
                                      RMTs_Vinv,
                                      eig_list,
                                      cgp::NTuple{2, Int},
                                      ::Type{T_out}) where {N, T_out}
    for (sector_qlabels, dim) in spaces
        sector_qlabels ∈ covered && continue

        cgt_dim = prod(dimension(symm[n], sector_qlabels[n]) for n in 1:N)
        for j in 1:dim
            push!(eig_list, (zero(T_out), cgt_dim, sector_qlabels, j))
        end

        rmt_eye = reshape(Matrix{T_out}(I, dim, dim), dim, dim, ones(Int, N)...)
        rmt_eye[:] .*= sqrt(Float64(cgt_dim))

        push!(qlabels_V, sector_qlabels)
        push!(RMTs_V, rmt_eye)

        if !isnothing(RMTs_Vinv)
            push!(qlabels_Vinv, sector_qlabels)
            push!(RMTs_Vinv, rmt_eye)
        end
    end
end

function _eig_diagonal_rmt(::Vector{<:Array}, vals::AbstractVector{T}, scale, ::Val{RD}) where {T, RD}
    return _dense_diagonal_rmt_from_values(vals, Val(RD), scale)
end

function _eig_diagonal_rmt(::Vector{<:DiagRMT}, vals::AbstractVector{T}, scale, ::Val{RD}) where {T, RD}
    return _diag_rmt_from_values(vals, Val(RD), (1, 2), scale)
end

# TODO: This is not optimized.
function _renumber_eig_entries(eig_entries)
    isempty(eig_entries) && return copy(eig_entries)
    sector_indices = Dict{typeof(eig_entries[1][3]), Vector{Int}}()
    for entry in eig_entries
        push!(get!(sector_indices, entry[3], Int[]), entry[4])
    end

    sector_maps = Dict{typeof(eig_entries[1][3]), Dict{Int, Int}}()
    for (sector, idxs) in sector_indices
        sector_maps[sector] = Dict(old_idx => new_idx for (new_idx, old_idx) in enumerate(sort(unique(idxs))))
    end

    out = similar(eig_entries)
    for (i, entry) in enumerate(eig_entries)
        ev, deg, sector, old_idx = entry
        out[i] = (ev, deg, sector, sector_maps[sector][old_idx])
    end
    return out
end

function _eig_qlabel_matrix(sectors::AbstractVector{QT}) where {QT}
    qlabels = Matrix{QT}(undef, 2, length(sectors))
    for (sector_index, sector) in enumerate(sectors)
        qlabels[1, sector_index] = sector
        qlabels[2, sector_index] = sector
    end
    return qlabels
end

_retag_qindex(idx::TLIndex, tag::AbstractString) = TLIndex(tag, idx.dir, idx.plev, idx.lock, idx.dual)

function _retag_eigen_result(result::EigenResult, eig_tag::AbstractString)
    d_inds = (_retag_qindex(inds(result.D)[1], eig_tag), _retag_qindex(inds(result.D)[2], eig_tag))
    D = TLArray(result.D, d_inds)

    v_inds = (inds(result.V)[1], _retag_qindex(inds(result.V)[2], eig_tag))
    V = TLArray(result.V, v_inds)

    V_inv = if isnothing(result.V_inv)
        nothing
    else
        vinv_inds = (_retag_qindex(inds(result.V_inv)[1], eig_tag), inds(result.V_inv)[2])
        TLArray(result.V_inv, vinv_inds)
    end

    return EigenResult(V, D, V_inv, result.eig_list)
end

"""Validate and orient rank-two tensor `q` for eigenvalue routines. `opname` is used in diagnostics; the result has incoming/outgoing leg order and identical space lists, with a lazy permutation applied when needed."""
function _prepare_eigen_input(q::AbstractTLArray{T, 2, N, RD},
                              opname::AbstractString) where {T, N, RD}
    qinds = inds(q)
    dirs = (qinds[1].dir, qinds[2].dir)
    @assert (dirs == ('+', '-') || dirs == ('-', '+')) "$opname requires one incoming ('+') and one outgoing ('-') leg"

    q_work = dirs == ('+', '-') ? q : permutedims(q, (2, 1))
    qspaces = spaces(q_work)
    @assert qspaces[1] == qspaces[2] "$opname: both legs of input TLArray must have the same space list (same sectors and dimensions)"
    return q_work
end

"""Validate that the two legs of rank-two `q` carry matching tag, prime, lock, and dual metadata required by a Hermitian eigendecomposition. `opname` labels assertion messages."""
function _check_hermitian_eigen_legs(q::AbstractTLArray,
                                     opname::AbstractString)
    idx1, idx2 = inds(q)
    @assert idx1.itags == idx2.itags "$opname: Hermitian eigendecomposition requires both legs to have the same itag"
    @assert idx1.plev == idx2.plev "$opname: Hermitian eigendecomposition requires both legs to have the same plev"
    @assert idx1.lock == idx2.lock "$opname: Hermitian eigendecomposition requires both legs to have the same lock"
    @assert idx1.dual == idx2.dual "$opname: Hermitian eigendecomposition requires both legs to have the same dual flag"
    return nothing
end

"""Materialize one eigenproblem sector as a dense matrix. Applies deferred RMT view/scale state for stable `sector_index` and returns `(matrix, left_dim, right_dim)` for block eigensolvers."""
function _eig_scaled_matrix(q::AbstractTLArray{T, 2, N, RD}, sector_index::Int) where {T, N, RD}
    rmt, scale = sector_rmt_permuted(q, sector_index, _identity_rmt_perm(Val(RD)))
    rmt_size = sector_rmt_dim(q, sector_index)
    sL, sR = rmt_size[1], rmt_size[2]
    mat = Matrix(reshape(rmt, sL, sR))
    scale != one(typeof(scale)) && (mat .*= scale)
    return mat, sL, sR
end

function _hermiticity_ratio(q::AbstractTLArray{T, 2, N, RD}) where {T, N, RD}
    idx1, idx2 = inds(q)
    if !(idx1.itags == idx2.itags &&
         idx1.plev == idx2.plev &&
         idx1.lock == idx2.lock &&
         idx1.dual == idx2.dual)
        return Inf
    end

    qnorm2 = 0.0
    diffnorm2 = 0.0
    for sector_index in sector_slots(q)
        is_sector_zero(q, sector_index) && continue
        mat, sL, sR = _eig_scaled_matrix(q, sector_index)
        sL == sR || return Inf
        qnorm2 += sum(abs2, mat)
        diffnorm2 += sum(abs2, mat - adjoint(mat))
    end
    qnorm = sqrt(qnorm2)
    diffnorm = sqrt(diffnorm2)
    if iszero(qnorm)
        return iszero(diffnorm) ? 0.0 : Inf
    end
    return diffnorm / qnorm
end

function _select_eig_sectors(template::TLArray{T, 2, N, RD, QT, PS, M, RMT},
                          picks::Dict{NTuple{N, Tuple{Vararg{Int}}}, Vector{Int}};
                          mode::Symbol) where {T, N, RD, QT, PS, M, RMT}
    qlabels_out = QT[]
    source_wmat_sectors = Int[]
    RMTs_out = mode == :diag && RMT <: DiagRMT ? DiagRMT{T, RD}[] : Array{T, RD}[]
    sector_counts = Dict{NTuple{N, Tuple{Vararg{Int}}}, Int}()

    for (sector, idxs) in picks
        unique_count = length(unique(sort(idxs)))
        unique_count == 0 && continue
        sector_counts[sector] = unique_count
    end

    for sector_index in sector_slots(template)
        template.iszero[sector_index] && continue
        sector = _eig_sector_qlabels(template, sector_index)
        idxs = get(picks, sector, Int[])
        isempty(idxs) && continue

        idxs_sorted = sort(idxs)
        rmt, scale = sector_rmt(template, sector_index)
        s1, s2 = sector_rmt_axis_dim(template, sector_index, 1), sector_rmt_axis_dim(template, sector_index, 2)
        rmt_new = if mode == :diag
            if rmt isa DiagRMT
                DiagRMT(T[scale * rmt.diag[i] for i in idxs_sorted], Val(RD), (1, 2))
            else
                mat = Matrix(reshape(rmt, s1, s2))
                scale != one(typeof(scale)) && (mat .*= scale)
                reshape(mat[idxs_sorted, idxs_sorted], length(idxs_sorted), length(idxs_sorted), ones(Int, N)...)
            end
        elseif mode == :cols
            mat = Matrix(reshape(rmt, s1, s2))
            scale != one(typeof(scale)) && (mat .*= scale)
            reshape(mat[:, idxs_sorted], s1, length(idxs_sorted), ones(Int, N)...)
        elseif mode == :firstdim
            mat = Matrix(reshape(rmt, s1, s2))
            scale != one(typeof(scale)) && (mat .*= scale)
            reshape(mat[idxs_sorted, :], length(idxs_sorted), s2, ones(Int, N)...)
        else
            error("Unknown eig sector selection mode: $mode")
        end

        push!(qlabels_out, sector)
        push!(RMTs_out, rmt_new)
        push!(source_wmat_sectors, sector_index)
    end

    function build_selected_space(base_space, dim_fn)
        selected = eltype(base_space)[]
        for (sector, dim) in base_space
            haskey(sector_counts, sector) || continue
            push!(selected, (sector, dim_fn(dim, sector_counts[sector])))
        end
        return selected
    end

    spaces_out = if mode == :diag
        (
            build_selected_space(spaces(template)[1], (_, count) -> count),
            build_selected_space(spaces(template)[2], (_, count) -> count),
        )
    elseif mode == :cols
        (
            copy(spaces(template)[1]),
            build_selected_space(spaces(template)[2], (_, count) -> count),
        )
    elseif mode == :firstdim
        (
            build_selected_space(spaces(template)[1], (_, count) -> count),
            copy(spaces(template)[2]),
        )
    else
        error("Unknown eig sector selection mode: $mode")
    end

    qlabels_mat = Matrix{QT}(undef, 2, length(qlabels_out))
    for (sector_index, sector) in enumerate(qlabels_out)
        qlabels_mat[1, sector_index] = sector
        qlabels_mat[2, sector_index] = sector
    end
    wmatdata_out, wmatinfo_out = _copy_wmat_storage(template, source_wmat_sectors; deep=true)
    return TLArray(symm(template), qlabels_mat, wmatdata_out, wmatinfo_out, RMTs_out, inds(template), spaces_out)
end

"""
    _effective_eigen_keep_count(eig_entries, Nkeep, tol; keep_low=true)

Choose a retained count from the count-expansion window. Entries must already
be stably sorted in the selection direction. Use signed adjacent gaps, choosing
the earliest cut on ties; a window reaching the spectrum end keeps everything.
Callers validate the nonnegative count and finite, nonnegative `tol`.
"""
function _effective_eigen_keep_count(eig_entries,
                                     Nkeep::Integer,
                                     tol::Real;
                                     keep_low::Bool=true)
    total = length(eig_entries)
    nkeep_eff = Int(min(Nkeep, total))
    if iszero(tol) || nkeep_eff == 0 || nkeep_eff == total
        return nkeep_eff
    end

    # Clamp before converting to Int, even if a large finite tol overflows the product.
    candidate_end = ceil(Int, min(total, nkeep_eff * (1 + Float64(tol))))
    candidate_end <= nkeep_eff && return nkeep_eff
    candidate_end == total && return candidate_end

    best_cut = nkeep_eff
    gap = first(eig_entries[best_cut + 1]) - first(eig_entries[best_cut])
    best_gap = keep_low ? gap : -gap
    for i in nkeep_eff+1:candidate_end
        gap = first(eig_entries[i + 1]) - first(eig_entries[i])
        directed_gap = keep_low ? gap : -gap
        if directed_gap > best_gap
            best_gap = directed_gap
            best_cut = i
        end
    end
    return best_cut
end

"""
    _split_eigen_result(result; Nkeep=nothing, threshold=nothing,
                        keep_low=true, tol=0.1)

Partition real eigenvalues globally by a count window or an inclusive threshold.
Exactly one selection argument is required. Preserve ascending output lists and
renumber local indices to match the selected factor storage without mutating
`result`. Irrep degeneracy does not multiply the selection count.
"""
function _split_eigen_result(result::EigenResult;
                             Nkeep::Union{Nothing, Integer}=nothing,
                             threshold::Union{Nothing, Real}=nothing,
                             keep_low::Bool=true,
                             tol::Real=0.1)
    isnothing(Nkeep) != isnothing(threshold) ||
        throw(ArgumentError("supply exactly one of Nkeep and threshold"))
    isfinite(tol) && tol >= 0 ||
        throw(ArgumentError("tol must be finite and nonnegative"))
    if !isnothing(Nkeep)
        Nkeep >= 0 || throw(ArgumentError("Nkeep must be nonnegative"))
    else
        isfinite(threshold) || throw(ArgumentError("threshold must be finite"))
    end

    eig_entries = copy(result.eig_list)
    # Stable sorting preserves the source order of ties in either direction.
    sort!(eig_entries; by=first, rev=!keep_low, alg=Base.Sort.MergeSort)
    nkeep_eff = if isnothing(threshold)
        _effective_eigen_keep_count(eig_entries, Nkeep, tol; keep_low=keep_low)
    else
        count(entry -> keep_low ? first(entry) <= threshold : first(entry) >= threshold,
              eig_entries)
    end
    kept_entries = eig_entries[1:nkeep_eff]
    discarded_entries = eig_entries[nkeep_eff+1:end]
    if !keep_low
        sort!(kept_entries; by=first, alg=Base.Sort.MergeSort)
        sort!(discarded_entries; by=first, alg=Base.Sort.MergeSort)
    end

    N = length(symm(result.D))
    kept_picks = Dict{NTuple{N, Tuple{Vararg{Int}}}, Vector{Int}}()
    for entry in kept_entries
        push!(get!(kept_picks, entry[3], Int[]), entry[4])
    end

    discarded_picks = Dict{NTuple{N, Tuple{Vararg{Int}}}, Vector{Int}}()
    for entry in discarded_entries
        push!(get!(discarded_picks, entry[3], Int[]), entry[4])
    end

    Vkeep = _select_eig_sectors(result.V, kept_picks; mode=:cols)
    Vdiscard = _select_eig_sectors(result.V, discarded_picks; mode=:cols)
    Dkeep = _select_eig_sectors(result.D, kept_picks; mode=:diag)
    Ddiscard = _select_eig_sectors(result.D, discarded_picks; mode=:diag)

    Vinv_keep = isnothing(result.V_inv) ? nothing : _select_eig_sectors(result.V_inv, kept_picks; mode=:firstdim)
    Vinv_discard = isnothing(result.V_inv) ? nothing : _select_eig_sectors(result.V_inv, discarded_picks; mode=:firstdim)

    kept = EigenResult(Vkeep, Dkeep, Vinv_keep, _renumber_eig_entries(kept_entries))
    discarded = EigenResult(Vdiscard, Ddiscard, Vinv_discard, _renumber_eig_entries(discarded_entries))
    return kept, discarded
end

function _eigen_hermitian(q::AbstractTLArray{T, 2, N, RD, QT},
                          eig_tag::AbstractString = "eig") where {T, N, RD, QT}

    _check_hermitian_eigen_legs(q, "eigen")

    symmetries = symm(q)
    qinds = inds(q)
    qspaces = spaces(q)
    dirs = (qinds[1].dir, qinds[2].dir)
    out_leg = 2
    cgp = (1, 2)

    # ── Decompose each sector ───────────────────────────────────────────────────
    T_out    = promote_type(T, Float64)
    qlabels_D = QT[]
    qlabels_V = QT[]
    RMTs_D = DiagRMT{T_out, 2 + N}[]
    RMTs_V = Array{T_out, 2 + N}[]
    # Eigenvalue, degeneracy, sector qlabels, in-sector index
    eig_list = Tuple{T_out, Int, NTuple{N, Tuple{Vararg{Int}}}, Int}[]

    for sector_index in sector_slots(q)
        is_sector_zero(q, sector_index) && continue
        sector_qlabels = _eig_sector_qlabels(q, sector_index)

        # Degeneracy = product of irrep dims across all symmetries for this sector
        cgt_dim = prod(
            dimension(symmetries[n], sector_qlabels[n])
            for n in 1:N)
        cgt_scale = sqrt(Float64(cgt_dim))

        rmt_mat, sL, sR = _eig_scaled_matrix(q, sector_index)
        @assert sL == sR "eigen: RMT must be square for eigendecomposition, got ($sL, $sR)"

        mat = rmt_mat ./ cgt_scale

        F = eigen(Hermitian(mat))
        eigenvalues  = T_out.(F.values)
        eigenvectors = T_out.(F.vectors)

        chi = length(eigenvalues)
        for (j, ev) in enumerate(eigenvalues)
            push!(eig_list, (ev, cgt_dim, sector_qlabels, j))
        end

        rmt_D = _eig_diagonal_rmt(RMTs_D, eigenvalues, cgt_scale, Val(2 + N))
        rmt_V = reshape(eigenvectors, sL, chi, ones(Int, N)...)
        rmt_V[:] .*= cgt_scale

        push!(qlabels_D, sector_qlabels)
        push!(qlabels_V, sector_qlabels)
        push!(RMTs_D, rmt_D)
        push!(RMTs_V, rmt_V)
    end

    covered = Set(_eig_sector_qlabels(q, sector_index) for sector_index in sector_slots(q) if !is_sector_zero(q, sector_index))
    _append_missing_eig_sectors!(symmetries, qspaces[1], covered,
        qlabels_D, RMTs_D,
        qlabels_V, RMTs_V,
        nothing, nothing,
        eig_list, cgp, T_out)

    # Sort eig_list ascending by eigenvalue.
    sort!(eig_list; by = x -> _eig_sort_value(x[1], true))

    # ── Inherit spaces from original q ──────────────────────────────────────
    spaces_D = (qspaces[1], qspaces[1])
    spaces_V = (qspaces[1], qspaces[1])

    # ── Build output TLIndex tuples ───────────────────────────────────────────
    inds_D = (TLIndex(eig_tag, dirs[1]), TLIndex(eig_tag, dirs[2]))

    orig_out_ind = qinds[out_leg]
    inds_V = (TLIndex(orig_out_ind.itags, dirs[1], orig_out_ind.plev, orig_out_ind.lock, orig_out_ind.dual),
              TLIndex(eig_tag, dirs[2]))

    wmatdata_D, wmatinfo_D = _unit_wmat_storage(productsymm(q), length(RMTs_D))
    wmatdata_V, wmatinfo_V = _unit_wmat_storage(productsymm(q), length(RMTs_V))
    D = TLArray(symmetries, _eig_qlabel_matrix(qlabels_D),
                wmatdata_D, wmatinfo_D, RMTs_D, inds_D, spaces_D)
    V = TLArray(symmetries, _eig_qlabel_matrix(qlabels_V),
                wmatdata_V, wmatinfo_V, RMTs_V, inds_V, spaces_V)

    return EigenResult(V, D, nothing, eig_list)
end

function _eigen_general(q::AbstractTLArray{T, 2, N, RD, QT},
                        eig_tag::AbstractString = "eig") where {T, N, RD, QT}

    symmetries = symm(q)
    qinds = inds(q)
    qspaces = spaces(q)
    dirs = (qinds[1].dir, qinds[2].dir)
    cgp = (1, 2)

    in_leg  = 1
    out_leg = 2

    T_out     = promote_type(T, ComplexF64)
    qlabels_D = QT[]
    qlabels_V = QT[]
    qlabels_Vinv = QT[]
    RMTs_D = DiagRMT{T_out, 2 + N}[]
    RMTs_V = Array{T_out, 2 + N}[]
    RMTs_Vinv = Array{T_out, 2 + N}[]
    eig_list  = Tuple{T_out, Int, NTuple{N, Tuple{Vararg{Int}}}, Int}[]

    for sector_index in sector_slots(q)
        is_sector_zero(q, sector_index) && continue
        sector_qlabels = _eig_sector_qlabels(q, sector_index)

        cgt_dim = prod(
            dimension(symmetries[n], sector_qlabels[n])
            for n in 1:N)
        cgt_scale = sqrt(Float64(cgt_dim))

        rmt_mat, sL, sR = _eig_scaled_matrix(q, sector_index)
        @assert sL == sR "eigen_general: RMT must be square"

        mat = rmt_mat ./ cgt_scale

        F = eigen(mat)
        eigenvalues      = T_out.(F.values)
        eigenvectors     = T_out.(F.vectors)
        eigenvectors_inv = T_out.(inv(F.vectors))

        chi = length(eigenvalues)
        for (j, ev) in enumerate(eigenvalues)
            push!(eig_list, (ev, cgt_dim, sector_qlabels, j))
        end

        rmt_D    = _eig_diagonal_rmt(RMTs_D, eigenvalues, cgt_scale, Val(2 + N))
        rmt_V    = reshape(eigenvectors, sL, chi, ones(Int, N)...)
        rmt_Vinv = reshape(eigenvectors_inv, chi, sL, ones(Int, N)...)
        rmt_V[:] .*= cgt_scale
        rmt_Vinv[:] .*= cgt_scale

        push!(qlabels_D, sector_qlabels)
        push!(qlabels_V, sector_qlabels)
        push!(qlabels_Vinv, sector_qlabels)
        push!(RMTs_D, rmt_D)
        push!(RMTs_V, rmt_V)
        push!(RMTs_Vinv, rmt_Vinv)
    end

    covered = Set(_eig_sector_qlabels(q, sector_index) for sector_index in sector_slots(q) if !is_sector_zero(q, sector_index))
    _append_missing_eig_sectors!(symmetries, qspaces[1], covered,
        qlabels_D, RMTs_D,
        qlabels_V, RMTs_V,
        qlabels_Vinv, RMTs_Vinv,
        eig_list, cgp, T_out)

    sort!(eig_list; by = x -> _eig_sort_value(x[1], false))

    spaces_D    = (qspaces[1], qspaces[1])
    spaces_V    = (qspaces[1], qspaces[1])
    spaces_Vinv = (qspaces[1], qspaces[1])

    inds_D = (TLIndex(eig_tag, dirs[1]), TLIndex(eig_tag, dirs[2]))

    orig_out_ind = qinds[out_leg]
    inds_V = (TLIndex(orig_out_ind.itags, dirs[1], orig_out_ind.plev, orig_out_ind.lock, orig_out_ind.dual),
              TLIndex(eig_tag, dirs[2]))

    orig_in_ind = qinds[in_leg]
    inds_Vinv = (TLIndex(eig_tag, dirs[1]),
                 TLIndex(orig_in_ind.itags, dirs[2], orig_in_ind.plev, orig_in_ind.lock, orig_in_ind.dual))

    wmatdata_D, wmatinfo_D = _unit_wmat_storage(productsymm(q), length(RMTs_D))
    wmatdata_V, wmatinfo_V = _unit_wmat_storage(productsymm(q), length(RMTs_V))
    wmatdata_Vinv, wmatinfo_Vinv = _unit_wmat_storage(productsymm(q), length(RMTs_Vinv))
    D    = TLArray(symmetries, _eig_qlabel_matrix(qlabels_D),
                   wmatdata_D, wmatinfo_D, RMTs_D, inds_D, spaces_D)
    V    = TLArray(symmetries, _eig_qlabel_matrix(qlabels_V),
                   wmatdata_V, wmatinfo_V, RMTs_V, inds_V, spaces_V)
    Vinv = TLArray(symmetries, _eig_qlabel_matrix(qlabels_Vinv),
                   wmatdata_Vinv, wmatinfo_Vinv, RMTs_Vinv, inds_Vinv, spaces_Vinv)

    return EigenResult(V, D, Vinv, eig_list)
end

function _eigen_unsymmetric(q::AbstractTLArray{T, 2, 0, RD},
                            eig_tag::AbstractString = "eig";
                            hermitian::Bool=false) where {T, RD}
    qinds = inds(q)
    dirs = (qinds[1].dir, qinds[2].dir)
    sL, sR = size(q)
    sL == sR || throw(ArgumentError("eigen: RMT must be square for eigendecomposition, got ($sL, $sR)"))
    # Unsymmetric TLArrays have no sector decomposition. In particular, a
    # zero tensor can have no evaluated RMT at all, so read its logical dense
    # value (which supplies zeros for absent sectors) instead of requesting
    # sector 1 directly.
    mat = Matrix(reshape(_unsymmetric_dense_data(q), sL, sR))

    F = hermitian ? eigen(Hermitian(mat)) : eigen(mat)
    vals = F.values
    vectors = F.vectors
    vectors_inv = hermitian ? nothing : inv(vectors)
    T_out = eltype(vals)
    eig_list = Tuple{T_out, Int, Tuple{}, Int}[(vals[j], 1, (), j) for j in eachindex(vals)]
    sort!(eig_list; by = x -> _eig_sort_value(x[1], hermitian))

    orig_out = qinds[2]
    orig_in = qinds[1]
    inds_D = (TLIndex(eig_tag, dirs[1]), TLIndex(eig_tag, dirs[2]))
    inds_V = (TLIndex(orig_out.itags, dirs[1], orig_out.plev, orig_out.lock, orig_out.dual),
              TLIndex(eig_tag, dirs[2]))
    D = TLArray(Matrix(Diagonal(vals)), inds_D)
    V = TLArray(vectors, inds_V)
    Vinv = if hermitian
        nothing
    else
        inds_Vinv = (TLIndex(eig_tag, dirs[1]),
                     TLIndex(orig_in.itags, dirs[2], orig_in.plev, orig_in.lock, orig_in.dual))
        TLArray(vectors_inv, inds_Vinv)
    end
    return EigenResult(V, D, Vinv, eig_list)
end

function LinearAlgebra.eigen(q::AbstractTLArray{T, 2, N, RD},
                             eig_tag::AbstractString = "eig";
                             hermitian::Bool = false) where {T, N, RD}
    q_work = _prepare_eigen_input(q, "eigen")
    materialize(q_work)

    use_hermitian = hermitian
    if !use_hermitian
        use_hermitian = _hermiticity_ratio(q_work) <= _EIG_HERMITIAN_RTOL
    end

    N == 0 && return _eigen_unsymmetric(q_work, eig_tag; hermitian=use_hermitian)

    return use_hermitian ? _eigen_hermitian(q_work, eig_tag) : _eigen_general(q_work, eig_tag)
end

"""
    discard_eigen(result::EigenResult, kept_tag="eigK", discarded_tag="eigD";
                  Nkeep=nothing, threshold=nothing, keep_low=true, tol=0.1)

Return `(kept, discarded)` eigen-subspaces without mutating `result`. Supply
exactly one of `Nkeep` and `threshold`. Compare real eigenvalues by signed value:
`keep_low=true` selects the lowest values, and `false` selects the highest.
Each eigenvalue entry counts once regardless of its irrep degeneracy.

With a nonnegative integer `Nkeep`, retain that many entries (or all available).
`tol=0` uses the exact count; positive `tol` chooses the largest adjacent gap
between cuts at `Nkeep` and `ceil(Nkeep * (1 + tol))`. Equal gaps choose the
smaller count. A window reaching the spectrum end keeps everything. `Nkeep=0`
keeps nothing; eigenvalues equal to zero are counted normally.

With a finite real `threshold`, keep values `<= threshold` when `keep_low=true`
or `>= threshold` otherwise, using direct comparisons. `tol` does not adjust an
explicit threshold, but must be finite and nonnegative in either mode. Invalid
selection combinations, negative counts, or invalid bounds raise `ArgumentError`.

Both results have ascending eigenvalue lists and matching renumbered factor
indices. `kept_tag` and `discarded_tag` label their eigenvector bonds. Preserve
`V_inv === nothing` when absent in the input. Complex spectra are outside this
interface's current scope.
"""
function discard_eigen(result::EigenResult,
                       kept_tag::AbstractString = "eigK",
                       discarded_tag::AbstractString = "eigD";
                       Nkeep::Union{Nothing, Integer}=nothing,
                       threshold::Union{Nothing, Real}=nothing,
                       keep_low::Bool=true,
                       tol::Real=0.1)
    kept, discarded = _split_eigen_result(result; Nkeep=Nkeep, threshold=threshold,
                                         keep_low=keep_low, tol=tol)
    return _retag_eigen_result(kept, kept_tag), _retag_eigen_result(discarded, discarded_tag)
end
