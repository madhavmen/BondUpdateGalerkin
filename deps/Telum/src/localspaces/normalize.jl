"""
    _check_localspace_shapes(symm, weights, lops, ops) -> Int

Validate counts, dimensions and finite entries in the original local basis.
Accept legacy integer weights or vectors of Cartan matrices; do not mutate input.
"""
function _check_localspace_shapes(symm, weights, lops, ops)
    length(symm) == length(weights) == length(lops) ||
        throw(ArgumentError("Symmetry, weight and lowering tuples must have equal lengths"))
    isempty(ops) && throw(ArgumentError("Local space has no operators"))
    d = size(first(values(ops)), 1)
    d > 0 || throw(ArgumentError("Local space must be nonempty"))
    for (name, op) in ops
        _check_local_matrix(op, d, "operator $name")
    end
    for i in eachindex(symm)
        w = weights[i]
        if w isa AbstractVector{<:Tuple}
            length(w) == d || throw(ArgumentError("weight[$i] has inconsistent state count"))
            all(v -> length(v) == nzops(symm[i]) && all(x -> x isa Int, v), w) ||
                throw(ArgumentError("weight[$i] must contain integer tuples of length $(nzops(symm[i]))"))
        elseif w isa AbstractVector{<:AbstractMatrix}
            length(w) == nzops(symm[i]) || throw(ArgumentError("weight[$i] has incorrect matrix count"))
            for (j, z) in enumerate(w)
                _check_local_matrix(z, d, "weight[$i][$j]")
            end
        else
            throw(ArgumentError("weight[$i] must be integer tuples or a vector of matrices"))
        end
        expected = isabelian(symm[i]) ? 0 : LurCGT.nlops(symm[i])
        length(lops[i]) == expected || throw(ArgumentError("symmetry $i has incorrect lowering-operator count"))
        for (j, op) in enumerate(lops[i])
            _check_local_matrix(op, d, "lowering[$i][$j]")
        end
    end
    return d
end

"""
    _check_local_matrix(A, d, label)

Require a finite square numerical matrix on a local space of dimension `d`.
"""
function _check_local_matrix(A, d, label)
    A isa AbstractMatrix{<:Number} && size(A) == (d, d) ||
        throw(ArgumentError("$label must be a $d × $d numerical matrix"))
    all(isfinite, A) || throw(ArgumentError("$label contains nonfinite entries"))
    return nothing
end

"""
    _local_zmatrices(T, symmetry, entry) -> Vector{Matrix{T}}

Copy one symmetry's Cartan matrices, expanding legacy diagonal weights only
on the matrix-input path. Component order follows the weight tuples.
"""
function _local_zmatrices(::Type{T}, s, w::AbstractVector{<:Tuple}) where T
    return Matrix{T}[Matrix{T}(Diagonal([v[j] for v in w])) for j in 1:nzops(s)]
end
function _local_zmatrices(::Type{T}, s, w::AbstractVector{<:AbstractMatrix}) where T
    return Matrix{T}[Matrix{T}(z) for z in w]
end

"""
    _check_local_commutation(zs, lops)

Check all Cartan pairs and cross-factor z/lowering, lowering/lowering and
lowering/raising commutators with absolute Frobenius tolerance 1e-12.
Own-factor lowering relations are deliberately excluded.
"""
function _check_local_commutation(zs, lops)
    for i in eachindex(zs), j in i:length(zs)
        for (a, A) in enumerate(zs[i]), (b, B) in enumerate(zs[j])
            norm(comm(A, B)) <= 1e-12 || throw(ArgumentError("weight[$i][$a] and weight[$j][$b] do not commute"))
        end
        i == j && continue
        for (a, A) in enumerate(zs[i]), (b, B) in enumerate(lops[j])
            norm(comm(A, B)) <= 1e-12 || throw(ArgumentError("weight[$i][$a] and lowering[$j][$b] do not commute"))
        end
        for (a, A) in enumerate(lops[i]), (b, B) in enumerate(zs[j])
            norm(comm(A, B)) <= 1e-12 || throw(ArgumentError("lowering[$i][$a] and weight[$j][$b] do not commute"))
        end
        for (a, A) in enumerate(lops[i]), (b, B) in enumerate(lops[j])
            norm(comm(A, B)) <= 1e-12 && norm(comm(A, B')) <= 1e-12 ||
                throw(ArgumentError("lowering[$i][$a] and lowering/raising[$j][$b] do not commute"))
        end
    end
    return nothing
end

"""
    _split_local_eigenvalues(values, indices; preserve_order=false)

Partition indices into clusters whose full eigenvalue spread is at most
1e-12. Preserve original index order for initial diagonal partitions.
"""
function _split_local_eigenvalues(values, indices; preserve_order=false)
    order = sortperm(values; alg=MergeSort)
    groups = Vector{Int}[]
    start = 1
    while start <= length(order)
        stop = start
        while stop < length(order) && values[order[stop+1]] - values[order[start]] <= 1e-12
            stop += 1
        end
        group = indices[order[start:stop]]
        preserve_order && sort!(group)
        push!(groups, group)
        start = stop + 1
    end
    preserve_order && sort!(groups; by=first)
    return groups
end

"""
    _local_integer_weight(value, label) -> Int

Check a finite, real, representable weight is strictly within 1e-12 of an
integer before conversion. The tolerance never scales with weight magnitude.
"""
function _local_integer_weight(value, label)
    x = real(value)
    isfinite(value) && abs(imag(value)) < 1e-12 && abs(x - round(x)) < 1e-12 &&
        typemin(Int) <= x < -float(typemin(Int)) ||
        throw(ArgumentError("$label is not a representable integer weight: $value"))
    return round(Int, x)
end

"""
    _normalize_localspace_data(symm, weights, lops, ops)

Return `(weights, lowering_ops, mwirops, U)` in a common Cartan eigenbasis.
`U` maps this basis into the caller's basis and is `nothing` when no rotation
is needed. Legacy input retains its arrays; matrix input is copied. Validate
before diagonalization and after integer recovery without mutating the hook.
"""
function _normalize_localspace_data(symm, weights, lops, ops)
    d = _check_localspace_shapes(symm, weights, lops, ops)
    all(w -> w isa AbstractVector{<:Tuple}, weights) && return weights, lops, ops, nothing
    complex = any(op -> !(eltype(op) <: Real), values(ops)) ||
        any(ls -> any(op -> !(eltype(op) <: Real), ls), lops) ||
        any(w -> w isa AbstractVector{<:AbstractMatrix} && any(z -> !(eltype(z) <: Real), w), weights)
    return _normalize_localspace_matrices(complex ? ComplexF64 : Float64, symm, weights, lops, ops, d)
end

"""
    _normalize_localspace_matrices(T, symm, weights, lops, ops, d)

Normalize matrix input using concrete payload type `T`. Refine degenerate
joint eigenspaces successively; rotate all physical operators only once.
"""
function _normalize_localspace_matrices(::Type{T}, symm::NTuple{N,Any}, weights, lops, ops, d) where {T,N}
    zs = ntuple(i -> _local_zmatrices(T, symm[i], weights[i]), N)
    for i in 1:N, (j, Z) in enumerate(zs[i])
        norm(Z - Z') <= 1e-12 || throw(ArgumentError("weight[$i][$j] is not Hermitian"))
    end
    _check_local_commutation(zs, lops)
    diagonal = all(group -> all(isdiag, group), zs)
    U = diagonal ? nothing : Matrix{T}(I, d, d)
    if !diagonal
        blocks = [collect(1:d)]
        # Already-diagonal generators constrain every subsequent rotation.
        for group in zs, Z in group
            isdiag(Z) || continue
            blocks = reduce(vcat, [_split_local_eigenvalues(real.(diag(Z)[b]), b; preserve_order=true) for b in blocks])
        end
        for group in zs, Z in group
            isdiag(Z) && continue
            refined = Vector{Int}[]
            for b in blocks
                Ub = U[:, b]
                E = eigen(Hermitian(Ub' * Z * Ub))
                U[:, b] = Ub * E.vectors
                append!(refined, _split_local_eigenvalues(E.values, b))
            end
            blocks = refined
        end
        # Fix column phases by their largest absolute entry; unresolved
        # degeneracies remain free, but all operators share this same basis.
        for j in axes(U,2)
            col = view(U,:,j)
            pivot = col[argmax(abs.(col))]
            col .*= conj(pivot) / abs(pivot)
        end
        norm(U' * U - I) <= 1e-12 || throw(ArgumentError("Common Cartan basis is not unitary"))
    end
    rotated_zs = ntuple(i -> [U === nothing ? Z : U' * Z * U for Z in zs[i]], N)
    for i in 1:N, (j, Z) in enumerate(rotated_zs[i])
        norm(Z - Diagonal(diag(Z))) <= 1e-12 || throw(ArgumentError("weight[$i][$j] failed simultaneous diagonalization"))
    end
    normalized = ntuple(N) do i
        [_local_weight_tuple(symm[i], rotated_zs[i], k, i) for k in 1:d]
    end
    integer_zs = ntuple(i -> _local_zmatrices(T, symm[i], normalized[i])::Vector{Matrix{T}}, N)
    for i in 1:N, j in eachindex(zs[i])
        norm(rotated_zs[i][j] - integer_zs[i][j]) <= 1e-12 ||
            throw(ArgumentError("weight[$i][$j] failed integer reconstruction"))
    end
    new_lops = ntuple(i -> SparseMatrixCSC{T,Int}[_transform_local_operator(T, op, U) for op in lops[i]], N)
    new_ops = Dict{Symbol,SparseMatrixCSC{T,Int}}(name => _transform_local_operator(T, op, U) for (name,op) in ops)
    _check_local_commutation(integer_zs, new_lops)
    return normalized, new_lops, new_ops, U
end

"""
    _local_weight_tuple(symmetry, zs, state, factor)

Recover one state's integer tuple in the symmetry's fixed Cartan order.
"""
function _local_weight_tuple(::Type{S}, zs, k, i) where S
    return ntuple(j -> _local_integer_weight(zs[j][k,k], "weight[$i][$j] state $k"), Val(nzops(S)))
end

"""
    _transform_local_operator(T, op, U)

Copy an operator to concrete sparse storage, optionally rotating by `U`.
Discard sub-1e-12 roundoff entries so weight-block extraction sees exact zeros.
"""
function _transform_local_operator(::Type{T}, op, U) where T
    result = sparse(T.(U === nothing ? op : U' * op * U))
    U === nothing || droptol!(result, 1e-12)
    return result
end
