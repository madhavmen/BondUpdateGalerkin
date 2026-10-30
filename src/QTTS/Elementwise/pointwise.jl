# Pointwise linear operations on QTT fields: scaling, conjugation, sums.
#
# Moved from BUG-Julia@b155550:src/RSVDCBEBondUpdate/qtt/qtt_cores.jl. `tt_density` follows with the
# zip-up products (`products.jl`) it is built on.

"`c A` (the scalar on site 1)."
tt_scale(A::AbstractVector, c::Number) = [k == 1 ? c .* A[1] : copy(A[k]) for k in eachindex(A)]

tt_conj(A::AbstractVector) = [conj.(a) for a in A]

"""
    tt_add(A, B) -> TT

`A + B` by direct sum of the bonds (rank adds; compress afterwards).
"""
function tt_add(A::AbstractVector, B::AbstractVector)
    L = length(A)
    length(B) == L || throw(DimensionMismatch("tt_add: $L sites against $(length(B))"))
    T = promote_type(eltype(A[1]), eltype(B[1]))
    L == 1 && return [T.(A[1]) .+ B[1]]
    C = Vector{Array{T,3}}(undef, L)
    for k = 1:L
        a, b = A[k], B[k]
        size(a, 2) == size(b, 2) || throw(DimensionMismatch("tt_add: site $k dimensions differ"))
        C[k] = if k == 1
            cat(T.(a), T.(b); dims = 3)
        elseif k == L
            cat(T.(a), T.(b); dims = 1)
        else
            c = zeros(T, size(a, 1) + size(b, 1), size(a, 2), size(a, 3) + size(b, 3))
            c[1:size(a, 1), :, 1:size(a, 3)] .= a
            c[(size(a, 1)+1):end, :, (size(a, 3)+1):end] .= b
            c
        end
    end
    return C
end

"`sum_t c_t A_t`, compressed once at `tol`."
function tt_lincomb(terms::AbstractVector; tol::Real = 1e-14, maxdim::Int = typemax(Int))
    acc = nothing
    for (c, A) in terms
        X = tt_scale(A, c)
        acc = acc === nothing ? X : tt_add(acc, X)
    end
    return first(tt_compress(acc; tol = tol, maxdim = maxdim))
end
