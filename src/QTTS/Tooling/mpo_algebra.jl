# Dense operator-core algebra: identity, sums, compression, adjoint, products and the exact
# operator-state product. An operator core is `(w_l, out, in, w_r)`, `(O psi)[out] = sum_in
# O[out, in] psi[in]`.
#
# Moved from BUG-Julia@b155550:src/RSVDCBEBondUpdate/qtt/qtt_cores.jl. `op_zipup` follows with the
# zip-up products (`Elementwise/products.jl`) it is built on.

"The `L`-site identity, bond dimension 1."
op_identity(L::Int; T::Type = Float64) = [reshape(Matrix{T}(I, 2, 2), 1, 2, 2, 1) for _ = 1:L]

op_scale(W::AbstractVector, c::Number) = [k == 1 ? c .* W[1] : copy(W[k]) for k in eachindex(W)]

"`A + B` (bond dimensions add)."
op_add(A::AbstractVector, B::AbstractVector) = [_as4(c) for c in tt_add(_as3.(A), _as3.(B))]

"`sum_t c_t W_t`, compressed once."
function op_lincomb(terms::AbstractVector; tol::Real = 1e-13, maxdim::Int = typemax(Int))
    acc = nothing
    for (c, W) in terms
        X = op_scale(W, c)
        acc = acc === nothing ? X : op_add(acc, X)
    end
    return op_compress(acc; tol = tol, maxdim = maxdim)
end

"SVD rounding of an operator chain (its cores read as a TT with `d = 4`)."
op_compress(W::AbstractVector; tol::Real = 1e-13, maxdim::Int = typemax(Int)) =
    [_as4(c) for c in first(tt_compress(_as3.(W); tol = tol, maxdim = maxdim))]

"The adjoint `O'` (conjugate transpose)."
op_adjoint(W::AbstractVector) = [conj.(permutedims(w, (1, 3, 2, 4))) for w in W]

"The transpose `O^T`."
op_transpose(W::AbstractVector) = [permutedims(w, (1, 3, 2, 4)) for w in W]

"""
    op_compose(A, B) -> operator

The product `A * B` (B acts first): bonds multiply, A's index fastest.
"""
function op_compose(A::AbstractVector, B::AbstractVector)
    T = promote_type(eltype(A[1]), eltype(B[1]))
    C = Vector{Array{T,4}}(undef, length(A))
    for k in eachindex(A)
        a, b = A[k], B[k]
        c = zeros(T, size(a, 1) * size(b, 1), 2, 2, size(a, 4) * size(b, 4))
        for o = 1:2, i = 1:2, m = 1:2
            c[:, o, i, :] .+= kron(b[:, m, i, :], a[:, o, m, :])
        end
        C[k] = c
    end
    return C
end

"`diag(f)` of a TT `f`: `W[a, o, i, b] = delta_{oi} f[a, o, b]`."
function op_diag(F::AbstractVector)
    W = Vector{Array{eltype(F[1]),4}}(undef, length(F))
    for k in eachindex(F)
        f = F[k]
        w = zeros(eltype(f), size(f, 1), 2, 2, size(f, 3))
        for s = 1:2
            w[:, s, s, :] .= f[:, s, :]
        end
        W[k] = w
    end
    return W
end

"""
    op_apply(W, A) -> TT

`W * A` exactly (bond `w chi`, W's index fastest); compress afterwards.
"""
function op_apply(W::AbstractVector, A::AbstractVector)
    T = promote_type(eltype(W[1]), eltype(A[1]))
    C = Vector{Array{T,3}}(undef, length(A))
    for k in eachindex(A)
        w, a = W[k], A[k]
        c = zeros(T, size(w, 1) * size(a, 1), 2, size(w, 4) * size(a, 3))
        for o = 1:2, i = 1:2
            c[:, o, :] .+= kron(a[:, i, :], w[:, o, i, :])
        end
        C[k] = c
    end
    return C
end

"""
    op_dense(W) -> Matrix

The `2^L x 2^L` matrix, rows and columns in the order of `tt_dense` (site 1 fastest). Small sizes
only -- tests.
"""
function op_dense(W::AbstractVector)
    size(W[1], 1) == 1 || throw(ArgumentError("op_dense: left boundary dimension $(size(W[1], 1))"))
    M = reshape(W[1], 2, 2, size(W[1], 4))
    P = 2
    for k = 2:length(W)
        wl, _, _, wr = size(W[k])
        X = reshape(reshape(M, P * P, wl) * reshape(W[k], wl, 4 * wr), P, P, 2, 2, wr)
        M = reshape(permutedims(X, (1, 3, 2, 4, 5)), 2P, 2P, wr)
        P *= 2
    end
    size(M, 3) == 1 || throw(ArgumentError("op_dense: right boundary dimension $(size(M, 3))"))
    return reshape(M, P, P)
end

"The rank-1 operator `|r><c|` on `n` bits (logical order), `r, c` 0-based."
function op_ket_bra(n::Int, r::Integer, c::Integer; T::Type = Float64)
    return [
        begin
            w = zeros(T, 1, 2, 2, 1)
            w[1, ((r>>k)&1)+1, ((c>>k)&1)+1, 1] = one(T)
            w
        end for k = 0:(n-1)
    ]
end
