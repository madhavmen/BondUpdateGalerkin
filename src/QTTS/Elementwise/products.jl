# Elementwise products and the compressed operator application of QTT functions.
#
# Moved from BUG-Julia@b155550:src/RSVDCBEBondUpdate/qtt/qtt_cores.jl (`tt_hadamard`, `tt_density`,
# `tt_product`, `op_zipup` and their zip-up kernels).
#
# A zip-up contracts and truncates site by site, left to right, over a right-canonical copy of the
# input, so the exact `chi_A chi_B` (or `w chi`) cores are never formed. Each bond's discarded
# singular values add to an error bound; the build is repeated once at a tighter cut if the bound
# misses `tol`, then recompressed.

"""
    tt_hadamard(A, B) -> TT

The elementwise product `A .* B`, exact: rank `chi_A chi_B` (compress afterwards). Small ranks only;
[`tt_product`](@ref) is the compressed product.
"""
function tt_hadamard(A::AbstractVector, B::AbstractVector)
    T = promote_type(eltype(A[1]), eltype(B[1]))
    C = Vector{Array{T,3}}(undef, length(A))
    for k in eachindex(A)
        a, b = A[k], B[k]
        c = zeros(T, size(a, 1) * size(b, 1), size(a, 2), size(a, 3) * size(b, 3))
        for s = 1:size(a, 2)
            c[:, s, :] .= kron(b[:, s, :], a[:, s, :])        # A's bond index fastest
        end
        C[k] = c
    end
    return C
end

"""
    tt_density(A; tol = 1e-12, maxdim) -> (rho, info)

The density `|A|^2 = A .* conj(A)` as a compressed TT, by a zip-up (file header). The relative
error bound is checked against `tol` and the build repeated once at a tighter cut if it misses.
`info = (chi, err_bound, seconds, retried)`; the build is a dominant `O(D^4)` cost of a nonlinear
step (Chen et al., arXiv:2507.04279v5 Sec. 3.1), so its time is returned for the step ledger.
"""
function tt_density(A0::AbstractVector; tol::Real = 1e-12, maxdim::Int = typemax(Int))
    start = time_ns()
    A, _ = tt_compress(A0; tol = 0.0)
    B = tt_conj(A)
    return _zip_rc(cut -> _product_zipup(A, B, cut, maxdim), tol, maxdim, start)
end

"""
    tt_product(A, B; tol = 1e-12, maxdim) -> (C, info)

`A .* B` compressed, by the zip-up of [`tt_density`](@ref) (same error bound, retry and `info`),
never the exact cores of [`tt_hadamard`](@ref).
"""
function tt_product(
    A0::AbstractVector,
    B0::AbstractVector;
    tol::Real = 1e-12,
    maxdim::Int = typemax(Int),
)
    start = time_ns()
    A, _ = tt_compress(A0; tol = 0.0)
    B = B0 === A0 ? A : first(tt_compress(B0; tol = 0.0))
    return _zip_rc(cut -> _product_zipup(A, B, cut, maxdim), tol, maxdim, start)
end

"""
    op_zipup(W, A; tol = 1e-12, maxdim) -> (C, info)

`W * A` compressed by a zip-up, with the retry and `info` of [`tt_product`](@ref): never the exact
`w chi` cores of [`op_apply`](@ref). The per-bond tails bound the error only when `W`'s right parts
are contractions, so for a general operator the bound is an estimate.
"""
function op_zipup(
    W::AbstractVector,
    A0::AbstractVector;
    tol::Real = 1e-12,
    maxdim::Int = typemax(Int),
)
    start = time_ns()
    A, _ = tt_compress(A0; tol = 0.0)
    return _zip_rc(cut -> _apply_zipup(W, A, cut, maxdim), tol, maxdim, start)
end

"A zip-up `zip(cut) -> (C, bound)`, retried once at a tighter cut, then recompressed."
function _zip_rc(zip, tol::Real, maxdim::Int, start)
    C, bound = zip(tol)
    retried = false
    if bound > tol && tol > 0
        C, bound = zip(0.25 * tol * tol / bound)
        retried = true
    end
    C, recompression = tt_compress(C; tol = 0.1 * tol, maxdim = maxdim)
    return C,
    (
        chi = maximum(tt_ranks(C); init = 1),
        err_bound = bound + recompression,
        seconds = (time_ns() - start) / 1e9,
        retried = retried,
    )
end

"Left zip-up of `A .* B`; returns the cores and the relative error bound."
function _product_zipup(A::AbstractVector, B::AbstractVector, tol::Real, maxdim::Int)
    L = length(A)
    T = promote_type(eltype(A[1]), eltype(B[1]))
    carry = ones(T, 1, 1, 1)                             # (r, a, b): a for A, b for B
    C = Vector{Array{T,3}}(undef, L)
    discarded = 0.0
    for k = 1:L
        a, b = A[k], B[k]
        a_left, d, a_right = size(a)
        b_left, _, b_right = size(b)
        r = size(carry, 1)
        # Y[r, a, s, b'] = sum_b carry[r, a, b] B[b, s, b']
        Y = reshape(
            reshape(carry, r * a_left, b_left) * reshape(b, b_left, d * b_right),
            r,
            a_left,
            d,
            b_right,
        )
        site = zeros(T, r, d, a_right, b_right)          # (r, s, a', b')
        for s = 1:d
            Ys = reshape(permutedims(Y[:, :, s, :], (1, 3, 2)), r * b_right, a_left)   # (r b', a)
            Ts = reshape(Ys * a[:, s, :], r, b_right, a_right)                        # (r, b', a')
            site[:, s, :, :] .= permutedims(Ts, (1, 3, 2))
        end
        if k == L
            C[k] = reshape(site, r, d, 1)
            break
        end
        F = svd(reshape(site, r * d, a_right * b_right))
        delta = tol * norm(F.S) / sqrt(max(L - 1, 1))
        kept = _qtt_rank(F.S, delta, maxdim)
        discarded += sqrt(sum(abs2, @view(F.S[(kept+1):end]); init = 0.0))
        C[k] = reshape(F.U[:, 1:kept], r, d, kept)
        carry = reshape(Diagonal(F.S[1:kept]) * F.Vt[1:kept, :], kept, a_right, b_right)
    end
    last_norm = norm(C[L])
    return C,
    (last_norm > discarded ? discarded / (last_norm - discarded) : (discarded > 0 ? Inf : 0.0))
end

"Left zip-up of `W * A`; returns the cores and the relative error estimate."
function _apply_zipup(W::AbstractVector, A::AbstractVector, tol::Real, maxdim::Int)
    L = length(A)
    T = promote_type(eltype(W[1]), eltype(A[1]))
    carry = ones(T, 1, 1, 1)                             # (r, a, w): a for A, w for W
    C = Vector{Array{T,3}}(undef, L)
    discarded = 0.0
    for k = 1:L
        w, a = W[k], A[k]
        a_left, d, a_right = size(a)
        w_left, d_out, _, w_right = size(w)
        r = size(carry, 1)
        # Y[r, w, i, a'] = sum_a carry[r, a, w] A[a, i, a']
        Y = reshape(
            reshape(permutedims(carry, (1, 3, 2)), r * w_left, a_left) *
            reshape(a, a_left, d * a_right),
            r,
            w_left,
            d,
            a_right,
        )
        # site[r, o, a', w'] = sum_{w, i} Y[r, w, i, a'] W[w, o, i, w']
        Yp = reshape(permutedims(Y, (1, 4, 2, 3)), r * a_right, w_left * d)
        Wp = reshape(permutedims(w, (1, 3, 2, 4)), w_left * d, d_out * w_right)
        site = permutedims(reshape(Yp * Wp, r, a_right, d_out, w_right), (1, 3, 2, 4))
        if k == L
            C[k] = reshape(site, r, d_out, 1)
            break
        end
        F = svd(reshape(site, r * d_out, a_right * w_right))
        delta = tol * norm(F.S) / sqrt(max(L - 1, 1))
        kept = _qtt_rank(F.S, delta, maxdim)
        discarded += sqrt(sum(abs2, @view(F.S[(kept+1):end]); init = 0.0))
        C[k] = reshape(F.U[:, 1:kept], r, d_out, kept)
        carry = reshape(Diagonal(F.S[1:kept]) * F.Vt[1:kept, :], kept, a_right, w_right)
    end
    last_norm = norm(C[L])
    return C,
    (last_norm > discarded ? discarded / (last_norm - discarded) : (discarded > 0 ? Inf : 0.0))
end
