# Elementwise products and the compressed operator application of QTT functions.
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

The input is first compressed at `tol / 10` (its discarded part, doubled, joins the bound): a frame
state `W Y Z` carries the frame widths `w` at every bond, and the zip-up's SVDs are `2r x w^2` (the
first L = 20 GP stage sat hours in `zgesdd` at 10 G with `tol = 0` here, which keeps every roundoff
direction of the frames).
"""
function tt_density(A0::AbstractVector; tol::Real = 1e-12, maxdim::Int = typemax(Int))
    start = time_ns()
    A, input_error = tt_compress(A0; tol = 0.1 * tol)
    B = tt_conj(A)
    C, info = _zip_rc(cut -> _product_zipup(A, B, cut, maxdim), tol, maxdim, start)
    return C, merge(info, (err_bound = info.err_bound + 2 * input_error,))
end

"""
    tt_product(A, B; tol = 1e-12, maxdim) -> (C, info)

`A .* B` compressed, by the zip-up of [`tt_density`](@ref) (same error bound, retry and `info`,
inputs compressed at `tol / 10` the same way), never the exact cores of [`tt_hadamard`](@ref).
"""
function tt_product(
    A0::AbstractVector,
    B0::AbstractVector;
    tol::Real = 1e-12,
    maxdim::Int = typemax(Int),
)
    start = time_ns()
    A, error_a = tt_compress(A0; tol = 0.1 * tol)
    B, error_b = B0 === A0 ? (A, error_a) : tt_compress(B0; tol = 0.1 * tol)
    C, info = _zip_rc(cut -> _product_zipup(A, B, cut, maxdim), tol, maxdim, start)
    return C, merge(info, (err_bound = info.err_bound + error_a + error_b,))
end

"""
    op_zipup(W, A; tol = 1e-12, maxdim) -> (C, info)

`W * A` compressed by a zip-up, with the retry and `info` of [`tt_product`](@ref): never the exact
`w chi` cores of [`op_apply`](@ref). The per-bond tails bound the error only when `W`'s right parts
are contractions, so for a general operator the bound is an estimate. `A` is compressed at
`tol / 10` first, as in [`tt_density`](@ref) (frame states carry the frame widths).
"""
function op_zipup(
    W::AbstractVector,
    A0::AbstractVector;
    tol::Real = 1e-12,
    maxdim::Int = typemax(Int),
)
    start = time_ns()
    A, input_error = tt_compress(A0; tol = 0.1 * tol)
    C, info = _zip_rc(cut -> _apply_zipup(W, A, cut, maxdim), tol, maxdim, start)
    return C, merge(info, (err_bound = info.err_bound + input_error,))
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

"""
    tt_hadamard_fit(A, B, init; tol = 1e-12, maxdim, maxsweeps = 6) -> (C, info)

`A .* B` by variational compression: two-site alternating fits of `C` to the Hadamard product,
warm-started from `init` (the previous stage iterate's density, or the step input's), never forming
the `chi_A chi_B` cores. Each two-site target is the exact projection of `A .* B` onto `C`'s
orthonormal left and right environments, and its SVD is `(chi_C d) x (d chi_C)` -- the zip-up's
are `(chi_C d) x chi_A chi_B`, which is where an L = 20 GP stage spent its hours. The two-site cut
keeps each bond's smallest rank within `tol / sqrt(L - 1)` of the norm, so ranks adapt in both
directions. Sweeps stop when the norm of `C` (the projection of `A .* B` at a fixed point) changes
by less than `tol` over a half-sweep. Inputs are compressed at `tol / 10` first, as in
[`tt_density`](@ref). `info = (chi, err_bound, seconds, retried, sweeps)`: `err_bound` is an
estimate (the last half-sweep's discarded weight plus the last norm change; a fit has no a-priori
bound).

Known limitation: the projections onto `C`'s own environments cannot see components of `A .* B`
outside them, so a warm start of much lower rank than the target stalls at a false fixed point
that the norm-change stop accepts. On a first imaginary-time step (state chi 14 -> 31-34, init
chi 10) the fit stopped at chi 16 after 2 sweeps with a relative error 2.7e-3 against the dense
`|psi|^2` while `err_bound` claimed 2e-11; more sweeps do not help, a zip-up warm start does.
Real-time steps, where the rank changes little per step, stay within the density budget.
"""
function tt_hadamard_fit(
    A0::AbstractVector,
    B0::AbstractVector,
    init::AbstractVector;
    tol::Real = 1e-12,
    maxdim::Int = typemax(Int),
    maxsweeps::Int = 6,
)
    start = time_ns()
    site_count = length(A0)
    length(B0) == site_count == length(init) ||
        throw(DimensionMismatch("tt_hadamard_fit: lengths differ"))
    A, error_a = tt_compress(A0; tol = 0.1 * tol)
    B, error_b = B0 === A0 ? (A, error_a) : tt_compress(B0; tol = 0.1 * tol)
    T = promote_type(eltype(A[1]), eltype(B[1]), eltype(init[1]))
    if site_count == 1
        C = [T.(A[1] .* B[1])]
        return C,
        (chi = 1, err_bound = 0.0, seconds = (time_ns() - start) / 1e9, retried = false, sweeps = 0)
    end
    # right-canonical start (site 1 holds the norm): the left-to-right half-sweep needs sites 2..L
    # right-orthonormal, so that each two-site target is an orthogonal projection
    C = [Array{T,3}(core) for core in first(tt_compress(init; tol = 0.0))]
    left_envs = Vector{Array{T,3}}(undef, site_count + 1)
    right_envs = Vector{Array{T,3}}(undef, site_count + 1)
    left_envs[1] = ones(T, 1, 1, 1)
    right_envs[site_count+1] = ones(T, 1, 1, 1)
    for site = site_count:-1:2
        right_envs[site] = _had_env_right(right_envs[site+1], C[site], A[site], B[site])
    end
    cut_of(target_norm) = tol * target_norm / sqrt(max(site_count - 1, 1))
    previous_norm, target_norm, discarded, sweeps, change = -1.0, 0.0, 0.0, 0, 1.0
    for half_sweep = 1:(2*maxsweeps)
        discarded = 0.0
        left_to_right = isodd(half_sweep)
        for site in (left_to_right ? (1:(site_count-1)) : ((site_count-1):-1:1))
            target = _had_two_site(
                left_envs[site],
                A[site],
                B[site],
                A[site+1],
                B[site+1],
                right_envs[site+2],
            )
            left_dim, d1, d2, right_dim = size(target)
            factorisation = svd(reshape(target, left_dim * d1, d2 * right_dim))
            target_norm = norm(factorisation.S)
            rank = _qtt_rank(factorisation.S, cut_of(target_norm), maxdim)
            discarded += sum(abs2, @view(factorisation.S[(rank+1):end]); init = 0.0)
            kept_u, kept_s = factorisation.U[:, 1:rank], factorisation.S[1:rank]
            kept_vt = factorisation.Vt[1:rank, :]
            if left_to_right
                C[site] = reshape(kept_u, left_dim, d1, rank)
                C[site+1] = reshape(Diagonal(kept_s) * kept_vt, rank, d2, right_dim)
                left_envs[site+1] = _had_env_left(left_envs[site], C[site], A[site], B[site])
            else
                C[site+1] = reshape(kept_vt, rank, d2, right_dim)
                C[site] = reshape(kept_u * Diagonal(kept_s), left_dim, d1, rank)
                right_envs[site+1] =
                    _had_env_right(right_envs[site+2], C[site+1], A[site+1], B[site+1])
            end
        end
        sweeps = half_sweep
        previous_norm >= 0 && (change = abs(target_norm - previous_norm) / max(target_norm, eps()))
        # stop on a right-to-left half-sweep (C right-canonical again, as tt_compress returns it)
        !left_to_right && previous_norm >= 0 && change <= tol && break
        previous_norm = target_norm
    end
    bound = sqrt(discarded) / max(target_norm, eps()) + change + error_a + error_b
    return C,
    (
        chi = maximum(tt_ranks(C); init = 1),
        err_bound = bound,
        seconds = (time_ns() - start) / 1e9,
        retried = false,
        sweeps = sweeps / 2,
    )
end

# `<C| A .* B>` environments `env[c, a, b]`, `C` conjugated (the bra), `A` and `B` sharing the bit.
function _had_env_left(env, Ck, Ak, Bk)
    T = promote_type(eltype(env), eltype(Ak), eltype(Bk))
    out = zeros(T, size(Ck, 3), size(Ak, 3), size(Bk, 3))
    for bit in axes(Ak, 2)
        partial = _tc(env, (2,), Ak[:, bit, :], (1,))                 # (c, b, a')
        partial = _tc(partial, (2,), Bk[:, bit, :], (1,))             # (c, a', b')
        out .+= _tc(conj(Ck[:, bit, :]), (1,), partial, (1,))         # (c', a', b')
    end
    return out
end

function _had_env_right(env, Ck, Ak, Bk)
    T = promote_type(eltype(env), eltype(Ak), eltype(Bk))
    out = zeros(T, size(Ck, 1), size(Ak, 1), size(Bk, 1))
    for bit in axes(Ak, 2)
        partial = _tc(Ak[:, bit, :], (2,), env, (2,))                 # (a, c', b')
        partial = _tc(partial, (3,), Bk[:, bit, :], (2,))             # (a, c', b)
        out .+= _tc(conj(Ck[:, bit, :]), (2,), partial, (2,))         # (c, a, b)
    end
    return out
end

# the two-site target `T[c, s1, s2, c'']` of `A .* B` between the environments
function _had_two_site(left_env, Ak, Bk, Ak1, Bk1, right_env)
    T = promote_type(eltype(left_env), eltype(Ak), eltype(right_env))
    d1, d2 = size(Ak, 2), size(Ak1, 2)
    left = Array{T,4}(undef, size(left_env, 1), d1, size(Ak, 3), size(Bk, 3))   # (c, s1, a', b')
    for bit = 1:d1
        partial = _tc(left_env, (2,), Ak[:, bit, :], (1,))            # (c, b, a')
        left[:, bit, :, :] .= _tc(partial, (2,), Bk[:, bit, :], (1,)) # (c, a', b')
    end
    # (a', b', s2, c'')
    right = Array{T,4}(undef, size(Ak1, 1), size(Bk1, 1), d2, size(right_env, 1))
    for bit = 1:d2
        partial = _tc(Ak1[:, bit, :], (2,), right_env, (2,))          # (a', c'', b'')
        partial = _tc(partial, (3,), Bk1[:, bit, :], (2,))            # (a', c'', b')
        right[:, :, bit, :] .= permutedims(partial, (1, 3, 2))        # (a', b', c'')
    end
    product =
        reshape(left, size(left_env, 1) * d1, :) * reshape(right, size(Ak1, 1) * size(Bk1, 1), :)
    return reshape(product, size(left_env, 1), d1, d2, size(right_env, 1))
end
