# Dense core algebra for quantics fields: construction, compression, contraction, evaluation, and
# placing 1D chains on a grid.
#
# Moved from BUG-Julia@b155550:src/RSVDCBEBondUpdate/qtt/qtt_cores.jl (the TT basics and the grid
# placement; operators are in `mpo_algebra.jl`, pointwise operations in `Elementwise/`). `qtt_embed`
# takes a `QTTOrdering` of any number of axes; the `QTTGrid` method is its 2D case.
#
# Fields and operators are built as plain Julia arrays and only then wrapped as Telum tensors
# (`state_adapter.jl`): under `:none` a Telum tensor is ONE dense block, so nothing is lost, and the
# construction never touches a 2^L object. Conventions:
#   TT core       (chi_l, d, chi_r);  site 1 is the FASTEST index of a dense contraction
#   operator core (w_l, out, in, w_r); (O psi)[out] = sum_in O[out, in] psi[in]
# A 1D chain is listed in LOGICAL order, bit 0 (least significant) first; `qtt_embed` places it on
# the ordering's sites.

# ── TT basics ──

"Smallest rank `r >= 1` whose discarded tail `sqrt(sum S[r+1:end]^2)` is at most `delta`, capped."
function _qtt_rank(S::AbstractVector{<:Real}, delta::Real, maxdim::Int)
    r = length(S)
    tail = 0.0
    while r > 1 && tail + S[r]^2 <= delta^2
        tail += S[r]^2
        r -= 1
    end
    return max(min(r, maxdim), 1)
end

"""
    tt_dense(C) -> Array

The full tensor `(d, ..., d)` of a TT, site 1 first (fastest). Small sizes only.
"""
function tt_dense(C::AbstractVector)
    L = length(C)
    d = size(C[1], 2)
    size(C[1], 1) == 1 ||
        throw(ArgumentError("tt_dense: left boundary rank is $(size(C[1], 1))"))
    M = reshape(C[1], d, size(C[1], 3))
    for k = 2:L
        chi, dk, chr = size(C[k])
        M = reshape(M * reshape(C[k], chi, dk * chr), :, chr)
    end
    size(M, 2) == 1 || throw(ArgumentError("tt_dense: right boundary rank is $(size(M, 2))"))
    return reshape(M, ntuple(k -> size(C[k], 2), L))
end

"""
    tt_from_dense(A; tol = 1e-14, maxdim) -> Vector{Array{T,3}}

TT-SVD of a `(d, ..., d)` array (site 1 = dimension 1) at relative Frobenius tolerance `tol`,
split evenly over the bonds. Fixture construction and 1D functions only.
"""
function tt_from_dense(A::AbstractArray{T}; tol::Real = 1e-14, maxdim::Int = typemax(Int)) where {T}
    L = ndims(A)
    nrm = norm(A)
    delta = tol * nrm / sqrt(max(L - 1, 1))
    C = Vector{Array{T,3}}(undef, L)
    L == 1 && (C[1] = reshape(Array(A), 1, size(A, 1), 1); return C)
    M = reshape(Array(A), size(A, 1), :)
    chi = 1
    for k = 1:(L-1)
        d = size(A, k)
        M = reshape(M, chi * d, :)
        F = svd(M)
        r = _qtt_rank(F.S, delta, maxdim)
        C[k] = reshape(F.U[:, 1:r], chi, d, r)
        M = Diagonal(F.S[1:r]) * F.Vt[1:r, :]
        chi = r
    end
    C[L] = reshape(M, chi, size(A, L), 1)
    return C
end

"""
    tt_from_samples(vals; tol = 1e-14, maxdim) -> TT

The 1D quantics TT of `vals[i + 1] = f(x_i)`, `length(vals) = 2^n`, in logical order (bit 0 first).
"""
function tt_from_samples(vals::AbstractVector; tol::Real = 1e-14, maxdim::Int = typemax(Int))
    n = round(Int, log2(length(vals)))
    2^n == length(vals) || throw(ArgumentError("need 2^n samples, got $(length(vals))"))
    return tt_from_dense(reshape(collect(vals), ntuple(_ -> 2, n)); tol = tol, maxdim = maxdim)
end

"""
    tt_compress(C; tol = 1e-14, maxdim, relative = true) -> (C, discarded)

Left QR sweep, then a right-to-left SVD sweep keeping each bond's smallest rank whose tail fits
`tol / sqrt(L - 1)` (of the norm when `relative`). The result is right-canonical (site 1 holds the
norm). `discarded` is the relative Frobenius error bound (the sum of the bonds' tails over the
norm). Works for any physical dimension `d` (operators go through `op_compress`).
"""
function tt_compress(
    C0::AbstractVector;
    tol::Real = 1e-14,
    maxdim::Int = typemax(Int),
    relative::Bool = true,
)
    L = length(C0)
    T = mapreduce(eltype, promote_type, C0)
    C = [Array{T,3}(c) for c in C0]
    for k = 1:(L-1)
        chi, d, r = size(C[k])
        F = qr(reshape(C[k], chi * d, r))
        Q = Matrix(F.Q)
        m = size(Q, 2)
        C[k] = reshape(Q, chi, d, m)
        c2, d2, r2 = size(C[k+1])
        C[k+1] = reshape(F.R * reshape(C[k+1], c2, d2 * r2), m, d2, r2)
    end
    nrm = norm(C[L])
    delta = (relative ? tol * nrm : tol) / sqrt(max(L - 1, 1))
    disc = 0.0
    for k = L:-1:2
        chi, d, r = size(C[k])
        F = svd(reshape(C[k], chi, d * r))
        rr = _qtt_rank(F.S, delta, maxdim)
        disc += sqrt(sum(abs2, @view(F.S[(rr+1):end]); init = 0.0))
        C[k] = reshape(F.Vt[1:rr, :], rr, d, r)
        c1, d1, _ = size(C[k-1])
        C[k-1] =
            reshape(reshape(C[k-1], c1 * d1, chi) * (F.U[:, 1:rr] * Diagonal(F.S[1:rr])), c1, d1, rr)
    end
    return C, (nrm > 0 ? disc / nrm : 0.0)
end

"Bond dimensions `chi_1 ... chi_{L-1}` of a TT or operator chain."
tt_ranks(C::AbstractVector) = [size(C[k], ndims(C[k])) for k = 1:(length(C)-1)]

"`<A|B>` (A conjugated) of two TTs with the same physical dimensions."
function tt_dot(A::AbstractVector, B::AbstractVector)
    E = ones(promote_type(eltype(A[1]), eltype(B[1])), 1, 1)
    for k in eachindex(A)
        a, b = A[k], B[k]
        X = reshape(E * reshape(b, size(b, 1), :), size(a, 1) * size(a, 2), size(b, 3))
        E = reshape(a, size(a, 1) * size(a, 2), size(a, 3))' * X
    end
    return E[1, 1]
end

tt_norm(A::AbstractVector) = sqrt(max(real(tt_dot(A, A)), 0.0))

"""
    tt_evaluate(C, bits) -> scalar

One amplitude: the product of the cores' `bits[k] + 1` slices. Never densifies.
"""
function tt_evaluate(C::AbstractVector, bits::AbstractVector{<:Integer})
    v = C[1][:, bits[1]+1, :]
    for k = 2:length(C)
        v = v * C[k][:, bits[k]+1, :]
    end
    return v[1, 1]
end

# Operator cores read as TT cores with `d = 4` (`op_compress`, `op_add`).
_as3(W::AbstractArray{<:Any,4}) = reshape(W, size(W, 1), size(W, 2) * size(W, 3), size(W, 4))
_as4(C::AbstractArray{<:Any,3}, d::Int = 2) = reshape(C, size(C, 1), d, d, size(C, 3))

# ── placing 1D chains on a grid ──

_bond_reverse(c::AbstractArray{<:Any,3}) = permutedims(c, (3, 2, 1))
_bond_reverse(c::AbstractArray{<:Any,4}) = permutedims(c, (4, 2, 3, 1))

"`site => core` of a 1D chain (logical order) on axis `a`'s sites, oriented in site order."
function _site_chain(o::QTTOrdering, a::Int, C::AbstractVector)
    sites = o.sites[a]
    length(C) == length(sites) || throw(
        DimensionMismatch("a $(o.axes[a]) chain needs $(length(sites)) cores, got $(length(C))"),
    )
    p = sortperm(sites)                                   # logical bits in site order
    if p == collect(1:length(sites))
        return Dict(sites[k] => C[k] for k in eachindex(C))
    elseif p == collect(length(sites):-1:1)
        return Dict(sites[k] => _bond_reverse(C[k]) for k in eachindex(C))
    end
    throw(ArgumentError("axis $(o.axes[a])'s bits are not monotone in site order"))
end

"""
    qtt_embed(o::QTTOrdering; axis = chain, ...) -> chain
    qtt_embed(g::QTTGrid; x = nothing, y = nothing) -> chain

The chain `C_1 (x) C_2 (x) ...` on the ordering's sites from 1D chains in logical order, passed by
axis name: TT cores (3-dim, a separable field `f(x) g(y)`) or operator cores (4-dim, `O_x (x) O_y`).
A missing axis is the constant 1 (fields) or the identity (operators). At every site the combined
bond is `(axis-1 bond, axis-2 bond, ...)`, axis 1 fastest, so the interleaved order carries the
other axes' bonds through.
"""
function qtt_embed(o::QTTOrdering{N}; chains...) where {N}
    for k in keys(chains)
        k in o.axes || throw(ArgumentError("no axis :$k in $(o.axes)"))
    end
    given = Any[get(chains, o.axes[a], nothing) for a = 1:N]
    all(isnothing, given) && throw(ArgumentError("qtt_embed needs a chain"))
    isop = ndims(first(c for c in given if c !== nothing)[1]) == 4
    T = promote_type((c === nothing ? Float64 : eltype(c[1]) for c in given)...)
    one_core = isop ? reshape(Matrix{T}(I, 2, 2), 1, 2, 2, 1) : ones(T, 1, 2, 1)
    placed = [
        _site_chain(o, a, given[a] === nothing ? [copy(one_core) for _ = 1:o.bits[a]] : given[a])
        for a = 1:N
    ]
    owner = zeros(Int, qtt_length(o))
    for a = 1:N, s in o.sites[a]
        owner[s] = a
    end
    out = Vector{Array{T,isop ? 4 : 3}}(undef, length(owner))
    bond = ones(Int, N)
    for s in eachindex(owner)
        a = owner[s]
        own = placed[a][s]
        # the passing axes' bonds: faster ones (before `a`) right of `own` in the kron
        Ilo = Matrix{T}(I, prod(bond[1:(a-1)]; init = 1), prod(bond[1:(a-1)]; init = 1))
        Ihi = Matrix{T}(I, prod(bond[(a+1):N]; init = 1), prod(bond[(a+1):N]; init = 1))
        m = size(Ilo, 1) * size(Ihi, 1)
        sl, sr = size(own, 1), size(own, ndims(own))
        if isop
            c = zeros(T, sl * m, 2, 2, sr * m)
            for p = 1:2, i = 1:2
                c[:, p, i, :] .= kron(Ihi, own[:, p, i, :], Ilo)
            end
        else
            c = zeros(T, sl * m, size(own, 2), sr * m)
            for b = 1:size(own, 2)
                c[:, b, :] .= kron(Ihi, own[:, b, :], Ilo)
            end
        end
        out[s] = c
        bond[a] = sr
    end
    return out
end

qtt_embed(g::QTTGrid; x = nothing, y = nothing) = qtt_embed(g.ordering; x = x, y = y)

"""
    qtt_function(g, f; tol = 1e-14, maxdim) -> TT

The field `f(x, y)` on the whole grid by TT-SVD of its dense samples, in site order. Fixture
construction on small grids only (`L <= 22`); a production field never goes through this.
"""
function qtt_function(g::QTTGrid, f; tol::Real = 1e-14, maxdim::Int = typemax(Int))
    L = qtt_length(g)
    L <= 22 || throw(ArgumentError("qtt_function densifies; L = $L is not a fixture size"))
    xs, ys = qtt_coords(g)
    p = qtt_site_permutation(g)
    v0 = f(xs[1], ys[1])
    A = Array{typeof(float(v0))}(undef, ntuple(_ -> 2, L))
    nx = 2^g.nx
    for c in eachindex(p)
        gi = p[c] - 1
        A[c] = f(xs[gi%nx+1], ys[gi÷nx+1])
    end
    return tt_from_dense(A; tol = tol, maxdim = maxdim)
end

"""
    qtt_separable(g, fx, fy; tol = 1e-14) -> TT

`fx(x) fy(y)` from two 1D TT-SVDs (`2^nx + 2^ny` samples; valid at production sizes). Pass
`fy = nothing` for a function of `x` alone, and vice versa.
"""
function qtt_separable(g::QTTGrid, fx, fy; tol::Real = 1e-14, maxdim::Int = typemax(Int))
    xs, ys = qtt_coords(g)
    tx = fx === nothing ? nothing : tt_from_samples(fx.(xs); tol = tol, maxdim = maxdim)
    ty = fy === nothing ? nothing : tt_from_samples(fy.(ys); tol = tol, maxdim = maxdim)
    return qtt_embed(g; x = tx, y = ty)
end

"""
    qtt_values(C, g) -> Matrix

The field on the grid, `M[ix + 1, iy + 1]`. Small grids only (tests, figures of fixtures).
"""
function qtt_values(C::AbstractVector, g::QTTGrid)
    A = tt_dense(C)
    p = qtt_site_permutation(g)
    M = zeros(eltype(A), 2^g.nx, 2^g.ny)
    for c in eachindex(p)
        M[p[c]] = A[c]
    end
    return M
end

"The value at grid point `(ix, iy)` (0-based) by one core contraction -- any size."
qtt_value(C::AbstractVector, g::QTTGrid, ix::Integer, iy::Integer) =
    tt_evaluate(C, qtt_bits(g, ix, iy))
