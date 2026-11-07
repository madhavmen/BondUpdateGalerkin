# Long-range two-body MPOs: explicit long-range terms and power-law couplings fitted by sums of
# exponentials.

"""
    LongRangeTerm(left, right, coeff, decay)

One exponentially decaying two-site term: `coeff * decay^(r-1) * left_i right_{i+r}` for
every `r >= 1`. `decay = 0` is the nearest-neighbour term with strength `coeff`.

`left` and `right` must be rank-2 (charge-neutral) operators -- see the block comment
above for why a charged channel is refused rather than approximated.
"""
struct LongRangeTerm
    left::Any
    right::Any
    coeff::Float64
    decay::Float64
end

"""
    long_range_mpo(L, nn, lr) -> MPO

`H = (nearest-neighbour terms `nn`) + (exponentially decaying terms `lr`)` as an MPO with
virtual dimension `2 + |nn| + |lr|`, independent of `L` and of the interaction range.

`nn` is an [`XXZChain`](@ref) (or an empty term list); `lr` a vector of
[`LongRangeTerm`](@ref). The coefficient is applied on the closing half of each term, as
in [`mpo_from_terms`](@ref), so the two builders cannot disagree about normalisation.
"""
function long_range_mpo(L::Int, nn::XXZChain, lr::Vector{LongRangeTerm})
    length(nn) == L ||
        throw(DimensionMismatch("nn term list is for $(length(nn)) sites, asked for $L"))
    for (k, t) in pairs(lr)
        (length(Telum.inds(t.left)) == 2 && length(Telum.inds(t.right)) == 2) || throw(
            ArgumentError(
                "long_range_mpo: term $k carries a rank-3 operator, so its channel would need " *
                "a CHARGED self-loop identity. Only charge-neutral (rank-2) long-range " *
                "operators are supported -- see the block comment above `LongRangeTerm`.",
            ),
        )
    end
    terms = nn.terms
    n, m = length(terms), length(lr)
    W = n + m + 2                                   # `id`, the channels, `done`
    Iloc = symmetry_mode() === :SU2 ? local_space(:SU2).I : local_space().I

    Ws = Any[]
    for i = 1:L
        tl, tr = "W,$(i - 1)", "W,$i"
        Iid = _mpo_block(Iloc, i, tl, tr, :none)
        Znil = TLArray(0.0 * Iid)
        mat = Matrix{Any}(nothing, W, W)
        i < L && (mat[1, 1] = Iid)
        i > 1 && (mat[W, W] = Iid)
        for t = 1:n
            i < L &&
                (mat[1, 1+t] = _mpo_block(terms[t].left, i, tl, tr, _side(terms[t].left, :left)))
            i > 1 && (
                mat[1+t, W] = TLArray(
                    terms[t].coeff *
                    _mpo_block(terms[t].right, i, tl, tr, _side(terms[t].right, :right)),
                )
            )
        end
        for k = 1:m
            c = 1 + n + k
            i < L && (mat[1, c] = _mpo_block(lr[k].left, i, tl, tr, :none))
            # THE SELF-LOOP. Needs both a predecessor and a successor site, so it exists
            # only in the interior -- at `i = 1` nothing has opened yet and at `i = L`
            # nothing may still be open, which the boundary trimming below enforces anyway.
            1 < i < L && lr[k].decay != 0.0 && (mat[c, c] = TLArray(lr[k].decay * Iid))
            i > 1 && (mat[c, W] = TLArray(lr[k].coeff * _mpo_block(lr[k].right, i, tl, tr, :none)))
        end
        if i == 1
            mat = reshape(mat[1, :], 1, W)
            mat[1, W] = Znil                        # cannot be `done` before anything ran
        elseif i == L
            mat = reshape(mat[:, W], W, 1)
            mat[1, 1] = Znil                        # nothing may stay `id` past the end
        end
        push!(Ws, TLArray(oplus(mat, (1, 4))))
    end
    return MPO(Ws)
end

"""
    long_range_zz_mpo(L; J=1.0, Jz=1.0, lambda=0.5) -> MPO

`H = J Σ_i (Sx Sx + Sy Sy)_{i,i+1} + Jz Σ_{i<j} lambda^(j-i-1) Sz_i Sz_j`.

A genuinely long-range, U(1)-symmetric test model: nearest-neighbour hopping with an
exponentially decaying Ising tail reaching every pair of sites. Virtual dimension `5`
whatever `L` is. `lambda = 0` reduces it to `xxz_mpo(L; J, delta = Jz/J)`.
"""
function long_range_zz_mpo(L::Int; J::Float64 = 1.0, Jz::Float64 = 1.0, lambda::Float64 = 0.5)
    symmetry_mode() === :SU2 &&
        throw(ArgumentError("long_range_zz_mpo needs a rank-2 Sz, which SU(2) does not have"))
    q = local_space()
    return long_range_mpo(
        L,
        xxz_chain(L; J = J, delta = 0.0),
        [LongRangeTerm(q.Sz, q.Sz, Jz, lambda)],
    )
end

"""
    LongRangeFit

What [`fit_long_range`](@ref) found, kept together so the error travels with the operator.

  - `coeffs`, `decays` -- the `c_k` and `lambda_k` of `sum_k c_k lambda_k^(r-1)`.
  - `target` -- the coupling asked for, `J(r)` for `r = 1 … R`.
  - `fitted` -- what the sum actually gives at those `r`.
  - `rel_errs` -- `|fitted - target| / |target|` per distance. The TAIL is where a fit
    fails, so the per-distance vector is kept rather than only its maximum.
  - `max_rel_err`, `l2_rel_err` -- the two summaries worth quoting.
"""
struct LongRangeFit
    coeffs::Vector{Float64}
    decays::Vector{Float64}
    target::Vector{Float64}
    fitted::Vector{Float64}
    rel_errs::Vector{Float64}
    max_rel_err::Float64
    l2_rel_err::Float64
end

"""
    default_decays(n) -> Vector{Float64}

`n` decay rates log-spaced in `(0, 1)`: `lambda_k = exp(-x_k)` with `x_k` geometric from
`0.05` to `4`. Short-range and long-range channels in one basis, so the fit has something
to put both the head and the tail of a power law on.
"""
default_decays(n::Int) =
    n <= 0 ? Float64[] :
    n == 1 ? [exp(-0.5)] : [exp(-0.05 * (4.0 / 0.05)^((k - 1) / (n - 1))) for k = 1:n]

"""
    fit_long_range(Jr; n_exp=4, decays=default_decays(n_exp)) -> LongRangeFit

Least-squares fit of the coupling sequence `Jr[r] = J(r)`, `r = 1 … length(Jr)`, by
`sum_k c_k lambda_k^(r-1)` at PRESCRIBED rates. One linear solve; see the block comment
above for why the rates are not fitted too.

The residual is reported per distance, because a power-law fit is good at the head and
poor at the tail and a single number hides that.
"""
function fit_long_range(
    Jr::AbstractVector{<:Real};
    n_exp::Int = 4,
    decays::AbstractVector{<:Real} = default_decays(n_exp),
)
    R = length(Jr)
    R >= 1 || throw(ArgumentError("fit_long_range needs at least one distance"))
    lam = collect(Float64, decays)
    all(0 .< lam .< 1) || throw(ArgumentError("decays must lie strictly in (0, 1)"))
    M = [lam[k]^(r - 1) for r = 1:R, k = 1:length(lam)]
    c = M \ collect(Float64, Jr)
    fitted = M * c
    tgt = collect(Float64, Jr)
    rel = abs.(fitted .- tgt) ./ max.(abs.(tgt), eps())
    l2 = norm(fitted .- tgt) / max(norm(tgt), eps())
    return LongRangeFit(c, lam, tgt, fitted, rel, maximum(rel; init = 0.0), l2)
end

"""
    long_range_terms(fit, left, right) -> Vector{LongRangeTerm}

The fit as one self-loop channel per exponential.
"""
long_range_terms(fit::LongRangeFit, left, right) =
    [LongRangeTerm(left, right, fit.coeffs[k], fit.decays[k]) for k = 1:length(fit.coeffs)]

"""
    power_law_zz_mpo(L; J=1.0, Jz=1.0, alpha=3.0, n_exp=4, decays=...) -> (MPO, LongRangeFit)

`H = J Σ_i (Sx Sx + Sy Sy)_{i,i+1} + Σ_{i<j} Jz |i-j|^-alpha Sz_i Sz_j`, with the power law
fitted by `n_exp` geometric channels.

RETURNS THE FIT AS WELL AS THE OPERATOR, and the two-tuple is the point: unlike
[`long_range_zz_mpo`](@ref) this MPO is NOT the Hamiltonian asked for, and how far off it is
at each distance is `fit.rel_errs`. Quoting a result from it without quoting that is
quoting an unstated model.
"""
function power_law_zz_mpo(
    L::Int;
    J::Float64 = 1.0,
    Jz::Float64 = 1.0,
    alpha::Float64 = 3.0,
    n_exp::Int = 4,
    decays::AbstractVector{<:Real} = default_decays(n_exp),
)
    symmetry_mode() === :SU2 &&
        throw(ArgumentError("power_law_zz_mpo needs a rank-2 Sz, which SU(2) does not have"))
    L >= 2 || throw(ArgumentError("power_law_zz_mpo needs at least two sites"))
    q = local_space()
    fit = fit_long_range([Jz * float(r)^(-alpha) for r = 1:(L-1)]; n_exp = n_exp, decays = decays)
    mpo = long_range_mpo(L, xxz_chain(L; J = J, delta = 0.0), long_range_terms(fit, q.Sz, q.Sz))
    return mpo, fit
end
