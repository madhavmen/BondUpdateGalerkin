# Moved from BUG-Julia@b155550:src/RSVDCBEBondUpdate/bug/krylov_enrich.jl, src/RSVDCBEBondUpdate/bug/krylov_closure.jl, src/RSVDCBEBondUpdate/bug/krylov_closure_loop.jl.
# krylov_recurrence.jl: the local Krylov recurrence behind `_krylov_enrich` -- Arnoldi chain, recycled-block complement solve, and their small linear-algebra helpers.

# The recurrence runs on the local tensors and, for the matrix tests, on plain vectors.
_kinner(a, b) = tensor_inner(a, b)
_kinner(a::Vector{<:Number}, b::Vector{<:Number}) = dot(a, b)
_kconc(x) = to_concrete(x)
_kconc(x::Vector{<:Number}) = x

"A recycled vector whose part outside the block is below this fraction of it adds nothing new."
const _RECYCLE_FLOOR = 1e-8

"""
    _krylov_basis(v0, applyH, tau, allowance, score; recycled = (), maxdepth, amp, state_seed,
                  fixed = false)
        -> (basis, weights, n_h1, closed, capped, n_recycled, rec_closed, rec_overlap, rec_defect)

The local Krylov recurrence of the frame stages. `score(v, weight, first) -> fits` takes each
direction to the frame (`first` = Test C's direction), `false` stopping the search as capped.
`basis[1]` is always the solve's input `v0 / ||v0||`, the last vector the active frontier.

Without `recycled` it is Arnoldi from `v0` ([`_arnoldi_chain`](@ref)). With `recycled` -- the
previous solve's input first, then its kept Krylov vectors, all in this round's gauge -- the input
is split as `v0 = alpha p + v_c` and only the complement `v_c` is extended
([`_rk_complement`](@ref)); when the previous block no longer holds `p`'s evolution it is Arnoldi
from `v0` after all.
"""
function _krylov_basis(
    v0,
    applyH,
    tau::ComplexF64,
    allowance::Float64,
    score;
    recycled = (),
    maxdepth::Int = _KRY_MAXDEPTH,
    amp::Float64 = 1.0,
    state_seed::Bool = true,
    fixed::Bool = false,
)
    b0 = norm(v0)
    b0 > 0 || return (
        basis = Any[],
        weights = Float64[],
        n_h1 = 0,
        closed = true,
        capped = false,
        n_recycled = 0,
        rec_closed = false,
        rec_overlap = 0.0,
        rec_defect = 0.0,
    )
    # `tau = 0` asks the unit-time closure question, as `basis_allocation` does.
    teff = iszero(tau) ? one(tau) : tau
    u0 = _kconc((1.0 / b0) * v0)
    rc = (n_h1 = 0, n_recycled = 0, rec_overlap = 0.0, rec_defect = 0.0)
    if !isempty(recycled)
        rc = _rk_complement(
            u0,
            recycled,
            applyH,
            teff,
            allowance,
            score;
            maxdepth = maxdepth,
            amp = amp,
            state_seed = state_seed,
        )
        haskey(rc, :basis) && return rc
    end
    ch = _arnoldi_chain(
        u0,
        applyH,
        teff,
        allowance,
        score;
        maxdepth = maxdepth,
        amp = amp,
        state_seed = state_seed,
        fixed = fixed,
    )
    return (
        basis = ch.basis,
        weights = ch.weights,
        n_h1 = rc.n_h1 + ch.n_h1,
        closed = ch.closed,
        capped = ch.capped,
        n_recycled = rc.n_recycled,
        rec_closed = false,
        rec_overlap = rc.rec_overlap,
        rec_defect = rc.rec_defect,
    )
end

"""
    _arnoldi_chain(u, applyH, tau, allowance, score; maxdepth, amp, state_seed, fixed)
        -> (basis, weights, n_h1, closed, capped)

Arnoldi from the unit seed `u`: each new vector weighed by `amp` times Saad's estimate
(`_kry_weight`) and offered to `score`, closed once that weight is within `allowance`. `fixed`
runs all `maxdepth` steps (the `fixed_depth` ablation of [`_krylov_enrich`](@ref)).
"""
function _arnoldi_chain(
    u,
    applyH,
    tau::ComplexF64,
    allowance::Float64,
    score;
    maxdepth::Int = _KRY_MAXDEPTH,
    amp::Float64 = 1.0,
    state_seed::Bool = true,
    fixed::Bool = false,
)
    nh, capped, closed = 0, false, false
    vs, ws = Any[u], Float64[amp]
    Hm = zeros(ComplexF64, maxdepth + 1, maxdepth + 1)
    for k = 1:maxdepth
        w = applyH(vs[end])
        nh += 1
        # ⛔ CGS2 against EVERY previous vector, the correction accumulated lazily and subtracted
        # ONCE per pass: `2m` materialisations per solve instead of `m^2` (a per-vector update
        # allocated 28.6 GiB per step at L=20, chi=32, d=4). Classical Gram-Schmidt repeated
        # twice is as stable as modified ("twice is enough"); the coefficients accumulate
        # across both passes, so column `k` of `Hm` is the true projection of `H v_k`.
        for _pass = 1:2
            corr = nothing
            for (t, x) in pairs(vs)
                ov = _kinner(x, w)
                Hm[t, k] += ov
                corr = corr === nothing ? ov * x : corr + ov * x
            end
            corr === nothing || (w = _kconc(w - corr))
        end
        b = norm(w)
        b <= _KRY_BREAKDOWN && (closed = true; break)    # invariant: nothing further exists
        Hm[k+1, k] = b
        v = _kconc((1.0 / b) * w)
        push!(vs, v)
        weight = amp * _kry_weight(Hm, k, tau; state_seed = state_seed)
        push!(ws, weight)
        score(v, weight, k == 1) || (capped = true; break)
        weight <= allowance && (closed = true; fixed || break)
    end
    closed || fixed || (capped = true)                    # depth bound with weight left
    return (basis = vs, weights = ws, n_h1 = nh, closed = closed, capped = capped)
end

"""
    _rk_complement(u0, recycled, applyH, tau, allowance, score; maxdepth, amp, state_seed)

The recycled solve (plan §1.2: "transport/reorthogonalize old useful directions into the new gauge,
and extend only from active frontiers"; §1.3: `tol_basis` carries "the recycled local Krylov basis
defect"). `recycled[1]` is the previous solve's input `p`, the rest its kept Krylov vectors. By
linearity `exp(tau H) u0 = alpha exp(tau H) p + exp(tau H) v_c` with `alpha = <p, u0>` and the
complement `v_c = u0 - alpha p`: the previous block `B = orth[p, recycled...]` already spans the
first term if its defect for `p` under THIS round's `H1`, `d = tau (HB - BA) phi_1(tau A) e1` with
`A = B'HB`, is small, and only `v_c` is new.

- `amp |alpha|` within the allowance: `p` carries nothing, nothing is tested (no `H1`).
- `amp |alpha| ||d|| <= allowance / 2` (one `H1` per block vector): the block joins the frame
  weighted by `amp |alpha| |exp(tau A) e1|`, and a fresh Arnoldi chain runs from `v_c` alone with
  `amp ||v_c||` and the allowance left -- unless `v_c`'s whole evolution fits in it (`2 amp ||v_c||`
  bounds it for a contraction), when `v_c` only joins the frame. Returns the solve, `rec_closed`.
- otherwise the block no longer holds `p`'s evolution (the generator moved between the rounds) and
  the caller runs Arnoldi from `u0`; returns only the counts.

The previous block is only ever tested, never extended: a block whose every vector leaves a
residual, continued one direction per step, ran to the depth bound in the matrix tests. Skipping
the test is no option either: with the generator moved by 1e-6 relative between the solves, the
complement's chain alone left errors 10x the allowance (1e-3: 1e4x).
"""
function _rk_complement(
    u0,
    recycled,
    applyH,
    tau::ComplexF64,
    allowance::Float64,
    score;
    maxdepth::Int = _KRY_MAXDEPTH,
    amp::Float64 = 1.0,
    state_seed::Bool = true,
)
    none = (n_h1 = 0, n_recycled = 0, rec_overlap = 0.0, rec_defect = 0.0)
    np = norm(recycled[1])
    np > 0 || return none
    p = _kconc((1.0 / np) * recycled[1])
    al = _kinner(p, u0)
    ampr = amp * abs(al)
    ampr > allowance || return merge(none, (rec_overlap = ampr,))
    B = Any[p]
    for c in Iterators.drop(recycled, 1)
        x = _orth_against(B, c)
        nx = norm(x)
        nx > _RECYCLE_FLOOR * max(norm(c), eps()) && push!(B, _kconc((1.0 / nx) * x))
    end
    HB = Any[_kconc(applyH(x)) for x in B]
    nh = length(B)
    A = ComplexF64[_kinner(B[r], HB[c]) for r in eachindex(B), c in eachindex(B)]
    g, ph = _exp_phi1_e1(tau .* A)
    dr = ampr * norm(_kconc(tau * (_lincomb(HB, ph) - _lincomb(B, A * ph))))
    counts = (n_h1 = nh, n_recycled = length(B), rec_overlap = ampr, rec_defect = dr)
    dr <= allowance / 2 || return counts
    gw = ampr .* abs.(g)
    capped, closed = false, true
    for (k, x) in pairs(B)
        score(x, gw[k], false) || (capped = true; break)
    end
    basis, weights = vcat(Any[u0], B), vcat(amp, gw)
    vc = _kconc(u0 - al * p)
    nc = norm(vc)
    rest = allowance - dr
    if !capped && nc > 0
        uc = _kconc((1.0 / nc) * vc)
        if 2 * amp * nc <= rest
            score(uc, amp * nc, false) || (capped = true)
            push!(basis, uc)
            push!(weights, amp * nc)
        else
            ch = _arnoldi_chain(
                uc,
                applyH,
                tau,
                rest,
                score;
                maxdepth = maxdepth,
                amp = amp * nc,
                state_seed = state_seed,
            )
            nh += ch.n_h1
            capped, closed = ch.capped, ch.closed
            append!(basis, ch.basis)
            append!(weights, ch.weights)
        end
    end
    return (
        basis = basis,
        weights = weights,
        n_h1 = nh,
        closed = closed,
        capped = capped,
        n_recycled = length(B),
        rec_closed = true,
        rec_overlap = ampr,
        rec_defect = dr,
    )
end

"`sum_k c[k] vs[k]`."
function _lincomb(vs, c)
    acc = nothing
    for (k, v) in pairs(vs)
        iszero(c[k]) && continue
        acc = acc === nothing ? c[k] * v : acc + c[k] * v
    end
    return acc === nothing ? _kconc(0.0 * vs[1]) : _kconc(acc)
end

"`(exp(A) e1, phi_1(A) e1)` from one augmented exponential: `exp([A e1; 0 0])`."
function _exp_phi1_e1(A::AbstractMatrix)
    n = size(A, 1)
    Z = zeros(ComplexF64, n + 1, n + 1)
    Z[1:n, 1:n] .= A
    Z[1, n+1] = 1
    E = exp(Z)
    return E[1:n, 1], E[1:n, n+1]
end
