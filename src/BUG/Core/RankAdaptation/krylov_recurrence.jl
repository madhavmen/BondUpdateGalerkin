# The local Krylov recurrence behind `_krylov_enrich` (`krylov_enrichment.jl`, which also holds the
# constants used here): the Arnoldi chain, the recycled-block complement solve, and their small
# linear-algebra helpers.

# The recurrence runs on the local tensors and, for the matrix tests, on plain vectors.
_kinner(left_vector, right_vector) = tensor_inner(left_vector, right_vector)
_kinner(left_vector::Vector{<:Number}, right_vector::Vector{<:Number}) =
    dot(left_vector, right_vector)
_kconc(vector) = to_concrete(vector)
_kconc(vector::Vector{<:Number}) = vector

"A recycled vector whose part outside the block is below this fraction of it adds nothing new."
const _RECYCLE_FLOOR = 1e-8

"""
    _krylov_basis(seed, applyH, tau, allowance, score; recycled = (), maxdepth, amp, state_seed,
                  fixed = false)
        -> (basis, weights, n_h1, closed, capped, n_recycled, rec_closed, rec_overlap, rec_defect)

The local Krylov recurrence of the frame stages. `score(direction, weight, first) -> fits` takes
each direction to the frame (`first` = Test C's direction), `false` stopping the search as capped.
`basis[1]` is always the solve's input `seed / ||seed||`, the last vector the active frontier.

Without `recycled` it is Arnoldi from `seed` ([`_arnoldi_chain`](@ref)). With `recycled` -- the
previous solve's input first, then its kept Krylov vectors, all in this round's gauge -- the input
is split as `seed = alpha p + v_c` and only the complement `v_c` is extended
([`_rk_complement`](@ref)); when the previous block no longer holds `p`'s evolution it is Arnoldi
from `seed` after all.
"""
function _krylov_basis(
    seed,
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
    seed_norm = norm(seed)
    seed_norm > 0 || return (
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
    unit_seed = _kconc((1.0 / seed_norm) * seed)
    rc = (n_h1 = 0, n_recycled = 0, rec_overlap = 0.0, rec_defect = 0.0)
    if !isempty(recycled)
        rc = _rk_complement(
            unit_seed,
            recycled,
            applyH,
            teff,
            allowance,
            score;
            maxdepth = maxdepth,
            amp = amp,
            state_seed = state_seed,
        )
        # A full result means the recycled block held; otherwise `rc` carries only its counts.
        haskey(rc, :basis) && return rc
    end
    ch = _arnoldi_chain(
        unit_seed,
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
    _arnoldi_chain(unit_seed, applyH, tau, allowance, score; maxdepth, amp, state_seed, fixed)
        -> (basis, weights, n_h1, closed, capped)

Arnoldi from the unit seed `unit_seed`: each new vector weighed by `amp` times Saad's estimate
(`_kry_weight`) and offered to `score`, closed once that weight is within `allowance`. `fixed`
runs all `maxdepth` steps (the `fixed_depth` ablation of [`_krylov_enrich`](@ref)).
"""
function _arnoldi_chain(
    unit_seed,
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
    vs, ws = Any[unit_seed], Float64[amp]
    Hm = zeros(ComplexF64, maxdepth + 1, maxdepth + 1)
    for krylov_step = 1:maxdepth
        image = applyH(vs[end])
        nh += 1
        # CGS2 against EVERY previous vector, the correction accumulated lazily and subtracted
        # ONCE per pass: `2m` materialisations per solve instead of `m^2` (a per-vector update
        # allocated 28.6 GiB per step at L=20, chi=32, d=4). Classical Gram-Schmidt repeated
        # twice is as stable as modified ("twice is enough"); the coefficients accumulate
        # across both passes, so column `krylov_step` of `Hm` is the true projection of `H v_k`.
        for _pass = 1:2
            corr = nothing
            for (row, basis_vector) in pairs(vs)
                ov = _kinner(basis_vector, image)
                Hm[row, krylov_step] += ov
                corr = corr === nothing ? ov * basis_vector : corr + ov * basis_vector
            end
            corr === nothing || (image = _kconc(image - corr))
        end
        residual_norm = norm(image)
        # Invariant subspace: nothing further exists.
        residual_norm <= _KRY_BREAKDOWN && (closed = true; break)
        Hm[krylov_step+1, krylov_step] = residual_norm
        new_vector = _kconc((1.0 / residual_norm) * image)
        push!(vs, new_vector)
        weight = amp * _kry_weight(Hm, krylov_step, tau; state_seed = state_seed)
        push!(ws, weight)
        score(new_vector, weight, krylov_step == 1) || (capped = true; break)
        weight <= allowance && (closed = true; fixed || break)
    end
    closed || fixed || (capped = true)                    # depth bound with weight left
    return (basis = vs, weights = ws, n_h1 = nh, closed = closed, capped = capped)
end

"""
    _rk_complement(unit_seed, recycled, applyH, tau, allowance, score; maxdepth, amp, state_seed)

The recycled solve transports and reorthogonalizes old useful directions into the new gauge and
extends only from active frontiers; `tol_basis` carries the recycled local Krylov basis defect.
`recycled[1]` is the previous solve's input `p`, the rest its kept Krylov vectors. By linearity
`exp(tau H) unit_seed = alpha exp(tau H) p + exp(tau H) v_c` with `alpha = <p, unit_seed>` and the
complement `v_c = unit_seed - alpha p`: the previous block
`B = orth[p, recycled...]` already spans the first term if its defect for `p` under THIS round's
`H1`, `d = tau (HB - BA) phi_1(tau A) e1` with `A = B'HB`, is small, and only `v_c` is new.
In the code `p` is `previous_input`, `B` is `block` and `A` is `block_operator`.

- `amp |alpha|` within the allowance: `p` carries nothing, nothing is tested (no `H1`).
- `amp |alpha| ||d|| <= allowance / 2` (one `H1` per block vector): the block joins the frame
  weighted by `amp |alpha| |exp(tau A) e1|`, and a fresh Arnoldi chain runs from `v_c` alone with
  `amp ||v_c||` and the allowance left -- unless `v_c`'s whole evolution fits in it (`2 amp ||v_c||`
  bounds it for a contraction), when `v_c` only joins the frame. Returns the solve, `rec_closed`.
- otherwise the block no longer holds `p`'s evolution (the generator moved between the rounds) and
  the caller runs Arnoldi from `unit_seed`; returns only the counts.

The previous block is only ever tested, never extended: a block whose every vector leaves a
residual, continued one direction per step, ran to the depth bound in the matrix tests. Skipping
the test is no option either: with the generator moved by 1e-6 relative between the solves, the
complement's chain alone left errors 10x the allowance (1e-3: 1e4x).
"""
function _rk_complement(
    unit_seed,
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
    previous_input = _kconc((1.0 / np) * recycled[1])
    al = _kinner(previous_input, unit_seed)
    ampr = amp * abs(al)
    ampr > allowance || return merge(none, (rec_overlap = ampr,))
    # Orthonormalise the recycled vectors against the block, dropping those it already spans.
    block = Any[previous_input]
    for recycled_vector in Iterators.drop(recycled, 1)
        outside_part = _orth_against(block, recycled_vector)
        nx = norm(outside_part)
        nx > _RECYCLE_FLOOR * max(norm(recycled_vector), eps()) &&
            push!(block, _kconc((1.0 / nx) * outside_part))
    end
    HB = Any[_kconc(applyH(block_vector)) for block_vector in block]
    nh = length(block)
    # The block's projected operator `B'HB`, then `exp(tau A) e1` and `phi_1(tau A) e1`.
    block_operator = ComplexF64[
        _kinner(block[row], HB[column]) for row in eachindex(block), column in eachindex(block)
    ]
    exp_e1, ph = _exp_phi1_e1(tau .* block_operator)
    # The block's defect `tau (HB - BA) phi_1(tau A) e1`, weighted by `p`'s share of the seed.
    dr = ampr * norm(_kconc(tau * (_lincomb(HB, ph) - _lincomb(block, block_operator * ph))))
    counts = (n_h1 = nh, n_recycled = length(block), rec_overlap = ampr, rec_defect = dr)
    dr <= allowance / 2 || return counts
    gw = ampr .* abs.(exp_e1)
    capped, closed = false, true
    for (index, block_vector) in pairs(block)
        score(block_vector, gw[index], false) || (capped = true; break)
    end
    basis, weights = vcat(Any[unit_seed], block), vcat(amp, gw)
    # The complement `v_c`, the only part whose evolution the block does not already hold.
    vc = _kconc(unit_seed - al * previous_input)
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
        n_recycled = length(block),
        rec_closed = true,
        rec_overlap = ampr,
        rec_defect = dr,
    )
end

"`sum_k coefficients[k] vectors[k]`, skipping zero coefficients."
function _lincomb(vectors, coefficients)
    acc = nothing
    for (index, vector) in pairs(vectors)
        iszero(coefficients[index]) && continue
        acc = acc === nothing ? coefficients[index] * vector : acc + coefficients[index] * vector
    end
    return acc === nothing ? _kconc(0.0 * vectors[1]) : _kconc(acc)
end

"""
`(exp(A) e1, phi_1(A) e1)` for `A = generator`, from one augmented exponential:
`exp([A e1; 0 0])`.
"""
function _exp_phi1_e1(generator::AbstractMatrix)
    matrix_size = size(generator, 1)
    augmented = zeros(ComplexF64, matrix_size + 1, matrix_size + 1)
    augmented[1:matrix_size, 1:matrix_size] .= generator
    augmented[1, matrix_size+1] = 1
    augmented_exp = exp(augmented)
    return augmented_exp[1:matrix_size, 1], augmented_exp[1:matrix_size, matrix_size+1]
end
