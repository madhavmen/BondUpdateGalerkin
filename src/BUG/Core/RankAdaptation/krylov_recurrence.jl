# The local Krylov recurrence behind `_krylov_enrich` (`krylov_enrichment.jl`, which also holds the
# constants used here): the Arnoldi chain and its inner products.

# The recurrence runs on the local tensors and, for the matrix tests, on plain vectors.
_kinner(left_vector, right_vector) = tensor_inner(left_vector, right_vector)
_kinner(left_vector::Vector{<:Number}, right_vector::Vector{<:Number}) =
    dot(left_vector, right_vector)
_kconc(vector) = TLArray(vector)
_kconc(vector::Vector{<:Number}) = vector

"""
    _krylov_basis(seed, applyH, tau, allowance, score; maxdepth, amp, state_seed)
        -> (basis, weights, n_h1, closed, capped)

The local Krylov recurrence of the frame stages: Arnoldi from `seed / ||seed||`
([`_arnoldi_chain`](@ref)). `score(direction, weight, first) -> fits` takes each direction to the
frame (`first` = Test C's direction), `false` stopping the search as capped. `basis[1]` is the
solve's input `seed / ||seed||`, the last vector the active frontier.
"""
function _krylov_basis(
    seed,
    applyH,
    tau::ComplexF64,
    allowance::Float64,
    score;
    maxdepth::Int = _KRY_MAXDEPTH,
    amp::Float64 = 1.0,
    state_seed::Bool = true,
)
    seed_norm = norm(seed)
    seed_norm > 0 ||
        return (basis = Any[], weights = Float64[], n_h1 = 0, closed = true, capped = false)
    # `tau = 0` asks the unit-time closure question, as `basis_allocation` does.
    teff = iszero(tau) ? one(tau) : tau
    unit_seed = _kconc((1.0 / seed_norm) * seed)
    return _arnoldi_chain(
        unit_seed,
        applyH,
        teff,
        allowance,
        score;
        maxdepth = maxdepth,
        amp = amp,
        state_seed = state_seed,
    )
end

"""
    _arnoldi_chain(unit_seed, applyH, tau, allowance, score; maxdepth, amp, state_seed)
        -> (basis, weights, n_h1, closed, capped)

Arnoldi from the unit seed `unit_seed`: each new vector weighed by `amp` times its residual integral
(`_kry_weight`) and offered to `score`, closed once that weight is within `allowance`.
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
        weight <= allowance && (closed = true; break)
    end
    closed || (capped = true)                    # depth bound with weight left
    return (basis = vs, weights = ws, n_h1 = nh, closed = closed, capped = capped)
end
