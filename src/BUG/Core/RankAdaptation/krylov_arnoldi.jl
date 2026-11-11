# The Arnoldi recurrence of the local Krylov enrichment (`_krylov_basis`, `krylov_recurrence.jl`):
# full CGS2 orthogonalisation, valid for any one-site generator. `cbe_bug_step!` uses it when the
# step is not Hermitian (`hermitian = false`: Lindbladians, non-Hermitian H_eff), and on request.

"""
    _arnoldi_chain(unit_seed, applyH, tau, allowance, score; maxdepth, amp, state_seed, fixed_depth)
        -> (basis, weights, n_h1, closed, capped)

Arnoldi from the unit seed `unit_seed`: each new vector is orthogonalised against EVERY previous
one (CGS2), weighed by `amp` times its residual integral (`_kry_weight`) and offered to `score`;
the stop is `_krylov_stop`'s.
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
    fixed_depth::Int = 0,
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
        stop, closed = _krylov_stop(weight, allowance, krylov_step, fixed_depth)
        stop && break
    end
    closed || (capped = true)                    # depth bound with weight left
    return (basis = vs, weights = ws, n_h1 = nh, closed = closed, capped = capped)
end
