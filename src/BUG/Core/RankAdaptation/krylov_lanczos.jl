# The Lanczos recurrence of the local Krylov enrichment (`_krylov_basis`, `krylov_recurrence.jl`):
# three-term, for HERMITIAN one-site generators only. `cbe_bug_step!` uses it by default when the
# step is Hermitian (`hermitian = true`, every closed-system real- or imaginary-time step). Measured
# on the XX domain wall at L = 64 (benchmarks/XXT5/results/dw_L64): the same directions, error and
# chi as Arnoldi at ~0.74x the Krylov-stage time.

"""
    _lanczos_chain(unit_seed, applyH, tau, allowance, score; maxdepth, amp, state_seed, fixed_depth)
        -> (basis, weights, n_h1, closed, capped)

Three-term Hermitian Lanczos from the unit seed `unit_seed`, for the one-site effective H. The
DD/frame projector is applied in `score`, not in `applyH`, so it does not break the Hermitian
recurrence. Each direction gets the same residual-integral weight and the same frame scoring as
[`_arnoldi_chain`](@ref); the stop is `_krylov_stop`'s. A complex diagonal element (a
non-Hermitian generator) throws. Past depth 20 every other column is reorthogonalised against
the whole basis, against the loss of orthogonality to converged Ritz directions.
"""
function _lanczos_chain(
    unit_seed,
    applyH,
    tau::ComplexF64,
    allowance::Float64,
    score;
    maxdepth::Int = _KRY_MAXDEPTH,
    amp::Float64 = 1.0,
    state_seed::Bool = true,
    fixed_depth::Int = 0,
    on_iteration = nothing,
)
    vs, weights = Any[unit_seed], Float64[amp]
    Hm = zeros(ComplexF64, maxdepth + 1, maxdepth + 1)
    closed, capped, n_h1 = false, false, 0
    previous_beta = 0.0
    for lanczos_step = 1:maxdepth
        image = applyH(vs[lanczos_step])
        n_h1 += 1
        lanczos_step > 1 && (image = _kconc(image - previous_beta * vs[lanczos_step-1]))
        alpha = _kinner(vs[lanczos_step], image)
        abs(imag(alpha)) <= 1e-9 * max(abs(alpha), 1.0) || throw(
            ArgumentError("Lanczos needs a Hermitian one-site generator (imag alpha = $alpha)"),
        )
        Hm[lanczos_step, lanczos_step] = real(alpha)
        image = _kconc(image - real(alpha) * vs[lanczos_step])
        # Very long Hermitian recurrences lose orthogonality to converged Ritz directions ("ghost"
        # Lanczos vectors). Correcting every other column past depth 20 avoids that without CGS2's
        # O(m^2) cost at small depths; a numerical safeguard, not a change of the tolerance.
        if lanczos_step >= 20 && iseven(lanczos_step)
            correction = nothing
            for basis_vector in vs
                coefficient = _kinner(basis_vector, image)
                correction =
                    correction === nothing ? coefficient * basis_vector :
                    correction + coefficient * basis_vector
            end
            correction === nothing || (image = _kconc(image - correction))
        end
        beta = norm(image)
        beta <= _KRY_BREAKDOWN && (closed = true; break)
        Hm[lanczos_step+1, lanczos_step] = beta
        Hm[lanczos_step, lanczos_step+1] = beta
        new_vector = _kconc((1.0 / beta) * image)
        push!(vs, new_vector)
        weight = amp * _kry_weight(Hm, lanczos_step, tau; state_seed = state_seed)
        push!(weights, weight)
        score(new_vector, weight, lanczos_step == 1) || (capped = true; break)
        on_iteration === nothing || on_iteration(lanczos_step, vs, copy(Hm[1:lanczos_step, 1:lanczos_step]))
        stop, closed = _krylov_stop(weight, allowance, lanczos_step, fixed_depth)
        stop && break
        previous_beta = beta
    end
    closed || (capped = true)                    # depth bound with weight left
    return (basis = vs, weights = weights, n_h1 = n_h1, closed = closed, capped = capped)
end
