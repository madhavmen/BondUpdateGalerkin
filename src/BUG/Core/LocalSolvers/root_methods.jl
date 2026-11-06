# root_methods.jl: the root integrators other than the exponential, the scaled-Krylov helpers
# `krylov.jl` uses (`_tail`, `_pick_tau`, `_hess_coeffs`) and the public `expv` dispatcher.
# Included right after `krylov.jl`.

# ── root integrators other than `:exp` ──
# `:trapezoid`, `:radau2`, `:radau3` are the rational stability functions of one implicit RK step,
# solved on the Krylov basis. The spectrum is always shifted to the extreme Ritz value: R is not
# shift-covariant, and an unshifted imaginary-time step hits R's pole. The global order is the
# smaller of root and closure.

"Root integrators [`expv`](@ref) accepts, see the block above."
const ROOT_METHODS = (:exp, :trapezoid, :radau2, :radau3, :radau2_gmres)

# Numerator / denominator coefficients of each stability function, constant term first.
# `:radau2_gmres` is the same Runge-Kutta method as `:radau2`, solved by its coupled stages.
const _ROOT_PADE = Dict(
    :trapezoid => ([1.0, 1 / 2], [1.0, -1 / 2]),
    :radau2 => ([1.0, 1 / 3], [1.0, -2 / 3, 1 / 6]),
    :radau2_gmres => ([1.0, 1 / 3], [1.0, -2 / 3, 1 / 6]),
    :radau3 => ([1.0, 2 / 5, 1 / 20], [1.0, -3 / 5, 3 / 20, -1 / 60]),
)

"Classical order of a root integrator (`typemax(Int)` for the exponential)."
root_order(root_method::Symbol) =
    root_method === :exp ? typemax(Int) :
    root_method === :trapezoid ? 2 :
    root_method in (:radau2, :radau2_gmres) ? 3 :
    root_method === :radau3 ? 5 : _bad_root(root_method)

# Throw the `ArgumentError` every entry point raises for an unknown `root_method`.
_bad_root(root_method) =
    throw(ArgumentError("root_method must be one of $(ROOT_METHODS), got :$root_method"))

"The stability function `R(z)` of `root_method` at `z = argument` (`exp(z)` for `:exp`)."
function root_stability(root_method::Symbol, argument::Number)
    root_method === :exp && return exp(ComplexF64(argument))
    haskey(_ROOT_PADE, root_method) || _bad_root(root_method)
    numerator, denominator = _ROOT_PADE[root_method]
    return evalpoly(ComplexF64(argument), numerator) / evalpoly(ComplexF64(argument), denominator)
end

"`R(matrix) rhs` for a small dense `matrix`: one solve with the denominator polynomial."
function _rational_apply(root_method::Symbol, matrix::AbstractMatrix, rhs::AbstractVector)
    numerator, denominator = _ROOT_PADE[root_method]
    # the polynomial with coefficients `poly_coeffs` (constant first) at `matrix`, by Horner
    poly(poly_coeffs) = (result = poly_coeffs[end] * Matrix{ComplexF64}(I, size(matrix)...);
    for power = (length(poly_coeffs)-1):-1:1
        result = result * matrix + poly_coeffs[power] * I
    end;
    result)
    return poly(denominator) \ (poly(numerator) * rhs)
end

"The shift of a rational root: the Ritz value maximising `Re(tau*lambda)` (none in real time)."
_rational_shift(vals, tau::ComplexF64) =
    (isempty(vals) || real(tau) == 0) ? zero(ComplexF64) :
    ComplexF64(vals[argmax([real(tau * ComplexF64(value)) for value in vals])])

"`R(tau*(T - sigma))e_1` for the Hermitian tridiagonal's eigendecomposition; scale not returned."
function _rational_coeffs(vals, vecs, tau::ComplexF64, root_method::Symbol)
    sigma = _rational_shift(vals, tau)
    # e_1 in the eigenbasis (first row of `vecs`), each component times R of its shifted value
    eigen_weights =
        root_stability.(root_method, tau .* (ComplexF64.(vals) .- sigma)) .* ComplexF64.(vecs[1, :])
    return ComplexF64.(vecs) * eigen_weights, zero(ComplexF64)
end

"""
Saad's a-posteriori estimate of the next basis vector's contribution: `beta_k*|[exp(tau*T_k)]_k1|`.
This is the CONVERGENCE measure. `beta` alone is a BREAKDOWN detector and does not fire on a
generic operator -- measured: identical results for thresholds from 1e-6 to 1e-13.
"""
_tail(coeff::Vector{ComplexF64}, betak::Float64) = isempty(coeff) ? 0.0 : betak * abs(coeff[end])

"Nodes `s = j / _TAIL_NODES` of [`_tail_sup`](@ref)'s guard over the step."
const _TAIL_NODES = 8

"""
    _tail_sup(evalcoeff, step, betak; above = Inf) -> Float64

Saad's tail [`_tail`](@ref) at `step`, guarded against cancellation in real time: the largest
`betak * |[exp(s step T_k)]_{k,1}|` over `s = j / _TAIL_NODES`, `j = 1, ..., _TAIL_NODES`. The
endpoint entry alone can vanish while the residual over the step does not (`T_2` with eigenvalue
gap `2 pi / |step|` stops at depth 2 with an O(1) error). In real time the error is at most
`|step| betak int_0^1 |[exp(s step T_k)]_{k,1}| ds`, which this samples. Other steps (imaginary
time, loss) keep the endpoint value: their coefficients carry per-`s` shifts. A stop test passes
its threshold as `above`: once the tail reaches it the answer is known, so a step that is not
converging costs one coefficient evaluation, as without the guard.
"""
function _tail_sup(evalcoeff, step::ComplexF64, betak::Float64; above::Float64 = Inf)
    tail = _tail(first(evalcoeff(step)), betak)
    iszero(real(step)) || return tail
    for node = 1:(_TAIL_NODES-1)
        tail >= above && return tail
        tail = max(tail, _tail(first(evalcoeff(step * (node / _TAIL_NODES))), betak))
    end
    return tail
end

"""
    _pick_tau(evalcoeff, tau_rem, betak, conv_tol) -> ComplexF64

Largest fraction of `tau_rem` this already-built basis can integrate accurately.

Bisects on the SMALL matrix only -- `evalcoeff(tau')` is `O(k^2)` from the cached
eigendecomposition -- so choosing the substep costs NO operator applications. That is the whole
economy of the scheme: the basis is built once and then asked how far it can go.
"""
function _pick_tau(evalcoeff, tau_rem::ComplexF64, betak::Float64, conv_tol::Float64)
    # whether the basis integrates a step of `trial_tau` within tolerance and representably
    accepted(trial_tau) = begin
        tl = _tail_sup(evalcoeff, trial_tau, betak; above = conv_tol > 0 ? conv_tol : Inf)
        # representability is a hard constraint even when `conv_tol` is off: an exponent past
        # `_EXPV_SAFE_EXPONENT` destroys the small coefficients whatever the tail estimate says
        isfinite(tl) && (conv_tol <= 0 || tl <= conv_tol)
    end
    accepted(tau_rem) && return tau_rem
    lo, hi = 0.0, 1.0                       # fraction of tau_rem
    for _ = 1:40
        mid = 0.5 * (lo + hi)
        accepted(mid * tau_rem) ? (lo = mid) : (hi = mid)
    end
    # never return zero: a step that cannot be taken at all must still make progress, and the
    # caller's `maxiter` is then the thing to raise
    return max(lo, 1e-6) * tau_rem
end

"""
`exp(tau*Hm)e_1` for an upper-Hessenberg `Hm`, shifted the same way the Hermitian path is.
Returns the LOG of the factored scale, for the reason [`_coeffs_from_eigen`](@ref) gives.
"""
function _hess_coeffs(Hm::Matrix{ComplexF64}, tau::ComplexF64)
    isempty(Hm) && return (ComplexF64[], zero(ComplexF64))
    sigma = _shift(eigvals(Hm), tau)
    propagator = exp(tau * (Hm - sigma * I))
    return propagator[:, 1], tau * sigma
end

"""
`R(tau*(Hm - sigma))e_1` on the Arnoldi Hessenberg, `sigma` the eigenvalue maximising
`Re(tau*lambda)` (see the root-integrator block); scale not returned.
"""
function _hess_rational(Hm::Matrix{ComplexF64}, tau::ComplexF64, root_method::Symbol)
    isempty(Hm) && return (ComplexF64[], zero(ComplexF64))
    sigma = _rational_shift(eigvals(Hm), tau)
    unit_vector = zeros(ComplexF64, size(Hm, 1))
    unit_vector[1] = 1
    return _rational_apply(root_method, tau * (Hm - sigma * I), unit_vector), zero(ComplexF64)
end

# ── `:radau2_gmres`: Radau IIA(2) coupled stages by matrix-free GMRES ──
# The stages solve `(I - tau A ⊗ B) Y = 1 ⊗ x`. Under `A = P Λ P^{-1}` they decouple into the
# shifted systems `(I - tau λ_k B) Z_k = d_k x`, `d = P^{-1} 1`, `Y = (P ⊗ I) Z`. A Krylov space is
# shift invariant and both right-hand sides are multiples of `x`, so ONE Arnoldi basis of `B` is
# recycled for both stages: an iteration costs one action, not the coupled system's two. The
# method is stiffly accurate, so the end state is `Y_2`. The spectral shift is `:radau2`'s
# (`_rational_shift`, none in real time): converged, both give the same rational step.
# In the code `P` is `stage_eigvecs`, `P^{-1}` is `stage_eigvecs_inv` and `Y` is `stage_values`.

# The Radau IIA(2) Butcher matrix `A`, and its eigendecomposition `(Λ, P, P^{-1})` computed once.
const _RADAU2_A = [5/12 -1/12; 3/4 1/4]
const _RADAU2_EIG = let decomposition = eigen(ComplexF64.(_RADAU2_A))
    (decomposition.values, decomposition.vectors, inv(decomposition.vectors))
end
"Bound on the restarted GMRES cycles of one Radau solve; missing `conv_tol` after it is reported."
const _RADAU_MAXCYCLE = 4

"""
    _shifted_gmres(apply, rhs, tau, zs, ds, Wt, sigma, target, maxiter, tol)
        -> (Zs, res, nmv, sigma)

One GMRES cycle for the systems `(I - z_k (B - sigma)) Z_k = d_k rhs` on one Arnoldi basis of `B`
from `rhs`. Each `Z_k` minimises its own residual `g_k`; the cycle stops when the stage residual
they make, `‖Σ_k Wt[:, k] ⊗ g_k‖` (`Wt` the columns of `P` they enter with), reaches `target`, at
breakdown (`tol`) or at `maxiter`. `sigma = nothing` takes `:radau2`'s shift from the current Ritz
values; a restart passes the first cycle's, so every cycle solves the same system.
"""
function _shifted_gmres(
    apply,
    rhs,
    tau::ComplexF64,
    zs,
    ds,
    Wt,
    sigma,
    target::Float64,
    maxiter::Int,
    tol::Float64,
)
    beta = norm(rhs)
    basis = Any[to_concrete((1.0 / beta) * rhs)]
    hess_matrix = zeros(ComplexF64, maxiter + 1, maxiter)
    ys, res, sig, nmv = Vector{Vector{ComplexF64}}(), Inf, zero(ComplexF64), 0
    for krylov_step = 1:maxiter
        next_vector = _cgs2!(hess_matrix, krylov_step, basis, apply(basis[krylov_step]))
        nmv += 1
        next_norm = norm(next_vector)
        hess_matrix[krylov_step+1, krylov_step] = next_norm
        # the (krylov_step + 1) x krylov_step Hessenberg and the matching rectangular identity
        Hb = hess_matrix[1:(krylov_step+1), 1:krylov_step]
        Ib = Matrix{ComplexF64}(I, krylov_step + 1, krylov_step)
        sig =
            sigma === nothing ?
            _rational_shift(eigvals(hess_matrix[1:krylov_step, 1:krylov_step]), tau) : sigma
        beta_e1 = zeros(ComplexF64, krylov_step + 1)        # beta * e_1, the projected rhs
        beta_e1[1] = beta
        stage_residuals = zeros(ComplexF64, krylov_step + 1, length(zs))
        empty!(ys)
        for (system_index, stage_factor) in pairs(zs)
            system_matrix = Ib - stage_factor * (Hb - sig * Ib)
            # the small least-squares problem for this shifted system's Krylov coordinates
            coords = system_matrix \ (ds[system_index] * beta_e1)
            push!(ys, coords)
            stage_residuals[:, system_index] = ds[system_index] * beta_e1 - system_matrix * coords
        end
        res = norm(stage_residuals * transpose(Wt))
        (next_norm < tol || res <= target || krylov_step == maxiter) && break
        push!(basis, to_concrete((1.0 / next_norm) * next_vector))
    end
    return ([_combine(coords, basis[1:length(coords)], 1.0) for coords in ys], res, nmv, sig)
end

"""
    radau2_gmres_expv(apply, tau, vector; tol = 1e-13, maxiter = 60, conv_tol = 0.0,
                      report = nothing)

One Radau IIA(2) step (order 3, L-stable) of `dx/dt = B x` over `tau` from `x = vector`, the
coupled stages solved matrix-free by GMRES (block above) to the relative stage residual `conv_tol`
(`tol_root`; `0` runs one cycle to breakdown or `maxiter`), restarted at most
`_RADAU_MAXCYCLE` times. `report` receives `(iterations, substeps = 1, err, cycles, stages)`: `err`
is the TRUE relative stage residual `‖1 ⊗ x - (I - tau A ⊗ (B - sigma)) Y‖ / ‖x‖`, recomputed with
two actions rather than read off the recurrence, and `stages = (Y_1, Y_2)`. As for `:radau2` the
shift's scale is not returned.
"""
function radau2_gmres_expv(
    apply,
    tau::ComplexF64,
    vector;
    tol::Float64 = 1e-13,
    maxiter::Int = 60,
    conv_tol::Float64 = 0.0,
    report::Union{Nothing,Base.RefValue} = nothing,
)
    xc = _as_complex(vector)
    beta0 = norm(xc)
    if beta0 == 0
        _record_krylov(0)
        _report!(report, 0, 0, 0.0)
        return xc
    end
    lam, stage_eigvecs, stage_eigvecs_inv = _RADAU2_EIG
    zs = tau .* lam
    target = conv_tol > 0 ? conv_tol * beta0 : 0.0
    Zs, _, nmv, sigma = _shifted_gmres(
        apply,
        xc,
        tau,
        zs,
        stage_eigvecs_inv * ones(ComplexF64, 2),
        stage_eigvecs,
        nothing,
        target,
        maxiter,
        tol,
    )
    # stage `stage_index` of `Y = (P ⊗ I) Z` from the decoupled solutions `Zs`
    stage(Zs, stage_index) =
        to_concrete(stage_eigvecs[stage_index, 1] * Zs[1] + stage_eigvecs[stage_index, 2] * Zs[2])
    stage_values = [stage(Zs, 1), stage(Zs, 2)]
    # the coupled system's residual `1 ⊗ x - stage_values + tau A ⊗ (B - sigma) stage_values`,
    # per stage and as one norm; two fresh actions
    function residual(stage_values)
        BY = [
            (image = apply(stage_value);
            iszero(sigma) ? image : to_concrete(image - sigma * stage_value)) for
            stage_value in stage_values
        ]
        nmv += 2
        stage_residuals = [
            to_concrete(
                xc - stage_values[stage_index] +
                tau * (_RADAU2_A[stage_index, 1] * BY[1] + _RADAU2_A[stage_index, 2] * BY[2]),
            ) for stage_index = 1:2
        ]
        return stage_residuals, sqrt(norm(stage_residuals[1])^2 + norm(stage_residuals[2])^2)
    end
    stage_residuals, res = residual(stage_values)
    cycles = 1
    while target > 0 && res > target && cycles < _RADAU_MAXCYCLE
        # Restart on the true residual: decoupled again, but the two right-hand sides are no longer
        # parallel, so each stage system gets its own basis.
        cycles += 1
        dZ = Any[]
        for stage_index = 1:2
            rho = to_concrete(
                stage_eigvecs_inv[stage_index, 1] * stage_residuals[1] +
                stage_eigvecs_inv[stage_index, 2] * stage_residuals[2],
            )
            if norm(rho) == 0
                push!(dZ, rho)
                continue
            end
            Zk, _, stage_matvecs, _ = _shifted_gmres(
                apply,
                rho,
                tau,
                zs[stage_index:stage_index],
                ComplexF64[1],
                stage_eigvecs[:, stage_index:stage_index],
                sigma,
                target / sqrt(2),
                maxiter,
                tol,
            )
            nmv += stage_matvecs
            push!(dZ, Zk[1])
        end
        stage_values =
            [to_concrete(stage_values[stage_index] + stage(dZ, stage_index)) for stage_index = 1:2]
        stage_residuals, res = residual(stage_values)
    end
    _record_krylov(nmv)
    report === nothing || (
        report[] = (
            iterations = nmv,
            substeps = 1,
            err = res / beta0,
            cycles = cycles,
            stages = (stage_values[1], stage_values[2]),
        )
    )
    return stage_values[2]
end

"""
    expv(apply, tau, vector; hermitian=true, maxiter=60, tol=1e-13, reorth=false,
         conv_tol=0.0, substeps=1)

`exp(tau*A)*vector` where `apply(v) = A*v` is matrix-free. `hermitian=true` uses Lanczos, `false`
Arnoldi (needed for the non-Hermitian K and L generators). `conv_tol > 0` stops on Saad's
estimate; `substeps = 0` splits `tau` adaptively. `report = Ref{Any}()` receives `(iterations,
substeps, err)` ([`_report!`](@ref)). `root_method` (`:exp` default) can also be `:trapezoid`,
`:radau2` or `:radau3`, which apply that implicit stability function in one stage and ignore
`substeps`.
"""
function expv(
    apply,
    tau::ComplexF64,
    vector;
    hermitian::Bool = true,
    maxiter::Int = 60,
    tol::Float64 = 1e-13,
    reorth::Bool = false,
    conv_tol::Float64 = 0.0,
    substeps::Int = 1,
    report::Union{Nothing,Base.RefValue} = nothing,
    root_method::Symbol = :exp,
)
    return hermitian ?
           lanczos_expv(
        apply,
        tau,
        vector;
        tol = tol,
        maxiter = maxiter,
        reorth = reorth,
        conv_tol = conv_tol,
        substeps = substeps,
        report = report,
        root_method = root_method,
    ) :
           arnoldi_expv(
        apply,
        tau,
        vector;
        tol = tol,
        maxiter = maxiter,
        conv_tol = conv_tol,
        substeps = substeps,
        report = report,
        root_method = root_method,
    )
end

# Any numeric `tau` (real time step, imaginary time step) is promoted to ComplexF64.
expv(apply, tau::Number, vector; kwargs...) = expv(apply, ComplexF64(tau), vector; kwargs...)
