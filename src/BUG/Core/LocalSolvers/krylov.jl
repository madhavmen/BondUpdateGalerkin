# Moved from BUG-Julia@b155550:src/BondUpdateBUG/expv.jl, src/BondUpdateBUG/expv_root_methods.jl.

# ── from src/BondUpdateBUG/expv.jl ──

# Matrix-free exp(tau*A)*x for Telum tensors, ported from Alice's tensor_lanczos_expv for
# trace-by-trace parity. Arnoldi serves the non-Hermitian K and L generators, where P_perp breaks
# Hermiticity. The breakdown-only exit is not a convergence test: a generic A never breaks down, so
# a large tau silently truncates the series at maxiter. `conv_tol` (Saad's estimate) and `substeps`
# (split tau, which also keeps imaginary-time exponents finite) fix it.

# Opt-in Krylov-depth instrumentation, mirroring krylov.py's KRYLOV_LOG /
# enable_krylov_log / disable_krylov_log / get_krylov_log. Off by default, so
# zero overhead. Every expv call appends its Krylov dimension (the number of
# matrix-free applications it actually performed).
const KRYLOV_LOG = Int[]
const _KRYLOV_RECORD = Ref(false)
# The parallel K/L/S step (plan Task 6) runs three `expv` solves at once; a racy `push!` would
# silently drop entries rather than fail.
const _KRYLOV_LOCK = ReentrantLock()

enable_krylov_log() = (_KRYLOV_RECORD[] = true; empty!(KRYLOV_LOG); nothing)
disable_krylov_log() = (_KRYLOV_RECORD[] = false; nothing)
get_krylov_log() = copy(KRYLOV_LOG)
_record_krylov(k::Int) =
    (_KRYLOV_RECORD[] && lock(() -> push!(KRYLOV_LOG, k), _KRYLOV_LOCK); nothing)

"""
Largest `|Re(tau*lambda)|` tolerated inside one substep's matrix exponential.

`exp(60)` is `1e26` -- large, but 280 orders of magnitude clear of the Float64 ceiling, so the
small coefficients keep their relative accuracy. This is the target the adaptive substep search
aims at when `conv_tol` is off and the only thing bounding the step is representability.
"""
const _EXPV_SAFE_EXPONENT = 60.0

"Exponent past which the spectral shift switches on. Below it the shift is exactly `0`, so short
steps reproduce the unshifted arithmetic bit for bit and nothing already measured moves."
const _EXPV_SHIFT_TRIGGER = 30.0

"Promote a tensor to ComplexF64 so real inputs can take a complex `tau`."
_as_complex(x) = to_concrete(x * (1.0 + 0.0im))

"""
    _shift(vals, tau) -> Float64

The spectral shift that keeps `exp(tau*(lambda - sigma))` representable: the eigenvalue whose
exponent `Re(tau*lambda)` is largest, so every shifted exponent is `<= 0`.

Returns `0.0` while the largest exponent is below [`_EXPV_SHIFT_TRIGGER`], so ordinary short
steps are untouched and stay bit-identical to the unshifted code.
"""
function _shift(vals::AbstractVector, tau::ComplexF64)
    isempty(vals) && return 0.0
    # an explicit vector, not a generator: `argmax` over a generator has no well-defined index
    ex = [real(tau * ComplexF64(v)) for v in vals]
    k = argmax(ex)
    return ex[k] > _EXPV_SHIFT_TRIGGER ? real(vals[k]) : 0.0
end

"""
    hermitian_tridiagonal_exp_coeffs(alpha, beta, tau) -> Vector{ComplexF64}

Krylov coefficients of `exp(tau*T)e_1` for the Hermitian tridiagonal `T`.
Ports `krylov.py::hermitian_tridiagonal_exp_coeffs`.
"""
function hermitian_tridiagonal_exp_coeffs(
    alpha::Vector{Float64},
    beta::Vector{Float64},
    tau::ComplexF64,
)
    isempty(alpha) && return ComplexF64[]
    vals, vecs = _tridiag_eigen(alpha, beta)
    c, ls = _coeffs_from_eigen(vals, vecs, tau)
    # this entry point's contract is the FULL coefficients, so the factored scale goes back in
    # here; an unrepresentable one is inherent to asking for them, and the substepping path
    # exists precisely so the integrators never do
    return c .* exp(ls)
end

"Eigendecomposition of the leading tridiagonal, done ONCE so the substep search can evaluate
`exp(tau'*T)e_1` for any trial `tau'` in `O(k^2)` instead of re-exponentiating the matrix."
function _tridiag_eigen(alpha::Vector{Float64}, beta::Vector{Float64})
    T = isempty(beta) ? Matrix{Float64}(reshape(alpha, 1, 1)) : Matrix(SymTridiagonal(alpha, beta))
    return eigen(Symmetric(T))
end

"""
    _coeffs_from_eigen(vals, vecs, tau) -> (coeff, logscale)

`exp(tau*T)e_1` split into a SHIFTED coefficient vector and the LOG of the scalar that was
factored out, `logscale = tau*sigma`. Keeping them apart is what lets the caller carry a huge
imaginary-time scale in log space rather than overflowing it into the vector.

⛔ THE LOG IS RETURNED, NOT THE SCALAR, and that is the whole point. Returning `exp(tau*sigma)`
for the caller to `log` again would overflow to `Inf` FIRST -- at exactly the large `tau` the
shift exists to survive -- and `log(Inf)` is `Inf`, so the accumulated scale would be destroyed
by the very round trip meant to preserve it.
"""
function _coeffs_from_eigen(vals, vecs, tau::ComplexF64)
    sigma = _shift(vals, tau)
    w = exp.(tau .* (ComplexF64.(vals) .- sigma)) .* ComplexF64.(vecs[1, :])
    return ComplexF64.(vecs) * w, tau * sigma
end

# ── from src/BondUpdateBUG/expv_root_methods.jl ──

"""
    lanczos_expv(apply, tau, x; tol=1e-13, maxiter=60, reorth=false,
                 conv_tol=0.0, substeps=1)

`exp(tau*A)*x` for a **Hermitian** action `apply(v) = A*v`. Three-term recurrence with a
breakdown-only exit, matching `krylov.py::tensor_lanczos_expv` and its defaults. `reorth=true`
orthogonalises twice against the whole basis: more stable, but not what the reference kernel does.
`conv_tol > 0` adds Saad's convergence estimate; `substeps != 1` splits `tau`.
"""
function lanczos_expv(
    apply,
    tau::ComplexF64,
    x;
    tol::Float64 = 1e-13,
    maxiter::Int = 60,
    reorth::Bool = false,
    conv_tol::Float64 = 0.0,
    substeps::Int = 1,
    report::Union{Nothing,Base.RefValue} = nothing,
    root_method::Symbol = :exp,
)
    # GMRES needs the full orthogonalisation whatever `B` is: the shifted stage systems are
    # complex non-Hermitian even for a Hermitian generator
    root_method === :radau2_gmres && return radau2_gmres_expv(
        apply,
        tau,
        x;
        tol = tol,
        maxiter = maxiter,
        conv_tol = conv_tol,
        report = report,
    )
    root_method in ROOT_METHODS || _bad_root(root_method)
    # An implicit root takes the WHOLE step in one stage -- that is what it is for -- so the
    # exponential's substep split does not apply to it.
    root_method === :exp || (substeps = 1)
    coeffs(vals, vecs, t) =
        root_method === :exp ? _coeffs_from_eigen(vals, vecs, t) :
        _rational_coeffs(vals, vecs, t, root_method)
    xc = _as_complex(x)
    beta0 = norm(xc)
    if beta0 == 0
        _record_krylov(0)
        _report!(report, 0, 0, 0.0)
        return xc
    end

    cur = xc
    tau_rem = tau
    logscale = 0.0 + 0.0im         # imaginary-time growth carried in log space, never overflowed
    total = 0
    nsub = 0
    err = 0.0

    while true
        nsub += 1
        b0 = norm(cur)
        b0 == 0 && break
        v = to_concrete((1.0 / b0) * cur)
        basis = Any[v]
        alpha = Float64[]
        betas = Float64[]

        w = apply(v)
        total += 1
        a = real(tensor_inner(v, w))
        push!(alpha, a)
        w = to_concrete(w + (-a) * v)

        # the step this basis will actually take; set once the loop ends (or converges early)
        for _ = 2:maxiter
            b = norm(w)
            b < tol && break                       # genuine breakdown: the space is exhausted
            if conv_tol > 0                        # genuine convergence: the answer stopped moving
                vals, vecs = _tridiag_eigen(alpha, betas)
                c, _ = coeffs(vals, vecs, tau_rem)
                _tail(c, b) < conv_tol && break
            end
            push!(betas, b)
            v = to_concrete((1.0 / b) * w)
            push!(basis, v)
            w = apply(v)
            total += 1
            a = real(tensor_inner(v, w))
            push!(alpha, a)
            # Materialise `w` before the reorth loop: Telum's +/- are lazy, so a symbolic `w` grew
            # by one term per step and made the loop O(k^3 N). Modified Gram-Schmidt is sequential,
            # so this changes only when each subtraction is evaluated, not the arithmetic.
            w = to_concrete(w + (-a) * v + (-b) * basis[end-1])
            if reorth
                # One materialisation per pass, not per basis vector. `w` stays concrete for the
                # whole pass and only the correction is lazy, so every inner product is against a
                # fixed `w`. That makes the pass classical Gram-Schmidt, applied twice (as stable as
                # MGS). The same fix is in `cbe_bug.jl`, which keeps BUG and tdvp2 on equal footing.
                for _pass = 1:2
                    corr = nothing
                    for u in basis
                        c = tensor_inner(u, w)
                        corr = corr === nothing ? c * u : corr + c * u
                    end
                    corr === nothing || (w = to_concrete(w - corr))
                end
            end
        end

        vals, vecs = _tridiag_eigen(alpha, betas)
        evalcoeff(t) = coeffs(vals, vecs, ComplexF64(t))
        betak = norm(w)

        tau_s = if substeps == 1
            tau_rem                                # single shot: exactly the old behaviour
        elseif substeps > 1
            tau_rem / (substeps - nsub + 1)        # fixed, equal split of what is left
        else
            _pick_tau(evalcoeff, tau_rem, betak, conv_tol)   # adaptive, free of extra matvecs
        end

        coeff, ls = evalcoeff(tau_s)
        err += _tail(coeff, betak) * b0 / beta0
        cur = _combine(coeff, basis, b0)
        logscale += ls                 # already a log: never round-trips through a huge scalar
        tau_rem -= tau_s
        (substeps == 1 || abs(tau_rem) <= 1e-14 * max(abs(tau), 1.0)) && break
        nsub >= 10_000 && error("expv: substep search failed to advance (tau_rem = $tau_rem)")
    end

    _record_krylov(total)
    _report!(report, total, nsub, err)
    return _apply_scale(cur, logscale)
end

"""
    _report!(report, iterations, substeps, err)

Hand a solve's own account back to a caller that passed `report = Ref{Any}()`:
`(iterations, substeps, err)`. `err` is Saad's a-posteriori estimate summed over the substeps,
relative to `||x||` -- exact bookkeeping in REAL time, where every substep's propagator is
unitary so the local errors cannot grow; in imaginary time the shift makes it a per-substep
relative figure. It is what `tol_expv` is checked against (plan §4, UT15).
"""
_report!(::Nothing, ::Int, ::Int, ::Float64) = nothing
_report!(r::Base.RefValue, total::Int, nsub::Int, err::Float64) =
    (r[] = (iterations = total, substeps = nsub, err = err); nothing)

"""
Largest norm a returned state may carry. The limit is `sqrt(floatmax)`, not `floatmax`: tensor-level
routines form unscaled sums of squares (inner products, the Gram products in the splitting SVD),
which overflow while entries are still finite. Julia's plain-array `norm` rescales and is safe. The
constant (1e150) sits below that limit.
"""
const _EXPV_MAX_NORM = 1e150

"""
    _apply_scale(cur, logscale) -> tensor

Put the factored scale back, unless doing so would push the state past [`_EXPV_MAX_NORM`].

⚠ WHEN IT IS DROPPED THE DIRECTION IS RIGHT AND THE NORM IS NOT. That is safe for IMAGINARY
time, where the norm grows like `exp(-tau*E_0)`, is not physical, and every caller renormalises.
In REAL time `logscale` is imaginary, `|sc| = 1` exactly, and this can never trigger -- so a
real-time norm, which IS physics, is never silently rescaled.
"""
function _apply_scale(cur, logscale::ComplexF64)
    sc = exp(logscale)
    (isfinite(abs(sc)) && sc != 0) || return cur
    n = norm(cur)
    (isfinite(n) && n > 0 && abs(sc) * n > _EXPV_MAX_NORM) && return cur
    return to_concrete(sc * cur)
end

"Recombine `out = beta0 * sum_i coeff[i] * basis[i]`."
function _combine(coeff, basis, beta0::Float64)
    out = (coeff[1] * beta0) * basis[1]
    for i = 2:length(coeff)
        out = out + (coeff[i] * beta0) * basis[i]
    end
    return to_concrete(out)
end

"""
    _cgs2!(H, j, basis, w) -> w

`w` orthogonalised against `basis` by classical Gram-Schmidt applied twice, the coefficients added
to column `j` of `H`. One materialisation per pass, as in `lanczos_expv`: a symbolic `w` would make
each inner product force a sum one term longer, so `w` stays concrete for the whole pass and only
`corr` is lazy.
"""
function _cgs2!(H::AbstractMatrix{ComplexF64}, j::Int, basis, w)
    for _pass = 1:2
        corr = nothing
        for i = 1:length(basis)
            h = tensor_inner(basis[i], w)
            H[i, j] += h
            corr = corr === nothing ? h * basis[i] : corr + h * basis[i]
        end
        corr === nothing || (w = to_concrete(w - corr))
    end
    return w
end

"""
    arnoldi_expv(apply, tau, x; tol=1e-13, maxiter=60, conv_tol=0.0, substeps=1)

`exp(tau*A)*x` for a **non-Hermitian** action, via the upper-Hessenberg Arnoldi factorisation. This
is the path for the K and L generators, where `P_perp = I - U0 U0'` breaks Hermiticity and Lanczos
is invalid. Each vector is orthogonalised twice against the full basis, since one MGS pass loses
orthogonality once the Krylov vectors align. `conv_tol` and `substeps` are as for
[`lanczos_expv`](@ref).
"""
function arnoldi_expv(
    apply,
    tau::ComplexF64,
    x;
    tol::Float64 = 1e-13,
    maxiter::Int = 60,
    conv_tol::Float64 = 0.0,
    substeps::Int = 1,
    report::Union{Nothing,Base.RefValue} = nothing,
    root_method::Symbol = :exp,
)
    root_method === :radau2_gmres && return radau2_gmres_expv(
        apply,
        tau,
        x;
        tol = tol,
        maxiter = maxiter,
        conv_tol = conv_tol,
        report = report,
    )
    root_method in ROOT_METHODS || _bad_root(root_method)
    root_method === :exp || (substeps = 1)     # an implicit root takes the whole step at once
    hcoeffs(Hm, t) = root_method === :exp ? _hess_coeffs(Hm, t) : _hess_rational(Hm, t, root_method)
    xc = _as_complex(x)
    beta0 = norm(xc)
    if beta0 == 0
        _record_krylov(0)
        _report!(report, 0, 0, 0.0)
        return xc
    end

    cur = xc
    tau_rem = tau
    logscale = 0.0 + 0.0im
    total = 0
    nsub = 0
    err = 0.0

    while true
        nsub += 1
        b0 = norm(cur)
        b0 == 0 && break
        v = to_concrete((1.0 / b0) * cur)
        basis = Any[v]
        H = zeros(ComplexF64, maxiter + 1, maxiter)
        m = 0
        betak = 0.0

        for j = 1:maxiter
            w = _cgs2!(H, j, basis, apply(basis[j]))
            total += 1
            m = j
            b = norm(w)
            betak = b
            b < tol && break
            if conv_tol > 0
                c = hcoeffs(H[1:m, 1:m], tau_rem)[1]
                _tail(c, b) < conv_tol && break
            end
            H[j+1, j] = b
            j == maxiter && break
            push!(basis, to_concrete((1.0 / b) * w))
        end

        Hm = H[1:m, 1:m]
        evalcoeff(t) = hcoeffs(Hm, ComplexF64(t))

        tau_s = if substeps == 1
            tau_rem
        elseif substeps > 1
            tau_rem / (substeps - nsub + 1)
        else
            _pick_tau(evalcoeff, tau_rem, betak, conv_tol)
        end

        coeff, ls = evalcoeff(tau_s)
        err += _tail(coeff, betak) * b0 / beta0
        cur = _combine(coeff, basis[1:m], b0)
        logscale += ls                 # already a log: never round-trips through a huge scalar
        tau_rem -= tau_s
        (substeps == 1 || abs(tau_rem) <= 1e-14 * max(abs(tau), 1.0)) && break
        nsub >= 10_000 && error("expv: substep search failed to advance (tau_rem = $tau_rem)")
    end

    _record_krylov(total)
    _report!(report, total, nsub, err)
    return _apply_scale(cur, logscale)
end
