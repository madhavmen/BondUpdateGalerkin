# ── shared run configuration ──────────────────────────────────────────────────
#
# Lives here, not in a mode submodule, because all three modes and both paths (gate,
# MPO-environment) are configured by the same expansion knobs. Only the fields that mean
# something different per mode are documented per mode.

"""
    CBEBugOptions

Controls for one run of any of the three modes, mirroring `BondUpdateOptions`. `dt` is a real step
(`tau = -im*dt`) in real time and an inverse-temperature step (`tau = -dt`) in imaginary time.
`maxdim`/`trunc_thresh` set the truncation; `normalize` rescales after each step (the norm is
recorded first). `maxiter`/`tol` budget the Krylov exponential; `seed` is one RNG per run.

There is no `order` field: the MPO path takes one projected exponential per step, so there is no
splitting to symmetrise. Ground state runs on the MPO path only, since a local eigensolve on a bare
gate would minimise a single term with frozen neighbours.
"""
Base.@kwdef struct CBEBugOptions
    dt::Float64 = 0.05
    n_steps::Int = 10
    centre_expand::Bool = true
    maxdim::Int = 200
    trunc_thresh::Float64 = 1e-12
    normalize::Bool = true
    maxiter::Int = 30
    tol::Float64 = 1e-15
    # S-step scheme, honoured by the `cbe_bond_update` path only. `s_iters` selects the solver
    # (see `cbe_bond_update`: 1 = one expansion + plain Lanczos, >1 = the outer-loop control,
    # <=0 = the expanding-basis Lanczos with `-s_iters` growth passes). `s_reorth` turns on full
    # reorthogonalisation in the plain Lanczos -- measured ~5.7x fewer matvecs at one bond, not
    # yet a default because it has not been checked across a sweep.
    s_iters::Int = 1
    s_reorth::Bool = false
    # PLAIN CBE instead of RSVD-CBE: the complete fused basis replaces the thin Gaussian, so
    # the preselection sees the exact `H*Theta` and its SVD returns the true leading directions
    # rather than an estimate. This FORMS THE RANK-4 OBJECT (cost `d*chi`, not `1.2*budget`),
    # which is precisely what the sketch exists to avoid -- it is the reference arm of the A/B,
    # not a better default. Makes `seed` and the RNG inert.
    exact::Bool = false
    seed::UInt = 0x5EED
    record_magnetisation::Bool = false
    observe::Union{Nothing,Function} = nothing
end
# ── PART II: CBE as the basis update of the gate-based bond_update_bug ──
# The three modes differ only in the operator posed to `_expanding_krylov` and in the S-step solver;
# the growth, sketch, projection, ranking and reorthogonalisation are shared code.

# ── the S step: what to do with the Krylov space once it is built ─────────────

"""
    SStepSolver

How [`_expanding_krylov`](@ref) turns its tridiagonal into the local answer.

The Krylov space, and every growth pass that built it, is identical across subtypes. Only
the reduction differs, which is why the modes share one solver.
"""
abstract type SStepSolver end

"""
    Propagate(tau)

`exp(tau * H_eff)` applied to the current local core. `tau = -im*dt` is real time,
`tau = -dt` is imaginary time; the solver does not care which, and neither does anything
below it.
"""
struct Propagate <: SStepSolver
    tau::ComplexF64
end

"""
    LowestEigen()

The lowest Ritz vector of `H_eff` in the accumulated Krylov space: a DMRG local solve that replaces
the exponential and keeps the basis. The result is normalised, since the sweep minimises a unit-norm
state.

The operator must include the environments (`L + h + R`). Given the bare two-site gate, each bond
would minimise its own pair and the next bond would undo it; see `apply_h_two_site` in `henv.jl`.
"""
struct LowestEigen <: SStepSolver end

"""
    _reduce_krylov(solver, alpha, betas, beta0) -> (coeffs, value)

`coeffs[k]` multiplies the `k`-th stored orthonormal Krylov vector to build the answer.
`value` is the Ritz value for [`LowestEigen`](@ref) and `NaN` when the solver has no
eigenvalue to report.
"""
_reduce_krylov(solver::Propagate, alpha, betas, beta0) = (
    # `beta0 exp(tau T) e_1`: the start vector's norm times the exponential's first column
    ComplexF64[
        beta0 * coeff for coeff in hermitian_tridiagonal_exp_coeffs(alpha, betas, solver.tau)
    ],
    NaN,
)

function _reduce_krylov(::LowestEigen, alpha, betas, beta0)
    # `beta0` is deliberately UNUSED: the propagator carries the state's norm through the
    # step, an eigensolve replaces the state outright and the Ritz vector is normalised.
    isempty(betas) && return (ComplexF64[1.0], alpha[1])
    decomposition = eigen(SymTridiagonal(alpha, betas))
    # eigenvalues come sorted ascending, so column 1 is the lowest Ritz pair
    return (ComplexF64.(@view decomposition.vectors[:, 1]), decomposition.values[1])
end
# ══════════════════════════════════════════════════════════════════════════════════════

# ── CBE as the basis update of the validated gate-based integrator ──
# A gate-only path has no environments, so a rank stall here isolates the CBE selection itself,
# which is shared with the MPO path. It replaces the K/L ODE solves and the random sector fill of
# `kls_bond_update`.

# ── the sketched gate action ──
# Never forms H*Theta: `Om` is folded into the gate first, so the largest intermediate is
# O(d^2 g r).
# Checked against `P_perp` of `apply_gate` on Theta, for arbitrary `Om`.

"Check `gate` is tagged for the bond `frame` describes, as `apply_gate` does for `theta`; returns
`nothing` or throws `ArgumentError`."
function _check_gate(frame::BondFrame, gate)
    tl, tr = frame.site_l.itags, frame.site_r.itags
    (Telum.inds(gate)[1].itags == tl && Telum.inds(gate)[2].itags == tr) || throw(
        ArgumentError(
            "gate is tagged for ($(gate.inds[1].itags), $(gate.inds[2].itags)), " *
            "bond frame carries ($tl, $tr)",
        ),
    )
    return nothing
end
