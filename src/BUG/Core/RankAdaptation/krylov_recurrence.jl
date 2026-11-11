# The local Krylov recurrence behind `_krylov_enrich` (`krylov_enrichment.jl`, which also holds the
# constants used here): the choice of recurrence and the inner products both recurrences share.
# One recurrence per file: `krylov_lanczos.jl` (three-term, Hermitian generators only) and
# `krylov_arnoldi.jl` (full CGS2, any generator), both included right after this file.

"The local recurrences `_krylov_basis` runs; `:auto` is resolved by `cbe_bug_step!` from `hermitian`."
const KRYLOV_RECURRENCES = (:lanczos, :arnoldi)

# The recurrence runs on the local tensors and, for the matrix tests, on plain vectors.
_kinner(left_vector, right_vector) = tensor_inner(left_vector, right_vector)
_kinner(left_vector::Vector{<:Number}, right_vector::Vector{<:Number}) =
    dot(left_vector, right_vector)
_kconc(vector) = TLArray(vector)
_kconc(vector::Vector{<:Number}) = vector

"""
    _krylov_basis(seed, applyH, tau, allowance, score; maxdepth, amp, state_seed, recurrence,
                  fixed_depth) -> (basis, weights, n_h1, closed, capped)

The local Krylov recurrence of the frame steps from `seed / ||seed||`: Lanczos
([`_lanczos_chain`](@ref)) or Arnoldi ([`_arnoldi_chain`](@ref)), by `recurrence`. Both weigh each
new direction by its residual integral over the step and offer it to `score(direction, weight,
first) -> fits` (`first` = Test C's direction); `false` stops the search as capped. `basis[1]` is
the solve's input `seed / ||seed||`, the last vector the active frontier. `fixed_depth = 0` stops
once a direction's weight is within `allowance` (the adaptive stop); `> 0` runs that many steps.
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
    recurrence::Symbol = :arnoldi,
    fixed_depth::Int = 0,
)
    recurrence in KRYLOV_RECURRENCES ||
        throw(ArgumentError("unknown local Krylov recurrence: $recurrence"))
    0 <= fixed_depth <= maxdepth || throw(ArgumentError("fixed_depth must be in 0:maxdepth"))
    seed_norm = norm(seed)
    seed_norm > 0 ||
        return (basis = Any[], weights = Float64[], n_h1 = 0, closed = true, capped = false)
    # `tau = 0` asks the unit-time closure question, as `basis_allocation` does.
    teff = iszero(tau) ? one(tau) : tau
    unit_seed = _kconc((1.0 / seed_norm) * seed)
    chain = recurrence === :lanczos ? _lanczos_chain : _arnoldi_chain
    return chain(
        unit_seed,
        applyH,
        teff,
        allowance,
        score;
        maxdepth = maxdepth,
        amp = amp,
        state_seed = state_seed,
        fixed_depth = fixed_depth,
    )
end

"""
    _krylov_stop(weight, allowance, step, fixed_depth) -> (stop, closed)

The stopping rule both recurrences share. Adaptive (`fixed_depth = 0`): stop, closed, once the new
direction's weight is within `allowance`. Fixed depth: stop at `step == fixed_depth`, closed only
if the weight is within `allowance` there -- a fixed depth never claims a tolerance it missed.
"""
function _krylov_stop(weight::Float64, allowance::Float64, step::Int, fixed_depth::Int)
    fixed_depth > 0 && return (step >= fixed_depth, step >= fixed_depth && weight <= allowance)
    return (weight <= allowance, weight <= allowance)
end
