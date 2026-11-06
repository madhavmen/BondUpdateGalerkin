# bug_step_variants.jl: thin entry points around `cbe_bug_step!` (`bug_step.jl`) -- the parallel
# K/L/S step -- and `_replace_state!`, which the QTT nonlinear step uses to write a result back.
# Moved out of `bug_step.jl` unchanged, to keep that file under the 600-line limit.

# ── from BUG/Core/Sweeps/parallel_kls.jl: the step entry point of the K/L/S kernel there ──

"""
    parallel_kls_step!(psi, mpo, tau; schedule = :concurrent, kwargs...) -> CBEBugSweepInfo

ONE step of the fully parallel K/L/S integrator: the preliminary half-sweeps (and,
for `order = p > 1`, the frontier closure) of [`cbe_bug_step!`](@ref), then the K, L and root S
equations from `psi` in those frames, the four-block reconstruction and the one compression.
Takes `cbe_bug_step!`'s normal-API keywords. `schedule` is `:sequential` (K, L, then S),
`:kl_then_s` (K and L concurrently, then S) or `:concurrent` (all three); the three give the
same state up to the order of floating-point reductions. `ledger.kls` reports the new directions,
the omitted block `missing` and whether the step fell back to the standard augmented root solve.
"""
parallel_kls_step!(psi::SymMPS, mpo, tau; schedule::Symbol = :concurrent, kwargs...) =
    cbe_bug_step!(psi, mpo, tau; kwargs..., kls = schedule)

"Overwrite `psi` with the tensors and gauge centre of `src` (same length)."
function _replace_state!(psi::SymMPS, src::SymMPS)
    length(psi) == length(src) ||
        throw(DimensionMismatch("_replace_state!: $(length(psi)) vs $(length(src)) sites"))
    for site = 1:length(psi)
        psi[site] = src[site]
    end
    psi.center = src.center
    return psi
end
