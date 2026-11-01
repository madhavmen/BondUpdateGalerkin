# Moved from BUG-Julia@b155550:src/RSVDCBEBondUpdate/bug/imaginary_time.jl.

# ── NORMALISED IMAGINARY-TIME CBE-BUG (imaginary-time plan, Task 1) ──
# One step is `cbe_bug_step!` at `tau = -dbeta` plus renormalisation, with energy shift `Eshift`.
# Outside defect `dbeta ||(I-P) H psi_1||` is charged to `tol_basis`; failures are never silent.
# This file holds the step-size schedules; the step itself is in `ite_step.jl`.

"""
    log_panel_steps(beta_max, panel_count; per_panel = 4) -> Vector{Float64}

The step sizes of arXiv:2606.02930's logarithmic PANEL grid: `P = panel_count` dyadic panels with
endpoints `0, beta_max/2^(P-1), ..., beta_max/2, beta_max`, each split into `per_panel` uniform
steps (four is the paper's trapezoidal reference). The first two panels are equal and every later
one doubles, so the early steps are `2^(P-1)` times finer than the last. The largest step is
`beta_max / (2 per_panel)` WHATEVER `P` is: `P` refines the start, `per_panel` the coarsest step.
`sum(steps) == beta_max` up to roundoff. (Same construction as `benchmarks/imaginary_time`'s
`panel_steps`, whose `n` is likewise the TOTAL step count, here `P * per_panel`.)
"""
function log_panel_steps(beta_max::Real, panel_count::Int; per_panel::Int = 4)
    beta_max > 0 || throw(ArgumentError("beta_max must be positive, got $beta_max"))
    panel_count >= 1 ||
        throw(ArgumentError("log_panel_steps needs panel_count >= 1, got $panel_count"))
    per_panel >= 1 || throw(ArgumentError("per_panel must be >= 1, got $per_panel"))
    # panel endpoints: 0, then beta_max / 2^(P - panel) for panel = 1..P
    edges = vcat(0.0, [Float64(beta_max) / 2.0^(panel_count - panel) for panel = 1:panel_count])
    steps = Float64[]
    for panel = 1:panel_count
        append!(steps, fill((edges[panel+1] - edges[panel]) / per_panel, per_panel))
    end
    return steps
end

"`step_count` equal steps reaching `beta_max`: the uniform-grid control."
function uniform_beta_steps(beta_max::Real, step_count::Int)
    beta_max > 0 || throw(ArgumentError("beta_max must be positive, got $beta_max"))
    step_count >= 1 || throw(ArgumentError("step_count must be >= 1, got $step_count"))
    return fill(Float64(beta_max) / step_count, step_count)
end

"The grid points `0, b_1, ..., beta_max` a step list reaches (`cumsum` with the origin)."
beta_grid(steps::AbstractVector{<:Real}) = vcat(0.0, cumsum(Float64.(steps)))

# From BUG-Julia@b155550:src/RSVDCBEBondUpdate/qtt/qtt_imaginary.jl (the ground-state schedules).

"`step_count` steps of `step_size` (the uniform policy)."
uniform_schedule(step_size::Real, step_count::Int) = fill(Float64(step_size), step_count)

"""
    geometric_schedule(first_step, ratio, hmax, step_count) -> Vector{Float64}

`dtau_k = min(hmax, first_step * ratio^k)`, `k = 0..step_count-1`: the logarithmically expanding
policy. `ratio` is an experiment-level schedule parameter, not an accuracy tolerance.
"""
geometric_schedule(first_step::Real, ratio::Real, hmax::Real, step_count::Int) =
    [min(Float64(hmax), first_step * ratio^step_index) for step_index = 0:(step_count-1)]
