# Moved from BUG-Julia@b155550:src/RSVDCBEBondUpdate/bug/imaginary_time.jl.

# ── NORMALISED IMAGINARY-TIME CBE-BUG (imaginary-time plan, Task 1) ──
# One step is `cbe_bug_step!` at `tau = -dbeta` plus renormalisation, with energy shift `Eshift`.
# Outside defect `dbeta ||(I-P) H psi_1||` is charged to `tol_basis`; failures are never silent.

"""
    log_panel_steps(beta_max, P; per_panel = 4) -> Vector{Float64}

The step sizes of arXiv:2606.02930's logarithmic PANEL grid: `P` dyadic panels with endpoints
`0, beta_max/2^(P-1), ..., beta_max/2, beta_max`, each split into `per_panel` uniform steps (four is
the paper's trapezoidal reference). The first two panels are equal and every later one doubles, so
the early steps are `2^(P-1)` times finer than the last. The largest step is
`beta_max / (2 per_panel)` WHATEVER `P` is: `P` refines the start, `per_panel` the coarsest step.
`sum(steps) == beta_max` up to roundoff. (Same construction as `benchmarks/imaginary_time`'s
`panel_steps`, whose `n` is likewise the TOTAL step count, here `P * per_panel`.)
"""
function log_panel_steps(beta_max::Real, P::Int; per_panel::Int = 4)
    beta_max > 0 || throw(ArgumentError("beta_max must be positive, got $beta_max"))
    P >= 1 || throw(ArgumentError("log_panel_steps needs P >= 1, got $P"))
    per_panel >= 1 || throw(ArgumentError("per_panel must be >= 1, got $per_panel"))
    edges = vcat(0.0, [Float64(beta_max) / 2.0^(P - j) for j = 1:P])
    steps = Float64[]
    for j = 1:P
        append!(steps, fill((edges[j+1] - edges[j]) / per_panel, per_panel))
    end
    return steps
end

"`n` equal steps reaching `beta_max`: the uniform-grid control."
function uniform_beta_steps(beta_max::Real, n::Int)
    beta_max > 0 || throw(ArgumentError("beta_max must be positive, got $beta_max"))
    n >= 1 || throw(ArgumentError("n must be >= 1, got $n"))
    return fill(Float64(beta_max) / n, n)
end

"The grid points `0, b_1, ..., beta_max` a step list reaches (`cumsum` with the origin)."
beta_grid(steps::AbstractVector{<:Real}) = vcat(0.0, cumsum(Float64.(steps)))

# From BUG-Julia@b155550:src/RSVDCBEBondUpdate/qtt/qtt_imaginary.jl (the ground-state schedules).

"`n` steps of `h` (the uniform policy)."
uniform_schedule(h::Real, n::Int) = fill(Float64(h), n)

"""
    geometric_schedule(h0, q, hmax, n) -> Vector{Float64}

`dtau_k = min(hmax, h0 q^k)`, `k = 0..n-1`: the logarithmically expanding policy. `q` is an
experiment-level schedule parameter, not an accuracy tolerance.
"""
geometric_schedule(h0::Real, q::Real, hmax::Real, n::Int) =
    [min(Float64(hmax), h0 * q^k) for k = 0:(n-1)]
