# The stage solve's vector field and Jacobian on fixed QTT frames, by root environments per summand.
#
# The stage solver (`BUG/TimeEvolution/stage_solvers.jl`) calls `stage_field_ops`, `stage_field`,
# `stage_jacobian` and `stage_linear_envs`; these methods implement them for dense QTT frames. A
# flow describes its summands through `_static_operator`, `_timed_operator`, `_density_coupling`
# and `_density_tol` (methods for `LinearFlow` here, for `GPFlow` beside the GP model).

"The state-independent operator a flow projects once per frame set (`LinearFlow`: its `H`)."
_static_operator(fl::LinearFlow) = fl.H
"The time-dependent operator projected once per stage time, or `nothing` (`LinearFlow`: none)."
_timed_operator(::LinearFlow, field_time) = nothing
"The coupling `g` of the flow's density term `g diag(|psi|^2)`; `0` for a linear flow."
_density_coupling(::LinearFlow) = 0.0
"The relative compression tolerance of the flow's density."
_density_tol(::LinearFlow) = 0.0
"""
`flow_op(fl, state_cores, field_time)` with the state's `density` already fitted (`nothing`: build
it); a flow without a density term ignores it.
"""
_flow_op_at(fl::QTTFlow, state_cores, field_time, density) = flow_op(fl, state_cores, field_time)

"""
    StageFieldOps(fl, fr; density = :variational, warm = Dict(), jacobian_psi2 = :fresh)

The stage solve's vector field and Jacobian on fixed frames, one root-environment set per summand:
the static part (kinetic, trap, rotation, `-mu`) or a linear flow's `H` is projected once per frame
set, the time-dependent part (the paddle) once per stage time, and the density `g diag(rho_j)` once
per stage iterate -- rebuilt from that iterate's `|Y_j|^2`, never reused across iterates. Field and
GMRES applications are then root-matrix products ([`root_apply`](@ref)), with no sweep over the
sites (the full path re-contracts the whole operator for every field and every GMRES application:
~170 s of a 250 s L = 16 root solve).

`density = :variational` fits `rho_j` and `psi_j^2` by [`tt_hadamard_fit`](@ref) warm-started from
the same stage slot's previous fit (`warm`, which the caller may carry across a step's root solves;
the density does not depend on the frames), and by the zip-up when there is none yet; `:zipup`
always zips up. `t_rho` counts every density second, `env_builds` the root-environment sets built,
`psi2_fits` the Jacobians' `psi^2` fits. `jacobian_psi2` is how the Jacobian's `g psi^2 conj(V)`
term gets its `psi^2`: `:fresh` (default) fits it at every Newton iterate; `:lagged` fits it at a
stage slot's first Jacobian of the root solve and keeps its environments for the later iterates
(a Jacobian is a linearisation inside Newton-Krylov; the residual still rebuilds `rho_j` at every
iterate, so the converged stages are unchanged).
"""
mutable struct StageFieldOps
    fl::QTTFlow
    fr::QTTFrames
    static::Any                        # root envs of the static operator
    timed::Dict{Float64,Any}           # stage time -> root envs of the timed operator, or `nothing`
    warm::Dict{Any,Any}                # (slot, :rho | :psi2) -> last fitted density / psi^2
    density::Symbol
    t_rho::Float64
    env_builds::Int
    jacobian_psi2::Symbol
    psi2_envs::Dict{Any,Any}           # slot -> root envs of diag(psi^2), kept under :lagged
    psi2_fits::Int
end

function StageFieldOps(
    fl::QTTFlow,
    fr::QTTFrames;
    density::Symbol = :variational,
    warm::Dict = Dict{Any,Any}(),
    jacobian_psi2::Symbol = :fresh,
)
    density in (:variational, :zipup) ||
        throw(ArgumentError("density must be :variational or :zipup, got :$density"))
    jacobian_psi2 in (:fresh, :lagged) ||
        throw(ArgumentError("jacobian_psi2 must be :fresh or :lagged, got :$jacobian_psi2"))
    static = frame_root_envs(fr, _static_operator(fl))
    return StageFieldOps(
        fl,
        fr,
        static,
        Dict{Float64,Any}(),
        warm,
        density,
        0.0,
        1,
        jacobian_psi2,
        Dict{Any,Any}(),
        0,
    )
end

stage_field_ops(fl::QTTFlow, fr::QTTFrames; kwargs...) = StageFieldOps(fl, fr; kwargs...)

"`frame_root_envs` counted in `ops.env_builds`."
_counted_envs(ops::StageFieldOps, H; kwargs...) =
    (ops.env_builds += 1; frame_root_envs(ops.fr, H; kwargs...))

"Whether the flow carries a density term (its coupling is nonzero)."
_nonlinear(ops::StageFieldOps) = _density_coupling(ops.fl) != 0

"Root environments of the state-independent part at `field_time` (static + timed)."
function _linear_envs(ops::StageFieldOps, field_time::Real)
    timed = get!(ops.timed, Float64(field_time)) do
        operator = _timed_operator(ops.fl, field_time)
        operator === nothing ? nothing : _counted_envs(ops, operator)
    end
    return timed === nothing ? ops.static : _envs_cat([(1.0, ops.static), (1.0, timed)])
end

"`A .* B` for the stage iterate in `slot` (`key` :rho or :psi2), warm-started when possible."
function _stage_product(ops::StageFieldOps, A, B, slot, key)
    tol = _density_tol(ops.fl)
    init = get(ops.warm, (slot, key), nothing)
    product, info =
        ops.density === :variational && init !== nothing ? tt_hadamard_fit(A, B, init; tol = tol) :
        (key === :rho ? tt_density(A; tol = tol) : tt_product(A, A; tol = tol))
    ops.warm[(slot, key)] = product
    ops.t_rho += info.seconds
    return product, info
end

"""
    stage_field(ops, Y, field_time, slot) -> (G, mu, info, aux)

`G(Y) = pref (W' H[rho(Y), t] W Y Z Z' - mu Y)` by root environments; `aux` carries what the
Jacobian at the same `Y` reuses (the density environments, `J1 Y`, the stage state).
"""
function stage_field(ops::StageFieldOps, Y::AbstractMatrix, field_time::Real, slot)
    fl = ops.fl
    linear = _linear_envs(ops, field_time)
    HY = root_apply(linear, Y)
    J1Y = HY                                          # (H_lin + 2 g rho) Y, for the normalised D mu
    rho_envs, state = nothing, nothing
    info = (chi_rho = 0, t_rho = 0.0, w = size(linear[1], 2))
    if _nonlinear(ops)
        coupling = _density_coupling(fl)
        state = frame_state(ops.fr, Y)
        rho, density_info = _stage_product(ops, state, tt_conj(state), slot, :rho)
        rho_envs = _counted_envs(ops, op_diag(rho))
        rhoY = root_apply(rho_envs, Y)
        HY = HY .+ coupling .* rhoY
        J1Y = HY .+ coupling .* rhoY
        info = (
            chi_rho = density_info.chi,
            t_rho = density_info.seconds,
            w = size(linear[1], 2) + size(rho_envs[1], 2),
        )
    end
    mu = flow_normalized(fl) ? real(dot(Y, HY)) / max(real(dot(Y, Y)), eps()) : 0.0
    aux = (linear = linear, rho_envs = rho_envs, J1Y = J1Y, state = state)
    return fl.pref .* (HY .- mu .* Y), mu, info, aux
end

"""
    stage_jacobian(ops, Y, field_time, mu, aux, slot) -> (V -> DG(Y)[V])

The real-linear Jacobian of [`stage_field`](@ref) at the same `Y`:
`pref [(H_lin + 2 g rho - mu) V + g psi^2 conj(V) - (D mu[V]) Y]`, the last term for the
normalised flow, `D mu[V] = 2 Re<V, J1 Y - mu Y> / <Y, Y>` (full Newton: without it Newton
diverged at `h mu ~ 1`, a trapped g = 100 normalised flow unconverged above h ~ 0.2). `psi^2` is
fitted once per iterate (warm-started like the density), or once per root solve under
`ops.jacobian_psi2 = :lagged`.
"""
function stage_jacobian(
    ops::StageFieldOps,
    Y::AbstractMatrix,
    field_time::Real,
    mu::Real,
    aux,
    slot,
)
    fl = ops.fl
    nonlinear = aux.rho_envs !== nothing
    coupling = nonlinear ? _density_coupling(fl) : 0.0
    J1 = nonlinear ? _envs_cat([(1.0, aux.linear), (2coupling, aux.rho_envs)]) : aux.linear
    J2 = nonlinear ? _psi2_envs(ops, aux.state, slot) : nothing
    normalized = flow_normalized(fl)
    y_norm_squared = max(real(dot(Y, Y)), eps())
    mu_direction = aux.J1Y .- mu .* Y                 # D mu[V] = 2 Re<V, mu_direction> / <Y, Y>
    return function (V)
        out = root_apply(J1, V) .- mu .* V
        nonlinear && (out = out .+ coupling .* root_apply(J2, conj(V)))
        normalized && (out = out .- (2 * real(dot(mu_direction, V)) / y_norm_squared) .* Y)
        return fl.pref .* out
    end
end

"""
Root environments of `diag(psi^2)` (conjugated ket) for the Jacobian at stage `slot`'s iterate
`state`: fitted now, or under `:lagged` the ones this root solve already built for the slot.
"""
function _psi2_envs(ops::StageFieldOps, state, slot)
    ops.jacobian_psi2 === :lagged && haskey(ops.psi2_envs, slot) && return ops.psi2_envs[slot]
    psi_squared, _ = _stage_product(ops, state, state, slot, :psi2)
    ops.psi2_fits += 1
    envs = _counted_envs(ops, op_diag(psi_squared); conj_ket = true)
    ops.psi2_envs[slot] = envs
    return envs
end

"""
    stage_linear_envs(ops, Y, field_time) -> envs

Root environments of `J1 = H_lin(t) + 2 g rho(Y)`, the stage preconditioner's frozen linear part.
A density already in the `:precond` slot (`qtt_bug_step!` puts `psi_n`'s there) is used as it is:
`Y` is the step input projected onto frames that contain it, and the preconditioner is approximate
anyway, so refitting it would be one density fit per root solve for no change.
"""
function stage_linear_envs(ops::StageFieldOps, Y::AbstractMatrix, field_time::Real)
    linear = _linear_envs(ops, field_time)
    _nonlinear(ops) || return linear
    rho = get(ops.warm, (:precond, :rho), nothing)
    if rho === nothing
        state = frame_state(ops.fr, Y)
        rho, _ = _stage_product(ops, state, tt_conj(state), :precond, :rho)
    end
    density_envs = _counted_envs(ops, op_diag(rho))
    return _envs_cat([(1.0, linear), (2 * _density_coupling(ops.fl), density_envs)])
end
