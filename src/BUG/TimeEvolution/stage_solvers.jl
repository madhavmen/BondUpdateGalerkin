# Implicit Runge-Kutta stage equations in the root space of a BUG step.
#
# The root space is spanned by the Galerkin frames of a step: left isometries W and right isometries
# Z, so a root matrix Y (r_L x r_R) is the state W Y Z. The stage solver only needs the frame
# interface below; a state representation (the dense QTT cores of QTTS) implements it. Stage
# equations Y_j = S + h sum_l A_jl G(t + c_l h, Y_l), with G(Y) = W' F(t, W Y Z) Z' the reduced
# field, are solved by Newton-GMRES on the doubled-real unknowns, since F is only real-linear
# (`stage_gmres.jl`).

"""
    frame_state(fr, Y) -> state

The state `W Y Z` of root matrix `Y` in the frames `fr`.
"""
function frame_state end

"""
    frame_project(fr, H, C) -> Matrix

The root matrix `W' H C Z'` of the Galerkin projection of operator `H` (`nothing` = identity)
applied to the state `C`, by environments: `H C` is never formed.
"""
function frame_project end

"""
    frame_operator_matrix(fr, H) -> Matrix

The root-space matrix of `W' H Z` in `vec(Y)` order (row index of `Y` fastest).
"""
function frame_operator_matrix end

"""
    frame_conj(fr, C) -> state

The complex conjugate of state `C`, in the representation of the frames `fr` (the `conj(d)` term
of a real-linear Jacobian).
"""
function frame_conj end

"""
    frame_root_envs(fr, H; conj_ket = false) -> envs

The environments of `W' H Z` at the root cut, from which the projected action, the root-space
matrix ([`root_envs_matrix`](@ref)) and the nearest Kronecker sum ([`root_envs_kron_sum`](@ref))
follow without another sweep over the sites.
"""
function frame_root_envs end

"""
    root_envs_matrix(envs) -> Matrix

The root-space matrix of the environments `envs` of [`frame_root_envs`](@ref), in `vec(Y)` order;
the dense stage preconditioner factorises it.
"""
function root_envs_matrix end

"""
    root_envs_kron_sum(envs) -> (Ml, Mr)

The Kronecker sum `Ml (x) I + I (x) Mr` nearest (Frobenius) to the root-space operator of `envs`,
acting on a root `Y` as `Ml Y + Y transpose(Mr)`: the separable stage preconditioner inverts it by
Sylvester solves ([`_separable_precond`](@ref)).
"""
function root_envs_kron_sum end

"""
    stage_field_ops(fl, fr; density, warm, jacobian_psi2) -> ops

The stage solve's field and Jacobian on fixed frames `fr`, by root environments per summand of the
flow's operator (QTTS: `StageFieldOps`). [`stage_field`](@ref), [`stage_jacobian`](@ref) and
[`stage_linear_envs`](@ref) evaluate them; `ops.t_rho`, `ops.env_builds` and `ops.psi2_fits`
count the density seconds, the environment sets built and the Jacobians' `psi^2` fits.
"""
function stage_field_ops end

"""
    stage_field(ops, Y, field_time, slot) -> (G, mu, info, aux)

The reduced field `G(Y)` of stage `slot` at `field_time`, its `mu`, the operator info and `aux`,
what [`stage_jacobian`](@ref) at the same `Y` reuses.
"""
function stage_field end

"""
    stage_jacobian(ops, Y, field_time, mu, aux, slot) -> (V -> DG(Y)[V])

The real-linear Jacobian of [`stage_field`](@ref) at the same `Y`, with the normalised flow's
`D mu` term.
"""
function stage_jacobian end

"""
    stage_linear_envs(ops, Y, field_time) -> envs

Root environments of the Jacobian's linear part `J1` at `Y`, the stage preconditioner's frozen
operator.
"""
function stage_linear_envs end

"""
    reduced_field(fl, fr, root_matrix, field_time) -> (G, mu, info)

The reduced vector field `G(Y) = W' F(t, W Y Z) Z'` at `Y = root_matrix`, `t = field_time`, the
subtracted `mu[psi]` (zero unless `flow_normalized(fl)`) and the flow's operator info. Every call
contracts the whole operator over the sites (`rk_stage_solve(...; envs = :full)`).
"""
function reduced_field(fl::QTTFlow, fr, root_matrix, field_time)
    state = frame_state(fr, root_matrix)
    field_operator, info = flow_op(fl, state, field_time)
    projected = frame_project(fr, field_operator, state)
    # Rayleigh quotient of the projected field: the shift that keeps a normalised flow's norm
    mu =
        flow_normalized(fl) ?
        real(dot(root_matrix, projected)) / max(real(dot(root_matrix, root_matrix)), eps()) : 0.0
    return fl.pref .* (projected .- mu .* root_matrix), mu, info
end

"""
    reduced_jacobian(fl, fr, root_matrix, field_time, mu; rho = nothing) -> Function

The reduced real-linear Jacobian at `Y = root_matrix` as a closure
`direction_matrix -> DG(Y)[direction_matrix]`. `dmu` is left out for a normalised flow: a
Newton-like solve whose residual is still exact. `rho` is the density of the same state, which the
field already built (see [`flow_jac_ops`](@ref)).
"""
function reduced_jacobian(fl::QTTFlow, fr, root_matrix, field_time, mu; rho = nothing)
    # `linear_jacobian` acts on the direction, `conjugate_jacobian` (or `nothing`) on its conjugate
    linear_jacobian, conjugate_jacobian =
        flow_jac_ops(fl, frame_state(fr, root_matrix), field_time; rho = rho)
    return function (direction_matrix)
        direction = frame_state(fr, direction_matrix)
        out = frame_project(fr, linear_jacobian, direction) .- mu .* direction_matrix
        conjugate_jacobian === nothing ||
            (out = out .+ frame_project(fr, conjugate_jacobian, frame_conj(fr, direction)))
        return fl.pref .* out
    end
end

"""
    RK_TABLEAUX

Butcher matrices `A` and nodes `c` of the root methods: `:trapezoid` (order 2, A-stable, not
L-stable; stage 1 explicit), `:radau2` (2-stage Radau IIA, order 3, L-stable), `:radau3` (3-stage
Radau IIA, order 5, L-stable). All three are stiffly accurate, so `S_{n+1}` is the last stage.
"""
const RK_TABLEAUX = let sqrt_six = sqrt(6.0)
    Dict(
        :trapezoid => (A = [0.0 0.0; 0.5 0.5], c = [0.0, 1.0]),
        :radau2 => (A = [5/12 -1/12; 3/4 1/4], c = [1 / 3, 1.0]),
        :radau3 => (
            A = [
                (88-7sqrt_six)/360 (296-169sqrt_six)/1800 (-2+3sqrt_six)/225
                (296+169sqrt_six)/1800 (88+7sqrt_six)/360 (-2-3sqrt_six)/225
                (16-sqrt_six)/36 (16+sqrt_six)/36 1/9
            ],
            c = [(4 - sqrt_six) / 10, (4 + sqrt_six) / 10, 1.0],
        ),
    )
end

"The root method of nonlinear BUG `order`: 2 trapezoid, 3 Radau IIA (2 stages), 4 Radau IIA (3)."
qtt_root_for_order(order::Int) =
    order == 2 ? :trapezoid :
    order == 3 ? :radau2 :
    order == 4 ? :radau3 : throw(ArgumentError("nonlinear BUG orders are 2, 3, 4; got $order"))

"""
Largest stacked stage dimension `s r_L r_R` for which `precond = :auto` factorises the dense
preconditioner; above it that mode runs GMRES unpreconditioned (and says so in `stats.precond`).
"""
const _STAGE_PRECOND_MAX = 3072

"""
Largest inexact-Newton forcing term of `rk_stage_solve` (GMRES tolerance relative to the
residual): each Newton iteration then removes at least ~90% of the linearised residual, so the
stall exit (`newton_stall`) and `maxnewton` keep their meaning.
"""
const _NEWTON_FORCING_MAX = 0.1

"""
    rk_stage_solve(fl, fr, root_start, start_time, step_size, method; tol = 1e-10,
                   maxnewton = 20, gmres_tol = nothing, gmres_maxiter = 600, Y0 = nothing,
                   precond = :separable, envs = :summand, density = :variational,
                   warm = Dict(), gmres_recycle = 0, newton_stall = 0.5,
                   jacobian_psi2 = :fresh) -> (stage_values, stats)

The nonlinear stage equations `Y_j = S + h sum_l A_jl G(t + c_l h, Y_l)` of `RK_TABLEAUX[method]`
(`S = root_start`, `t = start_time`, `h = step_size`, `Y_j` the returned `stage_values`) in the
root space of frames `fr`, by Newton-GMRES on the doubled-real unknowns with the reduced
real-linear Jacobian. The residual is the full nonlinear one, driven to `tol` relative to `||S||`.
`Y0` are starting stage values (default: `S` in every stage).

GMRES is right-preconditioned with `I - h A (x) pref (M - mu)`, `M` the frozen linear part `W' J1 Z`
of the Jacobian at the step input (`conj(d)` is left to GMRES):
- `precond = :separable` (default): `M` replaced by its nearest Kronecker sum
  ([`root_envs_kron_sum`](@ref)) and inverted by Sylvester solves, with no root-space matrix. It is
  exact for operators that are Kronecker sums across the root bond, at every grid size;
- `:dense`: the full `M` factorised as an `s r_L r_R` matrix; `:auto`: that only when
  `s r_L r_R <= _STAGE_PRECOND_MAX`, else none;
- `:none`: plain GMRES.

`envs = :summand` (default) evaluates fields and Jacobians through [`stage_field_ops`](@ref): root
environments per summand, the density fitted per `density` (`:variational`, warm-started from
`warm`, which the caller may carry across a step's root solves, or `:zipup`), `jacobian_psi2` how
the Jacobian's `psi^2` is refitted (`:fresh` or `:lagged`), and the normalised flow's `D mu` in the
Jacobian. `envs = :full` is [`reduced_field`](@ref) / [`reduced_jacobian`](@ref): every
application contracts the whole operator over the sites, without `D mu`.

`gmres_tol` is GMRES's tolerance relative to the current residual. `nothing` (default) is inexact
Newton: the Eisenstat-Walker forcing term (choice 2, `0.9 (res_k / res_{k-1})^2`, safeguarded, at
most `_NEWTON_FORCING_MAX`), never below `0.5 tol / res_k`, so the linear solve is only as accurate
as the next Newton iterate needs. A number fixes it at every Newton iteration.
`gmres_recycle = k > 0` carries `k` recycled directions across this call's Newton solves
([`GMRESRecycle`](@ref)); `0` is plain GMRES. Newton stops unconverged once the residual is above
`newton_stall` times its value three iterations earlier (`Inf`: run to `maxnewton`).

`stats` records Newton and GMRES iterations (with recycling, including the `recycled` products
that re-image the directions), the GMRES seconds `t_gmres`, the final residual, `converged`, the
field evaluations `nfield`, the density seconds `t_rho`, the last `mu`, the preconditioner, `envs`,
the `forcing` rule, `env_builds` and `psi2_fits` (`-1` under `envs = :full`), and `fields`, the
stage fields `G(Y_j)` at the returned stages.
"""
function rk_stage_solve(
    fl::QTTFlow,
    fr,
    root_start::AbstractMatrix,
    start_time::Real,
    step_size::Real,
    method::Symbol;
    tol::Real = 1e-10,
    maxnewton::Int = 20,
    gmres_tol::Union{Nothing,Real} = nothing,
    gmres_maxiter::Int = 600,
    Y0 = nothing,
    precond::Symbol = :separable,
    envs::Symbol = :summand,
    density::Symbol = :variational,
    warm::Dict = Dict{Any,Any}(),
    gmres_recycle::Int = 0,
    newton_stall::Real = 0.5,
    jacobian_psi2::Symbol = :fresh,
)
    precond in (:separable, :dense, :auto, :none) ||
        throw(ArgumentError("precond must be :separable, :dense, :auto or :none, got :$precond"))
    envs in (:summand, :full) || throw(ArgumentError("envs must be :summand or :full, got :$envs"))
    tableau = RK_TABLEAUX[method]
    nstages = length(tableau.c)
    butcher = tableau.A
    stage_times = [start_time + node * step_size for node in tableau.c]
    stage_values =
        Y0 === nothing ? [ComplexF64.(root_start) for _ = 1:nstages] :
        [ComplexF64.(start_value) for start_value in Y0]
    dims = [size(root_start) for _ = 1:nstages]
    start_norm = max(norm(root_start), eps())
    nfield, ngmres, t_rho, t_gmres, residual_norm = 0, 0, 0.0, 0.0, Inf
    # one recycle space per call: the frames (and so the unknowns' space) change between root solves
    recycle = gmres_recycle > 0 ? GMRESRecycle(gmres_recycle) : nothing
    ops =
        envs === :summand ?
        stage_field_ops(fl, fr; density = density, warm = warm, jacobian_psi2 = jacobian_psi2) :
        nothing
    apply_preconditioner, precond_name =
        _stage_preconditioner(fl, fr, ops, root_start, start_time, step_size, butcher, precond)
    residual_history = Float64[]           # residual per Newton iteration, for the stall exit
    forcing = _NEWTON_FORCING_MAX          # the inexact-Newton GMRES tolerance of this iteration
    for newton = 0:maxnewton
        fields, mus = Matrix{ComplexF64}[], Float64[]
        densities = Any[]                  # :full -- each stage's density, reused by its Jacobian
        auxes = Any[]                      # :summand -- what each stage's Jacobian reuses
        for stage = 1:nstages
            if ops === nothing
                field, mu, info = reduced_field(fl, fr, stage_values[stage], stage_times[stage])
                push!(densities, get(info, :rho, nothing))
                t_rho += info.t_rho
            else
                field, mu, _, aux = stage_field(ops, stage_values[stage], stage_times[stage], stage)
                push!(auxes, aux)
                t_rho = ops.t_rho
            end
            push!(fields, field)
            push!(mus, mu)
            nfield += 1
        end
        # the residual of each stage equation, Y_j - S - h sum_l A_jl G_l
        residuals = [
            stage_values[equation] .- root_start .-
            step_size .* sum(butcher[equation, stage] .* fields[stage] for stage = 1:nstages)
            for equation = 1:nstages
        ]
        residual_norm =
            sqrt(sum(norm(stage_residual)^2 for stage_residual in residuals)) / start_norm
        push!(residual_history, residual_norm)
        # a stalled Newton gives up: the residual did not fall below `newton_stall` times its value
        # three iterations earlier (the caller rejects the step)
        stalled = newton >= 3 && residual_norm > newton_stall * residual_history[end-3]
        if residual_norm <= tol || newton == maxnewton || stalled
            return stage_values,
            (
                newton = newton,
                gmres = ngmres,
                residual = residual_norm,
                converged = residual_norm <= tol,
                nfield = nfield,
                t_rho = t_rho,
                t_gmres = t_gmres,
                mu = mus[end],
                precond = precond_name,
                envs = envs,
                forcing = gmres_tol === nothing ? :eisenstat_walker : :fixed,
                env_builds = ops === nothing ? -1 : ops.env_builds,
                psi2_fits = ops === nothing ? -1 : ops.psi2_fits,
                recycled = recycle === nothing ? 0 : recycle.extra_products,
                fields = fields,
            )
        end
        # the Jacobian at the same stage values: its density is the one the field just built
        jacobians =
            ops === nothing ?
            [
                reduced_jacobian(
                    fl,
                    fr,
                    stage_values[stage],
                    stage_times[stage],
                    mus[stage];
                    rho = densities[stage],
                ) for stage = 1:nstages
            ] :
            [
                stage_jacobian(
                    ops,
                    stage_values[stage],
                    stage_times[stage],
                    mus[stage],
                    auxes[stage],
                    stage,
                ) for stage = 1:nstages
            ]
        # the stage system's Jacobian on packed directions; an all-zero column (the trapezoid's
        # explicit first stage) needs no Jacobian action
        function newton_operator(packed)
            directions = _unpack(packed, dims)
            actions = [
                any(!iszero, butcher[:, stage]) ? jacobians[stage](directions[stage]) :
                zero(directions[stage]) for stage = 1:nstages
            ]
            return _pack([
                directions[equation] .-
                step_size .* sum(butcher[equation, stage] .* actions[stage] for stage = 1:nstages)
                for equation = 1:nstages
            ])
        end
        # Eisenstat-Walker choice 2 once a previous residual exists; its safeguard keeps a large
        # previous forcing term from collapsing in one iteration
        if gmres_tol === nothing && newton >= 1
            walker = 0.9 * (residual_norm / residual_history[end-1])^2
            safeguard = 0.9 * forcing^2
            forcing = min(_NEWTON_FORCING_MAX, safeguard > 0.1 ? max(walker, safeguard) : walker)
        end
        # never solve the linear system past what the next residual needs to reach `tol`
        linear_tol =
            gmres_tol === nothing ? max(forcing, 0.5 * tol / residual_norm) : Float64(gmres_tol)
        # right preconditioning: solve (op o P^-1) y = -R, then dx = P^-1 y; the preconditioner and
        # the frames are fixed within this call, so recycled directions stay in the same space
        preconditioned(packed) = newton_operator(apply_preconditioner(packed))
        gmres_start = time_ns()
        dy, iterations, _ =
            recycle === nothing ?
            _gmres_real(
                preconditioned,
                -_pack(residuals);
                tol = Float64(linear_tol),
                maxiter = gmres_maxiter,
            ) :
            _gcro_real(
                preconditioned,
                -_pack(residuals),
                recycle;
                tol = Float64(linear_tol),
                maxiter = gmres_maxiter,
            )
        t_gmres += (time_ns() - gmres_start) / 1e9
        ngmres += iterations
        stage_values = [
            stage_values[stage] .+ correction for
            (stage, correction) in enumerate(_unpack(apply_preconditioner(dy), dims))
        ]
    end
end

"""
    _stage_preconditioner(fl, fr, ops, root_start, start_time, step_size, butcher, precond)
        -> (packed -> P^-1 packed, name)

The right preconditioner of [`rk_stage_solve`](@ref) on packed stage vectors, from the Jacobian's
linear part `J1` at the step input and midpoint time, with the normalised flow's shift `mu` frozen
at `root_start`; `name` is `:separable`, `:linear_part` (dense) or `:none`.
"""
function _stage_preconditioner(
    fl::QTTFlow,
    fr,
    ops,
    root_start::AbstractMatrix,
    start_time::Real,
    step_size::Real,
    butcher::AbstractMatrix,
    precond::Symbol,
)
    nstages, root_size = size(butcher, 1), length(root_start)
    dims = [size(root_start) for _ = 1:nstages]
    identity_map = (packed -> packed, :none)
    (precond === :none || (precond === :auto && nstages * root_size > _STAGE_PRECOND_MAX)) &&
        return identity_map
    midpoint = start_time + step_size / 2
    linear_envs =
        ops === nothing ?
        frame_root_envs(fr, first(flow_jac_ops(fl, frame_state(fr, root_start), midpoint))) :
        stage_linear_envs(ops, root_start, midpoint)
    start_norm_squared = max(norm(root_start)^2, eps())
    if precond === :separable
        Ml, Mr = root_envs_kron_sum(linear_envs)
        # the normalised flow's mu[psi_n] = <S, M S> / <S, S>, with M by its separable part
        mu0 =
            flow_normalized(fl) ?
            real(dot(root_start, Ml * root_start .+ root_start * transpose(Mr))) /
            start_norm_squared : 0.0
        solve = _separable_precond(Ml - mu0 * I, Mr, fl.pref, butcher, step_size)
        return (packed -> _pack(solve(_unpack(packed, dims)))), :separable
    end
    linear_part = root_envs_matrix(linear_envs)
    mu0 =
        flow_normalized(fl) ?
        real(dot(vec(root_start), linear_part * vec(root_start))) / start_norm_squared : 0.0
    stage_jacobian_matrix = fl.pref .* (linear_part - mu0 * I)
    factorised = lu(
        Matrix{ComplexF64}(I, nstages * root_size, nstages * root_size) .-
        step_size .* kron(butcher, stage_jacobian_matrix),
    )
    function apply_dense(packed)
        stacked = vcat((vec(block) for block in _unpack(packed, dims))...)
        solved = factorised \ stacked
        return _pack([
            reshape(solved[((stage-1)*root_size+1):(stage*root_size)], dims[stage]) for
            stage = 1:nstages
        ])
    end
    return apply_dense, :linear_part
end
