# Implicit Runge-Kutta stage equations in the root space of a BUG step.
#
# Moved from BUG-Julia@b155550:src/RSVDCBEBondUpdate/qtt/qtt_flows.jl (`reduced_field`,
# `reduced_jacobian`, `RK_TABLEAUX`, `rk_stage_solve`, `_gmres_real`, `_pack`, `_unpack`).
#
# The root space is spanned by the Galerkin frames of a step: left isometries W and right isometries
# Z, so a root matrix Y (r_L x r_R) is the state W Y Z. The stage solver only needs the frame
# interface below; a state representation (the dense QTT cores of QTTS) implements it. Stage
# equations Y_j = S + h sum_l A_jl G(t + c_l h, Y_l), with G(Y) = W' F(t, W Y Z) Z' the reduced
# field, are solved by Newton-GMRES on the doubled-real unknowns, since F is only real-linear.

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

The root-space matrix of `W' H Z` in `vec(Y)` order (row index of `Y` fastest), for the stage
preconditioner.
"""
function frame_operator_matrix end

"""
    frame_conj(fr, C) -> state

The complex conjugate of state `C`, in the representation of the frames `fr` (the `conj(d)` term
of a real-linear Jacobian).
"""
function frame_conj end

"""
    reduced_field(fl, fr, root_matrix, field_time) -> (G, mu, info)

The reduced vector field `G(Y) = W' F(t, W Y Z) Z'` at `Y = root_matrix`, `t = field_time`, the
subtracted `mu[psi]` (zero unless `flow_normalized(fl)`) and the flow's operator info.
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
    reduced_jacobian(fl, fr, root_matrix, field_time, mu) -> Function

The reduced real-linear Jacobian at `Y = root_matrix` as a closure
`direction_matrix -> DG(Y)[direction_matrix]`. `dmu` is left out for a normalised flow: a
Newton-like solve whose residual is still exact.
"""
function reduced_jacobian(fl::QTTFlow, fr, root_matrix, field_time, mu)
    # `linear_jacobian` acts on the direction, `conjugate_jacobian` (or `nothing`) on its conjugate
    linear_jacobian, conjugate_jacobian = flow_jac_ops(fl, frame_state(fr, root_matrix), field_time)
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
    _gmres_real(apply, target; tol = 1e-10, restart = 40, maxiter = 400)
        -> (solution, iterations, relres)

Restarted GMRES with Givens rotations on real vectors (the doubled-real form of a real-linear
complex map) for `apply(solution) = target`. Stops when the residual falls below `tol ||target||`.
"""
function _gmres_real(
    apply,
    target::Vector{Float64};
    tol::Float64 = 1e-10,
    restart::Int = 40,
    maxiter::Int = 400,
)
    dimension = length(target)
    solution = zeros(dimension)
    target_norm = norm(target)
    target_norm == 0 && return solution, 0, 0.0
    iterations = 0
    while iterations < maxiter
        residual = target .- apply(solution)
        beta = norm(residual)
        beta <= tol * target_norm && return solution, iterations, beta / target_norm
        cycle = min(restart, maxiter - iterations)
        basis = zeros(dimension, cycle + 1)
        hessenberg = zeros(cycle + 1, cycle)
        cosines, sines, rhs = zeros(cycle), zeros(cycle), zeros(cycle + 1)
        basis[:, 1] .= residual ./ beta
        rhs[1] = beta
        used = 0
        stalled = false
        for column = 1:cycle
            iterations += 1
            image = apply(basis[:, column])
            # modified Gram-Schmidt against the Krylov basis
            for row = 1:column
                hessenberg[row, column] = dot(view(basis, :, row), image)
                image .-= hessenberg[row, column] .* view(basis, :, row)
            end
            hessenberg[column+1, column] = norm(image)
            hessenberg[column+1, column] > 0 &&
                (basis[:, column+1] .= image ./ hessenberg[column+1, column])
            # apply the earlier rotations to the new column, then rotate its subdiagonal away
            for row = 1:(column-1)
                rotated =
                    cosines[row] * hessenberg[row, column] + sines[row] * hessenberg[row+1, column]
                hessenberg[row+1, column] =
                    -sines[row] * hessenberg[row, column] + cosines[row] * hessenberg[row+1, column]
                hessenberg[row, column] = rotated
            end
            radius = hypot(hessenberg[column, column], hessenberg[column+1, column])
            # The Krylov space closed (zero subdiagonal) on a direction the operator maps into the
            # earlier ones: the new column adds nothing, and keeping it would make the triangular
            # solve singular. The solution in the first column - 1 vectors is the best this space
            # holds.
            if radius == 0
                stalled = true
                break
            end
            used = column
            cosines[column] = hessenberg[column, column] / radius
            sines[column] = hessenberg[column+1, column] / radius
            hessenberg[column, column], hessenberg[column+1, column] = radius, 0.0
            rhs[column+1] = -sines[column] * rhs[column]
            rhs[column] = cosines[column] * rhs[column]
            abs(rhs[column+1]) <= tol * target_norm && break
        end
        if used > 0
            coefficients = UpperTriangular(hessenberg[1:used, 1:used]) \ rhs[1:used]
            solution .+= basis[:, 1:used] * coefficients
        end
        # a stalled space cannot improve on restart: return what it holds
        (stalled || abs(rhs[used+1]) <= tol * target_norm) &&
            return solution, iterations, abs(rhs[used+1]) / target_norm
    end
    return solution, iterations, norm(target .- apply(solution)) / target_norm
end

"Stack complex matrices as one real vector: real parts, then imaginary parts, matrix by matrix."
_pack(matrices) = vcat((vcat(vec(real(matrix)), vec(imag(matrix))) for matrix in matrices)...)

"The inverse of `_pack` for matrices of sizes `dims`."
function _unpack(packed::Vector{Float64}, dims)
    out = Matrix{ComplexF64}[]
    offset = 0
    for block_dims in dims
        entry_count = prod(block_dims)
        # this matrix's real parts, then its imaginary parts
        real_part = packed[(offset+1):(offset+entry_count)]
        imag_part = packed[(offset+entry_count+1):(offset+2*entry_count)]
        push!(out, reshape(real_part .+ 1im .* imag_part, block_dims))
        offset += 2 * entry_count
    end
    return out
end

"""
Largest stacked stage dimension `s r_L r_R` for which the dense preconditioner is factorised; above
it the stage GMRES runs unpreconditioned (and says so in `stats.precond`).
"""
const _STAGE_PRECOND_MAX = 3072

"""
    rk_stage_solve(fl, fr, root_start, start_time, step_size, method; tol = 1e-10,
                   maxnewton = 20, gmres_tol = 1e-3 tol, gmres_maxiter = 600, Y0 = nothing,
                   precond = :auto) -> (stage_values, stats)

The nonlinear stage equations `Y_j = S + h sum_l A_jl G(t + c_l h, Y_l)` of `RK_TABLEAUX[method]`
(`S = root_start`, `t = start_time`, `h = step_size`, `Y_j` the returned `stage_values`) in the
root space of frames `fr`, by Newton-GMRES on the doubled-real unknowns with the reduced
real-linear Jacobian. The residual is the full nonlinear one, driven to `tol` relative to `||S||`.
`precond = :auto` right-preconditions GMRES with `I - h A (x) pref (M - mu)`, `M = W' J1 Z` the
linear part of the Jacobian at the step midpoint, factorised once when `s r_L r_R <=
_STAGE_PRECOND_MAX`. `Y0` are starting stage values (default: `S` in every stage). `stats` records
Newton and GMRES iterations, the final residual, `converged`, the field evaluations `nfield`, the
density-build seconds `t_rho`, the last `mu` and the preconditioner used.
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
    gmres_tol::Real = 1e-3 * tol,
    gmres_maxiter::Int = 600,
    Y0 = nothing,
    precond::Symbol = :auto,
)
    tableau = RK_TABLEAUX[method]
    nstages = length(tableau.c)
    butcher = tableau.A
    stage_values =
        Y0 === nothing ? [ComplexF64.(root_start) for _ = 1:nstages] :
        [ComplexF64.(start_value) for start_value in Y0]
    dims = [size(root_start) for _ = 1:nstages]
    start_norm = max(norm(root_start), eps())
    nfield, ngmres, t_rho, residual_norm = 0, 0, 0.0, Inf
    root_size = length(root_start)
    factorised = nothing
    if precond === :auto && nstages * root_size <= _STAGE_PRECOND_MAX
        linear_jacobian, _ =
            flow_jac_ops(fl, frame_state(fr, root_start), start_time + step_size / 2)
        linear_part = frame_operator_matrix(fr, linear_jacobian)
        # the Rayleigh shift `mu` of a normalised flow, frozen at the start value
        mu0 =
            flow_normalized(fl) ?
            real(dot(vec(root_start), linear_part * vec(root_start))) /
            max(norm(root_start)^2, eps()) : 0.0
        stage_jacobian = fl.pref .* (linear_part - mu0 * I)
        factorised = lu(
            Matrix{ComplexF64}(I, nstages * root_size, nstages * root_size) .-
            step_size .* kron(butcher, stage_jacobian),
        )
    end
    # packed -> P^-1 packed on the packed real vector (identity without a preconditioner)
    function apply_preconditioner(packed)
        factorised === nothing && return packed
        stacked = vcat((vec(block) for block in _unpack(packed, dims))...)
        solved = factorised \ stacked
        return _pack([
            reshape(solved[((stage-1)*root_size+1):(stage*root_size)], dims[stage]) for
            stage = 1:nstages
        ])
    end
    for newton = 0:maxnewton
        fields, mus = Matrix{ComplexF64}[], Float64[]
        for stage = 1:nstages
            field, mu, info = reduced_field(
                fl,
                fr,
                stage_values[stage],
                start_time + tableau.c[stage] * step_size,
            )
            push!(fields, field)
            push!(mus, mu)
            nfield += 1
            t_rho += info.t_rho
        end
        # the residual of each stage equation, Y_j - S - h sum_l A_jl G_l
        residuals = [
            stage_values[equation] .- root_start .-
            step_size .* sum(butcher[equation, stage] .* fields[stage] for stage = 1:nstages)
            for equation = 1:nstages
        ]
        residual_norm =
            sqrt(sum(norm(stage_residual)^2 for stage_residual in residuals)) / start_norm
        if residual_norm <= tol || newton == maxnewton
            return stage_values,
            (
                newton = newton,
                gmres = ngmres,
                residual = residual_norm,
                converged = residual_norm <= tol,
                nfield = nfield,
                t_rho = t_rho,
                mu = mus[end],
                precond = factorised === nothing ? :none : :linear_part,
            )
        end
        jacobians = [
            reduced_jacobian(
                fl,
                fr,
                stage_values[stage],
                start_time + tableau.c[stage] * step_size,
                mus[stage],
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
        # right preconditioning: solve (op o P^-1) y = -R, then dx = P^-1 y
        dy, iterations, _ = _gmres_real(
            packed -> newton_operator(apply_preconditioner(packed)),
            -_pack(residuals);
            tol = Float64(gmres_tol),
            maxiter = gmres_maxiter,
        )
        ngmres += iterations
        stage_values = [
            stage_values[stage] .+ correction for
            (stage, correction) in enumerate(_unpack(apply_preconditioner(dy), dims))
        ]
    end
end
