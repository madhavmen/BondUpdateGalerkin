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
    reduced_field(fl, fr, Y, t) -> (G, mu, info)

The reduced vector field `G(Y) = W' F(t, W Y Z) Z'`, the subtracted `mu[psi]` (zero unless
`flow_normalized(fl)`) and the flow's operator info.
"""
function reduced_field(fl::QTTFlow, fr, Y, t)
    state = frame_state(fr, Y)
    H, info = flow_op(fl, state, t)
    projected = frame_project(fr, H, state)
    mu = flow_normalized(fl) ? real(dot(Y, projected)) / max(real(dot(Y, Y)), eps()) : 0.0
    return fl.pref .* (projected .- mu .* Y), mu, info
end

"""
    reduced_jacobian(fl, fr, Y, t, mu) -> (V -> DG(Y)[V])

The reduced real-linear Jacobian at `Y` as a closure. `dmu` is left out for a normalised flow: a
Newton-like solve whose residual is still exact.
"""
function reduced_jacobian(fl::QTTFlow, fr, Y, t, mu)
    J1, J2 = flow_jac_ops(fl, frame_state(fr, Y), t)
    return function (V)
        direction = frame_state(fr, V)
        out = frame_project(fr, J1, direction) .- mu .* V
        J2 === nothing || (out = out .+ frame_project(fr, J2, frame_conj(fr, direction)))
        return fl.pref .* out
    end
end

"""
    RK_TABLEAUX

Butcher matrices `A` and nodes `c` of the root methods: `:trapezoid` (order 2, A-stable, not
L-stable; stage 1 explicit), `:radau2` (2-stage Radau IIA, order 3, L-stable), `:radau3` (3-stage
Radau IIA, order 5, L-stable). All three are stiffly accurate, so `S_{n+1}` is the last stage.
"""
const RK_TABLEAUX = let s6 = sqrt(6.0)
    Dict(
        :trapezoid => (A = [0.0 0.0; 0.5 0.5], c = [0.0, 1.0]),
        :radau2 => (A = [5/12 -1/12; 3/4 1/4], c = [1 / 3, 1.0]),
        :radau3 => (
            A = [
                (88-7s6)/360 (296-169s6)/1800 (-2+3s6)/225
                (296+169s6)/1800 (88+7s6)/360 (-2-3s6)/225
                (16-s6)/36 (16+s6)/36 1/9
            ],
            c = [(4 - s6) / 10, (4 + s6) / 10, 1.0],
        ),
    )
end

"The root method of nonlinear BUG order `p`: 2 trapezoid, 3 Radau IIA (2 stages), 4 Radau IIA (3)."
qtt_root_for_order(p::Int) =
    p == 2 ? :trapezoid :
    p == 3 ? :radau2 :
    p == 4 ? :radau3 : throw(ArgumentError("nonlinear BUG orders are 2, 3, 4; got $p"))

"""
    _gmres_real(apply, b; tol = 1e-10, restart = 40, maxiter = 400) -> (x, iterations, relres)

Restarted GMRES with Givens rotations on real vectors (the doubled-real form of a real-linear
complex map). Stops when the residual falls below `tol ||b||`.
"""
function _gmres_real(
    apply,
    b::Vector{Float64};
    tol::Float64 = 1e-10,
    restart::Int = 40,
    maxiter::Int = 400,
)
    n = length(b)
    x = zeros(n)
    norm_b = norm(b)
    norm_b == 0 && return x, 0, 0.0
    iterations = 0
    while iterations < maxiter
        residual = b .- apply(x)
        beta = norm(residual)
        beta <= tol * norm_b && return x, iterations, beta / norm_b
        cycle = min(restart, maxiter - iterations)
        basis = zeros(n, cycle + 1)
        hessenberg = zeros(cycle + 1, cycle)
        cosines, sines, rhs = zeros(cycle), zeros(cycle), zeros(cycle + 1)
        basis[:, 1] .= residual ./ beta
        rhs[1] = beta
        used = 0
        for j = 1:cycle
            iterations += 1
            used = j
            w = apply(basis[:, j])
            # modified Gram-Schmidt against the Krylov basis
            for i = 1:j
                hessenberg[i, j] = dot(view(basis, :, i), w)
                w .-= hessenberg[i, j] .* view(basis, :, i)
            end
            hessenberg[j+1, j] = norm(w)
            hessenberg[j+1, j] > 0 && (basis[:, j+1] .= w ./ hessenberg[j+1, j])
            # apply the earlier rotations to the new column, then rotate its subdiagonal away
            for i = 1:(j-1)
                rotated = cosines[i] * hessenberg[i, j] + sines[i] * hessenberg[i+1, j]
                hessenberg[i+1, j] = -sines[i] * hessenberg[i, j] + cosines[i] * hessenberg[i+1, j]
                hessenberg[i, j] = rotated
            end
            radius = hypot(hessenberg[j, j], hessenberg[j+1, j])
            cosines[j], sines[j] = hessenberg[j, j] / radius, hessenberg[j+1, j] / radius
            hessenberg[j, j], hessenberg[j+1, j] = radius, 0.0
            rhs[j+1] = -sines[j] * rhs[j]
            rhs[j] = cosines[j] * rhs[j]
            abs(rhs[j+1]) <= tol * norm_b && break
        end
        coefficients = UpperTriangular(hessenberg[1:used, 1:used]) \ rhs[1:used]
        x .+= basis[:, 1:used] * coefficients
        abs(rhs[used+1]) <= tol * norm_b && return x, iterations, abs(rhs[used+1]) / norm_b
    end
    return x, iterations, norm(b .- apply(x)) / norm_b
end

"Stack complex matrices as one real vector: real parts, then imaginary parts, matrix by matrix."
_pack(Ys) = vcat((vcat(vec(real(Y)), vec(imag(Y))) for Y in Ys)...)

"The inverse of `_pack` for matrices of sizes `dims`."
function _unpack(x::Vector{Float64}, dims)
    out = Matrix{ComplexF64}[]
    offset = 0
    for d in dims
        n = prod(d)
        push!(out, reshape(x[(offset+1):(offset+n)] .+ 1im .* x[(offset+n+1):(offset+2n)], d))
        offset += 2n
    end
    return out
end

"""
Largest stacked stage dimension `s r_L r_R` for which the dense preconditioner is factorised; above
it the stage GMRES runs unpreconditioned (and says so in `stats.precond`).
"""
const _STAGE_PRECOND_MAX = 3072

"""
    rk_stage_solve(fl, fr, S, t, h, method; tol = 1e-10, maxnewton = 20, gmres_tol = 1e-3 tol,
                   gmres_maxiter = 600, Y0 = nothing, precond = :auto) -> (Y, stats)

The nonlinear stage equations `Y_j = S + h sum_l A_jl G(t + c_l h, Y_l)` of `RK_TABLEAUX[method]`
in the root space of frames `fr`, by Newton-GMRES on the doubled-real unknowns with the reduced
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
    S::AbstractMatrix,
    t::Real,
    h::Real,
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
    Y = Y0 === nothing ? [ComplexF64.(S) for _ = 1:nstages] : [ComplexF64.(y) for y in Y0]
    dims = [size(S) for _ = 1:nstages]
    norm_S = max(norm(S), eps())
    nfield, ngmres, t_rho, residual_norm = 0, 0, 0.0, Inf
    root_size = length(S)
    factorised = nothing
    if precond === :auto && nstages * root_size <= _STAGE_PRECOND_MAX
        J1, _ = flow_jac_ops(fl, frame_state(fr, S), t + h / 2)
        linear_part = frame_operator_matrix(fr, J1)
        mu0 =
            flow_normalized(fl) ?
            real(dot(vec(S), linear_part * vec(S))) / max(norm(S)^2, eps()) : 0.0
        stage_jacobian = fl.pref .* (linear_part - mu0 * I)
        factorised = lu(
            Matrix{ComplexF64}(I, nstages * root_size, nstages * root_size) .-
            h .* kron(butcher, stage_jacobian),
        )
    end
    # x -> P^-1 x on the packed real vector (identity without a preconditioner)
    function apply_preconditioner(x)
        factorised === nothing && return x
        stacked = vcat((vec(D) for D in _unpack(x, dims))...)
        solved = factorised \ stacked
        return _pack([reshape(solved[((j-1)*root_size+1):(j*root_size)], dims[j]) for j = 1:nstages])
    end
    for newton = 0:maxnewton
        fields, mus = Matrix{ComplexF64}[], Float64[]
        for l = 1:nstages
            G, mu, info = reduced_field(fl, fr, Y[l], t + tableau.c[l] * h)
            push!(fields, G)
            push!(mus, mu)
            nfield += 1
            t_rho += info.t_rho
        end
        residuals =
            [Y[j] .- S .- h .* sum(butcher[j, l] .* fields[l] for l = 1:nstages) for j = 1:nstages]
        residual_norm = sqrt(sum(norm(r)^2 for r in residuals)) / norm_S
        if residual_norm <= tol || newton == maxnewton
            return Y,
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
        jacobians = [reduced_jacobian(fl, fr, Y[l], t + tableau.c[l] * h, mus[l]) for l = 1:nstages]
        # the stage system's Jacobian on packed directions; an all-zero column (the trapezoid's
        # explicit first stage) needs no Jacobian action
        function newton_operator(x)
            directions = _unpack(x, dims)
            actions = [
                any(!iszero, butcher[:, l]) ? jacobians[l](directions[l]) : zero(directions[l]) for
                l = 1:nstages
            ]
            return _pack([
                directions[j] .- h .* sum(butcher[j, l] .* actions[l] for l = 1:nstages) for
                j = 1:nstages
            ])
        end
        # right preconditioning: solve (op o P^-1) y = -R, then dx = P^-1 y
        dy, iterations, _ = _gmres_real(
            x -> newton_operator(apply_preconditioner(x)),
            -_pack(residuals);
            tol = Float64(gmres_tol),
            maxiter = gmres_maxiter,
        )
        ngmres += iterations
        Y = [Y[j] .+ D for (j, D) in enumerate(_unpack(apply_preconditioner(dy), dims))]
    end
end
