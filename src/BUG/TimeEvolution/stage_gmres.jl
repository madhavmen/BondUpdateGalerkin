# The linear algebra of the stage solve's Newton steps: restarted GMRES on the doubled-real
# unknowns, its recycled variant (GCRO-DR) and the separable stage preconditioner. The stage
# solver (`stage_solvers.jl`) packs the complex stage roots into one real vector, since the
# Jacobian of a nonlinear flow is only real-linear.

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
    GMRESRecycle(count)

The recycled directions carried across the GMRES solves of one root solve (GCRO-DR, Parks et al.
2006): after each solve, the `count` harmonic Ritz directions of smallest magnitude of its last
cycle. Only the directions are kept: their images are recomputed under the current Jacobian at
every solve, so no product of an old operator is reused. `rk_stage_solve` makes a new one per
call, since the frames change between root solves. `extra_products` counts the products spent
re-imaging the directions (they are included in the GMRES iteration counts).
"""
mutable struct GMRESRecycle
    count::Int
    directions::Matrix{Float64}
    extra_products::Int
end
GMRESRecycle(count::Int) = GMRESRecycle(count, zeros(0, 0), 0)

"""
    _gcro_real(apply, target, recycle; tol, restart, maxiter) -> (solution, iterations, relres)

Restarted GMRES augmented by the recycled space of `recycle` (GCRO-DR): with the images
`C = A U` of the recycled directions `U` orthonormalised (and `U` rescaled so that `A U = C`), the
start is `x0 = U C' b` and every Arnoldi vector is kept orthogonal to `C`, so
`A (V - U B) = V_{m+1} H`. On return `recycle.directions` holds the new directions.
"""
function _gcro_real(
    apply,
    target::Vector{Float64},
    recycle::GMRESRecycle;
    tol::Float64 = 1e-10,
    restart::Int = 40,
    maxiter::Int = 400,
)
    dimension = length(target)
    target_norm = norm(target)
    target_norm == 0 && return zeros(dimension), 0, 0.0
    iterations = 0
    solution = zeros(dimension)
    # U (the rescaled recycled directions) and C = A U (orthonormal); empty without recycling
    recycled_basis, recycled_images = zeros(dimension, 0), zeros(dimension, 0)
    previous = recycle.directions
    if size(previous, 2) > 0 && size(previous, 1) == dimension
        # the recycled directions' images under the current operator
        images = reduce(hcat, [apply(previous[:, column]) for column = 1:size(previous, 2)])
        iterations += size(previous, 2)
        recycle.extra_products += size(previous, 2)
        factorisation = qr(images)
        triangular = Matrix(factorisation.R)
        pivots = abs.(diag(triangular))
        # a numerically dependent set of images is dropped rather than inverted
        if minimum(pivots) > 1e-12 * maximum(pivots)
            recycled_images = Matrix(factorisation.Q)
            recycled_basis = previous / triangular
            solution .= recycled_basis * (recycled_images' * target)
        end
    end
    recycled = size(recycled_images, 2)
    last_cycle = nothing
    relres = Inf
    while true
        residual = target .- apply(solution)
        beta = norm(residual)
        relres = beta / target_norm
        (beta <= tol * target_norm || iterations >= maxiter) && break
        cycle = min(restart, maxiter - iterations)
        basis = zeros(dimension, cycle + 1)
        unrotated = zeros(cycle + 1, cycle)       # the Hessenberg before rotation (harmonic Ritz)
        hessenberg = zeros(cycle + 1, cycle)
        couplings = zeros(recycled, cycle)        # B: the Arnoldi images' parts along C
        cosines, sines, rhs = zeros(cycle), zeros(cycle), zeros(cycle + 1)
        basis[:, 1] .= residual ./ beta
        rhs[1] = beta
        used = 0
        stalled = false
        for column = 1:cycle
            iterations += 1
            image = apply(basis[:, column])
            if recycled > 0
                couplings[:, column] .= recycled_images' * image
                image .-= recycled_images * view(couplings, :, column)
            end
            for row = 1:column
                hessenberg[row, column] = dot(view(basis, :, row), image)
                image .-= hessenberg[row, column] .* view(basis, :, row)
            end
            hessenberg[column+1, column] = norm(image)
            hessenberg[column+1, column] > 0 &&
                (basis[:, column+1] .= image ./ hessenberg[column+1, column])
            unrotated[1:(column+1), column] .= hessenberg[1:(column+1), column]
            for row = 1:(column-1)
                rotated =
                    cosines[row] * hessenberg[row, column] + sines[row] * hessenberg[row+1, column]
                hessenberg[row+1, column] =
                    -sines[row] * hessenberg[row, column] + cosines[row] * hessenberg[row+1, column]
                hessenberg[row, column] = rotated
            end
            radius = hypot(hessenberg[column, column], hessenberg[column+1, column])
            # the Krylov space closed on a direction mapped into the earlier ones: this column adds
            # nothing and would make the triangular solve singular (as in `_gmres_real`)
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
        used == 0 && break
        coefficients = UpperTriangular(hessenberg[1:used, 1:used]) \ rhs[1:used]
        solution .+= basis[:, 1:used] * coefficients
        # the update's recycled part, -U B y, keeps the correction's image orthogonal to C
        recycled > 0 && (solution .-= recycled_basis * (couplings[:, 1:used] * coefficients))
        last_cycle =
            (basis[:, 1:(used+1)], unrotated[1:(used+1), 1:used], couplings[:, 1:used], used)
        relres = abs(rhs[used+1]) / target_norm
        (stalled || relres <= tol) && break
    end
    last_cycle === nothing || _recycle_update!(recycle, recycled_basis, recycled_images, last_cycle)
    return solution, iterations, relres
end

"""
The next recycled directions: harmonic Ritz vectors of the augmented Arnoldi relation
`A W = What G`, with `W = [U, V_m]` and `What = [C, V_{m+1}]`.
"""
function _recycle_update!(recycle::GMRESRecycle, recycled_basis, recycled_images, last_cycle)
    basis, unrotated, couplings, used = last_cycle
    kept = size(recycled_basis, 2)
    scale = kept > 0 ? 1 ./ vec(sqrt.(sum(abs2, recycled_basis; dims = 1))) : Float64[]
    augmented = hcat(recycled_basis .* scale', basis[:, 1:used])       # W
    augmented_images = hcat(recycled_images, basis)                     # What
    relation = zeros(kept + used + 1, kept + used)                      # G
    relation[1:kept, 1:kept] .= Diagonal(scale)
    relation[1:kept, (kept+1):end] .= couplings
    relation[(kept+1):end, (kept+1):end] .= unrotated
    # GCRO-DR: G' G z = theta G' What' W z, keep the values of smallest magnitude
    values, vectors = eigen(relation' * relation, relation' * (augmented_images' * augmented))
    finite = findall(isfinite, values)
    isempty(finite) && return recycle
    smallest = finite[sortperm(abs.(values[finite]))[1:min(recycle.count, length(finite))]]
    chosen = vectors[:, smallest]
    # real arithmetic: a conjugate pair spans the real and imaginary parts
    factorisation = svd(augmented * hcat(real(chosen), imag(chosen)))
    nkeep = min(recycle.count, count(>(1e-10 * factorisation.S[1]), factorisation.S))
    recycle.directions = factorisation.U[:, 1:nkeep]
    return recycle
end

"""
    _separable_precond(Ml, Mr, pref, butcher, step_size) -> (stage_roots -> solved_roots)

`P^-1` for `P = I - h A (x) pref (Ml (x) I + I (x) Mr)` on the `s` stage roots, matrix-free: the
complex Schur forms of the Butcher matrix `A` (`s x s`), `Ml` and `transpose(Mr)` are taken once;
each application is then a block back substitution over the triangular stage matrix with one
triangular Sylvester solve (LAPACK `trsyl`) per stage, `O(s (r_L^2 r_R + r_L r_R^2))` -- never an
`s r_L r_R` matrix.
"""
function _separable_precond(
    Ml::AbstractMatrix,
    Mr::AbstractMatrix,
    pref::Number,
    butcher::AbstractMatrix,
    step_size::Real,
)
    left_schur = schur(Matrix{ComplexF64}(Ml))
    right_schur = schur(Matrix{ComplexF64}(transpose(Mr)))
    stage_schur = schur(Matrix{ComplexF64}(butcher))
    Ul, Tl, Ur, Tr = left_schur.Z, left_schur.T, right_schur.Z, right_schur.T
    stage_vectors, stage_triangle = stage_schur.Z, stage_schur.T
    nstages = size(butcher, 1)
    # the per-stage Sylvester coefficients: (1/2 - h T_jj pref Tl) X + X (1/2 - h T_jj pref Tr)
    half(triangle) = Matrix{ComplexF64}(I, size(triangle)...) ./ 2
    left_coefficients =
        [half(Tl) .- (step_size * stage_triangle[stage, stage] * pref) .* Tl for stage = 1:nstages]
    right_coefficients =
        [half(Tr) .- (step_size * stage_triangle[stage, stage] * pref) .* Tr for stage = 1:nstages]
    # the separable operator in the Schur bases
    separable(root) = pref .* (Tl * root .+ root * Tr)
    return function (stage_roots)
        # stages into the Schur basis of A, roots into the Schur bases of Ml and Mr^T
        rotated = [
            Ul' *
            sum(conj(stage_vectors[other, stage]) .* stage_roots[other] for other = 1:nstages) *
            Ur for stage = 1:nstages
        ]
        solved = Vector{Matrix{ComplexF64}}(undef, nstages)
        # back substitution over the upper-triangular stage matrix, last stage first
        for stage = nstages:-1:1
            rhs = copy(rotated[stage])
            for later = (stage+1):nstages
                coupling = stage_triangle[stage, later]
                iszero(coupling) || (rhs .+= (step_size * coupling) .* separable(solved[later]))
            end
            sylvester_solution, scale = LAPACK.trsyl!(
                'N',
                'N',
                copy(left_coefficients[stage]),
                copy(right_coefficients[stage]),
                rhs,
                1,
            )
            solved[stage] = sylvester_solution ./ scale
        end
        return [
            Ul * sum(stage_vectors[stage, other] .* solved[other] for other = 1:nstages) * Ur' for
            stage = 1:nstages
        ]
    end
end
