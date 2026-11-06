# The damped Gross-Pitaevskii equation on a periodic 2D grid.
#
# The frozen-density step is `frozen_flow`, stepped by `qtt_bug_step!`.
#
# The equation is never linearised:
#   d_t psi = F(t, psi) = (-i - gamma) H[psi, t] psi,
#   H[psi, t] = -Delta/2 + V_trap + V_pad(t) - mu - Omega L_z + g diag(rho),  rho = C(|psi|^2),
# and diag(rho) is rebuilt from the state it is applied to. The kinetic term is periodic.

"""
    GPModel

The damped GP problem on a grid: nonlinearity `g`, damping `gamma`, chemical potential `mu`, trap
frequency `omega`, rotation `Omega`, an optional `paddle`, the static linear part
`H0 = -Delta/2 + V_trap - mu - Omega L_z` as operator cores, and the relative compression tolerance
`rho_tol` of the density.
"""
struct GPModel
    grid::QTTGrid
    g::Float64
    gamma::Float64
    mu::Float64
    omega::Float64
    Omega::Float64
    paddle::Union{Nothing,GPPaddle}
    H0::Vector{Array{ComplexF64,4}}
    rho_tol::Float64
end

"""
    gp_model(g; gnl = 1.0, gamma = 0.1, mu = 0.0, omega = 1.0, Omega = 0.0, paddle = nothing,
             rho_tol = 1e-12) -> GPModel

The model on the periodic grid `g`, with `L_z = -i (x d_y - y d_x)`. `rho_tol` is the relative
compression tolerance of the density QTT; its action error is audited separately
([`gp_density_audit`](@ref)), never assumed.
"""
function gp_model(
    g::QTTGrid;
    gnl::Real = 1.0,
    gamma::Real = 0.1,
    mu::Real = 0.0,
    omega::Real = 1.0,
    Omega::Real = 0.0,
    paddle = nothing,
    rho_tol::Real = 1e-12,
)
    g.periodic_x && g.periodic_y || throw(
        ArgumentError(
            "gp_model: the GP fixtures are periodic; build the grid with periodic_x = periodic_y = true",
        ),
    )
    laplacian_x, laplacian_y = qtt_d(g, :x, 2; bc = :periodic), qtt_d(g, :y, 2; bc = :periodic)
    terms = Any[(-0.5, laplacian_x, nothing), (-0.5, nothing, laplacian_y)]
    if omega != 0
        push!(terms, (omega^2 / 2, qtt_coord_diag(g, :x, x -> x^2), nothing))
        push!(terms, (omega^2 / 2, nothing, qtt_coord_diag(g, :y, y -> y^2)))
    end
    mu != 0 && push!(terms, (-mu, nothing, op_identity(g.ny)))
    if Omega != 0
        # -Omega L_z = i Omega (x d_y - y d_x)
        first_x, first_y = qtt_d(g, :x, 1; bc = :periodic), qtt_d(g, :y, 1; bc = :periodic)
        push!(terms, (1im * Omega, qtt_coord_diag(g, :x), first_y))
        push!(terms, (-1im * Omega, first_x, qtt_coord_diag(g, :y)))
    end
    H0 = [ComplexF64.(core) for core in qtt_op(g, terms)]
    return GPModel(
        g,
        Float64(gnl),
        Float64(gamma),
        Float64(mu),
        Float64(omega),
        Float64(Omega),
        paddle,
        H0,
        Float64(rho_tol),
    )
end

"`V_pad(t)` of the model as operator cores (`nothing` without a paddle or once it is off)."
gp_paddle_op(m::GPModel, t::Real) = gp_paddle_op(m.grid, m.paddle, t)

"The state-independent part `H0 + V_pad(t)` as operator cores."
function gp_linear_op(m::GPModel, t::Real)
    paddle_op = gp_paddle_op(m, t)
    paddle_op === nothing && return m.H0
    return op_lincomb([(1.0, m.H0), (1.0, paddle_op)])
end

"""
    gp_density(m, A) -> (rho, info)

`rho = C(|psi|^2)` of the state cores `A` at `m.rho_tol` ([`tt_density`](@ref)).
"""
gp_density(m::GPModel, A::AbstractVector) = tt_density(A; tol = m.rho_tol)

"""
    gp_hamiltonian_op(m, A, t; rho = nothing) -> (H, info)

`H[psi, t] = H0 + V_pad(t) + g diag(rho(psi))` as operator cores. Pass `rho` to reuse a density
built elsewhere (the frozen-density ablation); by default it is rebuilt from `A`. `info` carries the
density's `chi_rho`, error bound `rho_err`, build seconds `t_rho`, `retried`, and the operator's
bond dimension `w`.
"""
function gp_hamiltonian_op(m::GPModel, A::AbstractVector, t::Real; rho = nothing)
    linear = gp_linear_op(m, t)
    if m.g == 0
        return linear,
        (
            chi_rho = 0,
            rho_err = 0.0,
            t_rho = 0.0,
            retried = false,
            w = maximum(tt_ranks(linear); init = 1),
        )
    end
    density_info = (
        chi = maximum(tt_ranks(rho === nothing ? A : rho); init = 1),
        err_bound = 0.0,
        seconds = 0.0,
        retried = false,
    )
    if rho === nothing
        rho, density_info = gp_density(m, A)
    end
    H = op_lincomb([(1.0, linear), (m.g, op_diag(rho))])
    return H,
    (
        chi_rho = density_info.chi,
        rho_err = density_info.err_bound,
        t_rho = density_info.seconds,
        retried = density_info.retried,
        w = maximum(tt_ranks(H); init = 1),
    )
end

"""
    GPHamiltonianCache

Memo of `H[psi, t]` MPOs keyed by `(state tag, t)`: the tag hashes the state's core data, so a
changed trial state or stage time is a miss and rebuilds the density and the MPO; a stale nonlinear
or moving-paddle operator cannot be reused. `hits`/`misses` count.
"""
mutable struct GPHamiltonianCache
    store::Dict{Tuple{UInt,Float64},Any}
    hits::Int
    misses::Int
end
GPHamiltonianCache() = GPHamiltonianCache(Dict{Tuple{UInt,Float64},Any}(), 0, 0)

"A content hash of a state's cores (gauge-dependent: a re-gauged state is a different tag)."
qtt_state_tag(A::AbstractVector) = foldl((h, c) -> hash(c, h), A; init = hash(length(A)))
qtt_state_tag(psi::SymMPS) = qtt_state_tag(qtt_cores(psi))

"""
    gp_hamiltonian(m, psi, t; cache = nothing, rho = nothing) -> (MPO, info)

The solver MPO of `H[psi, t]` for a state ([`gp_hamiltonian_op`](@ref)); with `cache`, an unchanged
`(psi, t)` returns the stored MPO.
"""
function gp_hamiltonian(m::GPModel, psi::SymMPS, t::Real; cache = nothing, rho = nothing)
    A = qtt_cores(psi)
    key = (qtt_state_tag(A), Float64(t))
    if cache !== nothing && rho === nothing && haskey(cache.store, key)
        cache.hits += 1
        return cache.store[key]
    end
    H, info = gp_hamiltonian_op(m, A, t; rho = rho)
    out = (qtt_to_mpo(H), info)
    if cache !== nothing && rho === nothing
        cache.misses += 1
        cache.store[key] = out
    end
    return out
end

"""
    gp_vector_field(m, A, t; tol = 1e-12) -> TT

`F(t, psi) = (-i - gamma) H[psi, t] psi` on TT cores, the density rebuilt from `A` itself.
"""
function gp_vector_field(m::GPModel, A::AbstractVector, t::Real; tol::Real = 1e-12)
    H, _ = gp_hamiltonian_op(m, A, t)
    return tt_scale(first(op_zipup(H, A; tol = tol)), -1im - m.gamma)
end

"""
    gp_jvp(m, A, dA, t; tol = 1e-12) -> TT

The real-linear derivative `DF(psi)[dpsi] = (-i - gamma) [(H0 + V_pad - mu + 2 g |psi|^2) dpsi +
g psi^2 conj(dpsi)]`. Not complex-linear: `gp_jvp(m, A, im*dA, t) != im*gp_jvp(m, A, dA, t)`
whenever `g != 0`.
"""
function gp_jvp(m::GPModel, A::AbstractVector, dA::AbstractVector, t::Real; tol::Real = 1e-12)
    linear = gp_linear_op(m, t)
    linear_part, _ = op_zipup(linear, dA; tol = tol)
    m.g == 0 && return tt_scale(linear_part, -1im - m.gamma)
    rho, _ = gp_density(m, A)
    psi_squared, _ = tt_product(A, A; tol = m.rho_tol)
    # diag(f) dA = f .* dA by zip-up products, never the exact chi_rho chi_dA cores
    density_term, _ = tt_product(rho, dA; tol = tol)
    conjugate_term, _ = tt_product(psi_squared, tt_conj(dA); tol = tol)
    total =
        tt_lincomb([(1.0, linear_part), (2 * m.g, density_term), (m.g, conjugate_term)]; tol = tol)
    return first(tt_compress(tt_scale(total, -1im - m.gamma); tol = tol))
end

"""
    gp_density_audit(m, A) -> Float64

`|g| ||(rho_QTT - |psi|^2) psi|| / ||psi||` from dense samples: the action audit of the density
compression. Small grids only.
"""
function gp_density_audit(m::GPModel, A::AbstractVector)
    psi_values = vec(tt_dense(A))
    rho, _ = gp_density(m, A)
    rho_values = vec(tt_dense(rho))
    return abs(m.g) * norm((rho_values .- abs2.(psi_values)) .* psi_values) /
           max(norm(psi_values), eps())
end

"""
    GPFlow(m; imaginary = false, normalized = imaginary)

The damped GP flow of `m` in real time (`pref = -i - gamma`), or in imaginary time (`pref = -1`):
`normalized = true` is the normalised gradient flow `-(H[psi] - mu[psi]) psi`, `normalized = false`
keeps the field `-H[psi] psi` with its density rebuilt at every stage (the caller restores the norm).
"""
struct GPFlow <: QTTFlow
    m::GPModel
    pref::ComplexF64
    normalized::Bool
end
GPFlow(m::GPModel; imaginary::Bool = false, normalized::Bool = imaginary) =
    GPFlow(m, imaginary ? ComplexF64(-1) : ComplexF64(-1im - m.gamma), normalized)

flow_op(fl::GPFlow, A, t) = gp_hamiltonian_op(fl.m, A, t)
flow_normalized(fl::GPFlow) = fl.normalized

"`J1 = H0 + V_pad + 2 g diag(rho)`, `J2 = g diag(psi^2)` of the real-linear Jacobian."
function flow_jac_ops(fl::GPFlow, A, t)
    m = fl.m
    linear = gp_linear_op(m, t)
    m.g == 0 && return linear, nothing
    rho, _ = gp_density(m, A)
    psi_squared, _ = tt_product(A, A; tol = m.rho_tol)
    return op_lincomb([(1.0, linear), (2 * m.g, op_diag(rho))]), op_scale(op_diag(psi_squared), m.g)
end
