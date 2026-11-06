# Shared pieces of the QTT benchmark drivers: environment parameters, CSV output, and independent
# sparse-grid references written from the finite-difference stencils (no QTT code).
#
# The stencils are written here instead of
# being taken from the tests, whose dense matrices are small-grid only.

using LinearAlgebra, SparseArrays, Random
using BondUpdateGalerkin.BUG, BondUpdateGalerkin.QTTS, BondUpdateGalerkin.Baselines
set_symmetry!(:none)

env_string(key, default) = get(ENV, key, default)
env_int(key, default) = parse(Int, env_string(key, string(default)))
env_float(key, default) = parse(Float64, env_string(key, string(default)))
env_list(key, default) =
    [parse(Float64, item) for item in split(env_string(key, default), ",") if !isempty(item)]

"Write NamedTuple rows to a CSV (header from the union of keys, in first-seen order)."
function write_csv(path, rows)
    isempty(rows) && return
    columns = collect(keys(rows[1]))
    for row in rows, key in keys(row)
        key in columns || push!(columns, key)
    end
    quoted(value) = (text = string(value); occursin(',', text) ? "\"" * text * "\"" : text)
    open(path, "w") do io
        println(io, join(columns, ","))
        for row in rows
            println(io, join((quoted(get(row, key, "")) for key in columns), ","))
        end
    end
end

"""
The sparse `N x N` finite-difference matrix of derivative order `order` (1 or 2) with spacing `h`
and boundary condition `bc` (`:periodic`, `:dirichlet0`, `:onesided`), row by row.
"""
function sparse_stencil(N::Int, h::Real, order::Int, bc::Symbol)
    rows, cols, weights = Int[], Int[], Float64[]
    central =
        order == 1 ? ((1, 1 / (2h)), (-1, -1 / (2h))) : ((1, 1 / h^2), (0, -2 / h^2), (-1, 1 / h^2))
    onesided_first = order == 1 ? [-3, 4, -1] ./ (2h) : [2, -5, 4, -1] ./ h^2
    for row = 1:N
        if bc === :onesided && (row == 1 || row == N)
            # second-order one-sided rows: the stencil reversed (and mirrored in sign for d/dx) at the end
            for (offset, weight) in enumerate(onesided_first)
                col = row == 1 ? offset : N - offset + 1
                push!(rows, row), push!(cols, col)
                push!(weights, row == 1 || order == 2 ? weight : -weight)
            end
            continue
        end
        for (offset, weight) in central
            col = row + offset
            if bc === :periodic
                col = mod1(col, N)
            elseif !(1 <= col <= N)
                continue
            end
            push!(rows, row), push!(cols, col), push!(weights, weight)
        end
    end
    return sparse(rows, cols, weights, N, N)
end

"`Ax (x) Ay` on the grid index `ix + 1 + Nx iy` (`nothing` = identity)."
grid_kron(g, Ax, Ay) = kron(
    Ay === nothing ? sparse(1.0I, 2^g.ny, 2^g.ny) : Ay,
    Ax === nothing ? sparse(1.0I, 2^g.nx, 2^g.nx) : Ax,
)

"Sparse GP linear operator `-Delta/2 + V_trap - mu - Omega L_z` (+ the paddle at `t`)."
function gp_sparse_linear(m::GPModel, t::Real)
    g = m.grid
    xs, ys = qtt_coords(g)
    second_x, second_y =
        sparse_stencil(2^g.nx, g.dx, 2, :periodic), sparse_stencil(2^g.ny, g.dy, 2, :periodic)
    X, Y = spdiagm(0 => collect(xs)), spdiagm(0 => collect(ys))
    H =
        -0.5 * (grid_kron(g, second_x, nothing) + grid_kron(g, nothing, second_y)) +
        (m.omega^2 / 2) * (grid_kron(g, X^2, nothing) + grid_kron(g, nothing, Y^2)) - m.mu * I
    if m.Omega != 0
        first_x, first_y =
            sparse_stencil(2^g.nx, g.dx, 1, :periodic), sparse_stencil(2^g.ny, g.dy, 1, :periodic)
        H = H + 1im * m.Omega * (grid_kron(g, X, first_y) - grid_kron(g, first_x, Y))
    end
    if m.paddle !== nothing
        amplitude, centre = paddle_state(m.paddle, t)
        if amplitude != 0
            p = m.paddle
            box_x = QTTS._box.(xs .- centre, p.width_x, p.smooth)
            box_y = QTTS._box.(ys .- p.y_c, p.width_y, p.smooth)
            H = H + amplitude * spdiagm(0 => vec(box_x * transpose(box_y)))
        end
    end
    return complex(H)
end

"Dense RK4 of `d_t v = (-i - gamma)(H(t) v + g |v|^2 v)` on the grid: the GP time reference."
function gp_dense_rk4(m::GPModel, v0::AbstractVector, t0::Real, T::Real; dt::Real = 1e-4)
    v = ComplexF64.(v0)
    pref = -1im - m.gamma
    nsteps = max(1, round(Int, (T - t0) / dt))
    h = (T - t0) / nsteps
    static_operator = m.paddle === nothing ? gp_sparse_linear(m, t0) : nothing
    field(t, w) =
        pref .* (
            (static_operator === nothing ? gp_sparse_linear(m, t) : static_operator) * w .+
            m.g .* abs2.(w) .* w
        )
    t = Float64(t0)
    for _ = 1:nsteps
        k1 = field(t, v)
        k2 = field(t + h / 2, v .+ h / 2 .* k1)
        k3 = field(t + h / 2, v .+ h / 2 .* k2)
        k4 = field(t + h, v .+ h .* k3)
        v = v .+ h / 6 .* (k1 .+ 2k2 .+ 2k3 .+ k4)
        t += h
    end
    return v
end

"A grid field of TT cores, flattened on the grid index."
grid_vec(A, g) = vec(qtt_values(A, g))

"The n-th harmonic-oscillator eigenfunction by the three-term recurrence."
phi_osc(n::Int, x) =
    n == 0 ? pi^(-0.25) * exp(-x^2 / 2) :
    n == 1 ? sqrt(2) * x * phi_osc(0, x) :
    (sqrt(2 / n) * x * phi_osc(n - 1, x) - sqrt((n - 1) / n) * phi_osc(n - 2, x))

"A rank-rich oscillator superposition: a nontrivial, entangled start."
function rank_rich_state(g; nmax = 4, seed = 1)
    rng = MersenneTwister(seed)
    terms = Any[]
    for mx = 0:nmax, my = 0:nmax
        coefficient = (randn(rng) + 1im * randn(rng)) / (1 + mx + my)^2
        push!(terms, (coefficient, qtt_separable(g, x -> phi_osc(mx, x), y -> phi_osc(my, y))))
    end
    return tt_lincomb(terms; tol = 1e-13)
end
