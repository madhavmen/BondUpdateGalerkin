# Gross-Pitaevskii observables: energies, chemical potential, stationarity residual, vortex count.
#
# Physical quantities use the discrete L2 inner product with the
# weight `dx dy` outside the cores.

const _GP_PARTS = IdDict{Any,Any}()

"The model's separate operators: kinetic `-Delta/2`, trap `V_trap`, rotation `-Omega L_z` (cores)."
function gp_parts(m::GPModel)
    get!(_GP_PARTS, m) do
        g = m.grid
        kinetic = qtt_op(
            g,
            [
                (-0.5, qtt_d(g, :x, 2; bc = :periodic), nothing),
                (-0.5, nothing, qtt_d(g, :y, 2; bc = :periodic)),
            ],
        )
        trap =
            m.omega == 0 ? nothing :
            qtt_op(
                g,
                [
                    (m.omega^2 / 2, qtt_coord_diag(g, :x, x -> x^2), nothing),
                    (m.omega^2 / 2, nothing, qtt_coord_diag(g, :y, y -> y^2)),
                ],
            )
        rotation =
            m.Omega == 0 ? nothing :
            qtt_op(
                g,
                [
                    (1im * m.Omega, qtt_coord_diag(g, :x), qtt_d(g, :y, 1; bc = :periodic)),
                    (-1im * m.Omega, qtt_d(g, :x, 1; bc = :periodic), qtt_coord_diag(g, :y)),
                ],
            )
        (K = kinetic, V = trap, R = rotation)
    end
end

"`<A| W |A>` for operator cores `W` (zero for `nothing`)."
_expect(W, A) = W === nothing ? 0.0 : real(_op_expect(W, A))

"""
    gp_observables(m, A, t; residual = true) -> NamedTuple

Particle number `N`; kinetic, trap, rotation, paddle and interaction energies; the grand-canonical
functional `F_gc = E_kin + E_trap + E_rot + E_pad - mu N + g/2 int |psi|^4`; the normalised energy
`E = (E_kin + E_trap + E_rot + E_pad + g/2 int |psi|^4) / N`, the functional the imaginary-time flow
descends, so the paddle is in it like the trap (without it every relaxation step around a paddle
reads as a rise and is rejected); the chemical potential
`mu[psi] = <psi, H[psi] psi> / <psi, psi>`; the stationarity residual
`R_mu = ||(H[psi] - mu[psi]) psi|| / ||psi||` (`NaN` with `residual = false`); the paddle's external
power `P_ext = int (d_t V_pad) |psi|^2` (central difference in t); and the largest rank `chi`.
"""
function gp_observables(m::GPModel, A::AbstractVector, t::Real; residual::Bool = true)
    g = m.grid
    cell = qtt_cell(g)
    parts = gp_parts(m)
    norm_squared = tt_norm(A)^2
    rho, _ = gp_density(m, A)
    quartic = real(tt_dot(rho, rho))                      # sum |psi|^4 (rho real)
    kinetic, trap, rotation =
        cell * _expect(parts.K, A), cell * _expect(parts.V, A), cell * _expect(parts.R, A)
    paddle = cell * _expect(gp_paddle_op(m, t), A)
    interaction = m.g / 2 * cell * quartic
    H, _ = gp_hamiltonian_op(m, A, t)
    mu = real(_op_expect(H, A)) / max(norm_squared, eps())
    # R_mu needs ||H psi||: the one compressed product here, skipped at production sizes
    stationarity =
        residual ?
        tt_norm(tt_lincomb([(1.0, first(op_zipup(H, A; tol = 1e-12))), (-mu, A)]; tol = 1e-12)) /
        sqrt(max(norm_squared, eps())) : NaN
    time_step = 1e-4
    paddle_later = cell * _expect(gp_paddle_op(m, t + time_step), A)
    paddle_earlier = cell * _expect(gp_paddle_op(m, max(t - time_step, 0.0)), A)
    power = (paddle_later - paddle_earlier) / (t + time_step - max(t - time_step, 0.0))
    N = cell * norm_squared
    return (
        N = N,
        E_kin = kinetic,
        E_trap = trap,
        E_rot = rotation,
        E_pad = paddle,
        E_int = interaction,
        F_gc = kinetic + trap + rotation + paddle - m.mu * N + interaction,
        E = (kinetic + trap + rotation + paddle + interaction) / max(N, eps()),
        mu = mu + m.mu,
        R_mu = stationarity,
        P_ext = power,
        chi = maximum(tt_ranks(A); init = 1),
    )
end

"""
    vortex_count(M; mask = 0.05, window = 2) -> (n_plus, n_minus, cores)

Signed winding `q = (1/2pi) sum arg-differences` of the phase of `M` around every grid plaquette,
skipping plaquettes outside the condensate, where the phase is ill-conditioned (density-zero
pixels). A plaquette is outside when the largest density within `window` pixels of each of its
corners is below `mask * max|M|^2`. A vortex core is itself a density zero, so testing the corners'
own density (`window = 0`) masks every core and counts no vortex at all (0 counted where the field
shows 6). `cores` lists `(i, j, q)` for every charged plaquette. `window` is in pixels, so the count
depends on the sampling: the rule is calibrated on 128^2 samples of the 64 x 64 box; at 256^2 the
window no longer reaches past a core and counts 0 where the 128^2 fields show 1-4 pairs.
"""
function vortex_count(M::AbstractMatrix; mask::Real = 0.05, window::Int = 2)
    nx, ny = size(M)
    density = abs2.(M)
    max_density = maximum(density)
    # the largest density within `window` pixels of each pixel (the condensate envelope of a core)
    envelope =
        window == 0 ? density :
        [
            maximum(
                @view density[
                    max(1, row - window):min(nx, row + window),
                    max(1, col - window):min(ny, col + window),
                ]
            ) for row = 1:nx, col = 1:ny
        ]
    wrap(angle_difference) = mod(angle_difference + pi, 2pi) - pi
    n_plus, n_minus = 0, 0
    cores = Tuple{Int,Int,Int}[]
    for i = 1:(nx-1), j = 1:(ny-1)
        corners = (M[i, j], M[i+1, j], M[i+1, j+1], M[i, j+1])
        corner_envelope =
            min(envelope[i, j], envelope[i+1, j], envelope[i+1, j+1], envelope[i, j+1])
        corner_envelope < mask * max_density && continue
        winding = sum(wrap(angle(corners[mod1(k + 1, 4)]) - angle(corners[k])) for k = 1:4)
        charge = round(Int, winding / 2pi)
        charge == 0 && continue
        push!(cores, (i, j, charge))
        charge > 0 ? (n_plus += charge) : (n_minus -= charge)
    end
    return n_plus, n_minus, cores
end
