# Independent Heston references: the exponential-affine solution, the semi-analytic call price,
# and the forward density's mass and affine moments.

"""
    heston_riccati(p, a, b, tau; nsteps = 20000, discount = true) -> (B, C)

The exact exponential-affine solution `u = exp(a x + B(tau) v + C(tau))` of `u_tau = A_H u` from
`u(0) = exp(a x + b v)`:
`B' = sigma^2 B^2/2 + (rho sigma a - kappa) B + (a^2 - a)/2`, `B(0) = b`;
`C' = (r - q) a + kappa theta B - r`, `C(0) = 0` (without `- r` when `discount = false`, the
forward-moment version). Integrated by classical RK4 with `nsteps` steps to near machine precision.
"""
function heston_riccati(
    p::HestonParams,
    a::Real,
    b::Real,
    tau::Real;
    nsteps::Int = 20000,
    discount::Bool = true,
)
    rate_B(B) = p.sigma^2 * B^2 / 2 + (p.rho * p.sigma * a - p.kappa) * B + (a^2 - a) / 2
    rate_C(B) = (p.r - p.q) * a + p.kappa * p.theta * B - (discount ? p.r : 0.0)
    B, C = Float64(b), 0.0
    step = tau / nsteps
    for _ = 1:nsteps
        # C' depends on B only, so both use the same four B stages
        k1 = rate_B(B)
        l1 = rate_C(B)
        k2 = rate_B(B + step / 2 * k1)
        l2 = rate_C(B + step / 2 * k1)
        k3 = rate_B(B + step / 2 * k2)
        l3 = rate_C(B + step / 2 * k2)
        k4 = rate_B(B + step * k3)
        l4 = rate_C(B + step * k3)
        B += step / 6 * (k1 + 2k2 + 2k3 + k4)
        C += step / 6 * (l1 + 2l2 + 2l3 + l4)
    end
    return B, C
end

"The affine field `exp(a x + B v + C)` on the grid (rank 1 at any size)."
heston_affine_field(g::QTTGrid, a::Real, B::Real, C::Real) =
    qtt_separable(g, x -> exp(a * x + C), v -> exp(B * v))

"""
    heston_call(p, S, K, v, T; umax = 200, n = 4000) -> Float64

The European call by Heston's semi-analytic formula (Gil-Pelaez inversion of the "little Heston
trap" characteristic function, midpoint rule with `n` nodes on `(0, umax]`): an independent
reference for the payoff, not a 2D closed form. Needs `sigma > 0`.
"""
function heston_call(
    p::HestonParams,
    S::Real,
    K::Real,
    v::Real,
    T::Real;
    umax::Real = 200.0,
    n::Int = 4000,
)
    log_spot = log(S)
    # characteristic function of log S_T
    function charfun(u)
        drift = p.kappa - p.rho * p.sigma * 1im * u
        d = sqrt((p.rho * p.sigma * 1im * u - p.kappa)^2 + p.sigma^2 * (1im * u + u^2))
        ratio = (drift - d) / (drift + d)
        decay = exp(-d * T)
        Cterm =
            (p.r - p.q) * 1im * u * T +
            p.kappa * p.theta / p.sigma^2 *
            ((drift - d) * T - 2 * log((1 - ratio * decay) / (1 - ratio)))
        Dterm = (drift - d) / p.sigma^2 * (1 - decay) / (1 - ratio * decay)
        return exp(Cterm + Dterm * v + 1im * u * log_spot)
    end
    du = umax / n
    log_strike = log(K)
    P1 = 0.0
    P2 = 0.0
    forward_factor = charfun(-1im)
    for k = 1:n
        u = (k - 0.5) * du
        P1 += real(exp(-1im * u * log_strike) * charfun(u - 1im) / (1im * u * forward_factor)) * du
        P2 += real(exp(-1im * u * log_strike) * charfun(u) / (1im * u)) * du
    end
    P1 = 0.5 + P1 / pi
    P2 = 0.5 + P2 / pi
    return S * exp(-p.q * T) * P1 - K * exp(-p.r * T) * P2
end

"Total mass `dx dv sum p` of a forward density (exactly conserved by `heston_op(...; adjoint)`)."
heston_mass(A::AbstractVector, g::QTTGrid) =
    qtt_cell(g) * real(tt_dot(qtt_separable(g, x -> 1.0, v -> 1.0), A))

"`dx dv sum exp(a x + b v) p(x, v)`: the affine moment of a forward density."
heston_moment(A::AbstractVector, g::QTTGrid, a::Real, b::Real) =
    qtt_cell(g) * real(tt_dot(qtt_separable(g, x -> exp(a * x), v -> exp(b * v)), A))
