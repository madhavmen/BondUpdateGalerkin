# `src/QTTS/Models/Heston/`

| file | contents |
|---|---|
| `heston.jl` | `HestonParams` and the generator on a log-price x variance grid, `u_tau = A_H u` in time to maturity (`heston_op` as operator cores, `heston_mpo` for the engine). The forward density uses the discrete transpose without the discount, which conserves mass exactly. |
| `heston_reference.jl` | Independent references: the exact exponential-affine solution (`heston_affine_field`, `heston_riccati`), the semi-analytic call price (`heston_call`), and the forward density's mass and moments (`heston_mass`, `heston_moment`). |
