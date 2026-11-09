# `src/QMB/Models/OpenSystem/`

Lindblad dynamics, in two ways that share the jump operators:

| path | contents |
|---|---|
| `jump_channels.jl` | The loss operators of the dissipative benchmarks (pair loss, collective and site emission): `Q = sum_mu L_mu' L_mu` as a compact MPO for the no-jump generator `no_jump_mpo(H, Q)`, and the jumps `L_mu` as charged MPOs for trajectories (`SiteOp`, `JumpChannel`, `jump_mpo`, `pair_loss_channels`, `collective_emission_channels`, `site_emission_channels`, `doublon_loss_mpo`, `collective_emission_mpo`). |
| [`MPDO/`](MPDO/README.md) | The density operator vectorised in Liouville space and evolved by the BUG step under the Lindblad MPO. |
| [`MCWF/`](MCWF/README.md) | Quantum-jump (Monte Carlo wave function) trajectories, with the BUG step as the no-jump integrator. |
