# `src/QMB/Models/OpenSystem/MCWF/`

Quantum-jump trajectories. A trajectory evolves `dpsi/dt = G psi` with the no-jump generator
until the survival `||psi||^2` reaches a random threshold, then applies a jump
`psi <- L_mu psi / ||L_mu psi||` chosen with probability `∝ ||L_mu psi||^2`. The segment is
renormalised every step and its log survival carried explicitly, so the threshold test is exact.

| file | contents |
|---|---|
| `quantum_jumps.jl` | `jump_trajectory`, `jump_ensemble` (many trajectories with an error bar), `apply_jump`, `jump_weight`, and the adaptive step control `step_doubling_error`, `next_step_size`. The no-jump step is any integrator, `cbe_bug_step!` by default. |

The jump operators are in `../jump_channels.jl`.
