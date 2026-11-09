# `src/QTTS/Models/GrossPitaevskii/`

The damped Gross-Pitaevskii equation `d_t psi = (-i - gamma) H[psi, t] psi`, with
`H = -Delta/2 + V_trap + V_paddle(t) - mu - Omega L_z + g |psi|^2`. The equation is never
linearised: `H` is rebuilt from the state at every evaluation.

| file | contents |
|---|---|
| `gross_pitaevskii.jl` | `GPModel` (`gp_model`), the flow `GPFlow` with its `flow_op`/`flow_jac_ops` methods, the Hamiltonian and its cache (`gp_hamiltonian`, `gp_hamiltonian_op`, `GPHamiltonianCache`, `gp_linear_op`, `gp_paddle_op`), the density (`gp_density`, `gp_density_audit`), `gp_vector_field`, `gp_jvp`. |
| `paddle.jl` | The moving paddle potential of the vortex-shedding runs (`GPPaddle`, `paddle_state`). |
| `ground_state.jl` | Ground states: imaginary-time BUG (`gp_ground_state!`) and QTT gradient descent (`gp_gradient_descent!`). |
| `observables.jl` | `gp_observables` (energies, chemical potential, stationarity residual), `gp_parts`, `vortex_count`. |
