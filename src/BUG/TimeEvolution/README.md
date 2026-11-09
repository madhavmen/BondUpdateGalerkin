# `src/BUG/TimeEvolution/`

The time steps. `cbe_bug_step!` is the public entry point for real time (`tau = -im*dt`) and the
building block of imaginary time (`../ITE/`) and of the nonlinear QTT step (`QTTS/Tooling/`).

## One `cbe_bug_step!`

1. Keyword checks and the budget (`_resolve_step_options`, `bug_step_stages.jl`).
2. The two half-sweeps (`Core/Sweeps/half_sweeps.jl`), sequential or concurrent.
3. For `order > 1`, the frontier closure (`_close_root_frontiers!`, `bug_step_stages.jl`).
4. The root solve on the centre bond (`expv`, `Core/LocalSolvers/`), or the K/L/S kernel.
5. Optional root report and outside-residual re-enrichment (`_report_root!`).
6. Assembly and one compression (`_assemble_and_compress!`, `Core/diagnostics.jl`), and the step
   record.

## Files

| file | contents |
|---|---|
| `bug_step.jl` | `cbe_bug_step!`: all its keywords (with comments on each), the half-sweeps, the closure call, the root solve and the hand-off to the compression. |
| `bug_step_stages.jl` | The stages between the half-sweeps and the compression: `_resolve_step_options`, `_close_root_frontiers!`, `_report_root!` (energies, outside residual and its re-enrichment), and the step record. |
| `bug_step_variants.jl` | `parallel_kls_step!` (the K/L/S integrator) and `_replace_state!`, which the QTT step uses to write a result back. |
| `flows.jl` | Vector fields for nonlinear steps: `QTTFlow`, `LinearFlow`, `flow_op`, `flow_jac_ops`, `flow_normalized`, `frozen_flow`. A domain (QTTS) supplies the nonlinear flows. |
| `stage_gmres.jl` | Restarted GMRES on the doubled-real unknowns, its recycled variant (GCRO-DR, `GMRESRecycle`) and the separable stage preconditioner. |
| `stage_solvers.jl` | Implicit Runge-Kutta stages in the root space of a step: the frame interface a state representation implements (`frame_state`, `frame_project`, `frame_operator_matrix`, `frame_conj`, `frame_root_envs`, ...), the stage-field interface (`stage_field_ops`, `stage_field`, `stage_jacobian`), `RK_TABLEAUX`, `qtt_root_for_order` and the Newton-GMRES solver `rk_stage_solve`. |
