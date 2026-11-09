# `src/BUG/ITE/` — imaginary time

Normalised imaginary-time evolution on top of `cbe_bug_step!` at `tau = -dbeta`.

| file | contents |
|---|---|
| `schedules.jl` | Step-size schedules: `uniform_beta_steps`, `log_panel_steps`, `beta_grid`, `uniform_schedule`, `geometric_schedule`. |
| `ite_step.jl` | `ite_step!`: one step with renormalisation and an energy shift; the outside defect `dbeta ||(I-P) H psi_1||` is charged to `tol_basis`, and a step that misses its budget is split and retried (`retries`, `on_fail`). `ite_evolve!` runs a whole schedule. |

Failures are reported, never silent: a step that still misses its budget after the retries is
flagged (`on_fail = :flag`) or raises (`on_fail = :error`).
