# Examples

Short scripts that use only the public API. Each checks its own answer against an exact or
independent reference, prints a final `...-OK` line, and exits non-zero if the check fails. The
first argument is an optional output directory; without it nothing is written.

```bash
julia --project examples/QMB/xx_quickstart.jl [OUTPUT_DIR] [L]
julia --project examples/QMB/xx_domain_wall.jl [OUTPUT_DIR] [L] [T]
julia --project examples/QTTS/heston_quickstart.jl [OUTPUT_DIR]
julia --project examples/QTTS/gp_ground_state.jl [OUTPUT_DIR]
```

| Script | Problem | Reference | What it shows |
|---|---|---|---|
| `QMB/xx_quickstart.jl` | XX chain from Néel, U(1) | free fermions | the three tolerances, the per-step `budget_status`, and a refused (`:budget_unmet`) request at `maxdim = 4` |
| `QMB/xx_domain_wall.jl` | XX chain from a domain wall, U(1) | free fermions | BUG order 2 and two-site TDVP on the same state, MPO, step and rank cap while the bond dimension grows |
| `QTTS/heston_quickstart.jl` | backward Heston on a 2D grid | the exact discrete flow, and the exponential-affine solution (Riccati) | the generator with one-sided boundary stencils, a QTT BUG step (`qtt_bug_step!`, order 3) on a linear flow, the BUG error kept apart from the spatial error |
| `QTTS/gp_ground_state.jl` | Gross-Pitaevskii in a harmonic trap, `g = 1` | QTT gradient descent | the nonlinear imaginary-time step with the density rebuilt at every stage, monotone energy |

Run times have not been measured yet. `julia --project examples/run_all.jl` runs every script at
small sizes (the manual pre-commit hook `julia-examples`); run it before a pull request.

The BUG-Julia examples `oat_z2.jl`, `tfim_z2.jl` and `observables_demo.jl`, their shared `common.jl`
and the plotting script `plot_examples.py` are not ported: their models are covered by the tests in
`test/QMB/ClosedSystem`, and they wrote CSVs into the source tree.
