# Examples

The scripts in `examples/` use only the public API, check their own answer against an exact or
independent reference, and exit non-zero when the check fails. CI runs all of them.

| Script | Problem | Reference |
|---|---|---|
| `examples/QMB/xx_quickstart.jl` | XX chain from Néel, U(1): tolerances, statuses, a refused request | free fermions |
| `examples/QMB/xx_domain_wall.jl` | XX chain from a domain wall: BUG and two-site TDVP side by side | free fermions |
| `examples/QTTS/heston_quickstart.jl` | backward Heston on a 2D grid; BUG error and spatial error reported apart | the exact flow of the discrete operator, and the exponential-affine (Riccati) solution |
| `examples/QTTS/gp_ground_state.jl` | Gross-Pitaevskii ground state in a harmonic trap | QTT gradient descent |

```bash
julia --project examples/QMB/xx_quickstart.jl OUTPUT_DIR
```

The optional first argument is the directory results are written to; without it nothing is
written.
