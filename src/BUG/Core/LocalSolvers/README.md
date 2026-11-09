# `src/BUG/Core/LocalSolvers/`

Matrix-free exponentials and implicit root integrators on Telum tensors. The root solve of every
step and the K/L/S stages call `expv`.

| file | contents |
|---|---|
| `krylov.jl` | `lanczos_expv` (Hermitian) and `arnoldi_expv` (non-Hermitian generators, e.g. open systems) for `exp(tau A) x`, with error estimate (`conv_tol`) and substepping (`substeps`) so a large `tau` is not silently truncated; `hermitian_tridiagonal_exp_coeffs`; the opt-in Krylov-depth log (`enable_krylov_log`, `get_krylov_log`, `KRYLOV_LOG`). |
| `root_methods.jl` | The root integrators other than the exponential (`ROOT_METHODS`: `:trapezoid`, `:radau2`, `:radau3`, `:radau2_gmres`) as rational stability functions on the Krylov basis, with `root_order` and `root_stability`; `radau2_gmres_expv`; the scaled-Krylov helpers; and the public dispatcher `expv(...; root_method)`. |

The local Krylov enrichment of the half-sweeps (a different Krylov recurrence, weighted by its
contribution to the step) is in `../RankAdaptation/krylov_enrichment.jl`.
