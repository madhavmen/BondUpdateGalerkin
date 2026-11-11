# `src/BUG/Randomised/` — controlled bond expansion

The one bond expansion of the package: the DD-QR CBE (`DDQRExpansion`, the default `expansion` of
`cbe_bug_step!`). At each bond and side it draws random probes, projects them out of the current
frame (the discarded space, "DD"), streams them through the MPO, and keeps the new directions a
sector-by-sector QR or SVD finds. `DDQRExpansion(mode)` decides how many probes it draws and how
many directions it keeps:

| `mode` | probes and kept rank | certificate |
|---|---|---|
| `:rank_tail` (default) | one sketch of `fixed_width = 8` columns; the rank bisected on that sketch's own tail (`tail_factor` of the bond's sample share) | none (uncertified heuristic); the order-2 global Krylov closure checks the state |
| `:adaptive` | incremental sketches grown until an independent holdout passes (`holdout_width`, `batch_blocks`) | independent holdout |
| `:rank_adaptive` | the minimal rank per sector from a pilot sketch, confirmed on an independent holdout | independent holdout |
| `:rank_fast` | as `:rank_adaptive` with a smaller pilot and holdout, at most 64 columns per draw | independent holdout |
| `:fixed` | `fixed_width` columns, `fixed_rank` kept | none (fixed baseline) |

The default is the most efficient setting measured on the U(1) XX chain (domain wall, L = 64, tol
1e-2 to 1e-4; `benchmarks/XXT5/results/dw_L64` and `dw_branch_claims`): it keeps 8-15x fewer
directions than `:fixed` from the same sketch, at the same error from tol 1e-3 down, and is the
fastest mode. At tol 1e-2 `:fixed` and `:adaptive` were up to 2x more accurate. Use `:rank_tail`
with `order >= 2`, where the global closure guards it. `krylov_mode` picks the local Krylov
recurrence (`Core/RankAdaptation/README.md`).

## Call path

```
Core/Sweeps/half_sweeps.jl ── expand_frame(::DDQRExpansion, ...)        (cbe.jl)
                                └─ dd_cbe_expand                          (dd_cbe_expand.jl)
                                     ├─ _dd_controller (default)          (dd_cbe_controller.jl)
                                     │    └─ _dd_probe, _dd_select        (dd_cbe.jl)
                                     └─ _certified_range (exact = true,    (range_finder.jl)
                                        or no allocation)
```

## Files (in include order)

| file | contents |
|---|---|
| `sketch.jl` | `sector_graded_sketch` (Gaussian probes allotted per charge sector), the range-finder pass `_range_find`, the residual test Test B (`_sample_residual`), the sector boosts after a failed holdout (`_target_sectors!`), and `_record_sample!`. |
| `range_finder.jl` | `_certified_range`, the numerical-rank search used by `exact = true` (the deterministic reference) and when the step gives no allocation; the expansion record `CBEExpansion`. |
| `cbe.jl` | The sketch closures (a built `H Theta`, or the probe folded into the MPO contraction), the column helpers (`_alloc_columns`, `_boost!`, `_randomize_like`, `_trim_total`), and the `DDQRExpansion` strategy with its `BasisExpansion` methods. |
| `dd_cbe.jl` | The kernels: `TwoSiteSketchOp` and `apply_two_site_sketch!` (the probe streamed through the MPO, never forming the two-site action), `RootOp`, `_dd_probe` (probes in the discarded space), `_thin_qr` and `_dd_select` (the sectorwise QR). |
| `dd_cbe_controller.jl` | `_dd_controller`: grows the probe width until an independent 4-column holdout passes, escalating to twice that and boosting under-sampled sectors. |
| `dd_cbe_expand.jl` | `dd_cbe_expand`, one side of one bond, and `_resolve_parallel`. |

`Core/` never names these functions: it calls the four `BasisExpansion` functions of
`Core/RankAdaptation/basis_expansion.jl`.
