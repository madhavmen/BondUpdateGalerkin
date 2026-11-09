# `src/BUG/Core/RankAdaptation/`

How the trial space grows and shrinks: the tolerance budget, the interface to the bond expansion,
the local Krylov enrichment, the order-p closure and the final truncation.

## The four tests of a step

Every basis decision is a test against a share of `tol_basis`:

- **Test A** — the retained-rank tail of a frame (`truncation.jl`).
- **Test B** — the sampled residual of the randomised bond expansion (`BUG/Randomised/sketch.jl`).
- **Test C / D** — the first and the later local Krylov directions (`krylov_enrichment.jl`).

The closure adds the frontier leak for `order > 1`. Every verdict lands in the step's
`AdaptiveLedger` (`basis_budget.jl`).

## Files

| file | contents |
|---|---|
| `basis_budget.jl` | `AdaptiveBudget` (the three tolerances), `TOL_DEFAULTS`, `basis_allocation`/`BasisAllocation` (how `tol_basis` is split over bonds and tests), `indicator`, the work ceiling (`work_ceiling`, `bond_work_ceilings`, `WorkCeiling`), and the ledger: `AdaptiveLedger`, `merge_ledger!`, `budget_status`, `budget_diagnostic`, `_log_bond!`. |
| `basis_expansion.jl` | The `BasisExpansion` interface the sweeps call: `expand_frame`, `sketch_closures`, `folded_sketch_closures`, `sampled_leak`. `BUG/Randomised` supplies the `DDQRExpansion` methods. |
| `krylov_enrichment.jl` | The half-sweep Krylov stage (Tests C/D): the step weight of a Krylov direction (`_kry_weight`, a residual integral), `_extend_frame`, `_append_ranked` (for `krylov_rank_tol > 0`), `_krylov_enrich`, and `_record_krylov!`. |
| `krylov_recurrence.jl` | The local Arnoldi recurrence `_krylov_enrich` drives: `_krylov_basis`, `_arnoldi_chain`. |
| `krylov_closure_frames.jl` | The order-p closure's frames (`KLFrames`), their environment caches, `KrylovFrontier`, `frames_from_state`, the projections of the state and its frontiers onto the frames. |
| `krylov_closure_loop.jl` | The closure drivers `_krylov_closure!` and `_single_pass_closure!`: loop over the frontiers `j = 1..p-1` until each one's outside action is within its share. |
| `krylov_frontier_bonds.jl` | One frontier bond: frames with a prescribed isometry, the leak measurements, the bond action, and `_frontier_expand` (the bond expansion inside the closure). |
| `truncation.jl` | The final rank truncation (`truncate_sweep!`, `truncate_recursive!`) and the deterministic rank and frame helpers shared with `BUG/Randomised` (preselection, the Test A tail, per-sector room, reorthonormalisation). |

The sweeps that call these, bond by bond, are in `../Sweeps/`.
