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

## The local Krylov recurrence

Tests C/D grow each K/L frame (and each frontier frame of the closure) by a local Krylov chain on
the one-site generator, stopped adaptively: a new direction is added until its residual-integral
weight over the step is within the bond's share of `tol_krylov` (`_krylov_stop`; a fixed depth,
`krylov_fixed_depth > 0`, is an experimental baseline). The recurrence is chosen per step by
`DDQRExpansion(krylov_mode)`: `:auto` (default) is resolved by `cbe_bug_step!` from `hermitian`,
Lanczos on a Hermitian step and Arnoldi otherwise (`resolve_krylov_recurrence`). Both build the
same directions; Lanczos needs one inner product per step instead of CGS2 against the whole basis
(XX domain wall, L = 64: ~0.74x the Krylov-stage time, same error and chi). One recurrence per file.

## The work ceiling

Every DD-QR CBE and local Krylov stage of bond `b` is capped (`cbe_bug_step!`'s
`work_ceiling_rule`, default `:bond`) at `min(ceil(1.2 d D_b), maxdim)`, `D_b` the bond's dimension
at the start of the step (`bond_work_ceilings`, `DEFAULT_BOND_CEILING_FACTOR`). It held on the XX
domain wall with no violations and no bond-dimension bloat. Two known limits, measured on a Neel
quench: a product state's first step gets `ceil(1.2 d)` per bond, which can be below what a bulk
quench needs, and a bond at `maxdim` leaves its frame no room. `work_ceiling_rule = :maxdim` (one
ceiling `d*maxdim`) avoids both.

## Files

| file | contents |
|---|---|
| `basis_budget.jl` | `AdaptiveBudget` (the three tolerances), `TOL_DEFAULTS`, `basis_allocation`/`BasisAllocation` (how `tol_basis` is split over bonds and tests), `indicator`, the work ceiling (`work_ceiling`, `bond_work_ceilings`, `WorkCeiling`), and the ledger: `AdaptiveLedger`, `merge_ledger!`, `budget_status`, `budget_diagnostic`, `_log_bond!`. |
| `basis_expansion.jl` | The `BasisExpansion` interface the sweeps call: `expand_frame`, `sketch_closures`, `folded_sketch_closures`, `sampled_leak`. `BUG/Randomised` supplies the `DDQRExpansion` methods. |
| `krylov_enrichment.jl` | The half-sweep Krylov stage (Tests C/D): the step weight of a Krylov direction (`_kry_weight`, a residual integral), `_extend_frame`, `_append_ranked` (for `krylov_rank_tol > 0`), `_krylov_enrich`, and `_record_krylov!`. |
| `krylov_recurrence.jl` | The choice of local recurrence `_krylov_enrich` drives (`_krylov_basis`, `KRYLOV_RECURRENCES`), the shared stopping rule `_krylov_stop` and inner products. |
| `krylov_lanczos.jl` | `_lanczos_chain`: three-term Lanczos, Hermitian one-site generators only (the default on a Hermitian step). |
| `krylov_arnoldi.jl` | `_arnoldi_chain`: CGS2 Arnoldi, any generator (the default when `hermitian = false`). |
| `krylov_closure_frames.jl` | The order-p closure's frames (`KLFrames`), their environment caches, `KrylovFrontier`, `frames_from_state`, the projections of the state and its frontiers onto the frames. |
| `krylov_closure_loop.jl` | The closure drivers `_krylov_closure!` and `_single_pass_closure!`: loop over the frontiers `j = 1..p-1` until each one's outside action is within its share. |
| `krylov_frontier_bonds.jl` | One frontier bond: frames with a prescribed isometry, the leak measurements, the bond action, and `_frontier_expand` (the bond expansion inside the closure). |
| `truncation.jl` | The final rank truncation (`truncate_sweep!`, `truncate_recursive!`) and the deterministic rank and frame helpers shared with `BUG/Randomised` (preselection, the Test A tail, per-sector room, reorthonormalisation). |

The sweeps that call these, bond by bond, are in `../Sweeps/`.
