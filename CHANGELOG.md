# Changelog

All notable changes, numerical behaviour first. Versions follow semantic versioning; before 1.0 a
minor version may change the public API.

## 0.1.0 (unreleased)

The package as moved from BUG-Julia `b155550` (tag `pre-bondupdategalerkin`), restructured into
`BUG`, `QMB`, `QTTS` and `Baselines` by `docs/dev/implementation_plan.md`. Every legacy file's
destination or archive decision is in `docs/dev/migration-map.md`; the moved engine reproduces the
frozen `b155550` runs in `test/fixtures/b155550`.

### Numerical behaviour

- The nonlinear stage solver's GMRES stops at an exact Krylov breakdown with a zero pivot and keeps
  the solution of the space it has; before, a stage system smaller than the GMRES tolerance could
  resolve failed with a singular triangular solve.
- The basis selection of a BUG step is a strategy (`expansion` keyword, default `DDQRExpansion`);
  the default reproduces the legacy randomised expansion.
- Unchanged, but now pinned: the compressed MPDO of test OS-U3 has a transient negative eigenvalue
  (-1.76e-4 after four steps at `b155550` too); the test compares with the frozen run instead of the
  legacy bound `-1e-4`, which `b155550` failed.

### API

- `mpo_from_terms` and `onsite_mpo` take the local identity as a required keyword `Iloc`; model
  MPOs come from `QMB` (`xxz_mpo`, `oat_mpo`, ...).
- The one-site CBE baselines (`tdvp_cbe1s_step!`, `dmrg_cbe1s!` and its named arms) take an `MPO`
  only; `tdvp1_cbe_step!`, `tdvp2_step!` and `Baselines.GroundState` keep `XXZChain`.
- `QTTOrdering` holds the bit placement of any number of axes; `qtt_embed` takes an ordering or a grid.
- The implicit Runge-Kutta stage solver (`rk_stage_solve`) works on any representation that
  implements the frame interface (`frame_state`, `frame_project`, `frame_operator_matrix`,
  `frame_conj`); `QTTFrames` implements it for dense QTT cores.
- Removed: `gp_bug_step!` (use `qtt_bug_step!` with `GPFlow`, or `frozen_flow` for the lagged
  density), `sector_report` and `sector_report_right` (broken at `b155550`), `ising_chain` (no
  method at `b155550`).

### Added

- Derivative MPOs with second-order one-sided boundary rows, checked for second-order convergence;
  a manufactured affine solution for the Heston generator.
- Self-checking examples (`examples/`), a Documenter manual (`docs/`), maintained benchmark drivers
  (`benchmarks/`), CI jobs for tests on two Julia versions, formatting, examples, docs, and a
  weekly benchmark workflow.
