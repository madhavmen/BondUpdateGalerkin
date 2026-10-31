# Changelog

All notable changes, numerical behaviour first. Versions follow semantic versioning; before 1.0 a
minor version may change the public API.

## Unreleased

### Changed

- No CI: the GitHub Actions workflows are removed. Formatting and the test suite (quality checks
  included) run on every commit through pre-commit hooks (`.pre-commit-config.yaml`); the examples
  (`examples/run_all.jl`) and the docs build are manual hooks. See `CONTRIBUTING.md`.
- The test suite is cut to about five minutes: the frozen-trajectory regressions compare their
  first steps (each frozen row is still checked on its own), and each test file's wall time is
  printed.
- The precompile workload is back (PrecompileTools, plan Section 8): the BUG steps (orders 1-2,
  midpoint, imaginary time), MPO products, the baselines, the Z2 Ising states and the QTT GP and
  Heston steps are compiled into the package image, so a new process no longer recompiles them.
  The test suite was almost all compilation (28 min warm on the cluster). New dependency:
  PrecompileTools.
- Aqua's `persistent_tasks` check is off: it rebuilt the package image, workload included, on
  every test run.

## 0.1.0 (31 October 2026)

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
  a manufactured affine solution for the Heston generator. Each stencil converges at second
  order, but the boundary closure without boundary data is not uniformly stable on the Heston
  generator: the semi-discrete solution's error plateaus (8.6e-5 at 16 x 16, 5.8e-5 at 32 x 32 in
  the Heston quick start) instead of falling at second order.
- Self-checking examples (`examples/`), a Documenter manual (`docs/`), maintained benchmark drivers
  (`benchmarks/`), CI jobs for tests on two Julia versions, formatting, examples, docs, and a
  weekly benchmark workflow.
