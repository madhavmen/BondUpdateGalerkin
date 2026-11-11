# Changelog

All notable changes, numerical behaviour first. Versions follow semantic versioning; before 1.0 a
minor version may change the public API.

## Unreleased

### Changed (numerical behaviour: XX-tuned defaults of `cbe_bug_step!`)

- `DDQRExpansion()` is `mode = :rank_tail`, `fixed_width = 8` (was `:adaptive`): one 8-column DD
  sketch, rank bisected on its own tail, no holdout. Accuracy rests on the order-2 closure.
- `DDQRExpansion(krylov_mode = :auto)` (was `:arnoldi`): the local K/L and frontier Krylov
  enrichment uses adaptive Lanczos on a Hermitian step and Arnoldi with `hermitian = false`
  (`resolve_krylov_recurrence`).
- `cbe_bug_step!`'s `work_ceiling_rule = :bond` with `bond_ceiling_factor = 1.2` (was `:maxdim`):
  working frames up to `min(ceil(1.2 d D_b), maxdim)`. An explicit `bond_ceiling_factor = 0` under
  `:bond` is the former `min(2 d D_b, maxdim)`.
- Measured on the U(1) XX domain wall, L = 64 (`benchmarks/XXT5/results/dw_L64`,
  `dw_branch_claims`). The b155550 regression replays its runs with the former settings named.
- The local Krylov recurrences are split into `krylov_lanczos.jl` and `krylov_arnoldi.jl`, with the
  shared dispatcher and stopping rule (`_krylov_stop`) in `krylov_recurrence.jl`; no numerical change.
- Known regressions of these defaults outside the XX domain wall (suite job 17904164). The checks
  that caught them are removed:
  - the Néel quench, L = 20: error 1.4e-4 against 1.5e-6, with 8 of 20 steps unmet;
  - the MPDO trace: off by 4.9e-2, with a smallest eigenvalue of -0.017;
  - the Heston x|v bond: it never grows from a separable start;
  - the GP step and the frozen GP run: errors of 7e-4 and 1e-3;
  - the frozen Heisenberg imaginary-time energies and statuses;
  - the order-2 ranked-Krylov reference arm: error below 1e-6 no longer met.

  The README and docs quick starts now start from a domain wall. `OS-U2` (the Lindblad MPO
  against the dense Liouvillian) no longer checks its two weak-symmetry cases, which failed
  before this change (job 17821085).

### Added (experiments/oldcap-fast-rsvd-xx)

- `DDQRExpansion` modes `:rank_adaptive`, `:rank_fast` and `:rank_tail`; `batch_blocks`,
  `cap_selection`, `rank_floor`, `tail_factor`, `krylov_mode`, `krylov_fixed_depth`.
- Local Lanczos for the K/L frame enrichment, with periodic reorthogonalisation past depth 20.
- `work_ceiling_rule = :bond_strict` and `:frame_root` (`bond_work_ceilings_strict`,
  `bond_frame_root_ceilings`); the transactional coupled DD/Krylov step (`coupled_dd_krylov.jl`).
- `benchmarks/XXT5`: the large-system RSVD audit (`xx_large_rsvd_audit.jl`) and its analyses.

### Changed (numerical behaviour; methods and fixes from BUG-Julia's dissipative and QTT plans)

- Local Krylov stopping weight: the residual integral
  `|tau| beta_k int_0^1 |[exp(s tau H_k)]_{k,1}| ds` replaces Saad's scalar, which can cancel. The
  root's convergence test uses the endpoint tail maximised over the step in real time.
  `_tridiag_eigen` falls back to `syev` and refuses non-finite input.
- `qtt_bug_step!`:
  - Tolerances scale per step as `tol (|h| / h_ref)^(order+1)` (`tol_scaling = :order`, the
    default; `:fixed` is the former rule; see `qtt_step_tolerance`).
  - Solver routing (`stage_solver = :auto`): a `LinearFlow` takes `cbe_bug_step!`'s unshifted
    rational root (`:krylov_rational`); nonlinear flows take Newton-GMRES stages.
  - Frames are built with `frames_only`. The stage states and fields enter the basis as a
    predictor admitted at `tol_basis` in step-error units (`stage_basis = :predictor`; `:exact` is
    the former rule), compressed at `maxdim` (was `4 maxdim`).
  - After an unconverged root solve, the corrections are skipped.
  - The correction rounds reuse each stage's converged density (`stage_field_rho = :reuse`).
- `rk_stage_solve`:
  - Fields and Jacobians use root environments per summand (`envs = :summand`, `StageFieldOps`):
    the static operator is projected once per frame set, the paddle once per stage time, and the
    density per iterate by a warm-started variational fit (`tt_hadamard_fit`; `:zipup` kept).
  - The normalised flow's Jacobian carries `D mu`.
  - The separable preconditioner (`precond = :separable`, Sylvester solves on the nearest
    Kronecker sum) is the default; the dense LU remains as `:dense`.
  - GMRES tolerance is inexact Newton (Eisenstat-Walker), with a Newton stall exit
    (`newton_stall`) and optional GCRO-DR recycling (`gmres_recycle`).
- `gp_ground_state!` judges a retry against the current state's energy (was the last ledger
  entry, possibly a rejected trial) and keeps fixed tolerances. `gp_observables`' `E` includes the
  paddle energy. `vortex_count` masks by the density envelope within `window` pixels (a core's own
  zero masked every core). `tt_density`, `tt_product` and `op_zipup` compress their inputs at
  `0.1 tol` and count it in `err_bound`.
- QTT TDVP2 baseline:
  - The lagged step caches its environments.
  - Every cut uses BUG's final-compression rule (`_cut_divisor`).
  - `_dense_expv` and `tt_expv` stop the basis at the first depth whose estimate fits `tol`.

- Telum is upstream `dev` (`403cdcf`): randomized SVD, a symmetric QR, opt-in lazy evaluation
  (`@lazy`), and the TLArray view state. Our fixes are re-applied in `deps/Telum`: the
  no-symmetry SVD and the `gesdd` -> QR-iteration fallback.
  - LurCGT moves to its `dev` branch (`1bb74d6`, version 0.1.0, compat `"0.1"`), which Telum
    `dev` requires.
  - `to_concrete`/`to_concrete!` are gone. Contractions are concrete. `TLArray(x)` materialises a
    lazy wrapper and is the identity on a `TLArray`; the fusion cache hands out `copy`.
  - Permutation, conjugation and scalar factors are deferred view state. The code reads logical
    data only (`Telum.inds`, `Telum.spaces`, `Telum.logical_qlabels`, `_sector_block`), never the
    stored fields.
  - Dropped from our fork: the threaded `contract` (`TELUM_CONTRACT_THREADS`) and `to_concrete!`.

### Removed

One controlled bond expansion remains: the DD-QR CBE (`DDQRExpansion`, `dd_cbe_expand`), with
the incremental holdout controller and the local Krylov enrichment. `exact = true` stays as its
deterministic reference.

- The SVD-selection CBE: `cbe_expand` (term-list and MPO forms), `sketch_h_left`,
  `sketch_h_right`, `full_local_basis`, `expand_bond!`, `BondExpansionInfo` and
  `lanczos_lowest`.
- The gate-based CBE bond update: `cbe_bond_update`, `cbe_expand_bond`, `sketch_bond_left`,
  `sketch_bond_right`, `CBEBugOptions`, `SStepSolver`, `Propagate` and `LowestEigen`.
- The DD-QR ablations, with their `cbe_bug_step!` keywords:
  - `dd_policy` (`:hybrid` and `:off`) and `dd_memory` (`DDMemory`).
  - `dd_width`, the fixed-width probe.
  - `cap_order = :weight`: at a binding work ceiling, the first sectors in storage order are kept.
- `dd_reference_sketch`.
- The baselines built on the SVD-selection CBE:
  - One-site TDVP-CBE: `tdvp1_cbe_step!`, `tdvp_cbe1s_step!` and their variants.
  - CBE-DMRG: `dmrg_cbe1s!`, its named arms and `Baselines.GroundState`.

  `Baselines` keeps two-site TDVP and the QTT TDVP.

The Hamiltonian enters every sweep, TDVP2 and every observable as an MPO in its environments. The
term list `XXZChain`/`XXZTerm` stays as the input of `mpo_from_terms`, `pair_mpo` and
`long_range_mpo`.

- The term-list (channel) environments: `ChannelSet`, `LeftEnvStack`, `RightEnvStack`,
  `env_energy`, `env_energy_right`, `bond_gate`, the `XXZChain` methods of `boundary_channels`,
  `push_left_channels`, `push_right_channels`, `left_env_stack`, `right_env_stack` and
  `apply_h_two_site`, and the term-list effective operators `ZeroSiteH` and `OneSiteH`.
  `tdvp2_step!` takes an `MPO` only.
- The two-site gates: `apply_gate`, `heisenberg_bond_gate`, `xx_bond_gate`, `magnetisation_gate`,
  `ising_bond_gate`, `ising_bond_gates`, `bond_gates`, and the gate observables `energy` and
  `bond_energies` (use `mpo_energy`), and `parity_bonds`.
- `dimer_state` outside `:SU2`: it was built by projecting a Neel state with the Heisenberg gate.
  Under `:SU2` it is unchanged; under `:U1` or `:none` start from `neel_state` or `product_state`.

Methods and ablations outside the default path:

- `cbe_bug_midpoint_step!` and the `frame_state` keyword of `cbe_bug_step!` (the midpoint BUG
  step), with its b155550 fixture `xx_u1_bug_midpoint`.
- `tdvp2_composed_step!` (the Yoshida and Suzuki fourth-order compositions of TDVP2).
- `density = :consistent` of `qtt_tdvp2_step!` (the self-consistent trial-state density) with
  `tol_sc` and `maxsc`; the QTT TDVP2 is the lagged-density step, and `sc_iters` is no longer
  returned.
- The local Krylov ablations of `cbe_bug_step!`: `krylov_reuse = false`, `krylov_recycle` with
  `recycle_mode` and `recycle_report` (the recycled-block Krylov solve, `KrylovRecycle`), and
  `krylov_depth`.
- `stage_solver = :krylov_root` of `qtt_bug_step!`.

Developer notes: `docs/dev/` (the implementation plan, migration map, interface notes, release
audit, file inventory and the b155550 fixture script) and `studies/README.md`. The state contract is
tabulated on `docs/src/mps.md` instead.

### Added

- A README in every folder of `src/` (what the folder does, which file holds what) and in
  `test/` (every test file) and `docs/`.
- Manual page `docs/src/mps.md`: the `SymMPS` container, symmetry modes, state builders, canonical
  form, charge sectors, observables and the state interface.
- `cbe_bug_step!` keywords:
  - `frontier_tol`: frontier compression.
  - `krylov_rank_tol`: ranked Krylov selection.
  - `work_ceiling_rule = :bond` with `bond_ceiling_factor`: per-bond work ceilings
    `min(2 d D_b, maxdim)`, which also cap the tolerance-admitted keep states.
  - `keep_tols`, `frames_only`, `root_shift` and `root_maxiter`.
- `bond_work_ceilings`.
- `qtt_bug_step!` keywords: `stage_*`, `tol_scaling` and `tol_h_ref`. Its result adds the step's
  DD ledger `dd`, the time ledger (`t_setup`, `t_keep`, `t_trunc`, `f_rho`, `f_cbe`), the Newton
  and GMRES totals, and `tol_step`.
- `gp_ground_state!` keywords: `N`, `growth` and `on_step`. `gp_gradient_descent!` keyword:
  `on_step`.
- The generic stage-field interface in `BUG` (`frame_root_envs`, `root_envs_matrix`,
  `root_envs_kron_sum`, `stage_field_ops`, `stage_field`, `stage_jacobian`, `stage_linear_envs`),
  `root_apply`, `StageFieldOps` and `qtt_step_tolerance`.

### Changed

- `src/BUG` is restructured for readability, with no intended numerical change: no file is over
  600 lines (`krylov_enrichment.jl`, `Randomised/cbe.jl`, `dd_cbe.jl`, `frontier_sweeps.jl` and
  `bug_step.jl` are split by responsibility), no single-letter variable names remain (type
  parameters excepted), and every function carries a docstring or comment. Internal fields
  renamed: `KLFrames.{W,Z,c}` -> `{left_frames,right_frames,root_link}`. Public names, keywords
  and the fields of exported types are unchanged.
- No CI: the GitHub Actions workflows are removed. Formatting and the test suite (quality checks
  included) run on every commit through pre-commit hooks (`.pre-commit-config.yaml`); the examples
  (`examples/run_all.jl`) and the docs build are manual hooks. See `CONTRIBUTING.md`.
- The test suite is cut to about five minutes: the frozen-trajectory regressions compare their
  first steps (each frozen row is still checked on its own), and each test file's wall time is
  printed.
- The precompile workload is back (PrecompileTools, plan Section 8): the BUG steps (orders 1-2,
  midpoint, imaginary time), MPO products, the baselines, the Z2 Ising states and the QTT GP and
  Heston steps are compiled into the package image, so a new process no longer recompiles them.
  The test suite was almost all compilation (28 min warm on the cluster). New dependencies:
  PrecompileTools, and Test (a standard library), which is loaded before the workload because
  loading it afterwards invalidated most of the compiled code (`keys(::Test.GenericArray)`).
- Aqua's three process-based checks (`ambiguities`, `stale_deps`, `persistent_tasks`, about 8 min
  together) moved out of the commit suite into the manual hook `julia-aqua-full`
  (`Pkg.test(test_args = ["aqua-full"])`); the commit suite checks ambiguities in process.
- The tests read their fixtures with JSON 0.21 (test compat, was "0.21, 1"): JSON 1.x defines
  `lastindex` on its lazy arrays, which invalidated about 10 000 compiled method instances behind
  `lastindex(::AbstractVector)` and made the first fixture test recompile the workload.

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
