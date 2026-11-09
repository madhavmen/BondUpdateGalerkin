# `test/`

`Pkg.test()` runs `runtests.jl`, which includes every test file in its own module (helpers of one
file cannot leak into another) and prints each file's wall time as `TESTFILE-TIME`. Keep the total
near five minutes. `Pkg.test(test_args = ["aqua-full"])` runs only the full Aqua checks.

The folders mirror `src/`: the tests of `src/X/Y/file.jl` live in `test/X/Y/`. Every test grades
against an independent reference (dense, analytic, free-fermion, manufactured or a frozen run),
never against the code under test.

## Layout

| folder | file | what it checks |
|---|---|---|
| `BUG/Core/TensorNetworks/` | `test_symmetric_mps.jl` | The `SymMPS` container and its canonical form. |
| | `test_sectors.jl` | Reachable sectors, and that `K1` lies inside `U0`'s range after `P_perp`. |
| | `test_fusion_cache.jl` | The fusion-basis memo: a hit returns exactly what a miss computes, the tag and the symmetry mode are part of the key, the memo stays inside its element budget. |
| | `test_mpo.jl` | An on-site MPO and the exact MPO-MPS product, with and without U(1). |
| `BUG/Core/Sweeps/` | `test_frames.jl` | The canonical two-site bond snapshot `BondFrame`. |
| | `test_frontier_compression.jl` | `frontier_tol`: compressed closure frontiers. |
| `BUG/Core/Interfaces/` | `test_state_conformance.jl` | The state contract, one suite for every state type and symmetry (QTT fields included). |
| `BUG/Core/LocalSolvers/` | `test_krylov.jl` | The Krylov exponential and the root methods' stability functions. |
| | `test_krylov_weights.jl` | The Krylov stopping weights against a dense exponential; the root's guarded tail; `_tridiag_eigen`. |
| `BUG/Core/RankAdaptation/` | `test_basis_budget.jl` | The budget object, the allocation, the work ceiling, the ledger, and refusal of retired keywords. |
| | `test_bond_work_ceiling.jl` | `work_ceiling_rule = :bond` and `bond_ceiling_factor`. |
| | `test_krylov_ranked.jl` | `krylov_rank_tol`: ranked against sequential Krylov directions at order 2 and 3. |
| `BUG/Randomised/` | `test_expansion.jl` | Test A on singular-value lists, and the DD-QR expansion on a two-spin entangling step. |
| `BUG/TimeEvolution/` | `test_step_keep_states.jl` | Kept states (`keep_states`, `keep_tols`) and frames-only builds. |
| | `test_parallel_half_sweeps.jl` | Parallel half-sweeps equal serial ones (order 1 and 2). |
| | `test_fixtures_b155550.jl` | Regression against the frozen runs in `fixtures/b155550/` (XX chain under U(1) and without symmetry). |
| | `test_stage_solvers.jl` | The implicit Runge-Kutta stage solver in the root space, GMRES variants included. |
| `BUG/ITE/` | `test_schedules.jl` | The imaginary-time schedules. |
| `QMB/ClosedSystem/` | `test_xx.jl` | The U(1) XX models at spin 1/2 and spin 3. |
| | `test_z2_ising.jl` | The Z2 Ising local space, states and the analytic energy. |
| | `test_tfim.jl` | The TFIM MPO against a dense TFIM, both symmetries, refusals. |
| | `test_oat.jl` | One-axis twisting: the MPO against `pair_mpo` and the dense operator. |
| `QMB/Observables/` | `test_local_observables.jl` | Site expectation values and the `S^z` observables. |
| `QMB/OpenSystem/MPDO/` | `test_liouville.jl` | Vectorisation, the Lindblad MPO against the dense Liouvillian, trace, Hermiticity and positivity through BUG steps. |
| `QMB/OpenSystem/MCWF/` | `test_quantum_jumps.jl` | The quantum-jump engine on a 4-site chain against dense operators. |
| `QTTS/Tooling/` | `test_grid.jl` | Bit orderings and grids round-trip in every layout. |
| | `test_cores.jl` | Sampled functions, separable embedding, the round trip through `SymMPS`. |
| | `test_operators.jl` | TT operators against dense ones in both layouts; the BUG MPO applies the same matrix. |
| | `test_derivatives.jl` | Derivative MPOs and 2D assembly against dense stencils; boundary-stencil convergence. |
| | `test_nonlinear_step.jl` | `qtt_bug_step!`: convergence on an exact nonlinear solution, tolerances, solver routing, and the frozen Heston and GP runs. |
| `QTTS/Models/Heston/` | `test_heston.jl` | The Heston generator against dense, mass conservation, a manufactured affine solution, the BUG step against the dense exponential. |
| `QTTS/Models/GrossPitaevskii/` | `test_gross_pitaevskii.jl` | The GP right-hand side on QTT against dense. |
| `Baselines/` | `test_baselines.jl` | TDVP2 on the XX chain against its frozen run and free fermions; a QTT TDVP2 step against the dense exponential. |
| `quality/` | `import_boundaries.jl` | Module boundaries: no forbidden module references, `BUG/Core` uses no higher-layer name, baselines never run a BUG step. |
| | `aqua.jl` | Package hygiene with Aqua (undefined exports, ambiguities, compat, piracy). |

## Shared helpers (`common/`)

Included by the test files that need them; test-side only, never reachable from `src/`.

| file | contents |
|---|---|
| `dense_reference.jl` | Dense `2^L` Hamiltonians and states (`dense_heisenberg`, `dense_state`, `dense_exact_propagate`, long-range chains) for small chains. |
| `free_fermion.jl` | The analytic XX reference via Jordan-Wigner (`xx_free_fermion_sz` and friends). |
| `free_fermion_open.jl` | Open-system free-fermion references: the time-dependent Lyapunov solution and the TFIM BdG reference. |
| `liouville_fixtures.jl` | Dense Liouvillians and the XX / Bose-Hubbard Lindblad models of the MPDO tests. |
| `qtt_fixtures.jl` | Grid re-indexing, random TTs, dense stencils and the Heston grid for the QTT tests. |

## Frozen runs (`fixtures/`)

`fixtures/b155550/` holds JSON trajectories frozen from BUG-Julia `b155550`; its README records how
they were made and the tolerances they are compared at.
