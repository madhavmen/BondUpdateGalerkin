# `src/`

The package source. `BondUpdateGalerkin.jl` defines the top module, includes the four submodules
below in dependency order, and holds the precompile workload.

| folder | module | what it is |
|---|---|---|
| [`BUG/`](BUG/README.md) | `BUG` | The basis-update-and-Galerkin engine: tensor-network storage, environments, local solvers, rank adaptation, sweeps, the randomised bond expansion and the time steps. Independent of what the tensor train represents. |
| [`QMB/`](QMB/README.md) | `QMB` | Quantum many-body problems: local spaces, initial states, model MPOs, open systems, observables. |
| [`QTTS/`](QTTS/README.md) | `QTTS` | Quantics tensor trains for PDEs: grids, QTT construction, operator algebra, the Heston and Gross-Pitaevskii models and the nonlinear QTT step. |
| [`Baselines/`](Baselines/README.md) | `Baselines` | Independent comparison integrators (two-site TDVP on MPS and on QTT). |

## Dependencies

```
BUG  <-  QMB
BUG  <-  QTTS
BUG, QMB, QTTS  <-  Baselines
```

`BUG` never names `QMB`, `QTTS` or `Baselines`; `QMB` and `QTTS` never name each other;
`Baselines` never runs a BUG step. `test/quality/import_boundaries.jl` checks all three.

## How a step flows through the folders

A real-time step `cbe_bug_step!(psi, mpo, tau)`:

1. `BUG/TimeEvolution/bug_step.jl` resolves the keywords and the tolerance budget
   (`BUG/Core/RankAdaptation/basis_budget.jl`).
2. The two half-sweeps (`BUG/Core/Sweeps/half_sweeps.jl`) grow a frame at every bond:
   the randomised bond expansion (`BUG/Randomised/`), then the local Krylov enrichment
   (`BUG/Core/RankAdaptation/krylov_enrichment.jl`).
3. For `order > 1`, the frontier closure (`BUG/Core/RankAdaptation/krylov_closure_*.jl` and
   `BUG/Core/Sweeps/frontier_*.jl`) grows the frames further.
4. The root solve on the centre bond (`BUG/Core/LocalSolvers/`), then one compression
   (`BUG/Core/diagnostics.jl`, `BUG/Core/RankAdaptation/truncation.jl`).

The model (an `MPO` from `QMB/` or `QTTS/`) and the state (a `SymMPS`) are the only inputs; the
engine reads the state only through `BUG/Core/Interfaces/`.
