# `src/BUG/` — the engine

Module `BUG` (`BUG.jl`): one rank-adaptive basis-update-and-Galerkin engine for any tensor-train
state. The folders are include directories, not submodules (a submodule named `Core` would shadow
Julia's `Core`), and they are layered: each may use only the ones above it in this table.

| folder | layer | what it holds |
|---|---|---|
| [`Core/`](Core/README.md) | 1 | State interfaces, MPS/MPO storage, environments, local Krylov solvers, rank adaptation, sweeps, step records. |
| [`Randomised/`](Randomised/README.md) | 2 | The randomised controlled bond expansion (DD-QR CBE), plugged into the sweeps as a `BasisExpansion` strategy. |
| [`TimeEvolution/`](TimeEvolution/README.md) | 3 | The real-time step `cbe_bug_step!`, its stages, nonlinear flows and the implicit Runge-Kutta stage solver. |
| [`ITE/`](ITE/README.md) | 3 | Imaginary time: `ite_step!`, `ite_evolve!` and the step-size schedules. |

`Core/` never uses a name that only a later layer defines (`test/quality/import_boundaries.jl`):
the sweeps reach the randomised expansion only through the four generic functions of
`Core/RankAdaptation/basis_expansion.jl`.

## The include order

`BUG.jl` includes the files in an order that matters: a file may use, in a signature or a struct
field, only the types of files above it. Function bodies resolve names at call time, so they are
free. When adding a file, put it after every file whose types it names.

## Where to start reading

- The step: `TimeEvolution/bug_step.jl` (`cbe_bug_step!`), then `Core/Sweeps/half_sweeps.jl`.
- The basis growth at one bond: `Randomised/dd_cbe_expand.jl`, then
  `Core/RankAdaptation/krylov_enrichment.jl`.
- The tolerances: `Core/RankAdaptation/basis_budget.jl`.
- The MPS container: `Core/TensorNetworks/symmetric_mps.jl` and the manual page `docs/src/mps.md`.
