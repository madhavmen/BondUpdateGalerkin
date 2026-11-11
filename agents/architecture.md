# Architecture and ownership

## Core invariant

The algorithm appears **once** under `src/BUG/`. QMB and QTTS supply the representation, evolution problem, operator actions, metric, geometry, constraints and observables needed by it. Baselines supply independent comparison methods.

## Allowed directions

- `BUG` depends on its interfaces and selected tensor backends, not QMB/QTTS model definitions.
- Inside `BUG`: `Core` <- `Randomised` <- `TimeEvolution`, `ITE` <- `solve.jl`. `Core` never references the later layers; its sweeps take a basis-selection strategy, and `Randomised` adds methods for it.
- `TimeEvolution` and `ITE` act on any state implementing `Core/Interfaces`, whether a QMB wavefunction MPS or a QTTS quantics TT, with no domain branches.
- `QMB` extends BUG interfaces for spin/fermion wavefunctions and density operators. `Models/OpenSystem/MPDO` and `Models/OpenSystem/MCWF` call the BUG step rather than carrying their own integrator.
- `QTTS` extends BUG interfaces for sampled/encoded fields, bit registers, PDE discretizations, boundaries, quadrature and non-Hermitian/nonlinear actions. Pointwise operations live in `QTTS/Elementwise`.
- `QMB` and `QTTS` never reference each other; shared MPS/MPO storage lives in `BUG/Core/TensorNetworks`.
- `Baselines` may use `BUG/Core`, `BUG/Randomised` and the flow definitions of `BUG/TimeEvolution/flows.jl` (a vector field, not an integrator), never a BUG step; `test/quality/import_boundaries.jl` checks that no baseline names one.
- The implicit Runge-Kutta stage solver (`BUG/TimeEvolution/stage_solvers.jl`) works through a frame interface (`frame_state`, `frame_project`, `frame_operator_matrix`, `frame_conj`); a representation implements it (`QTTS/Tooling/frames.jl` for dense QTT cores).
- Tests, examples and benchmarks consume public API, not the inverse.

## Source checklist

1. Is the mathematics generic TT/MPS linear algebra? Keep in `BUG` or use an upstream backend.
2. Is it a Hamiltonian, charge rule, quantum initialization or measurement? Put in `QMB`.
3. Is it a grid map, differential operator, physical PDE, BC or weighted inner product? Put in `QTTS`.
4. Is it a competing solver used for comparison? Put in `Baselines`.
5. Is it a configuration parser, plotter, Slurm launcher or publication script? Keep outside `src/` unless intentionally supported as public infrastructure.

## Boundary design

Before introducing new abstract types, enumerate the minimal needed methods for site access, canonicalization, local action, environments, projection, rank updates, inner product and diagnostics. Use Julia multiple dispatch/traits with thin adapters; favor code generation-free, type-stable specialization. No type piracy; no domain-name conditionals in the inner BUG sweep.

Document whether a projected tensor uses Euclidean or quadrature-weighted metric, whether QR/SVD preserves U(1)/Z2 sectors, and whether nonlinear RHS is evaluated at each required stage. Avoid assuming that Heston's non-Hermitian operator may be treated with Hermitian Lanczos or that nonlinear GPE is a constant linear MPO.

## Krylov recurrences

- One recurrence per file: the local frame enrichment's Lanczos (`Core/RankAdaptation/krylov_lanczos.jl`) and Arnoldi (`krylov_arnoldi.jl`) chains share only the dispatcher, the stopping rule and the inner products (`krylov_recurrence.jl`). A new recurrence gets its own file and an entry in `KRYLOV_RECURRENCES`.
- Lanczos is valid only for a Hermitian generator. The choice follows `cbe_bug_step!`'s `hermitian` keyword (`krylov_mode = :auto`, the default): every non-Hermitian step (Lindbladian, Heston, any non-Hermitian `H_eff`) must pass `hermitian = false`, which also switches the root solve to Arnoldi. Never hard-code `:lanczos` in a path that can see a non-Hermitian generator.
- Defaults are set from measured evidence and say where it is (`DDQRExpansion`'s docstring, `DEFAULT_BOND_CEILING_FACTOR`); a regression test that replays frozen runs names the settings it was frozen with instead of relying on the defaults.
