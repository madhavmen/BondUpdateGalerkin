# BondUpdateGalerkin Claude Code project instructions

@AGENTS.md
@agents/definition-of-done.md

## Mission

Maintain a scientifically correct, externally usable Julia package with a single representation-independent BUG algorithm supporting both quantum many-body MPS and PDE/QTT states. The above imported rules are **always applicable**.

## First steps for every implementation prompt

1. Determine target area and inspect only relevant files, call sites, tests, and docstrings.
2. Determine if this is a structural refactor or a numerical/mathematical change. For numerical changes, state the mathematical claim and independent verification before editing.
3. Choose appropriate on-demand workflow: `/julia-refactor`, `/scientific-change`, `/bug-validation`, `/qtt-pde-validation`, `/performance-regression`, or `/public-release` as relevant. Skills are not all activated for every task.
4. Implement the smallest coherent change, run available checks, and report evidence honestly.

## Package commands (verify against live repository before relying on them)

- Package load: `julia --project=. -e 'using BondUpdateGalerkin'`
- Default suite: `julia --project=. -e 'using Pkg; Pkg.test()'`
- Examples (each checks itself): `julia --project=. examples/QMB/xx_quickstart.jl`; list in `examples/README.md`.
- Manual: `julia --project=docs -e 'using Pkg; Pkg.instantiate()'`, then `julia --project=docs docs/make.jl`.

## Code locations

- `src/BUG/`: `Core/` (interfaces, MPS/MPO storage, environments, local solvers, rank adaptation, sweeps), `Randomised/` (sketch-based CBE), `TimeEvolution/` (real time), `ITE/` (imaginary time).
- `src/QMB/`: `Tooling/`, `Models/ClosedSystem/`, `Models/OpenSystem/{MPDO,MCWF}/`, `Observables/`.
- `src/QTTS/`: `Tooling/` (grids, QTT and MPO construction), `Elementwise/`, `Models/{Heston,GrossPitaevskii}/`.
- `src/Baselines/`: independent comparison integrators and exact references.
- `benchmarks/QMB/`, `benchmarks/QTTS/`, `benchmarks/kernels/`: maintained benchmark drivers (`benchmarks/README.md`).
- `examples/QMB/`, `examples/QTTS/`: self-checking examples; `studies/`: research code outside the package; `docs/src/`: the manual.

The full target tree, folder responsibilities and dependency rules are in `docs/dev/implementation_plan.md` Section 2; the legacy-file move map is in Section 8. Consult `agents/architecture.md` for dependency rules and `agents/scientific-validation.md` for scientifically consequential changes. The migration from BUG-Julia is complete; `docs/dev/migration-map.md` records every legacy file's destination or archive decision.

Do not run cluster-scale simulations or produce generated figures without an explicit task requirement. Prefer deterministic fast tests before expensive verification. Never call a change "validated" if the checks did not run.
