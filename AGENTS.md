# BondUpdateGalerkin contributor and coding-agent instructions

These rules apply to every code change. They are instructions, not evidence that checks ran.

1. **Inspect first.** Read the local implementation, caller interfaces, tests, and mathematical derivation before changing it. State whether the change is purely structural or alters numerical behavior.
2. **Respect ownership.** `src/BUG/` owns representation-independent algorithms; `src/QMB/` owns quantum models/states/observables; `src/QTTS/` owns grids, discretizations, PDEs, and QTT geometry; `src/Baselines/` owns comparison methods. During migration, map legacy paths before moving them. Do not duplicate BUG sweeps by domain.
3. **Preserve scientific contracts.** Never silently alter gauge conventions, conjugation, tensor-leg order, boundary conditions, charge labels, operator signs, time-stepping order, or truncation semantics. Explicitly distinguish unitary, dissipative, non-Hermitian, and nonlinear dynamics.
4. **Make results testable.** Supply deterministic tests and a suitably independent reference for each algorithmic change (dense, analytic, manufactured, or convergence reference). Record assumptions and numerical tolerance meanings.
5. **Write readable Julia.** Use descriptive identifiers, clear function boundaries, type-stable performance kernels, `!` for mutation, validated constructors, explicit errors, and docstrings for user-facing symbols. Document leg ordering and shape expectations where tensors are manipulated.
6. **Make small scoped changes.** Avoid unrelated formatting, generated artifacts, new dependencies, or large research experiments in a refactor PR. Retain historical reproductions before archiving.
7. **Separate errors.** Do not conflate time-step, local Krylov/implicit, spatial discretization, and rank truncation errors. Report `tol_basis`, rank limits, residuals, discarded weight, and tolerance saturation.
8. **Use proper checks.** Run relevant unit and reference tests, formatter, and package checks. Explicitly say which checks were run, failed, or unavailable; never assert green CI without evidence.
9. **Protect the public repository.** Do not commit secrets, local paths, private Telum/LurCGT code, license-restricted assets, large output folders, or unverified figures.
10. **Summarize work.** Finish with files changed, API/numerical impact, test commands and results, known limitations, and next action.

Detailed guidance: `agents/architecture.md`, `agents/code-quality.md`, `agents/scientific-validation.md`, `agents/testing.md`, `agents/definition-of-done.md`.
