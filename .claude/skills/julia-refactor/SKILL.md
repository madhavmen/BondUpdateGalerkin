---
name: julia-refactor
description: Use when reorganizing or polishing BondUpdateGalerkin Julia source files, modules, public interfaces, imports, documentation, tests, or folder structure without intending to alter numerical behavior. Keep changes minimal, type-stable, and fully traceable to the old implementation.
---

# Julia refactor workflow

1. Identify every source file, caller, export, include, test and README reference affected by the requested move. Produce a compact old-path to new-path map.
2. Read `agents/architecture.md` and `agents/code-quality.md`. Confirm the destination (`BUG`, `QMB`, `QTTS`, `Baselines`, or non-`src` research tooling) and avoid circular imports and type piracy.
3. Freeze a small old-behavior result before editing, with seed and package versions. Do not mix numerical improvements into a structure-only PR.
4. Move/split one responsibility at a time. Keep public docstrings and check mutating `!` functions, tensor leg names, type stability, and output side effects.
5. Run formatter, `using BondUpdateGalerkin`, focused `Pkg.test()` targets where configured, and all available frozen regression checks. State any unrun tests.
6. Report exact moved paths, new API decisions, changed behavior (prefer none), unresolved compatibility points, and tests.

Prefer an efficient focused inspection over reading the entire source tree. Do not reformat unrelated files.
