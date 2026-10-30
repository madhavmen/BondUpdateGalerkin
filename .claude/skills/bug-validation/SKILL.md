---
name: bug-validation
description: Use when editing or reviewing the BondUpdateGalerkin basis-update-and-Galerkin K/L/root sweeps, local Krylov or implicit solvers, symmetry-native MPS, randomized CBE, rank adaptation, recycling, truncation, or time-integration order. Produce evidence for numerical correctness and algorithmic choices.
---

# BUG/CBE validation workflow

1. Consult `agents/scientific-validation.md` and the current BUG mathematical derivation. Enumerate frame shapes, gauge/orthogonality identities, local action and root projection.
2. Identify precise effects on left/right sweeps and the root solve; verify that time integrator order is preserved and symmetry sectors are not mixed.
3. Test dense no-symmetry tensors and a small U(1) XX domain-wall MPS on the same shared algorithm interface; add a Z2 parity case if relevant.
4. Compare to an independent small exact/free-fermion result and pre-refactor fixture. Quantify observables, norm, discarded weight, final and maximum bond ranks.
5. Verify `tol_basis` semantics, rank caps, RSVD residual/tail estimators, deterministic sketch seed, local Krylov residuals and saturations. Include a rank/tolerance sensitivity check for logic changes.
6. For a performance claim, call `/performance-regression` rather than substituting a wall-time anecdote.
7. Report equations checked, measurable errors, test commands/results, and limitations.

Do not force all evolution operators into Hermitian Lanczos. Use general non-Hermitian or nonlinear solver capabilities as dictated by the problem.
