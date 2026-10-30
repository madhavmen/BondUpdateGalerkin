---
name: qtt-pde-validation
description: Use for QTT state adapters, quantics bit ordering, grids, MPO derivative assembly, PDE discretizations, Heston/Gross-Pitaevskii solvers, boundary conditions, nonlinear RHS, weighted metrics, or PDE validation in BondUpdateGalerkin.
---

# QTT/PDE scientific workflow

1. Read `agents/scientific-validation.md`, the PDE equation and chosen initial/boundary conditions; define forward/backward time and grid-space inner product.
2. Write the discrete site-to-grid map, register ordering and tensor leg convention. Verify small-grid encode/decode round trips for grouped and interleaved layout.
3. Cross-check every derivative and mixed term against dense stencils/matrices with correct boundary handling, including tensor-axis ordering.
4. Check linearity and symmetry before choosing local solvers: Heston is commonly non-Hermitian; cubic GPE is nonlinear and needs stage-consistent RHS evaluation.
5. Validate a minimal analytical/manufactured/known linear limit first, then time-step and grid convergence. For complex vortex shedding use a resolution-converged numerical reference and physical diagnostics, not a claimed general exact solution.
6. Compare fixed and adaptive ranks on a manageable case. Report rank growth, residuals, truncation saturation, physical observables, runtime, and memory.
7. Summarize measured results separately from conjectures and note unsupported boundary/time regimes.
