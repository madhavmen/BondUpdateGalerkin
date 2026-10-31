---
paths:
  - "src/QTTS/**/*.jl"
  - "test/QTTS/**/*.jl"
  - "examples/QTTS/**/*.jl"
---

# QTT/PDE rules

Always document grid, register/axis ordering, boundary conditions, quadrature/metric, time direction and operator linearity. Test grouped/interleaved bits with exact index maps before using them in physics results. Compare derivative/MPO actions and boundary treatment with a dense reference. Heston may be non-selfadjoint; avoid Hermitian-only local solvers. Gross-Pitaevskii with cubic interaction is nonlinear; ensure correct stage-wise RHS evaluation. Use `/scientific-change` and `/qtt-pde-validation` for changes to numerics or PDE operators.

`Tooling/` holds grids, QTT construction, MPO algebra, derivative MPOs and the adapter to `BUG/Core`. `Elementwise/` holds pointwise operations (Hadamard products, `|u|^2`), and each one is checked against the dense pointwise result. `Models/` holds Heston and Gross-Pitaevskii only. No QTTS file may contain its own sweep or Krylov exponential; those come from `BUG`. The one exception is `Tooling/frames.jl`: the dense-core contractions behind the stage solver's frame interface (recorded in the PR 8 row of `docs/dev/migration-map.md`; replacing them with `BUG/Core/Environments` is open work). Nonlinear flows subtype `QTTFlow` and rebuild `H[psi, t]` from the state at every evaluation.
