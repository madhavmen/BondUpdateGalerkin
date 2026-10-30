---
name: scientific-change
description: Apply to BondUpdateGalerkin numerical algorithm, physics model, PDE operator, symmetry, integration, truncation, or solver-stability changes. Verify the governing mathematics, independent reference, invariants, error accounting, and claimed scientific result before changing or approving the implementation.
---

# Scientific change workflow

1. Read the affected code/tests and `agents/scientific-validation.md`; identify the exact equation, assumptions, geometry, metric, sign/time convention, and operator properties.
2. Write a short scientific contract: what must remain invariant, what error changes are expected, and which local or global approximation is introduced.
3. Specify the smallest independent analytic/dense/manufactured test and an additional convergence test when applicable. Distinguish time, space, rank, Krylov and nonlinear-stage errors.
4. Implement only after the expected numerical comparison is defined. Preserve old fixtures unless there is a justified mathematical correction.
5. Run appropriate tests; inspect both measured error and diagnostics (rank, rejected CBE directions, discarded weight, local residual, tolerance saturation).
6. Return a compact evidence table: expected result, reference, numerical result if measured, tolerance, pass/fail/not-run, and remaining scientific limitation.

Never describe a plausible plot as proof. Never invent numerical data or claim a test ran when it did not.
