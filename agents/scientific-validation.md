# Scientific validity requirements

Before calling a numerical method or scientific result validated, document **what was verified**, against which reference, and within what tolerance.

## Mathematics

1. State `du/dt = F(t,u)` and identify Hamiltonian/operator/RHS, equation coefficients, initial and boundary conditions, real/complex domain, discretization and time direction.
2. State assumptions: Hermitian/non-Hermitian, linear/nonlinear, conservative/dissipative, symmetry support, orthogonality metric, and expected invariants.
3. For BUG changes, identify K/L sweep updates, root Galerkin solve, gauge changes, CBE selection, truncation placement, projection and effective local action.
4. For QTT changes, describe axes, bit-ordering, index mapping, quadrature/inner product, local derivative MPO, boundary treatment and nonlinear evaluation strategy.

## Evidence hierarchy

A. Independent analytic solution or manufactured solution when available.
B. Independent dense/sparse reference on small grids/chains.
C. Grid/time/bond/tolerance convergence with clearly identified reference error.
D. Cross-method consistency with assumptions and solver work matched.

C and D are **not** a substitute for A/B when A/B is available. Long-time plots or visually plausible vortices alone are not validation.

## QMB checks

- Correct Hamiltonian signs and local operator normalization; MPO vs dense benchmark.
- U(1)/Z2 charge preservation when theoretically exact; canonical-gauge and conjugation tests.
- Norm/energy conservation only for the relevant unitary/time-independent conditions; compare observables and analytic XX domain-wall profiles.

## QTTS checks

- Round-trip grouped/interleaved bit encoding; axis sizes and tensor register ordering.
- Stencils, mixed derivatives, BCs and weighted norms vs dense implementations.
- Heston: determine forward/backward PDE, non-Hermitian action, diffusion positivity and meaningful reference conditions.
- GPE: cubic interaction makes the full dynamics nonlinear; validate nonlinear RHS and stage consistency; vortex shedding requires a converged numerical reference rather than a presumed exact solution.

## Error accounting and reporting

Report grid/chain size, time step and order, scalar precision, RNG seed, rank cap, observed rank growth, `tol_basis`, Krylov/implicit residual, discarded weight, any physical invariants, and setup details. Distinguish temporal, spatial, rank and local-solve errors. Be explicit about unsatisfied tolerances and unsupported regimes. Attach commands and machine-readable data when feasible.
