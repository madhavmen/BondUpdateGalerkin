# Testing and reproducibility policy

## Minimal change verification

1. Unit test for each new API, boundary case and corrected bug.
2. Small integration test through the public `solve` interface for relevant QMB/QTTS path.
3. Independent numerical reference when algorithm, operator or discretization changed.
4. Format and package quality checks, verifying no unintended new dependencies or undefined exports.
5. Run relevant example in a clean environment before release.

## Test scope

- `test/BUG/{Core,Randomised,TimeEvolution,ITE}/`: local actions, frames, environments, Krylov, K/L/root, RSVD, CBE, rank and tolerance diagnostics.
- `test/QMB/{ClosedSystem,OpenSystem}/`: Hamiltonian/MPO vs dense, charge blocks, initial states, XX analytics, observables, MPDO trace and MCWF ensemble vs Lindblad references.
- `test/QTTS/{Tooling,Elementwise,Models}/`: grouped bits, PDE operators, BCs, elementwise products vs dense, nonlinear RHS, Heston/GPE reference problems.
- `test/quality/`: import-boundary checks (plan Section 2.2).
- `test/Baselines/`: independent comparison methods; matching problem construction.
- `test/integration/`: public API, output schemas, short cross-domain evolution.

## Frequency

On each PR: fast tests, formatter, package QA and appropriate small references. Scheduled or before scientific release: convergence sweeps, long-time stability, performance and large scientific studies. Do not run Slurm or publication-scale studies automatically for routine edits.

## Reliability requirements

- Use fixed RNG seeds for randomized algorithms in tests; assess sensitivity in separate stochastic tests.
- Use explicit tolerances based on floating-point expectations and problem scales; never weaken a failing test merely to make CI green.
- Keep reference data small, labeled by generator version/commit and scientifically independent when possible.
- Do not write generated files into tracked directories during tests; use temporary directories.
- Reports distinguish passed, failed, skipped and not-run checks.
