# `src/BUG/Core/` — the deterministic layer

Everything the BUG step needs that is not randomised and not a time-stepping policy. The later
layers (`Randomised/`, `TimeEvolution/`, `ITE/`) build on it; it never calls them.

| folder | what it holds |
|---|---|
| [`Interfaces/`](Interfaces/README.md) | The state contract and the symmetry traits the kernels dispatch on. |
| [`TensorNetworks/`](TensorNetworks/README.md) | `SymMPS`, its canonical form, charge sectors, and `MPO` storage and products. |
| [`Environments/`](Environments/README.md) | MPO environments and the one-, two- and zero-site effective Hamiltonians. |
| [`LocalSolvers/`](LocalSolvers/README.md) | Krylov exponentials (Lanczos, Arnoldi) and the rational root methods. |
| [`RankAdaptation/`](RankAdaptation/README.md) | The tolerance budget and ledger, the basis-expansion interface, the local Krylov enrichment, the order-p closure and the final truncation. |
| [`Sweeps/`](Sweeps/README.md) | The two half-sweeps, the frontier sweeps of the closure, bond frames and the parallel K/L/S kernel. |

## Files in this folder

| file | contents |
|---|---|
| `diagnostics.jl` | Step and run records: `KernelCounters`, `StepRecord`, `CBEBugSweepInfo`, `CBEBugRunInfo`, `OneSiteOp`, and `_assemble_and_compress!`, the closing assembly of the evolved state and its one compression at the end of every step. |
