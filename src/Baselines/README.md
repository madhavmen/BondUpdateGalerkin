# `src/Baselines/` — comparison methods

Module `Baselines` (`Baselines.jl`): integrators the BUG results are compared against. They reuse
`BUG/Core` (environments, Krylov solvers, truncation) but never a BUG step.

| folder | contents |
|---|---|
| [`TDVP/`](TDVP/README.md) | Two-site TDVP on symmetric MPS and on dense QTT cores. |
