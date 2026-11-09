# `src/Baselines/` — comparison methods

Module `Baselines` (`Baselines.jl`): integrators the BUG results are compared against. They reuse
`BUG/Core` (environments, Krylov solvers, truncation) but never a BUG step, so a comparison never
runs the method it is compared with (`test/quality/import_boundaries.jl` checks this).

| folder | contents |
|---|---|
| [`TDVP/`](TDVP/README.md) | Two-site TDVP on symmetric MPS and on dense QTT cores. |
