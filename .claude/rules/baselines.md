---
paths:
  - "src/Baselines/**/*.jl"
  - "test/Baselines/**/*.jl"
  - "benchmarks/**/*.jl"
---

# Baseline and benchmark rules

Baselines must be mathematically independent comparisons. They may use `BUG/Core` (environments, Krylov) and `BUG/Randomised` (CBE-TDVP and CBE-DMRG use controlled bond expansion), never `BUG/TimeEvolution` or `BUG/ITE`. Benchmark scripts live in `benchmarks/QMB/` and `benchmarks/QTTS/` and call the public API. Compare identical problems, boundary conditions, initial states, observables, precision and appropriately matched error targets. Label which methods are real time, imaginary time or ground state. In timings, separate compilation/warmup and repeated steady-state measurements, record allocations and hardware. Never claim speedup from unmatched tolerances or a single noisy run. Invoke `/performance-regression` for performance claims.
