---
name: performance-regression
description: Use when BondUpdateGalerkin changes claim improved speed, fewer Krylov iterations, better RSVD CBE efficiency, reduced allocations, different bond scaling, faster QTT time stepping, or when reviewing a performance regression.
---

# Performance evidence workflow

1. Define the numerical output being matched (not only the kernel). Fix model/grid, state, precision, dt, truncation and error targets, RNG seed, hardware and thread count.
2. Choose a representative medium-size case plus one smaller correctness reference. Warm Julia compilation before timing steady-state work.
3. Collect elapsed time (repeated with robust summary), allocations, peak memory if measurable, max rank, observed error, Krylov iterations, RSVD sketches and accepted rank directions.
4. Compare current implementation with the agreed reference branch or baseline on identical conditions. If errors differ, make the mismatch explicit; do not report an apples-to-oranges speedup.
5. Investigate changes in computational complexity in sites, bond dimension, symmetry block count and QTT ordering before tuning micro-kernels.
6. Provide a compact before/after comparison plus numerical-accuracy table and exact command to reproduce. Report uncertainty when repetitions are noisy.

Use BenchmarkTools or equivalent reproducible warmed timing. Do not launch expensive sweeps automatically for a routine code cleanup.
