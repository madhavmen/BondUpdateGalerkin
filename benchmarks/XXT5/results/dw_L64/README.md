# Main result: XX chain, U(1), domain wall, d = 2, L = 64

BUG (order 2, closure checks, per-bond work cap `min(ceil(1.2 d D_b), D_max)`, D_max = 256) against
TDVP2 with the same per-cut truncation allowance. T = 5, dt = 0.1, seed 0. Errors are the worst
sampled `max_i |<S^z_i> - exact|` against the free-fermion solution (sampled every 5 steps).
Driver `benchmarks/XXT5/xx_large_rsvd_audit.jl` (AUDIT_SET=krylov_factorial), analysis
`benchmarks/XXT5/analyze_rsvd_krylov_factorial.py`. Runs on the LMU cluster, Intel Xeon
E5-2680 v4, source snapshot cdcd0da.

## Serial, 1 CPU (`julia -t 1`, jobs 17881179_5 / _6)

18 BUG arms = 6 RSVD options (fixed8rank4, fixed12rank8, rank_adaptive, rank_fast3, rank_tail12,
rank_tail8) x 3 adaptive local Krylov solvers (Lanczos, Arnoldi, Lanczos with
`krylov_rank_tol = tol_basis`).

| tol_basis | | worst S^z error | wall over T | centre chi at T | exact chi at T | worst bond chi / exact |
|---|---|---|---|---|---|---|
| 1e-2 | 18 BUG arms | 3.9e-5 - 1.2e-4 | 31 - 35 s | 19 - 21 | 27 | 1.33 - 1.53 |
| 1e-2 | TDVP2 | 2.1e-4 | 87 s | 20 | 27 | 1.0 |
| 1e-3 | 18 BUG arms | 8.4e-6 - 1.1e-5 | 29 - 34 s | 25 - 28 | 50 | 1.00 - 1.03 |
| 1e-3 | TDVP2 | 2.1e-4 | 87 s | 27 | 50 | 1.0 |

- BUG is 1.7 - 5x (tol 1e-2) and 19 - 25x (tol 1e-3) more accurate than TDVP2 and 2.5 - 3x
  faster, at the same bond dimension.
  TDVP2's error is the same at both cuts: at dt = 0.1 it is its time-step error, not truncation.
- No bond-dimension bloat: the frames stay at 1.1 - 1.4x chi, the cap binds only at step 1, zero
  cap violations. "exact chi" is the free-fermion chi under the BUG's per-cut allowance.
- The RSVD and Krylov options barely matter here (errors within 1.3x at tol 1e-3, walls within
  18%). Lanczos spends 10 - 13% of the step in the Krylov stage, Arnoldi 13 - 17%; the closure is
  41 - 46%. The fixed-width sketches give the lowest error at tol 1e-2; rank_adaptive and
  rank_fast3 report 4 - 16 budget-unmet steps, the fixed and tail modes none.

Files: `serial/factorial_report.md` (every arm), `serial/factorial_summary.csv`,
`serial/error_vs_wall_factorial.png`, `serial/chi_vs_time.png`.

## Parallel modes, 3 CPUs (`julia -t 3`, jobs 17882318_1 / _2)

Serial, `parallel = true` (K and L half-sweeps concurrently) and `parallel = :full` (the parallel
K/L/S integrator) run back to back in one allocation, so the ratios are on one node. BLAS is
single-threaded (OMP_NUM_THREADS=1). Per-arm numbers: `parallel_modes.csv`; full reports:
`threads3_{false,true,full}/`.

| tol_basis | speedup `true` | speedup `:full` | error serial / true / full | chi serial / true / full |
|---|---|---|---|---|
| 1e-2 | 0.89 - 1.07 (median 0.95) | 0.84 - 1.02 (median 0.90) | 3.9e-5-1.2e-4 / 3.4e-5-1.4e-4 / 1.4e-5-5.5e-5 | 21-24 / 21-23 / 21-22 |
| 1e-3 | 0.87 - 1.04 (median 0.91) | 0.82 - 0.98 (median 0.87) | 8.4e-6-1.1e-5 / 8.4e-6-1.3e-5 / 9.0e-6-9.6e-6 | 28-31 / 28-31 / 26-28 |

- No speedup at this size: with chi ~ 20 - 30 a half-sweep is too little work to pay for the
  threading. `true` reproduces the serial results to sampling noise.
- `:full` is a different integrator: at tol 1e-2 it is 2 - 4x more accurate than serial at the same
  chi; at tol 1e-3 all three agree.
- TDVP2 (no `parallel` option) took 166 - 199 s inside the 3-thread processes vs 88 - 107 s in the
  serial one: wall-clock noise of the shared node, so only within-mode ratios are meaningful.

## Not part of the main result

The Neel start (L = 32) was examined and set aside (USER 11-10: keep the domain-wall result). Its
findings, for the record: the per-bond cap is 3 at a product state where the exact chi is 4, which
costs ~1000x in early error (ceiling factor 4 or `:maxdim` removes it); once a bond reaches D_max
the cap leaves the frame no room (error jumps to 2 - 7e-2 at T = 5 vs 2e-4 with frames up to
d D_max); and BUG's chi grows with the distance from the root to ~2x the exact chi even when
accurate. More RSVD columns (fixed32rank16, the exact probe) do not help.
