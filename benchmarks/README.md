# Benchmarks

Maintained drivers that call the public API, run without a cluster, and write results only to the
directory they are given. Timings exclude compilation (a warm-up run first) and report a median of
repeats; record the machine and thread count with every result and compare only like with like.

| Script | Measures | Compares |
|---|---|---|
| `QMB/ClosedSystem/xx_bug_vs_tdvp2.jl` | error against free fermions, rank and time on the XX chain | BUG order 2 and two-site TDVP, one per-cut truncation rule |
| `kernels/step_scaling.jl` | median time and allocations of one step against chi | BUG order 2 and two-site TDVP at the same chi |
| `QTTS/ordering_ranks.jl` | rank and error of a nonlinear GP run | grouped and interleaved QTT orderings |
| `QTTS/vortex_shedding.jl` | GP vortex shedding behind a paddle: observables, vortex charges, rank, error against dense RK4 | BUG p = 2/3/4, TDVP2 (lagged, consistent), split-dissipation TDVP |

`QTTS/common.jl` holds the shared CSV writer and the independent sparse finite-difference
references. Run times are not yet recorded.

At small sizes (a few minutes in all; `out` is any results directory):

```bash
out=/tmp/benchmarks
julia --project benchmarks/QMB/ClosedSystem/xx_bug_vs_tdvp2.jl "$out" 12 20
julia --project benchmarks/kernels/step_scaling.jl "$out" 16 "8,16"
OR_OUT="$out" OR_N=4 OR_T=0.4 julia --project benchmarks/QTTS/ordering_ranks.jl
VS_OUT="$out" VS_N=5 VS_BOX=16 VS_T=1 VS_SNAP=0.5 VS_HS=0.05 VS_ORDERS=2 VS_RELAX=0.5 VS_REFDT=1e-3 \n    julia --project benchmarks/QTTS/vortex_shedding.jl
```
