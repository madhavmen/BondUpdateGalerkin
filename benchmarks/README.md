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

Of the 237 benchmark files of BUG-Julia `b155550`, two are ported here and three became test references (`test/common`); the other 232 are not copied. They stay runnable at the tag
`pre-bondupdategalerkin` of BUG-Julia with its pinned `Manifest.toml`; `docs/dev/inventory_b155550.csv`
records the decision for each.
