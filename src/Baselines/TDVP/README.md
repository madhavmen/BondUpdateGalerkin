# `src/Baselines/TDVP/`

| file | contents |
|---|---|
| `tdvp2.jl` | `tdvp2_step!`: one symmetric two-site TDVP step (second order) on a `SymMPS` under an `MPO`, Lanczos or Arnoldi (`hermitian`), with the same truncation rule as the BUG step's compression; `TDVP2Info` records the bond, peak size, discarded weight and operator applications. |
| `qtt_tdvp.jl` | Two-site TDVP on dense QTT cores for nonlinear flows: `qtt_tdvp2_step!` (the density is held at `t_n` through the step, with backward one-site substeps), `qtt_split_tdvp_step!` (dissipation split off for the damped Gross-Pitaevskii equation), and the dense Krylov exponential `tt_expv`. |
