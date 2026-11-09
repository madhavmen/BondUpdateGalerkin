# `src/QTTS/Tooling/`

The QTT machinery, independent of any one PDE. Cores are plain Julia arrays `(chi_l, 2, chi_r)`;
operator cores are `(w_l, out, in, w_r)`. They become Telum tensors only at the adapter.

| file | contents |
|---|---|
| `grid.jl` | Bit orderings and grids: `QTTOrdering` (grouped or interleaved, any number of axes), `QTTGrid`, `qtt_grid`, `qtt_length`, `qtt_bits`, `qtt_point`, `qtt_coords`, `qtt_cell`, `qtt_sites`, `qtt_site_permutation`. |
| `cores.jl` | Dense core algebra: build (`tt_from_dense`, `tt_from_samples`, `qtt_function`, `qtt_separable`, `qtt_embed`), compress (`tt_compress`), contract and evaluate (`tt_dot`, `tt_norm`, `tt_dense`, `tt_evaluate`, `qtt_value`, `qtt_values`, `tt_ranks`). |
| `mpo_algebra.jl` | Operator-core algebra: `op_identity`, `op_add`, `op_lincomb`, `op_scale`, `op_compose`, `op_compress`, `op_adjoint`, `op_transpose`, `op_diag`, `op_ket_bra`, `op_apply`, `op_dense`. |
| `derivatives.jl` | 1D shift and finite-difference operators from the binary carry automaton (`qtt_shift`, `qtt_derivative`, `qtt_diag_1d`, boundary conditions `QTT_BCS`); no `2^n` object is formed. |
| `operator_assembly.jl` | 2D operators as sums of Kronecker products of 1D chains: `qtt_op`, `qtt_d`, `qtt_coord_diag`. |
| `state_adapter.jl` | Dense cores to and from `BUG` types under `:none`: `qtt_mps`, `qtt_cores`, `qtt_to_mpo`, `qtt_mpo_cores`, `qtt_apply`, `qtt_l2norm`. |
| `sampling.jl` | `qtt_normalize!` (in the physical quadrature) and `qtt_sample` (point reads, never densified). |
| `frames.jl` | `QTTFrames`: the root space of a BUG step on dense cores, implementing the stage solver's frame interface (`frame_state`, `frame_project`, `frame_operator_matrix`, `frame_conj`, `frame_root_envs`, `root_envs_matrix`, `root_envs_kron_sum`, `frame_defect`, `frame_miss`). |
| `stage_fields.jl` | `StageFieldOps`: the stage field and Jacobian on fixed frames by root environments per summand (`stage_field_ops`, `stage_field`, `stage_jacobian`, `stage_linear_envs`). |
| `nonlinear_step.jl` | `qtt_bug_step!`, one rank-adaptive BUG step of a flow (frames from the engine, the root from the stage solver or the rational root, one compression), and `qtt_step_tolerance`. |
