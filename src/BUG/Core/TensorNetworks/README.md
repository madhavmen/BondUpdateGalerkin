# `src/BUG/Core/TensorNetworks/`

The tensor-network objects: the symmetric MPS, its gauge, charge sectors and the MPO. All tensors
are Telum `TLArray`s. The full explanation of the MPS container is the manual page
`docs/src/mps.md`.

| file | contents |
|---|---|
| `symmetric_mps.jl` | The symmetry mode (`set_symmetry!`, `symmetry_mode`); the `SymMPS` container (legs `(link_l, site, link_r)`, itags `"L,i"`/`"S,i"`/`"L,i+1"`); `leg_dim`, `bond_dims`, `norm`, `tensor_inner`, `overlap`, `tensor_elements`; and the state-contract methods of `SymMPS`. |
| `canonical_form.jl` | Exact (truncation-free) centre moves `move_right!`, `move_left!`, `canonical!`; isometry checks `left_gram`, `right_gram`, `left_isometry_defect`, `right_isometry_defect`. |
| `sectors.jl` | Charge arithmetic (`add_charge`, `dual_charge`, `fuse_spaces`), `reachable_sectors` and `fusion_basis` with their memo (`clear_fusion_cache!`, `set_fusion_budget!`, `fusion_stats`), `perp_component` and `perp_component_right` (the part of new directions outside a frame), and the multiplet-versus-state counts `state_dim`, `state_bond_dims`, `local_dim`, `multiplet_ratio`. |
| `mpo.jl` | The `MPO` (one rank-4 tensor per site); the nearest-neighbour term list `XXZTerm`/`XXZChain`, the input of `mpo_from_terms`; on-site MPOs and MPO sums (`onsite_mpo`, `mpo_sum`, `no_jump_mpo`); the exact and compressed MPO-MPS products (`mpo_product`, `apply_mpo`, `factor_product`, `apply_factor`) and `global_krylov_states`. |

The local spaces and product states the MPS is built from are in `QMB/Tooling/`; the model MPOs
in `QMB/Models/` and `QTTS/`.
