# `src/QMB/Tooling/`

Building blocks shared by the models.

| file | contents |
|---|---|
| `local_spaces.jl` | The spin-1/2 local space per symmetry mode (`local_space`), spin-`s` spaces (`spin_space`), the Z2 transverse-field Ising space (`ising_local_space`), and the sector labels `SECTOR_UP`, `SECTOR_DOWN`, `SECTOR_SU2_HALF`, `SECTOR_SU2_SINGLET`. |
| `initial_states.jl` | Product states built by a fusion recursion: `product_state`, `neel_state`, `domain_wall_state`, `sector_product_state` (any abelian local space), and `dimer_state` (nearest-neighbour singlets, the `:SU2` start state). |
| `pair_mpo.jl` | `pair_mpo`: an exact MPO for any two-body coupling matrix `J[i,j]`, charged operators included; `PairVertex`, `pair_vertices`, `nn_coupling_matrix`, `distance_coupling_matrix`. |
| `long_range_mpo.jl` | Long-range two-body MPOs: explicit exponentially decaying terms (`LongRangeTerm`, `long_range_mpo`, `long_range_zz_mpo`) and power laws fitted by sums of exponentials (`fit_long_range`, `LongRangeFit`, `default_decays`, `long_range_terms`, `power_law_zz_mpo`). |

The `SymMPS` container itself and the nearest-neighbour MPO builder `mpo_from_terms` are in
`BUG/Core/TensorNetworks/`.
