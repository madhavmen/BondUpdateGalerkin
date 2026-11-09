# `src/QMB/Models/ClosedSystem/`

One file per Hamiltonian. Each returns an `MPO` and, where the model needs one, its start states
and observables.

| file | model | main functions |
|---|---|---|
| `xxz.jl` | XXZ / Heisenberg chain | `xxz_chain` (term list), `xxz_mpo`, `heisenberg_su2_chain`, `heisenberg_su2_mpo`; `cbe_bug_step!` also accepts an `XXZChain` and converts it to an MPO. |
| `xx.jl` | U(1) XX chain at spin 1/2 and spin 3, from a domain wall | `spin_xx_mpo`, `spin_current_mpo`, `spin_domain_wall_state`, `spin_sz_profile`. |
| `z2_ising.jl` | Transverse-field Ising states in the `sigma^x` basis (Z2 parity) | `ising_product_state`, `ising_polarised_state`, `ising_kink_state`, `x_profile`. |
| `tfim.jl` | Transverse-field Ising Hamiltonian | `tfim_mpo` (virtual dimension 3, under `:Z2` or `:none`). |
| `oat.jl` | One-axis twisting, `(sum S^z)^2 / 2` | `oat_mpo`, `oat_mpo_pairs`, `oat_couplings`, `oat_energy_shift`, `x_polarized_state`, `sx_operator`, `sx_profile`, `total_sx`. |
| `pxp.jl` | PXP model of a Rydberg blockade chain (scars) | `pxp_operators`, `pxp_mpo`, `z2_state`, `zk_state`. |
| `mbl.jl` | Disordered and quasiperiodic XXZ (MBL), and the spinless fermion chain they map to | `disordered_xxz_mpo`, `aubry_andre_xxz_mpo`, `field_mpo_from_terms`, `disorder_fields`, `quasiperiodic_fields`, `hubbard_aa_mpo`, `hubbard_cdw_state`. |
| `hubbard.jl` | Spinful Aubry-André-Hubbard chain on fused `d = 4` sites | `hub4_space`, `hub4_product_state`, `hubbard_aa_mpo_d4`, `hub4_density`, `hub4_imbalance`, `hub4_current_mpo`, `hub4_cdw_occupations`, `HUB4_*` labels. |
