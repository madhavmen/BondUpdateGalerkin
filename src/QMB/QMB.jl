"""
    QMB

Quantum many-body problems for the BUG engine: wavefunction MPS and vectorized density operators.

Include directories:

- `Tooling/`: local spaces, reusable lattice MPO builders, gates, initial states.
- `Models/ClosedSystem/`: one file per Hamiltonian for pure-state evolution.
- `Models/OpenSystem/`: Lindblad dynamics. `MPDO/` evolves the density operator in Liouville
  space; `MCWF/` samples quantum-jump trajectories. Both share `jump_channels.jl` and step with
  the BUG engine.
- `Observables/`: local and bond observables.
"""
module QMB

using ..BUG
# BUG internals the lattice MPO builders share (the automaton blocks and the boundary rule).
using ..BUG: _site_tag, _mpo_block, _mpo_retag, _side
using LurCGT, Telum
# The Z2 Ising local space: Telum cannot derive Z2 operators, so it is built from Telum's private
# block constructor (`ising_local_space`); its no-symmetry space takes sparse operators.
using Telum: _localspace_cgt_fields, _getLocalSpace_no_symmetry
using LurCGT: Z
using SparseArrays

include("Tooling/local_spaces.jl")
include("Tooling/initial_states.jl")
include("Tooling/gates.jl")
include("Tooling/pair_mpo.jl")
include("Tooling/long_range_mpo.jl")
include("Models/ClosedSystem/xxz.jl")
include("Models/ClosedSystem/xx.jl")
include("Models/ClosedSystem/z2_ising.jl")
include("Models/ClosedSystem/tfim.jl")
include("Models/ClosedSystem/oat.jl")
include("Models/ClosedSystem/pxp.jl")
include("Models/ClosedSystem/mbl.jl")
include("Models/ClosedSystem/hubbard.jl")
include("Observables/local_observables.jl")
include("Observables/bond_observables.jl")

export local_space, SECTOR_UP, SECTOR_DOWN, SECTOR_SU2_HALF, SECTOR_SU2_SINGLET
export product_state, sector_product_state, domain_wall_state, neel_state
export site_expval, sz_expectation, total_sz, apply_sz!, sz_overlaps
export dimer_state, heisenberg_bond_gate, xx_bond_gate, magnetisation_gate
export ising_local_space, spin_space
export PairVertex, pair_vertices, pair_mpo, nn_coupling_matrix, distance_coupling_matrix
export LongRangeTerm, long_range_mpo, long_range_zz_mpo, LongRangeFit, fit_long_range
export default_decays, long_range_terms, power_law_zz_mpo
# closed-system models (Models/ClosedSystem)
export xxz_chain, heisenberg_su2_chain, xxz_mpo, heisenberg_su2_mpo
export spin_xx_mpo, spin_current_mpo, spin_domain_wall_state, spin_sz_profile
export ising_bond_gate, ising_bond_gates, ising_product_state, ising_polarised_state
export ising_kink_state, x_profile
export tfim_mpo
export oat_couplings, oat_energy_shift, oat_mpo, oat_mpo_pairs, x_polarized_state, sx_operator
export sx_profile, total_sx
export pxp_operators, pxp_mpo, z2_state, zk_state
export disorder_fields, field_mpo_from_terms, disordered_xxz_mpo, quasiperiodic_fields
export aubry_andre_xxz_mpo, hubbard_aa_mpo, hubbard_cdw_state
export hub4_space, hub4_basis_index, hub4_sector, hub4_product_state, HUB4_EMPTY, HUB4_UP
export HUB4_DOWN, HUB4_BOTH, hubbard_aa_mpo_d4, hub4_density, hub4_imbalance
export hub4_cdw_occupations, hub4_current_mpo
export magnetisation, center_bond, center_bond_dimension, bond_spectrum, entanglement_spectrum
export parity_bonds, bond_gates, energy, bond_energies

end
