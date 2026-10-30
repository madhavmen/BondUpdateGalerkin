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
using ..BUG: _site_tag
using LurCGT, Telum

include("Tooling/local_spaces.jl")
include("Tooling/initial_states.jl")
include("Tooling/gates.jl")
include("Models/ClosedSystem/xxz.jl")
include("Observables/local_observables.jl")
include("Observables/bond_observables.jl")

export local_space, SECTOR_UP, SECTOR_DOWN, SECTOR_SU2_HALF, SECTOR_SU2_SINGLET
export product_state, sector_product_state, domain_wall_state, neel_state
export site_expval, sz_expectation, total_sz, apply_sz!, sz_overlaps
export dimer_state, heisenberg_bond_gate, xx_bond_gate, magnetisation_gate
export xxz_chain, heisenberg_su2_chain, xxz_mpo, heisenberg_su2_mpo
export magnetisation, center_bond, center_bond_dimension, bond_spectrum, entanglement_spectrum
export parity_bonds, bond_gates, energy, bond_energies

end
