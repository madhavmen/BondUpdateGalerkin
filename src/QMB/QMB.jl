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
using LurCGT, Telum

include("Tooling/local_spaces.jl")
include("Tooling/initial_states.jl")
include("Observables/local_observables.jl")

export local_space, SECTOR_UP, SECTOR_DOWN, SECTOR_SU2_HALF, SECTOR_SU2_SINGLET
export product_state, sector_product_state, domain_wall_state, neel_state
export site_expval, sz_expectation, total_sz, apply_sz!, sz_overlaps

end
