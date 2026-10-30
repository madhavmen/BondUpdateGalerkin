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

end
