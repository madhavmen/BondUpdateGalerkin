"""
    BUG

The one basis-update-and-Galerkin engine, independent of what the tensor train represents.

Include directories, layered in this order (each may use only the ones before it):

- `Core/`: state/operator/problem interfaces, MPS/MPO storage, environments, local solvers,
  deterministic rank adaptation, sweeps and diagnostics.
- `Randomised/`: sketch-based controlled bond expansion, plugged into the sweeps as a
  basis-selection strategy.
- `TimeEvolution/`: real-time steps for any state implementing the `Core` interfaces.
- `ITE/`: imaginary-time steps under the same state contract.

These are directories, not submodules: a submodule named `Core` would shadow Julia's `Core`.
"""
module BUG

end
