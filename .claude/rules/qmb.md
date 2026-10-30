---
paths:
  - "src/QMB/**/*.jl"
  - "test/QMB/**/*.jl"
  - "examples/QMB/**/*.jl"
---

# Quantum many-body rules

Model files define Hamiltonians, symmetries, initial states and observables, not numerical sweeps or plotting scripts. State local dimension, boundary conditions, coupling normalization and evolution convention. Keep U(1)/Z2 charge labels and MPO/MPS leg order explicit. Validate new Hamiltonian terms against independent small dense matrices and representative observable results. Use norm/energy conservation only where physically applicable. Include deterministic symmetry and reference tests; for numerical model changes use `/scientific-change`.

`Models/ClosedSystem/` holds pure-state Hamiltonians. `Models/OpenSystem/MPDO/` evolves the vectorized density operator, which preserves the trace, not the norm. `Models/OpenSystem/MCWF/` samples jump trajectories with an unnormalized no-jump segment whose norm decay sets the jump times. Both call the BUG step and share `OpenSystem/jump_channels.jl`. Reusable builders go in `Tooling/`, not in a model file.
