# `src/QMB/Models/`

Every model is an `MPO` (and its start states); the engine needs nothing else.

| folder | contents |
|---|---|
| [`ClosedSystem/`](ClosedSystem/README.md) | Hamiltonians for pure-state evolution: XX/XXZ/Heisenberg, transverse-field Ising, one-axis twisting, PXP scars, MBL and Aubry-André chains, the spinful Hubbard chain. |
| [`OpenSystem/`](OpenSystem/README.md) | Lindblad dynamics: density operators in Liouville space (`MPDO/`) and quantum-jump trajectories (`MCWF/`), sharing the jump operators of `jump_channels.jl`. |
