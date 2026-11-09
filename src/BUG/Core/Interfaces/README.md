# `src/BUG/Core/Interfaces/`

The contracts between the engine and the states it evolves. The engine reads a state only through
these functions, so a new state type that implements them runs unchanged; the kernels choose
their path from traits, never from which domain built the state.

| file | contents |
|---|---|
| `state.jl` | The state contract: `nsites`, `physical_dimension`, `bond_dimension`, `bond_dimensions`, `orthogonality_center`, `canonicalize!`, `inner_product`, `symmetry`. Also the copy rule: `copy(psi)` shares tensors, and every operation replaces a site tensor instead of mutating it. |
| `capabilities.jl` | Symmetry traits `NoSymmetry`, `U1Symmetry`, `Z2Symmetry`, `SU2Symmetry` and `symmetry_trait`, read off a tensor's own symmetry. Charge-aware kernels (sector-graded sketches, per-sector rank allocation) dispatch on them. |

`SymMPS` implements the contract in `../TensorNetworks/symmetric_mps.jl`; the table of the
contract is on the manual page `docs/src/mps.md`, and
`test/BUG/Core/Interfaces/test_state_conformance.jl` runs one suite against every state type.
