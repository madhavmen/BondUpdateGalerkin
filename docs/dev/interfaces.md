# Core interface contracts (WP2)

The contracts every state and operator type implements, so BUG kernels never branch on the domain that built a state. Source: `src/BUG/Core/Interfaces/`. Conformance: `test/BUG/Core/Interfaces/test_state_conformance.jl` runs the same checks on every state type and symmetry.

## State contract (`state.jl`)

| Function | Returns | Invariant checked by the conformance suite |
|---|---|---|
| `nsites(psi)` | `Int` | equals `length(psi)` |
| `physical_dimension(psi, i)` | `Int` | summed over the charge sectors of the site leg; spin-1/2 gives 2 |
| `bond_dimension(psi, b)`, `bond_dimensions(psi)` | `Int`, `Vector{Int}` | bond `b` links sites `b` and `b + 1`; `nsites - 1` entries |
| `orthogonality_center(psi)` | `Int` | tensors left of it are left isometries, right of it right isometries |
| `canonicalize!(psi, i)` | `psi` | moves the centre to `i`; the dense state is unchanged to 1e-12; isometry defects below 1e-12 |
| `inner_product(bra, ket)` | `ComplexF64` | equals the dense-vector product; Hermitian; an exact `0` across different total charges |
| `symmetry(psi)` | `SymmetryTrait` | matches the mode the state was built in |
| `copy(psi)` | same type | shares tensors; changing the copy's gauge leaves the original's tensors (`===`) and centre unchanged |

**Mutation rule.** Operations replace site tensors (`psi[i] = ...`) and never mutate a tensor in place, which is what makes the shared-tensor `copy` safe. A kernel that mutates a tensor must first own it.

**Metric.** `inner_product` is the Euclidean product of the tensor-train coefficients. For QTT states that represent sampled functions, a quadrature weight is applied by QTTS, not by BUG (plan Section 3.2).

**Leg layout.** Site tensors are `(link_l, site, link_r)` with itags `"L,i"`, `"S,i"`, `"L,i+1"`; the two legs of a bond carry opposite arrows. Bond frames (`Core/Sweeps/frames.jl`) give the exact two-site snapshot `theta = U0 * S0 * V0` used by the K/L/S kernels.

## Capability traits (`capabilities.jl`)

| Trait | Telum symmetry | Used by |
|---|---|---|
| `NoSymmetry` | `()` | QTT states; spin chains run without symmetry |
| `U1Symmetry` | `(U1,)` | charge-conserving spin and fermion chains |
| `Z2Symmetry` | `(Z{2},)` | transverse-field Ising |
| `SU2Symmetry` | `(SU{2},)` | Heisenberg multiplets |

## State types

| Type | Symmetries | Status |
|---|---|---|
| `SymMPS` (dense and block-symmetric MPS) | `NoSymmetry`, `U1Symmetry` tested; `SU2Symmetry`, `Z2Symmetry` once their states move (dimer and Ising states) | PR 2 |
| QTT field state (no-symmetry `SymMPS` with a grid `Ordering`) | `NoSymmetry` | PR 3 |

## Still to define

- Operator contract (`operator.jl`): local action, environment update, Hermiticity and linearity traits. Written with the MPO move (PR 4), against the MPO type it has to describe.
- `EvolutionProblem` (`problem.jl`): `du/dt = F(t, u)` for real time, imaginary time and linear and nonlinear PDEs (PR 4, with the flows of `qtt_flows.jl`).
