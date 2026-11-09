# `src/QMB/` — quantum many-body problems

Module `QMB` (`QMB.jl`): the physics side of the MPS runs. It builds local spaces, initial states
and model MPOs, and reads observables; the time evolution itself is `BUG`'s.

| folder | contents |
|---|---|
| [`Tooling/`](Tooling/README.md) | Local spaces and charge labels, initial states, general MPO builders (pair couplings, long-range fits). |
| [`Models/`](Models/README.md) | One file per Hamiltonian (`ClosedSystem/`) and the Lindblad dynamics (`OpenSystem/`). |
| [`Observables/`](Observables/README.md) | Site expectation values, magnetisation, Schmidt spectra. |

A typical run:

```julia
using BondUpdateGalerkin.BUG, BondUpdateGalerkin.QMB
set_symmetry!(:U1)
hamiltonian = xxz_mpo(20; J = 1.0, delta = 0.0)   # Models/ClosedSystem/xxz.jl
psi = neel_state(20)                              # Tooling/initial_states.jl
cbe_bug_step!(psi, hamiltonian, ComplexF64(-0.05im))
magnetisation(copy(psi))                          # Observables/bond_observables.jl
```
