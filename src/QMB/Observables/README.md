# `src/QMB/Observables/`

Read-out of a state. None of these change the represented vector, but they move the orthogonality
centre (`canonical!`); pass `copy(psi)` if the centre must stay where it is.

| file | contents |
|---|---|
| `local_observables.jl` | `site_expval` (any rank-2 local operator), `sz_expectation`, `total_sz`, `apply_sz!`, `sz_overlaps`. |
| `bond_observables.jl` | `magnetisation` (the `<S^z_j>` profile in one sweep), `center_bond`, `center_bond_dimension`, `bond_spectrum` (Schmidt values of one cut, expanded by multiplet degeneracy), `entanglement_spectrum`. |

The energy of a state is `mpo_energy(psi, mpo)` in `BUG/Core/Environments/`.
