# `src/QMB/Models/OpenSystem/MPDO/`

Matrix product density operators: `rho` stored as a `SymMPS` over a Liouville site space (one site
holds `|a><b|`), evolved by `cbe_bug_step!` under the Lindblad generator with `hermitian = false`.
Keeping `|I>>` among the step's kept states makes the step trace-preserving before compression.

| file | contents |
|---|---|
| `liouville_space.jl` | The symmetry-aware Liouville site space (`LiouvilleSpace`, `liouville_space`, with `:weak`, `:strong` or no charge grading), `liouville_index`, `liouville_pair`, `liouville_charge`, `liouville_op` (the stored tensor is the transpose of the applied matrix), `ket_super`, `bra_super`, `mpdo_apply!`, and the states `mpdo_product_state`, `mpdo_identity`, `mpdo_identity_sector`. |
| `lindblad_mpo.jl` | `dissipator_super` and `lindblad_mpo`: the Lindblad generator as an MPO in Liouville space. |
| `mpdo_observables.jl` | `mpdo_trace`, `mpdo_expect`, `mpdo_rdm`, `mpdo_dense`, `mpdo_normalise_trace!`, `mpdo_checks` (trace, Hermiticity, positivity), contracted by walking the abelian blocks. |
