# `src/QTTS/Elementwise/`

Pointwise operations on QTT fields.

| file | contents |
|---|---|
| `pointwise.jl` | Linear operations: `tt_scale`, `tt_conj`, `tt_add`, `tt_lincomb`. |
| `products.jl` | Elementwise products by zip-up (contract and truncate site by site, never forming the `chi_A chi_B` cores): `tt_hadamard`, `tt_product`, the variational fit `tt_hadamard_fit`, the density `tt_density` (`|psi|^2`), and the compressed operator application `op_zipup`. |
