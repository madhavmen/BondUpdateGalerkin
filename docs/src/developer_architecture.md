# Developer architecture

## Module dependencies

```
BUG  <-  QMB
BUG  <-  QTTS
BUG, QMB, QTTS  <-  Baselines
```

`BUG` never names `QMB`, `QTTS` or `Baselines`; `QMB` and `QTTS` never name each other or
`Baselines` (`test/quality/import_boundaries.jl`). Inside `BUG`, `Core` never uses a name that only
`Randomised`, `TimeEvolution` or `ITE` define: the sweeps reach the randomised
expansion through `BasisExpansion`.

## Extension points

| To add | Implement | Example |
|---|---|---|
| a basis-selection strategy | a `BasisExpansion` subtype with `expand_frame`, `sketch_closures`, `folded_sketch_closures`, `sampled_leak` | `DDQRExpansion` in `BUG/Randomised/cbe.jl` |
| a nonlinear vector field | a `QTTFlow` subtype with a `pref` field, `flow_op`, `flow_jac_ops` (and `flow_normalized` if it subtracts `mu`) | `GPFlow` in `QTTS/Models/GrossPitaevskii/gross_pitaevskii.jl` |
| a root space for the stage solver | `frame_state`, `frame_project`, `frame_operator_matrix`, `frame_conj` | `QTTFrames` in `QTTS/Tooling/frames.jl` |
| a quantum model | a function returning an `MPO` (`mpo_from_terms` with the model's local identity) and its initial states | `QMB/Models/ClosedSystem/xx.jl` |
| a PDE operator | a sum of Kronecker products of 1D chains (`qtt_op`), converted once with `qtt_to_mpo` | `heston_op` in `QTTS/Models/Heston/heston.jl` |

## State contract

Every state implements the contract of `src/BUG/Core/Interfaces/state.jl`, tabulated on the [MPS page](mps.md) (`nsites`, `physical_dimension`,
`bond_dimension`, `orthogonality_center`, `canonicalize!`, `inner_product`, `symmetry`);
`test/BUG/Core/Interfaces/test_state_conformance.jl` runs one suite against each state type.
Operations replace site tensors (`psi[i] = ...`) and never mutate one in place, so `copy(psi)`
shares tensors safely.

## Tests

`Pkg.test()` runs one file per behaviour, each in its own module (`test/runtests.jl`). The frozen
reference runs in `test/fixtures/b155550` pin the engine's answers; the other tests compare against
independent dense, free-fermion, closed-form or manufactured references.
