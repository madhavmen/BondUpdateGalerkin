# `src/QTTS/` — quantics tensor trains for PDEs

Module `QTTS` (`QTTS.jl`): a field on a `2^n x 2^n` grid is a tensor train with one site per bit,
stored as a no-symmetry `SymMPS` (`set_symmetry!(:none)`), and evolved by the BUG engine.

| folder | contents |
|---|---|
| [`Tooling/`](Tooling/README.md) | Grids and bit orderings, dense core algebra, derivative and operator MPOs, the adapter to `BUG` types, the frame and stage-field interfaces of the stage solver, and the nonlinear QTT step `qtt_bug_step!`. |
| [`Elementwise/`](Elementwise/README.md) | Pointwise operations: sums, scaling, Hadamard products, `|psi|^2`. |
| [`Models/`](Models/README.md) | Heston option pricing and the damped Gross-Pitaevskii equation. |

## How a QTT step runs

`qtt_bug_step!` (`Tooling/nonlinear_step.jl`) builds the frames with `cbe_bug_step!(...;
frames_only = true)`. A linear flow (Heston) then takes the engine's rational root; a nonlinear flow
(Gross-Pitaevskii) solves implicit Runge-Kutta stages by Newton-GMRES in the root space
(`BUG/TimeEvolution/stage_solvers.jl`, with the QTT methods of `Tooling/frames.jl` and
`Tooling/stage_fields.jl`). One compression follows.

Physical quantities use the discrete L2 inner product with the cell weight `dx dy` outside the
cores. No `2^L` object is formed except in small-grid references.
