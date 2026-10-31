# BondUpdateGalerkin

Rank-adaptive basis-update-and-Galerkin (BUG) time integration of tensor trains, with one numerical
engine for quantum many-body matrix product states and for quantics tensor trains of PDE fields.

| Module | Contents |
|---|---|
| `BUG` | the engine: states, MPOs, environments, Krylov solvers, the rank-adaptive BUG step (`cbe_bug_step!`), randomised controlled bond expansion, imaginary time, implicit Runge-Kutta roots for nonlinear flows |
| `QMB` | spin and fermion models (XX/XXZ, TFIM, OAT, PXP, MBL, Hubbard), Lindblad dynamics as MPDOs or quantum-jump trajectories, observables |
| `QTTS` | quantics grids and orderings, derivative MPOs with explicit boundary conditions, Heston and Gross-Pitaevskii models, the nonlinear QTT step |
| `Baselines` | TDVP (one- and two-site, with CBE), CBE-DMRG, QTT TDVP |

## A first run

```julia
using BondUpdateGalerkin.BUG, BondUpdateGalerkin.QMB

set_symmetry!(:U1)                              # conserve S^z_tot
hamiltonian = xxz_mpo(20; J = 1.0, delta = 0.0) # the XX chain
psi = neel_state(20)
for _ = 1:20
    info = cbe_bug_step!(psi, hamiltonian, ComplexF64(-im * 0.05))   # tau = -i dt
    budget_status(info.ledger)                  # :met, or :budget_unmet with the reason
end
```

A quantics field is a tensor train over the bits of its grid indices:

```jldoctest
julia> using BondUpdateGalerkin.QTTS

julia> g = qtt_grid(3, 2; x = (0.0, 1.0), y = (0.0, 1.0));

julia> qtt_length(g)
5

julia> qtt_point(g, qtt_bits(g, 5, 2))
(5, 2)
```

The [Examples](examples.md) page lists the scripts in `examples/`, each checked against an exact
or independent reference.

## Tolerances

`cbe_bug_step!` takes three accuracy tolerances and one resource limit:

- `tol_basis`: what the enriched basis may miss of the step (local Krylov share: `tol_krylov`);
- `tol_trunc`: the relative norm the final rank truncation may discard;
- `tol_expv`: the residual of the root solve;
- `maxdim`: the rank cap. When it binds past `tol_trunc`, the step reports `:budget_unmet`
  instead of claiming the tolerance.

## Limitations

- QTT states run without symmetry (`set_symmetry!(:none)`); MPS models choose `:U1`, `:Z2`, `:SU2`
  or `:none`.
- The Gross-Pitaevskii vortex-shedding study is a research benchmark, not a validated example.
- Run times of the examples and benchmarks are not yet recorded.
