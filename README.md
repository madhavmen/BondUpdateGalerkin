# BondUpdateGalerkin

Rank-adaptive basis-update-and-Galerkin (BUG) time integration of tensor trains, with randomised
controlled bond expansion and order-p integration, for quantum many-body matrix product states and
quantics tensor trains of PDE fields, on one shared engine.

## Install

Internal: access to this repository is required. Julia 1.11 or later.

```bash
git clone https://github.com/madhavmen/BondUpdateGalerkin.git
cd BondUpdateGalerkin
julia --project -e 'using Pkg; Pkg.instantiate()'
```

`Telum` is vendored in `deps/Telum` (our fork, MIT licence); `LurCGT` is fetched from GitHub at a
pinned commit. Both are resolved by `Project.toml`'s `[sources]`.

## Quick start: a spin chain

```julia
using BondUpdateGalerkin.BUG, BondUpdateGalerkin.QMB

set_symmetry!(:U1)
hamiltonian = xxz_mpo(20; J = 1.0, delta = 0.0)        # the XX chain
psi = neel_state(20)
for _ = 1:20
    info = cbe_bug_step!(psi, hamiltonian, ComplexF64(-im * 0.05); order = 2)
    @assert budget_status(info.ledger) === :met
end
magnetisation(psi)
```

## Quick start: a PDE on a quantics grid

```julia
using BondUpdateGalerkin.BUG, BondUpdateGalerkin.QTTS

set_symmetry!(:none)
grid = qtt_grid(5, 5; x = (-1.0, 1.0), y = (0.0, 0.5), periodic_x = false, periodic_y = false)
params = HestonParams(0.03, 0.01, 2.0, 0.09, 0.4, -0.7)  # r, q, kappa, theta, sigma, rho
psi = qtt_mps(qtt_separable(grid, x -> exp(0.5x), v -> exp(-v)))
for _ = 1:10
    cbe_bug_step!(psi, heston_mpo(grid, params), ComplexF64(0.02); hermitian = false)
end
```

## Choosing a method

| Task | Call |
|---|---|
| real or imaginary time on an MPS | `cbe_bug_step!(psi, mpo, tau; order)`, `tau = -im*dt` or `-dbeta` |
| normalised imaginary time with step control | `ite_step!`, `ite_evolve!` |
| Lindblad dynamics | `lindblad_mpo` + `cbe_bug_step!(...; hermitian = false)` (MPDO), or `jump_trajectory` (MCWF) |
| linear PDE on a quantics grid | the model's MPO + `cbe_bug_step!(...; hermitian = false)` |
| nonlinear PDE | `qtt_bug_step!(psi, GPFlow(model), t, h; order)` |
| comparison | `Baselines`: `tdvp2_step!`, `qtt_tdvp2_step!`, `qtt_split_tdvp_step!` |

## Diagnostics

Every `cbe_bug_step!` returns an info record whose `ledger` holds the step's verdict
(`budget_status`: `:met` or `:budget_unmet`, with `budget_diagnostic` naming the unmet test), the
discarded weight, the expanded and final ranks, and the Krylov work. `qtt_bug_step!` returns the
stage-solver statistics, the frame audits and timings.

## Documentation and examples

- `docs/`: the manual (`julia --project=docs docs/make.jl` after instantiating `docs/`); `docs/src/mps.md`
  explains the MPS container, symmetry modes, state builders and canonical form.
- `examples/`: runnable scripts, each checked against an exact or independent reference.
- `src/README.md` and the README in every folder below it: what each folder does and which file holds what; `test/README.md` maps the tests.

## Tests

```bash
julia --project -e 'using Pkg; Pkg.test()'
```

## Reproducibility

The randomised expansion draws from a seeded stream owned by the step (`rng` keyword); the same
inputs give the same result on the same machine. The frozen regression fixtures are in
`test/fixtures/b155550`, with the versions they were made with.
