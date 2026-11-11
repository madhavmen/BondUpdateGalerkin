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
psi = domain_wall_state(20)                            # up ... up down ... down
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

## Defaults of `cbe_bug_step!`

The defaults are the most efficient settings measured on the U(1) XX chain (domain wall, L = 64,
`tol_basis` 1e-2 to 1e-4, against the exact free-fermion solution and TDVP2;
`benchmarks/XXT5/results/dw_L64`, `dw_branch_claims`):

| setting | default | alternative |
|---|---|---|
| bond expansion | `DDQRExpansion(mode = :rank_tail, fixed_width = 8)`: one 8-column DD sketch, rank from its own tail | `mode = :adaptive` (holdout-certified), `:rank_adaptive`, `:rank_fast`, `:fixed` |
| local Krylov recurrence | `krylov_mode = :auto`: adaptive Lanczos on a Hermitian step, Arnoldi with `hermitian = false` | `:lanczos`, `:arnoldi` |
| work ceiling | `work_ceiling_rule = :bond`, `bond_ceiling_factor = 1.2`: frames up to `min(ceil(1.2 d D_b), maxdim)` | `:maxdim` (one ceiling `d*maxdim`), `:bond_strict`, `:frame_root` |

On that benchmark every BUG setting beat TDVP2 (1.3-25x lower error, 2.5-2.9x faster) without
bond-dimension bloat. `:rank_tail` has no independent certificate, so run it with `order >= 2`.
Pass `hermitian = false` for every non-Hermitian generator (it selects Arnoldi in both the frames
and the root). The per-bond cap starves the first step from a product state on a bulk quench (a
Neel state) and leaves a bond at `maxdim` no room; use `work_ceiling_rule = :maxdim` there.

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
