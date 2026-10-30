# BondUpdateGalerkin

Rank-adaptive basis-update-and-Galerkin (BUG) time integration for quantum many-body matrix product states and quantics tensor trains (QTT), with one shared numerical engine.

**Status:** package skeleton. Code is being migrated from the BUG-Julia research repository following [docs/dev/implementation_plan.md](docs/dev/implementation_plan.md), which also contains the target layout (Section 2) and the file-by-file move map (Section 8).

## Layout

| Module | Contents |
|---|---|
| `src/BUG/` | The engine: `Core/`, `Randomised/`, `TimeEvolution/`, `ITE/` |
| `src/QMB/` | Quantum many-body: `Tooling/`, `Models/ClosedSystem/`, `Models/OpenSystem/{MPDO,MCWF}/`, `Observables/` |
| `src/QTTS/` | Quantics tensor trains: `Tooling/`, `Elementwise/`, `Models/{Heston,GrossPitaevskii}/` |
| `src/Baselines/` | TDVP, DMRG and exact references for comparison |

## Tests

```julia
julia --project=. -e 'using Pkg; Pkg.test()'
```

## Contributing

Contributor and coding-agent rules are in [AGENTS.md](AGENTS.md) and [agents/](agents/).
