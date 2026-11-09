# `docs/`

The manual, built with Documenter:

```
julia --project=docs -e 'using Pkg; Pkg.instantiate()'
julia --project=docs docs/make.jl
```

The site is internal: it is built and checked locally (pre-commit hook `julia-docs`) and not
deployed. Output goes to `docs/build/`, which is not tracked.

| path | contents |
|---|---|
| `make.jl` | The build script and the page list. |
| `Project.toml` | The docs environment. |
| `src/index.md` | Home: a first run, tolerances, limitations. |
| `src/theory.md` | The method: the BUG step, controlled bond expansion, quantics tensor trains. |
| `src/mps.md` | The MPS container: legs and tags, symmetry modes, state builders, canonical form, sectors, observables, the state interface. |
| `src/examples.md` | The self-checking scripts in `examples/`. |
| `src/developer_architecture.md` | Module dependencies, extension points, the state contract, tests. |
| `src/api/*.md` | One API page per module (`BUG`, `QMB`, `QTTS`, `Baselines`), listing every docstring. |

For where each piece of code lives, see the README in each folder of `src/`.
