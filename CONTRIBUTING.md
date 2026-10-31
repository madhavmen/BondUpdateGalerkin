# Contributing

The rules for people and coding agents are in [AGENTS.md](AGENTS.md) and [agents/](agents/); the
architecture and dependency directions are in [agents/architecture.md](agents/architecture.md).

## Workflow

1. Branch from `main`, one branch per task.
2. Make the smallest coherent change. Say in the pull request whether it is structural or changes
   numerical behaviour; for the latter, state the claim and its independent reference (dense,
   analytic, free-fermion, manufactured or a frozen fixture).
3. Add or update a test for every behaviour change, beside the code it tests (`test/` mirrors `src/`).
4. Commit through the pre-commit hooks (below): every commit is formatted and passes the test
   suite. Before a pull request, also run the full Aqua checks, the examples and the docs build.
5. Open a pull request with the template filled in, listing the checks you ran.
6. Record numerical-behaviour and API changes in [CHANGELOG.md](CHANGELOG.md).

## Checks (pre-commit, no CI)

There is no CI; the checks run locally through [pre-commit](https://pre-commit.com)
(`.pre-commit-config.yaml`). One-time setup:

```bash
pip install pre-commit && pre-commit install
julia -e 'using Pkg; Pkg.add(name = "JuliaFormatter", version = "1")'
julia --project -e 'using Pkg; Pkg.instantiate()'
```

| Hook | When | Runs |
|---|---|---|
| `julia-format` | every commit with staged `.jl` files | JuliaFormatter (`.JuliaFormatter.toml`, margin 100) on the staged files; if it reformats, review and `git add` again |
| `julia-tests` | every commit with staged `.jl` or `.toml` files | `Pkg.test()`: unit, physics, PDE and frozen-fixture tests, module boundaries, Aqua without its process-based checks (about five minutes) |
| `julia-aqua-full` | `pre-commit run --hook-stage manual julia-aqua-full` | all of Aqua, including `ambiguities`, `stale_deps` and `persistent_tasks`, which each start a fresh Julia process |
| `julia-examples` | `pre-commit run --hook-stage manual julia-examples` | every example at small sizes (`examples/run_all.jl`) |
| `julia-docs` | `pre-commit run --hook-stage manual julia-docs` | the manual, doctests included |

`pre-commit run --all-files` checks the whole tree. `test/runtests.jl` prints each test file's wall
time (`TESTFILE-TIME`); keep the suite near five minutes.

## Commands

```bash
julia --project -e 'using Pkg; Pkg.instantiate()'
julia --project -e 'using Pkg; Pkg.test()'
julia --project examples/QMB/xx_quickstart.jl
julia --project=docs -e 'using Pkg; Pkg.instantiate()'; julia --project=docs docs/make.jl
```

## Releases

Bump `version` in `Project.toml`, move the `CHANGELOG.md` entries under the new version, merge,
then tag the merge commit `vX.Y.Z` on `main`.
