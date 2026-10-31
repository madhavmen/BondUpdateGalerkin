# Contributing

The rules for people and coding agents are in [AGENTS.md](AGENTS.md) and [agents/](agents/); the
architecture and dependency directions are in [agents/architecture.md](agents/architecture.md).

## Workflow

1. Branch from `main`, one branch per task.
2. Make the smallest coherent change. Say in the pull request whether it is structural or changes
   numerical behaviour; for the latter, state the claim and its independent reference (dense,
   analytic, free-fermion, manufactured or a frozen fixture).
3. Add or update a test for every behaviour change, beside the code it tests (`test/` mirrors `src/`).
4. Open a pull request with the template filled in. CI must pass: tests on Julia 1.11 and the
   current release, formatting, examples, docs, and a clean install from the README commands.
5. Record numerical-behaviour and API changes in [CHANGELOG.md](CHANGELOG.md).

## Commands

```bash
julia --project -e 'using Pkg; Pkg.instantiate()'
julia --project -e 'using Pkg; Pkg.test()'
julia --project examples/QMB/xx_quickstart.jl
julia --project=docs -e 'using Pkg; Pkg.instantiate()'; julia --project=docs docs/make.jl
```

Formatting is JuliaFormatter with `.JuliaFormatter.toml` (margin 100). When the `Format` job fails,
its `format-patch` artifact applies with `git apply`.

## Releases

Bump `version` in `Project.toml`, move the `CHANGELOG.md` entries under the new version, merge,
then tag the merge commit `vX.Y.Z` on `main`.
