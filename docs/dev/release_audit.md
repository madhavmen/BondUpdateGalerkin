# Dependency risk register and sharing checklist (WP0)

Source audited: BUG-Julia `main@b155550` (`Project.toml`, `Manifest.toml`, `deps/`). Checked 30 October 2026.

Distribution is internal sharing only; no public release is planned (decided 30 October 2026). The repository stays private.

## Dependency risk register

| Dependency | Source in BUG-Julia | License | Install path for colleagues | Risk | Action |
|---|---|---|---|---|---|
| LurCGT 1.0.0 | `repo-url = https://github.com/lurlurlurrrrr/LurCGT.jl`, `repo-rev = main` | MIT (GitHub metadata) | Public GitHub repository, not registered | Tracked `main` in BUG-Julia | **Done:** pinned to commit `1375987` in `[sources]` |
| Telum 0.2.1 | Our fork, vendored as `deps/Telum` (path dependency), original author Kiyeon Kim | MIT (`deps/Telum/LICENSE`) | Vendored in this repository with its MIT notice | BUG-Julia's `deps/build.jl` patches `svd.jl` in place for the no-symmetry case (`docs/telum_nosym_svd.patch`) | **Decided:** keep our fork. Vendor it with the patch applied in the source (no build-time patching) and record the BUG-Julia commit it came from. Later: open a pull request upstream to Telum with the fork's fixes |
| JSON, PrecompileTools, Preferences | General registry | MIT | Yes | Low | Add only when code that needs them moves in (PrecompileTools added for the precompile workload) |
| LinearAlgebra, Printf, Random, SparseArrays, Test | Julia stdlib | MIT | Yes | None | Test is a package dependency only so the precompile workload is compiled with it loaded (see `src/BondUpdateGalerkin.jl`) |

The no-symmetry SVD fix matters for this repository specifically: QTT states run as no-symmetry MPS (`qtt/qtt_telum.jl`), so the QTTS path depends on it.

## Sharing checklist (WP9 gate)

Status on 31 October 2026 (PR 13).

- [x] Every dependency installs with repository access only: `LurCGT` pinned to commit `1375987` in `[sources]`, `Telum` vendored in `deps/Telum` with its MIT `LICENSE`. Checked by the CI `Clean install` job (new depot, no cache, README commands only) up to its removal on 31 October 2026.
- [x] `[compat]` for every direct dependency (`LinearAlgebra`, `LurCGT`, `Random`, `SparseArrays`, `Telum`, `julia = 1.11`) and the test extras; CI tested Julia 1.11 and the current release until 31 October 2026.
- [x] Clean clone: install, load and the QMB quick start by the README commands (`Clean install` job); `Pkg.test()`, all four examples and the docs build in the `Tests`, `Examples` and `Docs` jobs (all green on release PR #13, run 41183438599; now pre-commit hooks). Repeated without CI on `main` 4e5d954 (31 October 2026, Julia 1.12.1): a checkout with a new, empty package depot and LurCGT cache, `Pkg.instantiate()` exactly as in the README, both README Julia blocks run from the README text, then `Pkg.test()` (5741/5741, tests 6m23).
- [x] No secrets, private cluster paths or generated output: the tree and the full history scanned for tokens, private keys, cluster host names and home or scratch paths on 31 October 2026 (none found); generated output is ignored (`.gitignore`) and examples write only to a given directory. One author in the history.
- [x] `CHANGELOG.md` (numerical behaviour first) and `CONTRIBUTING.md` present.
- [x] One dry run by a colleague unfamiliar with the code (plan WP9): install from the README, run the XX and Heston examples, read their output, add a small custom problem. Done by hand by the maintainer on 1 November 2026, together with a local run of the test suite.

Versioning: `0.1.0` is tagged on `main` once the migration pull requests are merged; later releases follow `CONTRIBUTING.md`.

## Quality gates (WP8; local pre-commit hooks since 31 October 2026)

The GitHub Actions workflows were removed on 31 October 2026 (USER decision, after Actions stopped running for billing reasons): the gates run locally through the pre-commit hooks of `.pre-commit-config.yaml` (setup in `CONTRIBUTING.md`).

| Gate | Where | When |
|---|---|---|
| JuliaFormatter on `src`, `test`, `docs`, `examples`, `benchmarks` | hook `julia-format` | every commit |
| Unit, physics, PDE and frozen-fixture tests | hook `julia-tests` (`Pkg.test()`, about five minutes) | every commit |
| Module boundaries; `BUG/Core` layering; no BUG step in `Baselines` | `test/quality/import_boundaries.jl`, in `julia-tests` | every commit |
| Aqua: undefined exports, ambiguities (in process), compat entries, piracy | `test/quality/aqua.jl`, in `julia-tests` | every commit |
| Aqua in full: also stale dependencies and persistent tasks (fresh processes) | `test/quality/aqua.jl` via `julia-aqua-full` | manual, before a pull request |
| Every example checks its own answer | hook `julia-examples` (`examples/run_all.jl`) | on demand, before a pull request |
| Manual builds, doctests pass | hook `julia-docs` | on demand, before a pull request (missing docstrings and cross-references warn) |
| Maintained benchmarks at small sizes | commands in `benchmarks/README.md` | on demand (last run: Actions run 41183449175, 31 October 2026, all four CSVs) |

No second Julia version is tested any more: the hooks use the developer's Julia. The `[compat]` floor `julia = 1.11` was last checked by CI on 31 October 2026 (release PR #13, run 41183438599).

**Gate check (31 October 2026).** Draft pull request #11 (`qa/broken-gate`, deleted after the check) put two deliberate defects on top of PR 12: the central first-derivative stencil with its sign flipped, and one mis-indented line, both in `src/QTTS/Tooling/derivatives.jl`. CI run 41158309943 rejected it: `Format` failed with exactly the indentation line in its patch, and `Tests` failed on both Julia versions in `test_derivatives.jl` (1D stencils, 2D derivatives, one-sided convergence) and `test_heston.jl` (generator vs dense). A first attempt used spacing inside a tuple, `( :periodic,:dirichlet0,   :onesided )`, which JuliaFormatter's default style leaves as it is; such a line is not a format violation and cannot test the gate.

JET is not configured: its supported Julia versions trail the ones this package tests, and plan Section 7 asks for it only where supported.

