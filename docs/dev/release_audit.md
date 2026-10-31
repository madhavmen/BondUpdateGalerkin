# Dependency risk register and sharing checklist (WP0)

Source audited: BUG-Julia `main@b155550` (`Project.toml`, `Manifest.toml`, `deps/`). Checked 30 October 2026.

Distribution is internal sharing only; no public release is planned (decided 30 October 2026). The repository stays private.

## Dependency risk register

| Dependency | Source in BUG-Julia | License | Install path for colleagues | Risk | Action |
|---|---|---|---|---|---|
| LurCGT 1.0.0 | `repo-url = https://github.com/lurlurlurrrrr/LurCGT.jl`, `repo-rev = main` | MIT (GitHub metadata) | Public GitHub repository, not registered | Tracked `main` in BUG-Julia | **Done:** pinned to commit `1375987` in `[sources]` |
| Telum 0.2.1 | Our fork, vendored as `deps/Telum` (path dependency), original author Kiyeon Kim | MIT (`deps/Telum/LICENSE`) | Vendored in this repository with its MIT notice | BUG-Julia's `deps/build.jl` patches `svd.jl` in place for the no-symmetry case (`docs/telum_nosym_svd.patch`) | **Decided:** keep our fork. Vendor it with the patch applied in the source (no build-time patching) and record the BUG-Julia commit it came from. Later: open a pull request upstream to Telum with the fork's fixes |
| JSON, PrecompileTools, Preferences | General registry | MIT | Yes | Low | Add only when code that needs them moves in |
| LinearAlgebra, Printf, Random, SparseArrays | Julia stdlib | MIT | Yes | None | — |

The no-symmetry SVD fix matters for this repository specifically: QTT states run as no-symmetry MPS (`qtt/qtt_telum.jl`), so the QTTS path depends on it.

## Sharing checklist (WP9 gate)

Status on 31 October 2026 (PR 13).

- [x] Every dependency installs with repository access only: `LurCGT` pinned to commit `1375987` in `[sources]`, `Telum` vendored in `deps/Telum` with its MIT `LICENSE`. Checked by the CI `Clean install` job (new depot, no cache, README commands only).
- [x] `[compat]` for every direct dependency (`LinearAlgebra`, `LurCGT`, `Random`, `SparseArrays`, `Telum`, `julia = 1.11`) and the test extras; CI tests Julia 1.11 and the current release.
- [x] Clean clone: install, load and the QMB quick start by the README commands (`Clean install` job); `Pkg.test()`, all four examples and the docs build in the `Tests`, `Examples` and `Docs` jobs.
- [x] No secrets, private cluster paths or generated output: the tree and the full history scanned for tokens, private keys, cluster host names and home or scratch paths on 31 October 2026 (none found); generated output is ignored (`.gitignore`) and examples write only to a given directory. One author in the history.
- [x] `CHANGELOG.md` (numerical behaviour first) and `CONTRIBUTING.md` present.
- [ ] One dry run by a colleague unfamiliar with the code (plan WP9): install from the README, run the XX and Heston examples, read their output, add a small custom problem. Needs a person; not done.

Versioning: `0.1.0` is tagged on `main` once the migration pull requests are merged; later releases follow `CONTRIBUTING.md`.

## Quality gates (WP8, PR 12)

| Gate | Where | Blocks a merge |
|---|---|---|
| JuliaFormatter on `src`, `test`, `docs`, `examples`, `benchmarks` | CI `Format` (the patch is kept as an artifact) | yes |
| Unit, physics, PDE and frozen-fixture tests on Julia 1.11 and the current release | CI `Tests` | yes |
| Module boundaries; `BUG/Core` layering; no BUG step in `Baselines` | `test/quality/import_boundaries.jl` | yes |
| Aqua: undefined exports, ambiguities, compat entries, piracy | `test/quality/aqua.jl` | yes |
| Every example checks its own answer | CI `Examples` | yes |
| Manual builds, doctests pass | CI `Docs` | yes (missing docstrings and cross-references warn) |
| Maintained benchmarks at small sizes, CSVs kept | `Benchmarks` workflow, weekly and on demand | no (timings on shared runners are noisy) |

**Gate check (31 October 2026).** Draft pull request #11 (`qa/broken-gate`, deleted after the check) put two deliberate defects on top of PR 12: the central first-derivative stencil with its sign flipped, and one mis-indented line, both in `src/QTTS/Tooling/derivatives.jl`. CI run 41158309943 rejected it: `Format` failed with exactly the indentation line in its patch, and `Tests` failed on both Julia versions in `test_derivatives.jl` (1D stencils, 2D derivatives, one-sided convergence) and `test_heston.jl` (generator vs dense). A first attempt used spacing inside a tuple, `( :periodic,:dirichlet0,   :onesided )`, which JuliaFormatter's default style leaves as it is; such a line is not a format violation and cannot test the gate.

JET is not configured: its supported Julia versions trail the ones this package tests, and plan Section 7 asks for it only where supported.

