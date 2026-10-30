# Dependency risk register and sharing checklist (WP0)

Source audited: BUG-Julia `main@b155550` (`Project.toml`, `Manifest.toml`, `deps/`). Checked 30 October 2026.

Distribution is internal sharing only; no public release is planned (decided 30 October 2026). The repository stays private.

## Dependency risk register

| Dependency | Source in BUG-Julia | License | Install path for colleagues | Risk | Action |
|---|---|---|---|---|---|
| LurCGT 1.0.0 | `repo-url = https://github.com/lurlurlurrrrr/LurCGT.jl`, `repo-rev = main` | MIT (GitHub metadata) | Public GitHub repository, not registered | Tracks `main`, so builds are not reproducible | Pin a commit in `[sources]`/Manifest when it is first added |
| Telum 0.2.1 | Our fork, vendored as `deps/Telum` (path dependency), original author Kiyeon Kim | MIT (`deps/Telum/LICENSE`) | Vendored in this repository with its MIT notice | BUG-Julia's `deps/build.jl` patches `svd.jl` in place for the no-symmetry case (`docs/telum_nosym_svd.patch`) | **Decided:** keep our fork. Vendor it with the patch applied in the source (no build-time patching) and record the BUG-Julia commit it came from. Later: open a pull request upstream to Telum with the fork's fixes |
| JSON, PrecompileTools, Preferences | General registry | MIT | Yes | Low | Add only when code that needs them moves in |
| LinearAlgebra, Printf, Random, SparseArrays | Julia stdlib | MIT | Yes | None | — |

The no-symmetry SVD fix matters for this repository specifically: QTT states run as no-symmetry MPS (`qtt/qtt_telum.jl`), so the QTTS path depends on it.

## Sharing checklist (WP9 gate)

- [ ] Every dependency installs with repository access only; Telum fork vendored with its MIT `LICENSE`.
- [ ] `[compat]` for every direct dependency; supported Julia versions tested in CI.
- [ ] Clean clone in a new depot: `Pkg.test()`, the QMB and QTTS examples, and the docs build pass using README commands only.
- [ ] No secrets, private cluster paths or generated output in the history.
- [ ] `CHANGELOG.md` and `CONTRIBUTING.md` present.
- [ ] One dry run by a colleague unfamiliar with the code (plan WP9).
