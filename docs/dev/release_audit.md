# Dependency risk register and release checklist (WP0)

Source audited: BUG-Julia `main@b155550` (`Project.toml`, `Manifest.toml`, `deps/`). Checked 30 October 2026.

## Dependency risk register

| Dependency | Source in BUG-Julia | License | Public install path | Risk | Action |
|---|---|---|---|---|---|
| LurCGT 1.0.0 | `repo-url = https://github.com/lurlurlurrrrr/LurCGT.jl`, `repo-rev = main` | MIT (GitHub metadata) | Yes: public GitHub repository, not registered | Tracks `main`, so builds are not reproducible | Pin a commit in `[sources]`/Manifest when it is first added; ask the author about registration |
| Telum 0.2.1 | Vendored directory `deps/Telum` (path dependency), author Kiyeon Kim | MIT (`deps/Telum/LICENSE`) | **None found**: no public upstream repository located | **Release blocker.** Outside users cannot install it, and BUG-Julia's `deps/build.jl` patches its `svd.jl` in place for the no-symmetry case (`docs/telum_nosym_svd.patch`) | Ask the author for a public repository and an upstream fix of the no-symmetry SVD; until then decide between vendoring with the MIT notice or an optional extension (WP0 exit gate) |
| JSON, PrecompileTools, Preferences | General registry | MIT | Yes | Low | Add only when code that needs them moves in |
| LinearAlgebra, Printf, Random, SparseArrays | Julia stdlib | MIT | Yes | None | — |

The no-symmetry SVD patch matters for this repository specifically: QTT states run as no-symmetry MPS (`qtt/qtt_telum.jl`), so the QTTS path depends on it.

## Release checklist (WP9 gate)

- [ ] Telum blocker above resolved; every dependency installable without private access.
- [ ] License chosen for this repository and compatible with all dependencies.
- [ ] `[compat]` for every direct dependency; supported Julia versions tested in CI.
- [ ] Clean clone in a new depot: `Pkg.test()`, the QMB and QTTS examples, and the docs build pass using README commands only.
- [ ] No secrets, private cluster paths, unpublished assets or generated output in the history.
- [ ] `CITATION.cff`, `CHANGELOG.md`, `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`, `SECURITY.md` present.
- [ ] One external-user dry run completed (plan WP9).
