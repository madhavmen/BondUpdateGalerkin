---
name: public-release
description: Use when preparing BondUpdateGalerkin for an external collaborator, open-source GitHub release, registered Julia package version, archived scientific reproduction, or tagged public release. Verify user onboarding, dependency/license access, docs, reproducibility, and CI evidence.
---

# Public release workflow

1. Read `agents/definition-of-done.md`, `README.md`, `CONTRIBUTING.md`, `Project.toml`, relevant manifests and CI. List all external dependencies and whether they can be installed without private access.
2. Check license and rights to redistribute the repository, Telum/LurCGT code access, borrowed figures, third-party data, notebooks and references. Flag uncertainties for human decision; do not make legal claims.
3. Test install in a clean Julia environment/new depot; run `using BondUpdateGalerkin`, package tests, simple QMB and QTT examples, and docs build according to verified README instructions.
4. Inspect exports, reproducible output files, help/docstrings, deprecation notes, changelog, citation, stable API and security-sensitive paths.
5. Verify CI matrix and extended numerical test evidence required for the version. Confirm no untracked generated plots and no private cluster/credential information.
6. Produce a release verdict: PASS / BLOCKED / INCOMPLETE with an evidence table and blocking actions. Do not claim public readiness without a clean-install check.
