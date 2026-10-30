# Julia code-quality standard

## Each public API

- Provide a Julia docstring with signature, purpose, arguments and defaults, returned type/shape, mutation behavior, error cases, numerical assumptions and a compact runnable example.
- Stable names must match documented meaning. For physical quantities identify units, normalization and sign conventions.
- Prefer a small number of orthogonal options; use typed configuration objects for genuinely complex settings. No new public keyword without a documented default and test.

## Each internal numerical function

- Prefer one responsibility, descriptive variables, explicit tensor dimensions/leg order, and a short explanation of non-obvious algebra.
- Use a `!` suffix when mutating; declare when aliased inputs and cache entries become invalid.
- Avoid `Any`, unnecessary abstract fields, global mutable state, hidden RNG and allocations in hot loops; establish type stability using representative calls.
- Use explicit scalar types and generic `eltype`/`zero`/`one` where possible, so real/complex and appropriate precision work predictably.
- Use `adjoint` vs `transpose` intentionally. Explain normalized vs unnormalized state representations.
- Fail early with `DimensionMismatch` or `ArgumentError`; surface tolerance saturation rather than silently returning low-quality results.
- Separate algorithm, model construction, diagnostics, persistence and plotting. Imports should be explicit and minimal.

## Readable source files

Use `snake_case.jl` filenames; `CamelCase` modules/types; Julia lower-case functions; meaningful names. Public code can use accessible ASCII identifiers such as `tol_basis`, `chi_max`, and `time_step`; define math symbols in docstrings, which can show equations. Use established mathematical symbols in short local expressions only when they improve understanding. Do not enforce arbitrary length or comment quotas.

## Formatting, docs and review

Adopt a single committed JuliaFormatter configuration. Use Julia's recommended docstring convention and tested doctests where practical. Review the public export surface, dependency compatibility, potential type piracy, numerical allocations and edge-case errors; run a formatter/test instead of relying on stylistic prose.
