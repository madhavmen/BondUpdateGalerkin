---
paths:
  - "src/BUG/**/*.jl"
  - "test/BUG/**/*.jl"
---

# BUG core rules

Read `agents/architecture.md` and, for mathematical changes, `agents/scientific-validation.md`. Keep the BUG sweep, K/L/local/root Galerkin steps, RSVD/CBE rank adaptation, Krylov and diagnostics representation-agnostic. Never import QMB/QTTS domain models. Do not assume unitary/Hermitian or symmetry-preserving evolution unless capability traits explicitly provide it. Report tensor-leg order, orthogonality metric, truncation location, actual residuals, rank cap and tolerance saturation. Run an analytic XX and a dense no-symmetry regression for algorithm changes. For consequential changes invoke `/scientific-change` and `/bug-validation`.

Layering: `Core/` <- `Randomised/` <- `TimeEvolution/`, `ITE/`. `Core/` never calls the later layers, and randomized expansion plugs into the sweeps as a strategy. `TimeEvolution/` and `ITE/` must accept both a wavefunction MPS and a quantics TT through `Core/Interfaces`. These folders are include directories inside module `BUG`, not submodules; a module named `Core` would shadow Julia's `Core`.
