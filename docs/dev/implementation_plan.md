# BondUpdateGalerkin: Repository Architecture and Implementation Plan

**Status:** Proposed migration plan (not yet applied to the source repository)  
**Date:** 30 October 2026  
**Distribution:** internal sharing only; no public release is planned (decided 30 October 2026). The repository stays private. Telum ships as our fork (BUG-Julia's `deps/Telum`, MIT, with the no-symmetry SVD fix applied in the source); the fixes go upstream later as a pull request to Telum. "External user" below means a colleague with repository access who does not know the code.  
**Source repository:** `BUG-Julia` (`github.com/madhavmen/BUG-Julia`), branch `main` at `b155550`. `Revival/` and `Experiments/` exist only on its older `scar-plan` branch.  
**Primary goal:** Turn the existing BUG-Julia research codebase into a reliable, readable, reproducible Julia package for **quantum many-body MPS** and **PDE/QTT** time integration, with one common rank-adaptive BUG engine.  
**Guiding principle:** Scientific correctness and a pleasant external-user experience take priority over superficial rearrangement of files.

> **Scope and evidence.** This plan is based on the supplied inventory, not a line-by-line inspection of the current repository. All proposed interfaces are design targets, not claims that the corresponding types or methods already exist. The first task is to verify the code and external dependency licenses/availability before making structural changes.

## 1. What success looks like

An external researcher should be able to clone/install the package, run an XX-chain MPS example and a 2D PDE/QTT example, find the governing equations and discretization, switch between BUG/CBE-BUG and applicable baseline integrators, obtain machine-readable diagnostics, and reproduce a small result **without editing package source, navigating research scripts, using Slurm, or understanding internal Telum/LurCGT types**.

### Non-negotiable design requirements

1. **One BUG numerical engine.** Do not copy the sweep, root solve, rank expansion, environment, or Krylov logic into QMB and QTTS. Specialization belongs in adapters, operator actions, and domain-specific observables.
2. **Correct mathematical contracts.** A many-body wavefunction and a PDE solution share tensor-train algebra, but can differ in normalization, inner product, operator linearity, conservation laws, boundary conditions, and stability requirements. Never assume all evolutions are unitary or Hermitian.
3. **Small stable public API.** Typical users need constructors, `solve`/`solve!`, solver options, observables, output, and examples. Experimental internals must not leak into the exported namespace.
4. **Clean research separation.** Package source is importable code; `examples/` is for teaching; `benchmarks/` is for performance; `studies/` is for reproducible research; temporary scripts, handoffs, and generated output do not belong on the public main branch.
5. **Every change is validated.** Compare old and new behavior on frozen reference cases before changing algorithms. Record intentional numerical differences with explanations and tests.
6. **Automated policy, not agent promises.** Instructions guide contributors and Claude; formatting, tests, numerical checks, and pull-request review enforce critical requirements.

### Explicit non-goals for this migration

- Do not redesign BUG mathematics, change truncation heuristics, or add new physics while extracting code; those are separate reviewed PRs.
- Do not expose all existing model experiments as supported public APIs.
- Do not require a cluster, notebook, Python plotting stack, or proprietary dependency for a default installation unless the dependency audit proves an unavoidable constraint and it is documented.
- Do not equate a successful compilation with a scientifically validated result.

---

## 2. Repository layout

**Preserve Julia package conventions:** keep `Project.toml` at the repository root, `src/BondUpdateGalerkin.jl` as the entry point, and the standard `test/runtests.jl`. The four domains are Julia modules **inside `src/`** (`BUG`, `QMB`, `QTTS`, `Baselines`), not four separate packages. That avoids a monorepo that is harder for new Julia users to install.

**Folders inside a module are include directories, not submodules.** Keeping one module per domain keeps dispatch and exports simple. It also avoids a real name clash: a submodule called `Core` would shadow Julia's built-in `Core` module. Source folders use `CamelCase` and files use `snake_case.jl`. Top-level non-source directories stay lowercase (`test/`, `benchmarks/`, `examples/`, `docs/`), and their domain subfolders mirror the `src/` names.

```text
BondUpdateGalerkin/
├── Project.toml                       # package metadata, [compat], minimal dependencies
├── README.md                          # entry point and two 2-minute quick starts
├── LICENSE                            # approved open-source license (WP9)
├── CITATION.cff                       # citation metadata (WP9)
├── CONTRIBUTING.md                    # development, testing, PR expectations (WP7)
├── CODE_OF_CONDUCT.md                 # (WP9)
├── SECURITY.md                        # responsible issue reporting (WP9)
├── CHANGELOG.md                       # (WP9)
├── AGENTS.md                          # portable, always-relevant contributor rules
├── CLAUDE.md                          # Claude Code entry point; imports key rules
├── .JuliaFormatter.toml
├── .gitignore
├── .github/
│   ├── workflows/
│   │   ├── ci.yml                     # installation, tests, Aqua, format
│   │   ├── docs.yml                   # Documenter build/deploy
│   │   └── extended-tests.yml         # scheduled/optional expensive science tests
│   ├── ISSUE_TEMPLATE/
│   └── pull_request_template.md
├── agents/                            # canonical rules and checklists (Section 6)
├── .claude/                           # rules/, agents/, skills/ (Section 6)
├── src/
│   ├── BondUpdateGalerkin.jl          # module wiring and selected exports
│   │
│   ├── BUG/                           # the one numerical engine
│   │   ├── BUG.jl                     # module, include order, curated exports
│   │   ├── solve.jl                   # solve/solve! public entry points and options
│   │   ├── Core/                      # representation-independent building blocks
│   │   │   ├── Interfaces/
│   │   │   │   ├── state.jl           # site/bond access, canonical form, copy semantics
│   │   │   │   ├── operator.jl        # local action, linearity, Hermiticity traits
│   │   │   │   ├── problem.jl         # EvolutionProblem: du/dt = F(t, u)
│   │   │   │   └── capabilities.jl    # symmetry, metric and solver traits
│   │   │   ├── TensorNetworks/        # MPS/MPO storage shared by QMB and QTTS
│   │   │   │   ├── symmetric_mps.jl   # SymMPS over Telum/LurCGT (U(1), Z2, no symmetry)
│   │   │   │   ├── canonical_form.jl  # move_left!/move_right!/canonical!
│   │   │   │   ├── sectors.jl         # charge fusion, reachable sectors, multiplets
│   │   │   │   └── mpo.jl             # MPO type, sums, onsite terms, MPO-MPS products
│   │   │   ├── Environments/
│   │   │   │   ├── environment_stacks.jl
│   │   │   │   ├── mpo_environments.jl
│   │   │   │   └── effective_operators.jl   # two-, one- and zero-site effective H
│   │   │   ├── LocalSolvers/
│   │   │   │   ├── krylov.jl          # Lanczos (Hermitian) and Arnoldi (general) expv
│   │   │   │   ├── root_methods.jl    # exp, trapezoid, Radau root updates
│   │   │   │   └── krylov_recycle.jl
│   │   │   ├── RankAdaptation/        # deterministic enrichment and truncation
│   │   │   │   ├── truncation.jl
│   │   │   │   ├── basis_budget.jl    # tol_basis allocation across bonds/sectors
│   │   │   │   └── krylov_enrichment.jl
│   │   │   ├── Sweeps/
│   │   │   │   ├── frames.jl          # left/right frames, projections
│   │   │   │   ├── half_sweeps.jl     # K/L half-sweeps
│   │   │   │   ├── frontier_sweeps.jl # order-p enriching sweeps
│   │   │   │   └── parallel_kls.jl
│   │   │   └── diagnostics.jl         # stable diagnostics schema (Section 3.6)
│   │   ├── Randomised/                # randomized sketching for CBE
│   │   │   ├── sketch.jl              # sector-graded sketches, seed ownership
│   │   │   ├── range_finder.jl        # RSVD range finder and residual estimators
│   │   │   ├── cbe.jl                 # controlled bond expansion as a strategy
│   │   │   └── dd_cbe.jl              # DD-CBE kernels and controller
│   │   ├── TimeEvolution/             # real time; works on any state in Core/Interfaces
│   │   │   ├── bug_step.jl            # BUG / CBE-BUG step
│   │   │   ├── midpoint.jl            # midpoint BUG
│   │   │   ├── flows.jl               # linear and nonlinear F(t, u), Jacobian action
│   │   │   ├── stage_solvers.jl       # implicit Runge-Kutta stages in reduced frames
│   │   │   └── steppers.jl            # time loop, step-size and error control
│   │   └── ITE/                       # imaginary time; same state contract
│   │       ├── ite_step.jl
│   │       ├── schedules.jl           # uniform, log-panel and geometric beta schedules
│   │       └── normalization.jl       # explicit renormalization policy
│   │
│   ├── QMB/                           # quantum many-body wavefunctions and density operators
│   │   ├── QMB.jl
│   │   ├── Tooling/                   # lattice MPO builders and states used by many models
│   │   │   ├── local_spaces.jl        # spin/fermion local Hilbert spaces and charges
│   │   │   ├── long_range_mpo.jl      # long-range and power-law terms, exponential fits
│   │   │   ├── pair_mpo.jl            # pair/triple-term MPOs from coupling matrices
│   │   │   ├── gates.jl               # two-site gates for gate-based checks
│   │   │   └── initial_states.jl      # product, sector-product and dimer states
│   │   ├── Models/
│   │   │   ├── ClosedSystem/          # Hamiltonians for pure-state evolution
│   │   │   │   ├── xx.jl
│   │   │   │   ├── xxz.jl
│   │   │   │   ├── tfim.jl
│   │   │   │   ├── z2_ising.jl
│   │   │   │   ├── oat.jl
│   │   │   │   ├── pxp.jl
│   │   │   │   ├── mbl.jl
│   │   │   │   └── hubbard.jl
│   │   │   └── OpenSystem/            # Lindblad dynamics
│   │   │       ├── jump_channels.jl   # jump operators, shared by MPDO and MCWF
│   │   │       ├── MPDO/              # density operator vectorized in Liouville space
│   │   │       │   ├── liouville_space.jl
│   │   │       │   ├── lindblad_mpo.jl
│   │   │       │   └── mpdo_observables.jl
│   │   │       └── MCWF/              # Monte Carlo wavefunction trajectories
│   │   │           └── quantum_jumps.jl
│   │   └── Observables/
│   │       ├── local_observables.jl   # site expectation values, profiles
│   │       └── bond_observables.jl    # bond energies, entanglement spectra
│   │
│   ├── QTTS/                          # quantics tensor trains for PDEs
│   │   ├── QTTS.jl
│   │   ├── Tooling/                   # grids, QTT construction and MPO construction
│   │   │   ├── grid.jl                # QTTGrid, grouped/interleaved bit maps, endianness
│   │   │   ├── cores.jl               # QTTs from functions/samples, compress, dot, evaluate
│   │   │   ├── mpo_algebra.jl         # add, compose, apply, adjoint, diagonal MPOs
│   │   │   ├── derivatives.jl         # shift and derivative MPOs, boundary conditions
│   │   │   ├── operator_assembly.jl   # multi-axis operators from derivative terms
│   │   │   ├── state_adapter.jl       # QTT <-> Core MPS/MPO (no-symmetry SymMPS)
│   │   │   └── sampling.jl            # normalization and grid sampling
│   │   ├── Elementwise/               # pointwise operations on QTT functions
│   │   │   ├── products.jl            # Hadamard product with zip-up compression
│   │   │   └── pointwise.jl           # conj, |u|^2, scaling, linear combinations
│   │   └── Models/
│   │       ├── Heston/
│   │       │   ├── heston.jl          # parameters, Feller check, Heston operator
│   │       │   └── heston_reference.jl  # Riccati/affine references, call price, moments
│   │       └── GrossPitaevskii/
│   │           ├── gross_pitaevskii.jl  # model, nonlinear Hamiltonian, vector field, JVP
│   │           ├── paddle.jl          # stirring potential and initial state
│   │           ├── ground_state.jl    # imaginary-time ground state via BUG/ITE
│   │           └── observables.jl     # energy parts, vortex count
│   │
│   └── Baselines/                     # independent comparison methods
│       ├── Baselines.jl
│       ├── TDVP/
│       │   ├── tdvp2.jl
│       │   └── tdvp1_cbe.jl
│       ├── DMRG/
│       │   └── dmrg_cbe.jl            # ground-state comparison only
│       └── exact.jl                   # small dense/sparse verification
│
├── test/
│   ├── runtests.jl                    # fast default tests
│   ├── common/                        # independently generated fixtures and references
│   ├── BUG/
│   │   ├── Core/
│   │   ├── Randomised/
│   │   ├── TimeEvolution/
│   │   └── ITE/
│   ├── QMB/
│   │   ├── ClosedSystem/
│   │   └── OpenSystem/
│   ├── QTTS/
│   │   ├── Tooling/
│   │   ├── Elementwise/
│   │   └── Models/
│   ├── Baselines/
│   ├── integration/                   # small end-to-end cross-domain tests
│   └── quality/                       # import boundaries, Aqua/JET selected checks
├── examples/
│   ├── README.md
│   ├── QMB/                           # xx_domain_wall.jl, tfim_z2.jl, oat_z2.jl
│   └── QTTS/                          # heston_2d.jl, gross_pitaevskii_2d.jl
├── benchmarks/
│   ├── Project.toml
│   ├── README.md
│   ├── QMB/
│   │   ├── ClosedSystem/              # benchmark scripts for closed-system models
│   │   └── OpenSystem/                # MPDO and MCWF benchmark scripts
│   ├── QTTS/                          # Heston and GP benchmark scripts
│   └── kernels/                       # allocation/time microbenchmarks of BUG kernels
├── studies/                           # curated publication-quality experiments
│   ├── README.md
│   ├── configs/
│   └── runners/
└── docs/
    ├── Project.toml
    ├── make.jl
    ├── src/                           # index, installation, quickstart, theory/, guides/, api, validation
    └── dev/
        └── implementation_plan.md     # this plan
```

Files are added when their content exists. The tree is the target, not a list of placeholders to create up front.

### 2.1 Folder responsibilities

| Folder | Owns | Must not contain |
|---|---|---|
| `BUG/Core` | State/operator/problem contracts, MPS/MPO storage, environments, local solvers, deterministic rank adaptation, sweeps, diagnostics | Model definitions, time-direction policy, randomized sketching |
| `BUG/Randomised` | Sketch-based basis selection (RSVD-CBE, DD-CBE), residual estimators, explicit RNG seeds | Sweep logic; it is called as a basis-selection strategy from `Core/Sweeps` |
| `BUG/TimeEvolution` | Real-time steps and time loops for **any** state implementing `Core/Interfaces`, whether a symmetric wavefunction MPS (QMB) or a quantics TT (QTTS); linear and nonlinear flows | `if domain == ...` branches; model-specific vector fields |
| `BUG/ITE` | Imaginary-time steps, beta schedules, explicit normalization, under the same state contract | Ground-state drivers for specific models |
| `QMB/Tooling` | Local spaces, reusable lattice MPO builders, gates, initial states | Hamiltonians of a specific model |
| `QMB/Models/ClosedSystem` | One file per Hamiltonian: terms, boundary conditions, symmetry, default initial state | Sweeps, plotting |
| `QMB/Models/OpenSystem/MPDO` | Liouville space, Lindblad MPO, MPDO trace/observables. The density operator is evolved by `BUG/TimeEvolution`; the trace is preserved, not the norm. | Its own integrator |
| `QMB/Models/OpenSystem/MCWF` | Trajectory sampling: jump times from norm decay, jump application, ensembles. Between jumps it calls `BUG/TimeEvolution` on the non-Hermitian effective Hamiltonian. | Its own integrator |
| `QMB/Models/OpenSystem/jump_channels.jl` | Jump operators/channels shared by MPDO and MCWF | — |
| `QTTS/Tooling` | Grids and bit orderings, QTT construction, MPO algebra and derivative MPOs, the adapter to `Core` | Model-specific operators |
| `QTTS/Elementwise` | Pointwise operations (Hadamard products, `|u|^2`, conjugation), e.g. the GP nonlinearity | Time stepping |
| `QTTS/Models` | Heston and Gross–Pitaevskii: equations, parameters, initial data, references, observables | Sweeps, Krylov, environments |
| `Baselines` | TDVP and DMRG comparisons, exact small references | Use of `BUG/TimeEvolution` or `BUG/ITE` |
| `benchmarks/QMB`, `benchmarks/QTTS` | Benchmark **scripts** per domain, calling the public API | Importable library code |

The QMB benchmark folder lives at `benchmarks/QMB/` rather than `src/QMB/` because `src/` holds importable code only (Section 1, requirement 4).

### 2.2 Dependency rules inside `src/`

```text
BUG/Core  <--  BUG/Randomised  <--  BUG/TimeEvolution, BUG/ITE  <--  BUG/solve.jl
   ^                 ^
   |                 |
QMB, QTTS (extend Core interfaces, call the public BUG API)
Baselines (BUG/Core and BUG/Randomised only)
```

- `BUG/Core` never references `Randomised`, `TimeEvolution` or `ITE`. Sweeps take a basis-selection strategy object, and `Randomised` adds methods for it (CBE is a strategy, not a fork; Section 3.4).
- `BUG` never references `QMB`, `QTTS` or `Baselines`.
- `QMB` and `QTTS` never reference each other.
- `Baselines` may use `BUG/Core` (environments, Krylov) and `BUG/Randomised` (CBE-TDVP and CBE-DMRG legitimately use CBE), never `TimeEvolution` or `ITE`.
- `test/quality/` enforces these rules by scanning the source for cross-references.

### 2.3 Deviations from the first draft of this layout, with reasons

1. **MPS/MPO storage sits in `BUG/Core/TensorNetworks`, not `QMB/states`.** QTT states are already carried as a no-symmetry `SymMPS` (`qtt/qtt_telum.jl`), so both domains need the storage, and `QTTS` must not depend on `QMB`.
2. **`Randomised` is separate from deterministic `RankAdaptation`.** Sketch-based selection carries RNG ownership and estimator checks that deterministic truncation does not.
3. **The QTT copy of the engine is not moved.** `qtt/qtt_flows.jl` contains its own BUG step, environments, Krylov exponential and TDVP (`qtt_bug_step!`, `_env_left`, `tt_expv`, `qtt_tdvp`), which is the second engine Section 1 forbids. Its generic parts (flows, reduced-frame stage solves) are ported onto `Core/Interfaces` in `TimeEvolution`. The rest is replaced by `Core` and `Baselines/TDVP` (see Section 8).
4. **Exploratory geometries go to `studies/`, not `Models/ClosedSystem`** (Kagome, breathing Kagome, square/triangular cylinders, Haldane–Shastry), following WP4.

**Naming:** `Baselines/` is plural because it contains several methods. `QTTS` is retained for the PDE/quantics layer; spell out "Quantics Tensor Trains" in user-facing text. `Randomised` uses the British spelling as requested. Type and module names are `CamelCase`, function names are idiomatic Julia.

**No mandatory `deps/` or `bin/`** unless the package truly requires a Julia build step or supported command-line tool. If a launcher is worth keeping, it should delegate to an importable function rather than contain physics logic.

---

## 3. Architecture: one engine, multiple physical domains

### 3.1 Dependency direction

```text
                             BondUpdateGalerkin public API
                                    |
                         +----------+-----------+
                         |                      |
                        QMB                    QTTS
              (Tooling, Models:        (Tooling, Elementwise,
               Closed/OpenSystem,       Models: Heston, GP)
               Observables)
                         |                      |
                         +----------+-----------+
                                    |
                                    v
                 BUG/TimeEvolution (real time)    BUG/ITE (imaginary time)
                                    |                  |
                                    +--------+---------+
                                             |
                                     BUG/Randomised (sketch-based CBE)
                                             |
                  BUG/Core: Interfaces, TensorNetworks, Environments,
                            LocalSolvers, RankAdaptation, Sweeps
                                             |
                                        Telum/LurCGT

              Baselines --> BUG/Core + BUG/Randomised; same domain problems and metrics
              Studies, benchmarks --> public API; never imported by src/
```

This diagram represents **logical contracts**, not Julia `include` order. Avoid circular module dependencies. Domain modules may extend interface functions owned by `BUG`; `BUG` must never import `QMB` or `QTTS`. Baselines may consume `BUG/Core` and `BUG/Randomised` but must not depend on `BUG/TimeEvolution` or `BUG/ITE` (Section 2.2).

### 3.2 Shared state/operator/problem contracts

Define a small, documented **interface using Julia multiple dispatch and/or traits** rather than forcing Telum or LurCGT types into a new inheritance hierarchy. Use thin wrappers where needed; avoid type piracy and prevent interface abstractions from creating hot-loop dynamic dispatch.

| Contract | Purpose | Important requirements |
|---|---|---|
| `TensorState`/adapter | Site tensors, bonds, scalar type, canonical form, copy | Dense and block-symmetric MPS; quantics TT; complex/real; gauge conventions |
| `TensorOperator`/action | Local operator action and environment updates | MPO or matrix-free callback; sum of MPOs; non-Hermitian actions allowed |
| `EvolutionProblem` | Definition of `du/dt = F(t,u)` | Real-time `-im*H*u`, imaginary-time `-H*u`, PDE linear/nonlinear RHS |
| `InnerProduct`/metric trait | Projection/orthogonality in the correct discretization | Default Euclidean only if correct; quadrature or mass-matrix metric when required |
| `SymmetryTrait` | Optional sector-aware kernels | `NoSymmetry`, U(1), Z2; no implicit dummy charge blocks |
| `Ordering/Geometry` | Site-to-grid/site-to-physical-index map | QTT grouped and interleaved bits, endianness, axis names |
| `RankPolicy` | Shared truncation and enrichment policy | `tol_basis` is the public rank-error budget; rank caps, diagnostics, deterministic seeds |
| `TimeStepper` | Appropriate local/root time updates | Explicit vs A-stable implicit; distinguish linear exponential from nonlinear ODE steps |
| `Result/Diagnostics` | Common run output | Bond growth, discarded weights, Krylov residuals, error estimators, timings, physical observables |

Minimum adapter capabilities to define and test: `nsites`, `physical_dimension`, `bond_dimension`, `canonicalize!`, `left_frame/right_frame`, `contract_environment!`, `apply_local!`, `project_local!`, `truncate!`/`enrich!`, `inner_product`, and reproducible `copy` semantics. **These are proposed interface functions**; their exact signatures must be designed against existing Telum/LurCGT APIs first.

**Critical design distinction:** An MPS many-body wavefunction is a state in a Hilbert space; a QTT is a tensor-train encoding of sampled function values or coefficients. Their core contractions can be shared, but QTT grids, boundary conditions, measure/weights, differential operators, and observables cannot be inferred from a quantum-state type. Conversely, PDE stability or normalization rules must not overwrite quantum dynamics.

### 3.3 Model an evolution law, not only a Hamiltonian

Use an abstract ODE formulation:

\[
\dot{y}(t)=F(t,y(t)),\qquad y(t_0)=y_0.
\]

- **Real-time Schrödinger:** `F(t,psi) = -im * H(t) * psi` (possibly time-dependent).
- **Imaginary-time evolution:** `F(t,psi) = -H * psi`; optional normalization must be explicit and documented.
- **Linear PDE:** `F(t,u) = L(t)u + f(t)`; may be non-selfadjoint or dissipative.
- **Nonlinear PDE:** `F(t,u) = L(t)u + N(t,u)` with stage-consistent nonlinear reevaluation where required.

The Gross–Pitaevskii equation is **nonlinear** when the cubic interaction is present. Do not declare the full GPE linear or force it into a fixed-MPO exponential integrator. Choose an explicit nonlinear local RHS, an appropriate projection/time integrator, or a clearly justified split, and test it. Heston operators typically include diffusion, drift, mixed derivatives, and non-selfadjoint terms; do not assume Hermitian Lanczos is valid. Complex and real arithmetic and adjoint/transpose handling must be explicit.

### 3.4 Representation and capability decisions

The core must select appropriate kernels via *capabilities*, not top-level `if domain == "QMB"` branches:

- **Symmetry present?** Use charge-reachable block sampling, per-sector CBE, sector-preserving QR/SVD, and global rank allocation across blocks.
- **No symmetry?** Use dense local tensor layouts and non-symmetry RSVD/CBE, respecting rank caps and a shared tolerance budget.
- **Linear self-adjoint generator?** Hermitian Krylov/Lanczos may be appropriate; confirm complex inner products and sign conventions.
- **General/non-Hermitian generator?** Use an Arnoldi or validated general-action solver; for stiff dissipative evolution, use suitable stable implicit solvers.
- **Nonlinear generator?** Recompute local contributions consistently at integration stages; make linearized/frozen-coefficient approximations opt-in and explicit.
- **Grid ordering?** Grouped/interleaved QTT bit index maps must be tested for round-trip fidelity and derivative placement.

### 3.5 Intended public API (illustrative; finalize after prototype)

```julia
using BondUpdateGalerkin

# Example A: quantum many-body MPS
problem = BondUpdateGalerkin.QMB.XXChain(; sites=16, boundary=:open)
state = BondUpdateGalerkin.QMB.domain_wall_state(problem; symmetry=:u1)
result = BondUpdateGalerkin.BUG.solve(
    problem, state;
    algorithm=BondUpdateGalerkin.BUG.CBEBUG(),
    tspan=(0.0, 4.0), dt=0.05,
    tol_basis=1e-7,
)

# Example B: a PDE encoded as a quantics tensor train
problem = BondUpdateGalerkin.QTTS.Heston2D(grid; parameters=parameters)
state = BondUpdateGalerkin.QTTS.encode_initial_state(problem; ordering=:grouped)
result = BondUpdateGalerkin.BUG.solve(
    problem, state;
    algorithm=BondUpdateGalerkin.BUG.CBEBUG(),
    tspan=(0.0, 1.0), dt=0.01,
    tol_basis=1e-7,
)
```

Examples above are **API sketches**, not guaranteed runnable code; numerical parameters must be checked per problem. The final API should have a small number of orthogonal keyword arguments, typed configuration objects for complex setups, explicit result/output paths, and documented mutating/non-mutating variants. Avoid requiring users to construct internal environment caches, CBE sampling workspaces, or Krylov recycling arrays.

### 3.6 Numerical tolerance hierarchy

Do not collapse mathematically distinct error sources into one ambiguous parameter. Expose `tol_basis` for rank/truncation budgeting; separately model **time discretization error**, **local Krylov/linear-solve residual**, and **nonlinear/implicit stage error**. Provide documented, conservative derived defaults when possible, rather than a dozen independent tuning knobs. Record **effective tolerances and whether each criterion was satisfied** in diagnostics. If a prescribed tolerance is unattainable due to rank caps, detect and report it.

For CBE-BUG, retain the existing policy where appropriate, but separate `select_directions`, `enrich_basis`, `project_root`, and `truncate_root`. Compare RSVD estimators with residual/tail-energy checks and guard against undersampling in weak/non-symmetric sectors. Preserve exact charge blocks in the symmetry path and avoid artificial sector enumeration in the QTT path. Profile memory and allocations at large bond dimension before committing to an abstract interface.

---

## 4. Detailed implementation work packages

### WP0 — Inventory, scientific baseline, and dependency audit (PR 0)

**Tasks**

- Build a machine-readable index of BUG-Julia `main@b155550`: `src/` (73), `tests/` (77) + `test/` (3), `examples/` (8), `benchmarks/` (237), `scripts/` (30) and `notebooks/` (11), plus `Revival/`/`Experiments/` from `scar-plan`. Index each file by ownership, imports, dependencies and purpose. Mark it as **core**, **domain**, **baseline**, **study**, **obsolete** or **unknown**; do not drop unknowns without review.
- Confirm Section 8's move map against call sites, with dependencies and intended API stability. Identify duplicated implementation paths between `BondUpdateBUG/`, `RSVDCBEBondUpdate/bug/` and `RSVDCBEBondUpdate/qtt/qtt_flows.jl`.
- Run and preserve the existing full test suite, or document environmental blockers. Freeze **numerical fixtures** (small-state trajectories, errors, ranks, discarded weights, representative plots/data, runtime baselines) at a known commit, recording Julia, Telum, and LurCGT versions.
- Audit Telum/LurCGT availability, licenses, unregistered dependencies, commit pins, credentials, data provenance, model implementations, figure assets, and third-party code. Every dependency must install for a colleague with repository access; Telum is vendored as our fork (see the header).
- Tag an archival pre-refactor commit. Separate scientific behavior changes from restructuring commits.

**Deliverables:** `docs/dev/migration-map.md` (temporary until completed), fixture directory, dependency risk register, release checklist.  
**Exit gate:** known install/test baseline; every existing source tree has a disposition; no unreviewed license/access blocker.

### WP1 — Clean Julia package skeleton and module boundaries (PR 1)

**Tasks**

- Establish `src/{BUG,QMB,QTTS,Baselines}` with explicit module entry points, minimal exports and clear include order. Keep `src/BondUpdateGalerkin.jl` small (no model registries, Slurm logic, I/O side effects, or giant precompile bodies).
- Migrate `tests/` to `test/` and wire `Pkg.test()` first. Use `test/Project.toml` / workspace capabilities only if compatible with the supported minimum Julia version.
- Move experiment runner and output-writing machinery out of the core package unless it is a deliberately supported, generic public API. Keep `Revival` analysis as a `studies/` or optional `analysis/` helper rather than a mandatory import.
- Introduce clear dependency directions; add a check or review test ensuring `src/BUG` does not import QMB/QTTS/Baselines/Experiments.
- Keep an optional backward-compatible alias/deprecation layer if a public API already exists; maintain a migration guide.

**Exit gate:** `using BondUpdateGalerkin` and smoke example load in a clean environment; no new numerical differences; no circular includes.

### WP2 — Shared state, operator, and evolution adapters (PRs 2–3)

**Tasks**

- Write interface contract tables first: required methods, tensor layouts, dimensional invariants, canonicalization rules, mutation/aliasing rules, scalar types, inner-product convention, environments and invalidation.
- Implement thin adapters for (a) symmetry-native MPS, (b) non-symmetric/dense MPS, and (c) QTT field states. Reuse Telum/LurCGT storage where sensible; do not rewrite their tensor algebra.
- Add `SymmetryTrait` and a QTT `Ordering` object with explicit axis lengths/bit counts, grouped/interleaved mapping, endianness, and geometry metadata.
- Define `EvolutionProblem`/operator action independent of a Hamiltonian. Include non-Hermitian and time-dependent cases and a pathway for nonlinear local RHS evaluations.
- Verify no unexpected copies, no type instability inside hot loops, and correct environment invalidation after basis changes.

**Tests:** interface conformance tests for **every state type** using the same small contraction, norm/inner product, canonicalization, projection, environment rebuild, and round-trip tests. For QTT, compare dense and TT operators on 1D/2D small grids in grouped/interleaved layouts.  
**Exit gate:** the exact same public local-action and contraction interface supports both quantum and PDE states without domain conditionals in BUG.

### WP3 — Consolidate BUG and CBE-BUG into one numerical backend (PRs 4–5)

**Tasks**

- Extract common K/L sweeps, local/root Galerkin updates, left/right frames, projections, and environment cache from the two existing BUG trees. Build a single implementation; represent CBE as a **basis-selection/enrichment strategy**, not a copied integrator.
- Consolidate Krylov (Lanczos/Arnoldi), exponential action, recycling, implicit stage solvers, SVD/QR, and diagnostics with explicit solver capability checks.
- Consolidate RSVD sampling (including no-symmetry operation), per-sector selection where symmetry exists, truncation, residual estimators, rank caps, and deterministic RNG ownership.
- Specify what is updated during half sweeps, at the root, and after truncation; capture gauge/orthogonality invariants; ensure that higher-order variants apply compatible root updates.
- Preserve existing working modes behind tests; do not silently deprecate TDVP baselines or make assumptions about imaginary-time and dissipative stability.
- Record `tol_basis`, projected residuals, discarded weight, `chi_max`, selected/rejected rank directions, Krylov iterations, time spent in contraction/sketch/QR/SVD, allocation, and stop reasons in a stable diagnostics schema.

**Numerical verification:** compare before/after evolution on XX U(1) domain wall, TFIM Z2, a small dense non-symmetric tensor state, a non-Hermitian linear operator, and a small nonlinear local RHS. Check norm/energy where applicable; check gauge and charge preservation; test bounded-rank and tolerance-saturated cases.  
**Exit gate:** same scientific answers as pre-refactor within justified tolerances; common algorithm code exists only once; benchmark results show no unreviewed severe regressions.

### WP4 — Clean and validate QMB (PR 6)

**Tasks**

- Give each maintained model one compact file defining Hamiltonian/MPO construction, boundary conditions, couplings, symmetry support, default initial states, and observable constructors. Remove solver and plotting logic from model files.
- Migrate in order: XX/XXZ small systems and domain wall -> TFIM Z2 -> OAT -> supported Hubbard/PXP examples -> advanced geometries only after validation. Keep Kagome, triangular/square cylinders, Haldane–Shastry, and exploratory scar analyses in `studies/` until their public maintenance burden is justified.
- Document quantum-state scalar convention, local Hilbert-space dimensions, `U(1)/Z2` charge labels and fusion rules, sign conventions, and boundary conditions.
- Provide named observables (`energy`, density/magnetization profiles, correlations, norm, charge-sector diagnostics) with shape/unit conventions.

**Tests:** MPO vs explicit dense/sparse Hamiltonian on small L; charge and parity invariance; analytic free-fermion XX domain-wall evolution; TFIM symmetry parity; time and bond convergence; consistency of observables and product-state initialization.  
**Exit gate:** one XX example is accurate and understandable without users touching `src/`.

### WP5 — Add QTT/PDE domain without duplicating BUG (PRs 7–8)

**Tasks**

- Implement `QuanticsState`, bit-ordering metadata and reversible encode/decode of sampled fields; define grid spacing, coordinate transforms, quadrature weights, and boundary metadata.
- Implement MPO/matrix-free derivative and mixed-derivative operators, boundary-condition application, and local-action/environment contractions. Make non-periodic conditions first-class rather than imposing MPS boundary semantics on PDEs.
- **Heston 2D first:** choose a precise forward or backward formulation, time direction, payoff/initial condition, degeneracy and boundary treatment. Validate operator terms (including cross derivative) against a dense reference, and use manufactured or analytical solutions for tractable subcases; use converged numerical references where no closed form applies.
- **Gross–Pitaevskii second:** implement the **nonlinear** cubic contribution explicitly in the evolution interface. Start with an analytic/controlled simple case (e.g., stationary or exactly tractable limit) before the more difficult 2D vortex-shedding study. Vortex shedding needs resolution-convergence/reference simulations, not an assumed closed-form exact solution.
- For shedding, use **grouped QTT ordering** as requested, also measure interleaved ordering on small setups. Track peak bond rank vs time, effective tolerance, vortex structure, density/phase, loss/conservation as appropriate, spatial error vs reference, and wall time/allocations.
- Decide and document how a non-Euclidean grid inner product affects Galerkin projections and diagnostics; test mass/energy behavior only where the chosen equation and boundary conditions actually imply it.

**Tests:** QTT index round trip; linear and mixed derivative stencil/MPO equivalence; boundary consistency; real/complex field tests; linear non-Hermitian Krylov action; GPE nonlinear RHS finite-difference check; step-size and spatial convergence; rank-adaptive vs fixed rank.  
**Exit gate:** a clean Heston demonstration and a simple validated GPE example run through the same BUG public API; vortex shedding is a separate research-grade benchmark, not a prerequisite for basic usability.

### WP6 — Establish fair independent baselines (PR 9)

**Tasks**

- Move TDVP1, TDVP2, TDVP-CBE, DMRG-CBE, dense, sparse, and exact references to clearly owned baseline/reference modules. Only expose methods whose mathematical scope is appropriate: DMRG for ground state, TDVP for dynamics, dense exact for small systems.
- Make baseline implementations consume identical model definitions, initial states, MPOs/operators, observables, and output metrics without reusing BUG-specific basis expansion internals.
- Define parity of effort: identical physical problem, grid/L, boundary conditions, initial state, observable definitions, hardware, precision, and specified error targets. Record when time-step order or rank policy differs.
- Separate performance figures from correctness plots and ensure warmup/compilation timing is reported separately in Julia benchmarks.

**Exit gate:** published baseline comparison scripts are rerunnable and honest about differences in numerical order and computational work.

### WP7 — Documentation, examples, and decluttering (PRs 10–11)

**Tasks**

- Rewrite README for the first-time user. Proposed order: one-line purpose; supported mathematical scope; install; 2 quick starts (QMB and QTTS); method selection; diagnostic outputs; links to docs/examples; reproducibility; cite; contribute; limitations.
- Build Documenter.jl pages for theory (BUG, CBE, QTT encoding), public API, developer architecture, extension guide (custom state/operator/problem), model equations, tested conservation identities, limitations, and FAQ. Prefer tested code snippets/doctests.
- Make examples short, parameterized, and side-effect controlled. Results go into an explicitly supplied `output_dir`, not into tracked source. Each example includes an expected output description and approximate resource needs **measured after implementation**.
- Remove generated notebooks from source history where reasonable, or maintain a single source of truth with a reproducible generation script. Keep only a few well-maintained tutorials, not parallel unreviewed `.jl` and `.ipynb` copies.
- Audit `benchmarks/` (162 files), `experiments/` (114), and `scripts/` (30). Promote maintained routines, archive frozen publication artifacts outside the main install path, and delete truly redundant generated files only after reproducibility is recorded. No blanket deletion.
- Add public install instructions for both registered and GitHub-based dependencies, with explicit fallback if Telum/LurCGT cannot be publicly installed.

**Exit gate:** a new collaborator can follow README to reproduce two small tests and find their method and observable documentation in under a few navigation steps.

### WP8 — Agent governance and automated quality checks (PRs 1 and 12)

Create the `agents/`, `.claude/`, and root instruction files specified in Section 6. Start in PR 1 so the migration itself follows the rules; audit/refine them in PR 12 once real tasks expose weaknesses. Establish automatic formatting and smoke/unit checks in CI; do not equate agent adherence with enforcement.

**Exit gate:** a test prompt run from repository root finds the correct rules, a QMB/QTTS edit loads the proper scoped rule/skill, and CI rejects a deliberately malformed formatting/test PR.

### WP9 — Sharing readiness, compatibility, and independent user trial (PR 13)

Internal sharing only (see the header): public-release items (open-source license choice, registration, DOI, `CODE_OF_CONDUCT.md`, `SECURITY.md`) are out of scope unless that decision changes.

- Verify installation with repository access only (vendored Telum fork included), repo history, secrets, authorship, and availability of referenced artifacts.
- Define `[compat]` for all direct dependencies and a supported Julia version matrix based on actual upstream support. Keep exact research pins in committed **study/benchmark manifests** (or archival release tags) rather than relying only on a root `Manifest.toml` for library consumer compatibility; document the deliberate policy.
- Test from a **clean clone/new Julia depot** and from a fresh environment using only README commands. Include a CPU-only path and separate optional HPC instructions.
- Build docs with no broken links or failing doctests. Establish tagged releases, changelog, semantic versioning, DOI/citation policy, and changelog of numerical behavior.
- Conduct at least one external-user dry run: ask someone unfamiliar with the code to install, run XX and Heston/QTT, interpret outputs, and add a minimal custom problem.

**Exit gate:** all release checklist items are green; no undisclosed private dependencies; a reviewer signs off on numerical evidence and usability.

---

## 5. Scientific coding standard for every Julia source file

Every file should have **one clear responsibility** and a predictable internal order. Do not enforce arbitrary line-count limits, but split files that mix unrelated concerns.

### File and API conventions

1. **Public functions/types:** Julia docstring containing signature, purpose, arguments/keywords, return values, units, tensor shapes, mutation behavior, and a minimal working example. Explain a method's assumptions and expected numerical regime.
2. **Internal numerical kernels:** a short header describing the equations/projection, dimensions of important intermediates, invariants, references to derivations, and any non-obvious complexity. Explain *why*, not what every obvious statement does.
3. **Variables:** prefer descriptive names (`left_frame`, `effective_operator`, `discarded_weight`, `bond_dimension`) to unexplained one-letter temporaries. Use conventional `L`, `D`, `d`, `χ`, `dt`, `τ` where genuinely clear, but define each symbol on first use. Prefer readable ASCII identifiers in stable user-facing API, with mathematical notation in docstrings; do not mechanically replace meaningful symbols if it reduces clarity.
4. **Functions:** small, single-purpose, predictable return types and mutation. Julia convention: suffix `!` for mutating functions. No file loading, output writing, or plotting as implicit side effects of solver calls.
5. **Typing:** use concrete/parametric fields, type-stable hot loops, function barriers where needed; avoid over-annotating every argument or introducing an abstract container that incurs dynamic dispatch in inner loops. Check via allocation tests and JET where supported.
6. **Tensor shapes:** document leg order, index direction, charge blocks, conjugation, permutations, and gauge. Assertions should detect incompatible dimensions and unsupported combinations near public boundaries.
7. **Numerical behavior:** avoid magic constants; define a tolerance's meaning (absolute/relative, norm scaling), separate discretization from solver errors, and record all relevant run parameters.
8. **Modularity:** model files do not perform sweeps; BUG core does not define XX/TFIM/Heston; baselines do not masquerade as BUG; tests do not rely on source-relative temporary directories.
9. **Error handling:** informative `ArgumentError`/`DimensionMismatch` on invalid inputs; warnings for tolerance saturation, singular geometry, and unsupported stable solver assumptions.
10. **Performance:** avoid avoidable materialization of dense states/operators, repeated environment rebuilds, unnecessary allocations in sweeps, and implicit `Any` in critical paths. Optimize only after a representative benchmark is preserved.

### Suggested source file ordering

```julia
# File purpose and mathematical assumptions (if needed)
# 1. Imports and explicitly scoped dependencies
# 2. Documented types/configurations with validated constructors
# 3. Small public functions with docstrings
# 4. Private helpers and numerical kernels
# 5. No top-level simulation, plotting, I/O or benchmark execution
```

A docstring should explain, for example, whether `left_frame` is orthonormal under `W'W = I` or under a weighted metric, which input tensor is modified, and whether `tol_basis` is a relative Frobenius-norm criterion. A function implementing RSVD should state sketch dimensions, random seed ownership, residual test, truncation decision, and its failure/reporting policy.

### Documentation quality bar

- Equations and notation must match implementation. Cite the derivation near the actual numerical algorithm, not only in the README.
- Public configuration options have defaults and explained physical/numerical implications, including stopping conditions.
- Examples are executable in CI (small mode); plots are optional outputs, not a prerequisite to using the library.
- Document unsupported methods instead of letting users discover them through obscure backend errors.
- Prefer one tested concept per guide over long duplicated narrative across README, docs, and source.

---

## 6. `agents/` + Claude Code: governance that actually activates

**Important implementation detail:** arbitrary files in `agents/` are useful as canonical guidance but **are not, by themselves, a guaranteed Claude Code auto-loaded instruction mechanism**. Place a short root `CLAUDE.md` in the repo and have it import/point to `AGENTS.md` and essential rules. Use `.claude/rules/` for path-specific guidance, `.claude/skills/<name>/SKILL.md` for invoked workflows, and `.claude/agents/*.md` for optional specialized subagents. These are different mechanisms; do not put every rule into every prompt.

### 6.1 Activation design

| Location | Loaded/used for | Content budget |
|---|---|---|
| `AGENTS.md` | Portable rules for people and compatible coding agents | ~1 page: scientific evidence, style, tests, permissions |
| `CLAUDE.md` | Claude Code permanent project entry point | ~60–120 lines: critical rules, imports, commands, scope |
| `agents/*.md` | Canonical detailed checklists and definitions | concise, non-overlapping reference |
| `.claude/rules/{bug,qmb,qtts,baselines}.md` | Scoped path-relevant constraints | short; map to `src/**` and relevant tests |
| `.claude/skills/*/SKILL.md` | Repeated, task-specific review/implementation playbooks | small metadata; full content on demand |
| `.claude/agents/*.md` | Optional specialist subagent for isolated review | read-only by default; reports findings |
| CI and PR template | Deterministic, enforceable gates | commands, artifacts, reviewer sign-off |

Use the root `CLAUDE.md` to import `@AGENTS.md` and at most one compact `@agents/definition-of-done.md` (or incorporate their most important sentences). Link other `agents/` files for on-demand reading, and let path-scoped rules cover domain-specific constraints. Avoid importing the entire folder: it increases context cost and hides important rules in noise.

### 6.2 Mandatory always-on rules (`AGENTS.md` / `CLAUDE.md`)

1. **Inspect before editing.** Read the relevant interfaces, call sites, tests, and equations. State whether a proposed change is structural or changes mathematics.
2. **Preserve correctness by default.** No silent changes to operators, sign conventions, boundary conditions, charge blocks, truncation, normalization, or integrator order.
3. **Use the existing abstraction.** New domain models go under QMB/QTTS, reusable numerical logic under BUG, comparisons under Baselines.
4. **Test changed behavior.** Add a minimal unit/regression test for every new bug fix or public API. For numerical algorithms, use independent dense/analytic/manufactured references whenever possible.
5. **Scientific reporting.** Report error definitions, grids, time steps, rank limits, tolerances, random seeds, baseline configurations, runtime environment, and whether results satisfy expectations. Never call unverified numbers "validated."
6. **Readable Julia.** Clear names, docstrings on exported APIs, documented tensor legs, typed hot paths, mutating `!` convention, no implicit outputs or hidden global state.
7. **Minimal diffs and no clutter.** Do not regenerate figures/notebooks or reformat unrelated code; do not move files without updating imports/docs/tests.
8. **Claim only checks that ran.** Clearly list verified checks, failed checks, and unavailable checks; never invent test results.
9. **No sensitive publication.** Do not expose secrets, private cluster details, unpublished licensed assets, or private Telum/LurCGT content without approval.
10. **End-of-task summary.** State files changed, design choice, numerical impact, tests run, remaining risk, and suggested follow-up.

### 6.3 Claude skills to implement (project-local)

| Skill directory | Trigger / intended invocation | Steps / required output |
|---|---|---|
| `scientific-change/` | Any algorithm/physics/discretization change; `/scientific-change` | Write equations and assumptions; independent reference; tests; expected invariants; report scientific limitations |
| `julia-refactor/` | Moving, splitting, renaming source; `/julia-refactor` | Dependency map -> minimal diff -> type stability/docs -> tests -> API/compat review |
| `bug-validation/` | Sweeps, root step, rank, RSVD, Krylov; `/bug-validation` | Gauge/charge/projection checks; analytic XX regression; tolerance and rank diagnostics; cost regression |
| `qtt-pde-validation/` | New QTT grid/operator/PDE; `/qtt-pde-validation` | Geometry/ordering/BCs; stencil vs MPO; exact/manufactured/dense reference; stability/convergence |
| `performance-regression/` | Suspected speed/allocation change; `/performance-regression` | Warmup, allocations and elapsed times, same tolerances/hardware, confidence/repetitions, before/after table |
| `public-release/` | Preparing tag/release; `/public-release` | clean clone; dependency/license audit; docs/build; examples; safety; citation; reproducibility |

Each skill needs YAML frontmatter (`name`, task-matching `description`) and compact imperative guidance. Use automatically discoverable descriptions, but for **critical changes require explicit invocation in the root instructions or PR checklist** because task matching is not a deterministic guarantee. Reference `agents/` files; do not copy huge checklists into skill bodies. For high-cost workflows, place command recipes in skill `scripts/` and reference documents in `references/` only when they add measurable value.

### 6.4 Optional specialist agents and cost policy

- `.claude/agents/scientific-reviewer.md`: read-only critique of equations, invariants, error metrics, numerical references and physical interpretation; use for algorithm/physics changes only.
- `.claude/agents/julia-reviewer.md`: read-only review of public API, type stability, type piracy, mutating semantics, file placement, imports, test completeness and docs; use for consequential refactors.
- **Cost discipline:** start with repo search and targeted file reads; use the least expensive capable model for narrow file mapping/docs, a stronger model when numerical reasoning is necessary; avoid spawning multiple agents for a 5-line fix; give agents a bounded list of files and questions; keep summaries short; run deterministic `JuliaFormatter`/tests before expensive review; reuse test outputs and avoid rerunning slow simulations unnecessarily.
- Review agents are not allowed to silently modify code, regenerate artifacts, or claim scientific validation without evidence. A single owner integrates changes to avoid contradictory edits.
- Use hooks sparingly for deterministic local checks (e.g. formatting on edited `.jl` files), but gate reliability in CI and run expensive tests only at appropriate boundaries, not on every keystroke.

### 6.5 Prompt-to-workflow routing

```text
User prompt
   |
   v
Root CLAUDE.md -> AGENTS.md (always-on essentials)
   |
   +-- Code location? -> scoped .claude/rules/*.md
   |
   +-- Pure refactor? -> julia-refactor
   +-- BUG numerical change? -> scientific-change + bug-validation
   +-- QTT/PDE change? -> scientific-change + qtt-pde-validation
   +-- Performance claim? -> performance-regression
   +-- Public release? -> public-release
   |
   v
Plan/inspect -> minimal edit -> unit tests -> reference tests
   -> format/API checks -> report evidence -> human review + CI
```

Not every skill should be invoked on every prompt: **always-on rules establish behavior; conditional skills focus effort and reduce token cost**. For changes involving both physics and BUG, invoke both appropriate checklists.

---

## 7. Automated scientific quality assurance

### Test layers and frequencies

| Layer | Core checks | When | Block release? |
|---|---|---|---|
| Fast unit | shapes, canonicalization, environments, local action, RSVD sketch, tolerance, edge cases | every PR | yes |
| Physics/unit | XX Hamiltonian vs dense; U(1)/Z2 sectors; TFIM; norm/energy expectations | every relevant PR | yes |
| PDE/unit | grouped bits, derivatives/mixed terms, BC, GPE nonlinear RHS, Heston local action | every relevant PR | yes |
| Small integration | two short end-to-end runs, QMB + QTT, output schema and restart if offered | every PR or merge | yes |
| Regression | frozen pre-refactor trajectories, ranks, diagnostics at fixed seeds | every relevant PR | yes |
| Type/API quality | JuliaFormatter, Aqua, selective JET, docs/docstrings, import boundaries | CI | yes for configured stable checks |
| Extended scientific | time/spatial convergence, rank/tolerance sweep, long-time stability, non-Hermitian | nightly / manual before releases | yes before claims/release |
| Performance | allocation/time vs baseline, memory peak, χ and site-count scaling | scheduled / performance PR | review threshold |
| Publication | reproduces figure/data with pinned study environment | release or paper milestone | yes for referenced publications |

**Quantitative acceptance criteria, to calibrate in WP0** (avoid arbitrary fixed tolerance across all problems):

- `QMB`: norm and energy errors for unitary real-time evolutions consistent with declared time/truncation tolerances; charge sectors preserved to machine precision where exact symmetry is required; analytic XX profiles agree within measured discretization/solver error.
- `QTTS`: dense-grid vs MPO operator application agrees at small resolution to an explicitly justified floating-point tolerance; ordering is round-trip exact for indexing; grid refinement yields expected rate in smooth test cases; boundary errors and mixed derivatives are explicitly checked.
- `BUG`: left/right frame orthogonality and root projection conditions; selected rank never exceeds declared cap; discarded weight is non-negative and honestly reported; tolerance saturation warns; results insensitive to gauge up to numerical error.
- `Performance`: no unexplained algorithmic scaling degradation in χ, local physical dimension, QTT bits or symmetry block count. Benchmark **before/after** relative to a frozen baseline rather than fixed arbitrary millisecond budgets.

### Recommended tooling

- `Pkg.test()` / Julia `Test` for defaults; `Documenter.jl` for documentation and doctests.
- `JuliaFormatter.jl` with a committed `.JuliaFormatter.toml` for formatting.
- `Aqua.jl` for obvious package hygiene issues such as undefined exports, method ambiguities, compatibility metadata and type piracy.
- `JET.jl` on a small set of representative hot-path calls when its selected version supports the project's Julia release; pin compatible test dependency and do not treat static analysis as a proof of performance or correctness.
- `BenchmarkTools.jl` for warmed benchmarks; GitHub Actions for CPU smoke and package quality; optional Slurm for documented large runs only.
- `CompatHelper`/`TagBot` if the repository is registered and the workflow has been verified.

### CI design

1. **Fast PR workflow:** clean instantiate, format check, package load, unit physics/QTT tests, small integration tests, Aqua, docs link/build check.
2. **Matrix:** supported Julia releases/OS after dependency support is verified; distinguish expected unsupported platforms rather than failing silently.
3. **Nightly/dispatch:** extended convergence and rank/performance tests with artifacts (JSON/CSV metrics, logs) and baseline comparisons.
4. **Release workflow:** clean-install smoke + full scientific/reference suite + docs + dependency/license audit + approved version/changelog.

Avoid storing large generated PNGs in the package repository or using tests that write files into tracked directories. Keep test reference fixtures small, versioned, documented, and independently generated.

---

## 8. Explicit old-to-new migration map

The source is BUG-Julia `main` at `b155550`. Paths on the left are relative to its `src/`, and paths on the right are relative to the new `src/`. "Split" means one legacy file feeds several new files, by mathematical ownership rather than by filename; names a Split row does not list go to its first destination. WP0 confirms every row against call sites before anything moves. A legacy file absent from this table is not copied.

### 8.1 Engine: `BondUpdateBUG/`, `RSVDCBEBondUpdate/{bug,expansion,budget,operators}/`

| Legacy file | New location | Treatment |
|---|---|---|
| `BUGJulia.jl` | `BondUpdateGalerkin.jl` | Replaced; its precompile workload is rebuilt later with PrecompileTools |
| `BondUpdateBUG/BondUpdateBUG.jl`, `RSVDCBEBondUpdate/RSVDCBEBondUpdate.jl` | `BUG/BUG.jl`, `QMB/QMB.jl` | Replaced by the new module entry points |
| `BondUpdateBUG/expv.jl` | `BUG/Core/LocalSolvers/krylov.jl` | Move |
| `BondUpdateBUG/expv_root_methods.jl` | `BUG/Core/LocalSolvers/krylov.jl` (Lanczos/Arnoldi), `root_methods.jl` (exp, trapezoid, Radau) | Split |
| `BondUpdateBUG/frame.jl` | `BUG/Core/Sweeps/frames.jl` | Move |
| `BondUpdateBUG/sectors.jl` | `BUG/Core/TensorNetworks/sectors.jl` | Move |
| `BondUpdateBUG/symmetric_mps.jl` | `BUG/Core/TensorNetworks/symmetric_mps.jl`; `product_state`, `sector_product_state`, `dimer_state` -> `QMB/Tooling/initial_states.jl` | Split |
| `BondUpdateBUG/symmetric_mps_canonical.jl` | `BUG/Core/TensorNetworks/canonical_form.jl`; `site_expval`, `apply_sz!`, `sz_overlaps` -> `QMB/Observables/local_observables.jl` | Split |
| `BondUpdateBUG/gates.jl` | `apply_gate` -> `BUG/Core/TensorNetworks/mpo.jl`; Heisenberg/magnetisation gates -> `QMB/Tooling/gates.jl` | Split |
| `BondUpdateBUG/observables.jl`, `BondUpdateBUG/sweep.jl` | `QMB/Observables/bond_observables.jl` | Move (`energy`, `parity_bonds`, bond spectra) |
| `BondUpdateBUG/z2_ising.jl` | `QMB/Models/ClosedSystem/z2_ising.jl`; `ising_local_space` -> `QMB/Tooling/local_spaces.jl` | Split |
| `budget/adaptive_budget.jl` | `BUG/Core/RankAdaptation/basis_budget.jl` | Move |
| `budget/multiplets.jl` | `BUG/Core/TensorNetworks/sectors.jl` | Merge |
| `bug/cbe_bug.jl` | `BUG/TimeEvolution/bug_step.jl` | Move; expansion calls go through the strategy interface |
| `bug/cbe_bug_midpoint.jl` | `BUG/TimeEvolution/midpoint.jl` | Move |
| `bug/cbe_bug_info.jl` | `BUG/Core/diagnostics.jl` | Move |
| `bug/half_sweeps.jl`, `bug/frontier_sweeps.jl`, `bug/parallel_kls.jl` | `BUG/Core/Sweeps/` | Move |
| `bug/krylov_closure.jl`, `bug/krylov_closure_loop.jl`, `bug/krylov_enrich.jl` | `BUG/Core/RankAdaptation/krylov_enrichment.jl` | Merge |
| `bug/krylov_recycle.jl` | `BUG/Core/LocalSolvers/krylov_recycle.jl` | Move |
| `bug/imaginary_time.jl` | `BUG/ITE/ite_step.jl`; `log_panel_steps`, `uniform_beta_steps` -> `BUG/ITE/schedules.jl` | Split |
| `bug/quantum_jumps.jl` | `QMB/Models/OpenSystem/MCWF/quantum_jumps.jl` | Move; the no-jump segment keeps calling the BUG step |
| `expansion/cbe_sketch.jl` | `BUG/Randomised/sketch.jl` | Move |
| `expansion/cbe_range.jl` | `BUG/Randomised/range_finder.jl` | Move |
| `expansion/cbe_core.jl`, `cbe_expand.jl`, `cbe_bond_update.jl`, `bond_expansion.jl` | `BUG/Randomised/cbe.jl` | Merge into one CBE strategy; `lanczos_lowest` goes to `Baselines/DMRG` if only DMRG uses it |
| `expansion/cbe_options.jl` | Solver options -> `BUG/solve.jl`; `CBEBugRunInfo` -> `BUG/Core/diagnostics.jl` | Split |
| `expansion/dd_cbe.jl`, `expansion/dd_cbe_controller.jl` | `BUG/Randomised/dd_cbe.jl` | Merge |
| `operators/henv.jl` | `XXZTerm`, `XXZChain`, `xxz_chain`, `heisenberg_su2_chain` -> `QMB/Models/ClosedSystem/xxz.jl`; `ZeroEnv`, `ChannelSet`, channel pushes -> `BUG/Core/Environments/environment_stacks.jl` | Split |
| `operators/henv_stacks.jl` | `BUG/Core/Environments/environment_stacks.jl` | Move |
| `operators/mpo.jl` | `MPO`, `mpo_from_terms`, `onsite_mpo`, `mpo_sum` -> `BUG/Core/TensorNetworks/mpo.jl`; `heisenberg_su2_mpo` -> `QMB/Models/ClosedSystem/xxz.jl`; long-range terms and fits -> `QMB/Tooling/long_range_mpo.jl` | Split |
| `operators/mpo_apply.jl` | `BUG/Core/TensorNetworks/mpo.jl` | Merge |
| `operators/mpo_env.jl` | `BUG/Core/Environments/mpo_environments.jl` | Move |
| `operators/mpo_fold.jl`, `operators/zero_site_core.jl` | `BUG/Core/Environments/effective_operators.jl`; `truncate_sweep!`, `truncate_recursive!` -> `BUG/Core/RankAdaptation/truncation.jl` | Split |

### 8.2 Quantum many-body models: `RSVDCBEBondUpdate/models/`

| Legacy file | New location | Treatment |
|---|---|---|
| `spin_xx.jl` | `QMB/Models/ClosedSystem/xx.jl`; `spin_space` -> `QMB/Tooling/local_spaces.jl` | Split |
| `tfim.jl`, `oat.jl` | `QMB/Models/ClosedSystem/tfim.jl`, `oat.jl` | Move |
| `pxp_scars.jl`, `mbl_scars.jl`, `hubbard_d4.jl` | `QMB/Models/ClosedSystem/pxp.jl`, `mbl.jl`, `hubbard.jl` | Move; `pair_loss_junction` -> `OpenSystem/jump_channels.jl` |
| `pair_mpo.jl` | `QMB/Tooling/pair_mpo.jl` | Move |
| `dissipative.jl` | `QMB/Models/OpenSystem/jump_channels.jl` | Move (channels and jump MPOs serve both MPDO and MCWF) |
| `liouville.jl` | `QMB/Models/OpenSystem/MPDO/liouville_space.jl`; `dissipator_super`, `lindblad_mpo` -> `lindblad_mpo.jl` | Split |
| `liouville_blocks.jl` | `QMB/Models/OpenSystem/MPDO/mpdo_observables.jl` | Move |
| `kagome.jl`, `breathing_kagome.jl`, `square_cylinder.jl`, `triangular_cylinder.jl`, `haldane_shastry.jl` | `studies/` | Not in the package until maintained (WP4) |

### 8.3 Quantics tensor trains: `RSVDCBEBondUpdate/qtt/`

| Legacy file | New location | Treatment |
|---|---|---|
| `qtt_grid.jl` | `QTTS/Tooling/grid.jl` | Move |
| `qtt_cores.jl` | `tt_from_*`, `tt_compress`, `tt_dot`, `tt_norm`, `tt_evaluate`, `qtt_embed`, `qtt_function`, `qtt_separable`, `qtt_values` -> `QTTS/Tooling/cores.jl`; `op_*` -> `QTTS/Tooling/mpo_algebra.jl`; `tt_hadamard`, `tt_product`, zip-up -> `QTTS/Elementwise/products.jl`; `tt_conj`, `tt_scale`, `tt_density`, `tt_add`, `tt_lincomb` -> `QTTS/Elementwise/pointwise.jl` | Split |
| `qtt_ops1d.jl` | `QTTS/Tooling/derivatives.jl` | Move |
| `qtt_telum.jl` | `QTTS/Tooling/state_adapter.jl` | Move |
| `qtt_models.jl` | `qtt_d`, `qtt_coord_diag`, `qtt_op` -> `QTTS/Tooling/operator_assembly.jl`; GP model/Hamiltonian/vector field/JVP -> `QTTS/Models/GrossPitaevskii/gross_pitaevskii.jl`; `GPPaddle`, `paddle_state`, `gp_paddle_op` -> `paddle.jl`; Heston -> `QTTS/Models/Heston/heston.jl` | Split; `gp_bug_step!` dropped (callers use `BUG.solve`) |
| `qtt_observables.jl` | `qtt_normalize!`, `qtt_sample` -> `QTTS/Tooling/sampling.jl`; `gp_parts`, `gp_observables`, `vortex_count` -> `GrossPitaevskii/observables.jl`; `heston_*` -> `Heston/heston_reference.jl` | Split |
| `qtt_imaginary.jl` | `uniform_schedule`, `geometric_schedule` -> `BUG/ITE/schedules.jl`; `gp_ground_state!`, `gp_gradient_descent!` -> `GrossPitaevskii/ground_state.jl` | Split |
| `qtt_flows.jl` | `QTTFlow`, `LinearFlow`, `flow_op`, `flow_jac_ops`, `frozen_flow` -> `BUG/TimeEvolution/flows.jl` (`GPFlow` -> `gross_pitaevskii.jl`); `QTTFrames`, `frame_*`, `reduced_field`, `reduced_jacobian`, `RK_TABLEAUX`, `rk_stage_solve`, `_gmres_real`, `_unpack` -> `BUG/TimeEvolution/stage_solvers.jl`, ported onto `Core/Interfaces`; `qtt_bug_step!`, `_replace_state!` -> merged into `BUG/TimeEvolution/bug_step.jl`; `_tc`, `_env_*`, `_eff1`, `_eff2`, `_op_expect` -> replaced by `BUG/Core/Environments`; `tt_expv`, `_dense_expv` -> replaced by `BUG/Core/LocalSolvers`; `qtt_tdvp2_step!`, `qtt_split_tdvp_step!` -> merged into `Baselines/TDVP` | Split, port and deduplicate (WP3); requires before/after fixtures from WP0 |

### 8.4 Baselines: `RSVDCBEBondUpdate/baselines/`

| Legacy file | New location | Treatment |
|---|---|---|
| `tdvp2.jl` | `Baselines/TDVP/tdvp2.jl` | Move |
| `tdvp1_cbe.jl`, `tdvp1_cbe_variants.jl`, `tdvp_cbe1s.jl` | `Baselines/TDVP/tdvp1_cbe.jl` | Merge; keep only variants used by a maintained comparison |
| `dmrg_cbe1s.jl`, `dmrg_cbe1s_driver.jl`, `ground_state.jl` | `Baselines/DMRG/dmrg_cbe.jl` | Merge |

### 8.5 Outside `src/`

| Existing area | New owner | Treatment |
|---|---|---|
| `Revival/`, `Experiments/` (`scar-plan` branch only) | `studies/` | Orchestration and analysis stay out of the package |
| `tests/BondUpdateBUG`, `tests/sweeps`, `test/` | `test/BUG/*`, `test/QMB/*`, `test/QTTS/*`, `test/Baselines` | Group by tested behavior, mirroring `src/`; file-level map produced in WP0 |
| `tests/common` | `test/common` | Retain independent analytic/dense/sparse/free-fermion/Lindblad references |
| `examples/` | `examples/QMB/`, `examples/QTTS/` | Keep as short, verified onboarding examples |
| `benchmarks/` (237 files) | `benchmarks/QMB/{ClosedSystem,OpenSystem}`, `benchmarks/QTTS`, `benchmarks/kernels`; the rest to `studies/` or the archive tag | Curate; no cluster requirement by default |
| `scripts/`, `bin/` | `studies/runners/` or the archive tag | Remove probes/wrappers from the user path |
| `deps/Telum` (our fork) | `deps/Telum` | Vendored with its MIT `LICENSE`; the no-symmetry SVD fix is already in its source at `b155550`, so `deps/build.jl` is not copied |
| `notebooks/` | `docs/` tutorials | Single authoritative source |
| `docs/` plans, handoffs, audits | Archive tag; stable content -> `docs/src/developer_architecture.md` | Not copied |
| `figures/`, `handover5_figs/`, `plots/` | Not copied | Curated figures only, later, into `docs/src/assets/` |
| `Manifest.toml` / pinned Telum/LurCGT | Root `Project.toml` `[compat]`; study/benchmark manifests | Decide after the access/license/Julia-version audit |

**Migration tactic:** file moves are separate commits from behavior changes. Because the new repository has no shared history with BUG-Julia, each moving commit names the source path and commit (`BUG-Julia@b155550:src/...`) in its message so that review can diff old against new. Update includes and doc links, and test every PR.

## 9. Implementation order, dependencies, and PR size

| Order | Main PR | Dependencies | Proof required |
|---|---|---|---|
| 0 | inventory + frozen science references | none | complete move map and existing test status |
| 1 | skeleton + agent rules + CI foundation | 0 | clean load and current examples unaffected |
| 2 | state/operator traits + dense MPS | 1 | state conformance suite |
| 3 | QTT adapter + ordering | 2 | exact index/operator checks on small grids |
| 4 | extract shared BUG sweeps/root | 2 | unchanged QMB numerical results |
| 5 | integrate CBE, RSVD, symmetry/no-symmetry | 3–4 | small rank-adaptive QMB and dense/QTT tests |
| 6 | QMB model/observables cleanup | 4–5 | analytic XX and symmetry regression |
| 7 | Heston2D QTT + derivatives/BC | 3–5 | dense/manufactured validation |
| 8 | GPE nonlinear QTT | 3–5 | nonlinear RHS + convergence test |
| 9 | baseline cleanup and parity of comparisons | 6–8 | fair same-problem comparisons |
| 10 | examples, public docs | 6–9 | readme/doctest onboarding checks |
| 11 | studies/benchmarks cleanup and archive | 0, 6–10 | reproducible retained scripts |
| 12 | CI hardening, scientific checklists, QA | 1–11 | gates detect deliberately broken PR |
| 13 | public release hardening | 0–12 | external clean-install and user trial |

**Recommended first thin vertical slice:** XX domain wall on the existing MPS path and a trivial 2D QTT linear operator on the new state interface, both entering *one* `BUG.solve` path. This proves architecture before broad model migration. Do not wait until all QMB models are ported to test the QTT abstraction.

## 10. Definition of done (per change, not only at release)

- [ ] The change has a short problem statement and belongs to the correct `src/` domain.
- [ ] The public API and mathematical assumptions are documented; exported names are intentional.
- [ ] Source naming, mutation semantics, tensor legs, tolerances, and errors are understandable to an outsider.
- [ ] A minimal deterministic test covers the new behavior; a physics/PDE reference exists where relevant.
- [ ] Previous behavior is compared to frozen baselines or the discrepancy is documented and reviewed.
- [ ] Numerical errors, invariants, and unsupported regimes are reported, not hidden.
- [ ] Formatting, import/API checks, and `Pkg.test()` (or relevant subset) have been run; unrun checks are disclosed.
- [ ] No large outputs, private paths/credentials, or obsolete research files were added.
- [ ] The PR explains performance impact, evidence, any migration/deprecations, and remaining limitations.
- [ ] For algorithmic changes, a scientific reviewer approves equations, tests, and interpretation.

## 11. Sources for conventions (consult during implementation)

- Julia package conventions, public APIs and testing: https://pkgdocs.julialang.org/v1/creating-packages/
- Julia package generation: https://julialang.org/contribute/developing_package/
- Julia documentation/docstrings: https://docs.julialang.org/en/v1/manual/documentation/
- Julia style guide: https://docs.julialang.org/en/v1/manual/style-guide/
- Julia performance and type stability: https://docs.julialang.org/en/v1/manual/performance-tips/
- Julia dependency and Manifest semantics: https://pkgdocs.julialang.org/v1/toml-files/
- Aqua.jl QA: https://github.com/JuliaTesting/Aqua.jl
- JET.jl (check Julia compatibility first): https://github.com/aviatesk/JET.jl
- Claude Code instructions, skills, rules and subagents: https://claude.com/resources/articles/steering-claude-code-skills-hooks-rules-subagents-and-more
- Claude Code project memory guide: https://support.claude.com/en/articles/14553240-give-claude-context-claude-md-and-better-prompts

---

**Final architectural rule:** `BUG` owns the **algorithm**, `QMB` and `QTTS` own **the mathematical problem and state representation**, and `Baselines` owns **independent comparisons**. Readability, scientific reproducibility, and external installability are part of the implementation—not optional post-processing.
