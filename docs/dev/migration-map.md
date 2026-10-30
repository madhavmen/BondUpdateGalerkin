# Migration map (WP0, temporary)

Every file of BUG-Julia `b155550` (tag `pre-bondupdategalerkin`) with its disposition. This file is deleted when WP8 closes; the plan's Section 8 stays the source of truth for `src/`.

- **Index:** [`inventory_b155550.csv`](inventory_b155550.csv), one row per file: `path, area, kind, target, treatment, references` (`kind` is core, domain, baseline, study, example, obsolete or unknown; `references` lists which of `BondUpdateBUG`, `RSVDCBEBondUpdate`, `Telum`, `LurCGT` the file names).
- **Regenerate:** `python -I tools/inventory.py implementation_plan.md TREE inventory_b155550.csv BUG-JULIA-CHECKOUT`, with `TREE` an extracted `git archive b155550 src tests test examples benchmarks`.
- **Check Section 8:** `python -I tools/check_map.py implementation_plan.md TREE` lists legacy `src/` files missing from Section 8, names a row assigns that its legacy files do not define, and names defined in more than one legacy file. It printed nothing on 30 October 2026. Control: restoring the old `heisenberg_su` and dropping the `BUGJulia.jl` row makes it report both.

## Counts (439 files)

| Area | Files | Disposition |
|---|---|---|
| `src/` | 73 | All in Section 8: 42 core, 19 domain, 7 baseline, 5 study (exploratory geometries) |
| `tests/`, `test/` | 80 | Mapped to `test/{BUG,QMB,QTTS,Baselines,common}` by tested behaviour; 2 go to `studies/` with their geometries |
| `examples/` | 8 | 7 ported to `examples/QMB/` in WP7; `plot_examples.py` not copied |
| `benchmarks/` | 237 | 3 independent references to `test/common/`; 128 study drivers to `benchmarks/QMB`, `benchmarks/QTTS`, `benchmarks/kernels` or `studies/`; 24 probes and wrappers not copied; **82 unknown, decided in WP8** |
| `scripts/` | 30 | 27 diagnostic probes not copied; `smoke_examples.jl` becomes the examples check; 2 sbatch wrappers only if a kept study needs them |
| `notebooks/` | 11 | Rewritten as `docs/` tutorials in WP7 |

## Corrections made to Section 8 in WP0

| Row | Was | Now | Evidence at `b155550` |
|---|---|---|---|
| (new) | `BUGJulia.jl` unmapped | `BondUpdateGalerkin.jl`; workload rebuilt later | Module root with the precompile workload |
| `operators/henv.jl` | `heisenberg_su` | `heisenberg_su2_chain` | `henv.jl:74` |
| `operators/mpo.jl` | `heisenberg_su2_mpo` unassigned | `QMB/Models/ClosedSystem/xxz.jl` | `mpo.jl:137` |
| `qtt/qtt_flows.jl` | `_eff` | `_tc`, `_eff1`, `_eff2` | `qtt_flows.jl:63,120,128` |
| `qtt/qtt_flows.jl` | `qtt_tdvp` | `qtt_tdvp2_step!` | `qtt_flows.jl:611` |
| `qtt/qtt_flows.jl` | `_gmres_real`, `_unpack`, `_replace_state!` unassigned | stage solvers and `bug_step.jl` | `qtt_flows.jl:271,319,410` |
| 8.5 `deps/` | archive tag | `deps/Telum` vendored (our fork) | No-symmetry SVD fix already in `deps/Telum/src/svd.jl`, so `deps/build.jl` is not needed |

Split rows now state the default: names a row does not list go to its first destination.

**Dropped (decided 30 October 2026):** `sector_report` and `sector_report_right` in `BondUpdateBUG/sectors.jl` are broken at `b155550`: commit `3024066` removed their helper `_complement_basis`, so they throw `UndefVarError`. Their only other callers are diagnostic scripts that are not copied. They and their 6 tests in `test_sectors.jl` are left out when `sectors.jl` moves.

## Test files

The `target` column of the index holds the per-file map. The grouping:

| New test directory | Legacy test files |
|---|---|
| `test/BUG/Core/TensorNetworks` | `test_symmetric_mps`, `test_sectors`, `test_fusion_cache`, the `apply_gate` part of `test_gates`, and `test/` (Telum kernel checks) |
| `test/BUG/Core/{LocalSolvers,Sweeps,Environments,RankAdaptation}` | `test_expv`, `test_adaptive_expv`, `test_krylov_recycle`, `test_hq_root`, `test_frame`, `test_parallel_kls`, `test_env_inplace`, `test_adaptive_krylov`, `test_krylov_closure*`, `test_adaptive_compression`; `test_adaptive_diagnostics` |
| `test/BUG/Randomised` | `test_adaptive_rsvd`, `test_adaptive_search`, `test_dd_cbe*`, `test_hq_kernels` |
| `test/BUG/TimeEvolution`, `test/BUG/ITE`, `test/BUG` | `test_sweeps` (BUG parts), `test_adaptive_parallel`; `test_imaginary_time`; `test_adaptive_api` |
| `test/Baselines` | `test_adaptive_dmrg`, `test_tdvp2_*`, the DMRG/TDVP parts of `test_sweeps` |
| `test/QMB/ClosedSystem`, `test/QMB/Tooling` | `test_z2_ising`, `test_spin_xx`, `test_spin_jw`, `test_xx_efficiency`, `test_tfim_mpo`, `test_oat`, `test_cbe_bug_exact_refs`, `test_symmetry_parity`, `test_physics` (Heisenberg; Haldane-Shastry to `studies/`); `test_pair_mpo`, the gate part of `test_gates` |
| `test/QMB/OpenSystem/MPDO` | `test_liouville` |
| `test/QMB/OpenSystem/MCWF` | `test_dissipative_t1`, `_t2`, `_t2_g12`, `_u18` (no-jump step); `_t3_a1`, `_t3_a2` (models); `_t4`, `_t4_b`, `_t4_conv`, `_t5`, `_t6` (jumps, controller, restart) |
| `test/QTTS` | `test_qtt_task1`, `test_qtt_task2` (stage-solver parts to `test/BUG/TimeEvolution`) |
| `test/common` | `tests/common/*`, plus `benchmarks/exact_sparse.jl`, `benchmarks/hubbard_aa_exact.jl`, `benchmarks/open_system/sparse_lindblad.jl`, which tests include |

## Moved so far

| PR | Legacy | New | Deviations from Section 8 |
|---|---|---|---|
| 2 | `BondUpdateBUG/symmetric_mps.jl`, `symmetric_mps_canonical.jl`, `sectors.jl`, `frame.jl`; `tensor_inner` from `expv.jl` | `BUG/Core/TensorNetworks/{symmetric_mps,canonical_form,sectors}.jl`, `BUG/Core/Sweeps/frames.jl`; `QMB/Tooling/{local_spaces,initial_states}.jl`, `QMB/Observables/local_observables.jl` | `tensor_inner` goes to TensorNetworks, not LocalSolvers, because `overlap` needs it. `dimer_state` waits for `gates.jl` (it projects with Heisenberg gates). Tests not yet moved: the dimer case and the `sz_overlaps` testset (need `tdvp2_step!`), and the d=4 allocation guard of `test_fusion_cache.jl` (needs `cbe_bug_step!`) |
| 3 | `qtt/qtt_grid.jl`, `qtt/qtt_telum.jl`; from `qtt/qtt_cores.jl` the TT basics, grid placement, `op_*` and `tt_scale`/`tt_conj`/`tt_add`/`tt_lincomb`; from `operators/mpo.jl` `MPO`, `mpo_virtual_dims`, the automaton blocks, `onsite_mpo`; `mpo_product` from `operators/mpo_apply.jl`; part of `tests/common/qtt_fixtures.jl` and U01 of `test_qtt_task1.jl` | `QTTS/Tooling/{grid,cores,mpo_algebra,state_adapter}.jl`, `QTTS/Elementwise/pointwise.jl`, `BUG/Core/TensorNetworks/mpo.jl`; `test/QTTS/Tooling/test_{grid,cores,operators}.jl`, `test/BUG/Core/TensorNetworks/test_mpo.jl`, `test/common/qtt_fixtures.jl` | New `QTTOrdering` (any number of axes) holds the site map that was in `QTTGrid`; `qtt_embed` takes an ordering. `onsite_mpo` takes `Iloc` as a required keyword (its default was QMB's spin space). `state_adapter.jl` builds its leg templates from Telum's site space, not `product_state`/`local_space` (QMB). `_as3`/`_as4` go to `cores.jl` under the first-destination rule. Not yet moved, with their dependencies: `qtt_apply`, `apply_mpo`, `apply_factor`, `global_krylov_states` (truncation, PR 4); `op_zipup`, `tt_hadamard`, `tt_product`, `tt_density` (zip-up products, PR 8); `mpo_from_terms`, `xxz_mpo`, `mpo_sum`, `no_jump_mpo`, `site_op_product` (models and open systems, PR 6); the stencil and environment fixtures and U01's `qtt_shift` wrap (derivatives, PR 7; PR 3 checks the wrap with a shift built from `op_ket_bra`). `xxz_mpo` needs `xxz_chain`, so it goes to `QMB/Models/ClosedSystem/xxz.jl`, not to BUG as the first-destination rule would say |

## Baseline

The pass/fail state of every legacy test at `b155550` comes from cluster job 17610132 and is recorded here when it finishes.
