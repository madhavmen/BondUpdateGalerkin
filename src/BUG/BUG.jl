"""
    BUG

The one basis-update-and-Galerkin engine, independent of what the tensor train represents.

Include directories, layered in this order (each may use only the ones before it):

- `Core/`: state/operator/problem interfaces, MPS/MPO storage, environments, local solvers,
  deterministic rank adaptation, sweeps and diagnostics.
- `Randomised/`: sketch-based controlled bond expansion, plugged into the sweeps as a
  basis-selection strategy.
- `TimeEvolution/`: real-time steps for any state implementing the `Core` interfaces.
- `ITE/`: imaginary-time steps under the same state contract.

These are directories, not submodules: a submodule named `Core` would shadow Julia's `Core`.
"""
module BUG

using LinearAlgebra, Random
using LurCGT, Telum

# The order is load-bearing: a file may use, in a signature or a struct field, only the types of
# the files above it (function bodies resolve names at call time, so they are free).
include("Core/Interfaces/capabilities.jl")
include("Core/Interfaces/state.jl")
include("Core/TensorNetworks/symmetric_mps.jl")
include("Core/TensorNetworks/canonical_form.jl")
include("Core/TensorNetworks/sectors.jl")
include("Core/Sweeps/frames.jl")
include("Core/TensorNetworks/mpo.jl")
include("Core/Environments/mpo_environments.jl")
include("Core/Environments/effective_operators.jl")
include("Core/LocalSolvers/krylov.jl")
include("Core/LocalSolvers/root_methods.jl")
include("Core/RankAdaptation/basis_budget.jl")
include("Core/RankAdaptation/truncation.jl")
include("Core/RankAdaptation/basis_expansion.jl")
include("Core/diagnostics.jl")
include("Core/RankAdaptation/krylov_enrichment.jl")
include("Core/RankAdaptation/krylov_recurrence.jl")
include("Core/RankAdaptation/krylov_lanczos.jl")
include("Core/RankAdaptation/krylov_arnoldi.jl")
include("Core/RankAdaptation/krylov_closure_frames.jl")
include("Core/RankAdaptation/krylov_frontier_bonds.jl")
include("Core/RankAdaptation/krylov_closure_loop.jl")
include("Core/Sweeps/half_sweeps.jl")
include("Core/Sweeps/frontier_sweeps.jl")
include("Core/Sweeps/frontier_closure.jl")
include("Core/Sweeps/parallel_kls.jl")
# The sweeps above call their basis selection only through a `BasisExpansion`; `Randomised/` adds
# the strategies and their methods.
include("Randomised/sketch.jl")
include("Randomised/range_finder.jl")
include("Randomised/cbe.jl")
include("Randomised/dd_cbe.jl")
include("Randomised/dd_cbe_controller.jl")
include("Randomised/dd_cbe_expand.jl")
include("Randomised/krylov_dd_cbe.jl")
include("TimeEvolution/bug_step.jl")
include("TimeEvolution/bug_step_stages.jl")
include("TimeEvolution/coupled_dd_krylov.jl")
include("TimeEvolution/neel_bulk_retry.jl")
include("TimeEvolution/bug_step_variants.jl")
include("TimeEvolution/flows.jl")
include("TimeEvolution/stage_gmres.jl")
include("TimeEvolution/stage_solvers.jl")
include("ITE/schedules.jl")
include("ITE/ite_step.jl")

# state interface (Core/Interfaces)
export SymmetryTrait, NoSymmetry, U1Symmetry, Z2Symmetry, SU2Symmetry, symmetry
export nsites,
    physical_dimension,
    bond_dimension,
    bond_dimensions,
    orthogonality_center,
    canonicalize!,
    inner_product
# MPS and MPO storage (Core/TensorNetworks)
export SymMPS, set_symmetry!, symmetry_mode, leg_dim, bond_dims, overlap, tensor_inner
export canonical!, move_left!, move_right!, tensor_elements
export left_gram, right_gram, left_isometry_defect, right_isometry_defect
export perp_component, perp_component_right
export reachable_sectors, fusion_basis, fuse_spaces, add_charge, dual_charge, sector_dim
export align_charge, fusion_stats, reset_fusion_stats!, clear_fusion_cache!, set_fusion_budget!
export state_dim, state_bond_dims, multiplet_ratio, local_dim
export XXZTerm, XXZChain, hamiltonian_terms
export MPO, mpo_from_terms, mpo_virtual_dims, onsite_mpo, mpo_sum, no_jump_mpo
export mpo_product, factor_product, apply_factor, apply_mpo, global_krylov_states
# bond frames and sweeps (Core/Sweeps)
export BondFrame, bond_frame, frame_theta, two_site_block, parallel_kls_step!
# environments and effective operators (Core/Environments)
export boundary_channels, left_channels, right_channels, push_left_channels, push_right_channels
export left_env_stack, right_env_stack
export apply_h_two_site, mpo_energy, mpo_matrix_element, mpo_variance
export MPOLink, MPOLeftEnvStack, MPORightEnvStack, MPOZeroSiteH, MPOOneSiteH, shift_h
export zero_site_h, apply_zero_site, one_site_h, apply_one_site
# local solvers (Core/LocalSolvers)
export expv, lanczos_expv, arnoldi_expv, radau2_gmres_expv, ROOT_METHODS, root_order
export root_stability, hermitian_tridiagonal_exp_coeffs
export enable_krylov_log, disable_krylov_log, get_krylov_log, KRYLOV_LOG
# rank adaptation (Core/RankAdaptation)
export truncate_sweep!, truncate_recursive!, AdaptiveBudget, BasisAllocation, basis_allocation
export indicator, AdaptiveLedger, merge_ledger!, work_ceiling, bond_work_ceilings, bulk_work_ceilings
export bond_work_ceilings_strict, bond_frame_root_ceilings
export budget_status, budget_diagnostic
export BasisExpansion, expand_frame, sketch_closures, folded_sketch_closures, sampled_leak
# diagnostics (Core/diagnostics.jl)
export CBEBugRunInfo, CBEBugSweepInfo, KernelCounters, StepRecord, OneSiteOp
# randomised controlled bond expansion, DD-QR (Randomised)
export sector_graded_sketch, CBEExpansion
export DDQRExpansion, dd_cbe_expand, TwoSiteSketchOp, RootOp
export apply_two_site_sketch!, apply_one_site!, apply_root!
# real and imaginary time (TimeEvolution, ITE)
export cbe_bug_step!, cbe_bug_coupled_step, cbe_bug_bulk_retry_step!
export ite_step!, ite_evolve!, log_panel_steps, uniform_beta_steps, beta_grid, uniform_schedule
export geometric_schedule
# flows and implicit stage equations in the root space (TimeEvolution)
export QTTFlow, LinearFlow, flow_op, flow_jac_ops, flow_normalized, frozen_flow
export frame_state, frame_project, frame_operator_matrix, frame_conj
export frame_root_envs, root_envs_matrix, root_envs_kron_sum
export stage_field_ops, stage_field, stage_jacobian, stage_linear_envs
export reduced_field, reduced_jacobian, RK_TABLEAUX, qtt_root_for_order, rk_stage_solve

end
