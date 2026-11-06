"""
    QTTS

Quantics tensor trains (QTT) for PDEs, evolved by the BUG engine.

Include directories:

- `Tooling/`: grids and bit orderings, QTT construction, MPO algebra, derivative MPOs and the
  adapter to the `BUG` state interface.
- `Elementwise/`: pointwise operations on QTT functions (Hadamard products, `|u|^2`).
- `Models/Heston/`, `Models/GrossPitaevskii/`: equations, parameters, initial data, references
  and observables.
"""
module QTTS

using ..BUG
# the frame and flow interface of the BUG stage solver, extended for dense QTT cores
import ..BUG: frame_state, frame_project, frame_operator_matrix, frame_conj
import ..BUG: frame_root_envs, root_envs_matrix, root_envs_kron_sum
import ..BUG: stage_field_ops, stage_field, stage_jacobian, stage_linear_envs
import ..BUG: flow_op, flow_jac_ops, flow_normalized
using ..BUG: KLFrames, _replace_state!
using LinearAlgebra
using Telum

include("Tooling/grid.jl")
include("Tooling/cores.jl")
include("Elementwise/pointwise.jl")
include("Elementwise/products.jl")
include("Tooling/mpo_algebra.jl")
include("Tooling/state_adapter.jl")
include("Tooling/sampling.jl")
include("Tooling/derivatives.jl")
include("Tooling/operator_assembly.jl")
include("Tooling/frames.jl")
include("Tooling/stage_fields.jl")
include("Tooling/nonlinear_step.jl")
include("Models/Heston/heston.jl")
include("Models/Heston/heston_reference.jl")
include("Models/GrossPitaevskii/paddle.jl")
include("Models/GrossPitaevskii/gross_pitaevskii.jl")
include("Models/GrossPitaevskii/observables.jl")
include("Models/GrossPitaevskii/ground_state.jl")

# grids and bit orderings (Tooling/grid.jl)
export QTTOrdering, QTTGrid, qtt_grid, qtt_length, qtt_sites, qtt_bits, qtt_point
export qtt_site_permutation, qtt_coords, qtt_cell
# TT cores (Tooling/cores.jl)
export tt_dense, tt_from_dense, tt_from_samples, tt_compress, tt_ranks, tt_dot, tt_norm
export tt_evaluate, qtt_embed, qtt_function, qtt_separable, qtt_values, qtt_value
# pointwise operations (Elementwise/pointwise.jl)
export tt_scale, tt_conj, tt_add, tt_lincomb
# operator cores (Tooling/mpo_algebra.jl)
export op_identity, op_scale, op_add, op_lincomb, op_compress, op_adjoint, op_transpose
export op_compose, op_diag, op_apply, op_dense, op_ket_bra
# the BUG state and MPO (Tooling/state_adapter.jl)
export qtt_mps, qtt_cores, qtt_to_mpo, qtt_mpo_cores, qtt_l2norm, qtt_apply
# derivatives and operator assembly (Tooling/derivatives.jl, Tooling/operator_assembly.jl)
export QTT_BCS, qtt_shift, qtt_derivative, qtt_diag_1d, qtt_d, qtt_coord_diag, qtt_op
# Heston (Models/Heston)
export HestonParams, feller, heston_op, heston_mpo
export heston_riccati, heston_affine_field, heston_call, heston_mass, heston_moment
# elementwise products and the compressed operator application (Elementwise/products.jl)
export tt_hadamard, tt_product, tt_density, tt_hadamard_fit, op_zipup
# normalisation and sampling (Tooling/sampling.jl)
export qtt_normalize!, qtt_sample
# dense root-space frames and the nonlinear step (Tooling/frames.jl, nonlinear_step.jl)
export QTTFrames, frame_miss, frame_defect, qtt_bug_step!, qtt_step_tolerance
# per-summand root environments of the stage solve (Tooling/frames.jl, Tooling/stage_fields.jl)
export root_apply, StageFieldOps
# Gross-Pitaevskii (Models/GrossPitaevskii)
export GPPaddle, paddle_state, gp_paddle_op, GPModel, gp_model, gp_linear_op, gp_density
export gp_hamiltonian_op, GPHamiltonianCache, qtt_state_tag, gp_hamiltonian
export gp_vector_field, gp_jvp, gp_density_audit, GPFlow
export gp_parts, gp_observables, vortex_count, gp_ground_state!, gp_gradient_descent!

end
