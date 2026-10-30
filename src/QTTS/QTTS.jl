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
using LinearAlgebra
using Telum

include("Tooling/grid.jl")
include("Tooling/cores.jl")
include("Elementwise/pointwise.jl")
include("Tooling/mpo_algebra.jl")
include("Tooling/state_adapter.jl")

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
export qtt_mps, qtt_cores, qtt_to_mpo, qtt_mpo_cores, qtt_l2norm

end
