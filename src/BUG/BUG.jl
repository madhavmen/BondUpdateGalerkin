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

using LinearAlgebra
using LurCGT, Telum

include("Core/Interfaces/capabilities.jl")
include("Core/Interfaces/state.jl")
include("Core/TensorNetworks/symmetric_mps.jl")
include("Core/TensorNetworks/canonical_form.jl")
include("Core/TensorNetworks/sectors.jl")
include("Core/TensorNetworks/mpo.jl")
include("Core/Sweeps/frames.jl")

# state interface (Core/Interfaces)
export SymmetryTrait, NoSymmetry, U1Symmetry, Z2Symmetry, SU2Symmetry, symmetry
export nsites,
    physical_dimension,
    bond_dimension,
    bond_dimensions,
    orthogonality_center,
    canonicalize!,
    inner_product
# MPS storage (Core/TensorNetworks)
export SymMPS, set_symmetry!, symmetry_mode, leg_dim, bond_dims, overlap, tensor_inner
export canonical!, move_left!, move_right!
export left_gram, right_gram, left_isometry_defect, right_isometry_defect
export perp_component, perp_component_right
export reachable_sectors, fusion_basis, fuse_spaces, add_charge, dual_charge, sector_dim
export align_charge, fusion_stats, reset_fusion_stats!, clear_fusion_cache!, set_fusion_budget!
export MPO, mpo_virtual_dims, onsite_mpo, mpo_product
# bond frames (Core/Sweeps)
export BondFrame, bond_frame, frame_theta, two_site_block

end
