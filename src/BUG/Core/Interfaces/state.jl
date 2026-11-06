# The state contract every tensor-train state implements. docs/dev/interfaces.md holds the
# contract table; test/BUG/Core/Interfaces runs one conformance
# suite against every state type.
#
# Copy semantics: `copy(psi)` copies the site-tensor vector and shares the tensors. That is safe
# because every operation in this package REPLACES a site tensor (`psi[site] = ...`) and never
# mutates one in place; a kernel that mutates a tensor must own it first.

"""
    nsites(psi) -> Int

Number of sites (tensors) of the tensor train.
"""
function nsites end

"""
    physical_dimension(psi, site) -> Int

Dimension of the physical leg of site `site`, summed over its charge sectors.
"""
function physical_dimension end

"""
    bond_dimension(psi, bond) -> Int

Dimension of bond `bond`, the link between sites `bond` and `bond + 1`, summed over charge sectors.
"""
function bond_dimension end

"""
    bond_dimensions(psi) -> Vector{Int}

The `nsites(psi) - 1` interior bond dimensions.
"""
function bond_dimensions end

"""
    orthogonality_center(psi) -> Int

The site every tensor to the left of is a left isometry of, and every tensor to the right a right
isometry of.
"""
function orthogonality_center end

"""
    canonicalize!(psi, site) -> psi

Move the orthogonality centre to site `site` without truncation. The state is unchanged; only its
gauge is.
"""
function canonicalize! end

"""
    inner_product(bra, ket) -> ComplexF64

`<bra|ket>` with `bra` conjugated, in the Euclidean metric of the tensor-train coefficients. A
structurally zero product (no shared charge sector) is exactly zero.
"""
function inner_product end

"""
    symmetry(psi) -> SymmetryTrait

The symmetry the state's tensors carry.
"""
function symmetry end
