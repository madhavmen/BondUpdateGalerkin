module Telum

using MKL
using HPTT_jll
using LinearAlgebra
using Random
using SparseArrays
using SparseArrayKit
using HDF5

import Base: lock, unlock
import LurCGT
import LurCGT: AbelianSymm, NonabelianSymm, Symmetry, SO, SU, Sp, U1, Z
import LurCGT: add_qn, decompose_irop, decompose_space, dimension
import LurCGT: get_CGTom, get_IROP, get_conj_perm, get_dualq
import LurCGT: getNsave_CGTperm, getNsave_CGTSVD, getNsave_CGTQR, getNsave_Xsymbol, getNsave_Conjperm
import LurCGT: getNsave_omlist, getNsave_validout, isabelian
import LurCGT: nzops, remove_zeros, totxt, transf_basis!

"""
    a ⊗ b

Return `kron(b, a)`, the Kronecker product ordered to match Telum's
column-major tensor-leg convention. `a` and `b` must both be matrices or both
vectors. The result is newly allocated and has element type promoted by `kron`.
"""
⊗(a::AbstractMatrix, b::AbstractMatrix) = kron(b, a)
⊗(a::AbstractVector, b::AbstractVector) = kron(b, a)
comm(A, B) = A * B - B * A

include("DiagRMT.jl")

const contraction_cost = Ref{Int}(0)
const svd_cost = Ref{Int}(0)
"""
    accumul_costs

Global `Ref{Bool}` controlling operation-cost accounting. Read or set its value
with `accumul_costs[]`; prefer `set_accumul_costs!` so callers do not depend on
the storage representation. When false, hot paths skip counter updates.
"""
const accumul_costs = Ref(false)

@doc """
    Z{N}

Cyclic Abelian symmetry family of order `N`, re-exported from LurCGT.
""" Z
@doc """`U1` is the one-dimensional continuous Abelian symmetry family re-exported from LurCGT. Its q-label is a one-element integer tuple representing charge.""" U1
@doc """`SU{N}` is the special-unitary symmetry family re-exported from LurCGT. `N` selects the defining-group dimension; q-labels use the package's Dynkin-label convention.""" SU
@doc """`SO{N}` is the special-orthogonal symmetry family re-exported from LurCGT. `N` selects the defining-group dimension and q-labels follow LurCGT's Dynkin convention.""" SO
@doc """`Sp{N}` is the compact symplectic symmetry family re-exported from LurCGT. `N` selects the family parameter and q-labels follow LurCGT's Dynkin convention.""" Sp

@inline function _add_contraction_cost!(cost::Integer)
    accumul_costs[] || return nothing
    contraction_cost[] += Int(cost)
    return nothing
end

@inline function _add_svd_cost!(cost::Integer)
    accumul_costs[] || return nothing
    svd_cost[] += Int(cost)
    return nothing
end

"""
    set_accumul_costs!(enabled) -> Bool

Enable or disable global contraction/SVD cost accounting. `enabled` must be a
`Bool`; the same value is stored in `accumul_costs[]` and returned. This mutates
global process-local state but does not reset existing counters.
"""
function set_accumul_costs!(enabled::Bool)
    accumul_costs[] = enabled
    return enabled
end

"""
    read_reset_costs!() -> (contraction_cost, svd_cost)

Return the accumulated integer cost counters as a two-tuple, then reset both
to zero. Counters are incremented only while `accumul_costs[]` is true; this
function always resets them and mutates global process-local state.
"""
function read_reset_costs!()
    costs = (contraction_cost[], svd_cost[])
    contraction_cost[] = 0
    svd_cost[] = 0
    return costs
end

include("TLArray.jl")
include("sparse_array.jl")
include("io_hdf5.jl")

export AbstractTLArray, TLArray, TLIndex, Itag, DiagRMT
export ProductSymm, productsymm, product_symms, symm, nsymms
export LocalSpaceOptions, SpinOptions, FermionOptions, FermionSOptions
export Z, U1, SU, SO, Sp
export getLocalSpace, getSymmetryInfo, getIdentity, get1jtensor, legflip, contract
export discard_eigen
export svd_leg, svd_cgtsvd
export empty_tlarray, qlabeltype, zero_qlabels, getvac, random_similar
export allsectors_tlarray, random_allsectors_tlarray
export addSingleton, deleteSingleton, ⊗
export fuselegs, splitleg
export oplus, getsub
export complete_oplus_matrix
export printmeta
export to_sparse_array
export findlegs, findleg, matchings, matching, unmatchings, unmatching
export contractables, contractable, uncontractables, uncontractable
export prime, setprime, noprime
export lock, lockp, unlock
export additag, removeitag, replaceitag, setitag
export accumul_costs, read_reset_costs!, set_accumul_costs!
export save_tlarray, load_tlarray
export @lazy

end
