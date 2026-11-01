# Capability traits (implementation plan, Sections 3.2 and 3.4). Kernels choose their path from
# these traits, never from which domain built the state.

"""
    SymmetryTrait

The symmetry a state's tensors carry, read off the tensors themselves by [`symmetry`](@ref).
Charge-aware kernels (sector-graded sketches, per-sector rank allocation) require a charged
trait; `NoSymmetry` states use dense local layouts with no dummy charge blocks.
"""
abstract type SymmetryTrait end

"One dense block per leg: no conserved charge (QTT states, and spin chains run without symmetry)."
struct NoSymmetry <: SymmetryTrait end
"Abelian U(1) charge, e.g. `2 S^z` for spins."
struct U1Symmetry <: SymmetryTrait end
"Abelian Z2 parity, e.g. the transverse-field Ising spin-flip parity."
struct Z2Symmetry <: SymmetryTrait end
"Non-abelian SU(2): blocks are multiplets and bond dimensions count multiplets."
struct SU2Symmetry <: SymmetryTrait end

"""
    symmetry_trait(syms::Tuple) -> SymmetryTrait

The trait for a Telum symmetry tuple as returned by `Telum.symm(tensor)`: `()`, `(U1,)`,
`(Z{2},)` or `(SU{2},)`. Product symmetries are not used by any state in this package and throw.
"""
symmetry_trait(::Tuple{}) = NoSymmetry()
function symmetry_trait(syms::Tuple)
    length(syms) == 1 ||
        throw(ArgumentError("product symmetry $syms has no SymmetryTrait; add one when needed"))
    symmetry_group = only(syms)
    symmetry_group === U1 && return U1Symmetry()
    symmetry_group === Z{2} && return Z2Symmetry()
    symmetry_group === SU{2} && return SU2Symmetry()
    throw(ArgumentError("symmetry $symmetry_group has no SymmetryTrait; add one when needed"))
end
