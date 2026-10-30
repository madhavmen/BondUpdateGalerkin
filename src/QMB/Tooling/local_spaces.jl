# Local Hilbert spaces and their charge labels.
#
# Moved from BUG-Julia@b155550:src/BondUpdateBUG/symmetric_mps.jl (the spin-1/2 local space).

const _LOCAL_SPACES = Dict{Symbol,Any}()

"""
    local_space(sym = symmetry_mode())

The spin-½ local space for mode `sym`, built once per mode and cached.
`SpinOptions` takes 2·S as an `Int`, so spin-½ is `SpinOptions(:U1, 1)` /
`SpinOptions(nothing, 1)`. Both branches expose `(:I, :Sp, :Sz, :Sm)`; under
`:U1` the raising/lowering operators are rank-3 (their op-leg carries the ±2
charge), under `:none` they are rank-2 plain matrices.
"""
function local_space(sym::Symbol = symmetry_mode())
    haskey(_LOCAL_SPACES, sym) && return _LOCAL_SPACES[sym]
    # `:Z2` IS NOT A SPIN LOCAL SPACE. It is the transverse-field Ising space, in the sigma^x
    # basis, with the Z2 spin-flip parity as its charge -- a different local space with
    # different operators (`I, X, Z`, no `Sp/Sz/Sm`). Asking for it here is asking for a
    # Heisenberg-style space that does not exist under Z2, so it refuses rather than returning
    # something spin-like.
    sym === :Z2 && throw(
        ArgumentError(
            "local_space is the spin-1/2 (Heisenberg) space and has no :Z2 form -- the TFIM has " *
            "no U(1)/SU(2) spin symmetry. Use the Z2 Ising local space instead.",
        ),
    )
    opts =
        sym === :SU2 ? SpinOptions(:SU2, 1) :
        sym === :U1 ? SpinOptions(:U1, 1) :
        sym === :none ? SpinOptions(nothing, 1) :
        throw(ArgumentError("symmetry must be :SU2, :U1 or :none, got $sym"))
    _LOCAL_SPACES[sym] = getLocalSpace(opts, ("s", "s", "op"))
    return _LOCAL_SPACES[sym]
end

"U(1) sector label for a single up spin (charge 2·Sz = +1)."
const SECTOR_UP = ((1,),)
"U(1) sector label for a single down spin (charge 2·Sz = -1)."
const SECTOR_DOWN = ((-1,),)
"SU(2) irrep label for a single spin-½ site (2S = 1). MEASURED from `local_space(:SU2)`."
const SECTOR_SU2_HALF = ((1,),)
"SU(2) irrep label for a singlet (2S = 0) -- what two spin-½ multiplets fuse down to."
const SECTOR_SU2_SINGLET = ((0,),)
