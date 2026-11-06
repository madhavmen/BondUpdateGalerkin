# Local Hilbert spaces and their charge labels.

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

const _Z2_Q0 = ((0,),)      # sigma^x = +1

const _Z2_Q1 = ((1,),)      # sigma^x = -1

# `_localspace_cgt_fields` wants `Vector{Tuple{NTuple{QD, NTuple{N, Tuple{Vararg{Int}}}}, ...}}`
# and Array is INVARIANT in its element type, so `Tuple{Tuple{Int}}` does NOT match
# `Tuple{Tuple{Vararg{Int}}}` and dispatch fails with a confusing MethodError. Spell them out.
const _QT = Tuple{Vararg{Int}}

const _DAT2 = Tuple{NTuple{2,NTuple{1,_QT}},Array{Float64,3}}

const _DAT3 = Tuple{NTuple{3,NTuple{1,_QT}},Array{Float64,4}}

const _ISING_SPACES = Dict{Symbol,Any}()

"Charge-preserving (diagonal) rank-2 operator with entries `v0`, `v1` on the two sectors."
function _z2_diag_op(v0::Float64, v1::Float64)
    sp = [(_Z2_Q0, 1), (_Z2_Q1, 1)]
    data = _DAT2[
        ((_Z2_Q0, _Z2_Q0), reshape([v0], 1, 1, 1)),
        ((_Z2_Q1, _Z2_Q1), reshape([v1], 1, 1, 1)),
    ]
    spaces = (sp, sp)
    f = _localspace_cgt_fields(data, (Z{2},), spaces)
    return TLArray(
        (Z{2},),
        f.qlabels,
        f.wmatdata,
        f.wmatinfo,
        f.RMTs,
        (TLIndex("s", '+'), TLIndex("s", '-')),
        spaces,
    )
end

"""
The charge-1 flip `Z`, as a rank-3 operator whose op leg carries charge 1.

BOTH matrix elements share one op-leg charge, which is the whole reason Z2 is representable
here: `1 = 0+1` and `0 = 1+1` are both true mod 2. Under U(1) this would need two different
op-leg charges (`+1` and `-1`) and would not be one irreducible operator at all.
"""
function _z2_flip_op()
    sp = [(_Z2_Q0, 1), (_Z2_Q1, 1)]
    opsp = [(_Z2_Q1, 1)]
    data = _DAT3[
        ((_Z2_Q1, _Z2_Q0, _Z2_Q1), reshape([1.0], 1, 1, 1, 1)),
        ((_Z2_Q0, _Z2_Q1, _Z2_Q1), reshape([1.0], 1, 1, 1, 1)),
    ]
    spaces = (sp, sp, opsp)
    f = _localspace_cgt_fields(data, (Z{2},), spaces)
    return TLArray(
        (Z{2},),
        f.qlabels,
        f.wmatdata,
        f.wmatinfo,
        f.RMTs,
        (TLIndex("s", '+'), TLIndex("s", '-'), TLIndex("op", '-')),
        spaces,
    )
end

"""
    ising_local_space(sym = symmetry_mode()) -> (; I, X, Z)

The spin-½ local space for the TFIM **in the sigma^x basis**, for `:Z2` or `:none`.

`:Z2` carries two charge sectors and a rank-3 `Z`; `:none` is one dense block with a rank-2
`Z`. THE SAME BASIS IN BOTH, deliberately: it is what makes a `:Z2`-vs-`:none` comparison a
statement about storage rather than about two different bases, so the two must agree digit for
digit on every observable.
"""
function ising_local_space(sym::Symbol = symmetry_mode())
    haskey(_ISING_SPACES, sym) && return _ISING_SPACES[sym]
    s = if sym === :Z2
        (; I = _z2_diag_op(1.0, 1.0), X = _z2_diag_op(1.0, -1.0), Z = _z2_flip_op())
    elseif sym === :none
        # Same operators, one dense block. Telum's own no-symmetry builder, so the tensors are
        # constructed exactly as `local_space(:none)`'s are.
        ops = Dict{Symbol,SparseMatrixCSC{Float64,Int}}(
            :I => spdiagm(0 => ones(Float64, 2)),
            :X => spdiagm(0 => Float64[1.0, -1.0]),
            :Z => sparse(Float64[0 1; 1 0]),
        )
        _getLocalSpace_no_symmetry(ops, ("s", "s", "op"))
    else
        throw(
            ArgumentError(
                "the transverse-field Ising local space is defined for :Z2 and :none, got $sym. " *
                "The TFIM has no U(1) spin symmetry -- Z_i Z_{i+1} does not conserve Sz -- and no " *
                "SU(2) symmetry either, since the transverse field singles out an axis.",
            ),
        )
    end
    _ISING_SPACES[sym] = s
    return s
end

const _SPIN_SPACES = Dict{Int,Any}()

"""
    spin_space(twoS) -> NamedTuple

The U(1) spin-`twoS/2` local space `(; I, Sp, Sz, Sm)`, charge `2 S^z` (`twoS:-2:-twoS`, each a
one-dimensional sector). `twoS = 1` IS `local_space(:U1)`, so the s = 1/2 chain is the package's
own. Cached, like `local_space`: build it before any sweep runs.
"""
function spin_space(twoS::Int)
    twoS >= 1 || throw(ArgumentError("twoS = 2s must be >= 1, got $twoS"))
    twoS == 1 && return local_space(:U1)
    return get!(() -> getLocalSpace(SpinOptions(:U1, twoS), ("s", "s", "op")), _SPIN_SPACES, twoS)
end
