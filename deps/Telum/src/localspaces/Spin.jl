# s2: spin * 2
"""
    get_SU2_symmops(s2::Int)

Build single-site spin operators for spin `s2 / 2`.

`s2` is twice the physical spin, so the local Hilbert-space dimension is
`s2 + 1`. The function obtains the SU(2) rank-1 CGT block from LurCGT,
converts it to floating point, fixes the overall sign convention, and returns
`(sp, sz, sm, I, lowering)`: raising, z-component, lowering, identity, and the
SU(2)-normalized lowering operator used as the irrep lowering block.
"""
function get_SU2_symmops(s2::Int)
    S = s2 / 2
    CGT = LurCGT.getNsave_cg3(SU{2}, BigInt, ((2,), (s2,)), [(s2,)])[(s2,)]
    arr = LurCGT.to_float(CGT, Float64)[1] * sqrt(S * (S + 1))
    if arr[2, 1, 1, 1] < 0 arr = -arr end

    sp = sparse(transpose(arr[1, :, :, 1]))
    sz = sparse(transpose(arr[2, :, :, 1]))
    sm = sparse(transpose(arr[3, :, :, 1]))
    I = spdiagm(0 => ones(Float64, s2 + 1))
    lowering = sm * sqrt(2)

    return sp, sz, sm, I, lowering
end

"""
    getSymmetryInfo(opts::SpinOptions)

Build symmetry metadata and maximal-weight IROPs for a spin local space.

`opts.spin` is twice the spin quantum number and must be positive.
`opts.symmetry` may be `:SU2`, `:U1`, or `nothing`. The return value is
`(symm, weights, lowering_ops, mwirops)`, where `symm` is the tuple of symmetry
types, `weights` gives one qlabel per local basis state, `lowering_ops` gives
the local lowering matrices for non-Abelian symmetry generation, and `mwirops`
contains the spin operators exposed to `getLocalSpace`.
"""
function getSymmetryInfo(opts::SpinOptions)
    @assert opts.symmetry === nothing || opts.symmetry == :SU2 || opts.symmetry == :U1 
    "SpinOptions currently only supports :SU2, :U1, or no symmetry"
    @assert opts.spin > 0 

    sp, sz, sm, I, lowering = get_SU2_symmops(opts.spin)

    # Basis: |up>, |down>
    weights = opts.symmetry === nothing ? () : ([(s,) for s in opts.spin:-2:-opts.spin],)
    
    if opts.symmetry == :SU2
        lowering_ops = ([lowering],)
    elseif opts.symmetry == :U1
        lowering_ops = (Matrix{Float64}[],)
    elseif opts.symmetry === nothing
        lowering_ops = ()
    else
        error("SpinOptions currently only supports :SU2, :U1, or no symmetry")
    end

    mwirops = Dict{Symbol, SparseMatrixCSC{Float64, Int}}()
    if opts.symmetry == :SU2
        mwirops[:S] = sp
        symm = (SU{2},)
    else 
        mwirops[:Sp] = sp
        mwirops[:Sz] = sz
        mwirops[:Sm] = sm
        if opts.symmetry == :U1
            symm = (U1,)
        elseif opts.symmetry === nothing
            symm = ()
        else
            error("SpinOptions currently only supports :SU2, :U1, or no symmetry")
        end
    end
    mwirops[:I] = I

    return symm, weights, lowering_ops, mwirops
end
