# Symmetry-native MPS container backed by Telum TLArrays. Site tensors are (link_l, site, link_r)
# with itags "L,i" / "S,i" / "L,i+1". Only the invariant that the two legs of a bond carry
# opposite arrows is kept, since svd's arrows depend on sweep direction. Contractions pair legs
# explicitly, so no code assumes which side is '+'.
#
# Moved from BUG-Julia@b155550:src/BondUpdateBUG/symmetric_mps.jl (container, symmetry mode,
# overlap) and src/BondUpdateBUG/expv.jl (tensor_inner). Local spaces and product states moved to
# QMB/Tooling.

# ── symmetry mode ──
# Set once, before building the state and its gates: both go through the local space and must
# match. Tensors carry their own symmetry (`symm(t)`), so the kernels need no mode flag; the mode
# is read only by state constructors and by the fusion memo's key.

const _SYMMETRY = Ref{Symbol}(:U1)

"The current symmetry mode, `:SU2`, `:U1`, `:Z2` or `:none`."
symmetry_mode() = _SYMMETRY[]

"""
    set_symmetry!(sym) -> Symbol

Select the local-space symmetry: `:SU2` (total-spin multiplets), `:U1` (charge-conserving spin
sectors), `:Z2` (Ising parity) or `:none` (one dense block). Call it before building the state
and its operators; they must share a mode.

`:SU2` is not just a faster `:U1`: it stores reduced elements on multiplets, so every state it can
represent has definite total spin. A Neel or domain-wall state is unwritable there. Bond
dimensions then count multiplets.
"""
function set_symmetry!(sym::Symbol)
    sym in (:SU2, :U1, :Z2, :none) ||
        throw(ArgumentError("symmetry must be :SU2, :U1, :Z2 or :none, got $sym"))
    _SYMMETRY[] = sym
    return sym
end

# ── container ───────────────────────────────────────────────────────────────

"""
    SymMPS(tensors, center)

A symmetric MPS. `tensors[i]` is rank-3 with legs `(link_l, site, link_r)`.
`center` is the orthogonality centre: tensors left of it are left isometries,
tensors right of it are right isometries.
"""
mutable struct SymMPS
    tensors::Vector{Any}
    center::Int
end

Base.length(psi::SymMPS) = length(psi.tensors)
Base.getindex(psi::SymMPS, i::Int) = psi.tensors[i]
Base.setindex!(psi::SymMPS, A, i::Int) = (psi.tensors[i] = A)
Base.eachindex(psi::SymMPS) = eachindex(psi.tensors)
Base.copy(psi::SymMPS) = SymMPS(copy(psi.tensors), psi.center)

"Total dimension of leg `l` of `t`, summed over its charge sectors."
leg_dim(t, l::Int) = sum(d for (_, d) in t.spaces[l]; init = 0)

"""
    bond_dims(psi) -> Vector{Int}

The `length(psi) - 1` interior bond dimensions.
"""
bond_dims(psi::SymMPS) = [leg_dim(psi[i], 3) for i = 1:(length(psi)-1)]

"""
    norm(psi::SymMPS)

Valid in canonical form, where every tensor other than the centre is an
isometry and the state norm is the Frobenius norm of the centre tensor.
"""
LinearAlgebra.norm(psi::SymMPS) = norm(psi[psi.center])

"""
    tensor_inner(a, b) -> ComplexF64

The canonical inner product `<a|b>` for two tensors of identical leg structure.

An inner product with no matching charge blocks is **exactly zero**, not an error. That case is
reachable -- e.g. `<v|H|v>` for a purely off-diagonal (pure XX flip-flop) `H` on an Sz-basis
product state -- and must not need a diagonal regulariser.
"""
function tensor_inner(a, b)
    n = length(a.inds)
    legs = ntuple(identity, n)
    r = contract(a', legs, b, legs)
    # `to_concrete!`, not `to_concrete`: `r` is a fresh contract result nothing else sees, so a copy
    # would be waste. This is the hottest allocation site, since re-orthogonalised Lanczos calls it
    # O(maxiter^2) times per solve.
    rc = to_concrete!(r)
    length(rc) == 0 && return 0.0 + 0.0im   # `length` of a TLArray is its sector count
    return ComplexF64(rc[])
end

"""
    overlap(bra::SymMPS, ket::SymMPS) -> ComplexF64

The full MPS inner product `<bra|ket>` (`bra` conjugated), contracted bond by bond from the left.
Lengths and physical spaces must match; bond dimensions may differ. `overlap(psi, psi)` is
`norm(psi)^2`. A structurally zero overlap (no matching charge blocks) is exactly `0`.

The package does not export `inner`: with `using ITensorMPS` that name resolves to ITensorMPS and
throws a MethodError. Use `overlap` or `LinearAlgebra.dot`.
"""
function overlap(bra::SymMPS, ket::SymMPS)
    length(bra) == length(ket) || throw(
        DimensionMismatch(
            "overlap needs equal-length states, got $(length(bra)) and $(length(ket))",
        ),
    )
    L = length(bra)
    # One site: every leg pairs (both links are dim-1 boundaries), so this is
    # exactly the single-tensor inner product.
    L == 1 && return tensor_inner(bra[1], ket[1])
    # The bra's internal links are retagged into a private "BL,i" namespace. Telum rejects two
    # identical indices in one tensor, and the transfer tensor carries bra and ket link_r with the
    # same "L,i+1" tag. The two boundary links keep their tags so bra and ket contract directly
    # there.
    braR = Vector{Any}(undef, L)
    for i = 1:L
        t = bra[i]
        i > 1 && (t = setitag(t, 1, "BL,$i"))
        i < L && (t = setitag(t, 3, "BL,$(i + 1)"))
        braR[i] = to_concrete(t)
    end
    # Left boundary link (leg 1) pairs here; carry (bra_link_r, ket_link_r) right.
    E = contract(braR[1]', (1, 2), ket[1], (1, 2))
    for i = 2:(L-1)
        tmp = contract(E, (1,), braR[i]', (1,))      # (ket_link, bra_site, bra_link_r)
        E = contract(tmp, (1, 2), ket[i], (1, 2))  # (bra_link_r, ket_link_r)
    end
    # Last site closes ALL remaining legs -- including the dim-1 RIGHT boundary
    # link (leg 3) -- so the result is a rank-0 scalar, not a dangling rank-2.
    tmp = contract(E, (1,), braR[L]', (1,))          # (ket_link, bra_site, bra_link_r)
    # The right boundary carries each state's TOTAL charge. If bra and ket share
    # no sector there they live in different charge sectors and the overlap is
    # structurally zero -- return it rather than let `contract` assert on the
    # mismatched space (the same zero `tensor_inner` gives for no matching blocks).
    braq = Set(first(sp) for sp in braR[L].spaces[3])
    ketq = Set(first(sp) for sp in ket[L].spaces[3])
    isdisjoint(braq, ketq) && return 0.0 + 0.0im
    s = contract(tmp, (1, 2, 3), ket[L], (1, 2, 3))
    r = to_concrete(s)
    length(r) == 0 && return 0.0 + 0.0im             # `length` of a TLArray is its sector count
    return ComplexF64(r[])
end

"`dot(bra, ket)` is `overlap(bra, ket)`. See `overlap`."
LinearAlgebra.dot(bra::SymMPS, ket::SymMPS) = overlap(bra, ket)

# ── the state contract (Core/Interfaces/state.jl) ───────────────────────────

nsites(psi::SymMPS) = length(psi)
physical_dimension(psi::SymMPS, i::Int) = leg_dim(psi[i], 2)
bond_dimension(psi::SymMPS, b::Int) = leg_dim(psi[b], 3)
bond_dimensions(psi::SymMPS) = bond_dims(psi)
orthogonality_center(psi::SymMPS) = psi.center
canonicalize!(psi::SymMPS, i::Int) = canonical!(psi, i)
inner_product(bra::SymMPS, ket::SymMPS) = overlap(bra, ket)
symmetry(psi::SymMPS) = symmetry_trait(symm(psi[1]))

# ── from BUG-Julia@b155550:src/RSVDCBEBondUpdate/baselines/tdvp2.jl ──

"Total stored elements of a TLArray -- the honest size of the object, block-sparse."
tensor_elements(t) = sum(length(r) for r in t.RMTs; init = 0)
