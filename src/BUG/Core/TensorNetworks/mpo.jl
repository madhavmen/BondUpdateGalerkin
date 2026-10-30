# MPO storage: the operator as one rank-4 tensor per site with fused virtual legs, on-site sums, and
# the exact MPO-MPS product.
#
# Moved from BUG-Julia@b155550:src/RSVDCBEBondUpdate/operators/mpo.jl (`MPO`, the automaton blocks,
# `onsite_mpo`) and operators/mpo_apply.jl (`mpo_product`). `onsite_mpo` takes the local identity
# `Iloc` as a required keyword: the spin-1/2 default lived in a model's local space, which BUG may not
# reference. `mpo_from_terms` and `mpo_sum` move with their first callers (the XXZ model and the
# open-system generators), `apply_mpo` with the truncation it compresses by.
# The leg and prime conventions are load-bearing: a wrong prime does not throw, it traces the wrong pair.

"""
    MPO(W)

An operator as a matrix product operator, one rank-4 tensor per site with legs
`(w_l, s_ket, s_bra, w_r)`. The boundary virtual legs of `W[1]` and `W[end]` are dim 1, from
trimming, so no separate boundary vector is carried.

The ket's site leg contracts with leg 2 and leg 3 is the output, so the tensor stores the
TRANSPOSE of the applied matrix on the site legs.
"""
struct MPO
    W::Vector{Any}
end

Base.length(mpo::MPO) = length(mpo.W)
Base.getindex(mpo::MPO, i::Int) = mpo.W[i]

"Virtual (MPO bond) dimension on each link `0 … L`."
mpo_virtual_dims(mpo::MPO) = vcat(
    [sum(d for (_, d) in mpo[1].spaces[1]; init = 0)],
    [sum(d for (_, d) in mpo[i].spaces[4]; init = 0) for i = 1:length(mpo)],
)

# Each automaton row is written out literally and `oplus` direct-sums them, inferring the zero blocks
# from the row and column spaces. The op-leg carries the charge (under `:U1` the `Sp` leg is its
# channel's sector; under `:SU2` the S=1 irrep), so nothing here is symmetry-specific.

"Retag an operator's two site legs onto site `i`, leaving any op-leg alone."
_mpo_retag(O, i::Int) = to_concrete(setitag(setitag(O, 1, "S,$i"), 2, "S,$i"))

"The ket-facing and bra-facing site legs of `O`, found by ARROW: a plain operator has the
ket leg at 1, an adjoint (`Sp'`) at 2. Hard-coding leg 1 breaks on the adjoint half of
every XY term."
function _op_site_legs(O)
    ket = O.inds[1].dir == '+' ? 1 : 2
    return ket, 3 - ket
end

"""
    _mpo_block(O, i, tl, tr, side) -> TLArray

One entry of the automaton matrix, as a rank-4 `(w_l, s_ket, s_bra, w_r)` tensor.

`side` says which virtual leg the operator's op-leg becomes: `:left` for the left half of
a term (op-leg '-' → `w_r`), `:right` for the right half (op-leg '+' → `w_l`), `:none`
for an operator without one (both virtual legs trivial). A rank-2 operator is `:none`
whatever `side` says, which is how `Sz Sz` and the whole `:none` symmetry mode work.
"""
function _mpo_block(O, i::Int, tl::AbstractString, tr::AbstractString, side::Symbol)
    B = _mpo_retag(O, i)
    ket, bra = _op_site_legs(B)
    if length(B.inds) == 2 || side === :none
        length(B.inds) == 2 || throw(
            ArgumentError(
                "_mpo_block: side=:none needs a rank-2 operator, got rank $(length(B.inds))",
            ),
        )
        B = to_concrete(permutedims(B, (ket, bra)))
        return to_concrete(addSingleton(B, (1, 4); itag = (tl, tr), dir = ('+', '-')))
    elseif side === :left
        B = to_concrete(permutedims(B, (ket, bra, 3)))            # (s_ket, s_bra, op)
        B = to_concrete(addSingleton(B, (1,); itag = tl, dir = '+'))
        return to_concrete(setitag(B, 4, tr))                     # op-leg becomes w_r
    elseif side === :right
        B = to_concrete(permutedims(B, (3, ket, bra)))             # (op, s_ket, s_bra)
        B = to_concrete(addSingleton(B, (4,); itag = tr, dir = '-'))
        return to_concrete(setitag(B, 1, tl))                      # op-leg becomes w_l
    else
        throw(ArgumentError("_mpo_block: side must be :left, :right or :none, got $side"))
    end
end

"""
    onsite_mpo(L, ops; Iloc) -> MPO

`Σ_i ops[i]` for rank-2 (charge-neutral) on-site operators, virtual dimension 2. `Iloc` is the
local identity of the site space the operators act on.
"""
function onsite_mpo(L::Int, ops::AbstractVector; Iloc)
    L >= 2 || throw(ArgumentError("onsite_mpo needs at least two sites, got $L"))
    length(ops) == L ||
        throw(DimensionMismatch("onsite_mpo got $(length(ops)) operators for $L sites"))
    all(length(O.inds) == 2 for O in ops) ||
        throw(ArgumentError("onsite_mpo: every operator must be rank 2 (charge-neutral)"))
    Ws = Any[]
    for i = 1:L
        tl, tr = "W,$(i - 1)", "W,$i"
        Iid = _mpo_block(Iloc, i, tl, tr, :none)
        O = _mpo_block(ops[i], i, tl, tr, :none)
        # rows/columns: 1 = id, 2 = done; the operator is the `id -> done` entry at EVERY site.
        # Filled by index: an `Any[Iid O]` literal concatenates the TLArrays instead.
        mat = Matrix{Any}(nothing, 2, 2)
        mat[1, 1], mat[1, 2], mat[2, 2] = Iid, O, Iid
        i == 1 && (mat = reshape(mat[1, :], 1, 2))
        i == L && (mat = reshape(mat[:, 2], 2, 1))
        push!(Ws, to_concrete(oplus(mat, (1, 4))))
    end
    return MPO(Ws)
end

"""
    mpo_product(mpo, psi) -> SymMPS

`mpo * psi`, exact: each site's MPS and MPO links are fused by a `getIdentity` isometry, so the bond
dimension is the product of the two. The result is NOT canonical (`norm` reads the centre, so use
`overlap`). The ket's site leg contracts with the MPO's leg 2, and leg 3 is the product's site leg.
A charged MPO (a jump changes the U(1) sector) has a charged right boundary leg, which the product
fuses into the state's own, so the result carries the shifted total charge.
"""
function mpo_product(mpo::MPO, psi::SymMPS)
    L = length(psi)
    length(mpo) == L ||
        throw(DimensionMismatch("MPO has $(length(mpo)) sites, state has $L"))
    out = Vector{Any}(undef, L)
    F = nothing
    for i = 1:L
        T = to_concrete!(contract(mpo[i], (2,), psi[i], (2,)))       # (w_l, s, w_r, ℓ_l, ℓ_r)
        T = if i == 1
            # the trimmed `w_l` of site 1 is the dim-1 vacuum; the state's own boundary stays
            to_concrete(permutedims(to_concrete(deleteSingleton(T, 1)), (3, 1, 4, 2)))
        else
            # undo the previous site's fusion on (ℓ_l, w_l): F'F is the identity on the pair
            to_concrete(permutedims(to_concrete!(contract(F', (1, 2), T, (4, 1))), (1, 2, 4, 3)))
        end                                                           # (ℓ_l, s, ℓ_r, w_r)
        # Placeholder tag: the fused link must not share `L,i+1` with the leg it replaces.
        F = to_concrete(getIdentity((T, 3), (T, 4); itag = "Jf,$(i + 1)"))
        out[i] = to_concrete!(contract(T, (3, 4), F, (1, 2)))        # (ℓ_l, s, fused)
    end
    for i = 1:L
        A = to_concrete(setitag(out[i], 3, "L,$(i + 1)"))
        out[i] = i == 1 ? A : to_concrete(setitag(A, 1, "L,$i"))
    end
    return SymMPS(out, L)
end
