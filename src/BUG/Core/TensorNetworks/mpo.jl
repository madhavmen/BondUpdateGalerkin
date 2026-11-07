# MPO storage: the operator as one rank-4 tensor per site with fused virtual legs; on-site sums,
# term-list MPOs and MPO sums; the exact MPO-MPS product, its compressed form and the global Krylov
# states. The nearest-neighbour term list `XXZChain` is defined here as the input of `mpo_from_terms`.
#
# `onsite_mpo` takes the local identity `Iloc` as a required keyword, since local spaces belong to
# the models, which BUG may not reference.
# The leg and prime conventions are load-bearing: a wrong prime does not throw, it traces the wrong
# pair.

"""
    MPO(W)

An operator as a matrix product operator, one rank-4 tensor per site (field `W`) with legs
`(w_l, s_ket, s_bra, w_r)`. The boundary virtual legs of `W[1]` and `W[end]` are dim 1, from
trimming, so no separate boundary vector is carried.

The ket's site leg contracts with leg 2 and leg 3 is the output, so the tensor stores the
TRANSPOSE of the applied matrix on the site legs.
"""
struct MPO
    W::Vector{Any}
end

# Vector-like access to the site tensors.
Base.length(mpo::MPO) = length(mpo.W)
Base.getindex(mpo::MPO, site::Int) = mpo.W[site]

"Virtual (MPO bond) dimension on each link `0 … length(mpo)`."
mpo_virtual_dims(mpo::MPO) = vcat(
    [sum(block_dim for (_, block_dim) in Telum.spaces(mpo[1])[1]; init = 0)],
    [
        sum(block_dim for (_, block_dim) in Telum.spaces(mpo[site])[4]; init = 0) for
        site = 1:length(mpo)
    ],
)

# Each automaton row is written out literally and `oplus` direct-sums them, inferring the zero
# blocks from the row and column spaces. The op-leg carries the charge (under `:U1` the `Sp` leg is
# its channel's sector; under `:SU2` the S=1 irrep), so nothing here is symmetry-specific.

"Retag an operator's two site legs onto site `site`, leaving any op-leg alone."
_mpo_retag(operator, site::Int) = TLArray(setitag(setitag(operator, 1, "S,$site"), 2, "S,$site"))

"The ket-facing and bra-facing site legs of `operator`, found by ARROW: a plain operator has
the ket leg at 1, an adjoint (`Sp'`) at 2. Hard-coding leg 1 breaks on the adjoint half of
every XY term."
function _op_site_legs(operator)
    ket = Telum.inds(operator)[1].dir == '+' ? 1 : 2
    return ket, 3 - ket
end

"""
    _mpo_block(operator, site, tl, tr, side) -> TLArray

One entry of the automaton matrix, as a rank-4 `(w_l, s_ket, s_bra, w_r)` tensor: `operator`
on site `site`, its virtual legs tagged `tl` and `tr`.

`side` says which virtual leg the operator's op-leg becomes: `:left` for the left half of
a term (op-leg '-' → `w_r`), `:right` for the right half (op-leg '+' → `w_l`), `:none`
for an operator without one (both virtual legs trivial). A rank-2 operator is `:none`
whatever `side` says, which is how `Sz Sz` and the whole `:none` symmetry mode work.
"""
function _mpo_block(operator, site::Int, tl::AbstractString, tr::AbstractString, side::Symbol)
    block = _mpo_retag(operator, site)
    ket, bra = _op_site_legs(block)
    if length(Telum.inds(block)) == 2 || side === :none
        length(Telum.inds(block)) == 2 || throw(
            ArgumentError(
                "_mpo_block: side=:none needs a rank-2 operator, got rank $(length(block.inds))",
            ),
        )
        # No op-leg: both virtual legs are dim-1 singletons.
        block = TLArray(permutedims(block, (ket, bra)))
        return TLArray(addSingleton(block, (1, 4); itag = (tl, tr), dir = ('+', '-')))
    elseif side === :left
        block = TLArray(permutedims(block, (ket, bra, 3)))       # (s_ket, s_bra, op)
        block = TLArray(addSingleton(block, (1,); itag = tl, dir = '+'))
        return TLArray(setitag(block, 4, tr))                    # op-leg becomes w_r
    elseif side === :right
        block = TLArray(permutedims(block, (3, ket, bra)))       # (op, s_ket, s_bra)
        block = TLArray(addSingleton(block, (4,); itag = tr, dir = '-'))
        return TLArray(setitag(block, 1, tl))                    # op-leg becomes w_l
    else
        throw(ArgumentError("_mpo_block: side must be :left, :right or :none, got $side"))
    end
end

"""
    onsite_mpo(site_count, ops; Iloc) -> MPO

`Σ_i ops[i]` for rank-2 (charge-neutral) on-site operators on `site_count` sites, virtual
dimension 2. `Iloc` is the local identity of the site space the operators act on.
"""
function onsite_mpo(site_count::Int, ops::AbstractVector; Iloc)
    site_count >= 2 || throw(ArgumentError("onsite_mpo needs at least two sites, got $site_count"))
    length(ops) == site_count ||
        throw(DimensionMismatch("onsite_mpo got $(length(ops)) operators for $site_count sites"))
    all(length(Telum.inds(operator)) == 2 for operator in ops) ||
        throw(ArgumentError("onsite_mpo: every operator must be rank 2 (charge-neutral)"))
    Ws = Any[]
    for site = 1:site_count
        tl, tr = "W,$(site - 1)", "W,$site"
        Iid = _mpo_block(Iloc, site, tl, tr, :none)
        operator_block = _mpo_block(ops[site], site, tl, tr, :none)
        # rows/columns: 1 = id, 2 = done; the operator is the `id -> done` entry at EVERY site.
        # Filled by index: an `Any[Iid operator_block]` literal concatenates the TLArrays instead.
        mat = Matrix{Any}(nothing, 2, 2)
        mat[1, 1], mat[1, 2], mat[2, 2] = Iid, operator_block, Iid
        site == 1 && (mat = reshape(mat[1, :], 1, 2))
        site == site_count && (mat = reshape(mat[:, 2], 2, 1))
        push!(Ws, TLArray(oplus(mat, (1, 4))))
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
    site_count = length(psi)
    length(mpo) == site_count ||
        throw(DimensionMismatch("MPO has $(length(mpo)) sites, state has $site_count"))
    out = Vector{Any}(undef, site_count)
    fusion = nothing
    for site = 1:site_count
        # (w_l, s, w_r, ℓ_l, ℓ_r)
        product_tensor = contract(mpo[site], (2,), psi[site], (2,))
        product_tensor = if site == 1
            # the trimmed `w_l` of site 1 is the dim-1 vacuum; the state's own boundary stays
            TLArray(permutedims(TLArray(deleteSingleton(product_tensor, 1)), (3, 1, 4, 2)))
        else
            # undo the previous site's fusion on (ℓ_l, w_l): F'F is the identity on the pair,
            # with F = fusion
            TLArray(permutedims(contract(fusion', (1, 2), product_tensor, (4, 1)), (1, 2, 4, 3)))
        end                                                           # (ℓ_l, s, ℓ_r, w_r)
        # Placeholder tag: the fused link must not share `L,i+1` with the leg it replaces.
        fusion =
            TLArray(getIdentity((product_tensor, 3), (product_tensor, 4); itag = "Jf,$(site + 1)"))
        # (ℓ_l, s, fused)
        out[site] = contract(product_tensor, (3, 4), fusion, (1, 2))
    end
    # Give the fused links their final "L,i" tags (site 1 keeps its left boundary tag).
    for site = 1:site_count
        site_tensor = TLArray(setitag(out[site], 3, "L,$(site + 1)"))
        out[site] = site == 1 ? site_tensor : TLArray(setitag(site_tensor, 1, "L,$site"))
    end
    return SymMPS(out, site_count)
end

# ── the Hamiltonian as a nearest-neighbour term list ─────────────────────────

"""
    XXZTerm(left, right, coeff)

One nearest-neighbour term `coeff · left_i ⊗ right_{i+1}`, repeated on every bond: one vertex of
the automaton [`mpo_from_terms`](@ref) builds.

`left` and `right` are bare local operators, with no site tags. Either may be rank-3, carrying an
`op` leg whose charge is what makes the pair U(1)-allowed; when both are, that leg becomes the
MPO's virtual leg (this is the XY hop, and it is why the two operators must be an `Sp`/`Sp'`
adjoint pair rather than `Sp`/`Sm`).
"""
struct XXZTerm
    left::Any
    right::Any
    coeff::Float64
end

"""
    XXZChain(L, terms, J, delta)

`H = J Σ_i (Sx Sx + Sy Sy + delta Sz Sz)` on `L` sites, as a term list. It is not an operator the
sweeps apply: it is the input of [`mpo_from_terms`](@ref) and of QMB's pair and long-range MPO
builders. Built by `xxz_chain`; `delta = 0` is XX.
"""
struct XXZChain
    L::Int
    terms::Vector{XXZTerm}
    J::Float64
    delta::Float64
end

# The number of sites, so a chain can be checked against the state's length.
Base.length(chain::XXZChain) = chain.L

"The term list of `chain`, in the order `mpo_from_terms` gives the terms their automaton states."
hamiltonian_terms(chain::XXZChain) = chain.terms

"`side` for one half of a term: rank-3 operators put their op-leg on the virtual leg."
_side(operator, half::Symbol) = length(Telum.inds(operator)) == 3 ? half : :none

"""
    mpo_from_terms(hamiltonian::XXZChain; Iloc) -> MPO

The MPO of the nearest-neighbour term list `hamiltonian`. The blocks ARE `hamiltonian.terms`, so
the MPO cannot disagree with the model about `J`, `delta`, a coefficient or Telum's operator
normalisation. `Iloc` is the local identity of the site space, as for [`onsite_mpo`](@ref) (the
model that built `hamiltonian` supplies it).

The coefficient is applied once, on the closing (right) half of each term.
"""
function mpo_from_terms(hamiltonian::XXZChain; Iloc)
    site_count, terms = length(hamiltonian), hamiltonian.terms
    term_count = length(terms)

    # Automaton states: 1 = id, 1 + term_index = inside that term, term_count + 2 = done.
    Ws = Any[]
    for site = 1:site_count
        tl, tr = "W,$(site - 1)", "W,$site"
        Iid = _mpo_block(Iloc, site, tl, tr, :none)
        # A zero block with the SAME spaces as `Iid`; `empty_tlarray` will not do, it
        # returns EMPTY space lists and the matrix oplus then cannot infer the column.
        Znil = TLArray(0.0 * Iid)
        mat = Matrix{Any}(nothing, term_count + 2, term_count + 2)
        site < site_count && (mat[1, 1] = Iid)              # transport the identity
        site > 1 && (mat[term_count+2, term_count+2] = Iid) # ... on the far side of a term
        for term_index = 1:term_count
            # Open the term with its left half, close it with its (weighted) right half.
            site < site_count && (
                mat[1, 1+term_index] = _mpo_block(
                    terms[term_index].left,
                    site,
                    tl,
                    tr,
                    _side(terms[term_index].left, :left),
                )
            )
            site > 1 && (
                mat[1+term_index, term_count+2] = TLArray(
                    terms[term_index].coeff * _mpo_block(
                        terms[term_index].right,
                        site,
                        tl,
                        tr,
                        _side(terms[term_index].right, :right),
                    ),
                )
            )
        end
        if site == 1
            mat = reshape(mat[1, :], 1, term_count + 2)             # the `id` row only
            mat[1, term_count+2] = Znil                             # ... and it cannot be `done`
        elseif site == site_count
            mat = reshape(mat[:, term_count+2], term_count + 2, 1)  # the `done` column only
            mat[1, 1] = Znil                                        # ... and nothing may stay `id`
        end
        push!(Ws, TLArray(oplus(mat, (1, 4))))
    end
    return MPO(Ws)
end

"""
    mpo_sum(first_mpo, second_mpo; a = 1, b = 1) -> MPO

`a * first_mpo + b * second_mpo` as one MPO: block-diagonal virtual legs, the end sites
concatenated. Both must act on the same site legs; the scalars sit on site 1. Every tensor is
complex, as `b` may be.
"""
function mpo_sum(first_mpo::MPO, second_mpo::MPO; a::Number = 1.0, b::Number = 1.0)
    site_count = length(first_mpo)
    length(second_mpo) == site_count ||
        throw(DimensionMismatch("mpo_sum: $site_count sites against $(length(second_mpo))"))
    site_count >= 2 || throw(ArgumentError("mpo_sum needs at least two sites, got $site_count"))
    Ws = Any[]
    for site = 1:site_count
        sa, sb = site == 1 ? (a, b) : (1, 1)
        Ai = TLArray(ComplexF64(sa) * first_mpo[site])
        Bi = TLArray(ComplexF64(sb) * second_mpo[site])
        # End sites: concatenate along the one non-trivial virtual leg; interior sites: block
        # diagonal on both virtual legs.
        summed = if site == 1 || site == site_count
            oplus(Any[Ai, Bi], (site == 1 ? 4 : 1,))
        else
            mat = Matrix{Any}(nothing, 2, 2)
            mat[1, 1], mat[2, 2] = Ai, Bi
            oplus(mat, (1, 4))
        end
        push!(Ws, TLArray(summed))
    end
    return MPO(Ws)
end

"""
    no_jump_mpo(hamiltonian, loss_operator) -> MPO

`H_eff = H - (i/2) Q` with `H = hamiltonian` and `Q = loss_operator`, so `tau = -im*dt`
integrates the no-jump equation `dψ/dt = Gψ`, `G = -iH - Q/2`, with `Q = Σ_μ L_μ' L_μ` (the
caller's MPO, positive semidefinite). Not Hermitian: pass `hermitian = false` to `cbe_bug_step!` /
`tdvp2_step!`, since Lanczos on it returns a wrong vector without an error.
"""
no_jump_mpo(hamiltonian::MPO, loss_operator::MPO) = mpo_sum(hamiltonian, loss_operator; b = -0.5im)

"`factor psi`, exact and not canonical: an `MPO` ([`mpo_product`](@ref)). The `SiteOp` method
is with the jump channels (QMB)."
factor_product(factor::MPO, psi::SymMPS) = mpo_product(factor, psi)

"""
    apply_factor(factor, psi; tol_trunc = 0.0, maxdim = 10_000, root = length(psi) ÷ 2 + 1)
        -> (SymMPS, discarded)

[`factor_product`](@ref), gauged (a right-to-left sweep from the last site makes every site an
isometry, whatever the product's gauge) and compressed once by `truncate_recursive!` at relative
discarded norm `tol_trunc` and rank `maxdim` (the step's own compression rule), centre `root`.
Returns the state and its relative discarded norm. A product that is exactly zero (a jump that
annihilates every component) is refused: its probability is zero, so it can never be drawn.
"""
function apply_factor(
    factor,
    psi::SymMPS;
    tol_trunc::Float64 = 0.0,
    maxdim::Int = 10_000,
    root::Int = length(psi) ÷ 2 + 1,
)
    phi = factor_product(factor, psi)
    real(overlap(phi, phi)) > 0 || throw(
        ArgumentError(
            "apply_factor: the product is exactly zero (the operator annihilates the state)",
        ),
    )
    canonical!(phi, 1)
    canonical!(phi, root)
    _, _, cum, _ = truncate_recursive!(phi, root; maxdim = maxdim, weight_tol = tol_trunc)
    return phi, cum
end

"`apply_factor` for an MPO: `mpo * psi`, gauged and compressed once."
apply_mpo(mpo::MPO, psi::SymMPS; kw...) = apply_factor(mpo, psi; kw...)

"""
    global_krylov_states(mpo, psi, krylov_count; tol_trunc = 1e-13, maxdim = 10_000)
        -> Vector{SymMPS}

`mpo^j psi` for `j = 1..krylov_count`, each by [`apply_mpo`](@ref) from the previous one and
normalised: the global Krylov vectors, to pass as `cbe_bug_step!(...; keep_states)`. From a U(1)
product state an order-`p` step's closure cannot see the long-range terms of a nonlocal generator
(both legs of `S⁺_a S⁻_b` leave their frames' sectors, and every two-site
window projects that away), so `mpo^p psi` is missing from the root space unless it is kept (dt5m:
one-step slope 2.7 -> 3.7, global error /2.7-18; keeping them on later steps changes nothing).
"""
function global_krylov_states(
    mpo::MPO,
    psi::SymMPS,
    krylov_count::Int;
    tol_trunc::Float64 = 1e-13,
    maxdim::Int = 10_000,
)
    ks, cur = SymMPS[], psi
    for _ = 1:krylov_count
        nxt, _ = apply_mpo(mpo, cur; tol_trunc = tol_trunc, maxdim = maxdim)
        canonical!(nxt, 1)
        nxt[1] = TLArray(nxt[1] / norm(nxt[1]))
        push!(ks, nxt)
        cur = nxt
    end
    return ks
end
