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
    length(mpo) == L || throw(DimensionMismatch("MPO has $(length(mpo)) sites, state has $L"))
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

# ── from BUG-Julia@b155550:src/RSVDCBEBondUpdate/operators/mpo.jl ──

"`side` for one half of a term: rank-3 operators put their op-leg on the virtual leg."
_side(O, s::Symbol) = length(O.inds) == 3 ? s : :none

"""
    mpo_from_terms(h::XXZChain; Iloc) -> MPO

The MPO of the same Hamiltonian `h` represents as a term list, so the two paths cannot
disagree about `J`, `delta`, a coefficient or Telum's operator normalisation: the blocks
ARE `h.terms`. `Iloc` is the local identity of the site space, as for [`onsite_mpo`](@ref)
(the model that built `h` supplies it).

The coefficient is applied once, on the closing (right) half of each term -- exactly where
`push_left_channels` applies it.
"""
function mpo_from_terms(h::XXZChain; Iloc)
    L, terms = length(h), h.terms
    n = length(terms)

    Ws = Any[]
    for i = 1:L
        tl, tr = "W,$(i - 1)", "W,$i"
        Iid = _mpo_block(Iloc, i, tl, tr, :none)
        # A zero block with the SAME spaces as `Iid`; `empty_tlarray` will not do, it
        # returns EMPTY space lists and the matrix oplus then cannot infer the column.
        Znil = to_concrete(0.0 * Iid)
        mat = Matrix{Any}(nothing, n + 2, n + 2)
        i < L && (mat[1, 1] = Iid)                       # transport the identity
        i > 1 && (mat[n+2, n+2] = Iid)               # ... on the far side of a term
        for t = 1:n
            i < L &&
                (mat[1, 1+t] = _mpo_block(terms[t].left, i, tl, tr, _side(terms[t].left, :left)))
            i > 1 && (
                mat[1+t, n+2] = to_concrete(
                    terms[t].coeff *
                    _mpo_block(terms[t].right, i, tl, tr, _side(terms[t].right, :right)),
                )
            )
        end
        if i == 1
            mat = reshape(mat[1, :], 1, n + 2)           # the `id` row only
            mat[1, n+2] = Znil                         # ... and it cannot be `done` yet
        elseif i == L
            mat = reshape(mat[:, n+2], n + 2, 1)       # the `done` column only
            mat[1, 1] = Znil                             # ... and nothing may stay `id`
        end
        push!(Ws, to_concrete(oplus(mat, (1, 4))))
    end
    return MPO(Ws)
end

"""
    mpo_sum(A, B; a = 1, b = 1) -> MPO

`a A + b B` as one MPO: block-diagonal virtual legs, the end sites concatenated. Both must act on
the same site legs; the scalars sit on site 1. Every tensor is complex, as `b` may be.
"""
function mpo_sum(A::MPO, B::MPO; a::Number = 1.0, b::Number = 1.0)
    L = length(A)
    length(B) == L || throw(DimensionMismatch("mpo_sum: $L sites against $(length(B))"))
    L >= 2 || throw(ArgumentError("mpo_sum needs at least two sites, got $L"))
    Ws = Any[]
    for i = 1:L
        sa, sb = i == 1 ? (a, b) : (1, 1)
        Ai = to_concrete(ComplexF64(sa) * A[i])
        Bi = to_concrete(ComplexF64(sb) * B[i])
        W = if i == 1 || i == L
            oplus(Any[Ai, Bi], (i == 1 ? 4 : 1,))
        else
            mat = Matrix{Any}(nothing, 2, 2)
            mat[1, 1], mat[2, 2] = Ai, Bi
            oplus(mat, (1, 4))
        end
        push!(Ws, to_concrete(W))
    end
    return MPO(Ws)
end

"""
    no_jump_mpo(H, Q) -> MPO

`H_eff = H - (i/2) Q`, so `tau = -im*dt` integrates the no-jump equation `dψ/dt = Gψ`,
`G = -iH - Q/2`, with `Q = Σ_μ L_μ' L_μ` (the caller's MPO, positive semidefinite). Not Hermitian:
pass `hermitian = false` to `cbe_bug_step!` / `tdvp2_step!`, since Lanczos on it returns a wrong
vector without an error.
"""
no_jump_mpo(H::MPO, Q::MPO) = mpo_sum(H, Q; b = -0.5im)

# ── from BUG-Julia@b155550:src/RSVDCBEBondUpdate/operators/mpo_apply.jl ──

"`F psi`, exact and not canonical: an `MPO` ([`mpo_product`](@ref)). The `SiteOp` method moves
with the jump channels (QMB, PR 6)."
factor_product(F::MPO, psi::SymMPS) = mpo_product(F, psi)

"""
    apply_factor(F, psi; tol_trunc = 0.0, maxdim = 10_000, root = L ÷ 2 + 1) -> (SymMPS, discarded)

[`factor_product`](@ref), gauged (a right-to-left sweep from site `L` makes every site an isometry,
whatever the product's gauge) and compressed once by `truncate_recursive!` at relative discarded
norm `tol_trunc` and rank `maxdim` (the step's own compression rule), centre `root`. Returns the
state and its relative discarded norm. A product that is exactly zero (a jump that annihilates every
component) is refused: its probability is zero, so it can never be drawn.
"""
function apply_factor(
    F,
    psi::SymMPS;
    tol_trunc::Float64 = 0.0,
    maxdim::Int = 10_000,
    root::Int = length(psi) ÷ 2 + 1,
)
    phi = factor_product(F, psi)
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
    global_krylov_states(mpo, psi, m; tol_trunc = 1e-13, maxdim = 10_000) -> Vector{SymMPS}

`mpo^j psi` for `j = 1..m`, each by [`apply_mpo`](@ref) from the previous one and normalised: the
global Krylov vectors, to pass as `cbe_bug_step!(...; keep_states)`. From a U(1) product state an
order-`p` step's closure cannot see the long-range terms of a nonlocal generator (dissipative
plan U10: both legs of `S⁺_a S⁻_b` leave their frames' sectors, and every two-site window projects
that away), so `mpo^p psi` is missing from the root space unless it is kept (dt5m: one-step slope
2.7 -> 3.7, global error /2.7-18; keeping them on later steps changes nothing).
"""
function global_krylov_states(
    mpo::MPO,
    psi::SymMPS,
    m::Int;
    tol_trunc::Float64 = 1e-13,
    maxdim::Int = 10_000,
)
    ks, cur = SymMPS[], psi
    for _ = 1:m
        nxt, _ = apply_mpo(mpo, cur; tol_trunc = tol_trunc, maxdim = maxdim)
        canonical!(nxt, 1)
        nxt[1] = to_concrete(nxt[1] / norm(nxt[1]))
        push!(ks, nxt)
        cur = nxt
    end
    return ks
end

# ── from BUG-Julia@b155550:src/BondUpdateBUG/gates.jl ──

# Two-site bond gates, mirroring bond.py::kernel_gate. Legs are (ket_l '+', ket_r '+', bra_l '-',
# bra_r '-'), as in Python's einsum, so the result drops back into the site legs. Telum separates
# legs by arrow alone. Telum's Sp is normalised so that Sp (x) Sp' = (1/2) S^- (x) S^+: the XY term
# is a plain sum, with no 0.5 factor.

# Itag of a site leg given as a `TLIndex`, an `Itag`, or a plain string.
# A plain string has to be normalised rather than compared directly:
# `Itag` is a SORTED TAG SET, so the leg written as "S,1" reads back as "1,S"
# and a raw `Itag == String` comparison would reject the very tag that built it.
# `Itag` applies the normalisation `setitag` uses (the legacy code tagged a local identity, which
# lives in QMB).
_site_tag(x::TLIndex) = x.itags
_site_tag(x::AbstractString) = Itag(x)
_site_tag(x) = x

"""
    apply_gate(gate, theta, site_l, site_r) -> TLArray

Act `gate` on the two-site block `theta`, legs `(link_l, site_l, site_r, link_r)`
in and out. Python's `_apply_gate_named`; the retag Python needs afterwards is
unnecessary here because the gate's bra legs already carry the site tags.

`site_l` / `site_r` are checked against `theta`, not used to build anything: a
gate tagged for the wrong bond otherwise contracts happily against the wrong legs
of a symmetric tensor and returns a plausible, silently wrong answer.
"""
function apply_gate(gate, theta, site_l, site_r)
    tl, tr = _site_tag(site_l), _site_tag(site_r)
    (gate.inds[1].itags == tl && gate.inds[2].itags == tr) || throw(
        ArgumentError(
            "gate is tagged for ($(gate.inds[1].itags), $(gate.inds[2].itags)), " *
            "asked to act on ($tl, $tr)",
        ),
    )
    (theta.inds[2].itags == tl && theta.inds[3].itags == tr) || throw(
        ArgumentError(
            "gate is tagged for ($tl, $tr) but theta carries " *
            "($(theta.inds[2].itags), $(theta.inds[3].itags))",
        ),
    )
    out = contract(gate, (1, 2), theta, (2, 3))   # (bra_l, bra_r, link_l, link_r)
    return to_concrete(permutedims(out, (3, 1, 2, 4)))
end
