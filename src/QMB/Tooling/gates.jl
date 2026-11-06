# Two-site Heisenberg-type gates and the dimer state built from them.
#
# `apply_gate` and `_site_tag` are in BUG/Core/TensorNetworks/mpo.jl.

"Retag both physical legs of a local operator, leaving any operator leg alone."
_retag_site(O, tag) = TLArray(setitag(setitag(O, 1, tag), 2, tag))

"""
    heisenberg_bond_gate(site_l, site_r; J=1.0, delta=1.0) -> TLArray

The nearest-neighbour term `J (Sx Sx + Sy Sy + delta Sz Sz)` on one bond, tagged for the two site
legs (`TLIndex` handles or bare itags). `delta = 0` is the XX model, `delta = 1` isotropic
Heisenberg.

Built as `Sp(x)Sp' + Sm(x)Sm' + delta Sz(x)Sz`: the two hops conserve charge in both directions, and
the op-leg charge (+/-2) forces each to pair with its own adjoint.
"""
function heisenberg_bond_gate(site_l, site_r; J::Float64 = 1.0, delta::Float64 = 1.0)
    tl, tr = _site_tag(site_l), _site_tag(site_r)
    tl == tr && throw(ArgumentError("the two sites of a bond need distinct itags, both are '$tl'"))
    q = local_space()

    # (ket_l, bra_l, ket_r, bra_r) -> (ket_l, ket_r, bra_l, bra_r).
    pair(A, B) = TLArray(
        permutedims(contract(_retag_site(A, tl), (3,), _retag_site(B, tr)', (3,)), (1, 4, 2, 3)),
    )

    # ── :SU2 ──
    # One term, not three: under SU(2) only `:S` (a rank-3 op-leg, S = 1 irrep) is exposed, and its
    # op-leg contraction sums the vector index, so one `pair` is the full `S_i . S_j`. `delta != 1`
    # is refused, not built isotropically: anisotropy breaks SU(2) to U(1), so the gate would be the
    # wrong Hamiltonian.
    if symmetry_mode() === :SU2
        delta == 1.0 || throw(
            ArgumentError(
                "SU(2) admits only the isotropic point, got delta = $delta; anisotropy breaks " *
                "the symmetry down to U(1), so run that case under set_symmetry!(:U1)",
            ),
        )
        return TLArray(J * pair(q.S, q.S) * (1.0 + 0.0im))
    end
    outer(A, B) = TLArray(
        permutedims(contract(_retag_site(A, tl), (), _retag_site(B, tr), ()), (1, 3, 2, 4)),
    )

    # Under :U1 the raising/lowering ops are rank 3 and `pair` already folds in the 1/2 (see the
    # header). Under :none they are plain matrices: Telum's Sp = -(1/√2) S⁺ and Sm = (1/√2) S⁻ make
    # the XY term an outer product with coefficient -1. The gate test pins this element for element
    # against :U1.
    xy = if length(Telum.inds(q.Sp)) == 3
        pair(q.Sp, q.Sp) + pair(q.Sm, q.Sm)
    else
        TLArray(-1.0 * (outer(q.Sp, q.Sm) + outer(q.Sm, q.Sp)))
    end
    G = delta == 0.0 ? xy : xy + delta * outer(q.Sz, q.Sz)
    return TLArray(J * G * (1.0 + 0.0im))
end

"""
    xx_bond_gate(site_l, site_r; J=1.0) -> TLArray

`J (Sx Sx + Sy Sy)`, the free-fermion XX term. `heisenberg_bond_gate` with
`delta = 0`.
"""
xx_bond_gate(site_l, site_r; J::Float64 = 1.0) =
    heisenberg_bond_gate(site_l, site_r; J = J, delta = 0.0)

"""
    magnetisation_gate(site_l, site_r) -> TLArray

`Sz (x) I + I (x) Sz`, the two-site magnetisation, in the same leg convention as
[`heisenberg_bond_gate`](@ref). Not part of the dynamics -- it is the observable
that pins U(1) charge conservation across a bond update.
"""
function magnetisation_gate(site_l, site_r)
    tl, tr = _site_tag(site_l), _site_tag(site_r)
    q = local_space()
    outer(A, B) = TLArray(
        permutedims(contract(_retag_site(A, tl), (), _retag_site(B, tr), ()), (1, 3, 2, 4)),
    )
    return TLArray((outer(q.Sz, q.I) + outer(q.I, q.Sz)) * (1.0 + 0.0im))
end

"""
    dimer_state(L) -> SymMPS

The product of nearest-neighbour singlets `(1,2)(3,4)...` under `:SU2`; `L` must be even. It is the
representable S = 0 starting point that replaces `product_state` (a Neel state has no SU(2) form).

Built by the fusion recursion of `_product_state_u1`, but the selection acts on the outgoing link,
since the physical leg has a single sector under SU(2). Odd sites keep spin 1/2 and even sites keep
spin 0, so after each even site the running total is a singlet: a chain of independent dimers.
Every link carries one multiplet, so `bond_dims` reads all ones, yet the state is entangled.
"""
function dimer_state(L::Int)
    # Available in EVERY mode, unlike `product_state` (which `:SU2` cannot represent) and unlike
    # the old `:SU2`-only version of this function. That matters because an SU(2)-vs-no-symmetry
    # comparison is only meaningful if both sides evolve the SAME state: the claim under test is
    # that symmetry changes the cost and not the physics. Under `:none`/`:U1` the construction is
    # by projection — see `_dimer_state_projected`.
    symmetry_mode() === :SU2 || return _dimer_state_projected(L)
    L >= 2 && iseven(L) ||
        throw(ArgumentError("dimer_state needs an even number of sites >= 2, got $L"))
    q = local_space(:SU2)

    tensors = Any[]
    prev = (getvac(q.I, ("L,0", "L,1")), 2)

    for i = 1:L
        F = TLArray(getIdentity(prev, (q.I, 1); itag = "L,$(i + 1)"))
        F = TLArray(setitag(F, 2, "S,$i"))
        # Select on leg 3, the FUSED total spin, not on the physical leg.
        target = isodd(i) ? SECTOR_SU2_HALF : SECTOR_SU2_SINGLET
        A = TLArray(getsub(F, 3, s -> s == target ? Colon() : nothing))
        A = TLArray(A * (1.0 + 0.0im))
        push!(tensors, A)
        prev = (A, 3)
    end

    return SymMPS(tensors, L)
end
