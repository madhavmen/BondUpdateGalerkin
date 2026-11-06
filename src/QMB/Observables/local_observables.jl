# Site expectation values and the S^z observables of spin-1/2 states.

"""
    site_expval(psi, j, O) -> ComplexF64

`⟨ψ|O_j|ψ⟩` for a rank-2 local operator `O` with legs `(site '+', site '-')`. Canonicalises to `j`
first, so the contraction is local. `O` must span the full local space (`product_state` builds with
`preserve_space=true`). `contract` needs matching itags, so `O` is retagged onto site `j`; equal
tags on its two legs are safe because their arrows differ.
"""
function site_expval(psi::SymMPS, j::Int, O)
    canonical!(psi, j)
    A = psi[j]
    Oj = setitag(O, "S,$j")
    T = contract(Oj, (1,), A, (2,))         # (op-out, L,j, L,j+1)
    return contract(T, (1, 2, 3), A', (2, 1, 3))[]
end

"""
    sz_expectation(psi, j) -> Float64

`⟨S_z^j⟩`, in units where an up spin gives `+1/2`.
"""
sz_expectation(psi::SymMPS, j::Int) = real(site_expval(psi, j, local_space().Sz))

"""
    total_sz(psi) -> Float64

`Σ_j ⟨S_z^j⟩`.
"""
total_sz(psi::SymMPS) = sum(sz_expectation(psi, j) for j = 1:length(psi))

"""
    apply_sz!(psi, j) -> psi

`S^z_j |ψ⟩` in place, NOT renormalised, by sector projection (`S^z = ½P↑ − ½P↓`). Contracting
`local_space().Sz` onto the site leg is wrong and silent: `contract` re-sorts the site leg's charge
sectors, giving a different state with a genuinely orthogonal component. `getsub(...; preserve_space
= true)` keeps the full site space, so the halves add and the leg keeps its sector order.
[`site_expval`](@ref) is unaffected because it closes the loop with the same tensor. Kept here so
the fix lives in one place.
"""
function apply_sz!(psi::SymMPS, j::Int)
    # Under `:SU2` both `getsub` filters silently turn `S^z` into `0.5 * I`, because `SECTOR_UP` and
    # the SU(2) spin-1/2 label are both `((1,),)` and `SECTOR_DOWN` selects nothing. `S^z` is the m
    # = 0 part of a rank-1 operator and leaves the total-spin sector, so there is no SU(2) version.
    # Use `<S.S> = 3 <S^z S^z>` instead (arXiv:2209.00739 Eq. (5) drops that factor of 3).
    symmetry_mode() === :SU2 && throw(
        ArgumentError(
            "apply_sz! is not available under :SU2 -- S^z is the m=0 component of a rank-1 tensor " *
            "operator, not a scalar, so it leaves the total-spin sector an :SU2 MPS is built on. " *
            "Without this guard the sector filters silently return (1/2)*psi. Use :U1 for any S^z " *
            "observable, or implement the vector correlator.",
        ),
    )
    canonical!(psi, j)
    psi[j] = _sz_site(psi[j])
    canonical!(psi, 1)
    return psi
end

"`S^z` on the site leg (leg 2) of one MPS tensor, by sector projection ([`apply_sz!`](@ref))."
function _sz_site(A)
    up = to_concrete(getsub(A, 2, s -> s == SECTOR_UP ? Colon() : nothing; preserve_space = true))
    dn = to_concrete(getsub(A, 2, s -> s == SECTOR_DOWN ? Colon() : nothing; preserve_space = true))
    return to_concrete(to_concrete(0.5 * up) + to_concrete(-0.5 * dn))
end

"""
    sz_overlaps(bra, ket) -> Vector{ComplexF64}

`[<bra| S^z_x |ket> for x in 1:L]` in ONE left and one right transfer pass, i.e. about three
[`overlap`](@ref)s instead of `L`. This is the correlator row `G(x,t) ~ <Om| S^z_x |phi(t)>`; the
route through `L` stored states `S^z_x|Om>` costs `L` overlaps per time and `L` copies of the
ground state (several GB at chi >= 512 on 40 sites). `S^z_x` acts on one site leg in any gauge,
so it is applied to the bra tensor alone ([`apply_sz!`](@ref)'s projection, Hermitian). States in
different total charge sectors give zeros, as `overlap` does.
"""
function sz_overlaps(bra::SymMPS, ket::SymMPS)
    length(bra) == length(ket) || throw(
        DimensionMismatch(
            "sz_overlaps needs equal-length states, got $(length(bra)) and $(length(ket))",
        ),
    )
    symmetry_mode() === :SU2 && throw(ArgumentError("sz_overlaps needs :U1, see apply_sz!"))
    L = length(bra)
    L >= 2 || return [ComplexF64(tensor_inner(_sz_site(bra[1]), ket[1]))]
    # bra links retagged into "BL,i" exactly as `overlap` does, so the transfer legs are distinct
    braR = Vector{Any}(undef, L)
    for i = 1:L
        t = bra[i]
        i > 1 && (t = setitag(t, 1, "BL,$i"))
        i < L && (t = setitag(t, 3, "BL,$(i + 1)"))
        braR[i] = to_concrete(t)
    end
    braq = Set(first(sp) for sp in braR[L].spaces[3])
    ketq = Set(first(sp) for sp in ket[L].spaces[3])
    isdisjoint(braq, ketq) && return zeros(ComplexF64, L)
    scalar(s) = (r = to_concrete(s); length(r) == 0 ? 0.0 + 0.0im : ComplexF64(r[]))
    # left: E[i] = sites 1..i, legs (bra_link_r, ket_link_r)
    lstep(E, b, i) = contract(contract(E, (1,), b', (1,)), (1, 2), ket[i], (1, 2))
    E = Vector{Any}(undef, L - 1)
    E[1] = contract(braR[1]', (1, 2), ket[1], (1, 2))
    for i = 2:(L-1)
        E[i] = lstep(E[i-1], braR[i], i)
    end
    # right: R[i] = sites i..L, legs (bra_link_l, ket_link_l)
    R = Vector{Any}(undef, L)
    R[L] = contract(braR[L]', (2, 3), ket[L], (2, 3))
    for i = (L-1):-1:2
        R[i] = contract(braR[i]', (2, 3), contract(ket[i], (3,), R[i+1], (2,)), (2, 3))
    end
    G = zeros(ComplexF64, L)
    for x = 1:L
        b = _sz_site(braR[x])
        if x == L
            G[x] = scalar(contract(contract(E[L-1], (1,), b', (1,)), (1, 2, 3), ket[L], (1, 2, 3)))
        else
            Ex = x == 1 ? contract(b', (1, 2), ket[1], (1, 2)) : lstep(E[x-1], b, x)
            G[x] = scalar(contract(Ex, (1, 2), R[x+1], (1, 2)))
        end
    end
    return G
end
