# The one- and zero-site effective operators of an MPO in its environments, and the shifted
# operator the root solves use.

"""
    MPOOneSiteH

The single-site effective Hamiltonian  held
as its three factors rather than contracted: the operand is a site tensor, so contracting them
first would build a `(chi d) x (chi d)` matrix where three sequential contractions cost
`O(chi^2 d w)`.

`nothing` on either environment is the chain boundary, where the MPO's own dim-1 leg is
dropped instead of contracted -- see the file header.
"""
struct MPOOneSiteH
    l::Any                    # (bra, mpo, ket) at link j, or `nothing`
    w::Any                    # W^[j]
    r::Any                    # (bra, mpo, ket) at link j+1, or `nothing`
end

"""
    one_site_h(mpo, site, lenv, renv) -> MPOOneSiteH

`H_eff` at `site`. `lenv` must sit on link `site` and `renv` on link `site+1` -- the same
convention the channel-path method of this function uses.
"""
one_site_h(mpo::MPO, site::Int, lenv::MPOLink, renv::MPOLink) =
    MPOOneSiteH(lenv.E, mpo[site], renv.E)

# The same operator with the site's two links read out of prebuilt stacks.
one_site_h(mpo::MPO, site::Int, lenv::MPOLeftEnvStack, renv::MPORightEnvStack) =
    one_site_h(mpo, site, left_channels(lenv, site), right_channels(renv, site + 1))

"""
    apply_one_site(effective_h::MPOOneSiteH, site_tensor) -> TLArray

`H_eff site_tensor` for a site tensor with legs `(link_l, site, link_r)`, in and out.
"""
function apply_one_site(effective_h::MPOOneSiteH, site_tensor)
    # `contract` returns a concrete tensor; `TLArray(...)` materialises the lazy singleton
    # deletion (and is the identity on a concrete tensor, so it never copies).
    contracted = if effective_h.l === nothing
        # (ℓ_l, ℓ_r, w_l, s_bra, w_r)
        partial = contract(site_tensor, (2,), effective_h.w, (2,))
        TLArray(deleteSingleton(partial, 3))            # (ℓ_l, ℓ_r, s_bra, w_r)
    else
        # (bra_l, w, s, ℓ_r), then (bra_l, ℓ_r, s_bra, w_r)
        partial = contract(effective_h.l, (3,), site_tensor, (1,))
        partial = contract(partial, (2, 3), effective_h.w, (1, 2))
        partial
    end                                                    # (link_l, ℓ_r, s_bra, w_r)

    contracted = if effective_h.r === nothing
        partial = TLArray(deleteSingleton(contracted, 4))  # (link_l, ℓ_r, s_bra)
        TLArray(permutedims(partial, (1, 3, 2)))
    else
        contract(contracted, (4, 2), effective_h.r, (2, 3))  # (link_l, s_bra, bra_r)
    end

    # Each environment's bra leg replaced a link leg: give it that link's prime level back.
    effective_h.l === nothing || (contracted = _unprime(contracted, 1))
    effective_h.r === nothing || (contracted = _unprime(contracted, 3))
    return contracted
end

"""
    MPOZeroSiteH

The centre bond's effective Hamiltonian, as the two environments already pushed through
the expanded frames:

    H_eff^{(0)} S = sum_w L[w] S R[w]

One contraction over the MPO leg at a bond, where the site tensor of the
1-site form is absent. Acting with it never touches a site tensor, so a Krylov matvec is
`O(chi^2 w)` rather than the `O(chi^2 d^2)` of a two-site apply.
"""
struct MPOZeroSiteH
    l::Any                    # (bra_bL, mpo, bL)
    r::Any                    # (bra_bR, mpo, bR)
end

"""
    zero_site_h(mpo, bond, lenv, renv, U_ex, V_ex) -> MPOZeroSiteH

Push the environments at links `bond` and `bond+2` onto the centre bond through the EXPANDED
frames. `U_ex` carries `(link_l, site_l, bond)` and `V_ex` carries `(bond, site_r,
link_r)` -- an MPS tensor's layout in each case -- so this is one more step of Eq. (1.8)
with the frame in place of a state tensor, and needs no separate code.
"""
zero_site_h(mpo::MPO, bond::Int, lenv::MPOLink, renv::MPOLink, U_ex, V_ex) = MPOZeroSiteH(
    _mpo_left_step(lenv.E, U_ex, mpo[bond]),
    _mpo_right_step(renv.E, V_ex, mpo[bond+1]),
)

"""
    apply_zero_site(effective_h::MPOZeroSiteH, core) -> TLArray

`H_eff core` for the centre core, legs `(bond_l, bond_r)` in and out. Each environment
contributes its bra leg in place of the ket leg it consumes, so the result carries the core's
own legs after unpriming.
"""
function apply_zero_site(effective_h::MPOZeroSiteH, core)
    contracted = contract(effective_h.l, (3,), core, (1,))  # (bra_l, mpo, bond_r)
    contracted = contract(contracted, (2, 3), effective_h.r, (2, 3))  # (bra_l, bra_r)
    return _unprime(_unprime(contracted, 1), 2)
end

"""
    shift_h(operator, shift) -> operator or ShiftedH

The effective Hamiltonian `H - shift` for the generators of an imaginary-time step. `shift = 0`
returns `operator` itself, so an unshifted step is bit-identical.

A shift by the identity leaves every Krylov space unchanged and only changes the weights read off
them. The CBE sketches are exact as they stand, since `P_perp (H - s) Theta = P_perp H Theta`.
"""
struct ShiftedH{T}
    h::T
    s::Float64
end

shift_h(operator, shift::Real) = iszero(shift) ? operator : ShiftedH(operator, Float64(shift))

# The shifted actions `(H - s) site_tensor` and `(H - s) core`.
apply_one_site(shifted::ShiftedH, site_tensor) =
    TLArray(apply_one_site(shifted.h, site_tensor) - shifted.s * site_tensor)
apply_zero_site(shifted::ShiftedH, core) =
    TLArray(apply_zero_site(shifted.h, core) - shifted.s * core)

# ── zero-site generator, frame construction and truncation sweeps ──
# The Lubich sweep that used to sit here was removed. Its re-ranking growth passes do not carry over
# to the Krylov-accumulated frame of `cbe_bug_step!`, so measurements do not port between the two.

# CBE-BUG: K and L basis sweeps in CBE frames, then ONE Galerkin step at the centre bond (only the
# centre tensor is evolved, so nothing is counted twice). The K/L ODEs are never solved:
# the DD-QR CBE (`dd_cbe_expand`) supplies the directions. The sweeps must not write back into `psi` (zero padding would
# collapse the rank), and no rank-4 tensor is ever allocated.

# ── writing an expansion back into the state ─────────────────────────────────

# the methods below are specifically for BUG 

"""
    _absorb_left!(psi, bond, U_ex) -> psi

Replace `psi[bond]` by the expanded left frame and push the (zero-padded) remainder into
`psi[bond+1]`, leaving the state unchanged and the centre at `bond+1`.

`U_ex†(A_i A_{i+1})` is evaluated as `(U_ex†A_i)A_{i+1}`: the two-site block is never
formed, and the intermediate is `bond_ex x bond_old`.
"""
function _absorb_left!(psi::SymMPS, bond::Int, U_ex)
    tag = Telum.inds(psi[bond])[3].itags
    overlap_matrix = contract(U_ex', (1, 2), psi[bond], (1, 2))  # (bond_ex, bond_old)
    next_tensor = contract(overlap_matrix, (2,), psi[bond+1], (1,))  # (bond_ex, site_r, link_r)
    psi[bond] = TLArray(setitag(U_ex, 3, tag))
    psi[bond+1] = TLArray(setitag(next_tensor, 1, tag))
    psi.center = bond + 1
    return psi
end

"""
    _absorb_right!(psi, bond, V_ex) -> psi

Mirror: `psi[bond+1]` becomes the expanded right frame and the remainder goes into `psi[bond]`,
leaving the centre at `bond`.
"""
function _absorb_right!(psi::SymMPS, bond::Int, V_ex)
    tag = Telum.inds(psi[bond])[3].itags
    overlap_matrix = contract(psi[bond+1], (2, 3), V_ex', (2, 3))  # (bond_old, bond_ex)
    site_tensor = contract(psi[bond], (3,), overlap_matrix, (1,))  # (link_l, site_l, bond_ex)
    psi[bond] = TLArray(setitag(site_tensor, 3, tag))
    psi[bond+1] = TLArray(setitag(V_ex, 1, tag))
    psi.center = bond
    return psi
end

"""Put a frame's two bond legs back on the state's own link tags.

BOTH legs, not one: adjacent tensors in a `SymMPS` must agree on the shared link tag, and
retagging one side only leaves `psi[i].inds[3] != psi[i+1].inds[1]`, which `contract` rejects on
the next sweep.

Lives here beside [`_frame_from`](@ref) because both are shared frame plumbing for every sweep.
It previously sat in `jan_bug.jl`, and deleting that file took `cbe_bug_step!` down with it --
the helper is not specific to any one integrator.
"""
_relink(tensor, left_tag, right_tag) = TLArray(setitag(setitag(tensor, 1, left_tag), 3, right_tag))

"""
    _frame_from(left, right) -> BondFrame

`bond_frame`'s factorisation on tensors handed in rather than read from `psi`. The basis sweeps must
not touch the state, so the tensor is a carry `C_{i-1} psi[i]`, which `bond_frame` cannot express.

Correctness needs exactly one of `left`/`right` to carry the amplitude while the other is an
isometry: the K-sweep (psi canonical at 1) carries it in `left`, the L-sweep (psi canonical at L)
in `right`.
"""
function _frame_from(left, right)
    res_l = svd(left, (1, 2), "bU,L", "bU,R"; cutoff = 0.0)
    res_r = svd(right, (1,), "bV,L", "bV,R"; cutoff = 0.0)
    left_isometry = TLArray(res_l.U)
    right_isometry = TLArray(res_r.Vd)
    # The bond core joins both SVDs' remainders: (S Vd)_left * (U S)_right.
    bond_core = TLArray(TLArray(res_l.S * res_l.Vd) * TLArray(res_r.U * res_r.S))
    return BondFrame(
        left_isometry,
        bond_core,
        right_isometry,
        Telum.inds(left)[1],
        Telum.inds(left)[2],
        Telum.inds(left)[3],
        Telum.inds(right)[2],
        Telum.inds(right)[3],
        leg_dim(left_isometry, 3),
        copy(Telum.spaces(left)[3]),
    )
end

"The BUG's root link (the step's split and compression root is one past it) when none is given."
_bug_root_link(site_count::Int) = max(1, min(site_count - 1, site_count ÷ 2))

# ── the step ─────────────────────────────────────────────────────────────────

"Total stored elements of every tensor in the state -- block-sparse, so the honest size.
(`tensor_elements` is in Core/TensorNetworks/symmetric_mps.jl.)"
_state_stored(psi::SymMPS) = sum(tensor_elements(psi[site]) for site = 1:length(psi); init = 0)
