# The one- and zero-site effective operators of an MPO in its environments, and the shifted
# operator the root solves use.

"""
    MPOOneSiteH

The single-site effective Hamiltonian of Eq. (1.7), `H_eff^[j] = L^[j-1] W^[j] R^[j+1]`, held
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

One contraction over the MPO leg -- Eq. (1.7) at a bond, where the site tensor of the
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
# `cbe_expand` supplies the directions. The sweeps must not write back into `psi` (zero padding would
# collapse the rank), and no rank-4 tensor is ever allocated.

# ── writing an expansion back into the state ─────────────────────────────────

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
_relink(tensor, left_tag, right_tag) =
    TLArray(setitag(setitag(tensor, 1, left_tag), 3, right_tag))

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

# ── the 0-site effective Hamiltonian ─────────────────────────────────────────

"""
    ZeroSiteH

The centre bond's effective Hamiltonian, as channel environments already pushed through
the expanded frames. Acting with it never touches a site tensor:

    S_out = done_L·S + S·done_Rᵀ + Σ_t c_t Σ_op open_L[t]·S·open_R[t]

Each term is a `bond x bond` product, so a Krylov matvec costs `O(χ²)` rather than the
`O(χ²d²)` of a two-site apply. This is what `RSVDpreBE0SiQS` returning `VLex, VRex` is
for: the whole point of expanding the bond is that the update afterwards is 0-site.
"""
struct ZeroSiteH
    done_l::Any
    done_r::Any
    open_l::Vector{Any}
    open_r::Vector{Any}
    coeffs::Vector{Float64}
end

"""
    zero_site_h(chain, bond, lenv, renv, U_ex, V_ex) -> ZeroSiteH

Push the channel environments at links `bond` and `bond+2` through the expanded frames onto the
centre bond. `U_ex` carries `(link_l, site_l, bond)` and `V_ex` carries
`(bond, site_r, link_r)` -- an MPS tensor's layout in each case -- so the same one-site
recursions the environment stacks are built from apply unchanged.
"""
function zero_site_h(chain::XXZChain, bond::Int, lch::ChannelSet, rch::ChannelSet, U_ex, V_ex)
    terms = chain.terms
    nt = length(terms)

    # Left: everything completed to the left of the centre bond, plus each half-term that
    # is still open there. A channel open at link `bond` is closed by `term.right` at that site.
    dl = lch.done === ZERO ? ZERO : _left_step(lch.done, U_ex, nothing, bond)
    for term_index = 1:nt
        lch.open[term_index] === ZERO && continue
        contribution = _left_step(lch.open[term_index], U_ex, terms[term_index].right, bond)
        dl = _accum(dl, TLArray(terms[term_index].coeff * contribution))
    end
    ol = Any[_left_step(lch.id, U_ex, terms[term_index].left, bond) for term_index = 1:nt]

    # Right: the mirror, through the right frame at site `bond+1`.
    dr = rch.done === ZERO ? ZERO : _right_step(rch.done, V_ex, nothing, bond + 1)
    for term_index = 1:nt
        rch.open[term_index] === ZERO && continue
        contribution = _right_step(rch.open[term_index], V_ex, terms[term_index].left, bond + 1)
        dr = _accum(dr, TLArray(terms[term_index].coeff * contribution))
    end
    or = Any[_right_step(rch.id, V_ex, terms[term_index].right, bond + 1) for term_index = 1:nt]

    return ZeroSiteH(dl, dr, ol, or, [term.coeff for term in terms])
end

# The same operator with the bond's two links read out of prebuilt stacks.
zero_site_h(chain::XXZChain, bond::Int, lenv::LeftEnvStack, renv::RightEnvStack, U_ex, V_ex) =
    zero_site_h(chain, bond, left_channels(lenv, bond), right_channels(renv, bond + 2), U_ex, V_ex)

"""
    apply_zero_site(effective_h::ZeroSiteH, core) -> TLArray

`H_eff core` for the centre core, legs `(bond_l, bond_r)` in and out.

Each environment contributes its bra leg in place of the ket leg it consumes, so the
result carries the core's own legs after unpriming -- the same convention `_env_on_link` uses.
"""
function apply_zero_site(effective_h::ZeroSiteH, core)
    acc = nothing
    add!(summand) = (acc = acc === nothing ? TLArray(summand) : TLArray(acc + summand))

    if effective_h.done_l !== ZERO
        # (bra_l, bond_r) <- done_l's bra replaces bond_l
        add!(_unprime(contract(effective_h.done_l, (2,), core, (1,)), 1))
    end
    if effective_h.done_r !== ZERO
        out = contract(core, (2,), effective_h.done_r, (2,))       # (bond_l, bra_r)
        add!(_unprime(TLArray(out), 2))
    end
    for term_index in eachindex(effective_h.open_l)
        ol, or = effective_h.open_l[term_index], effective_h.open_r[term_index]
        (ol === ZERO || or === ZERO) && continue
        # A term's two halves either BOTH carry an op-leg (the U(1) XY hop, where the leg
        # carries the +/-2 charge that makes the pair allowed) or NEITHER does (SzSz, and
        # every term without symmetry). One rank test covers both channels.
        left = contract(ol, (2,), core, (1,))          # (bra_l, [op,] bond_r)
        out = if length(Telum.inds(ol)) == 3
            contract(left, (2, 3), or, (3, 2))         # close the op-leg too
        else
            contract(left, (2,), or, (2,))
        end                                             # (bra_l, bra_r)
        add!(
            TLArray(
                effective_h.coeffs[term_index] * _unprime(_unprime(TLArray(out), 1), 2),
            ),
        )
    end
    acc === nothing && throw(
        ArgumentError(
            "apply_zero_site: the effective Hamiltonian has no contributions at this bond",
        ),
    )
    return acc
end

"The BUG's root link (the step's split and compression root is one past it) when none is given."
_bug_root_link(site_count::Int) = max(1, min(site_count - 1, site_count ÷ 2))

# ── the step ─────────────────────────────────────────────────────────────────

"Total stored elements of every tensor in the state -- block-sparse, so the honest size.
(`tensor_elements` lives in baselines/tdvp2.jl; both callers resolve it at run time.)"
_state_stored(psi::SymMPS) = sum(tensor_elements(psi[site]) for site = 1:length(psi); init = 0)

"""
    OneSiteH

Single-site effective Hamiltonian at site `site`, as channel environments on links `site` and
`site+1`. Used only for 2-site TDVP's backward half-step.

A single site cannot hold both halves of a nearest-neighbour term, so there is no
analogue of the two-site action's bond contribution -- four kinds of term, not five.
"""
struct OneSiteH
    site::Int
    done_l::Any
    done_r::Any
    open_l::Vector{Any}
    open_r::Vector{Any}
    terms::Vector{XXZTerm}
end

"""
    one_site_h(chain, site, lenv, renv) -> OneSiteH

The single-site effective Hamiltonian at `site`; `lenv` must be filled at link `site` and
`renv` at link `site+1`.
"""
one_site_h(chain::XXZChain, site::Int, lch::ChannelSet, rch::ChannelSet) =
    OneSiteH(site, lch.done, rch.done, lch.open, rch.open, chain.terms)

# The same operator with the site's two links read out of prebuilt stacks.
one_site_h(chain::XXZChain, site::Int, lenv::LeftEnvStack, renv::RightEnvStack) =
    one_site_h(chain, site, left_channels(lenv, site), right_channels(renv, site + 1))

"""
    apply_one_site(effective_h::OneSiteH, site_tensor) -> TLArray

`H_eff site_tensor` for a site tensor with legs `(link_l, site, link_r)`.

The site tensor shares its layout with both `K0` and `V0`, so `_env_on_link` applies on leg 1
and leg 3 unchanged, and it already handles closing an op-leg (or not) by rank.
"""
function apply_one_site(effective_h::OneSiteH, site_tensor)
    site = effective_h.site
    acc = nothing
    add!(summand) = (acc = acc === nothing ? TLArray(summand) : TLArray(acc + summand))

    # Completed terms on either side, then each open half-term closed on this site.
    effective_h.done_l === ZERO || add!(_env_on_link(site_tensor, effective_h.done_l, 1))
    effective_h.done_r === ZERO || add!(_env_on_link(site_tensor, effective_h.done_r, 3))
    for (term_index, term) in enumerate(effective_h.terms)
        open_env = effective_h.open_l[term_index]
        open_env === ZERO || add!(
            TLArray(
                term.coeff *
                _env_on_link(_apply_site_op(site_tensor, term.right, site), open_env, 1),
            ),
        )
        open_env = effective_h.open_r[term_index]
        open_env === ZERO || add!(
            TLArray(
                term.coeff *
                _env_on_link(_apply_site_op(site_tensor, term.left, site), open_env, 3),
            ),
        )
    end
    acc === nothing && throw(ArgumentError("apply_one_site: no contributions at site $site"))
    return acc
end
