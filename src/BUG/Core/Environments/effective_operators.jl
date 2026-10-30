# Moved from BUG-Julia@b155550:src/RSVDCBEBondUpdate/operators/mpo_fold.jl, src/RSVDCBEBondUpdate/operators/zero_site_core.jl, src/RSVDCBEBondUpdate/baselines/tdvp2.jl.

# ── from src/RSVDCBEBondUpdate/operators/mpo_fold.jl ──

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
    one_site_h(mpo, j, lenv, renv) -> MPOOneSiteH

`H_eff` at site `j`. `lenv` must sit on link `j` and `renv` on link `j+1` -- the same
convention the channel-path method of this function uses.
"""
one_site_h(mpo::MPO, j::Int, lenv::MPOLink, renv::MPOLink) =
    MPOOneSiteH(lenv.E, mpo[j], renv.E)

one_site_h(mpo::MPO, j::Int, lenv::MPOLeftEnvStack, renv::MPORightEnvStack) =
    one_site_h(mpo, j, left_channels(lenv, j), right_channels(renv, j + 1))

"""
    apply_one_site(H1::MPOOneSiteH, A) -> TLArray

`H_eff A` for a site tensor `A` with legs `(link_l, site, link_r)`, in and out.
"""
function apply_one_site(H1::MPOOneSiteH, A)
    # `to_concrete!` on the contraction results: this is the hot path, and `to_concrete` on an already
    # concrete tensor would deep-copy every block.
    # Safe only because every argument is a fresh `contract` result; the inputs belong to callers and keep `to_concrete`.
    T = if H1.l === nothing
        X = to_concrete!(contract(A, (2,), H1.w, (2,)))     # (ℓ_l, ℓ_r, w_l, s_bra, w_r)
        to_concrete(deleteSingleton(X, 3))                  # (ℓ_l, ℓ_r, s_bra, w_r)
    else
        X = to_concrete!(contract(H1.l, (3,), A, (1,)))      # (bra_l, w, s, ℓ_r)
        X = to_concrete!(contract(X, (2, 3), H1.w, (1, 2)))  # (bra_l, ℓ_r, s_bra, w_r)
        X
    end                                                    # (link_l, ℓ_r, s_bra, w_r)

    T = if H1.r === nothing
        X = to_concrete(deleteSingleton(T, 4))             # (link_l, ℓ_r, s_bra)
        to_concrete(permutedims(X, (1, 3, 2)))
    else
        to_concrete!(contract(T, (4, 2), H1.r, (2, 3)))    # (link_l, s_bra, bra_r)
    end

    H1.l === nothing || (T = _unprime(T, 1))
    H1.r === nothing || (T = _unprime(T, 3))
    return T
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
    zero_site_h(mpo, i, lenv, renv, U_ex, V_ex) -> MPOZeroSiteH

Push the environments at links `i` and `i+2` onto the centre bond through the EXPANDED
frames. `U_ex` carries `(link_l, site_l, bond)` and `V_ex` carries `(bond, site_r,
link_r)` -- an MPS tensor's layout in each case -- so this is one more step of Eq. (1.8)
with the frame in place of a state tensor, and needs no separate code.
"""
zero_site_h(mpo::MPO, i::Int, lenv::MPOLink, renv::MPOLink, U_ex, V_ex) =
    MPOZeroSiteH(_mpo_left_step(lenv.E, U_ex, mpo[i]),
                 _mpo_right_step(renv.E, V_ex, mpo[i + 1]))

"""
    apply_zero_site(H0::MPOZeroSiteH, S) -> TLArray

`H_eff S` for the centre core `S`, legs `(bond_l, bond_r)` in and out. Each environment
contributes its bra leg in place of the ket leg it consumes, so the result carries `S`'s
own legs after unpriming.
"""
function apply_zero_site(H0::MPOZeroSiteH, S)
    T = to_concrete!(contract(H0.l, (3,), S, (1,)))        # (bra_l, mpo, bond_r)
    T = to_concrete!(contract(T, (2, 3), H0.r, (2, 3)))    # (bra_l, bra_r)
    return _unprime(_unprime(T, 1), 2)
end

"""
    shift_h(H, s) -> H or ShiftedH

The effective Hamiltonian `H - s` for the generators of an imaginary-time step. `s = 0` returns `H`
itself, so an unshifted step is bit-identical.

A shift by the identity leaves every Krylov space unchanged and only changes the weights read off
them. The CBE sketches are exact as they stand, since `P_perp (H - s) Theta = P_perp H Theta`.
"""
struct ShiftedH{T}
    h::T
    s::Float64
end

shift_h(h, s::Real) = iszero(s) ? h : ShiftedH(h, Float64(s))

apply_one_site(H::ShiftedH, A) = to_concrete(apply_one_site(H.h, A) - H.s * A)
apply_zero_site(H::ShiftedH, S) = to_concrete(apply_zero_site(H.h, S) - H.s * S)

# ── from src/RSVDCBEBondUpdate/operators/zero_site_core.jl ──

# ── zero-site generator, frame construction and truncation sweeps ──
# The Lubich sweep that used to sit here was removed. Its re-ranking growth passes do not carry over
# to the Krylov-accumulated frame of `cbe_bug_step!`, so measurements do not port between the two.

# CBE-BUG: K and L basis sweeps in CBE frames, then ONE Galerkin step at the centre bond (only the
# centre tensor is evolved, so nothing is counted twice). The K/L ODEs are never solved:
# `cbe_expand` supplies the directions. The sweeps must not write back into `psi` (zero padding would
# collapse the rank), and no rank-4 tensor is ever allocated.

# ── writing an expansion back into the state ─────────────────────────────────

"""
    _absorb_left!(psi, i, U_ex) -> psi

Replace `psi[i]` by the expanded left frame and push the (zero-padded) remainder into
`psi[i+1]`, leaving the state unchanged and the centre at `i+1`.

`U_ex†(A_i A_{i+1})` is evaluated as `(U_ex†A_i)A_{i+1}`: the two-site block is never
formed, and the intermediate is `bond_ex x bond_old`.
"""
function _absorb_left!(psi::SymMPS, i::Int, U_ex)
    tag = psi[i].inds[3].itags
    M = contract(U_ex', (1, 2), psi[i], (1, 2))        # (bond_ex, bond_old)
    B = contract(M, (2,), psi[i + 1], (1,))            # (bond_ex, site_r, link_r)
    psi[i]     = to_concrete(setitag(U_ex, 3, tag))
    psi[i + 1] = to_concrete(setitag(B, 1, tag))
    psi.center = i + 1
    return psi
end

"""
    _absorb_right!(psi, i, V_ex) -> psi

Mirror: `psi[i+1]` becomes the expanded right frame and the remainder goes into `psi[i]`,
leaving the centre at `i`.
"""
function _absorb_right!(psi::SymMPS, i::Int, V_ex)
    tag = psi[i].inds[3].itags
    M = contract(psi[i + 1], (2, 3), V_ex', (2, 3))    # (bond_old, bond_ex)
    A = contract(psi[i], (3,), M, (1,))                # (link_l, site_l, bond_ex)
    psi[i]     = to_concrete(setitag(A, 3, tag))
    psi[i + 1] = to_concrete(setitag(V_ex, 1, tag))
    psi.center = i
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
_relink(T, t1, t3) = to_concrete(setitag(setitag(T, 1, t1), 3, t3))

"""
    _frame_from(left, right) -> BondFrame

`bond_frame`'s factorisation on tensors handed in rather than read from `psi`. The basis sweeps must
not touch the state, so the tensor is a carry `C_{i-1} psi[i]`, which `bond_frame` cannot express.

Correctness needs exactly one of `left`/`right` to carry the amplitude while the other is an
isometry: the K-sweep (psi canonical at 1) carries it in `left`, the L-sweep (psi canonical at L)
in `right`.
"""
function _frame_from(left, right)
    res_l = svd(left,  (1, 2), "bU,L", "bU,R"; cutoff = 0.0)
    res_r = svd(right, (1,),   "bV,L", "bV,R"; cutoff = 0.0)
    U0 = to_concrete(res_l.U)
    V0 = to_concrete(res_r.Vd)
    S0 = to_concrete(to_concrete(res_l.S * res_l.Vd) * to_concrete(res_r.U * res_r.S))
    return BondFrame(U0, S0, V0,
                     left.inds[1], left.inds[2], left.inds[3],
                     right.inds[2], right.inds[3],
                     leg_dim(U0, 3), copy(left.spaces[3]))
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
    zero_site_h(h, i, lenv, renv, U_ex, V_ex) -> ZeroSiteH

Push the channel environments at links `i` and `i+2` through the expanded frames onto the
centre bond. `U_ex` carries `(link_l, site_l, bond)` and `V_ex` carries
`(bond, site_r, link_r)` -- an MPS tensor's layout in each case -- so the same one-site
recursions the environment stacks are built from apply unchanged.
"""
function zero_site_h(h::XXZChain, i::Int, lch::ChannelSet, rch::ChannelSet,
                     U_ex, V_ex)
    terms = h.terms
    nt = length(terms)

    # Left: everything completed to the left of the centre bond, plus each half-term that
    # is still open there. A channel open at link i is closed by `term.right` at site i.
    dl = lch.done === ZERO ? ZERO : _left_step(lch.done, U_ex, nothing, i)
    for t in 1:nt
        lch.open[t] === ZERO && continue
        c = _left_step(lch.open[t], U_ex, terms[t].right, i)
        dl = _accum(dl, to_concrete(terms[t].coeff * c))
    end
    ol = Any[_left_step(lch.id, U_ex, terms[t].left, i) for t in 1:nt]

    dr = rch.done === ZERO ? ZERO :
         _right_step(rch.done, V_ex, nothing, i + 1)
    for t in 1:nt
        rch.open[t] === ZERO && continue
        c = _right_step(rch.open[t], V_ex, terms[t].left, i + 1)
        dr = _accum(dr, to_concrete(terms[t].coeff * c))
    end
    or = Any[_right_step(rch.id, V_ex, terms[t].right, i + 1) for t in 1:nt]

    return ZeroSiteH(dl, dr, ol, or, [t.coeff for t in terms])
end

zero_site_h(h::XXZChain, i::Int, lenv::LeftEnvStack, renv::RightEnvStack, U_ex, V_ex) =
    zero_site_h(h, i, left_channels(lenv, i), right_channels(renv, i + 2), U_ex, V_ex)

"""
    apply_zero_site(H0, S) -> TLArray

`H_eff S` for the centre core `S`, legs `(bond_l, bond_r)` in and out.

Each environment contributes its bra leg in place of the ket leg it consumes, so the
result carries `S`'s own legs after unpriming -- the same convention `_env_on_link` uses.
"""
function apply_zero_site(H0::ZeroSiteH, S)
    acc = nothing
    add!(x) = (acc = acc === nothing ? to_concrete(x) : to_concrete(acc + x))

    if H0.done_l !== ZERO
        # (bra_l, bond_r) <- done_l's bra replaces bond_l
        add!(_unprime(to_concrete!(contract(H0.done_l, (2,), S, (1,))), 1))
    end
    if H0.done_r !== ZERO
        out = contract(S, (2,), H0.done_r, (2,))       # (bond_l, bra_r)
        add!(_unprime(to_concrete(out), 2))
    end
    for t in eachindex(H0.open_l)
        ol, or = H0.open_l[t], H0.open_r[t]
        (ol === ZERO || or === ZERO) && continue
        # A term's two halves either BOTH carry an op-leg (the U(1) XY hop, where the leg
        # carries the +/-2 charge that makes the pair allowed) or NEITHER does (SzSz, and
        # every term without symmetry). One rank test covers both channels.
        left = contract(ol, (2,), S, (1,))             # (bra_l, [op,] bond_r)
        out = if length(ol.inds) == 3
            contract(left, (2, 3), or, (3, 2))         # close the op-leg too
        else
            contract(left, (2,), or, (2,))
        end                                             # (bra_l, bra_r)
        add!(to_concrete(H0.coeffs[t] * _unprime(_unprime(to_concrete(out), 1), 2)))
    end
    acc === nothing && throw(ArgumentError(
        "apply_zero_site: the effective Hamiltonian has no contributions at this bond"))
    return acc
end

"The BUG's root link `c` (the step's split and compression root is `c + 1`) when none is given."
_bug_root_link(L::Int) = max(1, min(L - 1, L ÷ 2))

# ── the step ─────────────────────────────────────────────────────────────────

"Total stored elements of every tensor in the state -- block-sparse, so the honest size.
(`tensor_elements` lives in baselines/tdvp2.jl; both callers resolve it at run time.)"
_state_stored(psi::SymMPS) = sum(tensor_elements(psi[i]) for i in 1:length(psi); init = 0)

# ── from src/RSVDCBEBondUpdate/baselines/tdvp2.jl ──

"""
    OneSiteH

Single-site effective Hamiltonian at site `j`, as channel environments on links `j` and
`j+1`. Used only for 2-site TDVP's backward half-step.

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
    one_site_h(h, j, lenv, renv) -> OneSiteH

The single-site effective Hamiltonian at site `j`; `lenv` must be filled at link `j` and
`renv` at link `j+1`.
"""
one_site_h(h::XXZChain, j::Int, lch::ChannelSet, rch::ChannelSet) =
    OneSiteH(j, lch.done, rch.done, lch.open, rch.open, h.terms)

one_site_h(h::XXZChain, j::Int, lenv::LeftEnvStack, renv::RightEnvStack) =
    one_site_h(h, j, left_channels(lenv, j), right_channels(renv, j + 1))

"""
    apply_one_site(H1, M) -> TLArray

`H_eff M` for a site tensor `M` with legs `(link_l, site, link_r)`.

`M` shares its layout with both `K0` and `V0`, so `_env_on_link` applies on leg 1 and
leg 3 unchanged, and it already handles closing an op-leg (or not) by rank.
"""
function apply_one_site(H1::OneSiteH, M)
    j = H1.site
    acc = nothing
    add!(x) = (acc = acc === nothing ? to_concrete(x) : to_concrete(acc + x))

    H1.done_l === ZERO || add!(_env_on_link(M, H1.done_l, 1))
    H1.done_r === ZERO || add!(_env_on_link(M, H1.done_r, 3))
    for (t, term) in enumerate(H1.terms)
        o = H1.open_l[t]
        o === ZERO || add!(to_concrete(term.coeff *
            _env_on_link(_apply_site_op(M, term.right, j), o, 1)))
        o = H1.open_r[t]
        o === ZERO || add!(to_concrete(term.coeff *
            _env_on_link(_apply_site_op(M, term.left, j), o, 3)))
    end
    acc === nothing && throw(ArgumentError(
        "apply_one_site: no contributions at site $j"))
    return acc
end
