# Channel environments of a nearest-neighbour term list (`XXZChain`) and their stacks.
#

# Channel environments for a nearest-neighbour XXZ chain: the MPO contraction with the virtual leg
# named, so each channel's charge rides on the op-leg of Telum's Sp/Sm.
# The bra leg is primed and the ket unprimed, so the bra tensor differs at each boundary. A wrong
# prime does not throw; it silently traces the wrong pair, which is why `env_energy` is checked
# against the gate energy.

# ── the Hamiltonian as a term list ───────────────────────────────────────────

"""
    XXZTerm(left, right, coeff)

One nearest-neighbour term `coeff · left_i ⊗ right_{i+1}`, repeated on every bond.

`left` and `right` are bare local operators; they are retagged onto their site at
contraction time. Either may be rank-3, carrying an `op` leg whose charge is what
makes the pair U(1)-allowed; when both are, that leg is contracted between them (this
is the XY hop, and it is why the two operators must be a `Sp`/`Sp'` adjoint pair
rather than `Sp`/`Sm` -- see `gates.jl`'s header on the 1/√2 normalisation).
"""
struct XXZTerm
    left::Any
    right::Any
    coeff::Float64
end

"""
    XXZChain(L, terms, J, delta)

`H = J Σ_i (Sx Sx + Sy Sy + delta Sz Sz)` on `L` sites, as a term list.
Built by [`xxz_chain`](@ref); `delta = 0` is XX.
"""
struct XXZChain
    L::Int
    terms::Vector{XXZTerm}
    J::Float64
    delta::Float64
end

# The number of sites, so a chain can be checked against the state's length.
Base.length(chain::XXZChain) = chain.L

"The term list of `chain`. `hamiltonian_terms(chain)[t]` pairs with a channel's `open[t]`."
hamiltonian_terms(chain::XXZChain) = chain.terms

"""
    bond_gate(chain, site_l, site_r) -> TLArray

The two-site gate of the term list `chain` on one bond, used by the term-list two-site action
(`apply_h_two_site`). The model that builds `chain` supplies the method (QMB's XXZ chain).
"""
function bond_gate end

# A channel with no contribution yet is not the boundary identity: `done` starts at ZERO, while `id`
# starts at the boundary identity (`nothing`). Conflating them is a silent factor-of-one bug.

# The marker type of an empty channel; `ZERO` is its only instance.
struct ZeroEnv end
const ZERO = ZeroEnv()

# Add `addend` into a running channel sum; returns the new sum. Adding into ZERO returns the addend
# itself, so an empty channel never contributes a spurious zero tensor.
_accum(accumulated::ZeroEnv, addend) = addend
_accum(accumulated, addend) = TLArray(accumulated + addend)

# A sweep carries the channels from link i to link i+1 through site i, O(1) per bond instead of an
# O(L) rebuild. `canonical!` rewrites tensors, so a stack built before it is stale: canonicalise
# first, then build the one stack the other side needs, then sweep.

"""
    ChannelSet

The channel environment on ONE link: `id` (the transported identity), `done` (completed
terms) and `open[t]` (term `t`'s dangling half). `id === nothing` is a boundary; `done` or
`open[t]` may be `ZERO`.
"""
struct ChannelSet
    id::Any
    done::Any
    open::Vector{Any}
end

"The channels on a chain boundary: identity transported from nothing, no terms yet."
boundary_channels(chain::XXZChain) = ChannelSet(nothing, ZERO, Any[ZERO for _ in chain.terms])

"""
    push_left_channels(cs, chain, site_tensor, site; open_next=true) -> ChannelSet

Carry the left channels on link `site` through `site_tensor` onto link `site+1`.

`open_next=false` suppresses opening new half-terms, for the last site of a chain where a
term opened would never be closed.
"""
function push_left_channels(
    cs::ChannelSet,
    chain::XXZChain,
    site_tensor,
    site::Int;
    open_next::Bool = true,
)
    terms = chain.terms
    # Completed terms ride through with an identity on this site.
    acc = cs.done === ZERO ? ZERO : _left_step(cs.done, site_tensor, nothing, site)
    # Each open half-term is closed here by its right operator.
    for (term_index, term) in enumerate(terms)
        cs.open[term_index] === ZERO && continue
        closed = _left_step(cs.open[term_index], site_tensor, term.right, site)
        acc = _accum(acc, TLArray(term.coeff * closed))
    end
    # The transported identity opens every term with its left operator on this site.
    nxt = Any[
        open_next ? _left_step(cs.id, site_tensor, terms[term_index].left, site) : ZERO for
        term_index in eachindex(terms)
    ]
    return ChannelSet(_left_step(cs.id, site_tensor, nothing, site), acc, nxt)
end

"""
    push_right_channels(cs, chain, site_tensor, site; open_next=true) -> ChannelSet

Mirror: carry the right channels on link `site+1` through `site` onto link `site`. A right
channel OPENS with `term.right` and CLOSES with `term.left` -- the asymmetry `_right_step`
documents.
"""
function push_right_channels(
    cs::ChannelSet,
    chain::XXZChain,
    site_tensor,
    site::Int;
    open_next::Bool = true,
)
    terms = chain.terms
    acc = cs.done === ZERO ? ZERO : _right_step(cs.done, site_tensor, nothing, site)
    for (term_index, term) in enumerate(terms)
        cs.open[term_index] === ZERO && continue
        closed = _right_step(cs.open[term_index], site_tensor, term.left, site)
        acc = _accum(acc, TLArray(term.coeff * closed))
    end
    nxt = Any[
        open_next ? _right_step(cs.id, site_tensor, terms[term_index].right, site) : ZERO for
        term_index in eachindex(terms)
    ]
    return ChannelSet(_right_step(cs.id, site_tensor, nothing, site), acc, nxt)
end

# ── local pieces ─────────────────────────────────────────────────────────────

"Retag an operator's two site legs onto site `site`, leaving any op-leg alone."
_retag_sites(operator, site::Int) =
    TLArray(setitag(setitag(operator, 1, "S,$site"), 2, "S,$site"))

"""
    _apply_site_op(site_tensor, operator, site) -> TLArray

`operator` acting on `site_tensor` at `site`, returned in `site_tensor`'s own layout with any
op-leg appended: `(link_l, site, link_r)` or `(link_l, site, link_r, op)`.

The ket-facing leg of `operator` is found by ARROW, not position: a plain operator has it at
leg 1, an adjoint (`Sp'`) at leg 2, and `contract` asserts opposite arrows, so
hard-coding leg 1 breaks on exactly the adjoint half of every XY term.
"""
function _apply_site_op(site_tensor, operator, site::Int)
    operator === nothing && return site_tensor
    Oi = _retag_sites(operator, site)
    ket_leg = Telum.inds(Oi)[1].dir == '+' ? 1 : 2
    contracted = contract(Oi, (ket_leg,), site_tensor, (2,))
    # contracted = (operator's remaining legs..., link_l, link_r). The surviving site leg of a
    # rank-3 operator is always at index < 3, so its remainder is (site_out, op) in order.
    return length(Telum.inds(Oi)) == 2 ? TLArray(permutedims(contracted, (2, 1, 3))) :
           TLArray(permutedims(contracted, (3, 1, 4, 2)))
end

"Bra tensor for an interior step: both link legs primed, so each pairs with the
environment's bra leg rather than with the ket."
_bra_interior(site_tensor) = TLArray(prime(prime(site_tensor, 1), 3)')

"Bra tensor at the left boundary: `link_l` unprimed, because there the bra and ket
links are the same dim-1 vacuum leg and must contract with each other."
_bra_left_boundary(site_tensor) = TLArray(prime(site_tensor, 3)')

"Bra tensor closing the right boundary: `link_r` unprimed, for the same reason."
_bra_right_boundary(site_tensor) = TLArray(prime(site_tensor, 1)')

# ── one left step ────────────────────────────────────────────────────────────

"""
    _left_step(environment, site_tensor, operator, site) -> TLArray

Push the left `environment` at link `site` through `site`, inserting `operator` on the ket;
`environment === nothing` is the boundary identity. Returns `(bra, ket [, op])`.

Distinguished by rank: a rank-2 environment with an op-free operator gives rank 2 (transport); a
rank-2 environment with a rank-3 operator opens a channel; rank-3 with rank-3 closes one. A rank-3
environment with an op-free operator would carry a half-open term past a site, so it is rejected
rather than silently building a longer-range Hamiltonian.
"""
function _left_step(environment, site_tensor, operator, site::Int)
    AO = _apply_site_op(site_tensor, operator, site)
    nAO = length(Telum.inds(AO))

    if environment === nothing
        # Boundary: contract link_l bra-to-ket along with the site leg.
        out = contract(AO, (1, 2), _bra_left_boundary(site_tensor), (1, 2))
    else
        nE = length(Telum.inds(environment))
        if nE == 3 && nAO == 4
            tmp = contract(AO, (1, 4), environment, (2, 3))   # close: pair the op-legs too
        elseif nE == 2
            tmp = contract(AO, (1,), environment, (2,))
        else
            error(
                "_left_step: rank-$nE environment with a rank-$nAO operator would " *
                "carry an open channel past site $site",
            )
        end
        # tmp = (site, link_r, [op], bra) -- the bra leg is always last.
        out = contract(tmp, (1, length(Telum.inds(tmp))), _bra_interior(site_tensor), (2, 1))
    end

    # out = (ket, [op], bra); put the bra first.
    leg_count = length(Telum.inds(out))
    return TLArray(permutedims(out, leg_count == 2 ? (2, 1) : (3, 1, 2)))
end

"""
    _left_close(environment, site_tensor, operator, site) -> ComplexF64

As `_left_step`, but at the last site: `link_r` is the dim-1 vacuum boundary, so it is
contracted bra-to-ket and the result is a scalar. A channel still open here would mean
a term running off the end of the chain, so that is an error.
"""
function _left_close(environment, site_tensor, operator, site::Int)
    AO = _apply_site_op(site_tensor, operator, site)
    nAO, nE = length(Telum.inds(AO)), length(Telum.inds(environment))
    tmp =
        (nE == 3 && nAO == 4) ? contract(AO, (1, 4), environment, (2, 3)) :
        nE == 2 ? contract(AO, (1,), environment, (2,)) :
        error("_left_close: rank-$nE environment with a rank-$nAO operator at site $site")
    length(Telum.inds(tmp)) == 3 ||
        error("_left_close: an operator channel is still open at the right boundary")
    scalar = contract(tmp, (1, 2, 3), _bra_right_boundary(site_tensor), (2, 3, 1))
    return ComplexF64(scalar[])
end

# The left and right environment stacks and the two-site effective action, built on the channel
# steps above.

# ── the left stack ───────────────────────────────────────────────────────────

"""
    LeftEnvStack

Left environments at links `1 … n`, one entry per link (`id[i]`, `done[i]`,
`open[i][t]`). `id[1] === nothing` is the boundary identity; `done`/`open` entries are
`ZERO` where the channel has no contribution yet.

`id` is stored even though it is the identity for a left-canonical chain: computing it
explicitly keeps the recursion valid with the orthogonality centre anywhere, and
`test_henv.jl` asserts it IS the identity in the canonical case rather than assuming it.
"""
struct LeftEnvStack
    id::Vector{Any}
    done::Vector{Any}
    open::Vector{Vector{Any}}
end

"""
    left_env_stack(psi, chain; upto=length(psi)-1) -> LeftEnvStack

Build the left channel environments at links `1 … upto+1` by sweeping sites `1 … upto`.

`upto = L-1` (the default) fills every link a bond update can need on its left, i.e.
up to link `L`. `upto = 0` is legal and gives just the boundary at link 1 -- what bond 1
needs, where there is nothing to its left.
"""
function left_env_stack(psi::SymMPS, chain::XXZChain; upto::Int = length(psi) - 1)
    site_count = length(psi)
    length(chain) == site_count ||
        throw(DimensionMismatch("chain has $(length(chain)) sites, state has $site_count"))
    0 <= upto <= site_count || throw(ArgumentError("upto must be in 0:$site_count, got $upto"))

    # The stack IS the carry, recorded link by link -- one implementation of the
    # recursion, so a sweep that carries and a sweep that prebuilds can never drift.
    cs = boundary_channels(chain)
    id = Any[cs.id]
    done = Any[cs.done]
    open = Vector{Any}[cs.open]
    for site = 1:upto
        # A channel opened at a site is closed at the next one, so nothing opens on the last
        # site of the chain.
        cs = push_left_channels(cs, chain, psi[site], site; open_next = site < site_count)
        push!(id, cs.id)
        push!(done, cs.done)
        push!(open, cs.open)
    end

    return LeftEnvStack(id, done, open)
end

"""
    env_energy(psi, chain) -> ComplexF64

`⟨ψ|H|ψ⟩` from the channel recursion, closing the right boundary at site `L`.

Requires `psi` left-canonical up to `L` in the sense that the recursion is exact for
any gauge -- it contracts bra against ket explicitly and never assumes an isometry.
Valid for an unnormalised state: this is the unnormalised expectation value, so
compare against `energy(psi, gates) * norm(psi)^2` unless the state is normalised.
"""
function env_energy(psi::SymMPS, chain::XXZChain)
    site_count = length(psi)
    st = left_env_stack(psi, chain; upto = site_count - 1)
    last_tensor = psi[site_count]
    # Completed terms close with an identity on the last site; open ones with their right half.
    expectation =
        st.done[site_count] === ZERO ? ComplexF64(0) :
        _left_close(st.done[site_count], last_tensor, nothing, site_count)
    for (term_index, term) in enumerate(chain.terms)
        open_env = st.open[site_count][term_index]
        open_env === ZERO && continue
        expectation += term.coeff * _left_close(open_env, last_tensor, term.right, site_count)
    end
    return expectation
end

# A right channel opens with `term.right` and closes with `term.left`, the mirror of the left step.
# Under U(1) the op-leg arrows reverse, which is why the term list stores both halves.

"""
    _right_step(environment, site_tensor, operator, site) -> TLArray

Push the right `environment` at link `site+1` through `site`, inserting `operator` on the ket.
`environment === nothing` means the right boundary identity. Returns `(bra, ket [, op])` on link
`site`, the same convention as [`_left_step`](@ref).
"""
function _right_step(environment, site_tensor, operator, site::Int)
    AO = _apply_site_op(site_tensor, operator, site)
    nAO = length(Telum.inds(AO))

    if environment === nothing
        # Boundary: contract link_r bra-to-ket along with the site leg.
        out = contract(AO, (2, 3), _bra_right_boundary(site_tensor), (2, 3))
    else
        nE = length(Telum.inds(environment))
        if nE == 3 && nAO == 4
            tmp = contract(AO, (3, 4), environment, (2, 3))   # close: pair the op-legs too
        elseif nE == 2
            tmp = contract(AO, (3,), environment, (2,))
        else
            error(
                "_right_step: rank-$nE environment with a rank-$nAO operator would " *
                "carry an open channel past site $site",
            )
        end
        # tmp = (link_l, site, [op], bra) -- the bra leg is always last.
        out = contract(tmp, (2, length(Telum.inds(tmp))), _bra_interior(site_tensor), (2, 3))
    end

    # out = (ket, [op], bra); put the bra first.
    leg_count = length(Telum.inds(out))
    return TLArray(permutedims(out, leg_count == 2 ? (2, 1) : (3, 1, 2)))
end

"""
    _right_close(environment, site_tensor, operator, site) -> ComplexF64

As `_right_step` at site 1, where `link_l` is the dim-1 vacuum boundary and is
contracted bra-to-ket, giving a scalar.
"""
function _right_close(environment, site_tensor, operator, site::Int)
    AO = _apply_site_op(site_tensor, operator, site)
    nAO, nE = length(Telum.inds(AO)), length(Telum.inds(environment))
    tmp =
        (nE == 3 && nAO == 4) ? contract(AO, (3, 4), environment, (2, 3)) :
        nE == 2 ? contract(AO, (3,), environment, (2,)) :
        error("_right_close: rank-$nE environment with a rank-$nAO operator at site $site")
    length(Telum.inds(tmp)) == 3 ||
        error("_right_close: an operator channel is still open at the left boundary")
    scalar = contract(tmp, (1, 2, 3), _bra_left_boundary(site_tensor), (1, 2, 3))
    return ComplexF64(scalar[])
end

# ── the right stack ──────────────────────────────────────────────────────────

"""
    RightEnvStack

Right environments, indexed by LINK like [`LeftEnvStack`](@ref) but filled from the
right: entry `i` is the environment on link `i`, i.e. everything at sites `i … L`.
`id[L+1] === nothing` is the boundary identity. Links below the sweep's reach hold
`missing`, so using one by mistake throws instead of being read as a boundary.
"""
struct RightEnvStack
    id::Vector{Any}
    done::Vector{Any}
    open::Vector{Vector{Any}}
end

"""
    right_env_stack(psi, chain; downto=2) -> RightEnvStack

Build the right channel environments on links `downto … L+1` by sweeping sites
`L … downto`. `downto = 2` (the default) fills every link a bond update can need on its
right, i.e. down to link 2. `downto = L+1` is legal and gives just the boundary -- what
bond `L-1` needs, where there is nothing to its right.
"""
function right_env_stack(psi::SymMPS, chain::XXZChain; downto::Int = 2)
    site_count = length(psi)
    length(chain) == site_count ||
        throw(DimensionMismatch("chain has $(length(chain)) sites, state has $site_count"))
    1 <= downto <= site_count + 1 ||
        throw(ArgumentError("downto must be in 1:$(site_count + 1), got $downto"))
    nt = length(chain.terms)

    id = Any[missing for _ = 1:(site_count+1)]
    done = Any[missing for _ = 1:(site_count+1)]
    open = Vector{Any}[Any[missing for _ = 1:nt] for _ = 1:(site_count+1)]

    cs = boundary_channels(chain)
    id[site_count+1] = cs.id
    done[site_count+1] = cs.done
    open[site_count+1] = cs.open
    for site = site_count:-1:downto
        # A channel opened at a site is closed at the one before, so nothing opens at site 1.
        cs = push_right_channels(cs, chain, psi[site], site; open_next = site > 1)
        id[site] = cs.id
        done[site] = cs.done
        open[site] = cs.open
    end

    return RightEnvStack(id, done, open)
end

"""
    env_energy_right(psi, chain) -> ComplexF64

`⟨ψ|H|ψ⟩` swept from the right instead of the left. Same number as
[`env_energy`](@ref); its purpose is to pin the right recursion independently, since
the two sweeps share no contraction.
"""
function env_energy_right(psi::SymMPS, chain::XXZChain)
    st = right_env_stack(psi, chain; downto = 2)
    first_tensor = psi[1]
    # Completed terms close with an identity on site 1; open ones with their left half.
    expectation =
        st.done[2] === ZERO ? ComplexF64(0) : _right_close(st.done[2], first_tensor, nothing, 1)
    for (term_index, term) in enumerate(chain.terms)
        open_env = st.open[2][term_index]
        open_env === ZERO && continue
        expectation += term.coeff * _right_close(open_env, first_tensor, term.left, 1)
    end
    return expectation
end

# The two-site effective action on bond (i, i+1): the done terms on either side, the two straddling
# channels, and the bond term, delegated to `apply_gate` (pinned element by element to the gate
# path). The far sides must be identities, i.e. the centre sits at i or i+1, which `bond_frame`
# already demands.

"""
    _apply_theta_op(Theta, operator, leg, site) -> TLArray

`operator` at `site` acting on `leg` (2 for the left site, 3 for the right) of the two-site
block, returned in `Theta`'s layout `(link_l, site_l, site_r, link_r)` with any op-leg appended.

As in [`_apply_site_op`](@ref), the ket-facing leg of `operator` is found by arrow, so an
adjoint half-term works unchanged.
"""
function _apply_theta_op(Theta, operator, leg::Int, site::Int)
    operator === nothing && return Theta
    Oi = _retag_sites(operator, site)
    ket_leg = Telum.inds(Oi)[1].dir == '+' ? 1 : 2
    contracted = contract(Oi, (ket_leg,), Theta, (leg,))
    rank3 = length(Telum.inds(Oi)) == 3
    # contracted = (site_out, [op], Theta's legs except `leg`, in order).
    perm = if leg == 2
        rank3 ? (3, 1, 4, 5, 2) : (2, 1, 3, 4)
    elseif leg == 3
        rank3 ? (3, 4, 1, 5, 2) : (2, 3, 1, 4)
    else
        throw(ArgumentError("_apply_theta_op: leg must be 2 or 3, got $leg"))
    end
    return TLArray(permutedims(contracted, perm))
end

"""
    _apply_theta_env(block, environment, link_leg) -> TLArray

Contract `environment` onto `link_leg` (1 or 4) of `block`, which carries
`(link_l, site_l, site_r, link_r [, op])`. If both carry an op-leg they are contracted (the
channel closes); the environment's bra leg replaces `link_leg` and is unprimed so the result
has exactly `Theta`'s legs again.
"""
function _apply_theta_env(block, environment, link_leg::Int)
    link_leg in (1, 4) || throw(ArgumentError("link_leg must be 1 or 4, got $link_leg"))
    nT, nE = length(Telum.inds(block)), length(Telum.inds(environment))
    out = if nT == 5 && nE == 3
        contract(block, (link_leg, 5), environment, (2, 3))
    elseif nE == 2 && nT == 4
        contract(block, (link_leg,), environment, (2,))
    else
        error(
            "_apply_theta_env: rank-$nT block against a rank-$nE environment leaves " *
            "an operator channel dangling",
        )
    end
    # `out` = the block's remaining legs in order, then the environment's bra leg.
    #   link_leg == 1: (site_l, site_r, link_r, bra) -> bra first
    #   link_leg == 4: (link_l, site_l, site_r, bra) -> already in place
    out = link_leg == 1 ? TLArray(permutedims(out, (4, 1, 2, 3))) : TLArray(out)
    return TLArray(prime(out, link_leg; inc = -1))
end

"""
    apply_h_two_site(Theta, chain, bond, lenv, renv) -> TLArray

`H Theta` for the two-site block at bond `(bond, bond+1)`, legs
`(link_l, site_l, site_r, link_r)` in and out.

`lenv` must be filled at link `bond` and `renv` at link `bond+2`; the state they were built
from must be canonical at the bond (see the note above). This is the whole chain's `H`,
not the bond's gate -- which is the point of the module.
"""
function apply_h_two_site(Theta, chain::XXZChain, bond::Int, lch::ChannelSet, rch::ChannelSet)
    tl, tr = Telum.inds(Theta)[2].itags, Telum.inds(Theta)[3].itags
    # Collect the terms and sum once: pairwise `acc + summand` rebuilt the metadata for
    # each of the roughly nine summands, which made the probe allocation-bound. Telum's `sum` aligns
    # all summands once.
    terms = AbstractTLArray[]
    add!(summand) = push!(terms, summand)

    # (a), (b): the completed channels, with an implicit identity on the far side.
    lch.done === ZERO || add!(_apply_theta_env(Theta, lch.done, 1))
    rch.done === ZERO || add!(_apply_theta_env(Theta, rch.done, 4))

    # (c), (d): the half-open channels, closed inside the block.
    for (term_index, term) in enumerate(chain.terms)
        open_env = lch.open[term_index]
        if open_env !== ZERO
            operated = _apply_theta_op(Theta, term.right, 2, bond)
            add!(TLArray(term.coeff * _apply_theta_env(operated, open_env, 1)))
        end
        open_env = rch.open[term_index]
        if open_env !== ZERO
            operated = _apply_theta_op(Theta, term.left, 3, bond + 1)
            add!(TLArray(term.coeff * _apply_theta_env(operated, open_env, 4)))
        end
    end

    # (e) the bond's own term, via the gate the Trotter path is pinned against.
    add!(apply_gate(bond_gate(chain, tl, tr), Theta, tl, tr))

    # `nothing` when every channel was ZERO and there is no gate -- preserved from the pairwise
    # version, which returned an unset `acc`.
    isempty(terms) && return nothing
    return length(terms) == 1 ? terms[1] : TLArray(sum(terms))
end

# The same action with the bond's two links read out of prebuilt stacks.
apply_h_two_site(Theta, chain::XXZChain, bond::Int, lenv::LeftEnvStack, renv::RightEnvStack) =
    apply_h_two_site(Theta, chain, bond, left_channels(lenv, bond), right_channels(renv, bond + 2))

# ── reading a link out of a prebuilt stack ───────────────────────────────────
#
# Defined here rather than beside `ChannelSet`, because a method signature's types are
# resolved when the method is defined and `LeftEnvStack` / `RightEnvStack` are declared
# further up this file.

"Link `link` of a prebuilt left stack, as a `ChannelSet`."
left_channels(lenv::LeftEnvStack, link::Int) =
    ChannelSet(lenv.id[link], lenv.done[link], lenv.open[link])

"Link `link` of a prebuilt right stack, as a `ChannelSet`."
right_channels(renv::RightEnvStack, link::Int) =
    ChannelSet(renv.id[link], renv.done[link], renv.open[link])
