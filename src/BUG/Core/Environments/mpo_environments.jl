# MPO environments (Eq. 1.8), the two-site effective Hamiltonian (Eq. 1.7) and the link-leg and
# streamed two-site contractions the Core sweeps share. Included right after
# `Core/TensorNetworks/mpo.jl`, whose `MPO` type the signatures here use.

# ── environments: Eq. (1.8) ──────────────────────────────────────────────────

"""
    MPOLink(E)

The MPO environment on ONE link: a rank-3 `(bra, mpo, ket)` tensor, or `nothing` at a
chain boundary (where it is the identity on a dim-1 vacuum leg).

The counterpart of [`ChannelSet`](@ref), and deliberately the same shape of object: a
sweep either CARRIES it bond to bond (`push_left_channels`) or reads a link out of a
prebuilt stack (`left_channels`).
"""
struct MPOLink
    E::Any
end

"The environment on a chain boundary: `L^[0] = 1` of Eq. (1.8)."
boundary_channels(::MPO) = MPOLink(nothing)

"""
    _mpo_left_step(environment, site_tensor, mpo_tensor, bra_tensor = site_tensor) -> TLArray

One step of Eq. (1.8): push the left `environment` on link `i` through `site_tensor` and
`mpo_tensor` at site `i`, returning the environment on link `i+1` with legs `(bra, mpo, ket)`.

`site_tensor` may be a state tensor OR an expanded frame -- both carry an MPS tensor's
`(link_l, site, link_r)` layout, which is what lets `zero_site_h` push an environment
through `U_ex` with this same function. `bra_tensor` is `site_tensor` unless a matrix element
between two states is wanted ([`mpo_matrix_element`](@ref)).
"""
function _mpo_left_step(environment, site_tensor, mpo_tensor, bra_tensor = site_tensor)
    if environment === nothing
        # Boundary: `link_l` is the dim-1 vacuum, contracted bra-to-ket, and the MPO's `w_l`
        # is the dim-1 leg left by trimming, dropped rather than contracted.
        contracted = contract(mpo_tensor, (2,), site_tensor, (2,))  # (w_l, s_bra, w_r, ℓ_l, ℓ_r)
        contracted =
            contract(to_concrete!(contracted), (2, 4), _bra_left_boundary(bra_tensor), (2, 1))
        contracted = to_concrete(deleteSingleton(to_concrete!(contracted), 1))  # (w_r, ℓ_r, ℓ_r')
        return to_concrete(permutedims(contracted, (3, 1, 2)))
    end
    # `to_concrete!` on each fresh `contract` result: the copying form deep-copied the
    # chi^2*w*d intermediates nobody else holds, on the most-called kernel of every sweep.
    # The outer `to_concrete` stays -- `permutedims`/`deleteSingleton` return views.
    contracted = contract(environment, (3,), site_tensor, (1,))         # (bra, w, s, ℓ_r)
    # (bra, ℓ_r, s_bra, w_r)
    contracted = contract(to_concrete!(contracted), (2, 3), mpo_tensor, (1, 2))
    # (ℓ_r, w_r, ℓ_r')
    contracted = contract(to_concrete!(contracted), (1, 3), _bra_interior(bra_tensor), (1, 2))
    return to_concrete(permutedims(to_concrete!(contracted), (3, 2, 1)))
end

"""
    _mpo_right_step(environment, site_tensor, mpo_tensor) -> TLArray

The mirror: push the right `environment` on link `i+1` through site `i`, giving link `i`.

No asymmetry to get backwards here -- unlike the channel recursion, which has to OPEN
with `term.right` and CLOSE with `term.left` on the way back. The MPO tensor is the same
object read the other way, which is the second reason this file exists.
"""
function _mpo_right_step(environment, site_tensor, mpo_tensor)
    if environment === nothing
        # Boundary: `link_r` is the dim-1 vacuum, contracted bra-to-ket; the MPO's dim-1 `w_r`
        # is dropped.
        contracted = contract(mpo_tensor, (2,), site_tensor, (2,))  # (w_l, s_bra, w_r, ℓ_l, ℓ_r)
        contracted =
            contract(to_concrete!(contracted), (2, 5), _bra_right_boundary(site_tensor), (2, 3))
        contracted = to_concrete(deleteSingleton(to_concrete!(contracted), 2))  # (w_l, ℓ_l, ℓ_l')
        return to_concrete(permutedims(contracted, (3, 1, 2)))
    end
    contracted = contract(site_tensor, (3,), environment, (3,))         # (ℓ_l, s, bra, w)
    # (ℓ_l, bra, w_l, s_bra)
    contracted = contract(to_concrete!(contracted), (2, 4), mpo_tensor, (2, 4))
    # (ℓ_l, w_l, ℓ_l')
    contracted = contract(to_concrete!(contracted), (2, 4), _bra_interior(site_tensor), (3, 2))
    return to_concrete(permutedims(to_concrete!(contracted), (3, 2, 1)))
end

"Carry the left environment on link `site` through `site` onto link `site+1`. `open_next` is
accepted for signature compatibility with the channel path and ignored: an MPO cannot
leave a term dangling past the boundary, the trimmed `W^[L]` forbids it."
push_left_channels(cs::MPOLink, mpo::MPO, site_tensor, site::Int; open_next::Bool = true) =
    MPOLink(_mpo_left_step(cs.E, site_tensor, mpo[site]))

"Mirror of [`push_left_channels`](@ref) for the right environment."
push_right_channels(cs::MPOLink, mpo::MPO, site_tensor, site::Int; open_next::Bool = true) =
    MPOLink(_mpo_right_step(cs.E, site_tensor, mpo[site]))

"""
    MPOLeftEnvStack / MPORightEnvStack

Prebuilt environments, indexed by LINK exactly as [`LeftEnvStack`](@ref) /
[`RightEnvStack`](@ref) are: entry `i` of the left stack is everything at sites `< i`,
entry `i` of the right stack everything at sites `>= i`. Links outside the sweep's reach
hold `missing`, so reading one by mistake throws instead of being taken for a boundary.
"""
struct MPOLeftEnvStack
    E::Vector{Any}
end

struct MPORightEnvStack
    E::Vector{Any}
end

"""
    left_env_stack(psi, mpo; upto=length(psi)-1) -> MPOLeftEnvStack

Build the left MPO environments on links `1 … upto+1`. The stack IS the carry, recorded
link by link, so a sweep that carries and a sweep that prebuilds cannot drift.
"""
function left_env_stack(psi::SymMPS, mpo::MPO; upto::Int = length(psi) - 1)
    site_count = length(psi)
    length(mpo) == site_count ||
        throw(DimensionMismatch("MPO has $(length(mpo)) sites, state has $site_count"))
    0 <= upto <= site_count || throw(ArgumentError("upto must be in 0:$site_count, got $upto"))
    environments = Any[missing for _ = 1:(site_count+1)]
    environments[1] = nothing
    cur = nothing
    for site = 1:upto
        cur = _mpo_left_step(cur, psi[site], mpo[site])
        environments[site+1] = cur
    end
    return MPOLeftEnvStack(environments)
end

"""
    right_env_stack(psi, mpo; downto=2) -> MPORightEnvStack

Build the right MPO environments on links `downto … L+1`.
"""
function right_env_stack(psi::SymMPS, mpo::MPO; downto::Int = 2)
    site_count = length(psi)
    length(mpo) == site_count ||
        throw(DimensionMismatch("MPO has $(length(mpo)) sites, state has $site_count"))
    1 <= downto <= site_count + 1 ||
        throw(ArgumentError("downto must be in 1:$(site_count + 1), got $downto"))
    environments = Any[missing for _ = 1:(site_count+1)]
    environments[site_count+1] = nothing
    cur = nothing
    for site = site_count:-1:downto
        cur = _mpo_right_step(cur, psi[site], mpo[site])
        environments[site] = cur
    end
    return MPORightEnvStack(environments)
end

"Link `link` of a prebuilt left stack."
left_channels(st::MPOLeftEnvStack, link::Int) = MPOLink(st.E[link])

"Link `link` of a prebuilt right stack."
right_channels(st::MPORightEnvStack, link::Int) = MPOLink(st.E[link])

"""
    _mpo_left_close(environment, site_tensor, mpo_tensor, bra_tensor = site_tensor) -> ComplexF64

As [`_mpo_left_step`](@ref) at the LAST site: `link_r` is the dim-1 vacuum boundary so it
is contracted bra-to-ket, and the MPO's `w_r` is the dim-1 leg left by trimming to the `done`
column, so it is dropped. The result is a scalar. `deleteSingleton` doubles as the
assertion that the trimming really did leave a singleton there.
"""
function _mpo_left_close(environment, site_tensor, mpo_tensor, bra_tensor = site_tensor)
    contracted = contract(environment, (3,), site_tensor, (1,))         # (bra, w, s, ℓ_r)
    # (bra, ℓ_r, s_bra, w_r)
    contracted = contract(to_concrete!(contracted), (2, 3), mpo_tensor, (1, 2))
    contracted = to_concrete(deleteSingleton(to_concrete!(contracted), 4))  # (bra, ℓ_r, s_bra)
    scalar = contract(contracted, (1, 3, 2), _bra_right_boundary(bra_tensor), (1, 2, 3))
    return ComplexF64(to_concrete(scalar)[])
end

"""
    mpo_energy(psi, mpo) -> ComplexF64

`⟨ψ|H|ψ⟩` from the MPO recursion, closing both boundaries. The independent check on
[`env_energy`](@ref): the two share no contraction.

Unnormalised, like `env_energy` -- compare against `energy(psi, gates) * norm(psi)^2`
unless the state is normalised. Exact in any gauge: bra and ket are contracted
explicitly and no isometry is assumed anywhere.
"""
mpo_energy(psi::SymMPS, mpo::MPO) = mpo_matrix_element(psi, mpo, psi)

"""
    mpo_matrix_element(bra, mpo, ket) -> ComplexF64

`⟨bra|mpo|ket⟩`, the [`mpo_energy`](@ref) recursion with a separate bra. Both states need the
same site legs and, under a symmetry, total charges that the MPO connects. Reads a non-Hermitian
MPO (`no_jump_mpo`) as a matrix on a small product basis.
"""
function mpo_matrix_element(bra::SymMPS, mpo::MPO, ket::SymMPS)
    site_count = length(ket)
    length(mpo) == site_count && length(bra) == site_count || throw(
        DimensionMismatch("MPO has $(length(mpo)) sites, ket $site_count, bra $(length(bra))"),
    )
    environment = nothing
    for site = 1:(site_count-1)
        environment = _mpo_left_step(environment, ket[site], mpo[site], bra[site])
    end
    return _mpo_left_close(environment, ket[site_count], mpo[site_count], bra[site_count])
end

"""
    _mpo2_left_step(environment, site_tensor, mpo_tensor) -> TLArray

[`_mpo_left_step`](@ref) with two copies of `mpo_tensor` between bra and ket: the left
environment of `<psi|H H|psi>` on link `i+1`, legs `(bra, w_a, w_b, ket)`.

The outer copy's virtual legs are primed (`_mpo_outer`): two copies of one MPO tensor carry
identical tags, which Telum refuses as a duplicate index.
"""
# `inc = 5`: far above any prime level the inner copy, the state or the bra carries, so the
# outer layer's virtual legs cannot coincide with any other leg whatever their direction.
_mpo_outer(mpo_tensor) = to_concrete(prime(prime(mpo_tensor, 1; inc = 5), 4; inc = 5))

function _mpo2_left_step(environment, site_tensor, mpo_tensor)
    Wa = _mpo_outer(mpo_tensor)
    if environment === nothing
        contracted = contract(mpo_tensor, (2,), site_tensor, (2,))  # (wb_l, s1, wb_r, ℓ_l, ℓ_r)
        # (wb_l, wb_r, ℓ_l, ℓ_r, wa_l, s2, wa_r)
        contracted = contract(to_concrete!(contracted), (2,), Wa, (2,))
        contracted =
            contract(to_concrete!(contracted), (6, 3), _bra_left_boundary(site_tensor), (2, 1))
        #                                                   (wb_l, wb_r, ℓ_r, wa_l, wa_r, ℓ_r')
        # (wb_r, ℓ_r, wa_l, wa_r, ℓ_r')
        contracted = to_concrete(deleteSingleton(to_concrete!(contracted), 1))
        contracted = to_concrete(deleteSingleton(contracted, 3))  # (wb_r, ℓ_r, wa_r, ℓ_r')
        return to_concrete(permutedims(contracted, (4, 3, 1, 2)))
    end
    contracted = contract(environment, (4,), site_tensor, (1,))  # (bra, wa, wb, s, ℓ_r)
    # (bra, wa, ℓ_r, s1, wb_r)
    contracted = contract(to_concrete!(contracted), (3, 4), mpo_tensor, (1, 2))
    # (bra, ℓ_r, wb_r, s2, wa_r)
    contracted = contract(to_concrete!(contracted), (2, 4), Wa, (1, 2))
    # (ℓ_r, wb_r, wa_r, ℓ_r')
    contracted = contract(to_concrete!(contracted), (1, 4), _bra_interior(site_tensor), (1, 2))
    return to_concrete(permutedims(to_concrete!(contracted), (4, 3, 2, 1)))
end

"""
    mpo_variance(psi, mpo) -> Float64

The exact energy variance `<H^2> - <H>^2` of the NORMALISED state, `<H^2> = <psi|H H|psi>` from a
two-layer MPO environment (no `H^2` MPO is formed). The accuracy measure a ground state carries
where exact diagonalisation is out of reach: zero exactly on an
eigenstate, and `variance / gap` bounds the energy error.

A difference of two `O(E^2)` numbers: its floor is `~eps * E^2`, i.e. ~1e-13 at `E ~ 30`.
"""
function mpo_variance(psi::SymMPS, mpo::MPO)
    site_count = length(psi)
    length(mpo) == site_count ||
        throw(DimensionMismatch("MPO has $(length(mpo)) sites, state has $site_count"))
    site_count >= 2 || throw(ArgumentError("mpo_variance needs at least two sites"))
    environment = nothing
    for site = 1:(site_count-1)
        environment = _mpo2_left_step(environment, psi[site], mpo[site])
    end
    # Close the two-layer environment on the last site, dropping both trimmed dim-1 `w_r` legs.
    last_tensor, last_mpo = psi[site_count], mpo[site_count]
    contracted = contract(environment, (4,), last_tensor, (1,))  # (bra, wa, wb, s, ℓ_r)
    # (bra, wa, ℓ_r, s1, wb_r)
    contracted = contract(to_concrete!(contracted), (3, 4), last_mpo, (1, 2))
    contracted = to_concrete(deleteSingleton(to_concrete!(contracted), 5))  # (bra, wa, ℓ_r, s1)
    contracted = contract(contracted, (2, 4), _mpo_outer(last_mpo), (1, 2))  # (bra, ℓ_r, s2, wa_r)
    contracted = to_concrete(deleteSingleton(to_concrete!(contracted), 4))  # (bra, ℓ_r, s2)
    h_squared = real(
        ComplexF64(
            to_concrete(
                contract(contracted, (1, 3, 2), _bra_right_boundary(last_tensor), (1, 2, 3)),
            )[],
        ),
    )
    norm_squared = norm(psi)^2
    mean_energy = real(mpo_energy(psi, mpo)) / norm_squared
    return h_squared / norm_squared - mean_energy^2
end

# ── the effective Hamiltonians: Eq. (1.7) ────────────────────────────────────

"""
    apply_h_two_site(Theta, mpo, bond, lenv, renv) -> TLArray

`H Theta` for the two-site block at bond `(bond, bond+1)`, legs
`(link_l, site_l, site_r, link_r)` in and out -- Eq. (1.7) with two MPO tensors between
the environments.

ONE contraction chain, against the channel version's five-case sum: the automaton's
states are summed over inside the MPO leg instead of being enumerated here. `lenv` must
sit on link `bond` and `renv` on link `bond+2`, and the state they were built from must be
canonical at the bond, since a boundary environment is the identity only in that gauge.
"""
function apply_h_two_site(Theta, mpo::MPO, bond::Int, lenv::MPOLink, renv::MPOLink)
    left_mpo_tensor, right_mpo_tensor = mpo[bond], mpo[bond+1]

    # Left end: attach the environment, or -- at the boundary -- keep `Theta`'s own link
    # leg and drop the MPO's dim-1 boundary leg.
    contracted = if lenv.E === nothing
        # (ℓ_l, s_r, ℓ_r, w_l, s_bra, w_mid)
        partial = to_concrete!(contract(Theta, (2,), left_mpo_tensor, (2,)))
        to_concrete(deleteSingleton(partial, 4))
    else
        partial = to_concrete!(contract(lenv.E, (3,), Theta, (1,)))  # (bra,w,s_l,s_r,ℓ_r)
        to_concrete!(contract(partial, (2, 3), left_mpo_tensor, (1, 2)))
    end                                                  # (link_l, s_r, ℓ_r, s_bra_l, w)
    # (link_l, ℓ_r, s_bra_l, s_bra_r, w)
    contracted = to_concrete!(contract(contracted, (5, 2), right_mpo_tensor, (1, 2)))

    contracted = if renv.E === nothing
        partial = to_concrete(deleteSingleton(contracted, 5))  # (link_l, ℓ_r, s_bra_l, s_bra_r)
        to_concrete(permutedims(partial, (1, 3, 4, 2)))
    else
        # (link_l, s_bra_l, s_bra_r, bra_r)
        to_concrete!(contract(contracted, (5, 2), renv.E, (2, 3)))
    end

    # Each environment's bra leg replaced a link leg: give it that link's prime level back.
    lenv.E === nothing || (contracted = _unprime(contracted, 1))
    renv.E === nothing || (contracted = _unprime(contracted, 4))
    return contracted
end

# The same action with the bond's two links read out of prebuilt stacks.
apply_h_two_site(Theta, mpo::MPO, bond::Int, lenv::MPOLeftEnvStack, renv::MPORightEnvStack) =
    apply_h_two_site(Theta, mpo, bond, left_channels(lenv, bond), right_channels(renv, bond + 2))

# ── from BUG/Randomised/cbe.jl: deterministic, used by the Core sweeps ──

"""
    _unprime(tensor, leg) -> TLArray

Drop one prime level from `leg`. An environment's bra leg is primed to keep it distinct
from the ket while both are open (`environment_stacks.jl`); once it has replaced a link leg it
must carry that link's original prime level again.
"""
#
# `to_concrete!`, not `to_concrete`: `prime` already returns a deep copy the caller owns, so a second
# copy is pure waste.
_unprime(tensor, leg::Int) = to_concrete!(prime(tensor, leg; inc = -1))

"""
    _env_on_link(tensor, environment, leg) -> TLArray

Contract `environment` onto `leg` of `tensor`, leaving the bra leg in `leg`'s place,
unprimed. If both carry an op-leg -- the tensor's last, the environment's third -- they are
contracted too (the channel closes).

`tensor` is `K0`-shaped `(link_l, site_l, bond [, op])` with `leg = 1`, or `V0`-shaped
`(bond, site_r, link_r [, op])` with `leg = 3`. In both cases the contracted link leg is
at an END of the tensor, so the bra comes back out in exactly the position it vacated
and no permutation is needed beyond the two written below.
"""
function _env_on_link(tensor, environment, leg::Int)
    nT, nE = length(tensor.inds), length(environment.inds)
    out = if nE == 3 && nT == 4
        contract(tensor, (leg, nT), environment, (2, 3))
    elseif nE == 2 && nT == 3
        contract(tensor, (leg,), environment, (2,))
    else
        error(
            "_env_on_link: rank-$nT tensor against a rank-$nE environment leaves an " *
            "operator channel dangling",
        )
    end
    # `out` = the tensor's remaining legs in order, then the environment's bra leg.
    #   leg == 1 (K0): (site_l, bond, bra) -> bra first
    #   leg == 3 (V0): (bond, site_r, bra) -> already in place
    if leg == 1
        return _unprime(to_concrete(permutedims(out, (3, 1, 2))), 1)
    elseif leg == 3
        return _unprime(to_concrete(out), 3)
    end
    throw(ArgumentError("_env_on_link: leg must be 1 or 3, got $leg"))
end

# ── from BUG/Randomised/dd_cbe.jl: deterministic, used by the Core sweeps ──
# The streamed two-site contractions: as `_fold_sketch_closures` (Randomised/cbe.jl), but with no
# final projection and Theta never formed, so every intermediate is thin in g.

"""
    _stream_left(left_amplitude, right_isometry, mpo, bond, lenv, renv, Om) -> TLArray

`X Om`, with `X` the two-site action on the block `left_amplitude * right_isometry` and `Om` in
`right_isometry`'s layout; `left_amplitude = U0 S0`, `right_isometry = V0` of the bond frame.
Legs `(link_l, site_l, g)`.
"""
function _stream_left(
    left_amplitude,
    right_isometry,
    mpo::MPO,
    bond::Int,
    lenv::MPOLink,
    renv::MPOLink,
    Om,
)
    left_mpo_tensor, right_mpo_tensor = mpo[bond], mpo[bond+1]
    Omd = Om'                                                 # (g, s_bra_r, bra_r)
    # Close the sketch against the right environment and the right MPO tensor.
    right_closure = if renv.E === nothing
        # (g, bra_r, w_mid, s_ket_r, w_r)
        partial = to_concrete!(contract(Omd, (2,), right_mpo_tensor, (3,)))
        to_concrete(deleteSingleton(partial, 5))             # (g, bra_r, w_mid, s_ket_r)
    else
        partial = to_concrete!(contract(Omd, (3,), renv.E, (1,)))   # (g, s_bra_r, w, ket_r)
        # (g, ket_r, w_mid, s_ket_r)
        to_concrete!(contract(partial, (2, 3), right_mpo_tensor, (3, 4)))
    end
    # (bond, g, w_mid), then (ℓ_l, s_l, g, w_mid), then (ℓ_l, g, w_l, s_bra_l)
    bond_partial = to_concrete!(contract(right_isometry, (2, 3), right_closure, (4, 2)))
    site_partial = to_concrete!(contract(left_amplitude, (3,), bond_partial, (1,)))
    with_left_mpo = to_concrete!(contract(site_partial, (2, 4), left_mpo_tensor, (2, 4)))
    if lenv.E === nothing
        partial = to_concrete(deleteSingleton(with_left_mpo, 3))   # (ℓ_l, g, s_bra_l)
        return to_concrete(permutedims(partial, (1, 3, 2)))  # (ℓ_l, s_bra_l, g)
    end
    partial = to_concrete!(contract(with_left_mpo, (1, 3), lenv.E, (3, 2)))  # (g, s_bra_l, bra)
    return _unprime(to_concrete(permutedims(partial, (3, 2, 1))), 1)
end

"""
    _stream_right(left_isometry, right_amplitude, mpo, bond, lenv, renv, Om) -> TLArray

`X' Om` as rows, with `X` the two-site action on the block `left_isometry * right_amplitude` and
`Om` in `left_isometry`'s layout; `left_isometry = U0`, `right_amplitude = S0 V0` of the bond
frame. Legs `(g, site_r, link_r)`.
"""
function _stream_right(
    left_isometry,
    right_amplitude,
    mpo::MPO,
    bond::Int,
    lenv::MPOLink,
    renv::MPOLink,
    Om,
)
    left_mpo_tensor, right_mpo_tensor = mpo[bond], mpo[bond+1]
    Omd = Om'                                                 # (bra_l, s_bra_l, g)
    # Close the sketch against the left environment and the left MPO tensor.
    left_closure = if lenv.E === nothing
        # (bra_l, g, w_l, s_ket_l, w_mid)
        partial = to_concrete!(contract(Omd, (2,), left_mpo_tensor, (3,)))
        to_concrete(deleteSingleton(partial, 3))             # (bra_l, g, s_ket_l, w_mid)
    else
        partial = to_concrete!(contract(Omd, (1,), lenv.E, (1,)))   # (s_bra_l, g, w, ket_l)
        # (g, ket_l, s_ket_l, w_mid)
        g_first = to_concrete!(contract(partial, (1, 3), left_mpo_tensor, (3, 1)))
        to_concrete(permutedims(g_first, (2, 1, 3, 4)))     # (ket_l, g, s_ket_l, w_mid)
    end
    # (bond, g, w_mid), then (s_r, ℓ_r, g, w_mid), then (ℓ_r, g, s_bra_r, w_r)
    bond_partial = to_concrete!(contract(left_isometry, (1, 2), left_closure, (1, 3)))
    site_partial = to_concrete!(contract(right_amplitude, (1,), bond_partial, (1,)))
    with_right_mpo = to_concrete!(contract(site_partial, (1, 4), right_mpo_tensor, (2, 1)))
    if renv.E === nothing
        partial = to_concrete(deleteSingleton(with_right_mpo, 4))  # (ℓ_r, g, s_bra_r)
        return to_concrete(permutedims(partial, (2, 3, 1)))  # (g, s_bra_r, ℓ_r)
    end
    partial = to_concrete!(contract(with_right_mpo, (1, 4), renv.E, (3, 2)))  # (g, s_bra_r, bra_r)
    return _unprime(partial, 3)
end
