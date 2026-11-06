# Read-out of physical and entanglement observables. None of it changes the represented vector, but
# it moves the orthogonality centre (`canonical!`, which preserves norm and state). Pass `copy(psi)`
# if the centre must stay put; the driver records from a copy for this reason.

"""
    magnetisation(psi) -> Vector{Float64}

`⟨S_z^j⟩` at every site `j`, up spin = `+1/2`. This is the site-resolved profile behind a light-cone
heat-map and the basis of most error metrics. One centre sweep gives the whole profile.

```julia
mz = magnetisation(psi)          # length N, one ⟨Sz⟩ per site
```
"""
magnetisation(psi::SymMPS) = [sz_expectation(psi, j) for j = 1:length(psi)]

"""
    center_bond(psi) -> Int

Index of the central interior bond, `length(psi) ÷ 2` (the cut between sites
`N÷2` and `N÷2 + 1`). Interior bonds are numbered `1 : length(psi)-1`, the same
numbering [`bond_dims`](@ref) uses.
"""
center_bond(psi::SymMPS) = length(psi) ÷ 2

"""
    center_bond_dimension(psi) -> Int

Bond dimension of the central cut -- the usual headline entanglement number,
equal to `bond_dims(psi)[center_bond(psi)]`.
"""
center_bond_dimension(psi::SymMPS) = leg_dim(psi[center_bond(psi)], 3)

"""
    bond_spectrum(psi, bond = center_bond(psi)) -> Vector{Float64}

Schmidt values of the cut at interior `bond`, largest first, expanded by symmetry-sector degeneracy
so the length equals the bond dimension. Their squares sum to the norm² across the cut (1 for a
normalised state).

```julia
s_center = bond_spectrum(psi)        # spectrum of the central cut
s_3      = bond_spectrum(psi, 3)     # spectrum of the cut between sites 3 and 4
```
"""
function bond_spectrum(psi::SymMPS, bond::Int = center_bond(psi))
    1 <= bond <= length(psi) - 1 ||
        throw(ArgumentError("bond $bond is outside the interior-bond range 1:$(length(psi) - 1)"))
    canonical!(psi, bond)
    # Split (left link, site) from the right link: the right-link singular values
    # ARE the Schmidt spectrum of the cut just right of `bond`. cutoff=0 keeps the
    # full spectrum; get_lists exposes it without densifying the symmetric tensor.
    res = svd(psi[bond], (1, 2); cutoff = 0.0, get_lists = true)
    svals = Float64[]
    for (sigma, degeneracy, _, _) in res.kept_list
        for _ = 1:Int(degeneracy)          # multiplet size (1 under U(1))
            push!(svals, Float64(sigma))
        end
    end
    return sort!(svals; rev = true)
end

"""
    entanglement_spectrum(psi) -> Vector{Vector{Float64}}

[`bond_spectrum`](@ref) at every interior bond, from one centre sweep: `spectrum[b]` is the cut
between sites `b` and `b+1`. It is far heavier than the scalar diagnostics, so record it at a few
timestamps.

```julia
spec = entanglement_spectrum(psi)    # spec[b] = Schmidt values of bond b
```
"""
entanglement_spectrum(psi::SymMPS) = [bond_spectrum(psi, b) for b = 1:(length(psi)-1)]

"""
    bond_energies(psi, gates) -> Vector{Float64}

`⟨ψ|h_b|ψ⟩ / ⟨ψ|ψ⟩` resolved by bond, i.e. [`energy`](@ref) without the final sum. `out[b]` is the
bond between sites `b` and `b+1`; bonds with a `nothing` gate give `0.0`.

This is the light-cone observable for symmetric runs. Under `:SU2` `⟨Sz_j⟩` is identically zero and
`Sz` is not exposed, but `S_j·S_{j+1}` is an SU(2) scalar, nonzero and defined in every mode, so
`:none`, `:U1` and `:SU2` runs are directly comparable. Like `energy`, it canonicalises `psi`: the
gauge changes, not the state.
"""
function bond_energies(psi::SymMPS, gates)
    out = zeros(Float64, length(psi) - 1)
    for i = 1:(length(psi)-1)
        gates[i] === nothing && continue
        canonical!(psi, i)
        f = bond_frame(psi, i)
        th = frame_theta(f)
        num = tensor_inner(th, apply_gate(gates[i], th, f.site_l, f.site_r))
        out[i] = real(num) / real(tensor_inner(th, th))
    end
    return out
end

# Odd/even Trotter sweep over the bonds, mirroring two_site_bug/scheme.py::parity_bonds / kls_bond /
# parity_sweep. Bonds of one parity act on disjoint site pairs, so each group is an exact factor of
# the Trotter step; the splitting error lies only between the groups.

"""
    parity_bonds(L, parity) -> Vector{Int}

Left-site indices of one commuting bond group. `:even` is `1, 3, 5, …` and
`:odd` is `2, 4, …` in Julia's 1-based indexing, matching Python's 0-based
`range(0, L-1, 2)` / `range(1, L-1, 2)` bond for bond.
"""
function parity_bonds(L::Int, parity::Symbol)
    parity === :even && return collect(1:2:(L-1))
    parity === :odd && return collect(2:2:(L-1))
    throw(ArgumentError("parity must be :even or :odd, got $parity"))
end

"""
    bond_gates(psi; J=1.0, delta=1.0) -> Vector

One gate per bond, tagged for that bond's own site legs. `delta = 0` gives XX.

Entries may be `nothing` for bonds carrying no Hamiltonian term; `parity_sweep!`
skips those, matching Python's `build_bond_generators`, which leaves untouched
bonds `None`.
"""
bond_gates(psi::SymMPS; J::Float64 = 1.0, delta::Float64 = 1.0) = Any[
    heisenberg_bond_gate(psi[i].inds[2], psi[i+1].inds[2]; J = J, delta = delta) for
    i = 1:(length(psi)-1)
]

"""
    energy(psi, gates) -> Float64

`Σ_b ⟨ψ|h_b|ψ⟩ / ⟨ψ|ψ⟩` over the bonds carrying a gate.

Divides by the norm bond by bond, so it is a true expectation value even if the
state is not normalised. Canonicalises `psi` as it goes -- exactly as
`sz_expectation` does -- which changes the gauge but not the state.
"""
function energy(psi::SymMPS, gates)
    E = 0.0
    for i = 1:(length(psi)-1)
        gates[i] === nothing && continue
        canonical!(psi, i)
        f = bond_frame(psi, i)
        th = frame_theta(f)
        num = tensor_inner(th, apply_gate(gates[i], th, f.site_l, f.site_r))
        E += real(num) / real(tensor_inner(th, th))
    end
    return E
end

"""
    _dimer_state_projected(L) -> SymMPS

The dimer covering for `:none` and `:U1`, made by projecting a Neel state onto singlets.
`dimer_state` is `:SU2`-only and `product_state` is guarded off under `:SU2`, so this is the start
state buildable in every mode. Each odd two-site block is refined by power iteration on `S·S` (the
singlet grows 3x over the triplet per application); the SVD supplies the rank-2 a singlet needs. Odd
bonds are disjoint, so no Trotter error arises.

Parity is 0-based as in Python: `parity_bonds(L, :even)` gives bonds 1,3,5,... A wrong parity
silently skips every gate and returns the input unchanged, which looks like a rank freeze.
"""
function _dimer_state_projected(L::Int)
    L >= 2 && iseven(L) ||
        throw(ArgumentError("dimer state needs an even number of sites >= 2, got $L"))
    psi = neel_state(L)
    g = bond_gates(psi; delta = 1.0)
    for i = 1:2:(L-1)
        canonical!(psi, i)
        f = bond_frame(psi, i)
        th = frame_theta(f)
        for _ = 1:60
            th = to_concrete(apply_gate(g[i], th, f.site_l, f.site_r))
            nn = sqrt(real(tensor_inner(th, th)))
            nn > 0 || error("dimer projection collapsed at bond $i")
            th = to_concrete((1.0 / nn) * th)
        end
        res = svd(th, (1, 2); cutoff = 1e-14, Nkeep = 4, get_lists = true)
        # Restore the bond tag: svd labels the new bond `svdL`, and `move_right!` then sees that tag
        # on two legs, which Telum rejects as a duplicate TLIndex. This fired only under
        # `:U1`/`:none`, since `dimer_state` reaches here only outside `:SU2`; every other svd
        # write-back already retags.
        bt = "L,$(i + 1)"
        psi[i] = to_concrete(setitag(res.U, 3, bt))
        psi[i+1] = to_concrete(setitag(to_concrete(res.S * res.Vd), 1, bt))
        psi.center = i + 1
    end
    n = norm(psi)
    n > 0 && (psi[psi.center] = to_concrete((1.0 / n) * psi[psi.center]))
    return psi
end
