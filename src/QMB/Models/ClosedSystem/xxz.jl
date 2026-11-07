# The XXZ chain: its term list and its MPO.
#
# The term-list types `XXZTerm`/`XXZChain` are in BUG (Core/TensorNetworks/mpo.jl), whose
# `mpo_from_terms` reads them; the model supplies the local identity they do not carry.

"""
    xxz_chain(L; J=1.0, delta=1.0) -> XXZChain

`H = J Σ_i (Sx Sx + Sy Sy + delta Sz Sz)` as a nearest-neighbour term list, the input of
[`xxz_mpo`](@ref). The XY term is factorised per mode: under `:U1` it is the adjoint pairs
`(Sp, Sp')` and `(Sm, Sm')` (Telum's normalisation already holds the 1/2); under `:none` the
coefficient is -1, since Telum normalises `Sp = -(1/sqrt2) S+_std`. `test_fixtures_b155550.jl`
evolves its MPO against the dense Hamiltonian.
"""
function xxz_chain(L::Int; J::Float64 = 1.0, delta::Float64 = 1.0)
    L >= 2 || throw(ArgumentError("xxz_chain needs at least two sites, got $L"))
    q = local_space()
    terms = XXZTerm[]
    if length(Telum.inds(q.Sp)) == 3
        push!(terms, XXZTerm(q.Sp, TLArray(q.Sp'), J))
        push!(terms, XXZTerm(q.Sm, TLArray(q.Sm'), J))
    else
        push!(terms, XXZTerm(q.Sp, q.Sm, -J))
        push!(terms, XXZTerm(q.Sm, q.Sp, -J))
    end
    delta == 0.0 || push!(terms, XXZTerm(q.Sz, q.Sz, J * delta))
    return XXZChain(L, terms, J, delta)
end

"""
    heisenberg_su2_chain(L; J=1.0) -> XXZChain

`H = J sum_i S_i . S_{i+1}` as ONE irreducible term for `symmetry_mode() === :SU2`, where Telum
exposes only `:S` and `:I`. Its two halves are joined by an `S = 1` op-leg, in place of the three
abelian terms; the downstream code is unchanged.

Only the isotropic point exists, so an anisotropic chain runs under `:U1` with [`xxz_chain`](@ref).
The overall constant is measured (Clebsch-Gordan scaling), pinned against dense diagonalisation.
"""
function heisenberg_su2_chain(L::Int; J::Float64 = 1.0)
    L >= 2 || throw(ArgumentError("heisenberg_su2_chain needs at least two sites, got $L"))
    symmetry_mode() === :SU2 || throw(
        ArgumentError(
            "heisenberg_su2_chain needs symmetry_mode() === :SU2, got $(symmetry_mode()); " *
            "use xxz_chain for :U1 or :none",
        ),
    )
    q = local_space(:SU2)
    # One term, op-leg contracted between the two halves -- the same shape the rank-3 `:U1`
    # branch of `xxz_chain` builds for `Sp`/`Sm`.
    terms = [XXZTerm(q.S, TLArray(q.S'), J)]
    return XXZChain(L, terms, J, 1.0)
end

"""
    xxz_mpo(L; J=1.0, delta=1.0) -> MPO

`H = J Σ_i (Sx Sx + Sy Sy + delta Sz Sz)` as an MPO. Convenience for
`mpo_from_terms(xxz_chain(L; J, delta))`.
"""
xxz_mpo(L::Int; J::Float64 = 1.0, delta::Float64 = 1.0) =
    mpo_from_terms(xxz_chain(L; J = J, delta = delta); Iloc = _xxz_identity())

"""
    heisenberg_su2_mpo(L; J=1.0) -> MPO

The SU(2) Heisenberg chain as an MPO: ONE term, whose `S=1` op-leg becomes the MPO bond,
so the virtual dimension is 3 multiplets rather than U(1)'s 5.
"""
heisenberg_su2_mpo(L::Int; J::Float64 = 1.0) =
    mpo_from_terms(heisenberg_su2_chain(L; J = J); Iloc = _xxz_identity())

# ── the term list as a step's operator ──

"The local identity of the spin-1/2 site space of the current mode."
_xxz_identity() = symmetry_mode() === :SU2 ? local_space(:SU2).I : local_space().I

BUG.cbe_bug_step!(psi::SymMPS, h::XXZChain, tau; kwargs...) =
    cbe_bug_step!(psi, mpo_from_terms(h; Iloc = _xxz_identity()), tau; kwargs...)

BUG.cbe_bug_midpoint_step!(psi::SymMPS, h::XXZChain, tau; kwargs...) =
    cbe_bug_midpoint_step!(psi, mpo_from_terms(h; Iloc = _xxz_identity()), tau; kwargs...)
