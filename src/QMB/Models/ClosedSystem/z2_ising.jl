# H = -J Σ Z Z - h Σ X made block-sparse in the sigma^x basis: |+> is Z2 charge 0, |-> charge 1.
# The Z2 local space (`ising_local_space`) is in QMB/Tooling/local_spaces.jl: Telum cannot derive
# Z2 operators, so its blocks come from private `Telum._localspace_cgt_fields`. The Hamiltonian is
# `tfim_mpo` (tfim.jl).

# ── states and observables ───────────────────────────────────────────────────────────

"""
    ising_product_state(dirs::Vector{Symbol}) -> SymMPS

Product state in the sigma^x basis; each entry is `:plus` (`X = +1`) or `:minus` (`X = -1`).

Same fusion recursion as `product_state`, but selecting on the Z2 charge (`:Z2`) or on the
dense index (`:none`) -- the two modes select differently because `:none` has ONE sector of
dimension 2, so there is no charge to filter on, only a basis index.
"""
function ising_product_state(dirs::Vector{Symbol})
    L = length(dirs)
    L >= 1 || throw(ArgumentError("ising_product_state needs at least one site"))
    all(d -> d in (:plus, :minus), dirs) || throw(
        ArgumentError(
            "each entry must be :plus or :minus, got $(filter(d -> !(d in (:plus, :minus)), dirs))",
        ),
    )
    q = ising_local_space()
    z2 = symmetry_mode() === :Z2

    tensors = Any[]
    prev = (getvac(q.I, ("L,0", "L,1")), 2)
    for i = 1:L
        F = TLArray(getIdentity(prev, (q.I, 1); itag = "L,$(i + 1)"))
        F = TLArray(setitag(F, 2, "S,$i"))
        A = if z2
            target = dirs[i] === :plus ? _Z2_Q0 : _Z2_Q1
            B = TLArray(getsub(F, 2, s -> s == target ? Colon() : nothing; preserve_space = true))
            present = Set(ql[3] for ql in Telum.logical_qlabels(B))
            TLArray(getsub(B, 3, s -> s in present ? Colon() : nothing))
        else
            # `_z2_diag_op`'s ordering is (|+>, |->), and `_getLocalSpace_no_symmetry` keeps it,
            # so index 1 is |+> and 2 is |->.
            TLArray(getsub(F, 3, s -> dirs[i] === :plus ? 1 : 2))
        end
        A = TLArray(A * (1.0 + 0.0im))
        push!(tensors, A)
        prev = (A, 3)
    end
    return SymMPS(tensors, L)
end

"""
    ising_polarised_state(L) -> SymMPS

`|+ + ... +>`, every spin along `+x`: the exact ground state at `J = 0` and the natural quench
start. Z2 charge 0, `chi = 1` -- a genuine product state, so it is also a test of whether the
integrator can GROW rank at all (without CBE it would sit at `chi = 1` forever).
"""
ising_polarised_state(L::Int) = ising_product_state(fill(:plus, L))

"""
    ising_kink_state(L) -> SymMPS

`|+ ... + - ... ->` with the flip in the middle: a domain wall in the sigma^x basis. Z2 charge
`(L ÷ 2) mod 2`. Gives a light cone under a TFIM quench in the same way the Heisenberg domain
wall does.
"""
ising_kink_state(L::Int) = ising_product_state([i <= L ÷ 2 ? :plus : :minus for i = 1:L])

"""
    x_profile(psi) -> Vector{Float64}

`<X_j>` on every site -- the transverse magnetisation, and the TFIM's light-cone observable.

WHY NOT `<Z_j>`: it is identically zero in any Z2-symmetric state, because `Z` changes the
charge and so has no diagonal matrix element at all. That is the symmetry, not a limitation --
the order parameter has to be probed with the CORRELATOR `<Z_i Z_j>`, never a single site.
"""
x_profile(psi::SymMPS) = [real(site_expval(psi, j, ising_local_space().X)) for j = 1:length(psi)]
