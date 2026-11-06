# Product and sector-product initial states.
#
# `dimer_state` is in QMB/Tooling/gates.jl, with the Heisenberg gates it is built from.

"""
    product_state(spins; eltype = ComplexF64) -> SymMPS

Product state from `spins::Vector{Symbol}`, each `:up` or `:down`. `eltype` sets the site-tensor
type: complex by default because BUG evolves complex amplitudes; `Float64` gives a real state.

Each site is a fusion isometry restricted to the requested physical sector. The physical leg keeps
its full space (`preserve_space=true`) so local operators still contract, and the outgoing link is
trimmed to the populated sectors, which is why the bond dimension is 1.
"""
function product_state(spins::Vector{Symbol}; eltype::Type{<:Number} = ComplexF64)
    L = length(spins)
    L >= 1 || throw(ArgumentError("product_state needs at least one site"))
    # A definite-Sz product state spans many total-S sectors, so it has no bond dimension under :SU2
    # and must throw. Under :SU2 the `:down` filter silently selects nothing, because SU(2)'s
    # spin-1/2 label coincides with `SECTOR_UP`, and a warning would let the wrong state through.
    symmetry_mode() === :SU2 && throw(
        ArgumentError(
            "product_state is not representable under :SU2 -- a definite-Sz product state has " *
            "weight in many total-spin sectors. Use a dimer state for an S = 0 start.",
        ),
    )
    return symmetry_mode() === :none ? _product_state_none(spins, eltype) :
           _product_state_u1(spins, eltype)
end

function _product_state_u1(spins::Vector{Symbol}, T::Type{<:Number} = ComplexF64)
    targets = map(enumerate(spins)) do (i, s)
        s === :up ? SECTOR_UP :
        s === :down ? SECTOR_DOWN : throw(ArgumentError("spin $i must be :up or :down, got $s"))
    end
    return sector_product_state(local_space(:U1), targets; eltype = T)
end

"""
    sector_product_state(q, sectors; eltype = ComplexF64) -> SymMPS

The product state with site `i` in the ONE-DIMENSIONAL U(1) sector `sectors[i]` of the local
space `q` -- `product_state` for any abelian site whose local states are distinct charges, such
as spin-`s` sites (charge `2 S^z`).
"""
function sector_product_state(q, sectors::AbstractVector; eltype::Type{<:Number} = ComplexF64)
    L, T = length(sectors), eltype
    L >= 1 || throw(ArgumentError("sector_product_state needs at least one site"))

    tensors = Any[]
    # Left boundary: a one-dimensional vacuum link.
    prev = (getvac(q.I, ("L,0", "L,1")), 2)

    for i = 1:L
        target = sectors[i]

        # Fuse the incoming link with a fresh physical leg. getIdentity returns
        # a lazy TLArrayView and getsub only accepts a concrete TLArray.
        F = to_concrete(getIdentity(prev, (q.I, 1); itag = "L,$(i + 1)"))
        F = to_concrete(setitag(F, 2, "S,$i"))

        # Keep only the requested spin, but preserve the physical leg's space.
        A = getsub(F, 2, s -> s == target ? Colon() : nothing; preserve_space = true)

        # Trim the outgoing link to the sectors that survived.
        A = to_concrete(A)
        # a label the site does not carry selects nothing, silently -- refuse it
        isempty(A.qlabels) &&
            throw(ArgumentError("site $i: sector $target is not a state of the local space"))
        present = Set(ql[3] for ql in A.qlabels)
        A = to_concrete(getsub(A, 3, s -> s in present ? Colon() : nothing))

        # BUG evolves complex amplitudes; promote once, here (unless a real state was asked for).
        A = to_concrete(A * one(T))

        push!(tensors, A)
        prev = (A, 3)
    end

    return SymMPS(tensors, L)
end

"""
    _dense_up_index(q) -> Int

Which index (1 or 2) of the `:none` dense physical sector is the up spin,
read off the sign of `Sz`'s diagonal. Avoids hard-coding the basis order that
`get_SU2_symmops` happens to produce.
"""
function _dense_up_index(q)
    Sz = Matrix(q.Sz.RMTs[1])
    return real(Sz[1, 1]) > real(Sz[2, 2]) ? 1 : 2
end

"""
    _product_state_none(spins, T = ComplexF64) -> SymMPS

No-symmetry product state. The physical leg is a single dense sector of
dimension 2, so a spin is selected by INDEX on the outgoing link (`getsub`
returns an `Int`), not by charge sector. The physical leg keeps its full
2-dimensional space so local operators stay contractible.
"""
function _product_state_none(spins::Vector{Symbol}, T::Type{<:Number} = ComplexF64)
    L = length(spins)
    q = local_space(:none)
    up = _dense_up_index(q)

    tensors = Any[]
    prev = (getvac(q.I, ("L,0", "L,1")), 2)
    for i = 1:L
        idx = if spins[i] === :up
            up
        elseif spins[i] === :down
            3 - up
        else
            throw(ArgumentError("spin $i must be :up or :down, got $(spins[i])"))
        end

        F = to_concrete(getIdentity(prev, (q.I, 1); itag = "L,$(i + 1)"))
        F = to_concrete(setitag(F, 2, "S,$i"))
        # One trivial sector on the outgoing link; keep the chosen spin index,
        # which trims that link to dimension 1 (product state ⇒ bond dim 1).
        A = to_concrete(getsub(F, 3, s -> idx))
        A = to_concrete(A * one(T))

        push!(tensors, A)
        prev = (A, 3)
    end

    return SymMPS(tensors, L)
end

"""
    domain_wall_state(L; eltype = ComplexF64) -> SymMPS

`↑…↑↓…↓` with the wall in the middle. Total Sz is 0 for even `L`.
"""
domain_wall_state(L::Int; eltype::Type{<:Number} = ComplexF64) =
    product_state([i <= L ÷ 2 ? :up : :down for i = 1:L]; eltype = eltype)

"""
    neel_state(L; eltype = ComplexF64) -> SymMPS

`↑↓↑↓…`.
"""
neel_state(L::Int; eltype::Type{<:Number} = ComplexF64) =
    product_state([isodd(i) ? :up : :down for i = 1:L]; eltype = eltype)
