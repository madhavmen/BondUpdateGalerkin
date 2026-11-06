# The Lindblad generator in Liouville space: the dissipator superoperator and the Lindblad MPO.

"`D[J] rho = J rho J' - {J'J, rho}/2`."
function dissipator_super(J::AbstractMatrix)
    JJ = J' * J
    return kron(J, conj(J)) - ket_super(JJ) / 2 - bra_super(JJ) / 2
end

"""
    lindblad_mpo(ls, L; pairs, onsite_h = nothing, jumps = []) -> MPO

The Lindbladian on `L` Liouville sites, exact, as a `pair_mpo` with the on-site terms in its field
slot. `pairs` are `(A, B, Jmat)` terms of H, `jumps` are `(Jop, rates)`, and `triples` are the
three-site terms of [`TripleTerm`](@ref). Every superoperator must pass `liouville_charge`, so a
model that breaks the grading is refused at build time.
"""
function lindblad_mpo(
    ls::LiouvilleSpace,
    L::Int;
    pairs = Any[],
    onsite_h = nothing,
    jumps = Any[],
    triples = Any[],
)
    L >= 2 || throw(ArgumentError("lindblad_mpo needs at least two sites"))
    isempty(pairs) &&
        isempty(triples) &&
        throw(ArgumentError("lindblad_mpo needs at least one two- or three-body term"))
    d = ls.d
    tts = TripleTerm[]
    for (A, B, C, w) in triples
        all(size(X) == (d, d) for X in (A, B, C)) ||
            throw(DimensionMismatch("triple operators must be $d x $d"))
        length(w) == L - 2 || throw(DimensionMismatch("triple weights must have length $(L - 2)"))
        for (sup, c) in ((ket_super, -1im), (bra_super, 1im))
            TA = liouville_op(ls, sup(A))
            length(Telum.inds(TA)) == 2 || throw(
                ArgumentError("a triple's first operator must be charge-neutral under :$(ls.sym)"),
            )
            push!(tts, TripleTerm(TA, _liouville_vertex(ls, sup(B), sup(C)), c .* ComplexF64.(w)))
        end
    end
    verts = PairVertex[]
    Js = AbstractMatrix[]
    for (A, B, Jm) in pairs
        size(A) == (d, d) && size(B) == (d, d) ||
            throw(DimensionMismatch("pair operators must be $d x $d"))
        size(Jm) == (L, L) || throw(DimensionMismatch("coupling matrix must be $L x $L"))
        # -i H rho  (ket side)  and  +i rho H  (bra side, operators transposed per site)
        for (SA, SB, c) in ((ket_super(A), ket_super(B), -1im), (bra_super(A), bra_super(B), 1im))
            push!(verts, _liouville_vertex(ls, SA, SB))
            push!(Js, c .* ComplexF64.(Jm))
        end
    end
    hs =
        onsite_h === nothing ? [zeros(d, d) for _ = 1:L] :
        onsite_h isa AbstractMatrix ? [onsite_h for _ = 1:L] : collect(onsite_h)
    length(hs) == L || throw(DimensionMismatch("need one on-site H per site"))
    field = Any[]
    for i = 1:L
        M = -1im .* (ket_super(ComplexF64.(hs[i])) - bra_super(ComplexF64.(hs[i])))
        for (Jop, rates) in jumps
            g = rates isa Number ? rates : rates[i]
            iszero(g) || (M = M + g .* dissipator_super(ComplexF64.(Jop)))
        end
        T = liouville_op(ls, M)
        length(Telum.inds(T)) == 2 || throw(
            ArgumentError(
                "the on-site superoperator at site $i is charged; H and every jump must be " *
                "covariant with a ZERO net shift under :$(ls.sym)",
            ),
        )
        push!(field, _mpo_block(T, i, "W,$(i - 1)", "W,$i", :none))
    end
    return pair_mpo(L, Vector{AbstractMatrix}(Js), verts; Iloc = ls.I, field = field, triples = tts)
end

"""
One `PairVertex` applying `SA_i SB_j`. Neutral halves pair directly; charged ones pair over the op
leg, the right half as the adjoint of the tensor of `SB'` (the `(Sp, Sp')` pattern of `xxz_chain`),
which requires the two shifts to cancel.
"""
function _liouville_vertex(ls::LiouvilleSpace, SA::AbstractMatrix, SB::AbstractMatrix)
    TA = liouville_op(ls, SA)
    if length(Telum.inds(TA)) == 2
        TB = liouville_op(ls, SB)
        length(Telum.inds(TB)) == 2 || throw(
            ArgumentError("pair halves must both be neutral or both charged under :$(ls.sym)"),
        )
        return PairVertex(TA, TB, 1.0)
    end
    # `liouville_op(ls, X)` stores transpose(X); the adjoint of the tensor for `SB'` stores
    # conj(transpose(SB')) = transpose(SB) as a matrix, i.e. it applies `SB`.
    TB = TLArray(liouville_op(ls, Matrix(SB'))')
    return PairVertex(TA, TB, 1.0)
end
