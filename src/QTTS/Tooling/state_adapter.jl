# Dense QTT cores <-> the BUG state and operator types under `:none`.
#
# Moved from BUG-Julia@b155550:src/RSVDCBEBondUpdate/qtt/qtt_telum.jl. The leg templates come from
# Telum's two-dimensional site space directly instead of a spin model's product state and local
# space, which QTTS may not reference.
#
# Under `:none` a TLArray is one dense block with `symm = ()`, so a core is wrapped by giving it the
# `inds` of a tensor of the same role: a product-state site (state) or an `onsite_mpo` site
# (operator). The arrows and tags are then exactly the solver's, and only the data are ours.
# ⚠ The MPO stores the TRANSPOSE of the applied matrix (`mpo_product`: the ket contracts leg 2, the
# output is leg 3), so an operator core `(w_l, out, in, w_r)` is stored as `(w_l, in, out, w_r)`.

const _BIT_SPACE = Ref{Any}(nothing)

"The no-symmetry two-dimensional site space of one quantics bit, built once."
function _bit_space()
    _BIT_SPACE[] === nothing &&
        (_BIT_SPACE[] = getLocalSpace(SpinOptions(nothing, 1), ("s", "s", "op")))
    return _BIT_SPACE[]
end

function _require_none(who::AbstractString)
    symmetry_mode() === :none || throw(
        ArgumentError(
            "$who needs set_symmetry!(:none) -- a quantics field has no charge, got :$(symmetry_mode())",
        ),
    )
end

"The legs `(\"L,i\", \"S,i\", \"L,i+1\")` of every site of an `L`-site state, built as a product state is."
function _bit_site_inds(L::Int)
    q = _bit_space()
    prev = (getvac(q.I, ("L,0", "L,1")), 2)
    inds = Vector{Any}(undef, L)
    for i = 1:L
        F = to_concrete(getIdentity(prev, (q.I, 1); itag = "L,$(i + 1)"))
        F = to_concrete(setitag(F, 2, "S,$i"))
        A = to_concrete(getsub(F, 3, s -> 1))
        inds[i] = A.inds
        prev = (A, 3)
    end
    return inds
end

"One dense `:none` TLArray holding `A`, with the legs `inds`."
function _none_tensor(A::AbstractArray{T,N}, inds::NTuple{N,TLIndex}) where {T,N}
    return TLArray(
        (),
        [ntuple(_ -> (), N)],
        Float64[],
        [()],
        [Array(A)],
        inds,
        ntuple(l -> [((), size(A, l))], N),
    )
end

"The dense block of a `:none` TLArray (zeros when it holds no sector)."
function _none_block(t)
    t = to_concrete(t)
    symm(t) === () || isempty(symm(t)) || throw(ArgumentError("_none_block: tensor is symmetric"))
    length(t.qlabels) <= 1 || throw(ArgumentError("_none_block: $(length(t.qlabels)) sectors"))
    isempty(t.qlabels) && return zeros(eltype(t), size(t))
    return Array(t.RMTs[1])
end

"""
    qtt_mps(cores; center = 1) -> SymMPS

The state of TT `cores` (`(chi_l, 2, chi_r)`, site order), gauged to `center` by exact SVD moves.
"""
function qtt_mps(cores::AbstractVector; center::Int = 1)
    _require_none("qtt_mps")
    L = length(cores)
    inds = _bit_site_inds(L)
    T = mapreduce(eltype, promote_type, cores; init = ComplexF64)
    ts = Any[_none_tensor(Array{T,3}(cores[i]), inds[i]) for i = 1:L]
    psi = SymMPS(ts, L)
    canonical!(psi, 1)                        # move_left! from L: exact, whatever the input gauge
    canonical!(psi, center)
    return psi
end

"The TT cores `(chi_l, 2, chi_r)` of `psi`, any gauge."
function qtt_cores(psi::SymMPS)
    _require_none("qtt_cores")
    return [_none_block(psi[i]) for i = 1:length(psi)]
end

"""
    qtt_to_mpo(W) -> MPO

The solver's MPO for operator cores `(w_l, out, in, w_r)` (the stored-transpose convention of the
file header is applied here, once).
"""
function qtt_to_mpo(W::AbstractVector)
    _require_none("qtt_to_mpo")
    L = length(W)
    L >= 2 || throw(ArgumentError("qtt_to_mpo needs at least two sites"))
    q = _bit_space()
    tmpl = onsite_mpo(L, [q.I for _ = 1:L]; Iloc = q.I)
    T = mapreduce(eltype, promote_type, W; init = ComplexF64)
    return MPO(
        Any[_none_tensor(Array{T,4}(permutedims(W[i], (1, 3, 2, 4))), tmpl[i].inds) for i = 1:L],
    )
end

"The operator cores `(w_l, out, in, w_r)` of a `:none` MPO (inverse of `qtt_to_mpo`)."
qtt_mpo_cores(mpo::MPO) = [permutedims(_none_block(mpo[i]), (1, 3, 2, 4)) for i = 1:length(mpo)]

"""
    qtt_apply(mpo, psi; tol_trunc = 1e-12, maxdim = 10_000) -> SymMPS

`mpo * psi` through the solver's own MPO product and compression ([`apply_mpo`](@ref)), for checks
that the BUG path and the dense core algebra agree.
"""
qtt_apply(mpo::MPO, psi::SymMPS; tol_trunc::Real = 1e-12, maxdim::Int = 10_000) =
    first(apply_mpo(mpo, psi; tol_trunc = Float64(tol_trunc), maxdim = maxdim))

"`sqrt(dx dy) ||psi||`, the physical L2 norm (the weight is kept OUT of every contraction)."
qtt_l2norm(psi::SymMPS, g::QTTGrid) = sqrt(qtt_cell(g)) * sqrt(max(real(overlap(psi, psi)), 0.0))
