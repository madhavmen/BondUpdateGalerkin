# The root space of a BUG step on dense QTT cores: the frame interface of the stage solver.
#
# The stage solver
# (`BUG/TimeEvolution/stage_solvers.jl`) only calls the frame interface; these methods implement it
# for dense cores under `:none`. Everything is `O(L chi^3 w)`; no `2^L` object is formed.

"`A` contracted with `B` over legs `ca` / `cb`; the free legs of `A` first, then those of `B`."
function _tc(A::AbstractArray, ca::Tuple, B::AbstractArray, cb::Tuple)
    free_a = Tuple(d for d = 1:ndims(A) if !(d in ca))
    free_b = Tuple(d for d = 1:ndims(B) if !(d in cb))
    rows = prod((size(A, d) for d in free_a); init = 1)
    inner = prod((size(A, d) for d in ca); init = 1)
    M =
        reshape(permutedims(A, (free_a..., ca...)), rows, inner) *
        reshape(permutedims(B, (cb..., free_b...)), inner, :)
    return reshape(M, (map(d -> size(A, d), free_a)..., map(d -> size(B, d), free_b)...))
end

"Left environment step `(bra, w, ket)` through site cores `Bk` (bra), `Hk`, `Ck` (ket)."
function _env_left(E, Bk, Hk, Ck)
    T = _tc(E, (3,), Ck, (1,))                     # (a', b', s', c)
    T = _tc(T, (2, 3), Hk, (1, 3))                 # (a', c, s, b)
    return permutedims(_tc(conj(Bk), (1, 2), T, (1, 3)), (1, 3, 2))    # (a, b, c)
end

"Right environment step, the mirror of `_env_left`."
function _env_right(F, Bk, Hk, Ck)
    T = _tc(Ck, (3,), F, (3,))                     # (c, s', a', b')
    T = _tc(T, (2, 4), Hk, (3, 4))                 # (c, a', b, s)
    return permutedims(_tc(conj(Bk), (2, 3), T, (4, 2)), (1, 3, 2))    # (a, b, c)
end

_unit_env(T = ComplexF64) = ones(T, 1, 1, 1)

"`<C| W |C>` for operator cores `W` by one environment sweep, never `W C` compressed."
function _op_expect(W, C)
    E = _unit_env()
    for k in eachindex(C)
        E = _env_left(E, C[k], W[k], C[k])
    end
    return E[1, 1, 1]
end

"""
    QTTFrames(W, Z)

The Galerkin frames of a BUG step as dense cores: left isometries `W[1..c]` `(link, site, bond)` and
right isometries `Z[1..L-c]` for sites `c+1..L` `(bond, site, link)`. A root matrix `Y`
(`r_L x r_R`) is the state `W Y Z`.
"""
struct QTTFrames
    W::Vector{Array{ComplexF64,3}}
    Z::Vector{Array{ComplexF64,3}}
end

"Dense frames of a step's `KLFrames` (`cbe_bug_step!(...; frames_report)`)."
function QTTFrames(fr::KLFrames)
    # `W` holds sites 1..c; `Z` has length L - 1 and `Z[j]` (j >= c) absorbs site j + 1
    W = [ComplexF64.(_none_block(fr.left_frames[i])) for i = 1:fr.root_link]
    Z = [ComplexF64.(_none_block(fr.right_frames[j])) for j = fr.root_link:length(fr.right_frames)]
    return QTTFrames(W, Z)
end

"The TT cores of `W Y Z`."
function frame_state(fr::QTTFrames, Y::AbstractMatrix)
    c = length(fr.W)
    centre = fr.W[c]
    root = reshape(
        reshape(centre, :, size(centre, 3)) * Y,
        size(centre, 1),
        size(centre, 2),
        size(Y, 2),
    )
    return vcat(fr.W[1:(c-1)], [root], fr.Z)
end

"`W' H C Z'` by environments, `H` operator cores or `nothing` (the identity)."
function frame_project(fr::QTTFrames, H, C::AbstractVector)
    c = length(fr.W)
    L = length(C)
    operator = H === nothing ? op_identity(L; T = ComplexF64) : H
    E = _unit_env()
    for k = 1:c
        E = _env_left(E, fr.W[k], operator[k], C[k])
    end
    F = _unit_env()
    for k = L:-1:(c+1)
        F = _env_right(F, fr.Z[k-c], operator[k], C[k])
    end
    return _tc(E, (2, 3), F, (2, 3))
end

"""
    frame_root_envs(fr, H; conj_ket = false) -> (E, F)

The environments of `W' H Z` at the root cut: `E[a, b, c]` over the `W` sites (bra `W`, operator
channel `b`, ket `W`, or `conj(W)` with `conj_ket` -- for the Jacobian's `psi^2 conj(dpsi)` term),
`F[a2, b, c2]` over the `Z` sites. Built per summand of an operator (the static part once per frame
set, the paddle per stage time, the density per stage iterate). With them the projected action is
[`root_apply`](@ref), `sum_b E_b Y transpose(F_b)`, which equals `frame_project(fr, H,
frame_state(fr, Y))` (with `conj_ket`: of `tt_conj(frame_state(fr, conj(Y)))`) at `O(w r^3)` per
application, with no sweep over the sites.
"""
function frame_root_envs(fr::QTTFrames, H; conj_ket::Bool = false)
    ket(core) = conj_ket ? conj(core) : core
    c = length(fr.W)
    E = _unit_env()
    for k = 1:c
        E = _env_left(E, fr.W[k], H[k], ket(fr.W[k]))
    end
    F = _unit_env()
    for k = length(H):-1:(c+1)
        F = _env_right(F, fr.Z[k-c], H[k], ket(fr.Z[k-c]))
    end
    return (E, F)
end

"`sum_b E_b Y transpose(F_b)`, the root action of [`frame_root_envs`](@ref)'s `(E, F)`."
function root_apply(envs, Y::AbstractMatrix)
    E, F = envs
    partial = _tc(E, (3,), Y, (1,))                  # (a, b, c2)
    return _tc(partial, (2, 3), F, (2, 3))           # (a, a2)
end

"Root environments of `sum_t c_t H_t` from those of each summand (channels side by side)."
_envs_cat(terms) = (
    cat((coefficient .* first(envs) for (coefficient, envs) in terms)...; dims = 2),
    cat((last(envs) for (_, envs) in terms)...; dims = 2),
)

"The root-space matrix of `W' H Z`: `M[(a,a2),(c,c2)] = sum_b E[a,b,c] F[a2,b,c2]`."
frame_operator_matrix(fr::QTTFrames, H) = root_envs_matrix(frame_root_envs(fr, H))

"The root-space matrix of the environments `(E, F)` ([`frame_operator_matrix`](@ref))."
function root_envs_matrix(envs)
    E, F = envs
    M = _tc(E, (2,), F, (2,))                        # (a, c, a2, c2)
    left_rank, right_rank = size(E, 1), size(F, 1)
    return reshape(permutedims(M, (1, 3, 2, 4)), left_rank * right_rank, left_rank * right_rank)
end

"""
    root_envs_kron_sum(envs) -> (Ml, Mr)

The Kronecker sum `Ml (x) I + I (x) Mr` nearest (Frobenius) to the root-space operator of the
environments `envs`, acting on a root `Y` as `Ml Y + Y transpose(Mr)`; the root matrix is never
formed. With the cut's channels `E_b` (left, `r_L x r_L`) and `F_b` (right, `r_R x r_R`),
`Ml = sum_b tr(F_b)/r_R E_b - kappa I`, `Mr = sum_b tr(E_b)/r_L F_b`,
`kappa = sum_b tr(E_b) tr(F_b) / (r_L r_R)`. It is exact when the operator is a Kronecker sum across
the root bond: the kinetic and trap terms at the grouped x|y interface, which is where the BUG root
sits (`L / 2 = n_x`); the rest (rotation, density, paddle, Heston's mixed term) is left to GMRES.
"""
function root_envs_kron_sum(envs)
    E, F = envs
    left_rank, channels, right_rank = size(E, 1), size(E, 2), size(F, 1)
    left_traces = [sum(E[row, channel, row] for row = 1:left_rank) for channel = 1:channels]
    right_traces = [sum(F[row, channel, row] for row = 1:right_rank) for channel = 1:channels]
    Ml = sum((right_traces[channel] / right_rank) .* E[:, channel, :] for channel = 1:channels)
    Mr = sum((left_traces[channel] / left_rank) .* F[:, channel, :] for channel = 1:channels)
    kappa = sum(left_traces .* right_traces) / (left_rank * right_rank)
    return Ml - kappa * I, Mr
end

frame_conj(::QTTFrames, C::AbstractVector) = tt_conj(C)

"""
    frame_miss(fr, X) -> Float64

`||X - W W' X Z' Z|| / ||X||`: what the root space cannot hold of the TT `X` (zero when `X` is in
the frames' span). The difference's norm is read after a QR sweep (`tt_compress`), because the Gram
`<D, D>` of a direct sum is a cancellation floored near 1e-8.
"""
function frame_miss(fr::QTTFrames, X::AbstractVector)
    norm_X = tt_norm(X)
    norm_X > 0 || return 0.0
    difference = tt_add(X, tt_scale(frame_state(fr, frame_project(fr, nothing, X)), -1.0))
    return norm(first(first(tt_compress(difference)))) / norm_X
end

"Largest isometry defect `||Q'Q - I||` over the frame cores (`W` left, `Z` right)."
function frame_defect(fr::QTTFrames)
    left_defect(Q) = (M = reshape(Q, :, size(Q, 3)); opnorm(M' * M - I))
    right_defect(Q) = (M = reshape(Q, size(Q, 1), :); opnorm(M * M' - I))
    return max(maximum(left_defect, fr.W; init = 0.0), maximum(right_defect, fr.Z; init = 0.0))
end
