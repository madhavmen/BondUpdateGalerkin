# Moved from BUG-Julia@b155550:src/RSVDCBEBondUpdate/bug/krylov_enrich.jl, src/RSVDCBEBondUpdate/bug/krylov_closure.jl, src/RSVDCBEBondUpdate/bug/krylov_closure_loop.jl.
# krylov_closure_frames.jl: the K/L frames of the order-p closure, their shared environment caches, and the projections of the state and its frontiers onto them.

# ── from src/RSVDCBEBondUpdate/bug/krylov_closure.jl ──

# ── GENERAL-ORDER KRYLOV CLOSURE (order-p condition) ──
# The order-p error is O(|tau|^(p+1)) once `(I - P) A B^j psi = 0` for j < p, with B = P A P.
# Frontiers are judged by `(I - P) A F_j`, not by a root residual (which concerns B, not A).

"A frontier below this fraction of the state norm is the zero vector (e.g. `H psi = 0`)."
const _FRONTIER_FLOOR = 1e-14

"Internal bound on closure passes. Reaching it with the frames still growing is `:closure`."
_closure_maxpass(order::Int) = 2 * order + 2

# Each enriching frontier sweep re-tags its frames with a new generation, so a frame and the
# frame it grew from never share a tag inside one overlap tensor.
_tagWg(i::Int, g::Int) = g == 0 ? _tagW(i) : "cbeW$g,$i"
_tagZg(j::Int, g::Int) = g == 0 ? _tagZ(j) : "cbeZ$g,$j"

"""
    KLFrames(W, Z, c, lenv, renv, gen)

The K/L frames of one step and the root environments they imply: `W[1..c]` left frames,
`Z[c..L-1]` right frames (`Z[j]` absorbs site `j+1`), `lenv` the environment at link `c` through
`W[1..c-1]`, `renv` the one at link `c+2` through `Z[c+1..L-1]`. `gen` counts the enrichments,
so a [`KrylovFrontier`](@ref) built in older frames is refused rather than silently misread.

`lcache` / `rcache` hold the environments every frontier sweep of ONE generation shares
([`_side_cache!`](@ref)); each half-sweep writes only its own, so the parallel halves never
write the same field.
"""
mutable struct KLFrames
    W::Vector{Any}
    Z::Vector{Any}
    c::Int
    lenv::Any
    renv::Any
    gen::Int
    lcache::Any
    rcache::Any
    # The generators' energy shift (`shift_h`): the root `B - eshift` and the frontier sweeps'
    # one-site Krylov operators. `0.0` is the unshifted step.
    eshift::Float64
end

KLFrames(W, Z, c, lenv, renv, gen) = KLFrames(W, Z, c, lenv, renv, gen, nothing, nothing, 0.0)

"""
Threads below which the closure keeps its frontier sweeps in sequence. Each sweep already runs
its two halves concurrently under `parallel = true`, so two concurrent sweeps need four threads
to gain anything; below that the 2-thread production path stays exactly what it was.
"""
const _FRONTIER_THREADS = 4

_frontier_parallel(parallel::Bool) = parallel && Threads.nthreads() >= _FRONTIER_THREADS

"""
Whether a measuring sweep runs its bonds as concurrent tasks ([`_bond_measurements`](@ref)) when
[`_frontier_parallel`](@ref) holds. A switch, not a tolerance: it changes no result at
`fold = false` and exists so a timing job can A/B it in one build (`bug_profile_2d.jl`,
`BP_BOND_TASKS`). Off by default: at 4 threads, D = 512 it gained nothing (job 17287169).
"""
const _BOND_TASKS = Ref(false)

"""
    _prime_caches!(fr, psi, mpo, core)

Build this generation's shared environments ([`_side_cache!`](@ref)) before frontier sweeps run
concurrently, so the concurrent sweeps only READ them. `core` is any root core: the cached
parts do not depend on it. The two sides are separate fields and are built concurrently.
"""
function _prime_caches!(fr::KLFrames, psi::SymMPS, mpo, core)
    Fm = _frontier_mps(psi, fr, core)
    t = Threads.@spawn _side_cache!(fr, Fm, mpo, :right)
    _side_cache!(fr, Fm, mpo, :left)
    fetch(t)
    return fr
end

"""
    _side_cache!(fr, Fm, mpo, side) -> (chain, stack)

The environments of one half-sweep that do not depend on the frontier, built once per frame
generation. ⛔ EVERY FRONTIER OF A GENERATION IS `(W ⊗ Z) core` WITH ONLY THE CORE DIFFERENT, so
re-gauging `Fm` to site 1 (left half) changes sites `1..c` only: the right stack on links
`c+1..L+1` and the frame chain `<W[1..i-1]| H |W[1..i-1]>` (the non-enriching sweep's `lch`)
are the same for every frontier. Rebuilding them per sweep was ~1/3 of a measuring sweep
(bug4 10x4 D=256 profile, job 17283685). `chain[i]` is the left channels at bond `i` (`i = 1..c`),
`stack` the right-stack entries on links `>= c+1`; the right half mirrors both (`chain[j]` the
right channels at bond `j = c..L-1`, `stack` the left-stack entries on links `<= c`).
"""
function _side_cache!(fr::KLFrames, Fm::SymMPS, mpo, side::Symbol)
    L, c = length(Fm), fr.c
    cur = side === :left ? fr.lcache : fr.rcache
    valid = cur !== nothing && cur.gen == fr.gen
    chain = valid ? cur.chain : nothing          # an enriching sweep seeds the chain only
    E = valid ? cur.stack : nothing
    (chain !== nothing && E !== nothing) && return (chain, E)
    if chain === nothing
        chain = Vector{Any}(undef, L - 1)
        ch = boundary_channels(mpo)
        if side === :left
            for i = 1:c
                chain[i] = ch
                i < c && (ch = push_left_channels(ch, mpo, fr.W[i], i))
            end
        else
            for j = (L-1):-1:c
                chain[j] = ch
                j > c && (ch = push_right_channels(ch, mpo, fr.Z[j], j + 1))
            end
        end
    end
    E = Any[missing for _ = 1:(L+1)]
    env = nothing
    if side === :left
        E[L+1] = nothing
        for k = L:-1:(c+1)
            env = _mpo_right_step(env, Fm[k], mpo[k])
            E[k] = env
        end
        fr.lcache = (gen = fr.gen, chain = chain, stack = E)
    else
        E[1] = nothing
        for i = 1:(c-1)
            env = _mpo_left_step(env, Fm[i], mpo[i])
            E[i+1] = env
        end
        fr.rcache = (gen = fr.gen, chain = chain, stack = E)
    end
    return (chain, E)
end

"""
    KrylovFrontier(core, degree, gen)

A projected Krylov frontier `F_j = (tau^j / j!) B^j S0`: its coefficients `core` in the root
basis of frames generation `gen` (a block-sparse `(b_L, b_R)` tensor, so its U(1) sectors travel
with it) and its degree `j`. The cached environments are the frames' own.
"""
struct KrylovFrontier
    core::Any
    degree::Int
    gen::Int
end

"""
    frames_from_state(psi, mpo, c) -> KLFrames

The state's OWN Schmidt frames at root bond `c` -- the subspace every step starts from, before
any expansion. Used to start a closure from scratch (`include_zero = true`) and by the tests.
"""
function frames_from_state(psi::SymMPS, mpo, c::Int)
    L = length(psi)
    1 <= c <= L - 1 || throw(ArgumentError("root must be a bond in 1:$(L - 1), got $c"))
    p = copy(psi)
    canonical!(p, c)
    W = Vector{Any}(undef, c)
    for i = 1:(c-1)
        W[i] = p[i]
    end
    W[c] = to_concrete(svd(p[c], (1, 2); cutoff = 0.0).U)
    q = copy(psi)
    canonical!(q, c + 1)
    Z = Vector{Any}(undef, L - 1)
    Z[c] = to_concrete(svd(q[c+1], (1,); cutoff = 0.0).Vd)
    for j = (c+1):(L-1)
        Z[j] = q[j+1]
    end
    # Frame tags on both ends of every frame link, as the half-sweeps write them.
    for i = 1:c
        W[i] = to_concrete(setitag(W[i], 3, _tagW(i)))
        i < c && (W[i+1] = to_concrete(setitag(W[i+1], 1, _tagW(i))))
    end
    for j = c:(L-1)
        Z[j] = to_concrete(setitag(Z[j], 1, _tagZ(j)))
        j > c && (Z[j-1] = to_concrete(setitag(Z[j-1], 3, _tagZ(j))))
    end
    lenv = boundary_channels(mpo)
    for i = 1:(c-1)
        lenv = push_left_channels(lenv, mpo, W[i], i)
    end
    renv = boundary_channels(mpo)
    for j = (L-1):-1:(c+1)
        renv = push_right_channels(renv, mpo, Z[j], j + 1)
    end
    return KLFrames(W, Z, c, lenv, renv, 0)
end

"`<W[1..c] | psi>` on the left of the root: `(b_L, link_{c+1})`. A full contraction, any gauge."
function _overlap_left(psi::SymMPS, W, c::Int)
    AL = nothing
    for i = 1:c
        T = AL === nothing ? psi[i] : to_concrete!(contract(AL, (2,), psi[i], (1,)))
        AL = to_concrete!(contract(W[i]', (1, 2), T, (1, 2)))         # (b_L, link_{i+1})
    end
    return AL
end

"`<Z[c..L-1] | psi>` on the right of the root: `(link_{c+1}, b_R)`. A full contraction, any gauge."
function _overlap_right(psi::SymMPS, Z, c::Int)
    AR = nothing
    for j = (length(psi)-1):-1:c
        B = AR === nothing ? psi[j+1] : to_concrete!(contract(psi[j+1], (3,), AR, (1,)))
        AR = to_concrete!(contract(B, (2, 3), Z[j]', (2, 3)))         # (link_j, b_j)
    end
    return AR
end

"""
    _projection_miss(psi, W, Z, c) -> Float64

`||(I - P) psi||` for `P` the projector onto the root space of `W[1..c]`, `Z[c..L-1]`, measured
DIRECTLY. `I - P = (I - P_L) ⊗ 1 + P_L ⊗ (I - P_R)`, and each side telescopes into orthogonal
one-site pieces `(1 - W_i W_i') T_i` / `B_j (1 - Z_j' Z_j)` that are formed explicitly, so the
roundoff floor is ~eps. The difference of squares `||psi||^2 - ||P psi||^2` cancels to
~sqrt(eps), the size of a tight `tol_basis`. The rest of `psi` beyond each piece enters through
a square-root factor of its Gram matrix (an SVD's `U S` / `S V'`), so any gauge works.
"""
function _projection_miss(psi::SymMPS, W, Z, c::Int)
    L = length(psi)
    R = Vector{Any}(undef, L)                    # R[k] R[k]' = Gram of psi[k..L] on link_k
    for k = L:-1:2
        T = k == L ? psi[L] : to_concrete!(contract(psi[k], (3,), R[k+1], (1,)))
        res = svd(T, (1,); cutoff = 0.0)
        R[k] = to_concrete(res.U * res.S)
    end
    miss2 = 0.0
    AL = nothing
    for i = 1:c
        T = AL === nothing ? psi[i] : to_concrete!(contract(AL, (2,), psi[i], (1,)))
        miss2 += norm(to_concrete!(contract(perp_component(W[i], T), (3,), R[i+1], (1,))))^2
        AL = to_concrete!(contract(W[i]', (1, 2), T, (1, 2)))         # (b_i, link_{i+1})
    end
    G = Vector{Any}(undef, L - 1)                # G[j]' G[j] = Gram of P_L psi on link_{j+1}
    res = svd(AL, (1,); cutoff = 0.0)
    G[c] = to_concrete(res.S * res.Vd)
    for j = (c+1):(L-1)
        res = svd(to_concrete!(contract(G[j-1], (2,), psi[j], (1,))), (1, 2); cutoff = 0.0)
        G[j] = to_concrete(res.S * res.Vd)
    end
    AR = nothing
    for j = (L-1):-1:c
        B = AR === nothing ? psi[j+1] : to_concrete!(contract(psi[j+1], (3,), AR, (1,)))
        miss2 += norm(to_concrete!(contract(G[j], (2,), perp_component_right(Z[j], B), (1,))))^2
        AR = to_concrete!(contract(B, (2, 3), Z[j]', (2, 3)))         # (link_{j+1}, b_j)
    end
    return sqrt(miss2)
end

"`S = (W ⊗ Z)' psi`, the root coefficients of `P psi` (plan: `project_root`)."
_project_root(psi::SymMPS, fr::KLFrames) = to_concrete!(
    contract(_overlap_left(psi, fr.W, fr.c), (2,), _overlap_right(psi, fr.Z, fr.c), (1,)),
)

"`B = P H P` at the root, from the frames' cached environments (plan: `root_generator`)."
_root_generator(mpo, fr::KLFrames) =
    shift_h(zero_site_h(mpo, fr.c, fr.lenv, fr.renv, fr.W[fr.c], fr.Z[fr.c]), fr.eshift)

_scaled(a::Number, x::Array{<:Number}) = a .* x
_scaled(a::Number, x) = to_concrete(a * x)

"""
    _projected_frontiers(applyB, S0, tau, m) -> [F_1, ..., F_m]

`F_j = (tau^j / j!) B^j S0` by the recursion `F_j = (tau/j) B F_{j-1}`. Generic in the vector
type (root cores here, plain vectors in the dense unit tests).
"""
function _projected_frontiers(applyB, S0, tau::Number, m::Int)
    Fs = Any[]
    F = S0
    for j = 1:m
        F = _scaled(tau / j, applyB(F))
        push!(Fs, F)
    end
    return Fs
end

"""
    _frontier_mps(psi, fr, core) -> SymMPS

The Galerkin vector `(W ⊗ Z) core` as an MPS on `psi`'s link tags, centre at the root -- the
same assembly `cbe_bug_step!` uses for the evolved core.
"""
function _frontier_mps(psi::SymMPS, fr::KLFrames, core)
    L, c = length(psi), fr.c
    T = Vector{Any}(undef, L)
    for i = 1:(c-1)
        T[i] = _relink(fr.W[i], psi[i].inds[1].itags, psi[i].inds[3].itags)
    end
    T[c] = _relink(to_concrete(fr.W[c] * core), psi[c].inds[1].itags, psi[c].inds[3].itags)
    for j = c:(L-1)
        T[j+1] = _relink(fr.Z[j], psi[j+1].inds[1].itags, psi[j+1].inds[3].itags)
    end
    return SymMPS(T, c)
end
