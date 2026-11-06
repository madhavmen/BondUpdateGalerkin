# The K/L frames of the order-p closure, their shared environment
# caches, and the projections of the state and its frontiers onto them.

# ── GENERAL-ORDER KRYLOV CLOSURE (order-p condition) ──
# The order-p error is O(|tau|^(p+1)) once `(I - P) A B^j psi = 0` for j < p, with B = P A P.
# Frontiers are judged by `(I - P) A F_j`, not by a root residual (which concerns B, not A).

"A frontier below this fraction of the state norm is the zero vector (e.g. `H psi = 0`)."
const _FRONTIER_FLOOR = 1e-14

"Internal bound on closure passes. Reaching it with the frames still growing is `:closure`."
_closure_maxpass(order::Int) = 2 * order + 2

# Each enriching frontier sweep re-tags its frames with a new generation, so a frame and the
# frame it grew from never share a tag inside one overlap tensor. Returns the link tag of left
# frame `site` / right frame `bond` in `generation` (generation 0 is the half-sweeps' own tag).
_tagWg(site::Int, generation::Int) = generation == 0 ? _tagW(site) : "cbeW$generation,$site"
_tagZg(bond::Int, generation::Int) = generation == 0 ? _tagZ(bond) : "cbeZ$generation,$bond"

"""
    KLFrames(left_frames, right_frames, root_link, lenv, renv, gen)

The K/L frames of one step and the root environments they imply, writing `c` for `root_link`:
`left_frames[1..c]` the left frames, `right_frames[c..L-1]` the right frames (`right_frames[j]`
absorbs site `j+1`); `lenv` is the environment at link `c` through `left_frames[1..c-1]`, `renv`
the one at link `c+2` through `right_frames[c+1..L-1]`. `gen` counts the enrichments, so a [`KrylovFrontier`](@ref) built in older
frames is refused rather than silently misread.

`lcache` / `rcache` hold the environments every frontier sweep of ONE generation shares
([`_side_cache!`](@ref)); each half-sweep writes only its own, so the parallel halves never
write the same field.
"""
mutable struct KLFrames
    left_frames::Vector{Any}
    right_frames::Vector{Any}
    root_link::Int
    lenv::Any
    renv::Any
    gen::Int
    lcache::Any
    rcache::Any
    # The generators' energy shift (`shift_h`): the root `B - eshift` and the frontier sweeps'
    # one-site Krylov operators. `0.0` is the unshifted step.
    eshift::Float64
end

# Frames with no cached environments yet and no energy shift.
KLFrames(left_frames, right_frames, root_link, lenv, renv, gen) =
    KLFrames(left_frames, right_frames, root_link, lenv, renv, gen, nothing, nothing, 0.0)

"""
Threads below which the closure keeps its frontier sweeps in sequence. Each sweep already runs
its two halves concurrently under `parallel = true`, so two concurrent sweeps need four threads
to gain anything; below that the 2-thread production path stays exactly what it was.
"""
const _FRONTIER_THREADS = 4

# Whether frontier sweeps may run concurrently: requested AND enough threads to gain from it.
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
    right_task = Threads.@spawn _side_cache!(fr, Fm, mpo, :right)
    _side_cache!(fr, Fm, mpo, :left)
    fetch(right_task)
    return fr
end

"""
    _side_cache!(fr, Fm, mpo, side) -> (chain, stack)

The environments of one half-sweep that do not depend on the frontier, built once per frame
generation. Every frontier of a generation is `(W ⊗ Z) core` with only the core different, so
re-gauging `Fm` to site 1 (left half) changes sites `1..c` only: the right stack on links
`c+1..L+1` and the frame chain `<W[1..i-1]| H |W[1..i-1]>` (the non-enriching sweep's `lch`)
are the same for every frontier. Rebuilding them per sweep was ~1/3 of a measuring sweep
(bug4 10x4 D=256 profile, job 17283685). `chain[bond]` is the left channels at `bond = 1..c`,
`stack` the right-stack entries on links `>= c+1`; the right half mirrors both (`chain[bond]`
the right channels at `bond = c..L-1`, `stack` the left-stack entries on links `<= c`).
"""
function _side_cache!(fr::KLFrames, Fm::SymMPS, mpo, side::Symbol)
    site_count, root_link = length(Fm), fr.root_link
    cur = side === :left ? fr.lcache : fr.rcache
    valid = cur !== nothing && cur.gen == fr.gen
    chain = valid ? cur.chain : nothing          # an enriching sweep seeds the chain only
    env_stack = valid ? cur.stack : nothing
    (chain !== nothing && env_stack !== nothing) && return (chain, env_stack)
    if chain === nothing
        chain = Vector{Any}(undef, site_count - 1)
        ch = boundary_channels(mpo)
        if side === :left
            for bond = 1:root_link
                chain[bond] = ch
                bond < root_link && (ch = push_left_channels(ch, mpo, fr.left_frames[bond], bond))
            end
        else
            for bond = (site_count-1):-1:root_link
                chain[bond] = ch
                bond > root_link &&
                    (ch = push_right_channels(ch, mpo, fr.right_frames[bond], bond + 1))
            end
        end
    end
    # The opposite side's MPO environments of `Fm` itself: entry `link` is the environment on
    # that link, `missing` where the half-sweep never reads it.
    env_stack = Any[missing for _ = 1:(site_count+1)]
    env = nothing
    if side === :left
        env_stack[site_count+1] = nothing
        for site = site_count:-1:(root_link+1)
            env = _mpo_right_step(env, Fm[site], mpo[site])
            env_stack[site] = env
        end
        fr.lcache = (gen = fr.gen, chain = chain, stack = env_stack)
    else
        env_stack[1] = nothing
        for site = 1:(root_link-1)
            env = _mpo_left_step(env, Fm[site], mpo[site])
            env_stack[site+1] = env
        end
        fr.rcache = (gen = fr.gen, chain = chain, stack = env_stack)
    end
    return (chain, env_stack)
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
    frames_from_state(psi, mpo, root_link) -> KLFrames

The state's OWN Schmidt frames at root bond `root_link` -- the subspace every step starts from,
before any expansion. Used to start a closure from scratch (`include_zero = true`) and by the
tests.
"""
function frames_from_state(psi::SymMPS, mpo, root_link::Int)
    site_count = length(psi)
    1 <= root_link <= site_count - 1 ||
        throw(ArgumentError("root must be a bond in 1:$(site_count - 1), got $root_link"))
    # Left frames: the left-canonical sites, closed at the root by the SVD's left isometry.
    left_gauged = copy(psi)
    canonical!(left_gauged, root_link)
    left_frames = Vector{Any}(undef, root_link)
    for site = 1:(root_link-1)
        left_frames[site] = left_gauged[site]
    end
    left_frames[root_link] = to_concrete(svd(left_gauged[root_link], (1, 2); cutoff = 0.0).U)
    # Right frames: the right-canonical sites, opened at the root by the SVD's right isometry.
    right_gauged = copy(psi)
    canonical!(right_gauged, root_link + 1)
    right_frames = Vector{Any}(undef, site_count - 1)
    right_frames[root_link] = to_concrete(svd(right_gauged[root_link+1], (1,); cutoff = 0.0).Vd)
    for bond = (root_link+1):(site_count-1)
        right_frames[bond] = right_gauged[bond+1]
    end
    # Frame tags on both ends of every frame link, as the half-sweeps write them.
    for site = 1:root_link
        left_frames[site] = to_concrete(setitag(left_frames[site], 3, _tagW(site)))
        site < root_link &&
            (left_frames[site+1] = to_concrete(setitag(left_frames[site+1], 1, _tagW(site))))
    end
    for bond = root_link:(site_count-1)
        right_frames[bond] = to_concrete(setitag(right_frames[bond], 1, _tagZ(bond)))
        bond > root_link &&
            (right_frames[bond-1] = to_concrete(setitag(right_frames[bond-1], 3, _tagZ(bond))))
    end
    lenv = boundary_channels(mpo)
    for site = 1:(root_link-1)
        lenv = push_left_channels(lenv, mpo, left_frames[site], site)
    end
    renv = boundary_channels(mpo)
    for bond = (site_count-1):-1:(root_link+1)
        renv = push_right_channels(renv, mpo, right_frames[bond], bond + 1)
    end
    return KLFrames(left_frames, right_frames, root_link, lenv, renv, 0)
end

"""
`<left_frames[1..root_link] | psi>` on the left of the root: `(b_L, link_{root_link+1})`. A full
contraction, any gauge.
"""
function _overlap_left(psi::SymMPS, left_frames, root_link::Int)
    AL = nothing
    for site = 1:root_link
        carried = AL === nothing ? psi[site] : to_concrete!(contract(AL, (2,), psi[site], (1,)))
        # (b_L, link_{site+1})
        AL = to_concrete!(contract(left_frames[site]', (1, 2), carried, (1, 2)))
    end
    return AL
end

"""
`<right_frames[root_link..L-1] | psi>` on the right of the root: `(link_{root_link+1}, b_R)`. A
full contraction, any gauge.
"""
function _overlap_right(psi::SymMPS, right_frames, root_link::Int)
    AR = nothing
    for bond = (length(psi)-1):-1:root_link
        carried = AR === nothing ? psi[bond+1] : to_concrete!(contract(psi[bond+1], (3,), AR, (1,)))
        # (link_bond, b_bond)
        AR = to_concrete!(contract(carried, (2, 3), right_frames[bond]', (2, 3)))
    end
    return AR
end

"""
    _projection_miss(psi, left_frames, right_frames, root_link) -> Float64

`||(I - P) psi||` for `P` the projector onto the root space of `left_frames[1..root_link]`,
`right_frames[root_link..L-1]`, measured DIRECTLY. `I - P = (I - P_L) ⊗ 1 + P_L ⊗ (I - P_R)`, and
each side telescopes into orthogonal one-site pieces `(1 - W_i W_i') T_i` / `B_j (1 - Z_j' Z_j)`
that are formed explicitly, so the roundoff floor is ~eps. The difference of squares
`||psi||^2 - ||P psi||^2` cancels to ~sqrt(eps), the size of a tight `tol_basis`. The rest of `psi`
beyond each piece enters through a square-root factor of its Gram matrix (an SVD's `U S` /
`S V'`), so any gauge works.
"""
function _projection_miss(psi::SymMPS, left_frames, right_frames, root_link::Int)
    site_count = length(psi)
    # right_factor[site] right_factor[site]' = Gram of psi[site..L] on link_site
    right_factor = Vector{Any}(undef, site_count)
    for site = site_count:-1:2
        carried =
            site == site_count ? psi[site_count] :
            to_concrete!(contract(psi[site], (3,), right_factor[site+1], (1,)))
        res = svd(carried, (1,); cutoff = 0.0)
        right_factor[site] = to_concrete(res.U * res.S)
    end
    miss2 = 0.0
    # Left side: the part of each carried site outside its left frame, weighted by the rest.
    AL = nothing
    for site = 1:root_link
        carried = AL === nothing ? psi[site] : to_concrete!(contract(AL, (2,), psi[site], (1,)))
        outside = perp_component(left_frames[site], carried)
        miss2 += norm(to_concrete!(contract(outside, (3,), right_factor[site+1], (1,))))^2
        # (b_site, link_{site+1})
        AL = to_concrete!(contract(left_frames[site]', (1, 2), carried, (1, 2)))
    end
    # left_factor[bond]' left_factor[bond] = Gram of P_L psi on link_{bond+1}
    left_factor = Vector{Any}(undef, site_count - 1)
    res = svd(AL, (1,); cutoff = 0.0)
    left_factor[root_link] = to_concrete(res.S * res.Vd)
    for bond = (root_link+1):(site_count-1)
        res = svd(
            to_concrete!(contract(left_factor[bond-1], (2,), psi[bond], (1,))),
            (1, 2);
            cutoff = 0.0,
        )
        left_factor[bond] = to_concrete(res.S * res.Vd)
    end
    # Right side, inside the left frames: each carried site's part outside its right frame.
    AR = nothing
    for bond = (site_count-1):-1:root_link
        carried = AR === nothing ? psi[bond+1] : to_concrete!(contract(psi[bond+1], (3,), AR, (1,)))
        outside = perp_component_right(right_frames[bond], carried)
        miss2 += norm(to_concrete!(contract(left_factor[bond], (2,), outside, (1,))))^2
        # (link_{bond+1}, b_bond)
        AR = to_concrete!(contract(carried, (2, 3), right_frames[bond]', (2, 3)))
    end
    return sqrt(miss2)
end

"`S = (W ⊗ Z)' psi`, the root coefficients of `P psi`."
_project_root(psi::SymMPS, fr::KLFrames) = to_concrete!(
    contract(
        _overlap_left(psi, fr.left_frames, fr.root_link),
        (2,),
        _overlap_right(psi, fr.right_frames, fr.root_link),
        (1,),
    ),
)

"`B = P H P` at the root, from the frames' cached environments."
_root_generator(mpo, fr::KLFrames) = shift_h(
    zero_site_h(
        mpo,
        fr.root_link,
        fr.lenv,
        fr.renv,
        fr.left_frames[fr.root_link],
        fr.right_frames[fr.root_link],
    ),
    fr.eshift,
)

# `factor * coefficients`: broadcast for the dense test vectors, a concrete tensor otherwise.
_scaled(factor::Number, coefficients::Array{<:Number}) = factor .* coefficients
_scaled(factor::Number, coefficients) = to_concrete(factor * coefficients)

"""
    _projected_frontiers(apply_generator, root_start, tau, frontier_count) -> Vector

`F_j = (tau^j / j!) B^j root_start` for `j = 1..frontier_count`, by the recursion
`F_j = (tau/j) B F_{j-1}` with `B` applied by `apply_generator`. Generic in the vector type (root
cores here, plain vectors in the dense unit tests).
"""
function _projected_frontiers(apply_generator, root_start, tau::Number, frontier_count::Int)
    Fs = Any[]
    frontier = root_start
    for degree = 1:frontier_count
        frontier = _scaled(tau / degree, apply_generator(frontier))
        push!(Fs, frontier)
    end
    return Fs
end

"""
    _frontier_mps(psi, fr, core) -> SymMPS

The Galerkin vector `(W ⊗ Z) core` as an MPS on `psi`'s link tags, centre at the root -- the
same assembly `cbe_bug_step!` uses for the evolved core.
"""
function _frontier_mps(psi::SymMPS, fr::KLFrames, core)
    site_count, root_link = length(psi), fr.root_link
    site_tensors = Vector{Any}(undef, site_count)
    for site = 1:(root_link-1)
        site_tensors[site] =
            _relink(fr.left_frames[site], psi[site].inds[1].itags, psi[site].inds[3].itags)
    end
    # The root core sits on the last left frame.
    site_tensors[root_link] = _relink(
        to_concrete(fr.left_frames[root_link] * core),
        psi[root_link].inds[1].itags,
        psi[root_link].inds[3].itags,
    )
    for bond = root_link:(site_count-1)
        site_tensors[bond+1] =
            _relink(fr.right_frames[bond], psi[bond+1].inds[1].itags, psi[bond+1].inds[3].itags)
    end
    return SymMPS(site_tensors, root_link)
end
