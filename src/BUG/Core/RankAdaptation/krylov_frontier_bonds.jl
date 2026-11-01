# Moved from BUG-Julia@b155550:src/RSVDCBEBondUpdate/bug/krylov_enrich.jl, src/RSVDCBEBondUpdate/bug/krylov_closure.jl, src/RSVDCBEBondUpdate/bug/krylov_closure_loop.jl.
# krylov_frontier_bonds.jl: one frontier bond of a closure sweep -- prescribed-isometry frames, leaks, the bond action, concurrent bond measurements, and the frontier CBE.

"""
    _frame_onto_left(U0, K, right) -> BondFrame

`_frame_from` with the LEFT isometry prescribed: `U0` is the current frame (embedded in the grown
basis of the bond before), which contains the range of the carry `K`, so `U0 S0 V0` is the exact
two-site block. `old_rank` is `U0`'s width -- the side being grown.
"""
function _frame_onto_left(U0, K, right)
    res_r = svd(right, (1,), "bV,L", "bV,R"; cutoff = 0.0)
    V0 = to_concrete(res_r.Vd)
    S0 =
        to_concrete(to_concrete!(contract(U0', (1, 2), K, (1, 2))) * to_concrete(res_r.U * res_r.S))
    return BondFrame(
        U0,
        S0,
        V0,
        K.inds[1],
        K.inds[2],
        K.inds[3],
        right.inds[2],
        right.inds[3],
        leg_dim(U0, 3),
        copy(K.spaces[3]),
    )
end

"Row mirror of [`_frame_onto_left`](@ref): the right isometry `V0` is prescribed."
function _frame_onto_right(left, B, V0)
    res_l = svd(left, (1, 2), "bU,L", "bU,R"; cutoff = 0.0)
    U0 = to_concrete(res_l.U)
    S0 = to_concrete(
        to_concrete(res_l.S * res_l.Vd) * to_concrete!(contract(B, (2, 3), V0', (2, 3))),
    )
    return BondFrame(
        U0,
        S0,
        V0,
        left.inds[1],
        left.inds[2],
        left.inds[3],
        B.inds[2],
        B.inds[3],
        leg_dim(V0, 1),
        copy(left.spaces[3]),
    )
end

"`||(I - U0 U0') X||` for a two-site `X = (link_l, site_l, site_r, link_r)`."
_left_leak(U0, X) = norm(perp_component(U0, X))

"`||X (I - V0' V0)||` for a two-site `X`, the right frame's miss."
function _right_leak(V0, X)
    ov = contract(X, (3, 4), V0', (2, 3))                               # (link_l, site_l, bond)
    return norm(to_concrete(X - to_concrete!(contract(ov, (3,), V0, (1,)))))
end

"""
    _bond_action(expansion, f, mpo, i, lch, rch, U0, V0, side, fold, rng) -> (skl, skr, leak, nh2)

One frontier bond's sketch closures and outside-space action on `side`. With `fold = false` the
exact two-site action `X = H Theta_F` is formed once, the leak is exact, and the closures share `X`
(`nh2 = 1`). With `fold = true` (`fold_omega = true` in `cbe_bug_step!`) `X` is never formed: the
probe is folded into the environment contractions, and the leak is Test B's holdout estimate at a
quarter of the frame width. This is the large-chi path, since `X` lives on the frame width (~2chi)
and the exact action costs far more than the folded probe. The closures and the estimate come from
the basis-selection strategy `expansion`.
"""
function _bond_action(
    expansion::BasisExpansion,
    f::BondFrame,
    mpo,
    i::Int,
    lch,
    rch,
    U0,
    V0,
    side::Symbol,
    fold::Bool,
    rng::AbstractRNG,
)
    if fold
        skl, skr = folded_sketch_closures(expansion, f, mpo, i, lch, rch)
        n = max(8, cld(leg_dim(side === :left ? U0 : V0, side === :left ? 3 : 1), 4))
        return (skl, skr, sampled_leak(expansion, f, skl, skr, side, n, rng), 0)
    end
    X = apply_h_two_site(frame_theta(f), mpo, i, lch, rch)
    skl, skr = sketch_closures(expansion, f, () -> X)
    return (skl, skr, side === :left ? _left_leak(U0, X) : _right_leak(V0, X), 1)
end

"""
    _bond_measurements(Fm, mpo, fr, chain, stack, side; expansion, fold, nsample, rng) -> Vector

A MEASURING half-sweep's bonds, CONCURRENTLY (parallel plan (b)). With the frames fixed, the only
bond-to-bond dependency is the carry (`C = W[i]' K` on the left, `D = B Z[j]'` on the right): a
cheap sequential chain. Each bond's frame, two-site action and leak (~1/3 of a measuring sweep,
`_bond_action`) then depend on nothing else, so they run one task per bond, each with its own RNG
drawn in bond order from `rng`. Entry `i` (a bond of `side`) is `(K, leak, nh2, sleak)`, `K` the
carry the sequential loop would have built there; `sleak = 0` unless `nsample > 0`.
"""
function _bond_measurements(
    Fm::SymMPS,
    mpo,
    fr::KLFrames,
    chain,
    stack,
    side::Symbol;
    expansion::BasisExpansion,
    fold::Bool,
    nsample::Int,
    rng::AbstractRNG,
)
    L, c = length(Fm), fr.c
    bonds = side === :left ? (1:c) : ((L-1):-1:c)
    carry = Vector{Any}(undef, L - 1)
    X = nothing
    for i in bonds
        if side === :left
            carry[i] = X === nothing ? Fm[i] : to_concrete!(contract(X, (2,), Fm[i], (1,)))
            X = to_concrete!(contract(fr.W[i]', (1, 2), carry[i], (1, 2)))
        else
            carry[i] = X === nothing ? Fm[i+1] : to_concrete!(contract(Fm[i+1], (3,), X, (1,)))
            X = to_concrete!(contract(carry[i], (2, 3), fr.Z[i]', (2, 3)))
        end
    end
    rngs = Vector{Any}(undef, L - 1)
    for i in bonds
        rngs[i] = MersenneTwister(rand(rng, UInt64))
    end
    function one(i)
        f, a = if side === :left
            f = _frame_onto_left(fr.W[i], carry[i], Fm[i+1])
            (
                f,
                _bond_action(
                    expansion,
                    f,
                    mpo,
                    i,
                    chain[i],
                    right_channels(stack, i + 2),
                    fr.W[i],
                    nothing,
                    :left,
                    fold,
                    rngs[i],
                ),
            )
        else
            f = _frame_onto_right(Fm[i], carry[i], fr.Z[i])
            (
                f,
                _bond_action(
                    expansion,
                    f,
                    mpo,
                    i,
                    left_channels(stack, i),
                    chain[i],
                    nothing,
                    fr.Z[i],
                    :right,
                    fold,
                    rngs[i],
                ),
            )
        end
        skl, skr, leak, nh2 = a
        sl = nsample > 0 ? sampled_leak(expansion, f, skl, skr, side, nsample, rngs[i]) : 0.0
        return (K = carry[i], leak = leak, nh2 = nh2, sleak = sl)
    end
    tasks = Dict(i => Threads.@spawn(one(i)) for i in bonds)
    out = Vector{Any}(undef, L - 1)
    for i in bonds
        out[i] = fetch(tasks[i])
    end
    return out
end

"How many charge sectors `new` carries on `leg` that `old` did not (plan §3.5)."
function _new_sectors(old, new, leg::Int)
    have = Set(first(s) for s in old.spaces[leg])
    return count(s -> !(first(s) in have), new.spaces[leg])
end

"""
    _frontier_expand(f, mpo, i, lch, rch, side, ckw, ledger, rng) -> CBEExpansion

A frontier bond's CBE (plan T4.1): `side`'s frame grows by [`expand_frame`](@ref) with the
step's strategy `ckw.expansion`, exactly as in the half-sweeps, so the Krylov stage after it
starts from those directions. `ckw` carries `cbe_bug_step!`'s expansion settings.
"""
function _frontier_expand(
    f::BondFrame,
    mpo,
    i::Int,
    lch,
    rch,
    side::Symbol,
    ckw::NamedTuple,
    ledger::AdaptiveLedger,
    rng::AbstractRNG,
)
    return expand_frame(
        ckw.expansion,
        f,
        mpo,
        i,
        lch,
        rch,
        side;
        ledger = ledger,
        rng = rng,
        alloc = get(ckw, :alloc, nothing),
        workcap = get(ckw, :workcap, 0),
        growth = get(ckw, :growth, ADAPTIVE_GROWTH),
        exact = get(ckw, :exact, false),
        version = i,
        cap_order = get(ckw, :cap_order, :sector),
        dd_width = get(ckw, :dd_width, 0),
        dd_policy = get(ckw, :dd_policy, :adaptive),
        dd_memory = get(ckw, :dd_memory, nothing),
    )
end

# Tag every bond entry a frontier sweep logged, after `_log_bond!` wrote it.
_tag_last!(lg::AdaptiveLedger, extra::NamedTuple) =
    (lg.bond_log[end] = merge(lg.bond_log[end], extra); lg)
