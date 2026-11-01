# Moved from BUG-Julia@b155550:src/RSVDCBEBondUpdate/bug/krylov_enrich.jl,
# src/RSVDCBEBondUpdate/bug/krylov_closure.jl, src/RSVDCBEBondUpdate/bug/krylov_closure_loop.jl.
# krylov_frontier_bonds.jl: one frontier bond of a closure sweep -- prescribed-isometry frames,
# leaks, the bond action, concurrent bond measurements, and the frontier CBE.

"""
    _frame_onto_left(left_isometry, carry, right) -> BondFrame

`_frame_from` with the LEFT isometry prescribed: `left_isometry` is the current frame (embedded in
the grown basis of the bond before), which contains the range of `carry`, so the frame's
`U0 S0 V0` (`left_isometry`, bond matrix, right isometry) is the exact two-site block.
`old_rank` is `left_isometry`'s width -- the side being grown.
"""
function _frame_onto_left(left_isometry, carry, right)
    res_r = svd(right, (1,), "bV,L", "bV,R"; cutoff = 0.0)
    right_isometry = to_concrete(res_r.Vd)
    # The carry seen through the prescribed left isometry, times the right site's `U S`.
    bond_matrix = to_concrete(
        to_concrete!(contract(left_isometry', (1, 2), carry, (1, 2))) *
        to_concrete(res_r.U * res_r.S),
    )
    return BondFrame(
        left_isometry,
        bond_matrix,
        right_isometry,
        carry.inds[1],
        carry.inds[2],
        carry.inds[3],
        right.inds[2],
        right.inds[3],
        leg_dim(left_isometry, 3),
        copy(carry.spaces[3]),
    )
end

"Row mirror of [`_frame_onto_left`](@ref): the right isometry `right_isometry` is prescribed."
function _frame_onto_right(left, carry, right_isometry)
    res_l = svd(left, (1, 2), "bU,L", "bU,R"; cutoff = 0.0)
    left_isometry = to_concrete(res_l.U)
    bond_matrix = to_concrete(
        to_concrete(res_l.S * res_l.Vd) *
        to_concrete!(contract(carry, (2, 3), right_isometry', (2, 3))),
    )
    return BondFrame(
        left_isometry,
        bond_matrix,
        right_isometry,
        left.inds[1],
        left.inds[2],
        left.inds[3],
        carry.inds[2],
        carry.inds[3],
        leg_dim(right_isometry, 1),
        copy(left.spaces[3]),
    )
end

"""
`||(I - U0 U0') X||` for `U0 = left_isometry` and a two-site
`X = two_site = (link_l, site_l, site_r, link_r)`.
"""
_left_leak(left_isometry, two_site) = norm(perp_component(left_isometry, two_site))

"""
`||X (I - V0' V0)||` for `V0 = right_isometry` and a two-site `X = two_site`, the right frame's
miss.
"""
function _right_leak(right_isometry, two_site)
    ov = contract(two_site, (3, 4), right_isometry', (2, 3))           # (link_l, site_l, bond)
    return norm(to_concrete(two_site - to_concrete!(contract(ov, (3,), right_isometry, (1,)))))
end

"""
    _bond_action(expansion, frame, mpo, bond, lch, rch, left_isometry, right_isometry, side, fold,
                 rng) -> (skl, skr, leak, nh2)

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
    frame::BondFrame,
    mpo,
    bond::Int,
    lch,
    rch,
    left_isometry,
    right_isometry,
    side::Symbol,
    fold::Bool,
    rng::AbstractRNG,
)
    if fold
        skl, skr = folded_sketch_closures(expansion, frame, mpo, bond, lch, rch)
        # Holdout width: a quarter of the grown side's frame width, at least 8.
        grown_width = leg_dim(
            side === :left ? left_isometry : right_isometry,
            side === :left ? 3 : 1,
        )
        sample_count = max(8, cld(grown_width, 4))
        leak = sampled_leak(expansion, frame, skl, skr, side, sample_count, rng)
        return (skl, skr, leak, 0)
    end
    action = apply_h_two_site(frame_theta(frame), mpo, bond, lch, rch)
    skl, skr = sketch_closures(expansion, frame, () -> action)
    leak = side === :left ? _left_leak(left_isometry, action) : _right_leak(right_isometry, action)
    return (skl, skr, leak, 1)
end

"""
    _bond_measurements(Fm, mpo, fr, chain, stack, side; expansion, fold, nsample, rng) -> Vector

A MEASURING half-sweep's bonds, CONCURRENTLY (parallel plan (b)). With the frames fixed, the only
bond-to-bond dependency is the carry (`C = W[i]' K` on the left, `D = B Z[j]'` on the right): a
cheap sequential chain. Each bond's frame, two-site action and leak (~1/3 of a measuring sweep,
`_bond_action`) then depend on nothing else, so they run one task per bond, each with its own RNG
drawn in bond order from `rng`. Entry `bond` (a bond of `side`) is `(K, leak, nh2, sleak)`, `K`
the carry the sequential loop would have built there; `sleak = 0` unless `nsample > 0`.
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
    site_count, root_link = length(Fm), fr.root_link
    bonds = side === :left ? (1:root_link) : ((site_count-1):-1:root_link)
    # The sequential chain: `carry[bond]` is `Fm`'s tensor at the bond with the sites already
    # passed contracted onto their frames (`projected`, the running overlap).
    carry = Vector{Any}(undef, site_count - 1)
    projected = nothing
    for bond in bonds
        if side === :left
            carry[bond] =
                projected === nothing ? Fm[bond] :
                to_concrete!(contract(projected, (2,), Fm[bond], (1,)))
            projected = to_concrete!(contract(fr.left_frames[bond]', (1, 2), carry[bond], (1, 2)))
        else
            carry[bond] =
                projected === nothing ? Fm[bond+1] :
                to_concrete!(contract(Fm[bond+1], (3,), projected, (1,)))
            projected = to_concrete!(contract(carry[bond], (2, 3), fr.right_frames[bond]', (2, 3)))
        end
    end
    rngs = Vector{Any}(undef, site_count - 1)
    for bond in bonds
        rngs[bond] = MersenneTwister(rand(rng, UInt64))
    end
    # One bond's measurement, independent of every other bond once the carries exist.
    function one(bond)
        frame, action_result = if side === :left
            frame = _frame_onto_left(fr.left_frames[bond], carry[bond], Fm[bond+1])
            (
                frame,
                _bond_action(
                    expansion,
                    frame,
                    mpo,
                    bond,
                    chain[bond],
                    right_channels(stack, bond + 2),
                    fr.left_frames[bond],
                    nothing,
                    :left,
                    fold,
                    rngs[bond],
                ),
            )
        else
            frame = _frame_onto_right(Fm[bond], carry[bond], fr.right_frames[bond])
            (
                frame,
                _bond_action(
                    expansion,
                    frame,
                    mpo,
                    bond,
                    left_channels(stack, bond),
                    chain[bond],
                    nothing,
                    fr.right_frames[bond],
                    :right,
                    fold,
                    rngs[bond],
                ),
            )
        end
        skl, skr, leak, nh2 = action_result
        sl =
            nsample > 0 ? sampled_leak(expansion, frame, skl, skr, side, nsample, rngs[bond]) :
            0.0
        return (site_core = carry[bond], leak = leak, nh2 = nh2, sleak = sl)
    end
    tasks = Dict(bond => Threads.@spawn(one(bond)) for bond in bonds)
    out = Vector{Any}(undef, site_count - 1)
    for bond in bonds
        out[bond] = fetch(tasks[bond])
    end
    return out
end

"How many charge sectors `new` carries on `leg` that `old` did not (plan §3.5)."
function _new_sectors(old, new, leg::Int)
    have = Set(first(sector) for sector in old.spaces[leg])
    return count(sector -> !(first(sector) in have), new.spaces[leg])
end

"""
    _frontier_expand(frame, mpo, bond, lch, rch, side, ckw, ledger, rng) -> CBEExpansion

A frontier bond's CBE (plan T4.1): `side`'s frame grows by [`expand_frame`](@ref) with the
step's strategy `ckw.expansion`, exactly as in the half-sweeps, so the Krylov stage after it
starts from those directions. `ckw` carries `cbe_bug_step!`'s expansion settings.
"""
function _frontier_expand(
    frame::BondFrame,
    mpo,
    bond::Int,
    lch,
    rch,
    side::Symbol,
    ckw::NamedTuple,
    ledger::AdaptiveLedger,
    rng::AbstractRNG,
)
    return expand_frame(
        ckw.expansion,
        frame,
        mpo,
        bond,
        lch,
        rch,
        side;
        ledger = ledger,
        rng = rng,
        alloc = get(ckw, :alloc, nothing),
        workcap = get(ckw, :workcap, 0),
        growth = get(ckw, :growth, ADAPTIVE_GROWTH),
        exact = get(ckw, :exact, false),
        version = bond,
        cap_order = get(ckw, :cap_order, :sector),
        dd_width = get(ckw, :dd_width, 0),
        dd_policy = get(ckw, :dd_policy, :adaptive),
        dd_memory = get(ckw, :dd_memory, nothing),
    )
end

# Tag every bond entry a frontier sweep logged, after `_log_bond!` wrote it.
_tag_last!(lg::AdaptiveLedger, extra::NamedTuple) =
    (lg.bond_log[end] = merge(lg.bond_log[end], extra); lg)
