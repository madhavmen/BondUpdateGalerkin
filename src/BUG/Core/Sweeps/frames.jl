# Canonical two-site bond snapshot, the input to the K/L/S kernel. Telum has no QR or LQ, so SVD
# stands in: the U of svd(A) is the QR's left isometry, and S*Vd its R factor. These are exact,
# truncation-free moves, so `cutoff=0.0` is passed explicitly; all rank adaptation belongs to
# the K/L/S step.

"""
    BondFrame

The canonical snapshot at bond `(bond, bond+1)`, with `theta == U0 * S0 * V0` exactly (no
truncation).
`U0` is a left isometry with legs `(link_l, site_l, bond_u)`, `V0` a right isometry with legs
`(bond_v, site_r, link_r)`, and `S0` the `(bond_u, bond_v)` centre.

`link_*` / `site_*` are `TLIndex` handles. A `TLIndex` carries no dimension, which lives in the
tensor's `spaces`, so `old_rank` and `link_mid_space` are stored separately.
"""
struct BondFrame
    U0::Any
    S0::Any
    V0::Any
    link_l::TLIndex
    site_l::TLIndex
    link_mid::TLIndex
    site_r::TLIndex
    link_r::TLIndex
    old_rank::Int
    link_mid_space::Vector
end

"""
    bond_frame(psi, bond) -> BondFrame

Snapshot bond `(bond, bond+1)`. Requires `psi.center == bond`.

The two SVDs are given **distinct** tag pairs. With Telum's default tags both
`S` factors would emit a `'+'` leg tagged `"svdL"`, and contracting them would
produce a tensor carrying two identical `TLIndex`es -- Telum rejects that with
"Duplicate TLIndex with non-empty itag".
"""
function bond_frame(psi::SymMPS, bond::Int)
    1 <= bond < length(psi) || throw(BoundsError(psi, bond))
    psi.center == bond || throw(
        ArgumentError(
            "bond_frame requires the orthogonality centre on site $bond, got $(psi.center)",
        ),
    )

    left = psi[bond]      # (link_l, site_l, link_mid)
    right = psi[bond+1]   # (link_mid, site_r, link_r)

    res_l = svd(left, (1, 2), "bU,L", "bU,R"; cutoff = 0.0)
    res_r = svd(right, (1,), "bV,L", "bV,R"; cutoff = 0.0)

    left_isometry = to_concrete(res_l.U)     # (link_l, site_l, bU,L)   -- the frame's U0
    right_isometry = to_concrete(res_r.Vd)   # (bV,R, site_r, link_r)   -- the frame's V0

    s_left = res_l.S * res_l.Vd          # (bU,L, link_mid)
    s_right = res_r.U * res_r.S          # (link_mid, bV,R)
    centre = to_concrete(s_left * s_right)   # (bU,L, bV,R)  -- the frame's S0

    return BondFrame(
        left_isometry,
        centre,
        right_isometry,
        left.inds[1],
        left.inds[2],
        left.inds[3],
        right.inds[2],
        right.inds[3],
        leg_dim(left_isometry, 3),
        copy(left.spaces[3]),
    )
end

"""
    frame_theta(frame) -> TLArray

Reassemble the two-site block `U0 * S0 * V0`, legs
`(link_l, site_l, site_r, link_r)`.
"""
frame_theta(frame::BondFrame) = to_concrete((frame.U0 * frame.S0) * frame.V0)

"""
    two_site_block(psi, bond) -> TLArray

The uncontracted two-site block `psi[bond] * psi[bond+1]`, same leg order as
`frame_theta`. Used to check the snapshot is exact.
"""
two_site_block(psi::SymMPS, bond::Int) = to_concrete(psi[bond] * psi[bond+1])
