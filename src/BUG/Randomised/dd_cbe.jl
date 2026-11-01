# Moved from BUG-Julia@b155550:src/RSVDCBEBondUpdate/expansion/dd_cbe.jl, src/RSVDCBEBondUpdate/expansion/dd_cbe_controller.jl.

# DD-QR CBE kernels: the sketch operators, discarded-space probes and the sectorwise thin QR
# (controller in `dd_cbe_controller.jl`, the bond expansion in `dd_cbe_expand.jl`).

# ── from src/RSVDCBEBondUpdate/expansion/dd_cbe.jl ──

# ── implicit DD-projected, MPO-streamed, QR-only CBE ──
# The frame grows by Om~ = P^D Om, Z = X Om~ streamed through the MPO, and Y = Q R: neither X nor the
# DD projector is formed. One side per call, since the two QRs belong to the two half-sweep orientations.

# ── one operator-action interface ──
# Each operator is immutable and built per bond visit, so a stale one cannot see a later gauge.
# Outputs go to a caller-held `Ref`, since Telum contractions allocate their result.

"""
    TwoSiteSketchOp(f, mpo, i, lenv, renv; adjoint = false, version = 0, counters)

The two-site effective Hamiltonian at bond `(i, i+1)` as a THIN-PROBE action.
`adjoint = false` is `Z = X Om` for a probe in `V0`'s layout `(g, site_r, link_r)`, returning
`(link_l, site_l, g)` (the K-sweep); `adjoint = true` is `X' Om` for a probe in `U0`'s layout,
returned as rows `(g, site_r, link_r)` (the L-sweep). The MPO channels are summed inside the
contraction chain against the SAME probe.
"""
struct TwoSiteSketchOp
    mpo::MPO
    bond::Int
    lenv::MPOLink
    renv::MPOLink
    U0::Any
    V0::Any
    core::Any            # U0*S0 (adjoint = false) or S0*V0 (adjoint = true)
    adjoint::Bool
    symmetry::Symbol
    version::Int
    counters::KernelCounters
end

function TwoSiteSketchOp(
    f::BondFrame,
    mpo::MPO,
    i::Int,
    lenv::MPOLink,
    renv::MPOLink;
    adjoint::Bool = false,
    version::Int = 0,
    counters::KernelCounters = KernelCounters(),
)
    core = adjoint ? to_concrete(f.S0 * f.V0) : to_concrete(f.U0 * f.S0)
    return TwoSiteSketchOp(
        mpo,
        i,
        lenv,
        renv,
        f.U0,
        f.V0,
        core,
        adjoint,
        symmetry_mode(),
        version,
        counters,
    )
end

"""
    apply_two_site_sketch!(Y::Ref, Om, op::TwoSiteSketchOp) -> Y

`Y[] = X Om` (or `X' Om` for `op.adjoint`), unprojected, streamed through the MPO channels.
"""
function apply_two_site_sketch!(Y::Base.RefValue, Om, op::TwoSiteSketchOp)
    op.symmetry === symmetry_mode() || throw(
        ErrorException(
            "TwoSiteSketchOp was built under $(op.symmetry), the active symmetry is $(symmetry_mode())",
        ),
    )
    Z, dt, db = _measure(
        () ->
            op.adjoint ? _stream_right(op.U0, op.core, op.mpo, op.bond, op.lenv, op.renv, Om) :
            _stream_left(op.core, op.V0, op.mpo, op.bond, op.lenv, op.renv, Om),
    )
    kc = op.counters
    kc.n_stream += 1
    kc.n_stream_cols += leg_dim(Om, op.adjoint ? 3 : 1)
    kc.t_stream += dt
    kc.b_stream += db
    Y[] = Z
    return Y
end

"`Y[] = H1 X` through the shared interface."
apply_one_site!(Y::Base.RefValue, X, op::OneSiteOp) = (Y[] = apply_one_site(op, X); Y)

"""
    RootOp(h, version, counters)

The root (zero-site) generator with counted applications; `apply_zero_site(op, S)` forwards.
"""
struct RootOp{H}
    h::H
    version::Int
    counters::KernelCounters
end

function apply_zero_site(op::RootOp, S)
    y, dt, db = _measure(() -> apply_zero_site(op.h, S))
    op.counters.n_root += 1
    op.counters.t_root += dt
    op.counters.b_root += db
    return y
end

"`Y[] = H0 S` through the shared interface."
apply_root!(Y::Base.RefValue, S, op::RootOp) = (Y[] = apply_zero_site(op, S); Y)

"""
    dd_reference_sketch(f, mpo, i, lenv, renv, Om, side) -> TLArray

TEST/DEBUG ONLY (plan T2.5 "full-DD exact tiny reference"): `P_L^D X P_R^D Om` (`side = :left`)
or `P_R^D X' P_L^D Om` (`:right`) through the MATERIALISED `H Theta` and the project-first
closures of `_sketch_closures`. Never on the production path.
"""
function dd_reference_sketch(
    f::BondFrame,
    mpo::MPO,
    i::Int,
    lenv::MPOLink,
    renv::MPOLink,
    Om,
    side::Symbol,
)
    skl, skr = _sketch_closures(f, () -> apply_h_two_site(frame_theta(f), mpo, i, lenv, renv))
    return side === :left ? skl(perp_component_right(f.V0, Om)) : skr(perp_component(f.U0, Om))
end

# ── T2.3: probes in the discarded space only ──────────────────────────────────────────────

"""
    _dd_probe(frame, side, w; rng, boost, exact, kc) -> P^D Om, or nothing

A sector-graded Gaussian probe in `frame`'s layout, drawn only in the discarded space: columns are
allotted by each sector's discarded dimension, then the frame is projected out. `exact = true`
projects the complete fused basis instead. Returns `nothing` when there is no discarded space.

Unlike `sector_graded_sketch(...; comp_ratio = 1.0)`, a short complement is not topped up from the
kept space.
"""
function _dd_probe(
    frame,
    side::Symbol,
    w::Int;
    rng::AbstractRNG,
    boost = nothing,
    exact::Bool = false,
    kc::KernelCounters = KernelCounters(),
)
    a, b = side === :left ? (1, 2) : (2, 3)
    F = fusion_basis(frame, a, b; tag = "cbeG")
    dfull, dT = _frame_sector_dims(frame, side, F, "_dd_probe")
    dC = Dict(q => max(dfull[q] - dT[q], 0) for q in keys(dfull))
    sum(values(dC); init = 0) == 0 && return nothing
    R = if exact
        F
    else
        cols = _alloc_columns(dC, w)
        boost === nothing || _boost!(cols, boost, dC)
        _trim_per_sector(_randomize_like(F, rng), 3, cols)
    end
    R === nothing && return nothing
    Om = side === :left ? R : to_concrete(permutedims(R, (3, 1, 2)))
    perp = side === :left ? perp_component : perp_component_right
    P, dt, db = _measure(() -> perp(frame, Om))
    kc.t_input_dd += dt
    kc.b_input_dd += db
    return norm(P) > 0 ? P : nothing
end

# ── T2.4: sectorwise thin Householder QR ──────────────────────────────────────────────────

const _TAG_DD_L, _TAG_DD_R = "cbeQ,L", "cbeQ,R"

"Block `rmt` as a `(rows, d_fl)` matrix: every leg but `fl` fused into rows, `fl` the columns."
function _block_matrix(rmt::AbstractArray, perm, rows::Int, dfl::Int)
    return reshape(permutedims(rmt, perm), rows, dfl)
end

"""
    _thin_qr(Y, fl, ref, tag, thr) -> Q or nothing

Orthonormal basis of the span of `Y`'s leg-`fl` "columns" (fl = 3: columns of the left
matricisation; fl = 1: rows of the right one), by column-pivoted Householder QR per charge
sector of leg `fl`. A direction is kept when its pivot `|R_kk|` exceeds `thr` -- an absolute,
machine-precision-scaled linear-dependence floor, not a physics cutoff; nothing is ranked.

The new leg is built in `ref`'s arrow convention (`align_charge`), so `[frame | Q]` direct-sums
directly. Abelian symmetries only (the block of a sector is then a plain matrix);
callers use `_thin_orth_nonabelian` otherwise.
"""
function _thin_qr(Y, fl::Int, ref::TLIndex, tag::AbstractString, thr::Float64)
    QD = length(Y.inds)
    syms = symm(Y)
    ydir = Y.inds[fl].dir
    others = Tuple(l for l = 1:QD if l != fl)
    groups = Dict{Any,Vector{Int}}()
    order = Any[]
    for k in eachindex(Y.qlabels)
        Y.isdefined[k] || continue
        q = Y.qlabels[k][fl]
        haskey(groups, q) || (groups[q] = Int[]; push!(order, q))
        push!(groups[q], k)
    end
    newq = similar(Y.qlabels, 0)
    newR = similar(Y.RMTs, 0)
    keep = Int[]
    space = similar(Y.spaces[fl], 0)
    for q in order
        ks = groups[q]
        rmt1 = Y.RMTs[ks[1]]
        nd = ndims(rmt1)
        all(size(rmt1, l) == 1 for l = (QD+1):nd) || throw(
            ArgumentError(
                "_thin_qr: sector multiplicity axes are not trivial -- non-abelian blocks",
            ),
        )
        perm = (others..., fl, ((QD+1):nd)...)
        dfl = size(rmt1, fl)
        mats = [
            _block_matrix(Y.RMTs[k], perm, prod(size(Y.RMTs[k], l) for l in others), dfl) for
            k in ks
        ]
        M = reduce(vcat, mats)
        size(M, 1) == 0 && continue
        F = qr(M, ColumnNorm())
        r = count(x -> abs(x) > thr, diag(F.R))
        r == 0 && continue
        Qm = Matrix(F.Q)[:, 1:r]
        qn = align_charge(syms, q, ydir, ref.dir)
        push!(space, (qn, r))
        row = 0
        for (k, Mk) in zip(ks, mats)
            nr = size(Mk, 1)
            sz = size(Y.RMTs[k])
            blk =
                reshape(Qm[(row+1):(row+nr), :], (map(l -> sz[l], others)..., r, sz[(QD+1):nd]...))
            row += nr
            push!(keep, k)
            push!(newR, permutedims(blk, invperm(collect(perm))))
            lab = Y.qlabels[k]
            push!(newq, ntuple(l -> l == fl ? qn : lab[l], QD))
        end
    end
    isempty(keep) && return nothing
    wd, wi = Telum._copy_wmat_storage(Y, keep; deep = true)
    inds = ntuple(l -> l == fl ? TLIndex(tag, ref.dir, ref.plev, 0, ref.dual) : Y.inds[l], QD)
    spaces = ntuple(l -> l == fl ? space : copy(Y.spaces[l]), QD)
    return TLArray(syms, newq, wd, wi, newR, inds, spaces)
end

"""
Plan T2.4 "twice is enough", read as the DGKS/Kahan-Parlett acceptance test: a unit column of `Q1`
is a direction only if it keeps at least this fraction of its norm through the second `P^D`. A
genuine direction keeps ~1. Roundoff that cleared the first floor collapses to the frame's own
orthogonality level, which an absolute `100 eps` floor does not reject: in a sector the frame
fills, that column is `P^D`'s cancellation noise renormalised, i.e. a unit vector INSIDE the frame.
MEASURED (dissipative A2 L=32 finite density, closure frontier sweep, dr6z): pass-1 residuals
3.0e-14 against a floor of 2.65e-14 in the filled sectors, kept by pass 2 with `|frame'Q| = 1`; the
controller's working frame absorbed them, the next selections projected against a non-isometric
frame, and the grown frame reached a defect of 14 at site 13 and 1e26 at the root.
"""
const _REORTH_KEEP = 1 / sqrt(2)

"""
    _dd_select(frame, Z, side, kc) -> (Q, backend)

Output DD projection and the two-pass thin QR of plan T2.4: `Y = P^D Z`, `Q1 = qr(Y)` at the
floor `100 eps max(||Z||, ||Y||)` (the cancellation noise `P^D` leaves when most of `Z` lies in
the frame), then `Q = qr(P^D Q1)` -- reorthogonalisation against the old frame, "twice is
enough", keeping the pivots `>= _REORTH_KEEP`. Returns `nothing` when `Y` is numerically zero.
Under SU(2) the sector blocks are not plain matrices, so the existing numerical-rank SVD
orthonormalisation (`_preselect_*` at `stol = 0`, no ranking) is used and reported as backend
`:svd_orth`, never silently.
"""
function _dd_select(frame, Z, side::Symbol, kc::KernelCounters)
    left = side === :left
    if symmetry_mode() === :SU2
        (Q, _), dt, db =
            _measure(() -> left ? _preselect_left(frame, Z, 0.0) : _preselect_right(frame, Z, 0.0))
        kc.t_qr += dt
        kc.b_qr += db
        kc.n_qr += 1
        Q === nothing || (kc.qr_rank += leg_dim(Q, left ? 3 : 1))
        return (Q, :svd_orth)
    end
    fl = left ? 3 : 1
    perp = left ? perp_component : perp_component_right
    Y, dt, db = _measure(() -> perp(frame, Z))
    kc.t_output_dd += dt
    kc.b_output_dd += db
    nz, ny = norm(Z), norm(Y)
    ny <= _CBE_EPS * max(nz, 1.0) && return (nothing, :householder)
    ref = frame.inds[fl]
    tag = left ? _TAG_DD_L : _TAG_DD_R
    Q, dt, db = _measure(
        () -> begin
            Q1 = _thin_qr(Y, fl, ref, tag, _CBE_EPS * max(nz, ny))
            Q1 === nothing ? nothing : _thin_qr(perp(frame, Q1), fl, ref, tag, _REORTH_KEEP)
        end,
    )
    kc.t_qr += dt
    kc.b_qr += db
    kc.n_qr += 1
    Q === nothing || (kc.qr_rank += leg_dim(Q, fl))
    return (Q, :householder)
end

"The DD-QR backend this symmetry uses (`StepRecord.qr_backend`)."
_qr_backend() = symmetry_mode() === :SU2 ? :svd_orth : :householder
