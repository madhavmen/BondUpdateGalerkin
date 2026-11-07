# DD-QR CBE kernels: the sketch operators, discarded-space probes and the sectorwise thin QR
# (controller in `dd_cbe_controller.jl`, the bond expansion in `dd_cbe_expand.jl`).

# ── implicit DD-projected, MPO-streamed, QR-only CBE ──
# The frame grows by Om~ = P^D Om, Z = X Om~ streamed through the MPO, and Y = Q R: neither X nor
# the DD projector is formed. One side per call, since the two QRs belong to the two half-sweep
# orientations.

# ── one operator-action interface ──
# Each operator is immutable and built per bond visit, so a stale one cannot see a later gauge.
# Outputs go to a caller-held `Ref`, since Telum contractions allocate their result.

"""
    TwoSiteSketchOp(frame, mpo, bond, lenv, renv; adjoint = false, version = 0, counters)

The two-site effective Hamiltonian at bond `(bond, bond+1)` as a THIN-PROBE action.
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

# Builds the operator from a bond frame, precomputing the core on the side the probe does not
# touch and recording the active symmetry so a later mode switch is caught.
function TwoSiteSketchOp(
    frame::BondFrame,
    mpo::MPO,
    bond::Int,
    lenv::MPOLink,
    renv::MPOLink;
    adjoint::Bool = false,
    version::Int = 0,
    counters::KernelCounters = KernelCounters(),
)
    core = adjoint ? TLArray(frame.S0 * frame.V0) : TLArray(frame.U0 * frame.S0)
    return TwoSiteSketchOp(
        mpo,
        bond,
        lenv,
        renv,
        frame.U0,
        frame.V0,
        core,
        adjoint,
        symmetry_mode(),
        version,
        counters,
    )
end

"""
    apply_two_site_sketch!(result_ref::Ref, Om, op::TwoSiteSketchOp) -> result_ref

`result_ref[] = X Om` (or `X' Om` for `op.adjoint`), unprojected, streamed through the MPO
channels.
"""
function apply_two_site_sketch!(result_ref::Base.RefValue, Om, op::TwoSiteSketchOp)
    op.symmetry === symmetry_mode() || throw(
        ErrorException(
            "TwoSiteSketchOp was built under $(op.symmetry), " *
            "the active symmetry is $(symmetry_mode())",
        ),
    )
    sketch, dt, db = _measure(
        () ->
            op.adjoint ? _stream_right(op.U0, op.core, op.mpo, op.bond, op.lenv, op.renv, Om) :
            _stream_left(op.core, op.V0, op.mpo, op.bond, op.lenv, op.renv, Om),
    )
    kc = op.counters
    kc.n_stream += 1
    kc.n_stream_cols += leg_dim(Om, op.adjoint ? 3 : 1)
    kc.t_stream += dt
    kc.b_stream += db
    result_ref[] = sketch
    return result_ref
end

"`result_ref[] = H1 vector` through the shared interface."
apply_one_site!(result_ref::Base.RefValue, vector, op::OneSiteOp) =
    (result_ref[] = apply_one_site(op, vector); result_ref)

"""
    RootOp(h, version, counters)

The root (zero-site) generator with counted applications; `apply_zero_site(op, core)` forwards.
"""
struct RootOp{H}
    h::H
    version::Int
    counters::KernelCounters
end

# Applies the wrapped root generator to `core`, counting the call, its time and its allocation.
function apply_zero_site(op::RootOp, core)
    image, dt, db = _measure(() -> apply_zero_site(op.h, core))
    op.counters.n_root += 1
    op.counters.t_root += dt
    op.counters.b_root += db
    return image
end

"`result_ref[] = H0 core` through the shared interface."
apply_root!(result_ref::Base.RefValue, core, op::RootOp) =
    (result_ref[] = apply_zero_site(op, core); result_ref)

# ── probes in the discarded space only ───────────────────────────────────────────────────

"""
    _dd_probe(frame, side, column_budget; rng, boost, exact, kc) -> P^D Om, or nothing

A sector-graded Gaussian probe in `frame`'s layout with `column_budget` columns, drawn only in the
discarded space: columns are allotted by each sector's discarded dimension, then the frame is
projected out. `exact = true` projects the complete fused basis instead. Returns `nothing` when
there is no discarded space.

Unlike `sector_graded_sketch(...; comp_ratio = 1.0)`, a short complement is not topped up from the
kept space.
"""
function _dd_probe(
    frame,
    side::Symbol,
    column_budget::Int;
    rng::AbstractRNG,
    boost = nothing,
    exact::Bool = false,
    kc::KernelCounters = KernelCounters(),
)
    # The two frame legs fused into the probe's rows: (link_l, site_l) or (site_r, link_r).
    left_leg, right_leg = side === :left ? (1, 2) : (2, 3)
    fused_basis = fusion_basis(frame, left_leg, right_leg; tag = "cbeG")
    dfull, dT = _frame_sector_dims(frame, side, fused_basis, "_dd_probe")
    dC = Dict(charge => max(dfull[charge] - dT[charge], 0) for charge in keys(dfull))
    sum(values(dC); init = 0) == 0 && return nothing
    probe = if exact
        fused_basis
    else
        cols = _alloc_columns(dC, column_budget)
        boost === nothing || _boost!(cols, boost, dC)
        _trim_per_sector(_randomize_like(fused_basis, rng), 3, cols)
    end
    probe === nothing && return nothing
    Om = side === :left ? probe : TLArray(permutedims(probe, (3, 1, 2)))
    perp = side === :left ? perp_component : perp_component_right
    projected, dt, db = _measure(() -> perp(frame, Om))
    kc.t_input_dd += dt
    kc.b_input_dd += db
    return norm(projected) > 0 ? projected : nothing
end

# ── sectorwise thin Householder QR ───────────────────────────────────────────────────────

const _TAG_DD_L, _TAG_DD_R = "cbeQ,L", "cbeQ,R"

"Block `rmt` as a `(rows, d_fl)` matrix: every leg but `fl` fused into rows, `fl` the columns."
function _block_matrix(rmt::AbstractArray, perm, rows::Int, dfl::Int)
    return reshape(permutedims(rmt, perm), rows, dfl)
end

"""
    _thin_qr(sketch, fl, ref, tag, thr) -> orthonormal_basis or nothing

Orthonormal basis of the span of `sketch`'s leg-`fl` "columns" (fl = 3: columns of the left
matricisation; fl = 1: rows of the right one), by column-pivoted Householder QR per charge
sector of leg `fl`. A direction is kept when its pivot `|R_kk|` exceeds `thr` -- an absolute,
machine-precision-scaled linear-dependence floor, not a physics cutoff; nothing is ranked.

The new leg is built in `ref`'s arrow convention (`align_charge`), so
`[frame | orthonormal_basis]` direct-sums directly. Abelian symmetries only (the block of a sector
is then a plain matrix); callers use `_thin_orth_nonabelian` otherwise.
"""
function _thin_qr(sketch, fl::Int, ref::TLIndex, tag::AbstractString, thr::Float64)
    QD = length(Telum.inds(sketch))
    syms = symm(sketch)
    ydir = Telum.inds(sketch)[fl].dir
    others = Tuple(leg for leg = 1:QD if leg != fl)
    # Group the defined blocks by their charge on leg `fl`, in first-seen order.
    groups = Dict{Any,Vector{Int}}()
    order = Any[]
    for block in eachindex(Telum.logical_qlabels(sketch))
        sketch.isdefined[block] || continue
        charge = Telum.logical_qlabels(sketch)[block][fl]
        haskey(groups, charge) || (groups[charge] = Int[]; push!(order, charge))
        push!(groups[charge], block)
    end
    # every defined block in logical leg order (Telum keeps permutation/conjugation/scale deferred)
    blocks =
        Dict(block => _sector_block(sketch, block) for charge in order for block in groups[charge])
    isempty(blocks) && return nothing
    newq = similar(Telum.logical_qlabels(sketch), 0)
    newR = Array{eltype(first(values(blocks))),ndims(first(values(blocks)))}[]
    keep = Int[]
    space = similar(Telum.spaces(sketch)[fl], 0)
    for charge in order
        ks = groups[charge]
        rmt1 = blocks[ks[1]]
        nd = ndims(rmt1)
        all(size(rmt1, leg) == 1 for leg = (QD+1):nd) || throw(
            ArgumentError(
                "_thin_qr: sector multiplicity axes are not trivial -- non-abelian blocks",
            ),
        )
        # Stack the sector's blocks row-wise into one matrix whose columns are leg `fl`.
        perm = (others..., fl, ((QD+1):nd)...)
        dfl = size(rmt1, fl)
        mats = [
            _block_matrix(
                blocks[block],
                perm,
                prod(size(blocks[block], leg) for leg in others),
                dfl,
            ) for block in ks
        ]
        stacked = reduce(vcat, mats)
        size(stacked, 1) == 0 && continue
        factorization = qr(stacked, ColumnNorm())
        rank_kept = count(pivot -> abs(pivot) > thr, diag(factorization.R))
        rank_kept == 0 && continue
        Qm = Matrix(factorization.Q)[:, 1:rank_kept]
        qn = align_charge(syms, charge, ydir, ref.dir)
        push!(space, (qn, rank_kept))
        # Cut `Qm` back into the sector's blocks, row range by row range.
        row = 0
        for (block, Mk) in zip(ks, mats)
            nr = size(Mk, 1)
            sz = size(blocks[block])
            blk = reshape(
                Qm[(row+1):(row+nr), :],
                (map(leg -> sz[leg], others)..., rank_kept, sz[(QD+1):nd]...),
            )
            row += nr
            push!(keep, block)
            push!(newR, permutedims(blk, invperm(collect(perm))))
            lab = Telum.logical_qlabels(sketch)[block]
            push!(newq, ntuple(leg -> leg == fl ? qn : lab[leg], QD))
        end
    end
    isempty(keep) && return nothing
    wd, wi = Telum._copy_wmat_storage(sketch, keep; deep = true)
    inds = ntuple(
        leg ->
            leg == fl ? TLIndex(tag, ref.dir, ref.plev, 0, ref.dual) : Telum.inds(sketch)[leg],
        QD,
    )
    spaces = ntuple(leg -> leg == fl ? space : copy(Telum.spaces(sketch)[leg]), QD)
    return TLArray(syms, newq, wd, wi, newR, inds, spaces)
end

"""
"Twice is enough", read as the DGKS/Kahan-Parlett acceptance test: a unit column of the
first-pass basis is a direction only if it keeps at least this fraction of its norm through the
second `P^D`. A genuine direction keeps ~1. Roundoff that cleared the first floor collapses to the
frame's own orthogonality level, which an absolute `100 eps` floor does not reject: in a sector the
frame fills, that column is `P^D`'s cancellation noise renormalised, i.e. a unit vector INSIDE the
frame. MEASURED (dissipative A2 L=32 finite density, closure frontier sweep, dr6z): pass-1
residuals 3.0e-14 against a floor of 2.65e-14 in the filled sectors, kept by pass 2 with
`|frame'Q| = 1`; the controller's working frame absorbed them, the next selections projected
against a non-isometric frame, and the grown frame reached a defect of 14 at site 13 and 1e26 at
the root.
"""
const _REORTH_KEEP = 1 / sqrt(2)

"""
    _dd_select(frame, sketch, side, kc) -> (orthonormal_basis, backend)

Output DD projection and the two-pass thin QR: `projected = P^D sketch`,
`first_pass = qr(projected)` at the floor `100 eps max(||sketch||, ||projected||)` (the
cancellation noise `P^D` leaves when most of `sketch` lies in the frame), then
`orthonormal_basis = qr(P^D first_pass)` -- reorthogonalisation against the old frame, "twice is
enough", keeping the pivots `>= _REORTH_KEEP`. Returns `nothing` when `projected` is numerically
zero. Under SU(2) the sector blocks are not plain matrices, so the existing numerical-rank SVD
orthonormalisation (`_preselect_*` at `stol = 0`, no ranking) is used and reported as backend
`:svd_orth`, never silently.
"""
function _dd_select(frame, sketch, side::Symbol, kc::KernelCounters)
    left = side === :left
    if symmetry_mode() === :SU2
        (orthonormal_basis, _), dt, db = _measure(
            () ->
                left ? _preselect_left(frame, sketch, 0.0) : _preselect_right(frame, sketch, 0.0),
        )
        kc.t_qr += dt
        kc.b_qr += db
        kc.n_qr += 1
        orthonormal_basis === nothing || (kc.qr_rank += leg_dim(orthonormal_basis, left ? 3 : 1))
        return (orthonormal_basis, :svd_orth)
    end
    fl = left ? 3 : 1
    perp = left ? perp_component : perp_component_right
    projected, dt, db = _measure(() -> perp(frame, sketch))
    kc.t_output_dd += dt
    kc.b_output_dd += db
    nz, ny = norm(sketch), norm(projected)
    ny <= _CBE_EPS * max(nz, 1.0) && return (nothing, :householder)
    ref = Telum.inds(frame)[fl]
    tag = left ? _TAG_DD_L : _TAG_DD_R
    orthonormal_basis, dt, db = _measure(
        () -> begin
            first_pass = _thin_qr(projected, fl, ref, tag, _CBE_EPS * max(nz, ny))
            first_pass === nothing ? nothing :
            _thin_qr(perp(frame, first_pass), fl, ref, tag, _REORTH_KEEP)
        end,
    )
    kc.t_qr += dt
    kc.b_qr += db
    kc.n_qr += 1
    orthonormal_basis === nothing || (kc.qr_rank += leg_dim(orthonormal_basis, fl))
    return (orthonormal_basis, :householder)
end

"The DD-QR backend this symmetry uses (`StepRecord.qr_backend`)."
_qr_backend() = symmetry_mode() === :SU2 ? :svd_orth : :householder
