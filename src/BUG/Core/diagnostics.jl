# Step and run records: kernel counters, the typed step record, run and sweep infos, and the closing
# assembly and compression of a step (`_assemble_and_compress!`).
#

# ── from BUG/Randomised/dd_cbe.jl: deterministic, used by the Core sweeps ──

"""
    KernelCounters

Operator applications, QR ranks, sector additions, seconds and allocated bytes per kernel.
Plain mutable fields so the hot loop records with no allocation. One instance per half-sweep
(they may run concurrently) and one for the root, merged after the join.

Bytes are `Base.gc_bytes()` differences: process-wide, so under `parallel = :half_sweeps`
each half-sweep's counts include the other thread's allocations in the same interval.
"""
mutable struct KernelCounters
    n_stream::Int          # MPO-streamed two-site actions (one per sketch call, all channels)
    n_stream_cols::Int     # probe columns those actions carried
    n_one_site::Int        # one-site H applications (local Krylov)
    n_root::Int            # zero-site H applications (root solve)
    n_qr::Int              # DD-QR selections
    qr_rank::Int           # directions those QRs kept (numerical rank, summed)
    n_sector_new::Int      # charge sectors a frame gained that it did not hold before
    t_input_dd::Float64    # P^D on the thin input probe
    t_stream::Float64
    t_output_dd::Float64   # P^D on the thin streamed output
    t_qr::Float64
    t_one_site::Float64
    t_root::Float64
    b_input_dd::Int
    b_stream::Int
    b_output_dd::Int
    b_qr::Int
    # Every phase is measured in bytes as well as seconds. The environment
    # and compression times are the step's `t_env` / `t_trunc`.
    b_one_site::Int
    b_root::Int
    b_env::Int             # environment stacks and channel pushes of the half-sweeps
    b_compress::Int        # the closing compression
end

# Every counter at zero.
KernelCounters() =
    KernelCounters(0, 0, 0, 0, 0, 0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, 0, 0, 0, 0, 0, 0, 0)

"Add every counter of `addend` into `total` (the join of concurrent half-sweeps); returns `total`."
function merge_counters!(total::KernelCounters, addend::KernelCounters)
    for field in fieldnames(KernelCounters)
        setfield!(total, field, getfield(total, field) + getfield(addend, field))
    end
    return total
end

"Run `thunk()` and return `(result, seconds, bytes)`."
@inline function _measure(thunk)
    start_bytes = Base.gc_bytes()
    start_time = time_ns()
    result = thunk()
    return result, (time_ns() - start_time) / 1e9, Base.gc_bytes() - start_bytes
end

"""
    OneSiteOp(h, version, counters)

A one-site effective Hamiltonian (`MPOOneSiteH`, possibly `shift_h`-ed) that counts its
applications. `apply_one_site(op, site_tensor)` forwards, so every existing caller works unchanged.
"""
struct OneSiteOp{H}
    h::H
    version::Int
    counters::KernelCounters
end

# Apply the wrapped one-site operator to `site_tensor`, recording one application, its seconds and
# its bytes; returns the image.
function apply_one_site(op::OneSiteOp, site_tensor)
    image, dt, db = _measure(() -> apply_one_site(op.h, site_tensor))
    op.counters.n_one_site += 1
    op.counters.t_one_site += dt
    op.counters.b_one_site += db
    return image
end

"""
    StepRecord

One `cbe_bug_step!`'s account, typed and allocation-light: which algorithm ran
(`policy`, `qr_backend`, `closure_policy` for the order > 1 frontier sweeps, `order`,
`parallel`), the kernel counters, the ranks `D_old` (largest bond before), `D_stage` (largest
trial-space frame, untruncated), `D_final` (after compression) and the two limits they are held
to, `D_max` (`maxdim`, the final rank) and `D_work` (the working-rank ceiling `work_ceiling`
puts on the frames), root iterations, total seconds and bytes,
the process `peak_rss` (`Sys.maxrss()`, a high-water mark, not per step), and `limits`: which
of `:basis_limited`, `:root_limited`, `:truncation_limited`, `:work_limited`, `:rank_limited`
held the step back (empty when the budget held), and `budget_rule`, how `tol_basis` was
aggregated over the shares (`:linear`).
"""
struct StepRecord
    policy::Symbol
    qr_backend::Symbol
    closure_policy::Symbol
    order::Int
    parallel::Symbol
    counters::KernelCounters
    D_old::Int
    D_stage::Int
    D_final::Int
    D_max::Int
    D_work::Int
    root_iters::Int
    t_step::Float64
    bytes_step::Int
    peak_rss::Int
    limits::Vector{Symbol}
    budget_rule::Symbol
end

"The separate limit statuses, read off the ledger's failed checks."
function _step_limits(ledger::AdaptiveLedger)
    lim = Symbol[]
    any(check -> check in (:sample, :tail, :krylov, :closure, :projection), ledger.unmet) &&
        push!(lim, :basis_limited)
    :expv in ledger.unmet && push!(lim, :root_limited)
    :trunc in ledger.unmet && push!(lim, :truncation_limited)
    ledger.maxdim_reached && push!(lim, :rank_limited)
    ledger.workcap_reached && push!(lim, :work_limited)
    return lim
end

"""
    CBEBugRunInfo

Per-step record of a CBE-BUG run; every field has length `n_steps`. `max_expanded` is the
headline rank diagnostic: the largest bond proposed before truncation (`r + Dex`). `krylov_dims` and
`mean_aug` form the cost pair, operator applications and the size they act on; wall time is roughly
their product.
"""
struct CBEBugRunInfo
    times::Vector{Float64}
    norms::Vector{Float64}
    bond_dims::Vector{Vector{Int}}
    max_bond_dims::Vector{Int}
    expanded::Vector{Vector{Int}}
    max_expanded::Vector{Int}
    centre_ranks::Vector{Int}
    err_pre::Vector{Float64}
    err_fnl::Vector{Float64}
    discarded::Vector{Float64}
    magnetisations::Vector{Vector{Float64}}
    observations::Vector{Any}
    krylov_dims::Vector{Int}
    mean_aug::Vector{Float64}
    # EMPTY unless the caller asked for it. Imaginary time fills it (energy is the convergence
    # signal there); real time leaves it empty rather than paying for a full contraction per
    # step that nothing reads.
    energies::Vector{Float64}
end

# The record without `energies` (real time): that field starts empty.
CBEBugRunInfo(
    times,
    norms,
    bds,
    maxbd,
    expanded,
    maxexp,
    centres,
    epre,
    efnl,
    disc,
    mags,
    obs,
    kdim,
    maug,
) = CBEBugRunInfo(
    times,
    norms,
    bds,
    maxbd,
    expanded,
    maxexp,
    centres,
    epre,
    efnl,
    disc,
    mags,
    obs,
    kdim,
    maug,
    Float64[],
)

# The number of recorded steps.
Base.length(info::CBEBugRunInfo) = length(info.times)

"""
    CBEBugSweepInfo

What one [`cbe_bug_step!`](@ref) did:

  - `expanded`, `n_new`: bond dims after the update (multiplets under SU(2)); directions admitted.
  - `root`, `root_rank`: the Galerkin bond and its kept rank.
  - `err_pre`, `err_fnl`: largest discarded weight in preselection / final selection.
  - `discarded`: largest weight one closing SVD discarded (cumulative: `ledger.discarded`).
  - `ledger`: the [`AdaptiveLedger`](@ref), read by [`budget_status`](@ref).
  - `krylov_dims`, `peak_elements`: applications and working set (not below 2-site TDVP's).
"""
struct CBEBugSweepInfo
    expanded::Vector{Int}
    n_new::Vector{Int}
    root::Int
    root_rank::Int
    err_pre::Float64
    err_fnl::Float64
    discarded::Float64
    krylov_dims::Int
    peak_elements::Int
    # Extra CBE growth passes of the removed fixed-knob path; always 0, kept so the record's
    # layout (and every recorded run) is unchanged.
    n_grow::Int
    # The root's right frame width, which nothing else recorded. `expanded[root_link]` is
    # `leg_dim(left_frames[root_link], 3)` -- the LEFT frame only -- and the right half-sweep skips
    # the root (`bond != root_link && (expanded[bond] = ...)`), so `b_R` was invisible. That
    # matters because the Galerkin space is `b_L x b_R`: judging completeness from `b_L` alone can
    # call a basis complete when it is not, which is exactly the wrong way for a diagnostic to fail.
    root_right::Int
    # Seconds inside the CBE (`dd_cbe_expand`), the only phase the randomised sketch makes cheaper. Without this
    # timer an rSVD study cannot tell a slow sketch from a fast sketch inside a small expansion
    # share, and those call for opposite responses: by Amdahl, the speedup is capped at 1/(1-f) for
    # an expansion share f.
    t_cbe::Float64
    # A phase breakdown of the step, because "the sketch did not help" has two very different
    # causes and only one is a code problem. The sketch makes the CBE cheaper and nothing
    # else, so Amdahl caps the whole-step speedup at `1/(1 - t_cbe/t_step)`. If `t_cbe` is a small
    # share, the answer is not a better sketch -- it is whatever these other phases are doing.
    t_kry::Float64      # building the Krylov frames (`_krylov_enrich`)
    t_root::Float64     # the root Galerkin solve (`expv` on the zero-site generator)
    t_trunc::Float64    # the closing `truncate_recursive!`
    # The timers that close the step's time accounting: without them a profile leaves a large
    # unattributed share. `t_frame` is an SVD too (`_frame_from` factorises both sides of a bond),
    # so it is not part of the CBE sketch. `t_split` is the removed fixed-knob path's frame splits
    # and is always 0.
    t_split::Float64    # the removed fixed-knob path's frame splits; always 0
    t_frame::Float64    # `_frame_from`, which factorises both sides of the bond
    t_env::Float64      # environment stacks and channel pushes
    t_step::Float64     # the whole call, so the shares can be checked to sum to 1
    # The step's verdict. Without it a step that failed Test A/B, or stopped at the work
    # ceiling, returns looking exactly like one that passed -- the silent convergence claim the
    # three-tolerance API exists to rule out.
    ledger::AdaptiveLedger
    # The typed record of the step -- CBE policy, kernel counters, ranks, limit
    # statuses (`StepRecord`, dd_cbe.jl).
    record::StepRecord
end

"""
    _assemble_and_compress!(psi, left_frames, right_frames, root_link, res, ledger, bwts, bd0;
                            maxdim, tol_trunc, truncate) -> (root_rank, discarded, t_trunc)

Stage 4 of [`cbe_bug_step!`](@ref): the frames become the new site tensors --
`left_frames[1:root_link-1]`, the split root core `left_frames[root_link] U` and
`S Vd right_frames[root_link]`, `right_frames[root_link+1:end]` -- and then ONE compression of the
evolved state. `res` is the SVD of the evolved root. `tol_trunc` is a budget on the
CUMULATIVE discarded norm over all bonds, shared out by `truncate_recursive!` into `bwts`; the
ledger records it and closes the per-bond log. `discarded` is the largest single relative weight
the root split or the compression threw away.
"""
function _assemble_and_compress!(
    psi,
    left_frames,
    right_frames,
    root_link::Int,
    res,
    ledger::AdaptiveLedger,
    bwts,
    bd0;
    maxdim::Int,
    tol_trunc::Float64,
    truncate::Bool,
)
    site_count = length(psi)
    discarded = _trunc_weight(res)
    # Left of the root: the left frames, relinked onto the state's own link tags.
    for site = 1:(root_link-1)
        psi[site] = _relink(left_frames[site], Telum.inds(psi[site])[1].itags, Telum.inds(psi[site])[3].itags)
    end
    # The root bond: U joins the left frame, S Vd the right frame, so the centre lands on
    # root_link + 1.
    t1c, t3c = Telum.inds(psi[root_link])[1].itags, Telum.inds(psi[root_link])[3].itags
    t3r = Telum.inds(psi[root_link+1])[3].itags
    psi[root_link] = _relink(TLArray(left_frames[root_link] * res.U), t1c, t3c)
    psi[root_link+1] = _relink(TLArray((res.S * res.Vd) * right_frames[root_link]), t3c, t3r)
    # Right of the root: right frame `bond` becomes site `bond + 1`.
    for bond = (root_link+1):(site_count-1)
        psi[bond+1] =
            _relink(right_frames[bond], Telum.inds(psi[bond+1])[1].itags, Telum.inds(psi[bond+1])[3].itags)
    end
    psi.center = root_link + 1
    root_rank = leg_dim(psi[root_link], 3)

    ttr = 0.0
    if truncate
        start_time = time_ns()
        _, wtr, cumw, capped = truncate_recursive!(
            psi,
            root_link + 1;
            maxdim = maxdim,
            weight_tol = tol_trunc,
            bond_weights = bwts,
        )
        ttr = (time_ns() - start_time) / 1e9
        discarded = max(discarded, wtr)
        ledger.n_compress += 1
        ledger.discarded = cumw
        ledger.maxdim_reached = capped
        # Relative slack for the rounding in the per-bond shares, which sum to `tol` exactly.
        cumw > tol_trunc * (1 + 1e-8) && _unmet!(ledger, :trunc)
    end
    bd1 = bond_dims(psi)
    ledger.r_final = maximum(bd1; init = 0)
    # Close the per-bond log with what only exists after the root solve and compression.
    ledger.bond_log = Any[
        merge(
            entry,
            (r_old = bd0[entry.bond], r_final = bd1[entry.bond], discarded = bwts[entry.bond]),
        ) for entry in ledger.bond_log
    ]
    return (root_rank, discarded, ttr)
end
