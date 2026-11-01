# Moved from BUG-Julia@b155550:src/RSVDCBEBondUpdate/budget/adaptive_budget.jl.

"""
    AdaptiveBudget(basis, trunc, expv)

The three accuracy tolerances of one CBE-BUG step. `basis` governs the trial space (what the step
may look for and must keep), `trunc` the relative cumulative discarded weight of the final
compression, and `expv` the error estimate of the root exponential. They are kept separate on
purpose.
"""
struct AdaptiveBudget
    basis::Float64
    trunc::Float64
    expv::Float64

    function AdaptiveBudget(basis::Real, trunc::Real, expv::Real)
        basis > 0 || throw(ArgumentError("tol_basis must be positive, got $basis"))
        trunc >= 0 || throw(ArgumentError("tol_trunc must be non-negative, got $trunc"))
        expv > 0 || throw(ArgumentError("tol_expv must be positive, got $expv"))
        return new(Float64(basis), Float64(trunc), Float64(expv))
    end
end

"""
    BasisAllocation

`tol_basis` apportioned across the step's bonds and four basis tests, with the residual scale:
`per_bond` (one bond's whole decision), `sample` (Test B), `tail` (Test A), `krylov` (Tests C/D),
`scale` (factor applied to raw Frobenius residuals).

The default `:rss` rule is `per_bond = tol_basis / sqrt(n_bonds * n_frontiers)`, each test `per_bond
/ sqrt(4)`: errors over bonds add in quadrature. That assumes independent residuals across bonds; a
systematic correlation makes it optimistic by up to `sqrt(n_bonds)`. The `:linear` alternative
([`basis_allocation`](@ref)) needs no such assumption. `rule` records which one bounded the step.
"""
struct BasisAllocation
    per_bond::Float64
    sample::Float64
    tail::Float64
    krylov::Float64
    scale::Float64
    rule::Symbol
end
# The five-share constructor records the default `:rss` rule.
BasisAllocation(per_bond, sample, tail, krylov, scale) =
    BasisAllocation(per_bond, sample, tail, krylov, scale, :rss)

"""
    basis_allocation(budget, nbonds; tau, nrm, nfront = 1, krylov = nothing) -> BasisAllocation

Apportion `budget.basis` over `nbonds` bonds, `nfront` frontiers and the four tests, and fix the
scale the residuals are measured in. The scale is `|tau| / ||psi||`: the residuals enter the step
through `exp(tau*H)` at O(tau), so omitting `|tau|` over-expands at small steps. Dividing by
`||psi||` keeps `tol_basis` meaningful for unnormalised states. `tau = 0` (basis-only sweeps) gives
scale 1, a pure closure question.

`krylov` is the tolerance Tests C/D (the recycled local Krylov directions) are apportioned from, by
the same rule; `nothing` is `budget.basis`. On a nonlocal generator the Krylov stage alone fills the
trial frame to the work ceiling (dr6ta: ~4000 Krylov vs ~0 DD directions per step), so it is tuned
on its own (USER 2026-10-27).
"""
function basis_allocation(
    budget::AdaptiveBudget,
    nbonds::Int;
    tau::Number = 0,
    nrm::Real = 1.0,
    nfront::Int = 1,
    rule::Symbol = :rss,
    krylov::Union{Nothing,Real} = nothing,
)
    nb = max(nbonds, 1)
    nf = max(nfront, 1)
    rule in (:rss, :linear) || throw(ArgumentError("rule must be :rss or :linear, got $rule"))
    kr = krylov === nothing ? 1.0 : Float64(krylov) / budget.basis
    kr > 0 || throw(ArgumentError("tol_krylov must be positive, got $krylov"))
    at = abs(tau)
    scale = iszero(at) ? 1.0 : at / max(Float64(nrm), eps(Float64))
    if rule === :linear
        per_test = budget.basis / (2 * (nb + 1) * nf)
        return BasisAllocation(2 * per_test, per_test, per_test, kr * per_test, scale, :linear)
    end
    per_bond = budget.basis / sqrt(nb * nf)
    # Four tests (A, B-left/right counted as one sampling decision, C/D as one closure
    # decision) -> sqrt(4) = 2. Kept as an explicit `sqrt(4)` so the count is visible.
    per_test = per_bond / sqrt(4)
    return BasisAllocation(per_bond, per_test, per_test, kr * per_test, scale, :rss)
end

"""
    indicator(alloc, resid) -> Float64

A raw Frobenius residual expressed as the scaled, comparable basis-error indicator. Every
Test A/B/C/D comparison goes through this, so the scaling convention lives in exactly one
place rather than being re-derived (and re-mistaken) at four call sites.
"""
indicator(alloc::BasisAllocation, resid::Real) = alloc.scale * Float64(resid)

# ── THE STEP LEDGER ──
# A solver that cannot meet its tolerance must say so. `maxdim` is a resource limit, never a
# substitute for an accuracy tolerance, so when it binds the status changes rather than the claim.

"""
    AdaptiveLedger

Per-step accounting of what the basis decisions actually cost and whether the budget held.

`tolerance_met` is false as soon as ANY bond exhausts its allowance without passing its
check; `workcap_reached` records that a rank/work ceiling -- not a tolerance -- ended a
search. The two are distinct: hitting the cap with the tolerance already satisfied is a
normal, healthy step.
"""
mutable struct AdaptiveLedger
    max_sample::Float64          # worst Test B indicator over the step, either side
    max_sample_l::Float64        # ...left expansions only (plan §8 `B_left`)
    max_sample_r::Float64        # ...right expansions only (plan §8 `B_right`)
    max_tail::Float64            # worst Test A indicator
    max_krylov::Float64          # worst Test C/D indicator
    max_krylov_c::Float64        # worst Test C (first Krylov direction vs the CBE frame)
    max_krylov_d::Float64        # worst Test D (later Krylov directions vs the grown frame)
    # The final compression's RELATIVE CUMULATIVE discarded norm, `sqrt(sum_b w_b) / ||psi||`
    # over every bond it cut -- the quantity `tol_trunc` bounds (plan §5, UT14). Not the
    # largest single bond's weight, which `CBEBugSweepInfo.discarded` still reports.
    discarded::Float64
    expv_err::Float64            # worst root `expv` error estimate (Saad, rel. to the root start)
    n_probes::Int                # sketch columns drawn, summed over bonds and sides
    n_holdout::Int               # independent validation probes drawn
    n_selected::Int              # CBE directions retained
    n_krylov::Int                # Krylov directions appended
    n_h1::Int                    # one-site H applications
    n_targeted::Int              # sectors Test B widened after a failed holdout
    # Plan §8: how often Test C finds CBE already captures the first one-site action, and how
    # often Test D must still grow the frame although Test C passed (the UT08 case).
    n_c_pass::Int
    n_d_after_c::Int
    workcap_reached::Bool
    tolerance_met::Bool
    # `maxdim` cut a bond of the final compression past the rank its `tol_trunc` share asked
    # for. Reported even when the cumulative total still fits (plan §7 UT14: a rank-cap
    # violation is reported, not masked).
    maxdim_reached::Bool
    # Plan §5/§8 `state_projection_error`: `||(1 - P) psi|| / ||psi||` for the frames the root
    # solve integrates in. Zero up to roundoff on the ordinary step (the frames contain `psi`);
    # on the midpoint step's second stage it measures how well the half-step frames hold the
    # ORIGINAL state, and it is charged against `tol_basis`.
    proj_err::Float64
    # Plan §8 `r_old`, `r_augmented`, `r_final`: largest bond rank before the step, of the
    # trial space (half-sweep frames), and after the final compression.
    r_old::Int
    r_aug::Int
    r_final::Int
    # Plan §8 `root_expv_iterations`: operator applications of the root solve (all substeps).
    expv_iters::Int
    # WHICH checks failed, so `:budget_unmet` comes with a reason (see `budget_diagnostic`):
    # `:sample` (Test B), `:tail` (Test A), `:krylov` (Tests C/D), `:trunc` (final
    # compression), `:projection` (midpoint frames), `:expv` (root solve), `:closure` (the
    # order-p Krylov frontiers did not close within the pass bound).
    unmet::Vector{Symbol}
    # One entry per Test B verdict (per bond and side): `(side, est, relunc, per, holdout,
    # sketch)` -- the indicator, a bound on its relative sampling error, each sector's scaled
    # indicator, and the holdout and sketch columns per sector. Plan §3 asks for sector-resolved
    # estimates WITH their uncertainty, and §8 for `n_reachable_sectors`/`probes_per_sector`;
    # a step-level maximum cannot say which sector or how sure.
    sample_log::Vector{Any}
    # Plan §8 "track by bond and timestep": one entry per half-sweep bond visit, written by
    # `_log_bond!` -- the bond's own Test A/B/C/D indicators, its counts, its verdict, the
    # width of the frame it produced (`r_augmented`), and the reachable sectors with their
    # probes; after the compression `cbe_bug_step!` adds `r_old`, `r_final` and that bond's
    # relative `discarded` norm. The step-level fields above are the max/sum over these.
    bond_log::Vector{Any}
    # ── General-order Krylov closure (`krylov_closure_loop.jl`; 1 / 0 / empty on `order = 1`) ──
    order::Int                   # the temporal order the step was asked for
    n_passes::Int                # closure passes over the frontiers j = 1..order-1
    n_closure_sweeps::Int        # frontier half-sweep pairs run (enriching or measuring)
    # Final time-weighted outside-space action of each frontier j = 0..order-1,
    # `|tau| ||(I - P) A F_j|| / ||psi||` (upper bound sqrt(left^2 + right^2)), measured on the
    # frames the root solve used; and the integrated-defect estimate `sum_j leak[j]/(j+1)`.
    leak::Vector{Float64}
    defect::Float64
    n_root_frontier::Int         # zero-site applications that built the frontiers
    n_h2_leak::Int               # two-site applications that measured the outside-space action
    n_h1_closure::Int            # one-site applications of the frontier Tests C/D
    # One entry per frontier sweep: pass, degree, leak before, directions added, widths,
    # newly occupied sectors, bonds skipped by the leak gate.
    closure_log::Vector{Any}
    t_closure::Float64
    n_compress::Int              # final `root_out` compressions (plan: exactly one per step)
    # The same frontiers measured on the frames the half-sweeps built, BEFORE any closure
    # enrichment: what the ordinary CBE + local-Krylov step already captures. `leak_initial` is
    # time-weighted like `leak`; `rho_initial[j]` is scale-free, `||(I - P) A F_j|| / ||F_j||`
    # (upper bound). Filled by `order > 1` with `closure_checks` and by `closure_probe` on `order =
    # 1`.
    leak_initial::Vector{Float64}
    rho_initial::Vector{Float64}
    # Plan Task 6: the parallel K/L/S step's account (`_kls_finish`), `nothing` on every other step.
    kls::Any
end

# A fresh ledger, field by field in declaration order: every indicator and count zero, the
# tolerance met, no checks failed, empty logs, order 1 and no K/L/S account.
AdaptiveLedger() = AdaptiveLedger(
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    false,
    true,
    false,
    0.0,
    0,
    0,
    0,
    0,
    Symbol[],
    Any[],
    Any[],
    1,
    0,
    0,
    Float64[],
    0.0,
    0,
    0,
    0,
    Any[],
    0.0,
    0,
    Float64[],
    Float64[],
    nothing,
)

"""
    _unmet!(ledger, what) -> ledger

Record that the check `what` failed. THE ONE PLACE `tolerance_met` IS CLEARED, so a failed
step always carries the reason with it; `nothing` (the legacy path) is a no-op.
"""
_unmet!(::Nothing, ::Symbol) = nothing
function _unmet!(ledger::AdaptiveLedger, what::Symbol)
    ledger.tolerance_met = false
    what in ledger.unmet || push!(ledger.unmet, what)
    return ledger
end

"""
    budget_status(ledger) -> Symbol

`:met` when every basis, compression and projection check of the step passed its share of
the budget, `:budget_unmet` otherwise (plan §1). `nothing` (the legacy path, which makes no
tolerance claim) gives `:legacy`. [`budget_diagnostic`](@ref) says which check failed and
what to change.
"""
budget_status(::Nothing) = :legacy
budget_status(ledger::AdaptiveLedger) = ledger.tolerance_met ? :met : :budget_unmet

"""
    budget_diagnostic(ledger) -> String

A one-line account of a `:budget_unmet` step: the failed checks with their worst indicators,
and the retry that addresses each (plan §1 "retry diagnostic"). Empty when the budget held.
"""
function budget_diagnostic(ledger::AdaptiveLedger)
    ledger.tolerance_met && return ""
    msg = String[]
    cap =
        ledger.workcap_reached ? " at the work ceiling (raise maxdim or reduce dt)" :
        " with the local space exhausted (reduce dt)"
    :sample in ledger.unmet && push!(msg, "Test B sampling residual $(ledger.max_sample)" * cap)
    :tail in ledger.unmet && push!(msg, "Test A retained-rank tail $(ledger.max_tail)" * cap)
    :krylov in ledger.unmet && push!(msg, "Tests C/D Krylov residual $(ledger.max_krylov)" * cap)
    :trunc in ledger.unmet && push!(
        msg,
        "final compression discarded $(ledger.discarded) relative norm" * (
            ledger.maxdim_reached ? " because maxdim bound (raise maxdim or loosen tol_trunc)" :
            " (loosen tol_trunc)"
        ),
    )
    :projection in ledger.unmet &&
        push!(msg, "midpoint frames miss the original state by $(ledger.proj_err) (reduce dt)")
    :expv in ledger.unmet && push!(
        msg,
        "root expv error estimate $(ledger.expv_err) after $(ledger.expv_iters) " *
        "applications (reduce dt)",
    )
    :closure in ledger.unmet && push!(
        msg,
        "order-$(ledger.order) Krylov closure did not close after " *
        "$(ledger.n_passes) passes, frontier leak $(ledger.leak)" *
        cap *
        " (or lower the order)",
    )
    return "budget unmet: " * join(msg, "; ")
end

# NOT `Base.merge!`: that method is for associatives, and adding a `TLArray`-free method to it
# from here would be type piracy in a module that already imports a great deal of Base.
"Fold a second ledger into the first. The two half-sweeps each keep their own and merge."
function merge_ledger!(ledger::AdaptiveLedger, other::AdaptiveLedger)
    ledger.max_sample = max(ledger.max_sample, other.max_sample)
    ledger.max_sample_l = max(ledger.max_sample_l, other.max_sample_l)
    ledger.max_sample_r = max(ledger.max_sample_r, other.max_sample_r)
    append!(ledger.sample_log, other.sample_log)
    ledger.max_tail = max(ledger.max_tail, other.max_tail)
    ledger.max_krylov = max(ledger.max_krylov, other.max_krylov)
    ledger.max_krylov_c = max(ledger.max_krylov_c, other.max_krylov_c)
    ledger.max_krylov_d = max(ledger.max_krylov_d, other.max_krylov_d)
    ledger.n_c_pass += other.n_c_pass
    ledger.n_d_after_c += other.n_d_after_c
    ledger.discarded = max(ledger.discarded, other.discarded)
    ledger.expv_err = max(ledger.expv_err, other.expv_err)
    ledger.n_probes += other.n_probes
    ledger.n_holdout += other.n_holdout
    ledger.n_selected += other.n_selected
    ledger.n_krylov += other.n_krylov
    ledger.n_h1 += other.n_h1
    ledger.n_targeted += other.n_targeted
    ledger.workcap_reached |= other.workcap_reached
    ledger.tolerance_met &= other.tolerance_met
    ledger.maxdim_reached |= other.maxdim_reached
    ledger.proj_err = max(ledger.proj_err, other.proj_err)
    ledger.r_old = max(ledger.r_old, other.r_old)
    ledger.r_aug = max(ledger.r_aug, other.r_aug)
    ledger.r_final = max(ledger.r_final, other.r_final)
    ledger.expv_iters += other.expv_iters
    union!(ledger.unmet, other.unmet)
    append!(ledger.bond_log, other.bond_log)
    ledger.order = max(ledger.order, other.order)
    ledger.n_passes += other.n_passes
    ledger.n_closure_sweeps += other.n_closure_sweeps
    append!(ledger.leak, other.leak)
    ledger.defect = max(ledger.defect, other.defect)
    ledger.n_root_frontier += other.n_root_frontier
    ledger.n_h2_leak += other.n_h2_leak
    ledger.n_h1_closure += other.n_h1_closure
    append!(ledger.closure_log, other.closure_log)
    ledger.t_closure += other.t_closure
    ledger.n_compress += other.n_compress
    append!(ledger.leak_initial, other.leak_initial)
    append!(ledger.rho_initial, other.rho_initial)
    other.kls === nothing || (ledger.kls = other.kls)
    return ledger
end

"""
    _log_bond!(ledger, bl, bond, side, width) -> ledger

Close one bond of a half-sweep: `bl` is the fresh ledger that bond's CBE (Tests A/B) and Krylov
(Tests C/D) decisions wrote into. Its summary goes into `bl.bond_log`, then `bl` is folded into
the half-sweep's `ledger` -- so the step-level maxima and sums are exactly the aggregate of the
logged bonds, by construction rather than by a second bookkeeping path. `nothing` (legacy) is a
no-op.
"""
_log_bond!(::Nothing, ::Nothing, ::Int, ::Symbol, ::Int) = nothing
function _log_bond!(ledger::AdaptiveLedger, bl::AdaptiveLedger, bond::Int, side::Symbol, width::Int)
    # Probes per reachable sector, from the FINAL Test B sketch of each expansion side (Test B
    # logs every widening attempt; the last one is what the frame was built from). Empty on
    # `exact = true`, which draws no sketch.
    final = Dict{Symbol,Any}()
    for sample_entry in bl.sample_log
        final[sample_entry.side] = sample_entry.sketch
    end
    probes = Dict{Symbol,Dict{Any,Int}}(
        sketch_side => sector_probes for (sketch_side, sector_probes) in final
    )
    push!(
        bl.bond_log,
        (
            bond = bond,
            side = side,
            sample = bl.max_sample,
            tail = bl.max_tail,
            krylov_c = bl.max_krylov_c,
            krylov_d = bl.max_krylov_d,
            n_probes = bl.n_probes,
            n_holdout = bl.n_holdout,
            n_selected = bl.n_selected,
            n_targeted = bl.n_targeted,
            n_krylov = bl.n_krylov,
            n_h1 = bl.n_h1,
            c_pass = bl.n_c_pass > 0,
            d_after_c = bl.n_d_after_c > 0,
            workcap = bl.workcap_reached,
            met = bl.tolerance_met,
            unmet = copy(bl.unmet),
            width = width,
            n_sectors = sum(length, values(probes); init = 0),
            probes_per_sector = probes,
        ),
    )
    return merge_ledger!(ledger, bl)
end

"""
    work_ceiling(maxdim, bond_rank, site_dim = 2) -> Int

The internal cap on temporary working rank at one bond, from the resource limit, the bond's
current rank `bond_rank` (`r` below) and the local dimension `site_dim` (`d` below, the
`local_dim` of the site). Not a user setting.

Expansion may overshoot `maxdim` while the trial space is built, since that is how a rank-`r` state
discovers rank-`r'` structure. It may not grow without bound, which an uncapped spectrum-driven rule
does on bonds whose singular values never decay. The policy `max(d*maxdim, d*r, 1)` admits the
whole one-site space a rank-`maxdim` frame reaches (`d*maxdim` directions, which is also the
two-site space TDVP2 keeps), and the `d*r` floor stops a bond already above `maxdim` from being
frozen. USER 10-28: `d`, not 2 -- at the d = 4 Hubbard site `2*maxdim` held A1 at the ceiling on
every step whatever `tol_krylov` (dq7a). It is a tuned policy, not a theorem; `workcap_reached` in
the ledger reports when it binds.
"""
work_ceiling(maxdim::Int, bond_rank::Int, site_dim::Int = 2) =
    max(site_dim * maxdim, site_dim * bond_rank, 1)

"""
The adaptive search's STARTING width schedule (`budget = ceil(growth*dmax) - r`, probe `1.2x` that).
USER 2026-10-16: 1.2, not the legacy 2.0 -- at 2.0 the first probe is ~1.2chi columns against a
local space of 2chi, so the RSVD starts as wide as the space it sketches. Only a start: Test B
widens a side that fails (`_certified_range(...; thin = true)`). `cbe_expand`'s own default
(2.0) is the fixed-count selection of the TDVP-CBE baselines.
"""
const ADAPTIVE_GROWTH = 1.2

"Default tolerances of the normal API. Plan §1; starting settings, not certified bounds."
const TOL_DEFAULTS = (basis = 1e-5, trunc = 1e-7, expv = 1e-10)
