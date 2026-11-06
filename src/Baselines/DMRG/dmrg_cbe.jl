# CBE-DMRG ground states: the one-site sweep (`dmrg_cbe1s_sweep!`, a port of DMRGSweepCBE1Si.m) and its
# driver, and the two-site `GroundState` module (RSVD-CBE expansion with the expanding Krylov
# local solve of the time-evolution modes, then a clean Lanczos in the fixed frames).
#
# The one-site paths take an `MPO` only (build it with QMB's `xxz_mpo`); `GroundState` accepts an
# `XXZChain` or an `MPO`.

"""
    DMRGCBEInfo

What one [`dmrg_cbe1s_sweep!`](@ref) half-sweep pair did.

  - `energy`: the lowest Ritz value in the sweep, a variational upper bound. Below the true ground
    energy it is a bug, not a lucky sweep.
  - `energies`: the Ritz value at each local solve; their spread is the convergence signal.
  - `bond_dims` (multiplets), `max_expanded`, `n_new`, `err_pre`, `err_fnl`, `discarded`,
    `krylov_dims`, and `ledger` ([`AdaptiveLedger`](@ref)).
"""
struct DMRGCBEInfo
    energy::Float64
    energies::Vector{Float64}
    bond_dims::Vector{Int}
    max_expanded::Int
    n_new::Vector{Int}
    err_pre::Float64
    err_fnl::Float64
    discarded::Float64
    krylov_dims::Int
    ledger::AdaptiveLedger
end

"""
Default `tol_basis` / `tol_trunc` of the ADAPTIVE CBE-DMRG -- medium-loose, NOT the time step's
`TOL_DEFAULTS` (1e-5 / 1e-7). A DMRG sweep is not a one-shot step: every bond is re-expanded and
re-solved on the next sweep, so a direction a loose sweep skips is offered again, and a tight
per-sweep budget mostly buys rank and matvecs the next sweep throws away. PROVISIONAL until
`benchmarks/ground_state/adaptive_dmrg_tolerance.jl` measures how loose each may be at a given
energy accuracy.
"""
const DMRG_TOL_DEFAULTS = (basis = 1e-3, trunc = 1e-4)

"""
    dmrg_etol(tol_basis, tol_trunc, N) -> Float64

The adaptive CBE-DMRG's sweep-to-sweep stopping threshold, DERIVED from its two tolerances
rather than set as a fourth accuracy knob. Measured (N = 20
Heisenberg / XX chains and 5x4 square cylinder): the returned state's energy error obeys
`dE/N <~ 2.5e-3 tol_basis^2 + 0.2 tol_trunc^2` (largest coefficients seen). The run stops once a
sweep changes the energy by less than a tenth of that, since further sweeps cannot buy more than
the tolerances allow. Floor `1e-12 N` (round-off). A fixed `etol = 1e-10` made loose settings
sweep up to the 40-sweep limit (34-40 sweeps at `tol_basis = 0.1`).
"""
dmrg_etol(tb::Real, tt::Real, N::Int) = max(0.1 * N * (2.5e-3 * tb^2 + 0.2 * tt^2), 1e-12 * N)

"""
    dmrg_cbe1s_sweep!(psi, mpo; kwargs...) -> DMRGCBEInfo

One full CBE-DMRG sweep (left to right, then right to left), in place, with the RSVD expansion engine
of `cbe_bug_step!`. Accuracy is set by `tol_basis`, `tol_trunc` and `maxdim`; the centre ends on site 1.

`tol_basis` is taken at unit time, so `tol_basis^2 / gap` bounds the energy error a met sweep can hide.

`restol = 1e-6` and `comp_ratio = 1.0` measured best. Do not copy `restol` to a time integrator: DMRG
re-solves each site on the next sweep, whereas a time step gets one shot and needs the Krylov depth.
"""
function dmrg_cbe1s_sweep!(
    psi::SymMPS,
    mpo::MPO;
    tol_basis::Union{Nothing,Float64} = nothing,
    tol_trunc::Union{Nothing,Float64} = nothing,
    exact::Bool = false,
    maxdim::Int = 128,
    maxiter::Int = 30,
    restol::Float64 = 1e-6,
    seed_expanded::Bool = false,
    # Adaptive search width from Test B (`_certified_range`); `false` = old.
    thin_search::Bool = true,
    rng::AbstractRNG = MersenneTwister(0x5EED),
)
    L = length(psi)
    length(mpo) == L || throw(DimensionMismatch("MPO has $(length(mpo)) sites, state has $L"))
    L >= 2 || throw(ArgumentError("dmrg_cbe1s_sweep! needs at least two sites"))

    # The GROUND-STATE defaults, not the time step's: a sweep is repeated until the energy
    # settles, so each one need not be tight (see `DMRG_TOL_DEFAULTS`).
    budget = AdaptiveBudget(
        something(tol_basis, DMRG_TOL_DEFAULTS.basis),
        something(tol_trunc, DMRG_TOL_DEFAULTS.trunc),
        TOL_DEFAULTS.expv,
    )
    energies = Float64[]
    n_new = zeros(Int, L - 1)
    maxexp = 0
    epre = 0.0
    efnl = 0.0
    disc = 0.0
    nmv = 0

    rmax = maximum(bond_dims(psi); init = 0)    # read before the gauge, as it always was
    canonical!(psi, 1)
    ledger = AdaptiveLedger()
    # Scale-free indicators: the Lanczos solves return unit vectors anyway, so normalising the
    # start only makes the first bond's residual comparable with the rest.
    n0 = norm(psi[1])
    n0 > 0 && (psi[1] = TLArray((1.0 / n0) * psi[1]))
    alloc = basis_allocation(budget, L - 1; tau = 1.0, nrm = 1.0)
    share = budget.trunc / sqrt(L - 1)          # each split's relative discarded-norm share
    swept = [0.0, 0.0]                          # squared discarded norm, per half-sweep

    # `rmax` read ONCE, before anything widens, so the work ceiling is a statement about the
    # state rather than about whichever bond is being looked at. The preselection cuts only at
    # the numerical rank floor (`stol_pre = 0`); Tests A/B against `tol_basis` decide
    # what matters.
    exkw = (
        growth = ADAPTIVE_GROWTH,
        comp_ratio = 1.0,
        stol_pre = 0.0,
        rmax = rmax,
        exact = exact,
        thin_search = thin_search,
        rng = rng,
        alloc = alloc,
        workcap = work_ceiling(maxdim, rmax),
    )

    function record!(info)
        maxexp = max(maxexp, info.dim_out)
        n_new[info.bond] += info.n_new
        epre = max(epre, info.err_pre)
        efnl = max(efnl, info.err_fnl)
    end

    # The budgeted split after the local solve. Logs the bond (Tests A/B from its expansion, plus
    # what the split discarded) into the sweep ledger.
    function split!(A, legs, i, bondlg, info, half)
        res, capped = _budget_svd(A, legs, maxdim, n2 -> share * sqrt(n2))
        capped && (bondlg.maxdim_reached = true; _unmet!(bondlg, :trunc))
        w = _trunc_weight(res)
        disc = max(disc, w)
        swept[half] += w^2
        _log_bond!(ledger, bondlg, i, info.dir, info.dim_out)
        ledger.bond_log[end] = merge(
            ledger.bond_log[end],
            (r_old = info.dim_in, r_final = length(res.kept_list), discarded = w),
        )
        return res
    end

    """
    The one-site eigensolve at site `j`, in the environments given.

    `seed_expanded` changes only the Lanczos start vector, never the operator or the space, so the Ritz
    value stays a variational upper bound (see [`lanczos_lowest`](@ref)). The default start is the
    zero-padded old vector, with no component along the directions the expansion just admitted;
    seeding with `v + eta * P_new(H v)` puts them in the space from the first vector.
    """
    function solve1(j, lch, rch; expanded::Bool = false)
        H1 = one_site_h(mpo, j, lch, rch)
        apply = x -> apply_one_site(H1, x)
        x0 = psi[j]
        if expanded && seed_expanded
            w = apply(x0)                      # the one extra application, counted below
            nmv += 1
            # Project out the current direction: what remains is what H mixes in from outside the old frame.
            # The denominator must be real: `tensor_inner` returns ComplexF64, and `max` on complex numbers is a
            # MethodError, so `norm(x0)^2` is used.
            ip = tensor_inner(x0, w) / max(norm(x0)^2, eps())
            r = TLArray(w + (-ip) * x0)
            nr = norm(r)
            if nr > 1e-14
                # 0.1 of the current norm: enough to enter the Krylov space at iteration 1,
                # small enough that the start is still essentially the current state.
                x0 = TLArray(x0 + (0.1 * norm(x0) / nr) * r)
            end
        end
        A, e, kd = lanczos_lowest(apply, x0; maxiter = maxiter, restol = restol)
        nmv += kd
        push!(energies, e)
        return A
    end

    # ── left to right ──
    # The right stack is prebuilt once and read at link i+2, which depends only on sites >= i+2. Reading it
    # backwards silently minimises the wrong Hamiltonian, since a stale environment does not throw.
    rstack = right_env_stack(psi, mpo; downto = 3)
    lch = boundary_channels(mpo)
    for i = 1:(L-1)
        bondlg = AdaptiveLedger()                           # this bond's Tests A/B
        rch1, info = expand_bond!(
            psi,
            mpo,
            i,
            :right,
            lch,
            right_channels(rstack, i + 2);
            exkw...,
            ledger = bondlg,
        )
        record!(info)

        A = solve1(i, lch, rch1; expanded = true)
        # Split with the amplitude moving RIGHT, truncating: this is the only place rank is
        # removed, and `Nkeep` is in MULTIPLETS under a non-abelian symmetry.
        res = split!(A, (1, 2), i, bondlg, info, 1)
        tag = Telum.inds(psi[i])[3].itags
        C = TLArray(res.S * res.Vd)
        psi[i] = TLArray(setitag(res.U, 3, tag))
        psi[i+1] = TLArray(setitag(contract(C, (2,), psi[i+1], (1,)), 1, tag))
        psi.center = i + 1

        lch = push_left_channels(lch, mpo, psi[i], i)
    end
    # Site L: the reference's `'vv'` -- solve and keep, no split, because there is no bond to
    # its right to move an orthogonality centre across.
    psi[L] = solve1(L, lch, boundary_channels(mpo))
    psi.center = L

    # ======== right to left ========
    # Built AFTER the closing solve, which rewrote only `psi[L]`; this stack reads sites
    # `1 … L-2`.
    lstack = left_env_stack(psi, mpo; upto = L - 2)
    rch = boundary_channels(mpo)
    for i = (L-1):-1:1
        bondlg = AdaptiveLedger()
        lch1, info = expand_bond!(
            psi,
            mpo,
            i,
            :left,
            left_channels(lstack, i),
            rch;
            exkw...,
            ledger = bondlg,
        )
        record!(info)

        A = solve1(i + 1, lch1, rch; expanded = true)
        res = split!(A, (1,), i, bondlg, info, 2)
        tag = Telum.inds(psi[i])[3].itags
        C = TLArray(res.U * res.S)
        psi[i+1] = TLArray(setitag(res.Vd, 1, tag))
        psi[i] = TLArray(setitag(contract(psi[i], (3,), C, (1,)), 3, tag))
        psi.center = i

        rch = push_right_channels(rch, mpo, psi[i+1], i + 1)
    end
    psi[1] = solve1(1, boundary_channels(mpo), rch)
    psi.center = 1

    # Each half-sweep cuts every bond once, against a unit-norm centre; the larger of the two
    # accumulations is what the sweep's splits cost the state (the `tol_trunc` budget).
    ledger.discarded = sqrt(maximum(swept))
    ledger.r_old = rmax
    ledger.r_aug = maxexp
    ledger.r_final = maximum(bond_dims(psi); init = 0)
    return DMRGCBEInfo(
        minimum(energies; init = Inf),
        energies,
        bond_dims(psi),
        maxexp,
        n_new,
        epre,
        efnl,
        disc,
        nmv,
        ledger,
    )
end

# ── driver ──

"""
    DMRGCBERun

Per-sweep history of [`dmrg_cbe1s!`](@ref). `energies[k]` is sweep `k`'s best Ritz value and
`sweep_spread[k]` the standard deviation across that sweep's local solves -- the reference's
`err_swp`, which is the convergence signal to read rather than the energy change alone: a
sweep whose bonds disagree has not converged even if its best value stopped moving.
"""
struct DMRGCBERun
    energies::Vector{Float64}
    sweep_spread::Vector{Float64}
    bond_dims::Vector{Vector{Int}}
    max_bond_dims::Vector{Int}
    # Under SU(2) `bond_dims` counts MULTIPLETS, so a reader comparing ranks across symmetry
    # modes needs the state count too -- recorded per sweep, because taking it from the final
    # `psi` afterwards reports one number for the whole history and hides the growth.
    state_bond_dims::Vector{Vector{Int}}
    max_state_bond_dims::Vector{Int}
    # Wall-clock seconds for each sweep on its own. The FIRST entry of a fresh session carries
    # Julia's compilation of the whole sweep stack (order 10^2 s here) and is NOT a cost datum.
    sweep_seconds::Vector{Float64}
    max_expanded::Vector{Int}
    err_pre::Vector{Float64}
    err_fnl::Vector{Float64}
    discarded::Vector{Float64}
    krylov_dims::Vector{Int}
    converged::Bool
    # Per sweep: `budget_status` (`:met` / `:budget_unmet`) and the sweep's ledger, so a
    # converged energy carries its basis verdict with it.
    status::Vector{Symbol}
    ledgers::Vector{AdaptiveLedger}
end

Base.length(r::DMRGCBERun) = length(r.energies)

"Sample standard deviation, written out rather than pulling in `Statistics` for one call."
function _spread(v::Vector{Float64})
    n = length(v)
    n < 2 && return 0.0
    m = sum(v) / n
    return sqrt(sum((x - m)^2 for x in v) / (n - 1))
end

"""
    dmrg_cbe1s!(psi, mpo; n_sweeps=20, etol=nothing, kwargs...) -> DMRGCBERun

Ground state of the Hamiltonian `mpo` by CBE-DMRG sweeps, in place. Convergence is tested on
the sweep-best energy, not bond to bond, since within a sweep the energy falls by construction.

`maxdim` is one value per call; a schedule is the caller's business. The particle number (U(1)) or
total spin (SU(2)) is pinned by the initial state, so this finds the ground state of that sector.
"""
function dmrg_cbe1s!(
    psi::SymMPS,
    mpo::MPO;
    n_sweeps::Int = 20,
    etol::Union{Nothing,Float64} = nothing,
    kwargs...,
)
    kw = values(kwargs)
    # The stopping threshold: an explicit `etol` wins, otherwise it is derived from the sweep's
    # own tolerances (`dmrg_etol`).
    etol = something(
        etol,
        dmrg_etol(
            something(get(kw, :tol_basis, nothing), DMRG_TOL_DEFAULTS.basis),
            something(get(kw, :tol_trunc, nothing), DMRG_TOL_DEFAULTS.trunc),
            length(psi),
        ),
    )
    es = Float64[]
    spread = Float64[]
    bds = Vector{Int}[]
    maxbd = Int[]
    sbds = Vector{Int}[]
    maxsbd = Int[]
    secs = Float64[]
    maxexp = Int[]
    epre = Float64[]
    efnl = Float64[]
    disc = Float64[]
    kd = Int[]
    status = Symbol[]
    ledgers = AdaptiveLedger[]
    converged = false
    prev = Inf

    for _ = 1:n_sweeps
        tsw = time()
        info = dmrg_cbe1s_sweep!(psi, mpo; kwargs...)
        push!(secs, time() - tsw)
        push!(es, info.energy)
        push!(spread, _spread(info.energies))
        push!(bds, info.bond_dims)
        push!(maxbd, maximum(info.bond_dims; init = 0))
        # `psi` is the state THIS sweep produced, so this is a per-sweep reading.
        sbd = state_bond_dims(psi)
        push!(sbds, sbd)
        push!(maxsbd, maximum(sbd; init = 0))
        push!(maxexp, info.max_expanded)
        push!(epre, info.err_pre)
        push!(efnl, info.err_fnl)
        push!(disc, info.discarded)
        push!(kd, info.krylov_dims)
        push!(status, budget_status(info.ledger))
        push!(ledgers, info.ledger)
        if abs(info.energy - prev) < etol
            converged = true
            break
        end
        prev = info.energy
    end
    return DMRGCBERun(
        es,
        spread,
        bds,
        maxbd,
        sbds,
        maxsbd,
        secs,
        maxexp,
        epre,
        efnl,
        disc,
        kd,
        converged,
        status,
        ledgers,
    )
end

"""
    rsvd_cbe_dmrg1s!(psi, mpo; kwargs...) -> DMRGCBERun

CBE-DMRG with the **RSVD** selection: the expansion is chosen from a thin Gaussian sketch of
`H*Theta`. **This is the method** -- `dmrg_cbe1s!`'s default, named so an A/B against the exact
arm reads as two arms of one comparison at the call site.
"""
rsvd_cbe_dmrg1s!(psi::SymMPS, mpo::MPO; kwargs...) = dmrg_cbe1s!(psi, mpo; exact = false, kwargs...)

"""
    exact_cbe_dmrg1s!(psi, mpo; kwargs...) -> DMRGCBERun

CBE-DMRG with the **exact** selection: the expansion uses the complete fused basis, so it
forms the rank-4 `H*Theta` -- the cost the randomised version exists to avoid. The A/B
reference arm, not a better default and not a fallback.
"""
exact_cbe_dmrg1s!(psi::SymMPS, mpo::MPO; kwargs...) = dmrg_cbe1s!(psi, mpo; exact = true, kwargs...)

"""
    rsvd_cbe_dmrg1s_seeded!(psi, mpo; kwargs...) -> DMRGCBERun

CBE-DMRG with the RSVD selection and the local eigensolve started from the expanded basis
(`seed_expanded = true`). Everything else matches [`rsvd_cbe_dmrg1s!`](@ref), so a difference is
attributable to the start vector alone.

The seed costs one extra operator application per local solve, counted in `krylov_dims`; compare
applications-to-accuracy, not sweeps.
"""
rsvd_cbe_dmrg1s_seeded!(psi::SymMPS, mpo::MPO; kwargs...) =
    dmrg_cbe1s!(psi, mpo; exact = false, seed_expanded = true, kwargs...)

# ── two-site CBE-DMRG ──

"""
    Baselines.GroundState

Two-site DMRG with RSVD-CBE expansion and the same expanding Krylov local solve as the
time-evolution modes (`_expanding_krylov`). The operator is `apply_h_two_site` (`L + h + R`) and the
reduction is `LowestEigen`, so it is a drop-in S step. MPO path only: gate environments are
identities and would miss the global ground state. Growth passes build the frames, then a clean
Lanczos in fixed frames keeps the Ritz value variational.
"""
module GroundState

using LinearAlgebra, Random
using LurCGT, Telum

using ...BUG
using ...BUG: _expanding_krylov, _reframe

"""
    GroundStateInfo

Per-sweep record. `energies[k]` is the lowest Ritz value seen during sweep `k`, which is a
variational upper bound on the ground energy (phase B makes it so -- see the module
docstring). `max_bond_dims` and `max_expanded` mirror `CBEBugRunInfo`.
"""
struct GroundStateInfo
    energies::Vector{Float64}
    bond_dims::Vector{Vector{Int}}
    max_bond_dims::Vector{Int}
    max_expanded::Vector{Int}
    krylov_dims::Vector{Int}
    grow_passes::Vector{Int}
    converged::Bool
end

Base.length(info::GroundStateInfo) = length(info.energies)

"""
    _bond_solve(f, applyH, expand_at, nmv; grow_iters, maxiter, tol) -> (U, V, S, energy, n_grow)

The local eigenproblem at one bond: phase A grows the frames, phase B solves exactly in them.
"""
function _bond_solve(f, applyH, expand_at, nmv; grow_iters::Int, maxiter::Int, tol::Float64)
    theta0 = frame_theta(f)

    # ---- phase A: grow the frames -------------------------------------------------
    # `maxiter = grow_iters` because this phase is here for its BASIS, not its answer; the
    # vector it returns is discarded and only `U`, `V` are kept. Running it longer would buy
    # iterations of an inconsistent tridiagonal.
    U, V, S_g, ga =
        grow_iters <= 0 ? (f.U0, f.V0, nothing, (; n_grow = 0)) :
        _expanding_krylov(
            f,
            applyH,
            expand_at,
            LowestEigen(),
            theta0,
            nmv;
            maxiter = grow_iters,
            tol = tol,
            grow_iters = grow_iters,
        )

    # ---- phase B: exact Rayleigh-Ritz in the FIXED frames -------------------------
    # `grow_iters = 0` makes `expand_at` unreachable, so the frames cannot move and the
    # tridiagonal is the true Galerkin matrix of the projected operator -- variational again.
    # `theta0` is the ORIGINAL two-site object: the grown frames contain the originals as
    # their first block, so projecting it in is lossless and it is the right start vector.
    fg = grow_iters <= 0 ? f : _reframe(f, U, S_g, V)
    U, V, S, info = _expanding_krylov(
        fg,
        applyH,
        identity,
        LowestEigen(),
        theta0,
        nmv;
        maxiter = maxiter,
        tol = tol,
        grow_iters = 0,
    )
    return U, V, S, info.value, ga.n_grow
end

"""
    _write_bond!(psi, i, f, U, V, S, moving; maxdim, trunc_thresh) -> (rank, expanded)

Truncate the solved core and put it back into the state, leaving the centre where the sweep
is heading: `:right` makes `psi[i]` a left isometry and parks the centre on `psi[i+1]`,
`:left` mirrors it. Getting this backwards silently breaks the next bond's `bond_frame`,
which assumes a true canonical snapshot.
"""
function _write_bond!(
    psi::SymMPS,
    i::Int,
    f,
    U_aug,
    V_aug,
    S,
    moving::Symbol;
    maxdim::Int,
    trunc_thresh::Float64,
    defer::Bool = false,
)
    # `defer = true` keeps every direction in this split, so rank control happens once at the end of
    # `solve!`. The CBE pass and the growth iterations find directions that a truncation at the next
    # bond could discard before the eigensolve sees them. The variational bound is unaffected, but
    # sweeps-to-convergence is not.
    res =
        defer ? svd(S, (1,); cutoff = 0.0, get_lists = true) :
        svd(S, (1,); cutoff = max(trunc_thresh, 1e-14), Nkeep = maxdim, get_lists = true)
    tag = f.link_mid.itags
    if moving === :right
        psi[i] = TLArray(setitag(TLArray(U_aug * res.U), 3, tag))
        psi[i+1] = TLArray(setitag(TLArray((res.S * res.Vd) * V_aug), 1, tag))
        psi.center = i + 1
    else
        psi[i] = TLArray(setitag(TLArray(U_aug * TLArray(res.U * res.S)), 3, tag))
        psi[i+1] = TLArray(setitag(TLArray(res.Vd * V_aug), 1, tag))
        psi.center = i
    end
    return leg_dim(psi[i], 3), max(leg_dim(U_aug, 3), leg_dim(V_aug, 1))
end

"""
    _half_sweep!(psi, h, moving, exkw, nmv; ...) -> (energy, max_expanded, n_grow)

One pass over every bond, left to right (`moving = :right`) or right to left.

`O(L)` environment work: the side the sweep moves away from is CARRIED bond to bond with one
`push_*_channels` each, and the side it moves toward is read from a single stack built at the
start of the half sweep. That stack stays valid for exactly this direction, because a
left-to-right pass only ever rewrites sites at or behind the bond it is on, and the stack is
read ahead of it.
"""
function _half_sweep!(
    psi::SymMPS,
    h::Union{XXZChain,MPO},
    moving::Symbol,
    exkw,
    nmv;
    grow_iters::Int,
    maxiter::Int,
    tol::Float64,
    maxdim::Int,
    trunc_thresh::Float64,
    defer::Bool = false,
)
    L = length(psi)
    e_best = Inf
    maxexp = 0
    grows = 0

    bonds = moving === :right ? (1:(L-1)) : ((L-1):-1:1)
    canonical!(psi, moving === :right ? 1 : L)

    # The stack for the side being swept TOWARD; the other side is carried.
    stack =
        moving === :right ? right_env_stack(psi, h; downto = 3) :
        left_env_stack(psi, h; upto = L - 1)
    carried = boundary_channels(h)

    for i in bonds
        lch = moving === :right ? carried : left_channels(stack, i)
        rch = moving === :right ? right_channels(stack, i + 2) : carried

        # Moving right, the centre is already on the bond's left site; moving left it is one site
        # off, so one QR move fixes either direction (idempotent when in place). It is safe against
        # the prebuilt stacks: moving right rewrites sites <= i while the right stack is read at
        # i+2, and moving left rewrites sites >= i while the left stack is read at 1..i-1.
        canonical!(psi, i)

        f = bond_frame(psi, i)
        applyH = Theta -> apply_h_two_site(Theta, h, i, lch, rch)
        expand_at = fr -> cbe_expand(fr, h, i, lch, rch; exkw...)

        U, V, S, e, ng = _bond_solve(
            f,
            applyH,
            expand_at,
            nmv;
            grow_iters = grow_iters,
            maxiter = maxiter,
            tol = tol,
        )
        grows += ng
        isnan(e) || (e_best = min(e_best, real(e)))

        _, exp_here = _write_bond!(
            psi,
            i,
            f,
            U,
            V,
            S,
            moving;
            maxdim = maxdim,
            trunc_thresh = trunc_thresh,
            defer = defer,
        )
        maxexp = max(maxexp, exp_here)

        # Carry the environment across the bond just written, in the sweep's direction.
        carried =
            moving === :right ? push_left_channels(carried, h, psi[i], i) :
            push_right_channels(carried, h, psi[i+1], i + 1)
    end
    return e_best, maxexp, grows
end

"""
CBE growth passes per local solve, the ground-state analogue of `s_iters = -g`. One is the minimum
viable setting: with zero, the frames never widen, the eigensolve minimises over a one-dimensional
space, and the sweep stays at the Neel product state while reporting convergence. The growth pass
does all of the rank generation, so `info.converged` is not evidence by itself. Raise it if a harder
model stalls.
"""
const _GROW_ITERS = 1

"""
    solve!(psi, h; opts=CBEBugOptions(), n_sweeps=20, etol=1e-10) -> GroundStateInfo

Ground state of `h` in place by two-site DMRG sweeps with RSVD-CBE expansion. `opts` gives the
truncation (`maxdim`, `trunc_thresh`), the Krylov budget and `exact`; the expansion uses
[`_GROW_ITERS`](@ref) growth passes per local solve. `opts.dt` and `opts.n_steps` are ignored.
Sweeps alternate direction and stop when the sweep-best energy moves by less than `etol`.

The result is a variational upper bound (the local solve's second phase runs in fixed frames), so a
reported energy below the true ground energy is a bug. U(1) pins the particle-number sector of
`psi`.
"""
function solve!(
    psi::SymMPS,
    h::Union{XXZChain,MPO};
    opts::CBEBugOptions = CBEBugOptions(),
    n_sweeps::Int = 20,
    etol::Float64 = 1e-10,
    defer_truncation::Bool = false,
)
    L = length(psi)
    length(h) == L || throw(DimensionMismatch("chain has $(length(h)) sites, state has $L"))
    L >= 2 || throw(ArgumentError("ground-state search needs at least two sites"))

    rng = MersenneTwister(opts.seed)
    exkw = (rmax = maximum(bond_dims(psi); init = 0), exact = opts.exact, rng = rng)

    energies = Float64[]
    bds = Vector{Int}[]
    maxbd = Int[]
    maxexp = Int[]
    kdim = Int[]
    grows = Int[]
    converged = false
    prev = Inf

    for sweep = 1:n_sweeps
        nmv = Ref(0)
        e, mx, ng = _half_sweep!(
            psi,
            h,
            isodd(sweep) ? :right : :left,
            exkw,
            nmv;
            grow_iters = _GROW_ITERS,
            maxiter = opts.maxiter,
            tol = opts.tol,
            maxdim = opts.maxdim,
            trunc_thresh = opts.trunc_thresh,
            defer = defer_truncation,
        )
        push!(energies, e)
        d = bond_dims(psi)
        push!(bds, d)
        push!(maxbd, maximum(d; init = 0))
        push!(maxexp, mx)
        push!(kdim, nmv[])
        push!(grows, ng)

        # Compared against the PREVIOUS SWEEP, not the previous bond: within a sweep the
        # energy falls monotonically by construction, so a per-bond test would stop early on
        # any flat stretch.
        if abs(e - prev) < etol
            converged = true
            break
        end
        prev = e
    end

    # ── deferred truncation: must not be skipped ──
    # One recursive root-to-leaves pass cuts each bond once (`Theta_tau`, arXiv:2201.10291 Alg. 7).
    # Recorded energies are unaffected (variational in the space used); the returned state is cut.
    if defer_truncation
        canonical!(psi, max(1, min(L - 1, L ÷ 2)))
        truncate_recursive!(
            psi,
            max(1, min(L - 1, L ÷ 2));
            maxdim = opts.maxdim,
            cutoff = opts.trunc_thresh,
        )
        # OVERWRITE, not push: these arrays are indexed BY SWEEP and an extra entry would
        # invent a sweep that never ran.
        d = bond_dims(psi)
        bds[end] = d
        maxbd[end] = maximum(d; init = 0)
    end

    return GroundStateInfo(energies, bds, maxbd, maxexp, kdim, grows, converged)
end

"""
    cbe_dmrg!(psi, h; opts=CBEBugOptions(), kwargs...) -> GroundStateInfo

Plain CBE-DMRG: the expansion uses the COMPLETE fused basis, so the preselection sees the
exact `H*Theta` and its SVD returns the true leading complement directions.

This is the reference arm. It forms the rank-4 two-site object (cost `d*chi` per side against
the sketch's `~1.2*budget`), which is exactly what the randomised version exists to avoid --
so it is the thing to measure against, not a better default.
"""
cbe_dmrg!(psi::SymMPS, h::Union{XXZChain,MPO}; opts::CBEBugOptions = CBEBugOptions(), kwargs...) =
    solve!(psi, h; opts = _with_exact(opts, true), kwargs...)

"""
    rsvd_cbe_dmrg!(psi, h; opts=CBEBugOptions(), kwargs...) -> GroundStateInfo

RSVD-CBE-DMRG: the sketched expansion, and [`solve!`](@ref)'s default. Named so the A/B
against [`cbe_dmrg!`](@ref) reads as two arms of one comparison at the call site.
"""
rsvd_cbe_dmrg!(
    psi::SymMPS,
    h::Union{XXZChain,MPO};
    opts::CBEBugOptions = CBEBugOptions(),
    kwargs...,
) = solve!(psi, h; opts = _with_exact(opts, false), kwargs...)

# Kept private: callers set `exact` in the options directly. The copy runs over `fieldnames`, not a
# hand-written field list, so a field added later is not silently reset to its default while its
# neighbours keep their values.
_with_exact(o::CBEBugOptions, e::Bool) = CBEBugOptions(;
    (f => (f === :exact ? e : getfield(o, f)) for f in fieldnames(CBEBugOptions))...,
)

export GroundStateInfo, solve!, cbe_dmrg!, rsvd_cbe_dmrg!

end # module GroundState
