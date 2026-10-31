# One-site TDVP with controlled bond expansion (CBE): the channel path (`tdvp1_cbe_step!`, nearest-
# neighbour `XXZChain`) and the MPO path (`tdvp_cbe1s_step!`, a port of TDVPSweepCBE1Si.m, for
# long-range or site-dependent H). Each bond: expand, forward one-site update, backward zero-site
# update (the -tau step removes the double count of the bond between two forward updates).
#
# Merged from BUG-Julia@b155550:src/RSVDCBEBondUpdate/baselines/{tdvp1_cbe,tdvp1_cbe_variants,
# tdvp_cbe1s}.jl. Not moved: the `XXZChain` method of `tdvp_cbe1s_step!` (`mpo_from_terms` now
# takes the local identity `Iloc`; build the MPO with QMB's `xxz_mpo`).

"""
    TDVP1CBEInfo

Per-step record. `max_expanded` is the widest bond CBE PROPOSED before the split truncated it,
the same quantity `CBEBugRunInfo.max_expanded` reports, and the one to read when asking whether
the expansion or the truncation is the rank limiter.

`krylov_dims` sums operator applications over both the one-site and zero-site solves, so it is
comparable with the other arms' `krylov_dims` only after remembering that a zero-site matvec is
`O(chi^2)` against a two-site `O(d^2 chi^3)` -- the counts are not the same unit of work.
"""
struct TDVP1CBEInfo
    max_bond::Int
    max_expanded::Int
    discarded::Float64
    err_pre::Float64
    err_fnl::Float64
    krylov_dims::Int
end

"""
One CBE + forward-one-site + backward-zero-site triple at bond `i`, sweeping in `dir`.

Takes the channels on the links it needs so the caller can CARRY them, and returns the ones
the next bond of the same sweep needs -- pushed through the tensor just written, never rebuilt.
`acc` collects diagnostics in place.
"""
function _tdvp1_cbe_bond!(
    psi::SymMPS,
    h::XXZChain,
    i::Int,
    tau::ComplexF64,
    dir::Symbol,
    lch::ChannelSet,
    rch::ChannelSet;
    exkw,
    maxdim,
    trunc_thresh,
    maxiter,
    tol,
    reorth = false,
    acc,
)
    nmv = 0

    # ---- CBE: widen the bond before anything is evolved --------------------------------
    # `_absorb_*!` keeps the STATE identical: the neighbour becomes the expanded frame and the
    # current site takes the (zero-padded) remainder. The new columns carry no amplitude yet,
    # which is precisely what the one-site update is then able to fill.
    canonical!(psi, i)
    f = bond_frame(psi, i)
    ex = cbe_expand(f, h, i, lch, rch; exkw...)
    acc[2] = max(acc[2], max(leg_dim(ex.U_ex, 3), leg_dim(ex.V_ex, 1)))
    acc[4] = max(acc[4], ex.err_pre)
    acc[5] = max(acc[5], ex.err_fnl)

    if dir === :right
        _absorb_right!(psi, i, ex.V_ex)              # psi[i+1] = V_ex, centre stays at i
        tag = psi[i].inds[3].itags


        rch1 = push_right_channels(rch, h, psi[i+1], i + 1)

        # ---- forward: one-site at site i, +tau ----
        M = expv(
            x -> (nmv += 1; apply_one_site(one_site_h(h, i, lch, rch1), x)),
            tau,
            psi[i];
            hermitian = true,
            maxiter = maxiter,
            tol = tol,
            reorth = reorth,
        )

        res = svd(M, (1, 2); cutoff = max(trunc_thresh, 1e-14), Nkeep = maxdim, get_lists = true)
        acc[3] = max(acc[3], _trunc_weight(res))

        psi[i] = res.U
        C = to_concrete(res.S * res.Vd)

        H0 = zero_site_h(h, i, lch, rch, psi[i], psi[i+1])
        C = expv(
            x -> (nmv += 1; apply_zero_site(H0, x)),
            -tau,
            C;
            hermitian = true,
            maxiter = maxiter,
            tol = tol,
            reorth = reorth,
        )

        # Absorb, then rename BOTH ends of the bond to `tag` in the same breath. Each result is
        # rank-3 with only one leg wanting `tag`, so neither construction sees a duplicate.
        psi[i+1] = to_concrete(setitag(to_concrete!(contract(C, (2,), psi[i+1], (1,))), 1, tag))
        psi[i] = to_concrete(setitag(psi[i], 3, tag))
        psi.center = i + 1

        lnext = push_left_channels(lch, h, psi[i], i)
        acc[6] += nmv
        return lnext
    else
        _absorb_left!(psi, i, ex.U_ex)               # psi[i] = U_ex, centre moves to i+1
        tag = psi[i].inds[3].itags
        lch1 = push_left_channels(lch, h, psi[i], i)

        # ---- forward: one-site at site i+1, +tau ----
        M = expv(
            x -> (nmv += 1; apply_one_site(one_site_h(h, i + 1, lch1, rch), x)),
            tau,
            psi[i+1];
            hermitian = true,
            maxiter = maxiter,
            tol = tol,
            reorth = reorth,
        )

        res = svd(M, (1,); cutoff = max(trunc_thresh, 1e-14), Nkeep = maxdim, get_lists = true)
        acc[3] = max(acc[3], _trunc_weight(res))
        # Mirror of the rightward branch: `C` is `(bond_ex, new)`, two different spaces, so the
        # SVD's tag stays on the new leg until after the zero-site step.
        psi[i+1] = res.Vd
        C = to_concrete(res.U * res.S)

        # ---- backward: zero-site, -tau ----
        H0 = zero_site_h(h, i, lch, rch, psi[i], psi[i+1])
        C = expv(
            x -> (nmv += 1; apply_zero_site(H0, x)),
            -tau,
            C;
            hermitian = true,
            maxiter = maxiter,
            tol = tol,
            reorth = reorth,
        )

        psi[i] = to_concrete(setitag(to_concrete!(contract(psi[i], (3,), C, (1,))), 3, tag))
        psi[i+1] = to_concrete(setitag(psi[i+1], 1, tag))
        psi.center = i

        rnext = push_right_channels(rch, h, psi[i+1], i + 1)
        acc[6] += nmv
        return rnext
    end
end

"""
    tdvp1_cbe_step!(psi, h, tau; exact=false, kwargs...) -> TDVP1CBEInfo

One symmetric second-order one-site TDVP-with-CBE step of `tau`, in place, with the time convention
and renormalisation of [`tdvp_cbe1s_step!`](@ref).

`exact = false` (default) is the RSVD selection: the sketch still forms the rank-4 two-site object,
and controls only the width of the candidate set. `exact = true` uses the complete fused basis.
"""
function tdvp1_cbe_step!(
    psi::SymMPS,
    h::XXZChain,
    tau::ComplexF64;
    exact::Bool = false,
    maxdim::Int = 200,
    trunc_thresh::Float64 = 1e-12,
    maxiter::Int = 30,
    tol::Float64 = 1e-15,
    # ── reorthogonalisation on by default: a correctness fix ──
    # Without it, Lanczos at maxiter = 30 loses norm and the exact and sketched arms diverge silently,
    # with a spurious extra bond. `maxiter = 200` fixes it at about 6x the matvecs.
    reorth::Bool = true,
    rng::AbstractRNG = MersenneTwister(0x5EED),
)
    L = length(psi)
    L >= 2 || throw(ArgumentError("tdvp1_cbe_step! needs at least two sites"))
    length(h) == L || throw(DimensionMismatch("chain has $(length(h)) sites, state has $L"))

    # acc = [unused, max_expanded, discarded, err_pre, err_fnl, nmv]
    acc = Any[0, 0, 0.0, 0.0, 0.0, 0]
    half = tau / 2

    # `rmax` read once, before either sweep widens anything, so it is a statement about the
    # state rather than a per-bond one. The expansion runs at `cbe_expand`'s own settings.
    exkw = (rmax = maximum(bond_dims(psi); init = 0), exact = exact, rng = rng)


    onesite!(j, lc, rc) = begin
        acc[6] += 1
        psi[j] = expv(
            x -> apply_one_site(one_site_h(h, j, lc, rc), x),
            half,
            psi[j];
            hermitian = true,
            maxiter = maxiter,
            tol = tol,
            reorth = reorth,
        )
        psi.center = j
    end

    canonical!(psi, 1)
    rstack = right_env_stack(psi, h; downto = 3)
    lch = boundary_channels(h)
    for i = 1:(L-1)
        lch = _tdvp1_cbe_bond!(
            psi,
            h,
            i,
            half,
            :right,
            lch,
            right_channels(rstack, i + 2);
            exkw,
            maxdim,
            trunc_thresh,
            maxiter,
            tol,
            reorth,
            acc,
        )
    end
    # Site L. `lch` is now the channels on link L, carried through the whole pass; to its right
    # is the chain end, so the right environment is the boundary.
    onesite!(L, lch, boundary_channels(h))

    # Safe to build AFTER the closing update: that update rewrote only `psi[L]`, and this stack
    # reads sites 1..L-2.
    lstack = left_env_stack(psi, h; upto = L - 2)
    rch = boundary_channels(h)
    for i = (L-1):-1:1
        rch = _tdvp1_cbe_bond!(
            psi,
            h,
            i,
            half,
            :left,
            left_channels(lstack, i),
            rch;
            exkw,
            maxdim,
            trunc_thresh,
            maxiter,
            tol,
            reorth,
            acc,
        )
    end
    # Site 1, the mirror.
    onesite!(1, boundary_channels(h), rch)

    return TDVP1CBEInfo(maximum(bond_dims(psi); init = 0), acc[2], acc[3], acc[4], acc[5], acc[6])
end

tdvp1_cbe_step!(psi::SymMPS, h::XXZChain, tau::Number; kwargs...) =
    tdvp1_cbe_step!(psi, h, ComplexF64(tau); kwargs...)

# ── the two selection rules of the channel path, named ──

"""
    tdvp1_rsvd_cbe_step!(psi, h, tau; kwargs...) -> TDVP1CBEInfo

One 1-site TDVP + **RSVD-CBE** step (the benchmark arm `1site_tdvp_cbe_rsvd`).

**This is the method.** The bond expansion is chosen from a thin Gaussian sketch of `H*Theta`,
the sketch probes the projector P_perp(H Theta), which forms the rank-4 object. `tau = -im*dt` is real time, `tau = -dt` imaginary;
nothing in the step cares which.

Identical to [`tdvp1_cbe_step!`](@ref) with `exact = false`, which is that function's default —
this name exists so the two selection rules are discoverable by name rather than by a boolean.
"""
tdvp1_rsvd_cbe_step!(psi::SymMPS, h::XXZChain, tau::ComplexF64; kwargs...) =
    tdvp1_cbe_step!(psi, h, tau; exact = false, kwargs...)

tdvp1_rsvd_cbe_step!(psi::SymMPS, h::XXZChain, tau::Number; kwargs...) =
    tdvp1_rsvd_cbe_step!(psi, h, ComplexF64(tau); kwargs...)

"""
    tdvp1_cbe_exact_step!(psi, h, tau; kwargs...) -> TDVP1CBEInfo

One 1-site TDVP + **plain (exact) CBE** step (the benchmark arm `1site_tdvp_cbe`).

The expansion uses the complete fused basis instead of a sketch, so it forms the rank-4
`H*Theta` — the cost the RSVD variant exists to avoid. **This is the A/B reference arm, not a
better default and not a fallback.** Reach for it to check that the sketch is selecting the
right directions; ship [`tdvp1_rsvd_cbe_step!`](@ref).

Identical to [`tdvp1_cbe_step!`](@ref) with `exact = true`.
"""
tdvp1_cbe_exact_step!(psi::SymMPS, h::XXZChain, tau::ComplexF64; kwargs...) =
    tdvp1_cbe_step!(psi, h, tau; exact = true, kwargs...)

tdvp1_cbe_exact_step!(psi::SymMPS, h::XXZChain, tau::Number; kwargs...) =
    tdvp1_cbe_exact_step!(psi, h, ComplexF64(tau); kwargs...)

# ── the MPO path ──

"""
    TDVPCBEInfo

What one [`tdvp_cbe1s_step!`](@ref) did.

  - `max_bond`, `max_expanded`: final rank, and the widest CBE proposal before the split. Compare
    them to see whether the expansion or the truncation is limiting.
  - `n_new`, `err_pre`, `err_fnl`, `discarded`: directions admitted and the largest discarded weights.
  - `krylov_dims`: operator applications over both solves; not comparable with a 0-site-only scheme.
"""
struct TDVPCBEInfo
    max_bond::Int
    max_expanded::Int
    n_new::Vector{Int}
    err_pre::Float64
    err_fnl::Float64
    discarded::Float64
    krylov_dims::Int
    peak_elements::Int
end

"""
    tdvp_cbe1s_step!(psi, mpo, tau; kwargs...) -> TDVPCBEInfo

One symmetric second-order one-site TDVP-with-CBE step of `tau`, in place. `tau = -im*dt` is real
time and `tau = -dt` imaginary time. The caller renormalises; for imaginary time that is load-bearing.

The expansion is [`cbe_expand`](@ref) with its fixed-count settings. `exact = false` (default) is the
RSVD selection; `exact = true` the full fused basis. `maxdim` and `trunc_thresh` bound the splits.

`reorth = true` is a correctness fix: without it the Lanczos silently loses norm.
"""
function tdvp_cbe1s_step!(
    psi::SymMPS,
    mpo::MPO,
    tau::ComplexF64;
    exact::Bool = false,
    # ── fold_omega: fold the sketch before the environment contractions ──
    # Without it every bond builds the full H*Theta for basis selection alone. Folding saves chi/g and so
    # grows with chi, but it can lose at small chi or a wide sketch; the default stays off.
    fold_omega::Bool = false,
    maxdim::Int = 200,
    trunc_thresh::Float64 = 1e-12,
    maxiter::Int = 30,
    tol::Float64 = 1e-15,
    reorth::Bool = true,
    # ── conv_tol: early exit of the Krylov solves, a fairness fix ──
    # `lanczos_expv` exits on breakdown only, so every solve burned `maxiter`. CBE-BUG already stops
    # adaptively, so timing against `conv_tol = 0` would credit that difference to the algorithm.
    conv_tol::Float64 = 0.0,
    substeps::Int = 1,
    # `hermitian = false` -> Arnoldi, for a non-Hermitian generator. See
    # the note in `cbe_bug_step!`: Lanczos does not fail on one, it lies.
    hermitian::Bool = true,
    rng::AbstractRNG = MersenneTwister(0x5EED),
)
    L = length(psi)
    length(mpo) == L || throw(DimensionMismatch("MPO has $(length(mpo)) sites, state has $L"))
    L >= 2 || throw(ArgumentError("tdvp_cbe1s_step! needs at least two sites"))

    cut = max(trunc_thresh, 1e-14)
    half = tau / 2
    n_new = zeros(Int, L - 1)
    maxexp = 0
    epre = 0.0
    efnl = 0.0
    disc = 0.0
    nmv = 0

    # `share_ht` is deliberately NOT in here: the MPO `cbe_expand` REFUSES the pair, and its
    # default (`nothing`) already means "share when project-first" -- so leaving it out is what
    # lets one `exkw` serve both paths.
    exkw = (
        rmax = maximum(bond_dims(psi); init = 0),
        exact = exact,
        fold_omega = fold_omega,
        rng = rng,
    )

    # Working set at this instant: the state as it stands plus the transients still alive. SAME
    # DEFINITION as `CBEBugSweepInfo.peak_elements` and `TDVP2Info.peak_elements`, which is the
    # only reason the three are comparable -- a memory claim measured three different ways is not
    # a claim. It was missing here, so the campaign's `peak` column read 0 for this sweep.
    peak = 0
    peak!(ts...) =
        (peak = max(peak, _state_stored(psi) + sum(tensor_elements(t) for t in ts; init = 0)))

    function record!(info)
        maxexp = max(maxexp, info.dim_out)
        n_new[info.bond] += info.n_new
        epre = max(epre, info.err_pre)
        efnl = max(efnl, info.err_fnl)
    end

    "One-site propagation at site `j`, no split -- the reference's `'vv'` closing update."
    function onesite!(j, lch, rch)
        H1 = one_site_h(mpo, j, lch, rch)
        psi[j] = expv(
            x -> (nmv += 1; apply_one_site(H1, x)),
            half,
            psi[j];
            hermitian = hermitian,
            maxiter = maxiter,
            tol = tol,
            reorth = reorth,
            conv_tol = conv_tol,
            substeps = substeps,
        )
        psi.center = j
    end

    # ======== left to right ========
    canonical!(psi, 1)
    rstack = right_env_stack(psi, mpo; downto = 3)
    lch = boundary_channels(mpo)
    for i = 1:(L-1)
        rch1, info = expand_bond!(psi, mpo, i, :right, lch, right_channels(rstack, i + 2); exkw...)
        record!(info)

        # --- forward one-site update at site i, +tau/2 ---
        H1 = one_site_h(mpo, i, lch, rch1)
        M = expv(
            x -> (nmv += 1; apply_one_site(H1, x)),
            half,
            psi[i];
            hermitian = hermitian,
            maxiter = maxiter,
            tol = tol,
            reorth = reorth,
            conv_tol = conv_tol,
            substeps = substeps,
        )

        peak!(M)
        res = svd(M, (1, 2); cutoff = cut, Nkeep = maxdim, get_lists = true)
        disc = max(disc, _trunc_weight(res))
        tag = psi[i].inds[3].itags
        psi[i] = res.U
        C = to_concrete(res.S * res.Vd)

        # --- backward bond (0-site) update, -tau/2 ---
        # The environments are pushed through the tensors AS THEY NOW STAND: `psi[i]` is the
        # freshly written left isometry and `psi[i+1]` the expanded right frame, so this is
        # exactly the reference's `Update0Si(OrthC, VL, VR)` with its updated `VL` and its
        # expanded `VR`.
        H0 = zero_site_h(mpo, i, lch, right_channels(rstack, i + 2), psi[i], psi[i+1])
        C = expv(
            x -> (nmv += 1; apply_zero_site(H0, x)),
            -half,
            C;
            hermitian = hermitian,
            maxiter = maxiter,
            tol = tol,
            reorth = reorth,
            conv_tol = conv_tol,
            substeps = substeps,
        )

        # Retag BOTH ends of the bond in one breath: each result is rank-3 with only one leg
        # wanting `tag`, so neither construction ever sees a duplicate index.
        psi[i+1] = to_concrete(setitag(to_concrete!(contract(C, (2,), psi[i+1], (1,))), 1, tag))
        psi[i] = to_concrete(setitag(psi[i], 3, tag))
        psi.center = i + 1

        lch = push_left_channels(lch, mpo, psi[i], i)
    end
    onesite!(L, lch, boundary_channels(mpo))

    # ======== right to left ========
    lstack = left_env_stack(psi, mpo; upto = L - 2)
    rch = boundary_channels(mpo)
    for i = (L-1):-1:1
        lch1, info = expand_bond!(psi, mpo, i, :left, left_channels(lstack, i), rch; exkw...)
        record!(info)

        H1 = one_site_h(mpo, i + 1, lch1, rch)
        M = expv(
            x -> (nmv += 1; apply_one_site(H1, x)),
            half,
            psi[i+1];
            hermitian = hermitian,
            maxiter = maxiter,
            tol = tol,
            reorth = reorth,
            conv_tol = conv_tol,
            substeps = substeps,
        )

        res = svd(M, (1,); cutoff = cut, Nkeep = maxdim, get_lists = true)
        disc = max(disc, _trunc_weight(res))
        tag = psi[i].inds[3].itags
        psi[i+1] = res.Vd
        C = to_concrete(res.U * res.S)

        H0 = zero_site_h(mpo, i, left_channels(lstack, i), rch, psi[i], psi[i+1])
        C = expv(
            x -> (nmv += 1; apply_zero_site(H0, x)),
            -half,
            C;
            hermitian = hermitian,
            maxiter = maxiter,
            tol = tol,
            reorth = reorth,
            conv_tol = conv_tol,
            substeps = substeps,
        )

        psi[i] = to_concrete(setitag(to_concrete!(contract(psi[i], (3,), C, (1,))), 3, tag))
        psi[i+1] = to_concrete(setitag(psi[i+1], 1, tag))
        psi.center = i

        rch = push_right_channels(rch, mpo, psi[i+1], i + 1)
    end
    onesite!(1, boundary_channels(mpo), rch)

    return TDVPCBEInfo(
        maximum(bond_dims(psi); init = 0),
        maxexp,
        n_new,
        epre,
        efnl,
        disc,
        nmv,
        peak,
    )
end

tdvp_cbe1s_step!(psi::SymMPS, mpo::MPO, tau::Number; kwargs...) =
    tdvp_cbe1s_step!(psi, mpo, ComplexF64(tau); kwargs...)
