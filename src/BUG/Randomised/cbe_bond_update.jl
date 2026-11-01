# Moved from BUG-Julia@b155550:src/RSVDCBEBondUpdate/expansion/cbe_core.jl, src/RSVDCBEBondUpdate/expansion/cbe_expand.jl, src/RSVDCBEBondUpdate/expansion/cbe_options.jl, src/RSVDCBEBondUpdate/expansion/cbe_bond_update.jl, src/RSVDCBEBondUpdate/operators/mpo_env.jl, src/RSVDCBEBondUpdate/operators/mpo_fold.jl, src/RSVDCBEBondUpdate/expansion/bond_expansion.jl.

# cbe_bond_update.jl: the gate-path sketches, `cbe_expand_bond`, and the gate-based CBE bond
# update with its expanding-basis Krylov (included after `cbe_expand.jl`).

# ── from src/RSVDCBEBondUpdate/expansion/cbe_options.jl ──

# cbe_options.jl: prebuilt-stack wrappers, the shared run configuration, the S step and the sketched gate action (from `cbe_core.jl`, included right after it).

# ── prebuilt-stack convenience wrappers ──────────────────────────────────────
#
# The kernels above take the channels on the two links they actually use, so a sweep can
# CARRY them (O(1) per bond) instead of rebuilding a stack (O(L) per bond). These wrappers
# keep the stack-indexed call sites -- the tests, and anything not sweeping -- unchanged.

sketch_h_left(f::BondFrame, h::XXZChain, i::Int, lenv::LeftEnvStack, renv::RightEnvStack, Om) =
    sketch_h_left(f, h, i, left_channels(lenv, i), right_channels(renv, i + 2), Om)

sketch_h_right(f::BondFrame, h::XXZChain, i::Int, lenv::LeftEnvStack, renv::RightEnvStack, Om) =
    sketch_h_right(f, h, i, left_channels(lenv, i), right_channels(renv, i + 2), Om)

cbe_expand(f::BondFrame, h::XXZChain, i::Int, lenv::LeftEnvStack, renv::RightEnvStack; kwargs...) =
    cbe_expand(f, h, i, left_channels(lenv, i), right_channels(renv, i + 2); kwargs...)

"""
    sketch_bond_left(f, gate, Om) -> TLArray

`Y = (gate Theta) Om†`, legs `(link_l, site_l, g)` -- the left matricization of the gate
action sketched from the right. `Om` carries `V0`'s layout `(g, site_r, link_r)`, i.e. it is
a candidate right frame with `g` columns instead of `r`; `Om = f.V0` recovers the exact left
factor of `gate Theta`.

Contraction order: absorb `Om` into the gate's right BRA leg, then close the gate's right
KET leg and `link_r` against `V0`, then `K0 = U0 S0`. Nothing rank-4 is built.
"""
function sketch_bond_left(f::BondFrame, gate, Om)
    _check_gate(f, gate)
    tl, tr = f.site_l.itags, f.site_r.itags
    HT = apply_gate(gate, frame_theta(f), tl, tr)       # (link_l, site_l, site_r, link_r)
    c = contract(f.U0', (1, 2), HT, (1, 2))            # (bond, site_r, link_r)
    A = to_concrete(HT - to_concrete!(contract(f.U0, (3,), c, (1,))))
    return to_concrete!(contract(A, (3, 4), Om', (2, 3)))          # (link_l, site_l, g)
end

"""
    sketch_bond_right(f, gate, Om) -> TLArray

Mirror of [`sketch_bond_left`](@ref): `Om` carries `U0`'s layout `(link_l, site_l, g)` and
the result is `(g, site_r, link_r)`.
"""
function sketch_bond_right(f::BondFrame, gate, Om)
    _check_gate(f, gate)
    tl, tr = f.site_l.itags, f.site_r.itags
    HT = apply_gate(gate, frame_theta(f), tl, tr)
    c = contract(HT, (3, 4), f.V0', (2, 3))            # (link_l, site_l, bond)
    A = to_concrete(HT - to_concrete!(contract(c, (3,), f.V0, (1,))))
    Y = contract(A, (1, 2), Om', (1, 2))               # (site_r, link_r, g)
    return to_concrete(permutedims(Y, (3, 1, 2)))       # (g, site_r, link_r)
end

"""
    cbe_expand_bond(f, gate; kwargs...) -> CBEExpansion

Controlled bond expansion of the frames at one bond, with the bare two-site gate standing in
for the environment-dressed effective Hamiltonian. Keywords are [`cbe_expand`](@ref)'s,
unchanged -- this is the same selection with different sketch closures.

The sketch probes the PROJECTOR (`P_perp (H Theta)`), which is the only path; there is no
`project_first` switch any more.
"""
cbe_expand_bond(f::BondFrame, gate; kwargs...) = cbe_expand(
    f,
    Om -> sketch_bond_left(f, gate, Om),
    Om -> sketch_bond_right(f, gate, Om);
    kwargs...,
)

# ── from src/RSVDCBEBondUpdate/expansion/cbe_bond_update.jl ──

# cbe_bond_update.jl: the gate-based CBE bond update and its sweep (from `cbe_core.jl`, included right after it).

# ── the bond update ──────────────────────────────────────────────────────────

"""
    cbe_bond_update(f::BondFrame, gate, tau; kwargs...) -> NamedTuple

One CBE-basis Galerkin update on the bond `f` snapshots, a drop-in for
`BondUpdateBUG.kls_bond_update` (same fields, plus `err_pre` / `err_fnl`). `maxdim`,
`trunc_thresh`, `s_tau`, `maxiter`, `tol` as there; `expand = false` freezes the frames; other
keywords forward to `cbe_expand`. `s_iters = 1` is one expansion then Lanczos, `n > 1` repeats
the expansion, `-g <= 0` uses [`_expanding_krylov`](@ref) with `g` growth passes.
"""
function cbe_bond_update(
    f::BondFrame,
    gate,
    tau::ComplexF64;
    maxdim::Int = 200,
    trunc_thresh::Float64 = 1e-14,
    s_tau::Union{Nothing,ComplexF64} = nothing,
    expand::Bool = true,
    maxiter::Int = 30,
    tol::Float64 = 1e-15,
    s_iters::Int = 1,
    iter_tol::Float64 = 1e-12,
    s_reorth::Bool = false,
    s_probe::Symbol = :residual,
    rng::AbstractRNG = MersenneTwister(0x5EED),
    expand_kwargs...,
)
    tau_s = s_tau === nothing ? tau : s_tau
    tl, tr = f.site_l.itags, f.site_r.itags
    theta0 = frame_theta(f)

    # Operator applications are counted here (`expv` returns only the vector), so cost can be split from
    # the number of steps.
    nmv = Ref(0)

    # The two closures that are all a mode gets to change. The gate path supplies the bare
    # two-site gate here (its environments are the identity by canonical form -- see
    # `CBEBugOptions`); the MPO path supplies the environment-dressed `apply_h_two_site`.
    applyH = th -> apply_gate(gate, th, tl, tr)
    expand_at = fr -> cbe_expand_bond(fr, gate; rng = rng, expand_kwargs...)

    # The Galerkin step in a GIVEN basis. Note `S_start` is rebuilt from `theta0` every time it
    # is called -- see the loop below for why that is the whole correctness argument.
    function galerkin(U_aug, V_aug)
        S_start =
            to_concrete!(contract(contract(U_aug', (1, 2), theta0, (1, 2)), (2, 3), V_aug', (2, 3)))
        function apply_s(x)
            nmv[] += 1
            theta = to_concrete((U_aug * x) * V_aug)
            evolved = apply_gate(gate, theta, tl, tr)
            proj = contract(U_aug', (1, 2), evolved, (1, 2))
            return to_concrete!(contract(proj, (2, 3), V_aug', (2, 3)))
        end
        return expv(
            apply_s,
            tau_s,
            S_start;
            hermitian = true,
            maxiter = maxiter,
            tol = tol,
            reorth = s_reorth,
        )
    end

    U_aug, V_aug = f.U0, f.V0
    err_pre, err_fnl, n_iter, iter_weight = 0.0, 0.0, 0, 0.0
    local S_new

    if !expand
        S_new = galerkin(U_aug, V_aug)
    elseif s_iters <= 0
        # ── GROWTH INSIDE THE LANCZOS ────────────────────────────────────────────────
        # `s_iters <= 0` selects the expanding-basis solver; see `_expanding_krylov`.
        U_aug, V_aug, S_new, info = _expanding_krylov(
            f,
            applyH,
            expand_at,
            Propagate(tau_s),
            theta0,
            nmv;
            maxiter = maxiter,
            tol = tol,
            grow_iters = -s_iters,
            probe = s_probe,
        )
        err_pre, err_fnl = info.err_pre, info.err_fnl
        n_iter, iter_weight = info.n_grow, info.beta_final
    else
        # ── one expansion, then a plain Lanczos (the default); `n > 1` repeats it ──
        # Each pass recomputes the budget from the current rank, compounding the growth schedule;
        # `galerkin` always starts from the original `theta0`, so the exponential is applied once.
        probe = f
        for k = 1:s_iters
            ex = expand_at(probe)
            U_prev, V_prev = U_aug, V_aug
            U_aug, V_aug = ex.U_ex, ex.V_ex
            err_pre, err_fnl, n_iter = ex.err_pre, ex.err_fnl, k

            S_new = galerkin(U_aug, V_aug)

            k == s_iters && break
            (ex.n_new_l + ex.n_new_r) == 0 && break
            iter_weight = _new_direction_weight(U_prev, V_prev, U_aug, V_aug, S_new)
            iter_weight < iter_tol && break

            probe = _reframe(f, U_aug, S_new, V_aug)
        end
    end

    res = svd(S_new, (1,); cutoff = max(trunc_thresh, 1e-14), Nkeep = maxdim, get_lists = true)
    tag = f.link_mid.itags
    left_core = to_concrete(setitag(to_concrete(U_aug * res.U), 3, tag))
    right_core = to_concrete(setitag(to_concrete((res.S * res.Vd) * V_aug), 1, tag))

    svals = [s for (s, _, _, _) in res.kept_list]
    return (;
        left_core,
        right_core,
        U_aug,
        V_aug,
        S_new,
        n_new_k = leg_dim(U_aug, 3) - f.old_rank,
        n_new_l = leg_dim(V_aug, 1) - f.old_rank,
        keep = leg_dim(left_core, 3),
        aug_k = leg_dim(U_aug, 3),
        aug_l = leg_dim(V_aug, 1),
        svals,
        discarded = _trunc_weight(res),
        err_pre,
        err_fnl,
        n_matvec = nmv[],
        n_iter,
        iter_weight,
    )
end

"""
    _expanding_krylov(f, applyH, expand, solver, theta0, nmv; ...) -> (U, V, S, info)

Lanczos for `exp(tau*H_eff)` whose basis grows with the Krylov space: before `H` is applied to
`v_k`, the frames are expanded with `v_k` as the probe, so `H v_k` fits. Stops when `beta_k < tol`
or `grow_iters` passes have run. Full reorthogonalisation is required because the projected
operator changes at every growth pass.
"""
function _expanding_krylov(
    f::BondFrame,
    applyH,
    expand,
    solver::SStepSolver,
    theta0,
    nmv;
    maxiter::Int,
    tol::Float64,
    grow_iters::Int,
    probe::Symbol = :residual,
)
    project(U, V, th) = to_concrete!(contract(contract(U', (1, 2), th, (1, 2)), (2, 3), V', (2, 3)))

    # Embed a core from the (Uo,Vo) basis into (Un,Vn) by computed overlaps (robust to a rotation by the
    # orthonormality guard); the overlaps are built once per growth pass, not per stored vector.
    embed_ops(Uo, Vo, Un, Vn) = (
        to_concrete!(contract(Un', (1, 2), Uo, (1, 2))),         # PL (new_l, old_l)
        to_concrete!(contract(Vo, (2, 3), Vn', (2, 3))),
    )         # PR (old_r, new_r)

    embed_with(PL, PR, S) = to_concrete!(contract(contract(PL, (2,), S, (1,)), (2,), PR, (1,)))

    U, V = f.U0, f.V0
    S0 = project(U, V, theta0)
    beta0 = norm(S0)
    err_pre, err_fnl, n_grow = 0.0, 0.0, 0
    # `value = NaN` is NOT decoration: every other exit carries it (the Ritz value for
    # `LowestEigen`), and a caller that reads `info.value` -- the ground-state solve does --
    # would hit an UndefVarError on this branch if the field were simply absent here.
    beta0 == 0 && return (
        U,
        V,
        S0,
        (; err_pre, err_fnl, n_grow, beta_final = 0.0, ranks = Int[], value = NaN),
    )

    basis = Any[to_concrete((1.0 / beta0) * S0)]
    alpha, betas, ranks = Float64[], Float64[], Int[leg_dim(U, 3)]
    b_last = 0.0

    image = nothing        # the PREVIOUS iteration's output, H*v_{k-1}, before orthogonalisation

    for it = 1:maxiter
        # ── step 0: grow the basis around the probe ──
        # `:residual` (default) probes with the orthonormal Lanczos vector; `:image` uses `H v_{k-1}` and
        # measured no different. The Lanczos recurrence itself is unchanged.
        if it <= grow_iters
            pv = (probe === :image && image !== nothing) ? image : basis[it]
            ex = expand(_reframe(f, U, pv, V))
            if (ex.n_new_l + ex.n_new_r) > 0
                Uo, Vo = U, V
                U, V = ex.U_ex, ex.V_ex
                PL, PR = embed_ops(Uo, Vo, U, V)
                for k in eachindex(basis)
                    basis[k] = embed_with(PL, PR, basis[k])
                end
                image === nothing || (image = embed_with(PL, PR, image))
                err_pre, err_fnl, n_grow = ex.err_pre, ex.err_fnl, n_grow + 1
            end
        end
        push!(ranks, leg_dim(U, 3))

        nmv[] += 1
        theta = to_concrete((U * basis[it]) * V)
        w = project(U, V, applyH(theta))
        image = w          # this iteration's output, carried to the next expansion

        a = real(tensor_inner(basis[it], w))
        push!(alpha, a)
        w = to_concrete(w + (-a) * basis[it])
        it > 1 && (w = to_concrete(w + (-betas[it-1]) * basis[it-1]))
        for _pass = 1:2, u in basis
            w = to_concrete(w - tensor_inner(u, w) * u)
        end

        # `b_last` is the TERMINATING residual, not the last accepted one. Reporting
        # `betas[end]` instead would show the last beta that was large enough to continue --
        # the opposite of the stopping signal it is meant to document.
        b = norm(w)
        b_last = b
        b < tol && break
        it == maxiter && break
        push!(betas, b)
        push!(basis, to_concrete((1.0 / b) * w))
    end

    # THE ONLY MODE-DEPENDENT LINE IN THE WHOLE FUNCTION. `coeff` already carries `beta0` for
    # the propagator and deliberately does not for the eigensolve, so the accumulation below
    # is identical either way.
    coeff, value = _reduce_krylov(solver, alpha, betas, beta0)
    S = to_concrete(coeff[1] * basis[1])
    for k = 2:length(coeff)
        S = to_concrete(S + coeff[k] * basis[k])
    end
    return (U, V, S, (; err_pre, err_fnl, n_grow, beta_final = b_last, ranks, value))
end

"""
    _new_direction_weight(U_prev, V_prev, U_aug, V_aug, S) -> Float64

Fraction of the evolved core's norm on the directions the last pass added,
`sqrt(‖S‖² - ‖P S Q‖²)/‖S‖`; small means the Krylov space has stopped delivering.
"""
function _new_direction_weight(U_prev, V_prev, U_aug, V_aug, S)
    n = norm(S)
    n == 0 && return 0.0
    PL = to_concrete!(contract(U_prev', (1, 2), U_aug, (1, 2)))     # (prev_l, new_l)
    PR = to_concrete!(contract(V_aug, (2, 3), V_prev', (2, 3)))     # (new_r, prev_r)
    S_old = to_concrete!(contract(contract(PL, (2,), S, (1,)), (2,), PR, (1,)))
    return sqrt(max(n^2 - norm(S_old)^2, 0.0)) / n
end

"""
    _reframe(f, U, S, V) -> BondFrame

`f` with frames and core replaced, so the next sketch probes the evolved state. `old_rank` is the
larger frame rank, so `room` stays a lower bound on each side.
"""
_reframe(f::BondFrame, U, S, V) = BondFrame(
    U,
    S,
    V,
    f.link_l,
    f.site_l,
    f.link_mid,
    f.site_r,
    f.link_r,
    max(leg_dim(U, 3), leg_dim(V, 1)),
    f.link_mid_space,
)

cbe_bond_update(f::BondFrame, gate, tau::Number; kwargs...) =
    cbe_bond_update(f, gate, ComplexF64(tau); kwargs...)
