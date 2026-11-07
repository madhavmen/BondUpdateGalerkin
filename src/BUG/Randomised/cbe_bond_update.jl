# cbe_bond_update.jl: the prebuilt-stack wrappers, the gate-path sketches, `cbe_expand_bond`, and
# the gate-based CBE bond update with its expanding-basis Krylov (included after `cbe_expand.jl`).

# ── prebuilt-stack convenience wrappers ──────────────────────────────────────
#
# The `ChannelSet` kernels (`sketch_h_*` in `cbe.jl`, the term-list `cbe_expand` in
# `range_finder.jl`) take the channels on the two links they actually use, so a sweep can
# CARRY them (O(1) per bond) instead of rebuilding a stack (O(L) per bond). These wrappers
# keep the stack-indexed call sites -- the tests, and anything not sweeping -- unchanged.

# `sketch_h_left` at bond `(bond, bond+1)`, reading its two channel sets out of full stacks.
sketch_h_left(
    frame::BondFrame,
    chain::XXZChain,
    bond::Int,
    lenv::LeftEnvStack,
    renv::RightEnvStack,
    Om,
) = sketch_h_left(frame, chain, bond, left_channels(lenv, bond), right_channels(renv, bond + 2), Om)

# `sketch_h_right` at bond `(bond, bond+1)`, reading its two channel sets out of full stacks.
sketch_h_right(
    frame::BondFrame,
    chain::XXZChain,
    bond::Int,
    lenv::LeftEnvStack,
    renv::RightEnvStack,
    Om,
) = sketch_h_right(
    frame,
    chain,
    bond,
    left_channels(lenv, bond),
    right_channels(renv, bond + 2),
    Om,
)

# The term-list `cbe_expand` at bond `(bond, bond+1)`, reading its channel sets out of full stacks.
cbe_expand(
    frame::BondFrame,
    chain::XXZChain,
    bond::Int,
    lenv::LeftEnvStack,
    renv::RightEnvStack;
    kwargs...,
) = cbe_expand(
    frame,
    chain,
    bond,
    left_channels(lenv, bond),
    right_channels(renv, bond + 2);
    kwargs...,
)

"""
    sketch_bond_left(frame, gate, Om) -> TLArray

`Y = (gate Theta) Om†`, legs `(link_l, site_l, g)` -- the left matricization of the gate
action sketched from the right. `Om` carries `V0`'s layout `(g, site_r, link_r)`, i.e. it is
a candidate right frame with `g` columns instead of `r`; `Om = frame.V0` recovers the exact left
factor of `gate Theta`.

Contraction order: absorb `Om` into the gate's right BRA leg, then close the gate's right
KET leg and `link_r` against `V0`, then `K0 = U0 S0`. Nothing rank-4 is built.
"""
function sketch_bond_left(frame::BondFrame, gate, Om)
    _check_gate(frame, gate)
    tl, tr = frame.site_l.itags, frame.site_r.itags
    HT = apply_gate(gate, frame_theta(frame), tl, tr)       # (link_l, site_l, site_r, link_r)
    # Remove the kept space: `perp_action = (I - U0 U0') HT`.
    kept_coeffs = contract(frame.U0', (1, 2), HT, (1, 2))   # (bond, site_r, link_r)
    perp_action = TLArray(HT - contract(frame.U0, (3,), kept_coeffs, (1,)))
    return contract(perp_action, (3, 4), Om', (2, 3))    # (link_l, site_l, g)
end

"""
    sketch_bond_right(frame, gate, Om) -> TLArray

Mirror of [`sketch_bond_left`](@ref): `Om` carries `U0`'s layout `(link_l, site_l, g)` and
the result is `(g, site_r, link_r)`.
"""
function sketch_bond_right(frame::BondFrame, gate, Om)
    _check_gate(frame, gate)
    tl, tr = frame.site_l.itags, frame.site_r.itags
    HT = apply_gate(gate, frame_theta(frame), tl, tr)
    # Remove the kept space: `perp_action = HT (I - V0' V0)`.
    kept_coeffs = contract(HT, (3, 4), frame.V0', (2, 3))   # (link_l, site_l, bond)
    perp_action = TLArray(HT - contract(kept_coeffs, (3,), frame.V0, (1,)))
    sketch = contract(perp_action, (1, 2), Om', (1, 2))     # (site_r, link_r, g)
    return TLArray(permutedims(sketch, (3, 1, 2)))      # (g, site_r, link_r)
end

"""
    cbe_expand_bond(frame, gate; kwargs...) -> CBEExpansion

Controlled bond expansion of the frames at one bond, with the bare two-site gate standing in
for the environment-dressed effective Hamiltonian. Keywords are [`cbe_expand`](@ref)'s,
unchanged -- this is the same selection with different sketch closures.

The sketch probes the PROJECTOR (`P_perp (H Theta)`), which is the only path; there is no
`project_first` switch any more.
"""
cbe_expand_bond(frame::BondFrame, gate; kwargs...) = cbe_expand(
    frame,
    Om -> sketch_bond_left(frame, gate, Om),
    Om -> sketch_bond_right(frame, gate, Om);
    kwargs...,
)

# ── the bond update ──────────────────────────────────────────────────────────

"""
    cbe_bond_update(frame::BondFrame, gate, tau; kwargs...) -> NamedTuple

One CBE-basis Galerkin update on the bond `frame` snapshots, a drop-in for
the K/L/S bond update (same fields, plus `err_pre` / `err_fnl`). `maxdim`,
`trunc_thresh`, `s_tau`, `maxiter`, `tol` as there; `expand = false` freezes the frames; other
keywords forward to `cbe_expand`. `s_iters = 1` is one expansion then Lanczos, `s_iters > 1`
repeats the expansion, `-g <= 0` uses [`_expanding_krylov`](@ref) with `g` growth passes.
"""
function cbe_bond_update(
    frame::BondFrame,
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
    tl, tr = frame.site_l.itags, frame.site_r.itags
    theta0 = frame_theta(frame)

    # Operator applications are counted here (`expv` returns only the vector), so cost can be split
    # from the number of steps.
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
            contract(contract(U_aug', (1, 2), theta0, (1, 2)), (2, 3), V_aug', (2, 3))
        # The projected gate `U_aug' gate (U_aug core_vector V_aug) V_aug'` on a root core.
        function apply_s(core_vector)
            nmv[] += 1
            theta = TLArray((U_aug * core_vector) * V_aug)
            evolved = apply_gate(gate, theta, tl, tr)
            proj = contract(U_aug', (1, 2), evolved, (1, 2))
            return contract(proj, (2, 3), V_aug', (2, 3))
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

    U_aug, V_aug = frame.U0, frame.V0
    err_pre, err_fnl, n_iter, iter_weight = 0.0, 0.0, 0, 0.0
    local S_new

    if !expand
        S_new = galerkin(U_aug, V_aug)
    elseif s_iters <= 0
        # ── GROWTH INSIDE THE LANCZOS ────────────────────────────────────────────────
        # `s_iters <= 0` selects the expanding-basis solver; see `_expanding_krylov`.
        U_aug, V_aug, S_new, info = _expanding_krylov(
            frame,
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
        # ── one expansion, then a plain Lanczos (the default); `s_iters > 1` repeats it ──
        # Each pass recomputes the budget from the current rank, compounding the growth schedule;
        # `galerkin` always starts from the original `theta0`, so the exponential is applied once.
        probe = frame
        for expansion_pass = 1:s_iters
            ex = expand_at(probe)
            U_prev, V_prev = U_aug, V_aug
            U_aug, V_aug = ex.U_ex, ex.V_ex
            err_pre, err_fnl, n_iter = ex.err_pre, ex.err_fnl, expansion_pass

            S_new = galerkin(U_aug, V_aug)

            expansion_pass == s_iters && break
            (ex.n_new_l + ex.n_new_r) == 0 && break
            iter_weight = _new_direction_weight(U_prev, V_prev, U_aug, V_aug, S_new)
            iter_weight < iter_tol && break

            probe = _reframe(frame, U_aug, S_new, V_aug)
        end
    end

    # Truncate the evolved core and absorb its factors into the augmented frames.
    res = svd(S_new, (1,); cutoff = max(trunc_thresh, 1e-14), Nkeep = maxdim, get_lists = true)
    tag = frame.link_mid.itags
    left_core = TLArray(setitag(TLArray(U_aug * res.U), 3, tag))
    right_core = TLArray(setitag(TLArray((res.S * res.Vd) * V_aug), 1, tag))

    svals = [singular_value for (singular_value, _, _, _) in res.kept_list]
    return (;
        left_core,
        right_core,
        U_aug,
        V_aug,
        S_new,
        n_new_k = leg_dim(U_aug, 3) - frame.old_rank,
        n_new_l = leg_dim(V_aug, 1) - frame.old_rank,
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
    _expanding_krylov(frame, applyH, expand, solver, theta0, nmv; ...)
        -> (left_frame, right_frame, root_result, info)

Lanczos for `exp(tau*H_eff)` whose basis grows with the Krylov space: before `H` is applied to
`v_k`, the frames are expanded with `v_k` as the probe, so `H v_k` fits. Stops when `beta_k < tol`
or `grow_iters` passes have run. Full reorthogonalisation is required because the projected
operator changes at every growth pass.
"""
function _expanding_krylov(
    frame::BondFrame,
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
    # Project a two-site tensor onto the root space of the given frames: `left' th right'`.
    project(left_basis, right_basis, th) =
        contract(contract(left_basis', (1, 2), th, (1, 2)), (2, 3), right_basis', (2, 3))

    # Embed a core from the (Uo,Vo) basis into (Un,Vn) by computed overlaps (robust to a rotation by
    # the orthonormality guard); the overlaps are built once per growth pass, not per stored vector.
    embed_ops(Uo, Vo, Un, Vn) = (
        contract(Un', (1, 2), Uo, (1, 2)),         # PL (new_l, old_l)
        contract(Vo, (2, 3), Vn', (2, 3)),
    )         # PR (old_r, new_r)

    # Apply the overlaps: `PL core PR`.
    embed_with(PL, PR, core) =
        contract(contract(PL, (2,), core, (1,)), (2,), PR, (1,))

    left_frame, right_frame = frame.U0, frame.V0
    root_start = project(left_frame, right_frame, theta0)
    beta0 = norm(root_start)
    err_pre, err_fnl, n_grow = 0.0, 0.0, 0
    # `value = NaN` is NOT decoration: every other exit carries it (the Ritz value for
    # `LowestEigen`), and a caller that reads `info.value` -- the ground-state solve does --
    # would hit an UndefVarError on this branch if the field were simply absent here.
    beta0 == 0 && return (
        left_frame,
        right_frame,
        root_start,
        (; err_pre, err_fnl, n_grow, beta_final = 0.0, ranks = Int[], value = NaN),
    )

    basis = Any[TLArray((1.0 / beta0) * root_start)]
    alpha, betas, ranks = Float64[], Float64[], Int[leg_dim(left_frame, 3)]
    b_last = 0.0

    image = nothing        # the PREVIOUS iteration's output, H*v_{k-1}, before orthogonalisation

    for it = 1:maxiter
        # ── step 0: grow the basis around the probe ──
        # `:residual` (default) probes with the orthonormal Lanczos vector; `:image` uses
        # `H v_{k-1}` and measured no different. The Lanczos recurrence itself is unchanged.
        if it <= grow_iters
            pv = (probe === :image && image !== nothing) ? image : basis[it]
            ex = expand(_reframe(frame, left_frame, pv, right_frame))
            if (ex.n_new_l + ex.n_new_r) > 0
                Uo, Vo = left_frame, right_frame
                left_frame, right_frame = ex.U_ex, ex.V_ex
                PL, PR = embed_ops(Uo, Vo, left_frame, right_frame)
                for column in eachindex(basis)
                    basis[column] = embed_with(PL, PR, basis[column])
                end
                image === nothing || (image = embed_with(PL, PR, image))
                err_pre, err_fnl, n_grow = ex.err_pre, ex.err_fnl, n_grow + 1
            end
        end
        push!(ranks, leg_dim(left_frame, 3))

        nmv[] += 1
        theta = TLArray((left_frame * basis[it]) * right_frame)
        residual = project(left_frame, right_frame, applyH(theta))
        image = residual          # this iteration's output, carried to the next expansion

        # Three-term recurrence, then two full reorthogonalisation passes against the basis.
        diagonal_entry = real(tensor_inner(basis[it], residual))
        push!(alpha, diagonal_entry)
        residual = TLArray(residual + (-diagonal_entry) * basis[it])
        it > 1 && (residual = TLArray(residual + (-betas[it-1]) * basis[it-1]))
        for _pass = 1:2, basis_vector in basis
            residual = TLArray(residual - tensor_inner(basis_vector, residual) * basis_vector)
        end

        # `b_last` is the TERMINATING residual, not the last accepted one. Reporting
        # `betas[end]` instead would show the last beta that was large enough to continue --
        # the opposite of the stopping signal it is meant to document.
        residual_norm = norm(residual)
        b_last = residual_norm
        residual_norm < tol && break
        it == maxiter && break
        push!(betas, residual_norm)
        push!(basis, TLArray((1.0 / residual_norm) * residual))
    end

    # THE ONLY MODE-DEPENDENT LINE IN THE WHOLE FUNCTION. `coeff` already carries `beta0` for
    # the propagator and deliberately does not for the eigensolve, so the accumulation below
    # is identical either way.
    coeff, value = _reduce_krylov(solver, alpha, betas, beta0)
    root_result = TLArray(coeff[1] * basis[1])
    for column = 2:length(coeff)
        root_result = TLArray(root_result + coeff[column] * basis[column])
    end
    return (
        left_frame,
        right_frame,
        root_result,
        (; err_pre, err_fnl, n_grow, beta_final = b_last, ranks, value),
    )
end

"""
    _new_direction_weight(U_prev, V_prev, U_aug, V_aug, core) -> Float64

Fraction of the evolved core's norm on the directions the last pass added,
`sqrt(‖core‖² - ‖P core Q‖²)/‖core‖`; small means the Krylov space has stopped delivering.
"""
function _new_direction_weight(U_prev, V_prev, U_aug, V_aug, core)
    core_norm = norm(core)
    core_norm == 0 && return 0.0
    PL = contract(U_prev', (1, 2), U_aug, (1, 2))     # (prev_l, new_l)
    PR = contract(V_aug, (2, 3), V_prev', (2, 3))     # (new_r, prev_r)
    S_old = contract(contract(PL, (2,), core, (1,)), (2,), PR, (1,))
    return sqrt(max(core_norm^2 - norm(S_old)^2, 0.0)) / core_norm
end

"""
    _reframe(frame, left_frame, core, right_frame) -> BondFrame

`frame` with frames and core replaced, so the next sketch probes the evolved state. `old_rank` is
the larger frame rank, so `room` stays a lower bound on each side.
"""
_reframe(frame::BondFrame, left_frame, core, right_frame) = BondFrame(
    left_frame,
    core,
    right_frame,
    frame.link_l,
    frame.site_l,
    frame.link_mid,
    frame.site_r,
    frame.link_r,
    max(leg_dim(left_frame, 3), leg_dim(right_frame, 1)),
    frame.link_mid_space,
)

# Any numeric `tau` is promoted to the `ComplexF64` method above.
cbe_bond_update(frame::BondFrame, gate, tau::Number; kwargs...) =
    cbe_bond_update(frame, gate, ComplexF64(tau); kwargs...)
