# Two-site TDVP on dense QTT cores (lagged and stage-consistent nonlinear density), the split-
# dissipation TDVP control of the damped GP equation, and the Krylov exponentials they use.
#
# The flows and the dense contractions come from BUG and QTTS.

# ── dense effective actions ──

"Two-site effective action at `(i, i+1)`."
function _eff2(E, Hi, Hj, F, Th)
    X = _tc(E, (3,), Th, (1,))                     # (a, b, t1, t2, e)
    X = _tc(X, (2, 3), Hi, (1, 3))                 # (a, t2, e, s1, b2)
    X = _tc(X, (2, 5), Hj, (3, 1))                 # (a, e, s1, s2, b3)
    return _tc(X, (2, 5), F, (3, 2))               # (a, s1, s2, d)
end

"One-site effective action."
function _eff1(E, Hk, F, M)
    X = _tc(E, (3,), M, (1,))                      # (a, b, t, e)
    X = _tc(X, (2, 3), Hk, (1, 3))                 # (a, e, s, b')
    return _tc(X, (2, 4), F, (3, 2))               # (a, s, d)
end

# ── Krylov exponentials ──

"""
    _dense_expv(apply, v, tau; tol = 1e-12, m = 30) -> (w, napply)

`exp(tau A) v` by Arnoldi with the a-posteriori estimate `|h_{m+1,m} [exp(tau H_m)]_{m,1}|`,
halving the substep until it fits `tol` (relative).
"""
function _dense_expv(apply, v::AbstractArray, tau::Number; tol::Real = 1e-12, m::Int = 30)
    nv = norm(v)
    nv == 0 && return copy(v), 0
    nap = 0
    w = ComplexF64.(v)
    rem = ComplexF64(tau)
    dt = rem
    while abs(rem) > 0
        abs(dt) > abs(rem) && (dt = rem)
        beta = norm(w)
        Vs = Array{ComplexF64}[w./beta]
        Hm = zeros(ComplexF64, m + 1, m)
        k = m
        happy = false
        for j = 1:m
            u = ComplexF64.(apply(Vs[j]))
            nap += 1
            for i = 1:j
                Hm[i, j] = dot(Vs[i], u)
                u .-= Hm[i, j] .* Vs[i]
            end
            for i = 1:j                                  # one reorthogonalisation pass
                c = dot(Vs[i], u)
                Hm[i, j] += c
                u .-= c .* Vs[i]
            end
            Hm[j+1, j] = norm(u)
            if abs(Hm[j+1, j]) <= 1e-14 * max(1.0, opnorm(Hm[1:j, 1:j], 1))
                k = j
                happy = true
                break
            end
            push!(Vs, u ./ Hm[j+1, j])
        end
        while true
            E = exp(dt .* Hm[1:k, 1:k])
            err = happy ? 0.0 : abs(Hm[k+1, k] * dt) * abs(E[k, 1])
            if err <= tol || abs(dt) < 1e-12 * abs(tau)
                w = beta .* sum(E[i, 1] .* Vs[i] for i = 1:k)
                rem -= dt
                break
            end
            dt /= 2
        end
    end
    return w, nap
end

"""
    tt_expv(apply, C, tau; m = 16, tol = 1e-10, maxdim = 256, tol_trunc = 1e-12) -> (TT, info)

`exp(tau A) C` by Arnoldi on TT vectors (every new vector compressed at `tol_trunc`), substeps
halved until the estimate `|h_{m+1,m} tau [exp(tau H_m)]_{m,1}|` fits `tol`. A stable global
reference with no backward substep.
"""
function tt_expv(
    apply,
    C::AbstractVector,
    tau::Number;
    m::Int = 16,
    tol::Real = 1e-10,
    maxdim::Int = 256,
    tol_trunc::Real = 1e-12,
)
    rem = ComplexF64(tau)
    dt = rem
    w = [ComplexF64.(c) for c in C]
    nap = 0
    while abs(rem) > 0
        abs(dt) > abs(rem) && (dt = rem)
        beta = tt_norm(w)
        beta == 0 && break
        Vs = Any[tt_scale(w, 1 / beta)]
        Hm = zeros(ComplexF64, m + 1, m)
        k = m
        happy = false
        for j = 1:m
            u = first(tt_compress(apply(Vs[j]); tol = tol_trunc, maxdim = maxdim))
            nap += 1
            for _ = 1:2, i = 1:j
                c = tt_dot(Vs[i], u)
                Hm[i, j] += c
                u = tt_lincomb([(1.0, u), (-c, Vs[i])]; tol = tol_trunc, maxdim = maxdim)
            end
            Hm[j+1, j] = tt_norm(u)
            if abs(Hm[j+1, j]) <= 1e-13 * max(1.0, opnorm(Hm[1:j, 1:j], 1))
                k, happy = j, true
                break
            end
            push!(Vs, tt_scale(u, 1 / Hm[j+1, j]))
        end
        while true
            E = exp(dt .* Hm[1:k, 1:k])
            err = happy ? 0.0 : abs(Hm[k+1, k] * dt) * abs(E[k, 1])
            if err <= tol || abs(dt) < 1e-12 * abs(tau)
                w = tt_lincomb(
                    [(beta * E[i, 1], Vs[i]) for i = 1:k];
                    tol = tol_trunc,
                    maxdim = maxdim,
                )
                rem -= dt
                break
            end
            dt /= 2
        end
    end
    return w, (applications = nap,)
end

# ── TDVP2 on dense cores ──

"""
    qtt_tdvp2_step!(psi, fl, t, h; density = :lagged, maxdim = 128, tol_trunc = 1e-10,
                    tol_krylov = 1e-12, tol_sc = 1e-10, maxsc = 12) -> NamedTuple

One symmetric two-site projector-splitting TDVP step (left-to-right then right-to-left, each `h/2`)
of the flow `fl`, with the explicitly BACKWARD one-site centre substeps `exp(-h/2 F_1)`:

- `density = :lagged` (TDVP2_lagged): `H[psi_n, t_n]` held through the step;
- `density = :consistent` (TDVP2_consistent): every substep re-evaluates `H` (and the paddle at the
  substep time) from the TRIAL state, by the self-consistent midpoint iteration
  `x' = exp(dt F[(x + x')/2]) x` to `tol_sc`, environments rebuilt at every iterate.

Records `A_back = ||M_after|| / ||M_before||` for every backward substep, the
self-consistency iterations and the discarded weight. "TDVP2" names the two-site update, not a
temporal order; the order is measured (V07).
"""
function qtt_tdvp2_step!(
    psi::SymMPS,
    fl::QTTFlow,
    t::Real,
    h::Real;
    density::Symbol = :lagged,
    maxdim::Int = 128,
    tol_trunc::Real = 1e-10,
    tol_krylov::Real = 1e-12,
    tol_sc::Real = 1e-10,
    maxsc::Int = 12,
)
    density in (:lagged, :consistent) ||
        throw(ArgumentError("density must be :lagged or :consistent, got :$density"))
    t0 = time_ns()
    C, _ = tt_compress(qtt_cores(psi); tol = 0.0)          # right-canonical, centre at site 1
    L = length(C)
    flow = density === :lagged ? frozen_flow(fl, C, t) : fl
    mu_of(H, X) = flow_normalized(fl) ? real(_op_expect(H, X)) / max(tt_norm(X)^2, eps()) : 0.0
    back = Float64[]
    nsc = Int[]
    disc2 = 0.0
    nap = 0
    half = h / 2
    # The whole state with the local block `x` at sites `lo..hi` (one or two sites), factorised
    # EXACTLY (no truncation): what the density of a trial state is built from.
    function assemble(x, lo, hi)
        X = copy(C)
        if lo == hi
            X[lo] = x
        else
            cl, _, _, cr = size(x)
            X[lo] = reshape(x, cl, 2, 2cr)
            X[hi] = reshape(Matrix{ComplexF64}(I, 2cr, 2cr), 2cr, 2, cr)
        end
        return X
    end
    # environments of the sites outside `lo..hi` for the operator H, bra = ket = the state
    function local_op(H, X, lo, hi)
        E = _unit_env()
        for k = 1:(lo-1)
            E = _env_left(E, X[k], H[k], X[k])
        end
        F = _unit_env()
        for k = L:-1:(hi+1)
            F = _env_right(F, X[k], H[k], X[k])
        end
        return lo == hi ? (z -> _eff1(E, H[lo], F, z)) : (z -> _eff2(E, H[lo], H[hi], F, z))
    end
    # one local substep `exp(dt F_local) x` at time `ts`, generator from the trial state
    function substep(x, lo, hi, dt, ts)
        X = assemble(x, lo, hi)
        H, _ = flow_op(flow, X, ts)
        mu = mu_of(H, X)
        app = local_op(H, X, lo, hi)
        y, k = _dense_expv(z -> fl.pref .* (app(z) .- mu .* z), x, dt; tol = tol_krylov)
        nap += k
        density === :lagged && (push!(nsc, 0); return y)
        it = 0
        for _ = 1:maxsc
            it += 1
            Xm = assemble((x .+ y) ./ 2, lo, hi)              # the midpoint trial state
            H, _ = flow_op(flow, Xm, ts)
            mu = mu_of(H, Xm)
            app = local_op(H, Xm, lo, hi)
            y2, k = _dense_expv(z -> fl.pref .* (app(z) .- mu .* z), x, dt; tol = tol_krylov)
            nap += k
            done = norm(y2 .- y) <= tol_sc * max(norm(y2), eps())
            y = y2
            done && break
        end
        push!(nsc, it)
        return y
    end
    # the truncating split of a two-site block (the TDVP2 rank control)
    function cut(Th, keep_left::Bool)
        cl, _, _, cr = size(Th)
        F = svd(reshape(Th, 2cl, 2cr))
        nrm = norm(F.S)
        r = _qtt_rank(F.S, tol_trunc * nrm / sqrt(max(L - 1, 1)), maxdim)
        disc2 += sum(abs2, @view(F.S[(r+1):end]); init = 0.0) / max(nrm^2, eps())
        U, S, Vt = F.U[:, 1:r], F.S[1:r], F.Vt[1:r, :]
        return keep_left ? (reshape(U, cl, 2, r), reshape(Diagonal(S) * Vt, r, 2, cr)) :
               (reshape(U * Diagonal(S), cl, 2, r), reshape(Vt, r, 2, cr))
    end
    theta(i) = reshape(
        reshape(C[i], :, size(C[i], 3)) * reshape(C[i+1], size(C[i+1], 1), :),
        size(C[i], 1),
        2,
        2,
        size(C[i+1], 3),
    )
    tmid = t + half / 2
    for i = 1:(L-1)                                     # left to right, h/2
        Th = substep(theta(i), i, i + 1, half, tmid)
        C[i], M = cut(Th, true)
        if i < L - 1
            C[i+1] = M                                   # in place for the environments
            n0 = norm(M)
            M = substep(M, i + 1, i + 1, -half, tmid)      # the BACKWARD centre substep
            push!(back, norm(M) / n0)
        end
        C[i+1] = M
    end
    tmid = t + 3half / 2
    for i = (L-1):-1:1                                  # right to left, h/2
        Th = substep(theta(i), i, i + 1, half, tmid)
        M, C[i+1] = cut(Th, false)
        if i > 1
            C[i] = M
            n0 = norm(M)
            M = substep(M, i, i, -half, tmid)
            push!(back, norm(M) / n0)
        end
        C[i] = M
    end
    _replace_state!(psi, qtt_mps(C; center = 1))
    return (
        A_back = back,
        A_max = maximum(back; init = 1.0),
        sc_iters = nsc,
        discarded = sqrt(disc2),
        applications = nap,
        density = density,
        t_step = (time_ns() - t0) / 1e9,
        chi = maximum(bond_dims(psi); init = 1),
    )
end

"""
    qtt_split_tdvp_step!(psi, m::GPModel, t, h; kwargs...) -> NamedTuple

The stabilized / split-dissipation TDVP control: Strang splitting of
`F = -i H - gamma H` into the conservative half `-i H` (two-site TDVP, whose backward substeps are
norm-preserving) between two dissipative half-steps `exp(-gamma h/2 H)` taken by the global TT
Krylov exponential (no backward substep). `H` is `H[psi, t]` at the start of each sub-flow
(lagged within it); labelled `:split_lagged`.
"""
function qtt_split_tdvp_step!(
    psi::SymMPS,
    m::GPModel,
    t::Real,
    h::Real;
    maxdim::Int = 128,
    tol_trunc::Real = 1e-10,
    tol_krylov::Real = 1e-12,
)
    t0 = time_ns()
    diss(A, ts) = begin
        H, _ = gp_hamiltonian_op(m, A, ts)
        first(
            tt_expv(
                X -> first(op_zipup(H, X; tol = tol_krylov)),
                A,
                -m.gamma * h / 2;
                maxdim = maxdim,
                tol = tol_krylov,
                tol_trunc = tol_trunc,
            ),
        )
    end
    A = diss(qtt_cores(psi), t)
    _replace_state!(psi, qtt_mps(A))
    cons = GPFlow(m, ComplexF64(-1im), false)
    r = qtt_tdvp2_step!(
        psi,
        cons,
        t,
        h;
        density = :lagged,
        maxdim = maxdim,
        tol_trunc = tol_trunc,
        tol_krylov = tol_krylov,
    )
    A = diss(qtt_cores(psi), t + h / 2)
    _replace_state!(psi, qtt_mps(first(tt_compress(A; tol = tol_trunc, maxdim = maxdim))))
    return merge(r, (density = :split_lagged, t_step = (time_ns() - t0) / 1e9))
end
