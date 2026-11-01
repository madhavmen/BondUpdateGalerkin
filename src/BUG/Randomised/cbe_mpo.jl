# Moved from BUG-Julia@b155550:src/RSVDCBEBondUpdate/expansion/cbe_core.jl, src/RSVDCBEBondUpdate/expansion/cbe_expand.jl, src/RSVDCBEBondUpdate/expansion/cbe_options.jl, src/RSVDCBEBondUpdate/expansion/cbe_bond_update.jl, src/RSVDCBEBondUpdate/operators/mpo_env.jl, src/RSVDCBEBondUpdate/operators/mpo_fold.jl, src/RSVDCBEBondUpdate/expansion/bond_expansion.jl.

# cbe_mpo.jl: the CBE sketch on MPO environments (shared `H*Theta` and fold-omega-first), the
# MPO `cbe_expand`, the per-bond `expand_bond!` and the DMRG eigensolve `lanczos_lowest`
# (included after `cbe_bond_update.jl`).

# ── from src/RSVDCBEBondUpdate/operators/mpo_env.jl ──

# The CBE sketch, unchanged in substance: only `H Theta` comes from the MPO now. The
# projector and the sketch are the reference's (`RSVDpreBE0SiQS.m`), and nothing in the
# selection knows how `H` is stored.

"See the [`sketch_h_left`](@ref) for a term list; this is the same sketch on MPO
environments. ⚠ A standalone call rebuilds `H*Theta`; inside a sweep the shared cache in
[`_sketch_closures`](@ref) is what runs."
sketch_h_left(f::BondFrame, mpo::MPO, i::Int, lenv::MPOLink, renv::MPOLink, Om) =
    first(_sketch_closures(f, () -> apply_h_two_site(frame_theta(f), mpo, i, lenv, renv)))(Om)

"Mirror of [`sketch_h_left`](@ref) on MPO environments."
sketch_h_right(f::BondFrame, mpo::MPO, i::Int, lenv::MPOLink, renv::MPOLink, Om) =
    last(_sketch_closures(f, () -> apply_h_two_site(frame_theta(f), mpo, i, lenv, renv)))(Om)

# ── from src/RSVDCBEBondUpdate/operators/mpo_fold.jl ──

# mpo_fold.jl: fold-omega-first: the sketch that never forms `H*Theta` (from `mpo.jl`, included right after it).

# ── fold-omega-first: the sketch that never forms H*Theta ──
# Contracting `Om` in first makes the cost scale with the sketch width g, not d*chi. It is exact, since
# P_perp and Om act on disjoint legs and commute.
# The three sketches use different Om and cannot share one folded chain, so it wins only for small dex (3g < chi).

"""
    _fold_sketch_closures(f, mpo, i, lenv, renv) -> (skl, skr)

The `(skl, skr)` pair [`cbe_expand`](@ref) wants, with `Om` folded into the contraction chain
instead of applied to a finished `H*Theta`.

`skl(Om)` takes `Om` in `V0`'s layout `(g, site_r, link_r)` and returns `(link_l, site_l, g)`;
`skr(Om)` takes `U0`'s layout `(link_l, site_l, g)` and returns `(g, site_r, link_r)`. Both
return the DISCARDED-space component, i.e. with `P_perp` already applied -- identical to
[`_sketch_closures`](@ref)'s output.
"""
function _fold_sketch_closures(f::BondFrame, mpo::MPO, i::Int, lenv::MPOLink, renv::MPOLink)
    W1, W2 = mpo[i], mpo[i+1]
    # ⚠ `Theta` IS SHARED, `H*Theta` IS NOT. `frame_theta` is `U0*S0*V0` -- an `O(chi^2 d^2)`
    # object the frame already implies, with no `H` in it. Rebuilding it per call would be pure
    # waste; caching the thing this function exists NOT to build would defeat the purpose.
    Theta = frame_theta(f)                       # (ℓ_l, s_l, s_r, ℓ_r)

    # ── LEFT: fold `Om` through renv and W2, then Theta, then W1 and lenv ────────────────
    function skl(Om)
        Omd = Om'                                # (g, s_bra_r, bra_r)
        # 1. the right cap. At the boundary `renv.E` is absent and `W2`'s `w_r` is the dim-1
        #    leg left by trimming, dropped rather than contracted -- the same asymmetry
        #    `apply_h_two_site` handles, and the reason this is not one expression.
        C = if renv.E === nothing
            X = to_concrete!(contract(Omd, (2,), W2, (3,)))   # (g,bra_r,w_mid,s_ket_r,w_r)
            to_concrete(deleteSingleton(X, 5))               # (g, bra_r, w_mid, s_ket_r)
        else
            X = to_concrete!(contract(Omd, (3,), renv.E, (1,)))  # (g, s_bra_r, w, ket_r)
            # W2 = (w_mid, s_ket_r, s_bra_r, w_r); close `s_bra_r` and `w_r`.
            to_concrete!(contract(X, (2, 3), W2, (3, 4)))     # (g, ket_r, w_mid, s_ket_r)
        end
        # 2. into Theta: close the ket site and ket link on the right.
        D = to_concrete!(contract(Theta, (3, 4), C, (4, 2)))  # (ℓ_l, s_l, g, w_mid)
        # 3. W1 = (w_l, s_ket_l, s_bra_l, w_mid); close the ket site and the shared MPO leg.
        E = to_concrete!(contract(D, (2, 4), W1, (2, 4)))     # (ℓ_l, g, w_l, s_bra_l)
        Y = if lenv.E === nothing
            X = to_concrete(deleteSingleton(E, 3))           # (ℓ_l, g, s_bra_l)
            to_concrete(permutedims(X, (1, 3, 2)))           # (ℓ_l, s_bra_l, g)
        else
            # lenv.E = (bra, w, ket); close the ket link and `w_l`.
            X = to_concrete!(contract(E, (1, 3), lenv.E, (3, 2)))  # (g, s_bra_l, bra)
            _unprime(to_concrete(permutedims(X, (3, 2, 1))), 1)   # (bra, s_bra_l, g)
        end
        return _project_left(f, Y)
    end

    # ── RIGHT: the mirror -- fold `Om` through lenv and W1 first ────────────────────────
    function skr(Om)
        Omd = Om'                                # (bra_l, s_bra_l, g)
        C = if lenv.E === nothing
            X = to_concrete!(contract(Omd, (2,), W1, (3,)))   # (bra_l,g,w_l,s_ket_l,w_mid)
            to_concrete(deleteSingleton(X, 3))               # (bra_l, g, s_ket_l, w_mid)
        else
            X = to_concrete!(contract(Omd, (1,), lenv.E, (1,)))  # (s_bra_l, g, w, ket_l)
            # W1 = (w_l, s_ket_l, s_bra_l, w_mid); close `s_bra_l` and `w_l`.
            Z = to_concrete!(contract(X, (1, 3), W1, (3, 1)))    # (g, ket_l, s_ket_l, w_mid)
            to_concrete(permutedims(Z, (2, 1, 3, 4)))           # (ket_l, g, s_ket_l, w_mid)
        end
        # into Theta: close the ket link and ket site on the left.
        D = to_concrete!(contract(Theta, (1, 2), C, (1, 3)))  # (s_r, ℓ_r, g, w_mid)
        # W2 = (w_mid, s_ket_r, s_bra_r, w_r); close the ket site and the shared MPO leg.
        E = to_concrete!(contract(D, (1, 4), W2, (2, 1)))     # (ℓ_r, g, s_bra_r, w_r)
        Y = if renv.E === nothing
            X = to_concrete(deleteSingleton(E, 4))           # (ℓ_r, g, s_bra_r)
            to_concrete(permutedims(X, (2, 3, 1)))           # (g, s_bra_r, ℓ_r)
        else
            X = to_concrete!(contract(E, (1, 4), renv.E, (3, 2)))  # (g, s_bra_r, bra_r)
            _unprime(X, 3)
        end
        return _project_right(f, Y)
    end

    return skl, skr
end

"`P_perp^L Y` with `P_perp^L = I - U0 U0'`, for `Y` of layout `(link_l, site_l, g)`."
function _project_left(f::BondFrame, Y)
    c = to_concrete!(contract(f.U0', (1, 2), Y, (1, 2)))      # (bond, g)
    return to_concrete(Y - to_concrete!(contract(f.U0, (3,), c, (1,))))
end

"`Y P_perp^R` with `P_perp^R = I - V0' V0`, for `Y` of layout `(g, site_r, link_r)`."
function _project_right(f::BondFrame, Y)
    c = to_concrete!(contract(Y, (2, 3), f.V0', (2, 3)))      # (g, bond)
    return to_concrete(Y - to_concrete!(contract(c, (2,), f.V0, (1,))))
end

"""
The bond expansion at bond `i`, driven by MPO environments.

`fold_omega = true` selects [`_fold_sketch_closures`](@ref), which folds the probe into the
contraction instead of building `H*Theta`. ⚠ It makes `share_ht` MEANINGLESS -- there is no
shared object to cache -- so passing both is refused rather than silently ignoring one.
"""
function cbe_expand(
    f::BondFrame,
    mpo::MPO,
    i::Int,
    lenv::MPOLink,
    renv::MPOLink;
    # ⛔ `nothing`, NOT `true`, SO "NOT PASSED" IS DISTINGUISHABLE FROM "PASSED
    # true". With a `Bool` default the contradiction check below cannot fire at
    # all -- `share_ht` is a NAMED kwarg, so it never appears in `kwargs...` and
    # `haskey(kwargs, :share_ht)` is false even when the caller passed it. That
    # is a guard that reads as protection and provides none.
    share_ht::Union{Nothing,Bool} = nothing,
    fold_omega::Bool = false,
    kwargs...,
)
    if fold_omega
        # ⛔ REFUSE THE CONTRADICTION RATHER THAN RESOLVING IT. A caller who sets `fold_omega`
        # alone is asserting nothing about sharing and is not warned; a caller who ALSO passes
        # `share_ht` explicitly believes something false about the run they are about to time,
        # and silently honouring one of the two is how a benchmark reports the wrong arm.
        share_ht === nothing || throw(
            ArgumentError(
                "fold_omega = true never forms H*Theta, so there is nothing for share_ht to " *
                "share; pass one or the other",
            ),
        )
        return cbe_expand(f, _fold_sketch_closures(f, mpo, i, lenv, renv)...; kwargs...)
    end
    return cbe_expand(
        f,
        _sketch_closures(
            f,
            () -> apply_h_two_site(frame_theta(f), mpo, i, lenv, renv);
            share = share_ht === nothing ? true : share_ht,
        )...;
        kwargs...,
    )
end

# ── from src/RSVDCBEBondUpdate/expansion/bond_expansion.jl ──

# ── the per-bond expansion, at the reference's granularity ──
# One function for every sweep: expand the bond, keep the state unchanged, and push the far
# environment through the widened tensor. The expansion is lossless; new columns carry no amplitude yet.

"""
    BondExpansionInfo

What one [`expand_bond!`](@ref) did at one bond, with field names following the reference's `Info`
struct so Julia and MATLAB traces can be read side by side.

  - `dim_in`, `dim_out`: bond dimension before and after, counting MULTIPLETS (not comparable with a
    U(1) number under SU(2)). Equal means the expansion declined.
  - `n_new`: directions admitted. `err_pre`/`err_fnl`: weight discarded (watch `err_fnl`).
"""
struct BondExpansionInfo
    bond::Int
    dir::Symbol
    dim_in::Int
    dim_out::Int
    n_new::Int
    err_pre::Float64
    err_fnl::Float64
end

"""
    expand_bond!(psi, mpo, i, dir, lch, rch; exkw...) -> (env_far, info)

Expand bond `(i, i+1)` in place, leaving the state unchanged, and return the environment on the side
the sweep is heading toward, pushed through the widened frame. `dir = :right` widens `psi[i+1]`, so
the centre ends on `i` and the returned environment is the right one at link `i+1`; `:left` mirrors it.

`lch` and `rch` are the caller's current environments, so the call costs one push and no rebuild.
"""
function expand_bond!(psi::SymMPS, mpo::MPO, i::Int, dir::Symbol, lch, rch; exkw...)
    dir in (:right, :left) || throw(ArgumentError("dir must be :right or :left, got $dir"))
    L = length(psi)
    1 <= i <= L - 1 || throw(ArgumentError("bond must be in 1:$(L - 1), got $i"))
    psi.center in (i, i + 1) || throw(
        ArgumentError(
            "expand_bond! needs the orthogonality centre on site $i or $(i + 1), " *
            "found it on $(psi.center)",
        ),
    )

    dim_in = leg_dim(psi[i], 3)
    f = _frame_from(psi[i], psi[i+1])
    ex = cbe_expand(f, mpo, i, lch, rch; exkw...)

    if dir === :right
        _absorb_right!(psi, i, ex.V_ex)              # psi[i+1] = V_ex, centre -> i
        env = push_right_channels(rch, mpo, psi[i+1], i + 1)
        n_new = ex.n_new_r
    else
        _absorb_left!(psi, i, ex.U_ex)               # psi[i] = U_ex, centre -> i+1
        env = push_left_channels(lch, mpo, psi[i], i)
        n_new = ex.n_new_l
    end

    return env, BondExpansionInfo(i, dir, dim_in, leg_dim(psi[i], 3), n_new, ex.err_pre, ex.err_fnl)
end

# ── the local eigensolve the DMRG sweep needs ──
# Same Krylov space as `lanczos_expv`, reduced to the lowest Ritz pair. It adds a Ritz-residual exit
# (beta_k |c_k|): `lanczos_expv` stops only on breakdown, which wastes the full `maxiter` in DMRG.

"""
    lanczos_lowest(apply, x; maxiter=30, tol=1e-13, restol=1e-10, reorth=true)
        -> (vector, ritz_value, krylov_dim)

Lowest eigenvector of a Hermitian `apply(v) = A*v` in the Krylov space of `x`, normalised. The Ritz
value is returned too, since recomputing it costs another application. It is a variational upper
bound, because the subspace is fixed.

`reorth = true` is the default: without it the recurrence loses orthogonality silently, giving a Ritz
value below the true eigenvalue that looks like better convergence.
"""
function lanczos_lowest(
    apply,
    x;
    maxiter::Int = 30,
    tol::Float64 = 1e-13,
    restol::Float64 = 1e-10,
    reorth::Bool = true,
)
    xc = to_concrete(ComplexF64(1.0) * x)
    beta0 = norm(xc)
    beta0 == 0 && throw(ArgumentError("lanczos_lowest: start vector has zero norm"))

    v = to_concrete((1.0 / beta0) * xc)
    basis = Any[v]
    alpha = Float64[]
    betas = Float64[]

    w = apply(v)
    a = real(tensor_inner(v, w))
    push!(alpha, a)
    w = to_concrete(w + (-a) * v)

    coeff, value = _reduce_krylov(LowestEigen(), alpha, betas, beta0)
    for _ = 2:maxiter
        b = norm(w)
        b < tol && break
        push!(betas, b)
        v = to_concrete((1.0 / b) * w)
        push!(basis, v)
        w = apply(v)
        a = real(tensor_inner(v, w))
        push!(alpha, a)
        w = w + (-a) * v + (-b) * basis[end-1]
        if reorth
            for _pass = 1:2, u in basis
                w = w - tensor_inner(u, w) * u
            end
        end
        w = to_concrete(w)
        coeff, value = _reduce_krylov(LowestEigen(), alpha, betas, beta0)
        # Ritz residual: `beta_k |c_k|` bounds `||A v - theta v||`. Cheap -- both factors are
        # already computed -- and it is the ANSWER's convergence, not the basis's.
        b * abs(coeff[end]) < restol && break
    end

    out = to_concrete(coeff[1] * basis[1])
    for k = 2:length(coeff)
        out = to_concrete(out + coeff[k] * basis[k])
    end
    n = norm(out)
    n > 0 && (out = to_concrete((1.0 / n) * out))
    return out, real(value), length(alpha)
end
