# Moved from BUG-Julia@b155550:src/RSVDCBEBondUpdate/operators/mpo_env.jl (the MPO sketch),
# src/RSVDCBEBondUpdate/operators/mpo_fold.jl and src/RSVDCBEBondUpdate/expansion/bond_expansion.jl.

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
sketch_h_left(frame::BondFrame, mpo::MPO, bond::Int, lenv::MPOLink, renv::MPOLink, Om) = first(
    _sketch_closures(frame, () -> apply_h_two_site(frame_theta(frame), mpo, bond, lenv, renv)),
)(Om)

"Mirror of [`sketch_h_left`](@ref) on MPO environments."
sketch_h_right(frame::BondFrame, mpo::MPO, bond::Int, lenv::MPOLink, renv::MPOLink, Om) = last(
    _sketch_closures(frame, () -> apply_h_two_site(frame_theta(frame), mpo, bond, lenv, renv)),
)(Om)

# ── from src/RSVDCBEBondUpdate/operators/mpo_fold.jl ──

# ── fold-omega-first: the sketch that never forms H*Theta ──
# Contracting `Om` in first makes the cost scale with the sketch width g, not d*chi. It is exact,
# since P_perp and Om act on disjoint legs and commute.
# The three sketches use different Om and cannot share one folded chain, so it wins only for small
# dex (3g < chi).

"""
    _fold_sketch_closures(frame, mpo, bond, lenv, renv) -> (skl, skr)

The `(skl, skr)` pair [`cbe_expand`](@ref) wants, with `Om` folded into the contraction chain
instead of applied to a finished `H*Theta`.

`skl(Om)` takes `Om` in `V0`'s layout `(g, site_r, link_r)` and returns `(link_l, site_l, g)`;
`skr(Om)` takes `U0`'s layout `(link_l, site_l, g)` and returns `(g, site_r, link_r)`. Both
return the DISCARDED-space component, i.e. with `P_perp` already applied -- identical to
[`_sketch_closures`](@ref)'s output.
"""
function _fold_sketch_closures(
    frame::BondFrame,
    mpo::MPO,
    bond::Int,
    lenv::MPOLink,
    renv::MPOLink,
)
    left_mpo, right_mpo = mpo[bond], mpo[bond+1]
    # ⚠ `Theta` IS SHARED, `H*Theta` IS NOT. `frame_theta` is `U0*S0*V0` -- an `O(chi^2 d^2)`
    # object the frame already implies, with no `H` in it. Rebuilding it per call would be pure
    # waste; caching the thing this function exists NOT to build would defeat the purpose.
    Theta = frame_theta(frame)                       # (ℓ_l, s_l, s_r, ℓ_r)

    # ── LEFT: fold `Om` through renv and right_mpo, then Theta, then left_mpo and lenv ─────
    function skl(Om)
        Omd = Om'                                # (g, s_bra_r, bra_r)
        # 1. the right cap. At the boundary `renv.E` is absent and `right_mpo`'s `w_r` is the
        #    dim-1 leg left by trimming, dropped rather than contracted -- the same asymmetry
        #    `apply_h_two_site` handles, and the reason this is not one expression.
        right_cap = if renv.E === nothing
            # (g, bra_r, w_mid, s_ket_r, w_r)
            partial = to_concrete!(contract(Omd, (2,), right_mpo, (3,)))
            to_concrete(deleteSingleton(partial, 5))         # (g, bra_r, w_mid, s_ket_r)
        else
            partial = to_concrete!(contract(Omd, (3,), renv.E, (1,)))  # (g, s_bra_r, w, ket_r)
            # right_mpo = (w_mid, s_ket_r, s_bra_r, w_r); close `s_bra_r` and `w_r`.
            # -> (g, ket_r, w_mid, s_ket_r)
            to_concrete!(contract(partial, (2, 3), right_mpo, (3, 4)))
        end
        # 2. into Theta: close the ket site and ket link on the right. -> (ℓ_l, s_l, g, w_mid)
        theta_capped = to_concrete!(contract(Theta, (3, 4), right_cap, (4, 2)))
        # 3. left_mpo = (w_l, s_ket_l, s_bra_l, w_mid); close the ket site and the shared MPO leg.
        #    -> (ℓ_l, g, w_l, s_bra_l)
        with_mpo = to_concrete!(contract(theta_capped, (2, 4), left_mpo, (2, 4)))
        sketch = if lenv.E === nothing
            partial = to_concrete(deleteSingleton(with_mpo, 3))   # (ℓ_l, g, s_bra_l)
            to_concrete(permutedims(partial, (1, 3, 2)))          # (ℓ_l, s_bra_l, g)
        else
            # lenv.E = (bra, w, ket); close the ket link and `w_l`. -> (g, s_bra_l, bra)
            partial = to_concrete!(contract(with_mpo, (1, 3), lenv.E, (3, 2)))
            _unprime(to_concrete(permutedims(partial, (3, 2, 1))), 1)   # (bra, s_bra_l, g)
        end
        return _project_left(frame, sketch)
    end

    # ── RIGHT: the mirror -- fold `Om` through lenv and left_mpo first ──────────────────
    function skr(Om)
        Omd = Om'                                # (bra_l, s_bra_l, g)
        left_cap = if lenv.E === nothing
            # (bra_l, g, w_l, s_ket_l, w_mid)
            partial = to_concrete!(contract(Omd, (2,), left_mpo, (3,)))
            to_concrete(deleteSingleton(partial, 3))         # (bra_l, g, s_ket_l, w_mid)
        else
            partial = to_concrete!(contract(Omd, (1,), lenv.E, (1,)))  # (s_bra_l, g, w, ket_l)
            # left_mpo = (w_l, s_ket_l, s_bra_l, w_mid); close `s_bra_l` and `w_l`.
            # -> (g, ket_l, s_ket_l, w_mid)
            capped = to_concrete!(contract(partial, (1, 3), left_mpo, (3, 1)))
            to_concrete(permutedims(capped, (2, 1, 3, 4)))  # (ket_l, g, s_ket_l, w_mid)
        end
        # into Theta: close the ket link and ket site on the left. -> (s_r, ℓ_r, g, w_mid)
        theta_capped = to_concrete!(contract(Theta, (1, 2), left_cap, (1, 3)))
        # right_mpo = (w_mid, s_ket_r, s_bra_r, w_r); close the ket site and the shared MPO leg.
        # -> (ℓ_r, g, s_bra_r, w_r)
        with_mpo = to_concrete!(contract(theta_capped, (1, 4), right_mpo, (2, 1)))
        sketch = if renv.E === nothing
            partial = to_concrete(deleteSingleton(with_mpo, 4))   # (ℓ_r, g, s_bra_r)
            to_concrete(permutedims(partial, (2, 3, 1)))          # (g, s_bra_r, ℓ_r)
        else
            # -> (g, s_bra_r, bra_r)
            partial = to_concrete!(contract(with_mpo, (1, 4), renv.E, (3, 2)))
            _unprime(partial, 3)
        end
        return _project_right(frame, sketch)
    end

    return skl, skr
end

"`P_perp^L sketch` with `P_perp^L = I - U0 U0'`, for `sketch` of layout `(link_l, site_l, g)`."
function _project_left(frame::BondFrame, sketch)
    kept_coeffs = to_concrete!(contract(frame.U0', (1, 2), sketch, (1, 2)))      # (bond, g)
    return to_concrete(sketch - to_concrete!(contract(frame.U0, (3,), kept_coeffs, (1,))))
end

"`sketch P_perp^R` with `P_perp^R = I - V0' V0`, for `sketch` of layout `(g, site_r, link_r)`."
function _project_right(frame::BondFrame, sketch)
    kept_coeffs = to_concrete!(contract(sketch, (2, 3), frame.V0', (2, 3)))      # (g, bond)
    return to_concrete(sketch - to_concrete!(contract(kept_coeffs, (2,), frame.V0, (1,))))
end

"""
    cbe_expand(frame, mpo, bond, lenv, renv; share_ht, fold_omega, kwargs...) -> CBEExpansion

The bond expansion at bond `(bond, bond+1)`, driven by MPO environments.

`fold_omega = true` selects [`_fold_sketch_closures`](@ref), which folds the probe into the
contraction instead of building `H*Theta`. ⚠ It makes `share_ht` MEANINGLESS -- there is no
shared object to cache -- so passing both is refused rather than silently ignoring one.
"""
function cbe_expand(
    frame::BondFrame,
    mpo::MPO,
    bond::Int,
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
        return cbe_expand(
            frame,
            _fold_sketch_closures(frame, mpo, bond, lenv, renv)...;
            kwargs...,
        )
    end
    return cbe_expand(
        frame,
        _sketch_closures(
            frame,
            () -> apply_h_two_site(frame_theta(frame), mpo, bond, lenv, renv);
            share = share_ht === nothing ? true : share_ht,
        )...;
        kwargs...,
    )
end

# ── from src/RSVDCBEBondUpdate/expansion/bond_expansion.jl ──

# ── the per-bond expansion, at the reference's granularity ──
# One function for every sweep: expand the bond, keep the state unchanged, and push the far
# environment through the widened tensor. The expansion is lossless; new columns carry no amplitude
# yet.

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
    expand_bond!(psi, mpo, bond, dir, lch, rch; exkw...) -> (env_far, info)

Expand bond `(bond, bond+1)` in place, leaving the state unchanged, and return the environment on
the side the sweep is heading toward, pushed through the widened frame. `dir = :right` widens
`psi[bond+1]`, so the centre ends on `bond` and the returned environment is the right one at link
`bond+1`; `:left` mirrors it.

`lch` and `rch` are the caller's current environments, so the call costs one push and no rebuild.
"""
function expand_bond!(psi::SymMPS, mpo::MPO, bond::Int, dir::Symbol, lch, rch; exkw...)
    dir in (:right, :left) || throw(ArgumentError("dir must be :right or :left, got $dir"))
    site_count = length(psi)
    1 <= bond <= site_count - 1 ||
        throw(ArgumentError("bond must be in 1:$(site_count - 1), got $bond"))
    psi.center in (bond, bond + 1) || throw(
        ArgumentError(
            "expand_bond! needs the orthogonality centre on site $bond or $(bond + 1), " *
            "found it on $(psi.center)",
        ),
    )

    dim_in = leg_dim(psi[bond], 3)
    frame = _frame_from(psi[bond], psi[bond+1])
    ex = cbe_expand(frame, mpo, bond, lch, rch; exkw...)

    if dir === :right
        _absorb_right!(psi, bond, ex.V_ex)              # psi[bond+1] = V_ex, centre -> bond
        env = push_right_channels(rch, mpo, psi[bond+1], bond + 1)
        n_new = ex.n_new_r
    else
        _absorb_left!(psi, bond, ex.U_ex)               # psi[bond] = U_ex, centre -> bond+1
        env = push_left_channels(lch, mpo, psi[bond], bond)
        n_new = ex.n_new_l
    end

    return env,
    BondExpansionInfo(bond, dir, dim_in, leg_dim(psi[bond], 3), n_new, ex.err_pre, ex.err_fnl)
end

# ── the local eigensolve the DMRG sweep needs ──
# Same Krylov space as `lanczos_expv`, reduced to the lowest Ritz pair. It adds a Ritz-residual
# exit (beta_k |c_k|): `lanczos_expv` stops only on breakdown, which wastes the full `maxiter` in
# DMRG.

"""
    lanczos_lowest(apply, start_vector; maxiter=30, tol=1e-13, restol=1e-10, reorth=true)
        -> (vector, ritz_value, krylov_dim)

Lowest eigenvector of a Hermitian `apply(vector) = A*vector` in the Krylov space of
`start_vector`, normalised. The Ritz value is returned too, since recomputing it costs another
application. It is a variational upper bound, because the subspace is fixed.

`reorth = true` is the default: without it the recurrence loses orthogonality silently, giving a
Ritz value below the true eigenvalue that looks like better convergence.
"""
function lanczos_lowest(
    apply,
    start_vector;
    maxiter::Int = 30,
    tol::Float64 = 1e-13,
    restol::Float64 = 1e-10,
    reorth::Bool = true,
)
    xc = to_concrete(ComplexF64(1.0) * start_vector)
    beta0 = norm(xc)
    beta0 == 0 && throw(ArgumentError("lanczos_lowest: start vector has zero norm"))

    lanczos_vector = to_concrete((1.0 / beta0) * xc)
    basis = Any[lanczos_vector]
    alpha = Float64[]
    betas = Float64[]

    residual = apply(lanczos_vector)
    diagonal_entry = real(tensor_inner(lanczos_vector, residual))
    push!(alpha, diagonal_entry)
    residual = to_concrete(residual + (-diagonal_entry) * lanczos_vector)

    coeff, value = _reduce_krylov(LowestEigen(), alpha, betas, beta0)
    for _ = 2:maxiter
        residual_norm = norm(residual)
        residual_norm < tol && break
        push!(betas, residual_norm)
        lanczos_vector = to_concrete((1.0 / residual_norm) * residual)
        push!(basis, lanczos_vector)
        # Three-term recurrence, optionally followed by two full reorthogonalisation passes.
        residual = apply(lanczos_vector)
        diagonal_entry = real(tensor_inner(lanczos_vector, residual))
        push!(alpha, diagonal_entry)
        residual =
            residual + (-diagonal_entry) * lanczos_vector + (-residual_norm) * basis[end-1]
        if reorth
            for _pass = 1:2, basis_vector in basis
                residual = residual - tensor_inner(basis_vector, residual) * basis_vector
            end
        end
        residual = to_concrete(residual)
        coeff, value = _reduce_krylov(LowestEigen(), alpha, betas, beta0)
        # Ritz residual: `beta_k |c_k|` bounds `||A v_k - theta v_k||`. Cheap -- both factors are
        # already computed -- and it is the ANSWER's convergence, not the basis's.
        residual_norm * abs(coeff[end]) < restol && break
    end

    # Ritz vector: the Lanczos basis weighted by the lowest eigenvector of the tridiagonal.
    out = to_concrete(coeff[1] * basis[1])
    for column = 2:length(coeff)
        out = to_concrete(out + coeff[column] * basis[column])
    end
    out_norm = norm(out)
    out_norm > 0 && (out = to_concrete((1.0 / out_norm) * out))
    return out, real(value), length(alpha)
end
