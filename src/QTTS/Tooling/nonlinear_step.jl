# One rank-adaptive BUG step of a flow on QTT states.
#
# The basis comes from the BUG engine (`cbe_bug_step!` with `frames_report`), the root from the
# generic stage solver (`rk_stage_solve`) or, for a linear flow, the engine's own rational root;
# this file holds the QTT glue (dense frames, kept stage states, compression).

"Safety factor on the local-Krylov share of `tol_basis` in `qtt_bug_step!`."
const _QTT_KRYLOV_MARGIN = 0.5

"""
Arnoldi basis cap of the unshifted rational root (`stage_solver = :krylov_rational`), a numerical
setting: one stage, no substep split. On the full 256^2 Heston grid from the call payoff at
h = 0.02, Radau IIA(2) reaches 3e-9 at 100 vectors and 1e-10 at 150; the Galerkin operator is
smaller.
"""
const _QTT_RATIONAL_MAXITER = 200

"""
Smallest per-step tolerance `qtt_step_tolerance` hands out: below it the Newton residual and the
relative discarded weight are at rounding level and cannot be met reliably.
"""
const _QTT_TOL_FLOOR = 1e-14

"""
    qtt_step_tolerance(tol, h, order; scaling = :order, h_ref = 0.1) -> Float64

The per-step value of a tolerance passed to [`qtt_bug_step!`](@ref). An order-p step's local
error is `O(h^(p+1))`; a tolerance held fixed while `h` shrinks puts a floor under the step test
(BUG order 4 stopped improving at ~2e-8 below h = 0.05). `scaling = :order` multiplies `tol` by
`(|h| / h_ref)^(p+1)`, so the error the tolerances allow per step falls at the method's own rate.
`h_ref` is the step at which the passed value applies; a longer step keeps `tol` (never looser).
The result never goes below `_QTT_TOL_FLOOR` (a `tol` already below it is returned as it is).
`scaling = :fixed` returns `tol`.
"""
function qtt_step_tolerance(
    tol::Real,
    h::Real,
    order::Int;
    scaling::Symbol = :order,
    h_ref::Real = 0.1,
)
    scaling in (:order, :fixed) ||
        throw(ArgumentError("tol_scaling must be :order or :fixed, got :$scaling"))
    h_ref > 0 || throw(ArgumentError("tol_h_ref must be positive, got $h_ref"))
    (scaling === :fixed || iszero(tol)) && return Float64(tol)
    factor = min(1.0, (abs(Float64(h)) / Float64(h_ref))^(order + 1))
    return max(Float64(tol) * factor, min(Float64(tol), _QTT_TOL_FLOOR))
end

"""
    qtt_bug_step!(psi, fl, t, h; order = 2, root = qtt_root_for_order(order), npc = order - 1,
                  tol_stage = 1e-10, tol_basis = 1e-6, tol_trunc = 1e-8, maxdim = 128,
                  keep_tol = 1e-10, keep = [], stage_solver = :auto, stage_basis = :predictor,
                  stage_envs = :summand, stage_density = :variational, stage_recycle = 0,
                  stage_gmres_tol = nothing, stage_jacobian_psi2 = :fresh,
                  stage_field_rho = :reuse, tol_scaling = :order, tol_h_ref = 0.1, kwargs...)
        -> NamedTuple

One rank-adaptive BUG step of the full flow `fl` from `t` to `t + h`, in place. With
`stage_solver = :newton_gmres`:

1. the basis: `cbe_bug_step!` at `order` on the predictor generator `H[psi_n, t_n]` (its DD-CBE,
   local Krylov and order-p frontier closure), frames only (`frames_only`), the frames taken from
   `frames_report`;
2. the root: `S_n = <frames|psi_n>`, the step input projected, and the nonlinear `root` stage
   equations solved with the density rebuilt at every stage iterate ([`rk_stage_solve`](@ref));
3. `npc` basis corrections: the frames are rebuilt with the stage states `Y_j` and their vector
   fields `F(t_j, Y_j)` kept (the full-space `H[Y_j, t_j] Y_j`: the root solve's projected fields
   lie in the old frames and would add no direction), and the root re-solved from the same `S_n`.
   An unconverged root solve skips the corrections (the caller rejects the step);
4. one compression of `W Y_s Z` (`truncate_recursive!` at `tol_trunc`, `maxdim`).

`stage_solver = :auto` is `:krylov_rational` for a `LinearFlow` and `:newton_gmres` otherwise.
`:krylov_rational` is `cbe_bug_step!` with the root method `root` (the trapezoid / Radau IIA
stability function `R(h A)` on the Arnoldi basis of the Galerkin operator) unshifted
(`root_shift = false`), so the step is `R(h A_G) S_n` with its decay kept, then compressed: for a
linear flow that is the implicit RK step in the frames, and `npc` does not apply. `:krylov_root` is
an ablation: `cbe_bug_step!` on `H[psi_n, t_n]` with its exponential root, which holds the density
at `t_n` through the step (first order in the nonlinearity).

`stage_basis` is how the correction rounds' `Y_j` and `F_j` enter the basis: `:predictor`
(default) compresses them at the step's `tol_trunc` and admits them at its `tol_basis` in the
units of the step error (`cbe_bug_step!`'s `keep_tols`: a state's missing part enters the step at
weight 1, a field's at `|h pref| max_i |a_ij|`); `:exact` compresses them at `keep_tol` and keeps
them exactly. Both are compressed at most `maxdim` wide. `stage_field_rho = :reuse` builds
`H[Y_j, t_j]` from the stage slot's last density fit, the converged iterate's density (`:rebuild`
fits it again; `stage_envs = :full` keeps no fits, so it always rebuilds).
`stage_envs`, `stage_density`, `stage_recycle`, `stage_gmres_tol` and `stage_jacobian_psi2` are
`rk_stage_solve`'s `envs`, `density`, `gmres_recycle`, `gmres_tol` and `jacobian_psi2`; the
density fits are warm-started from `psi_n`'s density and then from each stage slot's last fit,
across all of the step's root solves.
`tol_scaling`/`tol_h_ref`: `tol_basis`, `tol_trunc`, `tol_stage` (and a passed `tol_krylov`) are
applied as [`qtt_step_tolerance`](@ref) values; `tol_step` in the result records them.

The temporal order is a claim to be measured on the whole update, never assumed from `root`.
`kwargs` go to `cbe_bug_step!` (`tol_krylov` defaults to `_QTT_KRYLOV_MARGIN * tol_basis`).
Returns the stage statistics, `converged`, the frame widths, the discarded weight, the frame
audits (`miss_input`, `miss_keep`, `frame_defect`), the basis builds' DD ledger `dd`, `chi`, the
Newton and GMRES totals, and the step's time ledger: `t_step` split into `t_setup` (`H[psi_n]`
and its MPO), `t_basis`, `t_root` (with the density seconds `t_rho` and GMRES seconds `t_gmres`),
`t_keep` (the correction rounds' stage states and fields, with their density seconds `t_keep_rho`)
and `t_trunc`, with the shares `f_rho` (all density seconds) and `f_cbe` (`dd.t_cbe`).
"""
function qtt_bug_step!(
    psi::SymMPS,
    fl::QTTFlow,
    t::Real,
    h::Real;
    order::Int = 2,
    root::Symbol = qtt_root_for_order(order),
    npc::Int = order - 1,
    tol_stage::Real = 1e-10,
    tol_basis::Real = 1e-6,
    tol_trunc::Real = 1e-8,
    maxdim::Int = 128,
    keep_tol::Real = 1e-10,
    keep::AbstractVector = Any[],
    stage_solver::Symbol = :auto,
    stage_basis::Symbol = :predictor,
    stage_envs::Symbol = :summand,
    stage_density::Symbol = :variational,
    stage_recycle::Int = 0,
    stage_gmres_tol::Union{Nothing,Real} = nothing,
    stage_jacobian_psi2::Symbol = :fresh,
    stage_field_rho::Symbol = :reuse,
    tol_scaling::Symbol = :order,
    tol_h_ref::Real = 0.1,
    kwargs...,
)
    step_start = time_ns()
    # the step's tolerances in the units of its O(h^(order+1)) local error
    step_tol(tol) = qtt_step_tolerance(tol, h, order; scaling = tol_scaling, h_ref = tol_h_ref)
    tol_step = (
        basis = step_tol(tol_basis),
        trunc = step_tol(tol_trunc),
        stage = step_tol(tol_stage),
        scaling = tol_scaling,
        h_ref = Float64(tol_h_ref),
    )
    stage_field_rho in (:reuse, :rebuild) ||
        throw(ArgumentError("stage_field_rho must be :reuse or :rebuild, got :$stage_field_rho"))
    stage_basis in (:predictor, :exact) ||
        throw(ArgumentError("stage_basis must be :predictor or :exact, got :$stage_basis"))
    solver =
        stage_solver !== :auto ? stage_solver : fl isa LinearFlow ? :krylov_rational : :newton_gmres
    solver in (:newton_gmres, :krylov_rational, :krylov_root) || throw(
        ArgumentError(
            "stage_solver must be :auto, :newton_gmres, :krylov_rational or :krylov_root, " *
            "got :$stage_solver",
        ),
    )
    # R(h A) is the RK step only when the vector field is the fixed linear A
    solver === :krylov_rational &&
        !(fl isa LinearFlow) &&
        throw(
            ArgumentError("stage_solver = :krylov_rational needs a LinearFlow, got $(typeof(fl))"),
        )
    input_cores = qtt_cores(psi)
    predictor, predictor_info = flow_op(fl, input_cores, t)
    # density warm starts for the stage fits: psi_n's own density, then each slot's last fit
    warm = Dict{Any,Any}()
    input_density = get(predictor_info, :rho, nothing)
    if input_density !== nothing && solver === :newton_gmres
        for slot in Any[(1:length(RK_TABLEAUX[root].c))..., :precond]
            warm[(slot, :rho)] = input_density
        end
    end
    stage_keywords = (
        envs = stage_envs,
        density = stage_density,
        warm = warm,
        gmres_recycle = stage_recycle,
        gmres_tol = stage_gmres_tol,
        jacobian_psi2 = stage_jacobian_psi2,
    )
    # The frames' generator is shifted by the state's mean <H>: the Krylov spaces are unchanged, but
    # a stiff step's exponential root no longer underflows to an exactly zero matrix.
    mean_energy = real(_op_expect(predictor, input_cores)) / max(tt_norm(input_cores)^2, eps())
    predictor_mpo = qtt_to_mpo(predictor)
    t_setup = (time_ns() - step_start) / 1e9
    tau = ComplexF64(fl.pref * h)
    tol_krylov = step_tol(Float64(get(kwargs, :tol_krylov, _QTT_KRYLOV_MARGIN * tol_basis)))
    engine_kwargs = Base.structdiff(values(kwargs), NamedTuple{(:tol_krylov,)})
    nfallback = 0
    krylov_states = nothing
    # every frame set is audited: the step input and each keep state must lie in the span
    # (Galerkin invariants, e.g. Heston mass, hold only then) and the cores must be isometric
    keep_cores = [qtt_cores(state) for state in keep]
    miss_input, miss_keep, defect = 0.0, 0.0, 0.0
    function audit!(frames)
        miss_input = max(miss_input, frame_miss(frames, input_cores))
        miss_keep =
            max(miss_keep, maximum(kept -> frame_miss(frames, kept), keep_cores; init = 0.0))
        defect = max(defect, frame_defect(frames))
        return frames
    end
    # the basis builds' DD work and verdicts, summed (max for the residuals and the temporary rank)
    dd = (
        probes = 0,
        holdout = 0,
        selected = 0,
        resid_left = 0.0,
        resid_right = 0.0,
        r_aug = 0,
        t_cbe = 0.0,
    )
    function record_dd!(info)
        ledger = info.ledger
        dd = (
            probes = dd.probes + ledger.n_probes,
            holdout = dd.holdout + ledger.n_holdout,
            selected = dd.selected + ledger.n_selected,
            resid_left = max(dd.resid_left, ledger.max_sample_l),
            resid_right = max(dd.resid_right, ledger.max_sample_r),
            r_aug = max(dd.r_aug, ledger.r_aug),
            t_cbe = dd.t_cbe + info.t_cbe,
        )
    end
    # `final_root`: the engine's own root is the step (`nothing` = frames only, the stage path);
    # its evolved copy and info are kept in `engine_state`/`engine_info`
    engine_state, engine_info = Ref{Any}(nothing), Ref{Any}(nothing)
    # the <H> shift keeps the exponential root from underflowing; a rational root (trapezoid,
    # Radau) is applied unshifted, so its stability function is the plain R(h G)
    shift_of(final_root) = final_root === nothing || final_root === :exp ? mean_energy : 0.0
    # `tols`: one admission tolerance per kept state (`keep_tols`; empty = exact)
    function build_frames(keep_states; final_root = nothing, tols = Float64[])
        report = Ref{Any}(nothing)
        root_keywords =
            final_root === nothing ? NamedTuple() :
            final_root === :exp ? (root_method = :exp, tol_root = tol_step.stage) :
            (
                root_method = final_root,
                tol_root = tol_step.stage,
                root_shift = false,
                root_maxiter = _QTT_RATIONAL_MAXITER,
            )
        function run(basis_order, states, state_tols)
            engine_state[] = copy(psi)
            engine_info[] = cbe_bug_step!(
                engine_state[],
                predictor_mpo,
                tau;
                order = basis_order,
                tol_basis = tol_step.basis,
                tol_trunc = tol_step.trunc,
                maxdim = maxdim,
                hermitian = false,
                frames_report = report,
                keep_states = states,
                keep_tols = state_tols,
                # the stage path solves its own root on these frames
                frames_only = final_root === nothing,
                truncate = final_root !== nothing,
                eshift = shift_of(final_root),
                tol_krylov = tol_krylov,
                root_keywords...,
                engine_kwargs...,
            )
        end
        info = try
            run(order, keep_states, tols)
        catch err
            # The order-p frontier closure can carry an old frame wider than a near-saturated
            # site's new local space. The equivalent order-p basis -- order-1 frames that keep the
            # global Krylov vectors (tau H)^j psi, j < p -- is used instead and counted.
            order > 1 && occursin("cannot exceed its local space", sprint(showerror, err)) ||
                rethrow()
            nfallback += 1
            # built at most maxdim wide and admitted at the `keep_tol` they were compressed to
            if krylov_states === nothing
                krylov_states = global_krylov_states(
                    predictor_mpo,
                    psi,
                    order - 1;
                    tol_trunc = Float64(keep_tol),
                    maxdim = maxdim,
                )
            end
            run(
                1,
                vcat(Any[keep_states...], Any[krylov_states...]),
                vcat(
                    isempty(tols) ? zeros(length(keep_states)) : Float64.(tols),
                    fill(Float64(keep_tol), length(krylov_states)),
                ),
            )
        end
        record_dd!(info)
        return audit!(QTTFrames(report[]))
    end
    # the step result fields every solver reports
    function step_result(frames, stats, discarded, rank_capped, timings, used_root)
        t_step = (time_ns() - step_start) / 1e9
        t_rho_root = sum(stats_entry.t_rho for stats_entry in stats)
        return (
            stats = stats,
            converged = all(stats_entry.converged for stats_entry in stats),
            frame_width = (size(frames.W[end], 3), size(frames.Z[1], 1)),
            discarded = discarded,
            rank_capped = rank_capped,
            t_basis = timings.basis,
            t_root = timings.root,
            t_rho0 = predictor_info.t_rho,
            basis_fallback = nfallback,
            miss_input = miss_input,
            miss_keep = miss_keep,
            frame_defect = defect,
            t_step = t_step,
            chi = maximum(bond_dims(psi); init = 1),
            root = used_root,
            order = order,
            solver = solver,
            tol_step = tol_step,
            dd = dd,
            t_setup = t_setup,
            t_keep = timings.keep,
            t_keep_rho = timings.keep_rho,
            t_trunc = timings.trunc,
            t_rho = t_rho_root,
            f_rho = (predictor_info.t_rho + t_rho_root + timings.keep_rho) / max(t_step, eps()),
            f_cbe = dd.t_cbe / max(t_step, eps()),
            newton = sum(stats_entry.newton for stats_entry in stats),
            gmres = sum(stats_entry.gmres for stats_entry in stats),
            t_gmres = sum(stats_entry.t_gmres for stats_entry in stats),
            recycled = sum(stats_entry.recycled for stats_entry in stats),
            env_builds = sum(max(stats_entry.env_builds, 0) for stats_entry in stats),
            psi2_fits = sum(max(stats_entry.psi2_fits, 0) for stats_entry in stats),
        )
    end
    basis_start = time_ns()
    if solver !== :newton_gmres
        # the engine's root is the step: the rational root `root` of the linear flow, unshifted,
        # or the exponential root of the frozen generator (the ablation)
        final_root = solver === :krylov_rational ? root : :exp
        frames = build_frames(Any[keep...]; final_root = final_root)
        info, out = engine_info[], engine_state[]
        # the shifted exponential is exp(tau (H - <H>)): restore the scalar exp(tau <H>)
        shift = shift_of(final_root)
        iszero(shift) ||
            (out = qtt_mps(tt_scale(qtt_cores(out), exp(tau * shift)); center = out.center))
        _replace_state!(psi, out)
        root_error = info.ledger.expv_err
        engine_stats = (
            newton = 0,
            gmres = 0,
            residual = root_error,
            converged = solver === :krylov_root || root_error <= tol_step.stage,
            nfield = 0,
            t_rho = 0.0,
            t_gmres = 0.0,
            mu = NaN,
            precond = :none,
            envs = :none,
            env_builds = 0,
            psi2_fits = 0,
            recycled = 0,
            krylov = info.ledger.expv_iters,
            fields = Any[],
        )
        timings = (
            basis = info.t_step - info.t_root,
            root = info.t_root,
            keep = 0.0,
            keep_rho = 0.0,
            trunc = info.t_trunc,
        )
        return step_result(
            frames,
            Any[engine_stats],
            info.discarded,
            maximum(bond_dims(psi); init = 1) >= maxdim,
            timings,
            final_root,
        )
    end
    frames = build_frames(Any[keep...])
    t_basis = (time_ns() - basis_start) / 1e9
    tableau = RK_TABLEAUX[root]
    # A kept stage vector is admitted in the units of the step's error, as the DD tests measure:
    # a stage state's missing part enters the step at weight 1, a stage field's at
    # |h pref| max_i |a_ij| (F_j enters stage i as h pref a_ij F_j). The engine takes the tolerance
    # relative to the vector's own norm, hence the ratios.
    stage_keep_tol = stage_basis === :predictor ? tol_step.trunc : Float64(keep_tol)
    stage_admit_tol = stage_basis === :predictor ? tol_step.basis : 0.0
    input_norm = tt_norm(input_cores)
    field_weights = abs(tau) .* vec(maximum(abs, tableau.A; dims = 1))
    admit_tol(weight, cores) =
        iszero(stage_admit_tol) ? 0.0 :
        stage_admit_tol * input_norm / max(weight * tt_norm(cores), floatmin(Float64))
    stats = Any[]
    root_start = time_ns()
    Y, stage_stats = rk_stage_solve(
        fl,
        frames,
        frame_project(frames, nothing, input_cores),
        t,
        h,
        root;
        tol = tol_step.stage,
        stage_keywords...,
    )
    t_root = (time_ns() - root_start) / 1e9
    push!(stats, stage_stats)
    t_keep, t_keep_rho = 0.0, 0.0
    # each stage slot's last density fit is its converged iterate's density (summand path only)
    reuse_density =
        stage_field_rho === :reuse && stage_envs === :summand && _density_coupling(fl) != 0
    for _ = 1:npc
        # an unconverged root solve already rejects the step
        stage_stats.converged || break
        keep_start = time_ns()
        # the caller's `keep` states stay exact (Heston mass); the stages enter per `stage_basis`
        stage_keep = Any[keep...]
        stage_keep_tols = zeros(length(keep))
        for (slot, stage_root) in enumerate(Y)
            stage_cores = first(
                tt_compress(frame_state(frames, stage_root); tol = stage_keep_tol, maxdim = maxdim),
            )
            push!(stage_keep, qtt_mps(stage_cores))
            push!(stage_keep_tols, admit_tol(1.0, stage_cores))
            # F(t_j, Y_j) in the full space: the span of {Y_j, H[Y_j, t_j] Y_j} holds F for
            # every flow (pref and a normalised mu are scalars)
            stage_time = t + tableau.c[slot] * h
            density = reuse_density ? get(warm, (slot, :rho), nothing) : nothing
            field_operator, field_info = _flow_op_at(fl, stage_cores, stage_time, density)
            t_keep_rho += field_info.t_rho
            field_cores =
                first(op_zipup(field_operator, stage_cores; tol = stage_keep_tol, maxdim = maxdim))
            # a field no stage reads (a zero tableau column) does not enter the step at all
            if tt_norm(field_cores) > 0 && field_weights[slot] > 0
                push!(stage_keep, qtt_mps(field_cores))
                push!(stage_keep_tols, admit_tol(field_weights[slot], field_cores))
            end
        end
        t_keep += (time_ns() - keep_start) / 1e9
        basis_start = time_ns()
        corrected = build_frames(stage_keep; tols = stage_keep_tols)
        t_basis += (time_ns() - basis_start) / 1e9
        # the step input and the previous stages carried into the new frames (also t_keep)
        keep_start = time_ns()
        corrected_input = frame_project(corrected, nothing, input_cores)
        start_values = [frame_project(corrected, nothing, frame_state(frames, Yj)) for Yj in Y]
        t_keep += (time_ns() - keep_start) / 1e9
        root_start = time_ns()
        Y, stage_stats = rk_stage_solve(
            fl,
            corrected,
            corrected_input,
            t,
            h,
            root;
            tol = tol_step.stage,
            Y0 = start_values,
            stage_keywords...,
        )
        t_root += (time_ns() - root_start) / 1e9
        push!(stats, stage_stats)
        frames = corrected
    end
    trunc_start = time_ns()
    centre = length(frames.W) + 1
    out = qtt_mps(frame_state(frames, Y[end]); center = centre)
    _, _, cumulative, capped =
        truncate_recursive!(out, centre; maxdim = maxdim, weight_tol = tol_step.trunc)
    _replace_state!(psi, out)
    timings = (
        basis = t_basis,
        root = t_root,
        keep = t_keep,
        keep_rho = t_keep_rho,
        trunc = (time_ns() - trunc_start) / 1e9,
    )
    return step_result(frames, stats, cumulative, capped, timings, root)
end
