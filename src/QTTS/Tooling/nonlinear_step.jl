# One rank-adaptive BUG step of a nonlinear flow on QTT states.
#
# Moved from BUG-Julia@b155550:src/RSVDCBEBondUpdate/qtt/qtt_flows.jl (`qtt_bug_step!`). The basis
# comes from the BUG engine (`cbe_bug_step!` with `frames_report`), the root from the generic stage
# solver (`rk_stage_solve`); this file holds the QTT glue (dense frames, keep states, compression).

"Safety factor on the local-Krylov share of `tol_basis` in `qtt_bug_step!`."
const _QTT_KRYLOV_MARGIN = 0.5

"""
    qtt_bug_step!(psi, fl, t, h; order = 2, root = qtt_root_for_order(order), npc = order - 1,
                  tol_stage = 1e-10, tol_basis = 1e-6, tol_trunc = 1e-8, maxdim = 128,
                  keep_tol = 1e-10, keep = [], kwargs...) -> NamedTuple

One rank-adaptive BUG step of the full flow `fl` from `t` to `t + h`, in place:

1. the basis: `cbe_bug_step!` at `order` on the predictor generator `H[psi_n, t_n]` (its DD-CBE,
   local Krylov and order-p frontier closure), the frames taken from `frames_report`;
2. the root: `S_n = <frames|psi_n>`, the step input projected, and the nonlinear `root` stage
   equations solved with the density rebuilt at every stage iterate ([`rk_stage_solve`](@ref));
3. `npc` basis corrections: the frames are rebuilt keeping the stage states and their vector fields
   exactly (`keep_states`), and the root re-solved from the same `S_n`;
4. one compression of `W Y_s Z` (`truncate_recursive!` at `tol_trunc`, `maxdim`).

The temporal order is a claim to be measured on the whole update, never assumed from `root`.
`kwargs` go to `cbe_bug_step!` (`tol_krylov` defaults to `_QTT_KRYLOV_MARGIN * tol_basis`). Returns
the stage statistics, `converged`, the frame widths, the discarded weight, the frame audits
(`miss_input`, `miss_keep`, `frame_defect`), timings and `chi`.
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
    kwargs...,
)
    step_start = time_ns()
    input_cores = qtt_cores(psi)
    predictor, predictor_info = flow_op(fl, input_cores, t)
    # The frames' generator is shifted by the state's mean <H>: the Krylov spaces are unchanged, but
    # a stiff step's root no longer underflows to an exactly zero matrix.
    mean_energy = real(_op_expect(predictor, input_cores)) / max(tt_norm(input_cores)^2, eps())
    predictor_mpo = qtt_to_mpo(predictor)
    tau = ComplexF64(fl.pref * h)
    tol_krylov = Float64(get(kwargs, :tol_krylov, _QTT_KRYLOV_MARGIN * tol_basis))
    engine_kwargs = Base.structdiff(values(kwargs), NamedTuple{(:tol_krylov,)})
    nfallback = 0
    krylov_states = nothing
    # every frame set is audited: the step input and each keep state must lie in the span
    # (Galerkin invariants, e.g. Heston mass, hold only then) and the cores must be isometric
    keep_cores = [qtt_cores(state) for state in keep]
    miss_input, miss_keep, defect = 0.0, 0.0, 0.0
    function audit!(frames)
        miss_input = max(miss_input, frame_miss(frames, input_cores))
        miss_keep = max(miss_keep, maximum(X -> frame_miss(frames, X), keep_cores; init = 0.0))
        defect = max(defect, frame_defect(frames))
        return frames
    end
    function build_frames(keep_states)
        report = Ref{Any}(nothing)
        run(basis_order, states) = cbe_bug_step!(
            copy(psi),
            predictor_mpo,
            tau;
            order = basis_order,
            tol_basis = Float64(tol_basis),
            tol_trunc = Float64(tol_trunc),
            maxdim = maxdim,
            hermitian = false,
            frames_report = report,
            keep_states = states,
            truncate = false,
            eshift = mean_energy,
            tol_krylov = tol_krylov,
            engine_kwargs...,
        )
        try
            run(order, keep_states)
        catch err
            # The order-p frontier closure can carry an old frame wider than a near-saturated
            # site's new local space. The equivalent order-p basis -- order-1 frames that keep the
            # global Krylov vectors (tau H)^j psi, j < p, exactly -- is used instead and counted.
            order > 1 && occursin("cannot exceed its local space", sprint(showerror, err)) ||
                rethrow()
            nfallback += 1
            if krylov_states === nothing
                krylov_states = global_krylov_states(
                    predictor_mpo,
                    psi,
                    order - 1;
                    tol_trunc = Float64(keep_tol),
                    maxdim = 4 * maxdim,
                )
            end
            run(1, vcat(Any[keep_states...], Any[krylov_states...]))
        end
        return audit!(QTTFrames(report[]))
    end
    basis_start = time_ns()
    frames = build_frames(Any[keep...])
    t_basis = (time_ns() - basis_start) / 1e9
    stats = Any[]
    projected_input = frame_project(frames, nothing, input_cores)
    root_start = time_ns()
    Y, stage_stats = rk_stage_solve(fl, frames, projected_input, t, h, root; tol = tol_stage)
    t_root = (time_ns() - root_start) / 1e9
    push!(stats, stage_stats)
    tableau = RK_TABLEAUX[root]
    for _ = 1:npc
        # keep every stage state and its vector field exactly in the corrected frames
        stage_keep = Any[keep...]
        for (j, Yj) in enumerate(Y)
            stage_cores = frame_state(frames, Yj)
            push!(
                stage_keep,
                qtt_mps(first(tt_compress(stage_cores; tol = keep_tol, maxdim = 4 * maxdim))),
            )
            G, _, _ = reduced_field(fl, frames, Yj, t + tableau.c[j] * h)
            field_cores =
                first(tt_compress(frame_state(frames, G); tol = keep_tol, maxdim = 4 * maxdim))
            tt_norm(field_cores) > 0 && push!(stage_keep, qtt_mps(field_cores))
        end
        basis_start = time_ns()
        corrected = build_frames(stage_keep)
        t_basis += (time_ns() - basis_start) / 1e9
        corrected_input = frame_project(corrected, nothing, input_cores)
        start_values = [frame_project(corrected, nothing, frame_state(frames, Yj)) for Yj in Y]
        root_start = time_ns()
        Y, stage_stats = rk_stage_solve(
            fl,
            corrected,
            corrected_input,
            t,
            h,
            root;
            tol = tol_stage,
            Y0 = start_values,
        )
        t_root += (time_ns() - root_start) / 1e9
        push!(stats, stage_stats)
        frames = corrected
    end
    centre = length(frames.W) + 1
    out = qtt_mps(frame_state(frames, Y[end]); center = centre)
    _, _, cumulative, capped =
        truncate_recursive!(out, centre; maxdim = maxdim, weight_tol = Float64(tol_trunc))
    _replace_state!(psi, out)
    return (
        stats = stats,
        converged = all(s.converged for s in stats),
        frame_width = (size(frames.W[end], 3), size(frames.Z[1], 1)),
        discarded = cumulative,
        rank_capped = capped,
        t_basis = t_basis,
        t_root = t_root,
        t_rho0 = predictor_info.t_rho,
        basis_fallback = nfallback,
        miss_input = miss_input,
        miss_keep = miss_keep,
        frame_defect = defect,
        t_step = (time_ns() - step_start) / 1e9,
        chi = maximum(bond_dims(psi); init = 1),
        root = root,
        order = order,
    )
end
