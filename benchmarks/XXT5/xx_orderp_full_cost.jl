# Full-cost order-p BUG/RSVD audit on the optimized experiment branch.
# This is a measurement harness, NOT an implementation of joint-frontier selection.
# All reported elapsed times include SVD/QR, environments, validation and closure.
# Extra full-frontier audit is measured separately and included in audited total.
using BondUpdateGalerkin.BUG, BondUpdateGalerkin.QMB, BondUpdateGalerkin.QTTS
using LinearAlgebra, Random, Printf
include(joinpath(@__DIR__, "..", "..", "test", "common", "free_fermion.jl"))
include(joinpath(@__DIR__, "..", "..", "test", "common", "dense_reference.jl"))

mode = get(ENV, "BUG_ORDERP_MODE", "smoke")
mode in ("smoke", "physical", "stress") || error("unknown mode: $mode")
outdir = length(ARGS) >= 1 ? ARGS[1] : joinpath("results", "orderp-" * mode)
mkpath(outdir)

const HEADER = [
    "mode", "policy", "p", "L", "D_requested", "D_initial", "D_final", "D_stage",
    "D_root_left", "D_root_right", "D_work_max", "dt", "steps", "seed",
    "status", "unmet", "audit_met", "audit_leak_integral",
    "state_error", "sz_profile_error", "norm_error",
    "step_wall_s", "audit_wall_s", "full_audited_wall_s",
    "t_half_cbe_s", "t_half_krylov_s", "t_half_frame_s", "t_half_split_s",
    "t_half_env_s", "t_root_s", "t_trunc_s", "t_closure_s",
    "t_unattributed_s", "half_qr_count", "half_qr_s", "half_stream_cols",
    "half_h1", "closure_h1", "closure_h2", "root_h0", "frontier_root_h0",
    "sketch_probes_all", "holdout_probes_all", "dd_selected_all",
    "krylov_selected_all", "closure_sweeps", "closure_passes",
    "closure_sweep_wall_sum_s", "closure_added", "closure_skipped",
    "audit_h2", "audit_root_h0", "allocated_bytes_step", "peak_rss_high_water",
    "closure_checks", "frontier_tol", "krylov_rank_tol"
]
fmt(x::AbstractFloat) = isfinite(x) ? @sprintf("%.10g", x) : string(x)
fmt(x) = replace(string(x), ',' => ';', '\n' => ' ')
write_row(io, xs...) = (println(io, join(fmt.(xs), ",")); flush(io))
function policies()
    fixed = DDQRExpansion(mode=:fixed, fixed_width=8, fixed_rank=4,
                         krylov_mode=:lanczos, krylov_fixed_depth=8)
    fast = DDQRExpansion(mode=:rank_fast, rank_initial_width=3,
                        holdout_width=2, acceptance_factor=0.6,
                        krylov_mode=:lanczos)
    # Each case is a real available implementation; NO 'joint' algorithm is claimed.
    return [
        ("fixed_checked", fixed, true, 0.0, 0.0),
        ("fast_checked", fast, true, 0.0, 0.0),
        ("fast_ranked", fast, true, 0.0, 1e-6),
        ("fast_compressed", fast, true, 1e-6, 0.0),
        ("fast_unchecked", fast, false, 0.0, 0.0)
    ]
end
function random_mps(L, D, seed)
    rng = MersenneTwister(seed)
    dims = vcat([1], [min(D, 2^min(k, L-k)) for k in 1:L-1], [1])
    cores = [randn(rng, ComplexF64, dims[k], 2, dims[k+1]) ./
             sqrt(2 * dims[k]) for k in 1:L]
    psi = qtt_mps(cores; center = cld(L, 2))
    psi[psi.center] = psi[psi.center] * (1 / norm(psi))
    return psi
end
function run_case(io; mode, name, policy, checks, frontier_tol, rank_tol,
                  p, L, requested, dt, steps, seed, initial, mpo, basis_tol,
                  trunc_tol, exact_h = nothing, exact_state = nothing)
    tau = ComplexF64(-im * dt)
    state = copy(initial)
    maxdim = mode == "stress" ? requested + max(10, ceil(Int, 0.2 * requested)) : requested
    last_info = nothing
    summed_time = 0.0
    summed_audit = 0.0
    total_h1 = 0
    first_rank = maximum(bond_dims(initial); init=0)
    max_stage = 0
    last_audit = nothing
    probe_all = 0
    holdout_all = 0
    sel_all = 0
    ksel_all = 0
    closure_sweeps = 0
    closure_passes = 0
    closure_h1 = 0
    closure_h2 = 0
    half_h1 = 0
    half_qr = 0
    half_qr_s = 0.0
    half_stream_cols = 0
    root_h0 = 0
    frontier_root_h0 = 0
    half_cbe = 0.0
    half_kry = 0.0
    half_frame = 0.0
    half_split = 0.0
    half_env = 0.0
    root_time = 0.0
    trunc_time = 0.0
    closure_time = 0.0
    closure_logged = 0.0
    closure_added = 0
    closure_skipped = 0
    all_bytes = 0
    maxrss = 0
    audit_h2 = 0
    audit_root_h0 = 0
    bad = String[]
    audit_pass = true
    audit_integral = 0.0
    for s in 1:steps
        old = copy(state)
        fr = Ref{Any}(nothing)
        step_rng = MersenneTwister(seed + 1001 * s)
        info = cbe_bug_step!(
            state, mpo, tau;
            order=p, expansion=policy, rng=step_rng, parallel=false,
            maxdim=maxdim, tol_basis=basis_tol, tol_trunc=trunc_tol,
            tol_root=1e-11, closure_checks=checks,
            frontier_tol=frontier_tol, krylov_rank_tol=rank_tol,
            work_ceiling_rule=:bond, bond_ceiling_factor=1.2,
            frames_report=fr
        )
        last_info = info
        l = info.ledger
        cnt = info.record.counters
        summed_time += info.t_step
        max_stage = max(max_stage, info.record.D_stage)
        probe_all += l.n_probes
        holdout_all += l.n_holdout
        sel_all += l.n_selected
        ksel_all += l.n_krylov
        closure_sweeps += l.n_closure_sweeps
        closure_passes += l.n_passes
        closure_h1 += l.n_h1_closure
        closure_h2 += l.n_h2_leak
        half_h1 += cnt.n_one_site
        half_qr += cnt.n_qr
        half_qr_s += cnt.t_qr
        half_stream_cols += cnt.n_stream_cols
        root_h0 += cnt.n_root
        frontier_root_h0 += l.n_root_frontier
        half_cbe += info.t_cbe
        half_kry += info.t_kry
        half_frame += info.t_frame
        half_split += info.t_split
        half_env += info.t_env
        root_time += info.t_root
        trunc_time += info.t_trunc
        closure_time += l.t_closure
        closure_logged += sum(get(entry, :t, 0.0) for entry in l.closure_log)
        closure_added += sum(get(entry, :added, 0) for entry in l.closure_log)
        closure_skipped += sum(get(entry, :skipped, 0) for entry in l.closure_log)
        all_bytes += info.record.bytes_step
        maxrss = max(maxrss, info.record.peak_rss)
        budget_status(l) === :met || push!(bad, "step$(s):" * join(string.(l.unmet), "|"))
        # Independently remeasure ALL time-weighted frontiers on the FINAL frames,
        # even for the 'unchecked' optimization. This is extra cost, not hidden.
        start_audit = time_ns()
        final_frame = fr[]
        alloc = BUG.basis_allocation(
            BUG.AdaptiveBudget(basis_tol, trunc_tol, 1e-11), L-1;
            tau=tau, nrm=norm(old), nfront=p, rule=:linear
        )
        s0 = BUG._project_root(old, final_frame)
        audit = BUG._measure_frontiers(
            final_frame, old, mpo, s0, tau, alloc, (expansion=policy,), p;
            nrm=Float64(norm(old)), parallel=false, fold=true,
            rng=MersenneTwister(seed + 81000 + s)
        )
        summed_audit += (time_ns() - start_audit) / 1e9
        audit_h2 += audit.ledger.n_h2_leak
        audit_root_h0 += audit.n_root
        audit_pass &= audit.met && audit.met0
        audit_integral = sum(audit.leak[j] / j for j in 1:p)
        last_audit = audit
        println("STEP mode=$mode name=$name p=$p L=$L req=$requested " *
                "s=$s status=$(budget_status(l)) stage=$(info.record.D_stage) " *
                "final=$(info.record.D_final) step=$(round(info.t_step,digits=3)) " *
                "audit=$(round(summed_audit,digits=3)) closure_sweeps=$(l.n_closure_sweeps) " *
                "audit_met=$(audit.met && audit.met0)")
        flush(stdout)
    end
    errstate = NaN
    errprofile = NaN
    if mode == "smoke"
        truth = exp(tau * exact_h) * exact_state
        errstate = norm(dense_state(state) - truth) / norm(truth)
    elseif mode == "physical"
        exact = xx_free_fermion_sz(L, steps*dt; J=1.0, occupied=1:2:L)
        profile = real.(magnetisation(copy(state))) ./ real(overlap(state,state))
        errprofile = maximum(abs.(profile .- exact))
    end
    norm_error = abs(real(overlap(state,state)) - 1)
    l = last_info.ledger
    disjoint = half_cbe + half_kry + half_frame + half_split + half_env +
               root_time + trunc_time + closure_time
    # The individual component timers are subsets of wall time; the residual is
    # diagnostic, not a missing-work correction or a substitute for step_wall.
    una = summed_time - disjoint
    write_row(io,
        mode, name, p, L, requested, first_rank, maximum(bond_dims(state);init=0),
        max_stage, last_info.expanded[last_info.root], last_info.root_right,
        last_info.record.D_work, dt, steps, seed,
        budget_status(l), isempty(bad) ? "none" : join(bad,"|"),
        audit_pass, audit_integral, errstate, errprofile, norm_error,
        summed_time, summed_audit, summed_time + summed_audit,
        half_cbe, half_kry, half_frame, half_split, half_env,
        root_time, trunc_time, closure_time, una, half_qr, half_qr_s,
        half_stream_cols, half_h1, closure_h1, closure_h2, root_h0,
        frontier_root_h0, probe_all, holdout_all, sel_all, ksel_all,
        closure_sweeps, closure_passes, closure_logged, closure_added,
        closure_skipped, audit_h2, audit_root_h0, all_bytes, maxrss,
        checks, frontier_tol, rank_tol
    )
    return (status=budget_status(l), audit_met=audit_pass,
            state_error=errstate, profile_error=errprofile)
end

open(joinpath(outdir, "costs.csv"), "w") do io
    write_row(io, HEADER...)
    if mode == "stress"
        # Actual starting D=100 and D=500, NOT merely maxdim values.
        # Random :none states are stress tests, not a physical XX benchmark.
        set_symmetry!(:none)
        D = parse(Int, get(ENV,"BUG_ORDERP_D","100"))
        L = D >= 500 ? 18 : 14
        init = random_mps(L,D,85)
        actual = maximum(bond_dims(init))
        actual == D || error("stress rank not achieved: D=$D got=$actual")
        H = xxz_mpo(L; J=1.0, delta=0.0)
        for (name,policy,checks,ft,kt) in policies()[1:2]
            for p in (2,4)
                run_case(io;mode,name,policy,checks,frontier_tol=ft,rank_tol=kt,
                         p,L,requested=D,dt=0.005,steps=1,seed=31,
                         initial=init,mpo=H,basis_tol=1e-5,trunc_tol=1e-6)
            end
        end
    else
        set_symmetry!(:U1)
        L = mode == "smoke" ? 8 : 14
        init = neel_state(L)
        H = xxz_mpo(L; J=1.0, delta=0.0)
        H_exact = mode == "smoke" ? dense_heisenberg(L;J=1.0,delta=0.0) : nothing
        v0 = mode == "smoke" ? dense_state(init) : nothing
        candidate_policies = mode == "smoke" ? policies() : policies()
        requested_dims = mode == "smoke" ? (64,) : (100,500)
        for D in requested_dims, p in (2,3,4)
            for (name,policy,checks,ft,kt) in candidate_policies
                run_case(io;mode,name,policy,checks,frontier_tol=ft,rank_tol=kt,
                         p,L,requested=D,dt=mode=="smoke" ? 0.15 : 0.08,
                         steps=mode=="smoke" ? 1 : 2,seed=13,
                         initial=init,mpo=H,basis_tol=1e-6,trunc_tol=1e-10,
                         exact_h=H_exact,exact_state=v0)
            end
        end
    end
end
println("ORDERP_FULL_COST_COMPLETE output=$outdir")
