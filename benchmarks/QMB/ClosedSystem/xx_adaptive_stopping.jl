# Compare fixed DD oversampling/retention with adaptive DD and independent
# Lanczos/Arnoldi root exponential solves. Julia 1.11+, from repository root:
# julia --project benchmarks/QMB/ClosedSystem/xx_adaptive_stopping.jl OUTPUT_DIR
#
# The benchmark evaluates fixed final-state accuracy at matching final time.
# Short- and long-range models are BOTH genuine U(1) XX exchange operators;
# long-range spin exchange is interacting after Jordan-Wigner transformation.
using BondUpdateGalerkin.BUG, BondUpdateGalerkin.QMB, BondUpdateGalerkin.Baselines
using LinearAlgebra, Random, Statistics, Telum
include(joinpath(@__DIR__, "..", "..", "..", "test", "common", "dense_reference.jl"))

outdir = isempty(ARGS) ? "results/xx_adaptive_stopping" : ARGS[1]
mkpath(outdir)
set_symmetry!(:U1)

function couplings(L, longrange)
    W = zeros(Float64, L, L)
    for i in 1:L-1, j in i+1:L
        r = j - i
        W[i,j] = longrange ? r^(-3.0) : (r == 1 ? 1.0 : 0.0)
    end
    return W
end

function model(L, W)
    q = spin_space(1)
    v = [PairVertex(q.Sp, q.Sp', 1.0), PairVertex(q.Sm, q.Sm', 1.0)]
    return pair_mpo(L, W, v; Iloc=q.I)
end

# Independent ED Hamiltonian: up=1, down=0 in computational basis.
# A pair exchange has matrix element W[i,j]/2.
function exact_hamiltonian(L, W)
    N = 1 << L
    H = zeros(Float64, N, N)
    for k in 0:N-1, i in 1:L-1, j in i+1:L
        W[i,j] == 0 && continue
        bi = (k >> (L-i)) & 1
        bj = (k >> (L-j)) & 1
        bi == bj && continue
        m = xor(k, (1 << (L-i)) | (1 << (L-j)))
        H[m+1,k+1] += W[i,j]/2
    end
    return H
end

function reference_profile(H, psi, L, t)
    x = dense_state(psi)
    v = exp(-im*t*Hermitian(H))*x
    nv = real(dot(v,v))
    return [sum(((((k-1) >> (L-j)) & 1) == 0 ? 0.5 : -0.5)*abs2(v[k])
                 for k in eachindex(v))/nv for j in 1:L]
end
profile(psi) = real.(magnetisation(copy(psi))) ./ real(overlap(psi,psi))
csvrow(v) = join(string.(v), ",")
function write_rows(name, headers, rows)
    open(joinpath(outdir,name), "w") do io
        println(io,csvrow(headers))
        for r in rows
            println(io,csvrow(r))
        end
    end
end

const dt = parse(Float64, get(ENV, "BUG_XX_DT", "0.05"))
const nsteps = parse(Int, get(ENV, "BUG_XX_STEPS", "12"))
const nseeds = parse(Int, get(ENV, "BUG_XX_SEEDS", "3"))
const basis = parse(Float64, get(ENV, "BUG_XX_BASIS", "1e-5"))
const trunc = parse(Float64, get(ENV, "BUG_XX_TRUNC", "1e-8"))
const root_tol = parse(Float64, get(ENV, "BUG_XX_ROOT", "1e-10"))
const cap = parse(Int, get(ENV, "BUG_XX_MAXDIM", "64"))
const selected_policies = filter(!isempty, split(get(ENV, "BUG_XX_POLICIES", ""), ","))
const configs = (
    ("fixed2", DDQRExpansion(mode=:fixed,fixed_width=4,fixed_rank=2), NamedTuple()),
    ("fixed4", DDQRExpansion(mode=:fixed,fixed_width=8,fixed_rank=4), NamedTuple()),
    ("adaptive4", DDQRExpansion(holdout_width=4), NamedTuple()),
    ("adaptive2conservative", DDQRExpansion(holdout_width=2,acceptance_factor=0.6), NamedTuple()),
    # Rank control is an existing BUG feature; compare its extra small-core SVD cost.
    ("adaptive_ranked_krylov", DDQRExpansion(holdout_width=2,acceptance_factor=0.6),
        (krylov_rank_tol=basis,)),
    # This is a resource-cap ablation, not an accuracy stopping criterion.
    ("adaptive_bondcap", DDQRExpansion(holdout_width=2,acceptance_factor=0.6),
        (work_ceiling_rule=:bond,bond_ceiling_factor=1.2,)),
    ("adaptive_lanczos", DDQRExpansion(holdout_width=2,acceptance_factor=0.6,
        krylov_mode=:lanczos), NamedTuple()),
    ("fixed4_lanczos8", DDQRExpansion(mode=:fixed,fixed_width=8,fixed_rank=4,
        krylov_mode=:lanczos,krylov_fixed_depth=8), NamedTuple()),
    ("fixed4_lanczos12", DDQRExpansion(mode=:fixed,fixed_width=8,fixed_rank=4,
        krylov_mode=:lanczos,krylov_fixed_depth=12), NamedTuple()),
    # Verified global Krylov closure (the non-checking default can miss first-step directions).
    ("fixed4_checked", DDQRExpansion(mode=:fixed,fixed_width=8,fixed_rank=4),
        (closure_checks=true,)),
    ("adaptive_checked", DDQRExpansion(holdout_width=2,acceptance_factor=0.6),
        (closure_checks=true,)),
    ("lanczos_checked", DDQRExpansion(holdout_width=2,acceptance_factor=0.6,
        krylov_mode=:lanczos), (closure_checks=true,)),
    ("outside_checked", DDQRExpansion(holdout_width=2,acceptance_factor=0.6),
        (closure_checks=true,outside_residual=true,root_report=Ref{Any}())),
    ("order3_checked", DDQRExpansion(holdout_width=2,acceptance_factor=0.6),
        (closure_checks=true,order=3,)),
)
rows = Any[]
frames_rows = Any[]
profile_rows = Any[]
bond_stage_rows = Any[]
root_rows = Any[]
function dense_profile(v,L)
    nv = real(dot(v,v))
    return [sum(((((k-1) >> (L-j)) & 1) == 0 ? 0.5 : -0.5)*abs2(v[k])
                 for k in eachindex(v))/nv for j in 1:L]
end
function vector_errors(psi, ref)
    v = dense_state(psi)
    np, nr = norm(v), norm(ref)
    fidelity = abs2(dot(v, ref)) / (np*np * nr*nr)
    phase_error = sqrt(max(0.0, 1.0 - min(1.0, fidelity)))
    return (phase_error, abs(np-1.0))
end
for L in (8,), interaction in (:short,:long)
    longrange = interaction == :long
    W = couplings(L,longrange)
    mpo = model(L,W)
    H = exact_hamiltonian(L,W)
    initial = spin_domain_wall_state(L,1)
    # ED eigensystem reused for every step and every method. No approximation here.
    eig = eigen(Hermitian(H))
    u0 = dense_state(initial)
    c0 = eig.vectors' * u0
    exact_vec = [eig.vectors * (exp.(-im*dt*k .* eig.values) .* c0) for k in 1:nsteps]
    exact_at = [dense_profile(v,L) for v in exact_vec]
    for (method,policy,stepkw) in configs, seed in 0:(nseeds-1)
        (isempty(selected_policies) || method in selected_policies) || continue
        psi = spin_domain_wall_state(L,1)
        rng = MersenneTwister(seed)
        for step in 1:nsteps
            methodkw = merge((order=2,), stepkw)
            info = cbe_bug_step!(psi,mpo,-im*dt;
                maxdim=cap,tol_basis=basis,tol_trunc=trunc,
                tol_root=root_tol,expansion=policy,rng=rng,methodkw...)
            err = maximum(abs.(profile(psi).-exact_at[step]))
            state_err, norm_err = vector_errors(psi,exact_vec[step])
            d_final = bond_dims(psi)
            for (bond, d) in enumerate(d_final)
                frame = bond <= length(info.expanded) ? info.expanded[bond] : 0
                bond == info.root && (frame = max(frame, info.root_right))
                push!(frames_rows,(interaction,L,method,seed,step,step*dt,bond,frame,d))
            end
            # Actual K/L frame widths, logged before root solve by the two half-sweeps.
            for bl in info.ledger.bond_log
                push!(bond_stage_rows,(interaction,L,method,seed,step,step*dt,
                    bl.bond,bl.side,bl.width,bl.n_selected,bl.n_krylov,
                    bl.n_probes,bl.n_holdout,bl.n_h1,bl.met))
            end
            push!(root_rows,(interaction,L,method,seed,step,step*dt,
                info.expanded[info.root],info.root_right,
                info.expanded[info.root]*info.root_right,
                info.root_rank,maximum(d_final)))
            for (site, sz) in enumerate(profile(psi))
                push!(profile_rows,(interaction,L,method,seed,step,step*dt,site,sz,
                                    exact_at[step][site]))
            end
            Lg = info.ledger
            push!(rows,(interaction,L,method,seed,step,step*dt,
                err,state_err,norm_err,maximum(d_final),info.record.D_stage,
                Lg.n_probes,Lg.n_holdout,Lg.n_selected,Lg.n_krylov,
                Lg.n_h1,info.record.root_iters,info.t_cbe,info.t_kry,
                info.t_root,info.t_step,
                policy.mode === :fixed ? "uncertified" : string(budget_status(Lg))))
        end
    end
    (isempty(selected_policies) || "tdvp2" in selected_policies) || continue
    psi = spin_domain_wall_state(L,1)
    for step in 1:nsteps
        t0 = time_ns()
        info = tdvp2_step!(psi,mpo,-im*dt;maxdim=cap,weight_tol=trunc)
        elapsed = (time_ns()-t0)/1e9
        err = maximum(abs.(profile(psi).-exact_at[step]))
        state_err, norm_err = vector_errors(psi,exact_vec[step])
        for (bond, d) in enumerate(bond_dims(psi))
            push!(frames_rows,(interaction,L,"tdvp2",0,step,step*dt,bond,0,d))
        end
        for (site, sz) in enumerate(profile(psi))
            push!(profile_rows,(interaction,L,"tdvp2",0,step,step*dt,site,sz,
                                exact_at[step][site]))
        end
        push!(rows,(interaction,L,"tdvp2",0,step,step*dt,
            err,state_err,norm_err,maximum(bond_dims(psi)),0,
            0,0,0,0,0,0,0,0,0,elapsed,"reference"))
    end
end
write_rows("trajectories.csv",
    ("interaction","L","method","seed","step","t","max_sz_error",
     "sqrt_one_minus_fidelity","norm_error","D_final","D_stage","n_probes",
     "n_holdout","n_dd","n_krylov","n_h1","root_iters","t_cbe",
     "t_kry","t_root","t_step","budget_status"),rows)
write_rows("frame_ranks.csv",
    ("interaction","L","method","seed","step","t","bond","D_frame","D_final"),
    frames_rows)
write_rows("profiles.csv",
    ("interaction","L","method","seed","step","t","site","sz","sz_exact"),
    profile_rows)
write_rows("bond_stages.csv",
    ("interaction","L","method","seed","step","t","bond","side","D_frame",
     "n_dd","n_krylov","n_probes","n_holdout","n_h1","budget_met"),
    bond_stage_rows)
write_rows("root_ranks.csv",
    ("interaction","L","method","seed","step","t",
     "D_root_left","D_root_right","root_core_elements","D_root_svd_before_global_trunc",
     "D_final_max"),root_rows)
println("Wrote ",joinpath(outdir,"trajectories.csv"))

# Independently compare existing Lanczos and Arnoldi local exponential solvers,
# against exact exp on the SAME Hermitian XX matrix.
kr = Any[]
for L in (8,), interaction in (:short,:long)
    H = exact_hamiltonian(L,couplings(L,interaction==:long))
    u = dense_state(spin_domain_wall_state(L,1))
    E = eigen(Hermitian(H))
    uc = E.vectors' * u
    for t in (0.5,2.0,4.0), method in (:lanczos,:arnoldi)
        ref = E.vectors * (exp.(-im*t .* E.values) .* uc)
        solve = method == :lanczos ? lanczos_expv : arnoldi_expv
        for (policy, tol, depth) in
            vcat([("fixed",0.0,depth) for depth in (4,8,12,20,30,40)],
                 [("adaptive",tol,40) for tol in (1e-4,1e-6,1e-8,1e-10)])
            report = Ref{Any}()
            v = solve(x->TLArray(H*Array(x)),ComplexF64(-im*t),u;
                conv_tol=tol,maxiter=depth,report=report)
            push!(kr,(interaction,L,t,method,policy,tol,depth,report[].iterations,
                      norm(Array(v)-ref)/norm(ref),report[].err))
        end
    end
end
write_rows("krylov.csv",
    ("interaction","L","t","method","policy","conv_tol","maxiter",
     "applications","relative_error","reported_estimate"),kr)
println("Wrote ",joinpath(outdir,"krylov.csv"))
