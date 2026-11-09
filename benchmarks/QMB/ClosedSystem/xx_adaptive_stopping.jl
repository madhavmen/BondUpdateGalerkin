# Compare fixed DD oversampling/retention with adaptive DD and independent
# Lanczos/Arnoldi root exponential solves. Julia 1.11+, from repository root:
# julia --project benchmarks/QMB/ClosedSystem/xx_adaptive_stopping.jl OUTPUT_DIR
#
# The benchmark evaluates fixed final-state accuracy at matching final time.
# Short- and long-range models are BOTH genuine U(1) XX exchange operators;
# long-range spin exchange is interacting after Jordan-Wigner transformation.
using BondUpdateGalerkin.BUG, BondUpdateGalerkin.QMB, BondUpdateGalerkin.Baselines
using LinearAlgebra, Random, Statistics
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
    return [sum(((((k-1) >> (L-j)) & 1) == 1 ? 0.5 : -0.5)*abs2(v[k])
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

const dt = 0.05
const nsteps = 12
const basis = 1e-5
const trunc = 1e-8
const root_tol = 1e-10
const cap = 64
const configs = (
    ("fixed2", DDQRExpansion(mode=:fixed,fixed_width=4,fixed_rank=2)),
    ("fixed4", DDQRExpansion(mode=:fixed,fixed_width=8,fixed_rank=4)),
    ("adaptive4", DDQRExpansion(holdout_width=4)),
    ("adaptive2conservative", DDQRExpansion(holdout_width=2,acceptance_factor=0.6)),
)
rows = Any[]
for L in (8,), interaction in (:short,:long)
    longrange = interaction == :long
    W = couplings(L,longrange)
    mpo = model(L,W)
    H = exact_hamiltonian(L,W)
    initial = spin_domain_wall_state(L,1)
    # Small ED reference is computed at each sampled time, not only at t_final.
    exact_at = [reference_profile(H, initial, L, dt*k) for k in 1:nsteps]
    for (method,policy) in configs, seed in 0:2
        psi = spin_domain_wall_state(L,1)
        rng = MersenneTwister(seed)
        for step in 1:nsteps
            info = cbe_bug_step!(psi,mpo,-im*dt;order=2,
                maxdim=cap,tol_basis=basis,tol_trunc=trunc,
                tol_root=root_tol,expansion=policy,rng=rng)
            err = maximum(abs.(profile(psi).-exact_at[step]))
            Lg = info.ledger
            push!(rows,(interaction,L,method,seed,step,step*dt,
                err,maximum(bond_dims(psi)),info.record.D_stage,
                Lg.n_probes,Lg.n_holdout,Lg.n_selected,Lg.n_krylov,
                Lg.n_h1,info.record.root_iters,info.t_cbe,info.t_kry,
                info.t_root,info.t_step,budget_status(Lg)))
        end
    end
    psi = spin_domain_wall_state(L,1)
    for step in 1:nsteps
        info = tdvp2_step!(psi,mpo,-im*dt;maxdim=cap,weight_tol=trunc)
        err = maximum(abs.(profile(psi).-exact_at[step]))
        push!(rows,(interaction,L,"tdvp2",0,step,step*dt,
            err,maximum(bond_dims(psi)),0,0,0,0,0,0,0,0,0,0,0,"reference"))
    end
end
write_rows("trajectories.csv",
    ("interaction","L","method","seed","step","t","max_sz_error",
     "D_final","D_stage","n_probes","n_holdout","n_dd","n_krylov",
     "n_h1","root_iters","t_cbe","t_kry","t_root","t_step","budget_status"),rows)
println("Wrote ",joinpath(outdir,"trajectories.csv"))

# Independently compare existing Lanczos and Arnoldi local exponential solvers,
# against exact exp on the SAME Hermitian XX matrix.
kr = Any[]
for L in (8,), interaction in (:short,:long)
    H = exact_hamiltonian(L,couplings(L,interaction==:long))
    u = dense_state(spin_domain_wall_state(L,1))
    ref = exp(-im*dt*Hermitian(H))*u
    for method in (:lanczos,:arnoldi), tol in (1e-4,1e-6,1e-8,1e-10)
        report = Ref{Any}()
        solve = method == :lanczos ? lanczos_expv : arnoldi_expv
        v = solve(x->H*x,ComplexF64(-im*dt),u;
            conv_tol=tol,maxiter=40,report=report)
        push!(kr,(interaction,L,method,tol,report[].iterations,
                  norm(v-ref)/norm(ref),report[].err))
    end
end
write_rows("krylov.csv",
    ("interaction","L","method","conv_tol","applications","relative_error",
     "reported_estimate"),kr)
println("Wrote ",joinpath(outdir,"krylov.csv"))
