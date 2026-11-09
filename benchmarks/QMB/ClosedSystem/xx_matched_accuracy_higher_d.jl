# Spin-S U(1) XX benchmark: fixed vs adaptive CBE and local Arnoldi/Lanczos,
# equal-achieved-error comparison with independent exact diagonalization.
# d=twoS+1; default 2:6 = spin 1, 3:6 = spin 3/2, both S^z_total = 0.
#
# Julia 1.11 from repository root:
#   julia --project benchmarks/QMB/ClosedSystem/xx_matched_accuracy_higher_d.jl OUTPUT_DIR
# Environment controls:
# BUG_HIGHER_CASES="2:6,3:6"   BUG_HIGHER_INTERACTIONS="short,long"
# BUG_HIGHER_TOLS="1e-2,1e-3"  BUG_HIGHER_STEPS="6"
# BUG_HIGHER_SEEDS="2"        BUG_HIGHER_DT="0.08"
# BUG_HIGHER_POLICIES="fixed4_depth8,adaptive_lanczos,..."
# BUG_HIGHER_SAMPLE_EVERY="2"
using BondUpdateGalerkin.BUG, BondUpdateGalerkin.QMB, BondUpdateGalerkin.Baselines
using LinearAlgebra, Random, Statistics, Telum
include(joinpath(@__DIR__, "..", "..", "..", "test", "common", "dense_reference.jl"))

outdir = isempty(ARGS) ? "results/xx-higher-d" : ARGS[1]
mkpath(outdir)
set_symmetry!(:U1)
parse_list(name, default) = filter(!isempty, strip.(split(get(ENV,name,default),',')))
cases = [(parse(Int,split(s,':')[1]),parse(Int,split(s,':')[2]))
         for s in parse_list("BUG_HIGHER_CASES","2:6,3:6")]
interactions = parse_list("BUG_HIGHER_INTERACTIONS","short,long")
tolerances = parse.(Float64,parse_list("BUG_HIGHER_TOLS","1e-2,1e-3"))
nsteps = parse(Int,get(ENV,"BUG_HIGHER_STEPS","6"))
nseeds = parse(Int,get(ENV,"BUG_HIGHER_SEEDS","2"))
dt = parse(Float64,get(ENV,"BUG_HIGHER_DT","0.08"))
sample_every = parse(Int,get(ENV,"BUG_HIGHER_SAMPLE_EVERY","2"))
maxdim = parse(Int,get(ENV,"BUG_HIGHER_MAXDIM","128"))
selected = parse_list("BUG_HIGHER_POLICIES","")
sample_steps = Set(vcat(1,[k for k in 1:nsteps if k % sample_every==0],nsteps))

# Site digit a=0..twoS means |m=s-a>, so the basis is ordered from spin up down.
function sector_basis(L::Int,twoS::Int)
    d=twoS+1
    target=div(L*twoS,2)
    B=Vector{Vector{Int}}()
    for raw in 0:(d^L-1)
        x=raw
        digits=zeros(Int,L)
        for j in L:-1:1
            digits[j] = x % d
            x = div(x,d)
        end
        sum(digits)==target && push!(B,digits)
    end
    return B
end

function couplings(L, kind)
    J=zeros(Float64,L,L)
    for i in 1:L-1, j in i+1:L
        r=j-i
        J[i,j] = kind=="short" ? (r==1 ? 1.0 : 0.0) :
                 kind=="long" ? float(r)^(-3.0) :
                 throw(ArgumentError("unknown interaction $kind"))
    end
    return J
end

function xx_mpo(L,twoS,J)
    q=spin_space(twoS)
    vertices=[PairVertex(q.Sp,TLArray(q.Sp'),1.0),
              PairVertex(q.Sm,TLArray(q.Sm'),1.0)]
    return pair_mpo(L,J,vertices;Iloc=q.I)
end

# Independent sector ED: no MPO builder or tensor routines are used here.
# H_ij=(J_ij/2)(S_i^+ S_j^- + S_i^- S_j^+).
function sector_hamiltonian(B,twoS,J)
    n=length(B)
    L=length(B[1])
    d=twoS+1
    s=twoS/2
    index=Dict(Tuple(b)=>k for (k,b) in enumerate(B))
    H=zeros(Float64,n,n)
    for (k,b) in enumerate(B), i in 1:L-1, j in i+1:L
        iszero(J[i,j]) && continue
        ai,aj=b[i],b[j]
        mi,mj=s-ai,s-aj
        if ai>0 && aj<d-1
            # S_i^+ S_j^- turns (a_i,a_j) -> (a_i-1,a_j+1).
            next=copy(b); next[i]-=1; next[j]+=1
            fi=sqrt((s-mi)*(s+mi+1))
            fj=sqrt((s+mj)*(s-mj+1))
            H[index[Tuple(next)],k] += (J[i,j]/2)*fi*fj
        end
        if ai<d-1 && aj>0
            next=copy(b); next[i]+=1; next[j]-=1
            fi=sqrt((s+mi)*(s-mi+1))
            fj=sqrt((s-mj)*(s+mj+1))
            H[index[Tuple(next)],k] += (J[i,j]/2)*fi*fj
        end
    end
    norm(H-H',Inf)<1e-12 || error("ED Hamiltonian not Hermitian")
    return H
end

# Pre-compute the physical product bras once per sector. Each object contains
# exactly one U(1)-allowed computational configuration (all charge labels).
function sector_bras(B,twoS)
    q=spin_space(twoS)
    return [_full_tensor(sector_product_state(q,
       [((twoS-2*a,),) for a in b])) for b in B]
end
function sector_state(psi,bras)
    T=_full_tensor(psi)
    return ComplexF64[tensor_inner(bra,T) for bra in bras]
end
function exact_profiles(v,B,twoS)
    nrm2=real(dot(v,v))
    return [sum((twoS/2-b[j])*abs2(v[k]) for (k,b) in enumerate(B))/nrm2
            for j in 1:length(B[1])]
end
state_fidelity_error(a,b) =
    sqrt(max(0.0,1-min(1.0,abs2(dot(a,b))/real(dot(a,a)*dot(b,b)))))
fmt(x) = string(x)
function write_row(io,args...)
    println(io,join(fmt.(args),','))
    flush(io)
end

function policy_specs(basis_tol)
    return [
        ("fixed4_depth4",DDQRExpansion(mode=:fixed,fixed_width=8,fixed_rank=4,
                    krylov_mode=:lanczos,krylov_fixed_depth=4),NamedTuple()),
        ("fixed4_depth8",DDQRExpansion(mode=:fixed,fixed_width=8,fixed_rank=4,
                    krylov_mode=:lanczos,krylov_fixed_depth=8),NamedTuple()),
        ("fixed8_depth8",DDQRExpansion(mode=:fixed,fixed_width=12,fixed_rank=8,
                    krylov_mode=:lanczos,krylov_fixed_depth=8),NamedTuple()),
        ("fixed4_arnoldi8",DDQRExpansion(mode=:fixed,fixed_width=8,fixed_rank=4,
                    krylov_mode=:arnoldi,krylov_fixed_depth=8),NamedTuple()),
        ("fixed4_adaptK",DDQRExpansion(mode=:fixed,fixed_width=8,fixed_rank=4,
                    krylov_mode=:lanczos),NamedTuple()),
        ("adaptive_lanczos",DDQRExpansion(holdout_width=2,acceptance_factor=0.6,
                    krylov_mode=:lanczos),NamedTuple()),
        ("adaptive_arnoldi",DDQRExpansion(holdout_width=2,acceptance_factor=0.6,
                    krylov_mode=:arnoldi),NamedTuple()),
        ("adaptive_ranked",DDQRExpansion(holdout_width=2,acceptance_factor=0.6,
                    krylov_mode=:lanczos),(krylov_rank_tol=basis_tol,)),
        ("adaptive_unchecked",DDQRExpansion(holdout_width=2,acceptance_factor=0.6,
                    krylov_mode=:lanczos),(closure_checks=false,)),
        # Per-bond ceil(1.2*current_rank), applied to BOTH DD and K/L and root
        # closure frames. The cap is a resource restriction, not a tolerance.
        ("fixed4_cap12",DDQRExpansion(mode=:fixed,fixed_width=8,fixed_rank=4,
                    krylov_mode=:lanczos),
                    (work_ceiling_rule=:bond,bond_ceiling_factor=1.2,)),
        ("sequential_cap12",DDQRExpansion(holdout_width=2,acceptance_factor=0.6,
                    batch_blocks=1,krylov_mode=:lanczos),
                    (work_ceiling_rule=:bond,bond_ceiling_factor=1.2,)),
        ("batch2_cap12",DDQRExpansion(holdout_width=2,acceptance_factor=0.6,
                    batch_blocks=2,krylov_mode=:lanczos),
                    (work_ceiling_rule=:bond,bond_ceiling_factor=1.2,)),
        ("batch3_cap12",DDQRExpansion(holdout_width=2,acceptance_factor=0.6,
                    batch_blocks=3,krylov_mode=:lanczos),
                    (work_ceiling_rule=:bond,bond_ceiling_factor=1.2,)),
        ("batch2_uncapped",DDQRExpansion(holdout_width=2,acceptance_factor=0.6,
                    batch_blocks=2,krylov_mode=:lanczos),NamedTuple()),
        ("batch3_uncapped",DDQRExpansion(holdout_width=2,acceptance_factor=0.6,
                    batch_blocks=3,krylov_mode=:lanczos),NamedTuple()),
        ("fixed4_cap15",DDQRExpansion(mode=:fixed,fixed_width=8,fixed_rank=4,
                    krylov_mode=:lanczos),
                    (work_ceiling_rule=:bond,bond_ceiling_factor=1.5,)),
        ("batch2_cap15",DDQRExpansion(holdout_width=2,acceptance_factor=0.6,
                    batch_blocks=2,krylov_mode=:lanczos),
                    (work_ceiling_rule=:bond,bond_ceiling_factor=1.5,)),
    ]
end

rowfile=open(joinpath(outdir,"trajectories.csv"),"w")
write_row(rowfile,"d","twoS","L","interaction","basis_tol","method","seed",
 "step","t","state_error","profile_error","norm_error","D_stage","D_final",
 "D_root_left","D_root_right","root_core_elements","D_root_svd",
 "h1","h1_closure","root","root_frontier","h2_leak","dd_probes",
 "dd_holdout","mpo_stream_cols","mpo_stream_calls","n_krylov","n_dd",
 "step_sec","budget_status","unmet")
bondfile=open(joinpath(outdir,"bond_frames.csv"),"w")
write_row(bondfile,"d","L","interaction","basis_tol","method","seed","step",
 "t","bond","side","D_frame","D_final","n_dd","n_krylov","n_h1")
refft=open(joinpath(outdir,"reference.csv"),"w")
write_row(refft,"d","L","interaction","step","t","site","sz_exact")
reftest=open(joinpath(outdir,"reference_checks.csv"),"w")
write_row(reftest,"d","L","interaction","sector_dim","ed_hermitian_defect","init_vector_error","variance_ED","variance_MPO")
for (twoS,L) in cases
    iseven(L) || throw(ArgumentError("Need even L for S_z=0 domain wall"))
    d=twoS+1
    B=sector_basis(L,twoS)
    bras=sector_bras(B,twoS)
    wall=[j<=div(L,2) ? 0 : twoS for j in 1:L]
    index0=findfirst(b->b==wall,B)
    index0===nothing && error("Domain wall absent from S_z=0 basis")
    v0=zeros(ComplexF64,length(B))
    v0[index0]=1
    for interaction in interactions
        J=couplings(L,interaction)
        mpo=xx_mpo(L,twoS,J)
        H=sector_hamiltonian(B,twoS,J)
        psi0=spin_domain_wall_state(L,twoS)
        initial=sector_state(psi0,bras)
        init_error=norm(initial-v0)
        init_error <= 1e-10 || error("U1 basis/configuration ordering mismatch: $init_error")
        E=eigen(Hermitian(H))
        c0=E.vectors' * v0
        ref=[E.vectors*(exp.(-im*dt*k .* E.values).*c0) for k in 1:nsteps]
        variance_ed=real(dot(v0,H*H*v0))
        variance_mpo=mpo_variance(psi0,mpo)
        var_error=abs(variance_ed-variance_mpo)
        var_error <= 1e-9*max(1.0,abs(variance_ed)) ||
             error("Mismatch ED vs MPO action: ED=$variance_ed MPO=$variance_mpo")
        write_row(reftest,d,L,interaction,length(B),norm(H-H',Inf),
                  init_error,variance_ed,variance_mpo)
        for k in sample_steps
            p=exact_profiles(ref[k],B,twoS)
            for j in 1:L
                write_row(refft,d,L,interaction,k,k*dt,j,p[j])
            end
        end
        println("REFERENCE_OK d=$d L=$L kind=$interaction sector=$(length(B)) var=$variance_ed")
        for basis_tol in tolerances
            trunc_tol=basis_tol*1e-3
            for (method,policy,extra) in policy_specs(basis_tol)
                (!isempty(selected) && !(method in selected)) && continue
                for seed in 0:nseeds-1
                    psi=spin_domain_wall_state(L,twoS)
                    rng=MersenneTwister(seed+71)
                    for k in 1:nsteps
                        start_ns=time_ns()
                        kw=merge((order=2,closure_checks=true),extra)
                        info=cbe_bug_step!(psi,mpo,ComplexF64(-im*dt);
                           maxdim=maxdim,tol_basis=basis_tol,tol_trunc=trunc_tol,
                           tol_root=1e-10,expansion=policy,rng=rng,kw...)
                        elapsed=(time_ns()-start_ns)/1e9
                        ledger=info.ledger
                        cnt=info.record.counters
                        D=maximum(bond_dims(psi))
                        Dstage=info.record.D_stage
                        state_err,profile_err,norm_err=NaN,NaN,NaN
                        if k in sample_steps
                            v=sector_state(psi,bras)
                            state_err=state_fidelity_error(v,ref[k])
                            local_exact=exact_profiles(ref[k],B,twoS)
                            local_approx=spin_sz_profile(psi,twoS) ./ real(dot(v,v))
                            profile_err=maximum(abs.(local_approx-local_exact))
                            norm_err=abs(norm(v)-1)
                        end
                        write_row(rowfile,d,twoS,L,interaction,basis_tol,method,seed,
                            k,k*dt,state_err,profile_err,norm_err,Dstage,D,
                            info.expanded[info.root],info.root_right,
                            info.expanded[info.root]*info.root_right,info.root_rank,
                            cnt.n_one_site,ledger.n_h1_closure,cnt.n_root,
                            ledger.n_root_frontier,ledger.n_h2_leak,
                            ledger.n_probes,ledger.n_holdout,cnt.n_stream_cols,
                            cnt.n_stream,ledger.n_krylov,ledger.n_selected,
                            elapsed,budget_status(ledger),join(string.(ledger.unmet),";"))
                        # Keep left/right separate: this resolves actual root widths
                        # and distinguishes pre-root augmentation from SVD/compression.
                        for log in ledger.bond_log
                            write_row(bondfile,d,L,interaction,basis_tol,method,seed,
                                k,k*dt,log.bond,log.side,log.width,
                                bond_dims(psi)[log.bond],log.n_selected,
                                log.n_krylov,log.n_h1)
                        end
                        println("RUN d=$d L=$L $interaction tol=$basis_tol method=$method seed=$seed step=$k err=$state_err frame=$Dstage final=$D")
                    end
                end
            end
            if isempty(selected) || "tdvp2" in selected
                psi=spin_domain_wall_state(L,twoS)
                for k in 1:nsteps
                    start_ns=time_ns()
                    tdvp2_step!(psi,mpo,ComplexF64(-im*dt);
                        maxdim=maxdim,weight_tol=trunc_tol)
                    elapsed=(time_ns()-start_ns)/1e9
                    v=sector_state(psi,bras)
                    err=state_fidelity_error(v,ref[k])
                    local_exact=exact_profiles(ref[k],B,twoS)
                    local_approx=spin_sz_profile(psi,twoS)./real(dot(v,v))
                    prof_err=maximum(abs.(local_approx-local_exact))
                    write_row(rowfile,d,twoS,L,interaction,basis_tol,"tdvp2",0,
                        k,k*dt,err,prof_err,abs(norm(v)-1),0,
                        maximum(bond_dims(psi)),0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                        elapsed,"reference","")
                    println("TDVP d=$d L=$L $interaction step=$k err=$err")
                end
            end
        end
    end
end
close(rowfile); close(bondfile); close(refft); close(reftest)
println("XX_HIGHER_D_BENCHMARK_COMPLETE output=$outdir")
