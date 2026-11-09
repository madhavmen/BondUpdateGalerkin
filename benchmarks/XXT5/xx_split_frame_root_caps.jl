# Matched-state-error BUG order-2 versus TDVP2 with two-tier per-bond caps:
# nonroot DD/KL = 2*D_b_old; root input frames W/Z = ceil(1.2*D_root_old).
# Baseline: old strict ceil(1.2*D_b_old) at ALL bonds.
# U(1) XX spin 1/2, 1, 3/2, L=4, short and 1/r^3 long interactions.
# Julia: julia --project benchmarks/XXT5/xx_split_frame_root_caps.jl OUT
using BondUpdateGalerkin.BUG, BondUpdateGalerkin.QMB, BondUpdateGalerkin.Baselines
using LinearAlgebra, Random, Telum
include(joinpath(@__DIR__,"..","..","test","common","dense_reference.jl"))
set_symmetry!(:U1)
outdir=isempty(ARGS) ? "results/split-frame-root" : ARGS[1]
mkpath(outdir)
function xx_couplings(L, kind)
    J = zeros(Float64, L,L)
    for a in 1:L-1,b in a+1:L
        r=b-a
        J[a,b] = kind === :short ? (r==1 ? 1.0 : 0.0) : Float64(r)^(-3)
    end
    return J
end
function xx_mpo(L,twoS,J)
    q=spin_space(twoS)
    vertices=[PairVertex(q.Sp,TLArray(q.Sp'),1.0),
              PairVertex(q.Sm,TLArray(q.Sm'),1.0)]
    pair_mpo(L,J,vertices; Iloc=q.I)
end
function digits_of(k,d,L)
    digits=zeros(Int,L)
    for j in L:-1:1
        digits[j]=k % d
        k=div(k,d)
    end
    return digits
end
function full_hamiltonian(L,twoS,J)
    d=twoS+1
    H=zeros(ComplexF64,d^L,d^L)
    s=twoS/2
    for state in 0:d^L-1, a in 1:L-1, b in a+1:L
        J[a,b]==0 && continue
        dig=digits_of(state,d,L)
        da,db=dig[a],dig[b]
        ma,mb=s-da,s-db
        for direction in (-1,1)
            aa=da+direction
            bb=db-direction
            0<=aa<d && 0<=bb<d || continue
            next=copy(dig);next[a]=aa;next[b]=bb
            idx=foldl((v,w)->v*d+w,next;init=0)
            if direction == -1
                fa=sqrt((s-ma)*(s+ma+1));fb=sqrt((s+mb)*(s-mb+1))
            else
                fa=sqrt((s+ma)*(s-ma+1));fb=sqrt((s-mb)*(s+mb+1))
            end
            H[idx+1,state+1] += J[a,b]*fa*fb/2
        end
    end
    norm(H-H',Inf)<1e-12 || error("ED generator is not Hermitian")
    return Hermitian(H)
end

# U(1)-aware exact bra contractions; unlike generic dense_state(), these retain
# each physical charge label and have already been tested on d=3 and d=4.
function sector_indices_and_bras(L,twoS)
    d=twoS+1
    target=div(L*twoS,2)
    q=spin_space(twoS)
    indices=Int[]
    bras=Any[]
    for k in 0:d^L-1
        b=digits_of(k,d,L)
        sum(b)==target || continue
        push!(indices,k+1)
        push!(bras,_full_tensor(sector_product_state(q,
                        [((twoS-2*a,),) for a in b])))
    end
    return indices,bras
end
function sector_state(psi,bras)
    T=_full_tensor(psi)
    return ComplexF64[tensor_inner(bra,T) for bra in bras]
end
function error_and_norm(psi,ref,bras)
    v=sector_state(psi,bras)
    f=min(1.0,abs2(dot(v,ref))/(real(dot(v,v))*real(dot(ref,ref))))
    return sqrt(max(0.0,1.0-f)),abs(norm(v)-1.0)
end
const header=("d","L","range","dt","basis_tol","policy","seed",
  "accepted","attempts","budget_status","unmet",
  "error","norm_error","DD_discovery","DD_holdouts","K_L_H",
  "closure_H","root_H","max_aug","root_left","root_right",
  "root_core_elements","final_bond","wall_s")
function write_line(io,vals...)
    println(io,join(string.(vals),","))
    flush(io)
end

function record_rows(io,values...)
    println(io,join(string.(values),","))
    flush(io)
end
policies=(:old_strict,:frame2_root12,:rank_frame2_root12,:coupled,:tdvp2)
raw_tols=get(ENV,"BUG_SPLIT_TOLS","1e-2,1e-3")
tols=parse.(Float64,split(raw_tols,","))
nsteps=parse(Int,get(ENV,"BUG_SPLIT_STEPS","6"))
nseeds=parse(Int,get(ENV,"BUG_SPLIT_SEEDS","2"))
dt=parse(Float64,get(ENV,"BUG_SPLIT_DT","0.05"))
open(joinpath(outdir,"split_frame_root_trajectories.csv"),"w") do io
    record_rows(io,"d","L","range","dt","basis_tol","policy","seed","step","t",
       "accepted","attempts","budget","unmet","state_error","norm_error",
       "DD_discovery","DD_holdouts","K_L_H","closure_H","root_H",
       "peak_aug","root_left","root_right","root_core_elements",
       "peak_final_bond","frame_cap_excess","root_cap_excess","wall_s")
    for twoS in (1,2,3), kind in (:short,:long)
        d=twoS+1; L=4
        start=spin_domain_wall_state(L,twoS)
        J=xx_couplings(L,kind)
        mpo=xx_mpo(L,twoS,J)
        Hfull=full_hamiltonian(L,twoS,J)
        indices,bras=sector_indices_and_bras(L,twoS)
        H=Hermitian(Matrix(Hfull)[indices,indices])
        wall=[j<=div(L,2) ? 0 : twoS for j in 1:L]
        k0=findfirst(i -> digits_of(indices[i]-1,d,L)==wall,eachindex(indices))
        v0=zeros(ComplexF64,length(indices)); v0[k0]=1
        norm(sector_state(start,bras)-v0)<1e-10 ||
            error("Sector/domain wall mismatch d=$d")
        isapprox(real(dot(v0,H*(H*v0))),mpo_variance(start,mpo);
                 rtol=1e-9,atol=1e-10) ||
            error("MPO/ED initial variance mismatch d=$d $kind")
        ev=eigen(H);c0=ev.vectors'*v0
        for tol in tols, seed in 0:nseeds-1, policy in policies
            psi=spin_domain_wall_state(L,twoS)
            for k in 1:nsteps
                oldb=bond_dims(psi)
                root=fld(L,2)
                caps=policy===:old_strict ?
                    bond_work_ceilings_strict(oldb,128;factor=1.2) :
                    bond_frame_root_ceilings(oldb,128,root;frame_factor=2.0,root_factor=1.2)
                info=nothing;accepted=true;ntries=1;reason="reference"
                w=(DD_discovery=0,DD_holdouts=0,K_L_H=0,closure_H=0,root_H=0)
                tic=time_ns()
                if policy===:tdvp2
                    tdvp2_step!(psi,mpo,-im*dt;maxdim=128,weight_tol=tol*1e-3)
                elseif policy===:coupled
                    run=cbe_bug_coupled_step(psi,mpo,-im*dt;
                       tol_basis=tol,tol_trunc=tol*1e-3,tol_root=1e-10,
                       rank_initial_width=4,holdout_width=4,
                       acceptance_factor=0.7,maxdim=128,max_retries=4,
                       rng=MersenneTwister(991+seed))
                    info=run.info
                    psi=run.state
                    accepted=run.accepted
                    ntries=length(run.attempts)
                    reason=accepted ? "met" : "rejected"
                    w=run.totals
                else
                    dd=policy===:rank_frame2_root12 ?
                        DDQRExpansion(mode=:rank_adaptive,rank_initial_width=4,
                            holdout_width=4,acceptance_factor=0.7,
                            krylov_mode=:lanczos) :
                        DDQRExpansion(mode=:fixed,fixed_width=8,fixed_rank=4,
                            krylov_mode=:lanczos)
                    info=cbe_bug_step!(psi,mpo,-im*dt;
                        order=2,closure_checks=true,
                        tol_basis=tol,tol_trunc=tol*1e-3,tol_root=1e-10,
                        expansion=dd,maxdim=128,
                        work_ceiling_rule=policy===:old_strict ? :bond_strict : :frame_root,
                        bond_ceiling_factor=policy===:old_strict ? 1.2 : 2.0,
                        root_ceiling_factor=1.2,
                        rng=MersenneTwister(991+seed))
                    accepted=info.ledger.tolerance_met
                    reason=join(string.(info.ledger.unmet),";")
                    w=(DD_discovery=info.ledger.n_probes,
                       DD_holdouts=info.ledger.n_holdout,
                       K_L_H=info.ledger.n_h1,
                       closure_H=info.ledger.n_h1_closure,
                       root_H=info.record.counters.n_root)
                end
                wall_s=(time_ns()-tic)/1e9
                rleft=info===nothing ? 0 : info.expanded[info.root]
                rright=info===nothing ? 0 : info.root_right
                aug=info===nothing ? 0 : info.record.D_stage
                badframe=info===nothing ? 0 : maximum(info.expanded .- caps)
                badroot=info===nothing ? 0 : max(rleft,rright)-caps[root]
                if info!==nothing
                    badframe<=0 || error("frame ceiling exceeded")
                    badroot<=0 || error("root frame ceiling exceeded")
                    for entry in info.ledger.bond_log
                        entry.width<=caps[entry.bond] || error("logged augmented frame exceeds ceiling")
                    end
                end
                # Rejected coupled steps are not advanced: no t+dt error exists.
                if policy===:coupled && !accepted
                    state_error=NaN;norm_error=NaN
                else
                    ref=ev.vectors*(exp.(-im*(k*dt).*ev.values).*c0)
                    state_error,norm_error=error_and_norm(psi,ref,bras)
                end
                record_rows(io,d,L,kind,dt,tol,policy,seed,k,k*dt,
                   accepted,ntries,info===nothing ? "reference" : string(budget_status(info.ledger)),
                   reason,state_error,norm_error,
                   w.DD_discovery,w.DD_holdouts,w.K_L_H,w.closure_H,w.root_H,
                   aug,rleft,rright,rleft*rright,maximum(bond_dims(psi)),
                   badframe,badroot,wall_s)
                println("SPLIT d=$d $kind policy=$policy step=$k tol=$tol error=$state_error accepted=$accepted root=$rleft,$rright work=$(w.K_L_H+w.closure_H)")
                policy===:coupled && !accepted && break
            end
        end
    end
end
println("FRAME2_ROOT1p2_XX_BENCHMARK_COMPLETE")
