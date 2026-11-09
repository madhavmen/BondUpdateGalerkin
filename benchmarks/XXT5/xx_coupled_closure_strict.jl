# Executable physical U(1) XX test of coupled DD rank/checked Krylov closure.
# Small exact full-space ED verifies the state error, rather than using the
# closure diagnostic as an 'exact' accuracy measure.
# Run from repo root:
#   julia --project benchmarks/XXT5/xx_coupled_closure_strict.jl results/coupled
using BondUpdateGalerkin.BUG, BondUpdateGalerkin.QMB, BondUpdateGalerkin.Baselines
using LinearAlgebra, Random, Telum
include(joinpath(@__DIR__, "..", "..", "test", "common", "dense_reference.jl"))
set_symmetry!(:U1)
outdir = isempty(ARGS) ? "results/coupled-closure" : ARGS[1]
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
open(joinpath(outdir,"physical_one_step.csv"),"w") do io
    write_line(io,header...)
    for twoS in (1,2,3), kind in (:short,:long)
        d=twoS+1
        L=4
        initial=spin_domain_wall_state(L,twoS)
        J=xx_couplings(L,kind)
        mpo=xx_mpo(L,twoS,J)
        Hfull=full_hamiltonian(L,twoS,J)
        indices,bras=sector_indices_and_bras(L,twoS)
        H=Hermitian(Matrix(Hfull)[indices,indices])
        wall=[j<=div(L,2) ? 0 : twoS for j in 1:L]
        wall_idx=findfirst(i -> digits_of(indices[i]-1,d,L)==wall, eachindex(indices))
        wall_idx === nothing && error("Domain-wall state not in S_z=0 sector")
        v0=zeros(ComplexF64,length(indices));v0[wall_idx]=1
        norm(sector_state(initial,bras)-v0)<1e-11 ||
           error("U1 MPS sector basis differs from exact domain-wall state")
        varED=real(dot(v0,H*(H*v0)))
        varMPO=mpo_variance(initial,mpo)
        isapprox(varED,varMPO;rtol=1e-9,atol=1e-10) ||
           error("MPO/ED generator mismatch d=$d $kind ($varED vs $varMPO)")
        E=eigen(H)
        c0=E.vectors'*v0
        for dt in (0.02,0.10), tol in (1e-2,1e-3), seed in 0:1
            ref=E.vectors * (exp.(-im*dt.*E.values).*c0)
            caps=bond_work_ceilings_strict(bond_dims(initial),128;factor=1.2)
            @assert all(caps .== 2) # product-state wall, first step d-independent
            for policy in (:fixed,:rank_only,:coupled,:tdvp2)
                psi=spin_domain_wall_state(L,twoS)
                t0=time_ns()
                info=nothing
                accepted=true
                natt=1
                work=(DD_discovery=0,DD_holdouts=0,K_L_H=0,
                      closure_H=0,root_H=0,wall_s=0.0)
                if policy === :coupled
                    run=cbe_bug_coupled_step(
                        psi,mpo,-im*dt;tol_basis=tol,
                        tol_trunc=tol*1e-3,tol_root=1e-10,
                        maxdim=128,max_retries=4,
                        rank_initial_width=4,holdout_width=4,
                        acceptance_factor=0.7,
                        rng=MersenneTwister(seed+17),
                    )
                    info=run.info;psi=run.state
                    accepted=run.accepted;natt=length(run.attempts)
                    work=run.totals
                    @assert (accepted || norm(sector_state(psi,bras)-v0)<1e-12)
                elseif policy === :tdvp2
                    tdvp2_step!(psi,mpo,-im*dt;maxdim=128,weight_tol=tol*1e-3)
                else
                    dd=policy===:fixed ?
                       DDQRExpansion(mode=:fixed,fixed_width=8,fixed_rank=4,
                                     krylov_mode=:lanczos) :
                       DDQRExpansion(mode=:rank_adaptive,rank_initial_width=4,
                                     holdout_width=4,acceptance_factor=0.7,
                                     krylov_mode=:lanczos)
                    info=cbe_bug_step!(psi,mpo,-im*dt;
                        order=2,closure_checks=true,
                        tol_basis=tol,tol_trunc=tol*1e-3,tol_root=1e-10,
                        expansion=dd,maxdim=128,
                        work_ceiling_rule=:bond_strict,bond_ceiling_factor=1.2,
                        rng=MersenneTwister(seed+17))
                    accepted=info.ledger.tolerance_met
                    work=(DD_discovery=info.ledger.n_probes,
                          DD_holdouts=info.ledger.n_holdout,
                          K_L_H=info.ledger.n_h1,
                          closure_H=info.ledger.n_h1_closure,
                          root_H=info.record.counters.n_root,
                          wall_s=info.t_step)
                end
                elapsed=(time_ns()-t0)/1e9
                err,normerr=error_and_norm(psi,ref,bras)
                if info===nothing
                    wleft=0;wright=0;stage=0;reason="reference";status="reference"
                else
                    wleft=info.expanded[info.root]
                    wright=info.root_right
                    stage=info.record.D_stage
                    reason=join(string.(info.ledger.unmet),";")
                    status=string(budget_status(info.ledger))
                    if policy !== :coupled || accepted
                        @assert all(info.expanded .<= caps)
                        @assert info.root_right <= caps[info.root]
                        @assert all(bond_dims(psi) .<= caps)
                    end
                end
                write_line(io,d,L,kind,dt,tol,policy,seed,accepted,natt,
                    status,reason,err,normerr,work.DD_discovery,
                    work.DD_holdouts,work.K_L_H,work.closure_H,
                    work.root_H,stage,wleft,wright,wleft*wright,
                    maximum(bond_dims(psi)),elapsed)
                println("COUPLED_SMOKE d=$d $kind dt=$dt tol=$tol $policy "*
                        "accepted=$accepted error=$err attempts=$natt")
            end
        end
    end
end
println("COUPLED_STRICT_1p2_BOND_BENCHMARK_COMPLETE")
