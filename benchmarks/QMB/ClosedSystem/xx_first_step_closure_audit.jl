# Physical first-step mechanism audit: exact DD vs randomized, closure checks,
# second/third order, root outside-residual enrichment and TDVP2.
# Julia 1.11; run from the repository root:
# julia --project benchmarks/QMB/ClosedSystem/xx_first_step_closure_audit.jl RESULTS_DIR
using BondUpdateGalerkin.BUG, BondUpdateGalerkin.QMB, BondUpdateGalerkin.Baselines
using LinearAlgebra, Random, Telum
include(joinpath(@__DIR__, "..", "..", "..", "test", "common", "dense_reference.jl"))
outdir = isempty(ARGS) ? "results/xx-first-step" : ARGS[1]
mkpath(outdir)
set_symmetry!(:U1)
L, dt = 8, 0.05
Jmat(kind) = [j > i ? ((kind == :short && j-i==1) ? 1.0 :
     kind == :long ? Float64(j-i)^(-3.0) : 0.0) : 0.0 for i=1:L,j=1:L]
function h_exact(J)
    H=zeros(ComplexF64,2^L,2^L)
    for k in 0:(2^L-1),i in 1:(L-1),j in (i+1):L
        J[i,j] == 0 && continue
        ((k>>(L-i))&1)==((k>>(L-j))&1) && continue
        next=xor(k,(1<<(L-i))|(1<<(L-j)))
        H[next+1,k+1] += J[i,j]/2
    end
    return H
end
function h_mpo(J)
    q=spin_space(1)
    v=[PairVertex(q.Sp,q.Sp',1.0),PairVertex(q.Sm,q.Sm',1.0)]
    pair_mpo(L,J,v;Iloc=q.I)
end
function error_measures(a,b)
    na,nb=norm(a),norm(b)
    f=min(1.0,abs2(dot(a,b))/(na^2*nb^2))
    return sqrt(max(0.0,1-f)), abs(na-1.0)
end
cases=[
    ("default_order2",(order=2,closure_checks=false,exact=false,outside_residual=false,
         expansion=DDQRExpansion())),
    ("conservative_order2",(order=2,closure_checks=false,exact=false,outside_residual=false,
         expansion=DDQRExpansion(holdout_width=2,acceptance_factor=0.6))),
    ("checked_closure",(order=2,closure_checks=true,exact=false,outside_residual=false,
         expansion=DDQRExpansion(holdout_width=2,acceptance_factor=0.6))),
    ("root_outside",(order=2,closure_checks=true,exact=false,outside_residual=true,
         expansion=DDQRExpansion(holdout_width=2,acceptance_factor=0.6))),
    ("exact_DD",(order=2,closure_checks=false,exact=true,outside_residual=false,
         expansion=DDQRExpansion())),
    ("exact_DD_checked",(order=2,closure_checks=true,exact=true,outside_residual=false,
         expansion=DDQRExpansion())),
    ("order3_checked",(order=3,closure_checks=true,exact=false,outside_residual=false,
         expansion=DDQRExpansion(holdout_width=2,acceptance_factor=0.6))),
]
open(joinpath(outdir,"first_step_audit.csv"),"w") do io
    println(io,join(("interaction","method","error_full","norm_error","D_stage","D_final",
           "D_root_left","D_root_right","D_root_svd","n_h1","n_probes","n_holdout",
           "n_krylov","n_dd","root_iters","t_step","budget_status","unmet"),","))
    for kind in (:short,:long)
        J=Jmat(kind);mpo=h_mpo(J)
        start=spin_domain_wall_state(L,1)
        exact=exp(-im*dt*h_exact(J))*dense_state(start)
        for (name,stepkw) in cases
            psi=spin_domain_wall_state(L,1)
            rr=Ref{Any}()
            st=time_ns()
            info=cbe_bug_step!(psi,mpo,-im*dt;maxdim=64,tol_basis=1e-5,
                 tol_trunc=1e-8,tol_root=1e-10,rng=MersenneTwister(7),
                 root_report=stepkw.outside_residual ? rr : nothing,stepkw...)
            elapsed=(time_ns()-st)/1e9
            err,normerr=error_measures(dense_state(psi),exact)
            Lg=info.ledger
            row=(kind,name,err,normerr,info.record.D_stage,maximum(bond_dims(psi)),
                info.expanded[info.root],info.root_right,info.root_rank,
                Lg.n_h1,Lg.n_probes,Lg.n_holdout,Lg.n_krylov,Lg.n_selected,
                info.record.root_iters,elapsed,budget_status(Lg),
                join(string.(Lg.unmet),";"))
            println(io,join(row,","))
            println(join(row,","))
        end
        psi=spin_domain_wall_state(L,1)
        st=time_ns()
        tdvp2_step!(psi,mpo,-im*dt;maxdim=64,weight_tol=1e-8)
        err,normerr=error_measures(dense_state(psi),exact)
        row=(kind,"tdvp2",err,normerr,0,maximum(bond_dims(psi)),0,0,0,
             0,0,0,0,0,0,(time_ns()-st)/1e9,:reference,"")
        println(io,join(row,","));println(join(row,","))
    end
end
println("FIRST-STEP-AUDIT-DONE")
