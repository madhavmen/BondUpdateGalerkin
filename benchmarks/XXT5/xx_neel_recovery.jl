# Exact U(1) spin-1/2 XX Neel/domain-wall bulk-quench recovery.
# d=2, nearest-neighbor or power-law XX, OBC; tiny exact ED for L<=10.
# Compare legacy :bond, proposed :bulk, unrestricted working-frame :maxdim,
# transactional accepted-step retry, TDVP2 and A/B Krylov DD at matched error.
# Run: BUG_NEEL_L=10 BUG_NEEL_STEPS=50 julia --project benchmarks/XXT5/xx_neel_recovery.jl OUT
using BondUpdateGalerkin.BUG, BondUpdateGalerkin.QMB, BondUpdateGalerkin.Baselines
using LinearAlgebra, Random, Telum
include(joinpath(@__DIR__,"..","..","test","common","dense_reference.jl"))
set_symmetry!(:U1)
L=parse(Int,get(ENV,"BUG_NEEL_L","8"))
iseven(L) || error("Need even L for U1 zero charge")
nsteps=parse(Int,get(ENV,"BUG_NEEL_STEPS","2"))
dt=parse(Float64,get(ENV,"BUG_NEEL_DT","0.1"))
seeds=parse(Int,get(ENV,"BUG_NEEL_SEEDS","1"))
interaction_kind=get(ENV,"BUG_NEEL_RANGE","short")
initial_kind=get(ENV,"BUG_NEEL_START","neel")
maxdim=parse(Int,get(ENV,"BUG_NEEL_MAXDIM","64"))
tol=parse(Float64,get(ENV,"BUG_NEEL_TOL","1e-3"))
sample_every=parse(Int,get(ENV,"BUG_NEEL_SAMPLE_EVERY","5"))
mode_filter=filter(!isempty,split(get(ENV,"BUG_NEEL_MODES",""),','))
interaction_kind in ("short","long") || error("range short/long")
initial_kind in ("neel","domain_wall") || error("start neel/domain_wall")
outdir=isempty(ARGS) ? "results/neel-recovery" : ARGS[1]
mkpath(outdir)
make_initial()=initial_kind=="neel" ? neel_state(L) : spin_domain_wall_state(L,1)
J=zeros(Float64,L,L)
for i=1:L-1,j=i+1:L
    J[i,j]=interaction_kind=="short" ? (j==i+1 ? 1. : 0.) : (j-i)^(-3.)
end
sp=spin_space(1)
mpo=pair_mpo(L,J,[PairVertex(sp.Sp,TLArray(sp.Sp'),1.0),
                   PairVertex(sp.Sm,TLArray(sp.Sm'),1.0)];Iloc=sp.I)
function reference_builder()
    # Hamiltonian built independently in the S_z=0 computational sector.
    # For NN: free-fermion validation also available at L16.
    L<=10 || return nothing
    configurations=[k for k=0:(2^L-1) if count_ones(k)==L÷2]
    indices=Dict(k=>j for (j,k) in enumerate(configurations))
    H=zeros(Float64,length(configurations),length(configurations))
    for (j,bits) in enumerate(configurations),i=1:L-1,k=i+1:L
        J[i,k]==0 && continue
        bi=(bits>>(L-i)) & 1
        bk=(bits>>(L-k)) & 1
        bi==bk && continue
        other=xor(bits, (1<<(L-i)) | (1<<(L-k)))
        H[indices[other],j]+=J[i,k]*.5
    end
    # A fixed spin-basis configuration, independent from the MPO constructor.
    v0=zeros(ComplexF64,length(configurations))
    pattern= initial_kind=="neel" ? [isodd(i) ? 0 : 1 for i=1:L] :
                [i<=L÷2 ? 0 : 1 for i=1:L]
    initial=sum(pattern[i]<<(L-i) for i=1:L)
    v0[indices[initial]]=1.0
    eig=eigen(Symmetric(H))
    coeff=eig.vectors' * v0
    exact(t)=eig.vectors * (exp.(-im*t .* eig.values).*coeff)
    return (configurations=configurations,initial=v0,exact=exact)
end
ref=reference_builder()
if ref !== nothing
    v0=dense_state(make_initial())
    norm(v0[[k+1 for k in ref.configurations]]-ref.initial)<1e-11 ||
        error("Physical Neel initial-state convention mismatch")
end
# Root/intermediate caps may exceed maxdim; the final MPS never may.
policies=[
    ("bond_tail", :rank_tail,:bond,:none),
    ("bulk_tail", :rank_tail,:bulk,:none),
    ("maxdim_tail", :rank_tail,:maxdim,:none),
    ("retry_tail", :rank_tail,:retry,:none),
    ("warmstart_tail", :rank_tail,:bond,:none),
    ("bulk_rank_adaptive", :rank_adaptive,:bulk,:none),
    ("bulk_A", :rank_tail,:bulk,:interleaved),
    ("bulk_B", :rank_tail,:bulk,:post),
    ("tdvp2", :none,:none,:none),
]
choose=isempty(mode_filter) ? policies : filter(x->x[1] in mode_filter,policies)
open(joinpath(outdir,"trajectories.csv"),"w") do io
    println(io,"initial,L,range,method,seed,step,t,error,state_error,sz_error,wall_s,h1,h1_closure,root_h,stream_cols,stream_calls,frame_dim,bond_dim,met,unmet,attempts")
    for (name,rsvd,cap,extra) in choose,seed=0:seeds-1
        psi=make_initial()
        rng=MersenneTwister(71+seed)
        for step=1:nsteps
            attempts=1
            h1=0;h1cl=0;root=0;sc=0;calls=0
            met=true;unmet=""
            begin_ns=time_ns()
            if name=="tdvp2" || (name=="warmstart_tail" && step==1)
                tdvp2_step!(psi,mpo,ComplexF64(-im*dt);
                            maxdim=maxdim,weight_tol=tol*1e-3)
                D_stage=maximum(bond_dims(psi))
            else
                p=DDQRExpansion(mode=rsvd,
                     fixed_width=8,tail_factor=0.1,
                     rank_initial_width=4,holdout_width=4,
                     acceptance_factor=0.7,krylov_mode=:lanczos)
                kwargs=(order=2,closure_checks=true,tol_basis=tol,
                        tol_trunc=tol*1e-3,tol_root=1e-10,maxdim=maxdim,
                        expansion=p,extra_dd_mode=extra,
                        extra_dd_width=8,extra_dd_holdout=4,
                        extra_dd_max_checks=2)
                if cap==:retry
                    run=cbe_bug_bulk_retry_step!(psi,mpo,ComplexF64(-im*dt);
                        modes=(:bulk,:maxdim),bond_ceiling_factor=1.2,
                        bulk_min_width=8,bulk_headroom=0.25,rng=rng,
                        kwargs...)
                    attempts=length(run.history)
                    h1=run.h1;root=run.root_h;sc=run.stream_columns
                    calls=sum(a.stream_calls for a in run.history)
                    met=run.accepted
                    unmet=met ? "" : join(string.(run.history[end].unmet),";")
                    if !met
                        println("REJECTED_ALL name=$name at step=$step -- trajectory NOT advanced")
                        break
                    end
                    info=run.info
                else
                    info=cbe_bug_step!(psi,mpo,ComplexF64(-im*dt);
                        work_ceiling_rule=cap,
                        bond_ceiling_factor=cap==:maxdim ? 0.0 : 1.2,
                        bulk_min_width=8,bulk_headroom=0.25,
                        rng=rng,kwargs...)
                    met=info.ledger.tolerance_met
                    unmet=join(string.(info.ledger.unmet),";")
                    h1=info.record.counters.n_one_site
                    root=info.record.counters.n_root
                    sc=info.record.counters.n_stream_cols
                    calls=info.record.counters.n_stream
                end
                h1cl=info.ledger.n_h1_closure
                D_stage=info.record.D_stage
            end
            elapsed=(time_ns()-begin_ns)/1e9
            D=maximum(bond_dims(psi))
            D<=maxdim || error("final bond dimension exceeds maxdim")
            if ref !== nothing && (step==1 || step==nsteps || step%sample_every==0)
                approximate=dense_state(psi)[[k+1 for k in ref.configurations]]
                exact=ref.exact(step*dt)
                state_error=sqrt(max(0.,1-min(1.,abs2(dot(exact,approximate))/
                   (real(dot(exact,exact))*real(dot(approximate,approximate))))))
                mags=[sum((0.5-((bits>>(L-j))&1)) * abs2(exact[k])
                     for (k,bits) in enumerate(ref.configurations)) for j=1:L]
                prof=spin_sz_profile(psi,1)./real(overlap(psi,psi))
                sz_error=maximum(abs.(prof.-mags))
            else
                state_error=NaN;sz_error=NaN
            end
            println(io,join((initial_kind,L,interaction_kind,name,seed,step,step*dt,
                state_error,state_error,sz_error,elapsed,h1,h1cl,root,
                sc,calls,D_stage,D,met,unmet,attempts),","))
            flush(io)
            println("NEEL_XX name=$name range=$interaction_kind k=$step fid=$state_error sz=$sz_error met=$met t=$elapsed")
        end
    end
end
println("NEEL_RECOVERY_FINISHED")
