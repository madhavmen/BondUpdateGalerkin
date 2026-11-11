# Executed by GitHub Actions on the real BUGJulia implementation.
# Uniform ASSET PRICE S grid (not a uniform log(S) grid); fixed time step h = 0.1.
# Compare real qtt_bug_step! local solver variants against independent sparse Radau IIA.
# Usage: julia --project benchmarks/QTTS/heston_bug_stiff_dt01.jl mode bits nsteps initial bc outfile [frame_rule] [recycle_previous]
using LinearAlgebra, SparseArrays, Random
using BondUpdateGalerkin.BUG, BondUpdateGalerkin.QTTS
include(joinpath(@__DIR__, "common.jl"))

const H = 0.1
const P = HestonParams(0.03, 0.01, 3.0, 0.25, 1.0, -0.8)
const SMIN, SMAX = 0.5, 1.5
const VMIN, VMAX = 0.0, 0.5

mode = Symbol(get(ARGS, 1, "rational"))
bits = parse(Int, get(ARGS, 2, "4"))
nsteps = parse(Int, get(ARGS, 3, "10"))
initial = Symbol(get(ARGS, 4, "gaussian"))
bc = Symbol(get(ARGS, 5, "dirichlet0"))
outfile = get(ARGS, 6, "results/heston-bug.csv")
frame_rule = Symbol(get(ARGS, 7, "bond"))
recycle_previous = parse(Bool, get(ARGS, 8, "false"))
@assert mode in (:rational, :gmres, :stage)
@assert initial in (:gaussian, :multimode, :affine)
@assert bc in (:onesided, :dirichlet0)
@assert frame_rule in (:bond, :maxdim)

set_symmetry!(:none)
clear_fusion_cache!()
g = qtt_grid(bits, bits; x=(SMIN, SMAX), y=(VMIN, VMAX),
             periodic_x=false, periodic_y=false)
xs, vs = qtt_coords(g)
nx, nv = length(xs), length(vs)

# The Heston generator on UNIFORM SPOT S:
# (v S^2 /2) u_SS + rho sigma v S u_Sv + (sigma^2 v/2) u_vv
# +(r-q) S u_S + kappa(theta-v) u_v - r u.
# qtt_coord_diag and qtt_d use coordinates S and v uniformly; there is NO log transform.
function heston_spot_op(g, p; bc=:dirichlet0)
    ds, dss = qtt_d(g, :x, 1; bc=bc), qtt_d(g, :x, 2; bc=bc)
    dv, dvv = qtt_d(g, :y, 1; bc=bc), qtt_d(g, :y, 2; bc=bc)
    spot, variance = qtt_coord_diag(g, :x), qtt_coord_diag(g, :y)
    ss = op_compress(op_compose(spot, spot))
    sds = op_compress(op_compose(spot, ds))
    vdv = op_compress(op_compose(variance, dv))
    vdvv = op_compress(op_compose(variance, dvv))
    return qtt_op(g, Any[
        (0.5, op_compress(op_compose(ss, dss)), variance),
        (p.rho*p.sigma, sds, vdv),
        (0.5*p.sigma^2, nothing, vdvv),
        (p.r-p.q, sds, nothing),
        (p.kappa*p.theta, nothing, dv),
        (-p.kappa, nothing, vdv),
        (-p.r, nothing, op_identity(g.ny)),
    ])
end

function heston_spot_sparse(g, p; bc=:dirichlet0)
    ns, nv = 2^g.nx, 2^g.ny
    s, v = qtt_coords(g)
    ds=sparse_stencil(ns,g.dx,1,bc); dss=sparse_stencil(ns,g.dx,2,bc)
    dv=sparse_stencil(nv,g.dy,1,bc); dvv=sparse_stencil(nv,g.dy,2,bc)
    S=spdiagm(0=>collect(s)); V=spdiagm(0=>collect(v))
    kr(ax,av)=grid_kron(g,ax,av)
    return sparse(
        0.5*kr(S^2*dss,V) + p.rho*p.sigma*kr(S*ds,V*dv) +
        0.5*p.sigma^2*kr(nothing,V*dvv) +
        (p.r-p.q)*kr(S*ds,nothing) +
        p.kappa*p.theta*kr(nothing,dv) -
        p.kappa*kr(nothing,V*dv) -
        p.r*spdiagm(0=>ones(ns*nv))
    )
end

function initial_cores(g, kind)
    if kind === :gaussian
        return qtt_function(g,
            (s,v)->exp(-((s-1.03)/0.13)^2-((v-0.24-0.16*(s-1.03))/0.070)^2);
            tol=1e-12,maxdim=160)
    elseif kind === :multimode
        # 72 independent separated sine products in 128 x 128, for a genuine rank-stress test.
        # End values vanish at price and high-variance absorbing boundaries.
        nmodes=min(72,2^g.nx-2)
        return qtt_function(g, (s,v)->begin
            sx=(s-SMIN)/(SMAX-SMIN); vy=(v-VMIN)/(VMAX-VMIN)
            sum(((isodd(k) ? 1.0 : -1.0)/(1+0.015*k)) *
                sinpi(k*sx)*sinpi(k*vy) for k in 1:nmodes)
        end;tol=1e-12,maxdim=160)
    else
        return qtt_separable(g,s->s^0.5,v->exp(-v);tol=1e-12)
    end
end

op=heston_spot_op(g,P;bc=bc)
Aref = bits<=5 ? heston_spot_sparse(g,P;bc=bc) : nothing
init=initial_cores(g,initial)
psi=qtt_mps(init)
flow=LinearFlow(op)
u0=vec(qtt_values(init,g))
@assert all(isfinite,u0)
@assert g.dx ≈ (SMAX-SMIN)/(2^bits-1) # uniform S (never log S)
@assert g.dy ≈ (VMAX-VMIN)/(2^bits-1)
initial_rank=maximum(bond_dims(psi);init=1)
println("GRID price=uniform_S, NOT_log_S; ns=$nx nv=$nv dS=$(g.dx) dv=$(g.dy), dt=$H")
println("MODEL parameters=$(P), boundary=$bc, mode=$mode, start_max_bond=$initial_rank")
println("MODEL grid values left=$(first(xs)) right=$(last(xs)); numel=$(nx*nv)")

# Independent finite-difference reference, assembled without the QTT MPO, and
# evolved with a sparse, unprojected, two-stage Radau IIA step.
refstate=copy(u0)
ref_factor=nothing
operator_error=NaN
if Aref !== nothing
    action=vec(qtt_values(op_apply(op,init),g))
    operator_error=norm(action-Aref*u0)/max(norm(Aref*u0),eps())
    println("OPERATOR independent-sparse relative error=$operator_error")
    operator_error<1e-7 || error("Uniform-S QTT operator differs from independently assembled sparse operator")
    N=nx*nv
    Id=sparse(1.0I,N,N)
    r1=hcat(Id-H*(5/12)*Aref, H*(1/12)*Aref)
    r2=hcat(-H*(3/4)*Aref, Id-H*(1/4)*Aref)
    ref_factor=lu(vcat(r1,r2))
else
    println("REFERENCE skip full-grid sparse factorization for bits=$bits; only rank and local convergence audited")
end

mkpath(dirname(outfile))
columns=(
    "mode","initial","boundary","grid","bits","step","time","dS","dt",
    "chi_initial","chi_retained","chi_augmented","frame_left","frame_right",
    "local_residual","converged","rank_capped","discarded","miss_input",
    "frame_defect","relative_error_discrete_radau","root_seconds","step_seconds",
    "basis_fallback","status"
)
open(outfile,"w") do io
    println(io,join(columns,","))
end
csv(row)=open(outfile,"a") do io
    println(io,join(string.(row),","))
end

peak_rank=initial_rank
peak_aug=initial_rank
last_rank=initial_rank
stop_reason="completed"
wall_start=time()
previous_state=nothing
function run_benchmark!()
    global psi, refstate, peak_rank, peak_aug, last_rank, stop_reason, previous_state
for k=1:nsteps
    # Fix h = 0.1: no time adaptivity, no logarithmic grid in price.
    kwargs=(; order=3,tol_basis=1e-6,tol_trunc=1e-8,
            tol_stage=1e-9,maxdim=160,
            work_ceiling_rule=frame_rule,
            bond_ceiling_factor=frame_rule===:bond ? 1.2 : 0.0)
    try
        saved_input=copy(psi)
        retained= if recycle_previous && previous_state!==nothing
            Any[previous_state]
        else
            Any[]
        end
        info= if mode === :stage
            qtt_bug_step!(psi,flow,(k-1)*H,H;
                kwargs...,root=:radau2,npc=1,stage_solver=:newton_gmres,
                stage_recycle=4,stage_basis=:predictor,
                keep=retained)
        else
            qtt_bug_step!(psi,flow,(k-1)*H,H;
                kwargs...,root=(mode === :gmres ? :radau2_gmres : :radau2),
                stage_solver=:krylov_rational,keep=retained)
        end
        previous_state=saved_input
        if ref_factor!==nothing
            stack=ref_factor\vcat(refstate,refstate)
            refstate=stack[(nx*nv+1):(2*nx*nv)]
        end
        actual=vec(qtt_values(qtt_cores(psi),g))
        err=ref_factor===nothing ? NaN : norm(actual-refstate)/max(norm(refstate),eps())
        chi=maximum(bond_dims(psi);init=1)
        aug=max(info.dd.r_aug,info.frame_width...)
        root_res=maximum((s.residual for s in info.stats);init=0.0)
        peak_rank=max(peak_rank,chi)
        peak_aug=max(peak_aug,aug)
        last_rank=chi
        good=info.converged && !info.rank_capped && isfinite(root_res) &&
             (ref_factor===nothing || (isfinite(err) && err<=2e-2))
        status=good ? "ok" : "unmet"
        csv((mode,initial,bc,"uniform_S",bits,k,k*H,g.dx,H,initial_rank,chi,
            aug,info.frame_width[1],info.frame_width[2],root_res,info.converged,
            info.rank_capped,info.discarded,info.miss_input,info.frame_defect,
            err,info.t_root,info.t_step,info.basis_fallback,status))
        println("STEP k=$k t=$(round(k*H,digits=2)) retained=$chi augmented=$aug err=$err residual=$root_res converged=$(info.converged) capped=$(info.rank_capped) wall=$(round(time()-wall_start,digits=1))")
        flush(stdout)
        if !good
            stop_reason="failed_tol_step_$k"
            break
        end
    catch exc
        stop_reason="exception_step_$k"
        println(stderr,"FAIL k=$k exception=$(sprint(showerror,exc,catch_backtrace()))")
        csv((mode,initial,bc,"uniform_S",bits,k,k*H,g.dx,H,initial_rank,
             maximum(bond_dims(psi);init=1),-1,-1,-1,NaN,false,true,
             NaN,NaN,NaN,NaN,NaN,NaN,-1,"exception"))
        break
    end
end
end
run_benchmark!()
finish=stop_reason==="completed"
contracted=last_rank<initial_rank
println("RESULT mode=$mode bits=$bits form=$initial boundary=$bc result=$stop_reason initial_rank=$initial_rank final_rank=$last_rank peak_rank=$peak_rank peak_aug=$peak_aug contracted=$contracted elapsed=$(round(time()-wall_start,digits=1))")
println("RESULT output=$outfile")
# Do not silently accept a failed solve, but do not force monotonic rank contraction
# on initial data that is already very low rank (e.g. affine or a smooth Gaussian).
finish || exit(2)
