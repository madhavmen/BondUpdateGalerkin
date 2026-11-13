# d=3/4, L=16, short/long XX benchmark without exponential-size ED.
# The independent d=3/4 L=6 exact-sector ED is in xx_T5_matched_work.jl.
# Here compare wall time, H applications, sampling, budget flags and U(1) physics
# across matched initial states. Magnetisation differences are diagnostics,
# NOT a substitute for exact-state error or a converged large-system reference.
using BondUpdateGalerkin.BUG, BondUpdateGalerkin.QMB
using LinearAlgebra, Random, Telum
set_symmetry!(:U1)

L = parse(Int, get(ENV, "BUG_DD_AB_L", "16"))
twoS = parse(Int, get(ENV, "BUG_DD_AB_TWOS", "2"))
twoS in (2, 3) || error("This benchmark is for d=3 and d=4")
d = twoS + 1
range = Symbol(get(ENV, "BUG_DD_AB_RANGE", "short"))
range in (:short, :long) || error("range must be short or long")
nsteps = parse(Int, get(ENV, "BUG_DD_AB_STEPS", "5"))
dt = parse(Float64, get(ENV, "BUG_DD_AB_DT", "0.1"))
seed = parse(Int, get(ENV, "BUG_DD_AB_SEED", "0"))
tol = parse(Float64, get(ENV, "BUG_DD_AB_TOL", "1e-3"))
maxdim = parse(Int, get(ENV, "BUG_DD_AB_MAXDIM", "96"))
outdir = isempty(ARGS) ? "results/dd-ab-d34-l16" : ARGS[1]
mkpath(outdir)

J = zeros(Float64, L, L)
for i=1:L-1, j=i+1:L
    J[i, j] = range == :short ? (j == i+1 ? 1.0 : 0.0) : (j-i)^(-3.0)
end
q = spin_space(twoS)
mpo = pair_mpo(
    L, J, [PairVertex(q.Sp, TLArray(q.Sp'), 1.0),
           PairVertex(q.Sm, TLArray(q.Sm'), 1.0)]; Iloc=q.I,
)

modes = (:none, :interleaved, :post)
open(joinpath(outdir, "metrics.csv"), "w") do io
    println(io, "d,L,range,mode,step,t,wall_s,krylov_h1,closure_h1,stream_calls,stream_cols,dd_probes,dd_holdout,root_h,closure_sweeps,D_stage,D_final,root_left,root_right,norm_error,sz_total_error,budget_met,unmet")
    open(joinpath(outdir, "profiles.csv"), "w") do po
        println(po, "d,L,range,mode,step,t,site,sz")
        for mode in modes
            psi = spin_domain_wall_state(L, twoS)
            rng = MersenneTwister(seed+71)
            policy = DDQRExpansion(mode=:fixed, fixed_width=8,
                                   fixed_rank=4, krylov_mode=:lanczos)
            for k=1:nsteps
                before = bond_dims(psi)
                t0 = time_ns()
                info = cbe_bug_step!(
                    psi, mpo, ComplexF64(-im * dt);
                    order=2, closure_checks=true,
                    tol_basis=tol, tol_trunc=tol * 1e-3, tol_root=1e-10,
                    maxdim=maxdim,
                    work_ceiling_rule=:bond, bond_ceiling_factor=1.2,
                    expansion=policy, extra_dd_mode=mode,
                    extra_dd_width=8, extra_dd_holdout=4,
                    extra_dd_max_checks=2,
                    rng=rng,
                )
                wall=(time_ns()-t0)/1e9
                caps = bond_work_ceilings(before,d,maxdim;factor=1.2)
                all(info.expanded .<= caps) || error("left augmented cap exceeded")
                info.root_right <= caps[info.root] || error("right root cap exceeded")
                all(bond_dims(psi) .<= caps) || error("final bond cap exceeded")
                nrm2 = real(overlap(psi,psi))
                abs(nrm2) > 0 || error("zero norm")
                profile = spin_sz_profile(psi,twoS) ./ nrm2
                phys_total = abs(sum(profile))
                lg=info.ledger
                c=info.record.counters
                println(io, join((d,L,range,mode,k,k*dt,wall,
                  lg.n_h1,lg.n_h1_closure,c.n_stream,c.n_stream_cols,
                  lg.n_probes,lg.n_holdout,c.n_root,lg.n_closure_sweeps,
                  info.record.D_stage,maximum(bond_dims(psi)),
                  info.expanded[info.root],info.root_right,
                  abs(sqrt(nrm2)-1.0),phys_total,lg.tolerance_met,
                  join(string.(lg.unmet),";")),","))
                flush(io)
                for j=1:L
                    println(po,join((d,L,range,mode,k,k*dt,j,profile[j]),","))
                end
                flush(po)
                println("DDAB_LARGE d=$d range=$range mode=$mode k=$k t=$(round(wall,digits=3)) D=$(maximum(bond_dims(psi))) met=$(lg.tolerance_met)")
            end
        end
    end
end
println("DD_AB_HIGHER_D_LARGE_COMPLETE")
