# Heisenberg-cylinder RSVD x Krylov benchmark driver for BondUpdateGalerkin.
# IMPORTANT: this driver is PREPARED but NOT EXECUTED against Julia in this environment.
# Methods A/B are written as NOT_IMPLEMENTED until they have real core implementations;
# they must never be silently aliased to existing Lanczos/Arnoldi.
#
# julia --project heisenberg_cylinder_factorial.jl RESULTS_DIR U1 2 4 0.75 smoke
# julia --project heisenberg_cylinder_factorial.jl RESULTS_DIR SU2 3 4 0.75 full
#
# The full factorial is 6 RSVD x 4 Krylov policies x 2 recurrences x 2 symmetries
# = 96 logical cells. Fixed/adaptive local Krylov already run on both recurrences;
# A and B require implementation in the core before their cells can be measured.

using BondUpdateGalerkin.BUG
using BondUpdateGalerkin.QMB
using BondUpdateGalerkin.Baselines
using Telum
using LinearAlgebra
using Random
using Statistics
using Printf

const OUT = length(ARGS) >= 1 ? ARGS[1] : error("RESULTS_DIR required")
const SYM = length(ARGS) >= 2 ? Symbol(ARGS[2]) : error("U1 or SU2 required")
const LX = length(ARGS) >= 3 ? parse(Int, ARGS[3]) : 2
const LY = length(ARGS) >= 4 ? parse(Int, ARGS[4]) : 4
const TFINAL = length(ARGS) >= 5 ? parse(Float64, ARGS[5]) : 0.75
const SUITE = length(ARGS) >= 6 ? Symbol(ARGS[6]) : :smoke
SYM in (:U1, :SU2) || error("symmetry must be U1 or SU2")
SUITE in (:smoke, :full) || error("suite must be smoke or full")
LY >= 3 || error("Use LY >= 3 so cylinder periodic bonds are unambiguous")
iszero((LX * LY) % 2) || error("An even number of sites is required for the singlet-dimer start")

# MPS order winds back and forth around the circumference, with dimers on consecutive sites.
function cylinder_matrix(lx, ly)
    site_order = [(x, y) for x in 1:lx for y in (isodd(x) ? (1:ly) : (ly:-1:1))]
    position = Dict(s => i for (i, s) in enumerate(site_order))
    pairs = Set{Tuple{Int, Int}}()
    for x in 1:lx, y in 1:ly
        a = position[x, y]
        if x < lx
            b = position[x+1, y]
            push!(pairs, (min(a,b), max(a,b)))
        end
        b = position[x, mod1(y+1, ly)]
        push!(pairs, (min(a,b), max(a,b)))
    end
    jmat = zeros(Float64, lx * ly, lx * ly)
    for (i,j) in pairs
        jmat[i,j] = 1.0
    end
    for i in 1:2:(lx * ly)
        (i,i+1) in pairs || error("consecutive singlet ($i,$(i+1)) is not a cylinder bond")
    end
    return jmat
end

# U(1) representation of exactly the SAME physical nearest-neighbour singlets
# that QMB.dimer_state represents under SU(2):
# (|up,down> - |down,up>)/sqrt(2) per consecutive pair.
# Telum's U(1) fusion recursions are the same as QMB/Tooling/initial_states.jl.
function u1_singlet_dimers(n)
    symmetry_mode() === :U1 || error("u1_singlet_dimers requires U1")
    iseven(n) || error("even n required")
    q = local_space(:U1)
    prev = (getvac(q.I, ("L,0", "L,1")), 2)
    tensors = Any[]
    for i in 1:n
        f = TLArray(getIdentity(prev, (q.I, 1); itag="L,$(i+1)"))
        f = TLArray(setitag(f, 2, "S,$i"))
        a = if isodd(i)
            up = TLArray(getsub(f, 2, s -> s == SECTOR_UP ? Colon() : nothing;
                                preserve_space=true))
            dn = TLArray(getsub(f, 2, s -> s == SECTOR_DOWN ? Colon() : nothing;
                                preserve_space=true))
            TLArray((up - dn) / sqrt(2.0))
        else
            TLArray(getsub(f, 3, s -> s == ((0,),) ? Colon() : nothing))
        end
        push!(tensors, TLArray(a * (1.0 + 0.0im)))
        prev = (tensors[end], 3)
    end
    return SymMPS(tensors, n)
end

function make_model(lx, ly)
    n = lx * ly
    jmat = cylinder_matrix(lx, ly)
    terms = SYM === :SU2 ? heisenberg_su2_chain(n; J=1.0) :
                            xxz_chain(n; J=1.0, delta=1.0)
    h = pair_mpo(n, jmat, terms)
    psi = SYM === :SU2 ? dimer_state(n) : u1_singlet_dimers(n)
    nrm = real(overlap(psi, psi))
    energy = real(mpo_energy(psi, h)) / nrm
    expected = -0.375 * n
    @printf("initial |psi|^2=%.14g, energy=%.14g, expected=%.14g\n",nrm,energy,expected)
    abs(nrm - 1.0) < 1e-9 || error("Initial state not normalized: $nrm")
    abs(energy - expected) < 1e-8 ||
        error("Cylinder MPO/U1-dimer initialization invalid: $energy != $expected")
    return h, psi
end

phase_distance(a, b) = sqrt(max(0.0, 2.0 - 2.0 * min(1.0,
    abs(overlap(a,b)) / sqrt(real(overlap(a,a)) * real(overlap(b,b))))))

function tdvp_reference(psi0, h, numsteps)
    psi = copy(psi0)
    tau = ComplexF64(-im * TFINAL / numsteps)
    for _ in 1:numsteps
        tdvp2_step!(psi, h, tau; maxdim=512, weight_tol=1e-13,
                    maxiter=30, tol=1e-14, conv_tol=1e-13)
    end
    return psi
end

function physical_stats(psi, psi0, psi_ref, h)
    nrm = real(overlap(psi,psi))
    return (
        error=phase_distance(psi, psi_ref),
        energy=real(mpo_energy(psi,h))/nrm,
        loschmidt=abs(overlap(psi0,psi))^2/(real(overlap(psi0,psi0))*nrm),
        norm_drift=abs(nrm-1.0),
        state_chi=maximum(state_bond_dims(psi); init=1),
        multiplet_chi=maximum(bond_dims(psi); init=1)
    )
end

function policy(name, backend, fixed_depth)
    kw = (krylov_mode=backend, krylov_fixed_depth=fixed_depth)
    if name === :fixed
        return DDQRExpansion(;kw...,mode=:fixed,fixed_width=8,fixed_rank=4)
    elseif name === :sequential
        return DDQRExpansion(;kw...,mode=:adaptive,batch_blocks=1,holdout_width=4)
    elseif name === :batched
        return DDQRExpansion(;kw...,mode=:adaptive,batch_blocks=2,holdout_width=4)
    elseif name === :rank_adaptive
        return DDQRExpansion(;kw...,mode=:rank_adaptive,rank_initial_width=4,holdout_width=4)
    elseif name === :rank_tail
        return DDQRExpansion(;kw...,mode=:rank_tail,fixed_width=8,tail_factor=0.10)
    elseif name === :rank_fast
        return DDQRExpansion(;kw...,mode=:rank_fast,rank_initial_width=4,holdout_width=2)
    end
    error("unknown RSVD $name")
end

const RSVD = [:fixed,:sequential,:batched,:rank_adaptive,:rank_tail,:rank_fast]
const KPOLICIES = [:fixed_depth,:adaptive,:interleaved_A,:post_krylov_B]
const BACKENDS = [:lanczos,:arnoldi]
# Full scan allows matched-accuracy selection over time-step/rank/tolerance choices.
const SCAN_FULL = [(0.05, 1e-3, 64), (0.025, 1e-4, 128), (0.0125, 1e-5, 256)]
const SCAN_SMOKE = [(0.05, 1e-4, 64)]
const SCAN = SUITE === :full ? SCAN_FULL : SCAN_SMOKE
const REPEATS = SUITE === :full ? 3 : 1

function attempt(psi0,h,psi_ref, method, rsvd, rec, kpolicy, dt, tb, maxdim)
    ns = round(Int, TFINAL/dt)
    abs(ns*dt - TFINAL) < 1e-10 || error("Time grid does not divide TFINAL")
    tr = min(1e-10, tb * 0.1)
    te = 1e-12
    function evolve()
        psi = copy(psi0)
        rng = MersenneTwister(0x05EED)
        h1, h0, stream, qr, t_cbe, t_kry, peak = 0, 0, 0, 0, 0.0, 0.0, 0
        failed = false
        unmet = Set{String}()
        for _ in 1:ns
            info = if method === :tdvp2
                tdvp2_step!(psi,h,ComplexF64(-im*dt);
                    maxdim=maxdim,weight_tol=tr,maxiter=30,tol=1e-14,conv_tol=te)
            elseif method === :bug_default
                cbe_bug_step!(psi,h,ComplexF64(-im*dt);
                    order=2,maxdim=maxdim,tol_basis=tb,tol_trunc=tr,tol_expv=te,
                    rng=rng)
            else
                depth = kpolicy === :fixed_depth ? 8 : 0
                p = policy(rsvd,rec,depth)
                cbe_bug_step!(psi,h,ComplexF64(-im*dt);
                    order=2,maxdim=maxdim,tol_basis=tb,tol_trunc=tr,tol_expv=te,
                    expansion=p,closure_checks=true,rng=rng)
            end
            if method === :tdvp2
                h1 += info.krylov_dims
                peak=max(peak,info.peak_elements)
            else
                ctr=info.record.counters
                h1 += ctr.n_one_site
                h0 += ctr.n_root
                stream += ctr.n_stream_cols
                qr += ctr.n_qr
                t_cbe += info.t_cbe
                t_kry += info.t_kry
                peak = max(peak, info.peak_elements)
                if budget_status(info.ledger) !== :met
                    failed = true
                    union!(unmet,string.(info.ledger.unmet))
                end
            end
        end
        return (physics=physical_stats(psi,psi0,psi_ref,h),h1=h1,h0=h0,
                stream=stream,qr=qr,t_cbe=t_cbe,t_kry=t_kry,peak=peak,
                failed=failed,unmet=join(sort(collect(unmet)),"+"))
    end
    # Warm up exactly this code path before measuring. This is excluded from timings.
    evolve()
    secs = Float64[]
    last = nothing
    for _ in 1:REPEATS
        GC.gc()
        t0 = time_ns()
        last = evolve()
        push!(secs,(time_ns()-t0)/1e9)
    end
    p=last.physics
    return (median(secs),p,last)
end

mkpath(OUT)
set_symmetry!(SYM)
h, psi0 = make_model(LX, LY)
# The independent reference is high-accuracy TDVP2, NOT any of the RSVD policies.
# Convergence is measured by a step-halving comparison and a separate ED Loschmidt check.
ref1 = tdvp_reference(psi0,h,60)
ref2 = tdvp_reference(psi0,h,120)
ref_error = phase_distance(ref1,ref2)
losch_ref = physical_stats(ref2,psi0,ref2,h).loschmidt
const LOSCH_ED = Dict((2,4)=>0.17817819561842602,
                      (3,4)=>0.0444712959655938,
                      (4,4)=>0.011070426101091057)
losch_diff = isapprox(TFINAL, 0.75; atol=1e-12) ?
    abs(losch_ref - get(LOSCH_ED,(LX,LY),losch_ref)) : NaN
@printf("TDVP2 reference step-halving disagreement=%.3e; ED Loschmidt discrepancy=%.3e\n",
        ref_error,losch_diff)
open(joinpath(OUT,"reference_$(SYM)_$(LX)x$(LY).txt"),"w") do io
    println(io,"reference_tdvp2_step_halving_l2=$ref_error")
    println(io,"reference_loschmidt=$losch_ref")
    println(io,"ed_loschmidt_discrepancy=$losch_diff")
    println(io,"symmetry=$SYM")
    println(io,"geometry=$(LX)x$(LY)")
    println(io,"julia=$(VERSION) threads=$(Threads.nthreads()) cpu=$(Sys.CPU_NAME)")
end

filename = joinpath(OUT,"factorial_$(SYM)_$(LX)x$(LY).csv")
open(filename,"w") do io
    println(io,"symmetry,geometry,method,rsvd,backend,krylov_policy,dt,tol_basis,maxdim,status,phase_l2_error,energy,loschmidt,norm_drift,state_chi,multiplet_chi,median_s,h1,hroot,stream_cols,qr_calls,t_cbe,t_kry,peak_elements,unmet")
    function write_result(method,rsvd,back,kpol,dt,tb,md,status,vals)
        ident=string.( (SYM,"$(LX)x$(LY)",method,rsvd,back,kpol,dt,tb,md,status) )
        payload = vals === nothing ? fill("",15) : begin
            sec,p,w=vals
            string.((p.error,p.energy,p.loschmidt,p.norm_drift,p.state_chi,
                     p.multiplet_chi,sec,w.h1,w.h0,w.stream,w.qr,w.t_cbe,w.t_kry,
                     w.peak,w.unmet))
        end
        println(io,join(vcat(ident,payload),","))
        flush(io)
    end
    for (dt,tb,md) in SCAN
        for method in (:bug_default,:tdvp2)
            try
                vals=attempt(psi0,h,ref2,method,:none,:none,:none,dt,tb,md)
                status=vals[3].failed ? "budget_unmet" : "executed"
                write_result(method,:none,:none,:none,dt,tb,md,status,vals)
            catch e
                @warn "Baseline failed" method exception=(e,catch_backtrace())
                write_result(method,:none,:none,:none,dt,tb,md,"runtime_error",nothing)
            end
        end
        arms = SUITE === :smoke ? [(:rank_tail,:lanczos,:adaptive)] :
               [(r,b,k) for r in RSVD for b in BACKENDS for k in KPOLICIES]
        for (r,b,k) in arms
            if k in (:interleaved_A,:post_krylov_B)
                # These are not supplied by KRYLOV_RECURRENCES or half_sweeps.jl.
                write_result(:bug_factorial,r,b,k,dt,tb,md,"NOT_IMPLEMENTED",nothing)
                continue
            end
            try
                vals=attempt(psi0,h,ref2,:bug_factorial,r,b,k,dt,tb,md)
                status=vals[3].failed ? "budget_unmet" : "executed"
                write_result(:bug_factorial,r,b,k,dt,tb,md,status,vals)
            catch e
                @warn "Policy failed" r b k exception=(e,catch_backtrace())
                write_result(:bug_factorial,r,b,k,dt,tb,md,"runtime_error",nothing)
            end
        end
    end
end
println("Results: ",filename)