# Local Krylov-directed extra-DD variants, physical XX domain-wall evolution.
# Independent free-fermion exact density reference, avoiding a dense 12,870^2 ED matrix.
# RUN: BUG_DD_AB_L=16 BUG_DD_AB_STEPS=50 julia --project benchmarks/XXT5/xx_L16_krylov_dd_ab.jl results/krylov-dd-ab
using BondUpdateGalerkin.BUG, BondUpdateGalerkin.QMB
using LinearAlgebra, Random, Telum

set_symmetry!(:U1)
L = parse(Int, get(ENV, "BUG_DD_AB_L", "16"))
iseven(L) || error("Domain-wall chain length must be even")
nsteps = parse(Int, get(ENV, "BUG_DD_AB_STEPS", "50"))
nseeds = parse(Int, get(ENV, "BUG_DD_AB_SEEDS", "3"))
dt = parse(Float64, get(ENV, "BUG_DD_AB_DT", "0.1"))
maxdim = parse(Int, get(ENV, "BUG_DD_AB_MAXDIM", "128"))
tols = parse.(Float64, split(get(ENV, "BUG_DD_AB_TOLS", "1e-3"), ','))
outdir = isempty(ARGS) ? "results/krylov-dd-ab" : ARGS[1]
mkpath(outdir)

# Open-boundary H = 1/2 sum_j (S+_j S-_j+1 + S-_j S+_j+1).
J = zeros(Float64, L, L)
for b = 1:L-1
    J[b, b+1] = 1.0
end
sp = spin_space(1)
mpo = pair_mpo(
    L, J, [PairVertex(sp.Sp, TLArray(sp.Sp'), 1.0),
           PairVertex(sp.Sm, TLArray(sp.Sm'), 1.0)]; Iloc = sp.I,
)
# Jordan-Wigner exact reference: an N=L/2 free-fermion Slater state.
h = zeros(Float64, L, L)
for b=1:L-1
    h[b,b+1] = h[b+1,b] = 0.5
end
H1 = eigen(Symmetric(h))
propagator(t) = H1.vectors * Diagonal(exp.(-im*t .* H1.values)) * H1.vectors'
function reference_sz(t)
    orbitals = propagator(t)[:, 1:(L÷2)]
    return vec(sum(abs2.(orbitals); dims = 2)) .- 0.5
end
function slater_vector(t)
    orbitals = propagator(t)[:, 1:(L÷2)]
    result = zeros(ComplexF64, 2^L)
    for index = 0:(2^L - 1)
        occupied = [j for j=1:L if ((index >> (L-j)) & 1) == 0]
        length(occupied) == L÷2 || continue
        result[index+1] = det(orbitals[occupied, :])
    end
    return result
end

methods = (:none, :interleaved, :post)
outfile = open(joinpath(outdir, "xx16_ab.csv"), "w")
println(outfile, join((
    "method","seed","step","t","basis_tol","sz_err","state_err",
    "time_s","krylov_h1","closure_h1","stream_calls","stream_columns",
    "dd_discovery","dd_holdouts","root_h","closure_sweeps","D_stage",
    "D_final","root_left","root_right","met","unmet",
), ","))
flush(outfile)
for tol in tols, seed in 0:nseeds-1, method in methods
    psi = spin_domain_wall_state(L, 1)
    policy = DDQRExpansion(
        mode = :fixed, fixed_width = 8, fixed_rank = 4,
        krylov_mode = :lanczos,
    )
    rng = MersenneTwister(seed + 71)
    for k=1:nsteps
        tic = time_ns()
        info = cbe_bug_step!(
            psi, mpo, ComplexF64(-im * dt);
            order = 2,
            closure_checks = true,
            tol_basis = tol,
            tol_trunc = tol * 1e-3,
            tol_root = 1e-10,
            maxdim = maxdim,
            work_ceiling_rule = :bond,
            bond_ceiling_factor = 1.2,
            expansion = policy,
            extra_dd_mode = method,
            extra_dd_width = 8,
            extra_dd_holdout = 4,
            extra_dd_max_checks = 2,
            rng = rng,
        )
        tsec = (time_ns()-tic)/1e9
        nrm = norm(psi[psi.center])^2
        computed = spin_sz_profile(psi, 1) ./ nrm
        szerr = maximum(abs.(computed .- reference_sz(k*dt)))
        # Exact global-state fidelity is checked for the small test. The L16
        # production run reports exact free-fermion physical densities without
        # expensive per-sector product bra contractions on every step.
        stateerr = NaN
        if L <= 8 && (k == 1 || k == nsteps)
            approx = dense_state(psi)
            ref = slater_vector(k * dt)
            stateerr = sqrt(max(0, 1-abs2(dot(ref, approx)) /
                                      (real(dot(ref,ref))*real(dot(approx,approx)))))
        end
        lg = info.ledger
        ctr = info.record.counters
        println(outfile, join((
            method, seed, k, k*dt, tol, szerr, stateerr,
            tsec, lg.n_h1, lg.n_h1_closure,
            ctr.n_stream, ctr.n_stream_cols,
            lg.n_probes, lg.n_holdout, ctr.n_root,
            lg.n_closure_sweeps, info.record.D_stage,
            maximum(bond_dims(psi)), info.expanded[info.root],
            info.root_right, lg.tolerance_met, join(string.(lg.unmet), ";"),
        ), ","))
        flush(outfile)
        println("AB method=$(method) L=$L seed=$seed k=$k sz_error=$(szerr) " *
                "time=$(round(tsec,digits=3)) met=$(lg.tolerance_met)")
    end
end
close(outfile)
println("XX_KRYLOV_DD_AB_FINISHED")
