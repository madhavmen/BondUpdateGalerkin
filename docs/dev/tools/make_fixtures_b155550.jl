# WP0 frozen numerical fixtures from BUG-Julia b155550 (tag pre-bondupdategalerkin).
#
# Run from a BUG-Julia b155550 checkout (it includes that checkout's tests/common helpers):
#   FX_OUT=<dir> julia -t 2 --project=<BUG-Julia> <this file> [case ...]
# Writes one JSON file per case plus versions.json. Every case seeds the global RNG, so a rerun on
# the same commit and thread count reproduces it; the new package's regression tests compare
# against these files with tolerances set per quantity, never bit-for-bit.

using LinearAlgebra, Printf, Pkg
import JSON, Random
using LurCGT, Telum
using BUGJulia, BUGJulia.BondUpdateBUG, BUGJulia.RSVDCBEBondUpdate

const RC = BUGJulia.RSVDCBEBondUpdate
const OUT = get(ENV, "FX_OUT", "fixtures_b155550")
const COMMON = joinpath(pwd(), "tests", "common")
for f in (
    "adaptive_fixtures",
    "dense_reference",
    "free_fermion",
    "dissipative_fixtures",
    "liouville_fixtures",
)
    include(joinpath(COMMON, f * ".jl"))
end

cplx(v) = Dict("re" => real.(collect(v)), "im" => imag.(collect(v)))
infofield(info, f) = hasproperty(info, f) ? getproperty(info, f) : nothing
status(info) = hasproperty(info, :ledger) ? String(budget_status(info.ledger)) : nothing
sz_profile(p) = real.(magnetisation(copy(p))) ./ real(overlap(p, p))

"Real-time XX chain under U(1): Neel start, exact free-fermion <S^z_j(t)> as the reference."
function case_xx_u1(; order = 1, midpoint = false)
    set_symmetry!(:U1)
    L, dt, n = 10, 0.05, 20
    W = xxz_mpo(L; J = 1.0, delta = 0.0)
    psi = neel_state(L)
    rows = Any[]
    for k = 1:n
        info =
            midpoint ? cbe_bug_midpoint_step!(psi, W, ComplexF64(-im * dt)) :
            cbe_bug_step!(psi, W, ComplexF64(-im * dt); order = order)
        ex = xx_free_fermion_sz(L, k * dt; J = 1.0, occupied = 1:2:L)
        sz = sz_profile(psi)
        push!(
            rows,
            Dict(
                "t" => k * dt,
                "sz" => sz,
                "err_exact" => maximum(abs.(sz .- ex)),
                "bond_dims" => bond_dims(psi),
                "status" => status(info),
                "discarded" => infofield(info, :discarded),
            ),
        )
    end
    return Dict(
        "model" => "XX chain, J=1, U(1)",
        "L" => L,
        "dt" => dt,
        "steps" => n,
        "start" => "neel",
        "order" => order,
        "midpoint" => midpoint,
        "options" => "defaults",
        "trajectory" => rows,
    )
end

"The same flow without symmetry (the storage QTT states use), final state kept densely."
function case_xx_none()
    set_symmetry!(:none)
    clear_fusion_cache!()
    L, dt, n = 8, 0.05, 10
    W = xxz_mpo(L; J = 1.0, delta = 0.0)
    psi = neel_state(L)
    rows = Any[]
    for k = 1:n
        info = cbe_bug_step!(psi, W, ComplexF64(-im * dt); order = 2)
        push!(
            rows,
            Dict(
                "t" => k * dt,
                "bond_dims" => bond_dims(psi),
                "status" => status(info),
                "norm" => sqrt(real(overlap(psi, psi))),
            ),
        )
    end
    v = dense_state(psi)
    ref =
        exp(-im * n * dt * Matrix(dense_heisenberg(L; J = 1.0, delta = 0.0))) *
        dense_state(neel_state(L))
    out = Dict(
        "model" => "XX chain, J=1, no symmetry",
        "L" => L,
        "dt" => dt,
        "steps" => n,
        "start" => "neel",
        "order" => 2,
        "trajectory" => rows,
        "final_state" => cplx(v),
        "err_dense" => norm(v - ref) / norm(ref),
    )
    set_symmetry!(:U1)
    return out
end

"Imaginary time on the Heisenberg chain: energy per site along beta."
function case_heisenberg_ite()
    set_symmetry!(:U1)
    L = 10
    W = xxz_mpo(L; J = 1.0, delta = 1.0)
    psi = neel_state(L)
    e0 = minimum(eigvals(Hermitian(Matrix(dense_heisenberg(L; J = 1.0, delta = 1.0)))))
    rows = Any[]
    beta = 0.0
    for _ = 1:8
        hist = ite_evolve!(psi, W, uniform_beta_steps(0.25, 2); order = 2)
        beta += 0.25
        e = real(mpo_energy(psi, W)) / real(overlap(psi, psi))
        push!(
            rows,
            Dict(
                "beta" => beta,
                "energy" => e,
                "bond_dims" => bond_dims(psi),
                "status" => [String(r.status) for r in hist],
            ),
        )
    end
    return Dict(
        "model" => "Heisenberg chain, J=1, Delta=1, U(1)",
        "L" => L,
        "start" => "neel",
        "order" => 2,
        "dbeta" => 0.125,
        "e_exact_ground" => e0,
        "trajectory" => rows,
    )
end

"MPDO: driven-dissipative XX chain in Liouville space (weak U(1)); trace and positivity checks."
function case_mpdo()
    set_symmetry!(:U1)
    L = 6
    pairs, hs, jumps = _xx_model(L)
    ls = liouville_space([1, -1]; sym = :weak)
    W = lindblad_mpo(ls, L; pairs = pairs, onsite_h = hs, jumps = jumps)
    rho = mpdo_product_state(ls, [isodd(i) ? _NUP : I - _NUP for i = 1:L])
    rows = Any[]
    for k = 1:10
        cbe_bug_step!(
            rho,
            W,
            ComplexF64(0.1);
            order = 2,
            hermitian = false,
            tol_basis = 1e-6,
            tol_trunc = 1e-6,
            tol_expv = 1e-8,
            maxdim = 64,
        )
        c = mpdo_checks(rho, ls)
        push!(
            rows,
            Dict(
                "t" => 0.1k,
                "trace" => real(c.trace),
                "herm" => c.herm,
                "min_eig" => c.min_eig,
                "bond_dims" => bond_dims(rho),
            ),
        )
    end
    return Dict(
        "model" => "XX + dephasing 0.3 + gain 0.5 at site 1 + loss 0.7 at site L, field 0.2 i",
        "L" => L,
        "dt" => 0.1,
        "order" => 2,
        "start" => "Neel product (up on odd sites)",
        "options" => "tol_basis 1e-6, tol_trunc 1e-6, tol_expv 1e-8, maxdim 64",
        "trajectory" => rows,
    )
end

"MCWF: one seeded quantum-jump trajectory of the lossy XX chain from the domain wall."
function case_mcwf()
    set_symmetry!(:U1)
    L = 8
    g = _loss_rates(L)
    _, _, G, _, _ = _lossy_xx(L; gamma = g)
    ch = site_emission_channels(L; gamma = g)
    obs(p) = vcat(real(overlap(p, p)), sz_profile(p))
    tg = collect(0.0:0.25:2.0)
    step_kw = (closure_checks = true, tol_basis = 1e-10, tol_root = 1e-12)
    tr = jump_trajectory(
        domain_wall_state(L),
        G,
        ch,
        tg;
        dt = 0.05,
        seed = 7,
        observe = obs,
        step_kw = step_kw,
    )
    return Dict(
        "model" => "XX, site loss gamma_j = 1 (odd j) / 0.2 (even j)",
        "L" => L,
        "dt" => 0.05,
        "seed" => 7,
        "times" => tr.times,
        "events" => [Dict("t" => e.t, "channel" => e.channel) for e in tr.events],
        "njumps" => tr.njumps,
        "obs" => tr.obs,
        "nsteps" => tr.nsteps,
    )
end

"Two-site TDVP baseline on the XX chain (U(1), Neel)."
function case_tdvp2()
    set_symmetry!(:U1)
    L, dt, n = 10, 0.05, 20
    W = xxz_mpo(L; J = 1.0, delta = 0.0)
    psi = neel_state(L)
    rows = Any[]
    for k = 1:n
        tdvp2_step!(psi, W, ComplexF64(-im * dt); maxdim = 64, trunc_thresh = 1e-12)
        ex = xx_free_fermion_sz(L, k * dt; J = 1.0, occupied = 1:2:L)
        sz = sz_profile(psi)
        push!(
            rows,
            Dict(
                "t" => k * dt,
                "sz" => sz,
                "err_exact" => maximum(abs.(sz .- ex)),
                "bond_dims" => bond_dims(psi),
            ),
        )
    end
    return Dict(
        "model" => "XX chain, J=1, U(1)",
        "L" => L,
        "dt" => dt,
        "steps" => n,
        "options" => "maxdim 64, trunc_thresh 1e-12",
        "trajectory" => rows,
    )
end

"QTT Heston: backward pricing of a softened call, N = 5 bits per axis, BUG order 3."
function case_heston()
    set_symmetry!(:none)
    clear_fusion_cache!()
    N = 5
    p = HestonParams(0.03, 0.0, 1.5, 0.09, 0.5, -0.7)
    g = qtt_grid(N, N; x = (-3.0, 3.0), y = (0.0, 1.0), periodic_x = false, periodic_y = false)
    softplus(z) = z > 30 ? z : log1p(exp(z))
    u0 = qtt_separable(g, x -> 0.02 * softplus((exp(x) - 1.0) / 0.02), v -> 1.0; tol = 1e-12)
    fl = LinearFlow(heston_op(g, p))
    psi = qtt_mps(u0)
    h, n = 0.02, 10
    rows = Any[]
    for k = 0:(n-1)
        r = qtt_bug_step!(psi, fl, k * h, h; order = 3, maxdim = 64, tol_trunc = 1e-10)
        push!(rows, Dict("t" => (k + 1) * h, "chi" => r.chi, "bond_dims" => bond_dims(psi)))
    end
    v = vec(tt_dense(qtt_cores(psi)))
    out = Dict(
        "model" => "Heston backward, r 0.03 q 0 kappa 1.5 theta 0.09 sigma 0.5 rho -0.7",
        "N" => N,
        "h" => h,
        "steps" => n,
        "order" => 3,
        "trajectory" => rows,
        "final_state" => cplx(v),
    )
    set_symmetry!(:U1)
    return out
end

"QTT Gross-Pitaevskii: nonlinear real-time steps and the imaginary-time ground state (N = 3)."
function case_gp()
    set_symmetry!(:none)
    clear_fusion_cache!()
    g = qtt_grid(3, 3; x = (-5.0, 5.0), y = (-5.0, 5.0))
    m = gp_model(g; gnl = 1.0, gamma = 0.1, mu = 0.0, omega = 1.0, rho_tol = 1e-13)
    phi0(x) = pi^(-0.25) * exp(-x^2 / 2)
    phi1(x) = sqrt(2) * x * phi0(x)
    A0 = tt_lincomb([
        (1.0, qtt_separable(g, phi0, phi0)),
        (0.25, qtt_separable(g, phi1, phi0)),
        (0.15im, qtt_separable(g, phi0, phi1)),
    ])
    rows = Any[]
    psi = qtt_mps(A0)
    for k = 0:3
        r = qtt_bug_step!(
            psi,
            GPFlow(m),
            k * 0.1,
            0.1;
            order = 2,
            maxdim = 64,
            tol_basis = 1e-10,
            tol_trunc = 1e-12,
        )
        push!(rows, Dict("t" => (k + 1) * 0.1, "chi" => r.chi, "bond_dims" => bond_dims(psi)))
    end
    v = vec(tt_dense(qtt_cores(psi)))
    lg = gp_ground_state!(qtt_mps(A0), m, uniform_schedule(0.5, 10); order = 2, maxdim = 64)
    out = Dict(
        "model" => "GP, gnl 1, gamma 0.1, mu 0, omega 1, box [-5,5]^2",
        "N" => 3,
        "h" => 0.1,
        "order" => 2,
        "trajectory" => rows,
        "final_state" => cplx(v),
        "ground_state_energies" => [e.E for e in lg if e.accepted],
    )
    set_symmetry!(:U1)
    return out
end

const CASES = [
    "xx_u1_bug_order1" => () -> case_xx_u1(order = 1),
    "xx_u1_bug_order2" => () -> case_xx_u1(order = 2),
    "xx_u1_bug_midpoint" => () -> case_xx_u1(midpoint = true),
    "xx_none_bug_order2" => case_xx_none,
    "heisenberg_ite" => case_heisenberg_ite,
    "mpdo_xx_driven" => case_mpdo,
    "mcwf_xx_loss" => case_mcwf,
    "baseline_tdvp2_xx" => case_tdvp2,
    "qtt_heston_backward" => case_heston,
    "qtt_gp" => case_gp,
]

mkpath(OUT)
deps =
    Dict(d.name => string(d.version) for d in values(Pkg.dependencies()) if d.version !== nothing)
open(joinpath(OUT, "versions.json"), "w") do io
    JSON.print(
        io,
        Dict(
            "bug_julia_commit" => get(ENV, "FX_COMMIT", "unknown"),
            "julia" => string(VERSION),
            "threads" => Threads.nthreads(),
            "Telum" => get(deps, "Telum", "?"),
            "LurCGT" => get(deps, "LurCGT", "?"),
            "cpu" => Sys.CPU_NAME,
        ),
    )
end
want = isempty(ARGS) ? first.(CASES) : ARGS
bad = false
for (name, f) in CASES
    name in want || continue
    Random.seed!(1234)
    t0 = time()
    try
        res = f()
        res["wall_s"] = time() - t0
        open(io -> JSON.print(io, res), joinpath(OUT, name * ".json"), "w")
        @printf("FIXTURE-OK %s %.1f s\n", name, time() - t0)
    catch err
        global bad = true
        println("FIXTURE-FAILED ", name)
        showerror(stdout, err, catch_backtrace())
        println()
    end
    flush(stdout)
end
exit(bad ? 1 : 0)
