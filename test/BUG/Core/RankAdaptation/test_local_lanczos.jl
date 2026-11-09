# Hermitian Krylov recurrence: genuine 3-term Lanczos local frame enrichment.
# Standalone small test; no MPS/MPO, no hidden adaptive thresholds.
using Test, LinearAlgebra, Random, BondUpdateGalerkin.BUG
import BondUpdateGalerkin.BUG: _lanczos_chain, _arnoldi_chain
rng = MersenneTwister(45912)
for n in (24,40), t in (0.1,1.0,4.0)
    A = randn(rng,ComplexF64,n,n)
    H = Hermitian((A + A')/2)
    seed = randn(rng,ComplexF64,n)
    seed /= norm(seed)
    tau = ComplexF64(-im*t)
    target = 1e-9
    score = (_,_,_)->true
    ar = _arnoldi_chain(seed,x->H*x,tau,target,score;maxdepth=n)
    la = _lanczos_chain(seed,x->H*x,tau,target,score;maxdepth=n)
    ref = exp(tau*Matrix(H))*seed
    # Projection on the Krylov basis gives an independent necessary error check.
    QA = hcat(ar.basis...); QL=hcat(la.basis...)
    errA = norm(ref - QA*(QA' * ref))
    errL = norm(ref - QL*(QL' * ref))
    @test ar.closed && la.closed
    @test errA <= 2e-7
    @test errL <= 2e-7
    @test abs(la.n_h1 - ar.n_h1) <= 4
    @test all(isfinite,la.weights)
    @test norm(QL' * QL - I) <= 1e-7
end
# A fixed depth that does not meet the tolerance MUST report it as uncapped/unmet.
let n=20,H=Matrix{Float64}(SymTridiagonal(zeros(n),ones(n-1))),
    seed=ComplexF64[1;zeros(n-1)], tau=ComplexF64(-2im)
    r=_lanczos_chain(seed,x->H*x,tau,1e-12,(_,_,_)->true;
                     maxdepth=n,fixed_depth=4)
    @test r.n_h1==4
    @test r.capped && !r.closed
end
println("LOCAL-LANCZOS-OK")
