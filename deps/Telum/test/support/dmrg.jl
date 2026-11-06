using LinearAlgebra
using Telum

include(joinpath(@__DIR__, "..", "..", "benchmark", "two_site_dmrg.jl"))
using .BenchmarkDMRG

# DMRG validation support for integration/stress tests.
# These helpers intentionally live outside src/: they exercise public Telum
# operations together without becoming part of the package API.

# MPO builders used by DMRG tests and contraction/SVD stress fixtures.
function build_MPO(qss::Matrix{<:AbstractTLArray}, N::Int)
    println("Building MPO...")
    t = oplus(qss, (3, 4))
    zq = zero_qlabels(t)
    MPO = Vector{TLArray{Float64, 4}}(undef, N)
    for i in 1:N
        tags = ("S,$i", "S,$i", i > 1 ? "OB,$(i - 1)" : "OLeft",
                i < N ? "OB,$i" : "ORight")

        if i == 1
            MPO[i] = TLArray(getsub(t, 3, x -> x == zq ? -1 : nothing), tags)
        elseif i == N
            MPO[i] = TLArray(getsub(t, 4, x -> x == zq ? 1 : nothing), tags)
        else
            MPO[i] = TLArray(t, tags)
        end
    end
    return MPO
end

function MajumdarGhoshMPO(J, N)
    option = SpinOptions(SU{2}, 1 // 2)
    q = getLocalSpace(option, ("site", "site", "op"))

    qss = Matrix{AbstractTLArray{Float64, 4, 1, 5}}(undef, 4, 4)

    i4d = addSingleton(q.I; nlegs=2, itag=("left", "right"), dir=('+', '-'))
    s4d = addSingleton(q.S, 3; itag="left", dir='+')
    s4d = setitag(s4d, 4, "right")
    sc4d = addSingleton(q.S'; itag="right", dir='-')
    sc4d = permutedims(setitag(sc4d, 3, "left"), (2, 1, 3, 4))
    opid = TLArray(getIdentity((q.S, 3)), ("left", "right"))

    qss[1, 1] = qss[4, 4] = i4d
    qss[2, 1] = sc4d
    qss[3, 2] = q.I ⊗ opid
    qss[4, 2] = J * s4d
    qss[4, 3] = J / 2 * s4d

    return build_MPO(qss, N)
end

function HubbardMPO(U, μ, t, N,
                    opt::FermionSOptions=FermionSOptions(1, :U1, :SU2, nothing))
    q = getLocalSpace(opt, ("site", "site", "op"))
    nloc = lock(q.F', 2) * q.F

    qss = Matrix{AbstractTLArray{Float64, 4}}(undef, 4, 4)
    i4d = addSingleton(q.I; nlegs=2, itag=("left", "right"), dir=('+', '-'))
    ZF_flip = setitag(legflip(lock(q.Z, 1) * q.F, 3), 3, "left")
    FcZ = permutedims(q.F' * lock(q.Z, 2), (1, 3, 2))

    f4d = addSingleton(q.F, 3; itag="left", dir='+')
    f4d = setitag(f4d, 4, "right")
    fc_flip = addSingleton(legflip(q.F', 3), 3; itag="left", dir='+')
    fc_flip = setitag(permutedims(fc_flip, (2, 1, 3, 4)), 4, "right")

    qss[1, 1] = qss[4, 4] = i4d
    qss[2, 1] = setitag(addSingleton(FcZ; itag="right", dir='-'), 3, "left")
    qss[3, 1] = addSingleton(ZF_flip; itag="right", dir='-')
    qss[4, 1] = addSingleton(lock(nloc - 1, 1) * nloc * U / 2 - μ * nloc;
        nlegs=2, itag=("left", "right"), dir=('+', '-'))
    qss[4, 2] = -t * f4d
    qss[4, 3] = -t * fc_flip

    return build_MPO(qss, N)
end

function XYMPO(J, N)
    option = SpinOptions(:U1, 1)
    q = getLocalSpace(option, ("site", "site", "op"))

    i4d = addSingleton(q.I, (3, 4); itag=("left", "right"), dir=('+', '-'))

    sm4d = addSingleton(q.Sm, 3; itag="left", dir='+')
    sm4d = setitag(sm4d, 4, "right")
    sp4d = addSingleton(q.Sp, 3; itag="left", dir='+')
    sp4d = setitag(sp4d, 4, "right")

    smc4d = addSingleton(q.Sm', 4; itag="right", dir='-')
    smc4d = permutedims(setitag(smc4d, 3, "left"), (2, 1, 3, 4))
    spc4d = addSingleton(q.Sp', 4; itag="right", dir='-')
    spc4d = permutedims(setitag(spc4d, 3, "left"), (2, 1, 3, 4))

    qss = Matrix{AbstractTLArray{Float64, 4, 1, 5}}(undef, 4, 4)

    qss[1, 1] = qss[4, 4] = i4d
    qss[2, 1] = spc4d
    qss[3, 1] = smc4d
    qss[4, 2] = J * sp4d
    qss[4, 3] = J * sm4d
    return build_MPO(qss, N)
end

function XXZMPO(δ, h, N)
    option = SpinOptions(:U1, 1)
    q = getLocalSpace(option, ("site", "site", "op"))

    i4d = addSingleton(q.I, (3, 4); itag=("left", "right"), dir=('+', '-'))

    sm4d = addSingleton(q.Sm, 3; itag="left", dir='+')
    sm4d = setitag(sm4d, 4, "right")
    sz4d = addSingleton(q.Sz, (3, 4); itag=("left", "right"), dir=('+', '-'))
    sp4d = addSingleton(q.Sp, 3; itag="left", dir='+')
    sp4d = setitag(sp4d, 4, "right")

    smc4d = addSingleton(q.Sm', 4; itag="right", dir='-')
    smc4d = permutedims(setitag(smc4d, 3, "left"), (2, 1, 3, 4))
    spc4d = addSingleton(q.Sp', 4; itag="right", dir='-')
    spc4d = permutedims(setitag(spc4d, 3, "left"), (2, 1, 3, 4))

    qss = Matrix{AbstractTLArray{Float64, 4, 1, 5}}(undef, 5, 5)

    qss[1, 1] = qss[5, 5] = i4d
    qss[2, 1] = spc4d
    qss[3, 1] = smc4d
    qss[4, 1] = sz4d
    qss[5, 1] = h * sz4d
    qss[5, 2] = sp4d
    qss[5, 3] = sm4d
    qss[5, 4] = δ * sz4d
    return build_MPO(qss, N)
end

function central_bond_rmt_dimensions(MPS::AbstractVector{<:TLArray}; bond::Int=length(MPS) ÷ 2)
    1 <= bond < length(MPS) || throw(ArgumentError(
        "central bond index must satisfy 1 <= bond < $(length(MPS)), got $bond"))
    leg = findleg(MPS[bond]; itag="SB,$bond")
    isnothing(leg) && throw(ArgumentError("MPS[$bond] has no leg tagged SB,$bond"))
    bond_space = Telum.spaces(MPS[bond])[leg]
    qlabels = first.(bond_space)
    dimensions = last.(bond_space)
    perm = sortperm(dimensions; rev=true)
    return dimensions[perm], qlabels[perm]
end

# Structured test entry point. Each result is a NamedTuple so tests can assert
# energies while retaining central-bond and cost diagnostics for debugging.
function run_dmrg_validation(MPO=HubbardMPO(4.0, 1.5, 1.0, 40);
                             time_blocks=false, nkeeps=[20, 50],
                             nsweep=4)
    read_reset_costs!()
    return map(nkeeps) do Nkeep_DMRG
        println("Nkeep_DMRG = $Nkeep_DMRG")
        set_accumul_costs!(true)
        Nkeep_init = max(50, Nkeep_DMRG)
        MPS, energy = try
            @time do_dmrg_result(MPO, Nkeep_init, Nkeep_DMRG, nsweep; time_blocks)
        finally
            set_accumul_costs!(false)
        end
        dimensions, qlabels = central_bond_rmt_dimensions(MPS)
        contraction_cost, svd_cost = read_reset_costs!()
        GC.gc()
        return (; nkeep=Nkeep_DMRG, energy, dimensions, qlabels,
                contraction_cost, svd_cost)
    end
end

function run(MPO=HubbardMPO(4.0, 1.5, 1.0, 40); time_blocks=false, nkeeps=[20, 50])
    results = run_dmrg_validation(MPO; time_blocks, nkeeps)
    return getproperty.(results, :dimensions),
           getproperty.(results, :qlabels),
           getproperty.(results, :contraction_cost),
           getproperty.(results, :svd_cost)
end
