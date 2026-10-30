using Test
using BondUpdateGalerkin

# Each file runs in its own module, so helpers of one file cannot leak into another.
function run_testfile(path)
    m = Module(Symbol(basename(path)))
    Core.eval(m, :(include(p) = Base.include($m, p)))
    Base.include(m, joinpath(@__DIR__, path))
    return nothing
end

@testset "BondUpdateGalerkin" begin
    @testset "package loads" begin
        for name in (:BUG, :QMB, :QTTS, :Baselines)
            @test isdefined(BondUpdateGalerkin, name)
        end
    end

    # BUG/Core, primitives first so a failure localises
    run_testfile("BUG/Core/TensorNetworks/test_symmetric_mps.jl")
    run_testfile("BUG/Core/TensorNetworks/test_sectors.jl")
    run_testfile("BUG/Core/TensorNetworks/test_fusion_cache.jl")
    run_testfile("BUG/Core/TensorNetworks/test_mpo.jl")
    run_testfile("BUG/Core/Sweeps/test_frames.jl")
    run_testfile("BUG/Core/Interfaces/test_state_conformance.jl")
    # BUG/TimeEvolution and BUG/ITE: the moved engine against the frozen b155550 runs
    run_testfile("BUG/TimeEvolution/test_fixtures_b155550.jl")
    # QMB
    run_testfile("QMB/Observables/test_local_observables.jl")
    # QTTS
    run_testfile("QTTS/Tooling/test_grid.jl")
    run_testfile("QTTS/Tooling/test_cores.jl")
    run_testfile("QTTS/Tooling/test_operators.jl")

    run_testfile("quality/import_boundaries.jl")
    run_testfile("quality/aqua.jl")
end
