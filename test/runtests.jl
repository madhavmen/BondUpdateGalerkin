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
    run_testfile("BUG/Core/LocalSolvers/test_krylov.jl")
    run_testfile("BUG/Core/RankAdaptation/test_basis_budget.jl")
    # BUG/Randomised, BUG/TimeEvolution and BUG/ITE; the last compares the moved engine against
    # the frozen b155550 runs
    run_testfile("BUG/Randomised/test_expansion.jl")
    run_testfile("BUG/ITE/test_schedules.jl")
    run_testfile("BUG/TimeEvolution/test_parallel_half_sweeps.jl")
    run_testfile("BUG/TimeEvolution/test_fixtures_b155550.jl")
    run_testfile("BUG/TimeEvolution/test_stage_solvers.jl")
    # QMB
    run_testfile("QMB/Observables/test_local_observables.jl")
    run_testfile("QMB/ClosedSystem/test_xx.jl")
    run_testfile("QMB/ClosedSystem/test_z2_ising.jl")
    run_testfile("QMB/ClosedSystem/test_tfim.jl")
    run_testfile("QMB/ClosedSystem/test_oat.jl")
    run_testfile("QMB/OpenSystem/MPDO/test_liouville.jl")
    run_testfile("QMB/OpenSystem/MCWF/test_quantum_jumps.jl")
    # QTTS
    run_testfile("QTTS/Tooling/test_grid.jl")
    run_testfile("QTTS/Tooling/test_cores.jl")
    run_testfile("QTTS/Tooling/test_operators.jl")
    run_testfile("QTTS/Tooling/test_derivatives.jl")
    run_testfile("QTTS/Models/Heston/test_heston.jl")
    run_testfile("QTTS/Models/GrossPitaevskii/test_gross_pitaevskii.jl")
    run_testfile("QTTS/Tooling/test_nonlinear_step.jl")
    # Baselines
    run_testfile("Baselines/test_baselines.jl")

    run_testfile("quality/import_boundaries.jl")
    run_testfile("quality/aqua.jl")
end
