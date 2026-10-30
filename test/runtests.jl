using Test
using BondUpdateGalerkin

@testset "BondUpdateGalerkin" begin
    @testset "package loads" begin
        for name in (:BUG, :QMB, :QTTS, :Baselines)
            @test isdefined(BondUpdateGalerkin, name)
        end
    end

    include("quality/import_boundaries.jl")
    include("quality/aqua.jl")
end
