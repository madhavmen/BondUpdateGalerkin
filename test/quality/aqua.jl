# Package hygiene: undefined exports, ambiguities, compat entries, type piracy.
#
# Aqua runs `ambiguities`, `stale_deps` and `persistent_tasks` each in a fresh Julia process, which
# cost 152 s, 160 s and ~140 s of a ~20 min suite on the cluster (17634349, 17632492). On every
# commit the ambiguity check therefore runs here, in this process, through the same
# `Test.detect_ambiguities` call Aqua makes; the three process-based checks run in full with
# `Pkg.test(test_args = ["aqua-full"])`, the manual pre-commit hook `julia-aqua-full`.
using Test, Aqua
using BondUpdateGalerkin

const FULL_AQUA = "aqua-full" in ARGS

@testset "Aqua" begin
    if FULL_AQUA
        Aqua.test_all(BondUpdateGalerkin)
    else
        Aqua.test_all(
            BondUpdateGalerkin;
            ambiguities = false,
            stale_deps = false,
            persistent_tasks = false,
        )
        @testset "method ambiguities (in process)" begin
            @test isempty(Test.detect_ambiguities(BondUpdateGalerkin; recursive = true))
        end
    end
end
