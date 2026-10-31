# Package hygiene: undefined exports, ambiguities, compat entries, type piracy (plan Section 7).
#
# `persistent_tasks` is off: it precompiles the package again in a scratch environment, workload
# included, which cost 463 s of a 26 min suite on the cluster (17632492) on every run, warm or
# cold. What it guards, precompilation left waiting on a running task, still shows: `Pkg.test`
# precompiles the package first, and Julia reports a precompile that does not finish.
using Test, Aqua
using BondUpdateGalerkin

@testset "Aqua" begin
    Aqua.test_all(BondUpdateGalerkin; persistent_tasks = false)
end
