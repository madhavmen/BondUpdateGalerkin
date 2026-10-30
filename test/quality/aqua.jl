# Package hygiene: undefined exports, ambiguities, compat entries, type piracy (plan Section 7).
using Aqua

@testset "Aqua" begin
    Aqua.test_all(BondUpdateGalerkin)
end
