"""
    BondUpdateGalerkin

Rank-adaptive basis-update-and-Galerkin (BUG) time integration for quantum many-body matrix
product states and quantics tensor trains, with one shared numerical engine.

Modules: [`BUG`](@ref) (the engine), [`QMB`](@ref) (quantum many-body models),
[`QTTS`](@ref) (quantics tensor trains for PDEs), [`Baselines`](@ref) (comparison methods).
The layout and migration status are in `docs/dev/implementation_plan.md`.
"""
module BondUpdateGalerkin

# BUG first: the domain modules and the baselines build on it, never the reverse.
include("BUG/BUG.jl")
include("QMB/QMB.jl")
include("QTTS/QTTS.jl")
include("Baselines/Baselines.jl")

end
