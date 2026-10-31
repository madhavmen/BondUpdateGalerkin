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

# ── precompile workload (rebuilt from BUG-Julia@b155550:src/BUGJulia.jl, plan Section 8) ──
# Loading the package is cheap; the cost is specialising Telum's block-sparse kernels the first
# time each entry point runs, and a plain `using` caches none of it. Without this every process
# paid it again: on the cluster the test suite took 28 min warm, the same as cold, with 4 min in a
# file that only builds 6-site product states. Running the entry points here moves that work into
# the package image, paid once per source change.
#
# Kept small (6 sites, maxdim 8): the types, not the sizes, decide what is compiled, and the
# workload reruns after every edit to `src/`. Guarded: a workload that throws must leave a slow
# package, never one that fails to precompile. `finally` restores the module state (symmetry
# mode, fusion memo) that would otherwise be saved into the image.
using PrecompileTools: @setup_workload, @compile_workload

@setup_workload begin
    @compile_workload begin
        try
            for symmetry in (:U1, :none)
                BUG.set_symmetry!(symmetry)
                initial_state = QMB.neel_state(6)
                hamiltonian = QMB.xxz_mpo(6; J = 1.0, delta = 1.0)
                time_step = ComplexF64(-0.05im)
                # order 1 and 2, sequential and concurrent half-sweeps
                for order in (1, 2), parallel in (false, true)
                    state = copy(initial_state)
                    BUG.cbe_bug_step!(
                        state,
                        hamiltonian,
                        time_step;
                        order = order,
                        parallel = parallel,
                        maxdim = 8,
                    )
                    QMB.magnetisation(state)
                end
                # the baselines, Lanczos and Arnoldi branches
                for hermitian in (true, false)
                    Baselines.tdvp2_step!(
                        copy(initial_state),
                        hamiltonian,
                        time_step;
                        maxdim = 8,
                        hermitian = hermitian,
                    )
                end
                Baselines.tdvp_cbe1s_step!(copy(initial_state), hamiltonian, time_step; maxdim = 8)
            end
        catch err
            @warn "BondUpdateGalerkin precompile workload did not complete; the first step in " *
                  "each process pays full compilation again" exception = err
        finally
            BUG.set_symmetry!(:U1)
            BUG.clear_fusion_cache!()
        end
    end
end

end
