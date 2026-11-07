"""
    BondUpdateGalerkin

Rank-adaptive basis-update-and-Galerkin (BUG) time integration for quantum many-body matrix
product states and quantics tensor trains, with one shared numerical engine.

Modules: [`BUG`](@ref) (the engine), [`QMB`](@ref) (quantum many-body models),
[`QTTS`](@ref) (quantics tensor trains for PDEs), [`Baselines`](@ref) (comparison methods).
"""
module BondUpdateGalerkin

# BUG first: the domain modules and the baselines build on it, never the reverse.
include("BUG/BUG.jl")
include("QMB/QMB.jl")
include("QTTS/QTTS.jl")
include("Baselines/Baselines.jl")

# ── precompile workload ──
# Loading the package is cheap; the cost is specialising Telum's block-sparse kernels the first
# time each entry point runs, and a plain `using` caches none of it. Without this every process
# paid it again: on the cluster the test suite took 28 min warm, the same as cold, with 4 min in a
# file that only builds 6-site product states. Running the entry points here moves that work into
# the package image, paid once per source change.
#
# Kept small (6 sites, maxdim 8; 3+3 QTT bits): the types, not the sizes, decide what is compiled,
# and the workload reruns after every edit to `src/`. It covers the entry points the tests and
# examples call; a compile-vs-compute profile of the test files (17633220) showed the remaining
# per-process cost was in exactly these (midpoint and imaginary-time steps, MPO products, Z2,
# baselines, QTT steps). Each block is guarded on its own: a workload that throws must leave a
# slower package, never one that fails to precompile, and one failing block must not drop the
# others. `finally` restores the module state (symmetry mode, fusion memo) that would otherwise
# be saved into the image.
using PrecompileTools: @setup_workload, @compile_workload

# Test is loaded before the workload so the image is compiled in a world that already has it.
# Test defines `keys(::Test.GenericArray)`, and loading it afterwards invalidated 9069 compiled
# method instances behind every `keys(::AbstractVector)` call in the Telum kernels (SnoopCompile,
# 17634477): the first BUG step went from 6 s to 95 s, 49 s of it recompilation (17634373), so
# the test suite, which always loads Test, recompiled most of the workload.
import Test

"Run one workload block; a failure is reported at precompile time and otherwise ignored."
function _precompile_block(run_block, name)
    try
        run_block()
    catch err
        @warn "BondUpdateGalerkin precompile workload: the $name block did not complete; its " *
              "first call in each process pays full compilation again" exception = err
    end
    return nothing
end

@setup_workload begin
    @compile_workload begin
        try
            # MPS: the BUG steps, observables and baselines, with and without U(1)
            _precompile_block("MPS") do
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
                        BUG.mpo_energy(state, hamiltonian)
                        BUG.overlap(state, state)
                    end
                    BUG.cbe_bug_midpoint_step!(
                        copy(initial_state),
                        hamiltonian,
                        time_step;
                        maxdim = 8,
                    )
                    BUG.ite_evolve!(
                        copy(initial_state),
                        hamiltonian,
                        BUG.uniform_beta_steps(0.25, 2);
                        order = 2,
                        maxdim = 8,
                    )
                    # an on-site MPO and the exact MPO-MPS product
                    spin = QMB.local_space()
                    total_sz = BUG.onsite_mpo(6, [spin.Sz for _ = 1:6]; Iloc = spin.I)
                    BUG.mpo_product(total_sz, initial_state)
                    # the baselines: TDVP2 (Lanczos and Arnoldi, composed order 4)
                    for hermitian in (true, false)
                        Baselines.tdvp2_step!(
                            copy(initial_state),
                            hamiltonian,
                            time_step;
                            maxdim = 8,
                            hermitian = hermitian,
                        )
                    end
                    Baselines.tdvp2_composed_step!(
                        copy(initial_state),
                        hamiltonian,
                        time_step;
                        order = 4,
                        maxdim = 8,
                    )
                end
            end
            # Z2: the transverse-field Ising states and gate energy
            _precompile_block("Z2") do
                BUG.set_symmetry!(:Z2)
                polarised = QMB.ising_polarised_state(6)
                QMB.energy(polarised, QMB.ising_bond_gates(polarised; J = 1.0, h = 1.0))
                QMB.x_profile(polarised)
            end
            # QTT: the nonlinear GP flow and the linear Heston flow, BUG and TDVP2 steps
            _precompile_block("QTT") do
                BUG.set_symmetry!(:none)
                gp_grid = QTTS.qtt_grid(3, 3; x = (-5.0, 5.0), y = (-5.0, 5.0))
                gp = QTTS.gp_model(gp_grid; gnl = 1.0, gamma = 0.1, mu = 0.0, omega = 1.0)
                gp_initial = QTTS.qtt_separable(gp_grid, x -> exp(-x^2 / 2), y -> exp(-y^2 / 2))
                QTTS.qtt_bug_step!(QTTS.qtt_mps(gp_initial), QTTS.GPFlow(gp), 0.0, 0.1; maxdim = 16)
                Baselines.qtt_tdvp2_step!(
                    QTTS.qtt_mps(gp_initial),
                    QTTS.GPFlow(gp),
                    0.0,
                    0.05;
                    maxdim = 16,
                )
                heston_grid = QTTS.qtt_grid(
                    3,
                    3;
                    x = (-1.0, 1.0),
                    y = (0.0, 0.5),
                    periodic_x = false,
                    periodic_y = false,
                )
                heston_params = QTTS.HestonParams(0.03, 0.01, 2.0, 0.09, 0.4, -0.7)
                heston = QTTS.heston_op(heston_grid, heston_params)
                QTTS.qtt_bug_step!(
                    QTTS.qtt_mps(QTTS.qtt_separable(heston_grid, x -> exp(x), v -> 1.0)),
                    BUG.LinearFlow(heston),
                    0.0,
                    0.01;
                    order = 3,
                    maxdim = 16,
                )
            end
        finally
            BUG.set_symmetry!(:U1)
            BUG.clear_fusion_cache!()
        end
    end
end

end
