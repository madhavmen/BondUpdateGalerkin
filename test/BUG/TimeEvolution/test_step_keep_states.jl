# `cbe_bug_step!`'s kept states and frames-only builds, sized to run in seconds (Heisenberg, L = 8,
# U(1), exact selection):
#   - a kept state with `keep_tols = 0` is contained exactly; admitted at a tolerance it widens the
#     frames no more, and under `work_ceiling_rule = :bond` no frame exceeds its bond's ceiling;
#   - `frames_only = true` skips the root solve: the state comes back as `psi` in the frames, and
#     the frames are those of the full step;
#   - mismatched `keep_tols`, and `frames_only` with a truncation, are refused.

using Test, LinearAlgebra
using BondUpdateGalerkin.BUG, BondUpdateGalerkin.QMB

@testset "cbe_bug_step!: keep_tols and frames_only" begin
    set_symmetry!(:U1)
    clear_fusion_cache!()
    chain_length, site_dim = 8, 2
    hamiltonian = xxz_mpo(chain_length; J = 1.0, delta = 1.0)
    time_step = ComplexF64(-0.2im)
    step_settings = (order = 2, tol_basis = 1e-8, tol_trunc = 1e-12, maxdim = 64, exact = true)
    start_state = neel_state(chain_length)
    cbe_bug_step!(start_state, hamiltonian, time_step; step_settings...)
    # a more entangled state of the same sector, with directions outside the step's frames
    kept_state = copy(start_state)
    for _ = 1:3
        cbe_bug_step!(kept_state, hamiltonian, ComplexF64(-0.3im); step_settings...)
    end
    start_bonds = bond_dims(start_state)

    # frame widths of one step from `start_state` and the frames it reports
    function keep_step(; keywords...)
        frames = Ref{Any}(nothing)
        info = cbe_bug_step!(
            copy(start_state),
            hamiltonian,
            time_step;
            step_settings...,
            truncate = false,
            frames_report = frames,
            keywords...,
        )
        return info.expanded, frames[]
    end

    plain_widths, _ = keep_step()
    exact_widths, _ = keep_step(; keep_states = [kept_state])
    admitted_widths, _ = keep_step(; keep_states = [kept_state], keep_tols = [1e-6])
    @test any(exact_widths .> plain_widths)                   # fixture guard: the state widens
    @test all(admitted_widths .<= exact_widths)
    # the cap: under the per-bond rule an admitted state fills a frame only up to the ceiling
    small_maxdim = max(maximum(start_bonds), 2)
    ceilings = bond_work_ceilings(start_bonds, site_dim, small_maxdim)
    capped_widths, _ = keep_step(;
        keep_states = [kept_state],
        keep_tols = [1e-12],
        maxdim = small_maxdim,
        work_ceiling_rule = :bond,
    )
    @test all(capped_widths .<= ceilings)

    # frames only: no root solve, so the state is the input in the frames
    _, full_frames = keep_step()
    frames_state = copy(start_state)
    frames_ref = Ref{Any}(nothing)
    cbe_bug_step!(
        frames_state,
        hamiltonian,
        time_step;
        step_settings...,
        truncate = false,
        frames_only = true,
        frames_report = frames_ref,
    )
    overlap_ratio =
        abs(overlap(frames_state, start_state)) / (norm(frames_state) * norm(start_state))
    @test 1 - overlap_ratio < 1e-12
    @test [leg_dim(frame, 3) for frame in frames_ref[].left_frames] ==
          [leg_dim(frame, 3) for frame in full_frames.left_frames]

    @test_throws ArgumentError keep_step(; keep_states = [kept_state], keep_tols = [1e-6, 1e-6])
    @test_throws ArgumentError cbe_bug_step!(
        copy(start_state),
        hamiltonian,
        time_step;
        step_settings...,
        frames_only = true,
    )
end
