# `src/BUG/Core/Sweeps/`

The sweeps over the chain that build the step's trial space. They only build bases; the state is
evolved once, by the root solve on the centre bond.

| file | contents |
|---|---|
| `frames.jl` | `BondFrame`, `bond_frame`, `frame_theta`, `two_site_block`: the exact canonical two-site snapshot of one bond, the input of the bond expansion. |
| `half_sweeps.jl` | `_half_left!` and `_half_right!`, the two half-sweeps of `cbe_bug_step!`. At each bond: the bond expansion (`expand_frame`), then the Krylov enrichment (`_krylov_enrich`), then any kept states (`keep_states`). The two share no data, so `parallel = true` runs them concurrently. `_HalfSweepCtx` holds what both read. |
| `frontier_sweeps.jl` | `_frontier_left`, `_frontier_right`: the half-sweeps of one closure frontier, which embed the old frames into the grown ones and expand where the frontier's outside action is too large; `_polar_frame` keeps the grown frames orthonormal. |
| `frontier_closure.jl` | `_frontier_sweep!` (both sides of one frontier), the measurements built on it (`estimate_global_complement_action`), and the closure's termination loop. |
| `parallel_kls.jl` | The fully parallel K/L/S integrator (`KLS_SCHEDULES`): the K, L and root equations from `Y_n` at once in the preliminary frames, with a fallback to the standard root solve when the omitted block is too large. Entry point: `parallel_kls_step!` in `../../TimeEvolution/bug_step_variants.jl`. |
