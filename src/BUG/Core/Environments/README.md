# `src/BUG/Core/Environments/`

The Hamiltonian enters the engine only here: as an MPO contracted into left and right environments,
and the effective operators built from them. There is no other Hamiltonian representation (no
gates, no term-list environments).

| file | contents |
|---|---|
| `mpo_environments.jl` | Site-operator and bra-tensor helpers (`_apply_site_op`, `_bra_*`); the environment on one link (`MPOLink`) and how a sweep carries it (`boundary_channels`, `push_left_channels`, `push_right_channels`); prebuilt stacks (`left_env_stack`, `right_env_stack`, `MPOLeftEnvStack`, `MPORightEnvStack`); `mpo_energy`, `mpo_matrix_element`, `mpo_variance`; the two-site action `apply_h_two_site`; and the link-leg and streamed two-site contractions the sweeps share. |
| `effective_operators.jl` | The one-site (`MPOOneSiteH`, `one_site_h`, `apply_one_site`) and zero-site (`MPOZeroSiteH`, `zero_site_h`, `apply_zero_site`) effective Hamiltonians; the shifted operator `shift_h` the root solve uses for `H - eshift`; writing an expansion back into the state (`_absorb_left!`, `_absorb_right!`, `_frame_from`, `_relink`); the default root link `_bug_root_link`. |

The two-site action never forms a dense two-site matrix: environments are held as factors and
applied one contraction at a time.
