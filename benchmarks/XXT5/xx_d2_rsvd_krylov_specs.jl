# Spin-1/2 XX U(1): 4 local Krylov strategies x 6 DD-RSVD policies.
# All arms use the same initial state, tolerances, work ceiling and root method.
# The extra DD probes in A/B use their own copied RNG stream inside the sweeps.
function d2_rsvd_krylov_specs(basis_tol::Real)
    rsvd = [
        ("fixed", (mode = :fixed, fixed_width = 8, fixed_rank = 4)),
        ("sequential", (
            mode = :adaptive, holdout_width = 2, acceptance_factor = 0.6,
            batch_blocks = 1,
        )),
        ("batched_svd", (
            mode = :adaptive, holdout_width = 2, acceptance_factor = 0.6,
            batch_blocks = 2, cap_selection = :svd,
        )),
        ("rank_adaptive", (
            mode = :rank_adaptive, rank_initial_width = 4,
            holdout_width = 4, acceptance_factor = 0.7,
        )),
        ("rank_tail", (
            mode = :rank_tail, fixed_width = 8, tail_factor = 0.10,
        )),
        ("rank_fast", (
            mode = :rank_fast, rank_initial_width = 3,
            holdout_width = 2, acceptance_factor = 0.6,
        )),
    ]
    krylov = [
        ("fixed8", 8, :none),
        ("adaptive", 0, :none),
        ("interleaved", 0, :interleaved),
        ("post", 0, :post),
    ]
    arms = Tuple{String,DDQRExpansion,NamedTuple}[]
    common_step = (
        work_ceiling_rule = :bond,
        bond_ceiling_factor = 1.2,
    )
    for (rname, rspec) in rsvd, (kname, depth, ddmode) in krylov
        policy = DDQRExpansion(;
            merge(rspec, (krylov_mode = :lanczos, krylov_fixed_depth = depth))...,
        )
        more = ddmode === :none ? NamedTuple() : (
            extra_dd_mode = ddmode,
            extra_dd_width = 8,
            extra_dd_holdout = 4,
            extra_dd_max_checks = 2,
        )
        push!(arms, ("$(rname)__$(kname)", policy, merge(common_step, more)))
    end
    return arms
end
