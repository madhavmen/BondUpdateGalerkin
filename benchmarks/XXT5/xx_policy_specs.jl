# The DD/Krylov policy arms of the XX T=5 benchmarks, shared by xx_T5_matched_work.jl and
# xx_large_rsvd_audit.jl so both run exactly the same expansion settings.
# Each entry is (name, DDQRExpansion, extra cbe_bug_step! keywords).

include(joinpath(@__DIR__, "xx_d2_rsvd_krylov_specs.jl"))

function policy_specs(basis_tol)
    if get(ENV, "BUG_D2_FACTORIAL", "0") == "1"
        return d2_rsvd_krylov_specs(basis_tol)
    end
    return [
        (
            "fixed4_depth4",
            DDQRExpansion(
                mode = :fixed,
                fixed_width = 8,
                fixed_rank = 4,
                krylov_mode = :lanczos,
                krylov_fixed_depth = 4,
            ),
            NamedTuple(),
        ),
        (
            "fixed4_depth8",
            DDQRExpansion(
                mode = :fixed,
                fixed_width = 8,
                fixed_rank = 4,
                krylov_mode = :lanczos,
                krylov_fixed_depth = 8,
            ),
            NamedTuple(),
        ),
        (
            "fixed8_depth8",
            DDQRExpansion(
                mode = :fixed,
                fixed_width = 12,
                fixed_rank = 8,
                krylov_mode = :lanczos,
                krylov_fixed_depth = 8,
            ),
            NamedTuple(),
        ),
        (
            "fixed4_arnoldi8",
            DDQRExpansion(
                mode = :fixed,
                fixed_width = 8,
                fixed_rank = 4,
                krylov_mode = :arnoldi,
                krylov_fixed_depth = 8,
            ),
            NamedTuple(),
        ),
        (
            "fixed4_adaptK",
            DDQRExpansion(mode = :fixed, fixed_width = 8, fixed_rank = 4, krylov_mode = :lanczos),
            NamedTuple(),
        ),
        (
            "adaptive_lanczos",
            DDQRExpansion(holdout_width = 2, acceptance_factor = 0.6, krylov_mode = :lanczos),
            NamedTuple(),
        ),
        (
            "adaptive_arnoldi",
            DDQRExpansion(holdout_width = 2, acceptance_factor = 0.6, krylov_mode = :arnoldi),
            NamedTuple(),
        ),
        (
            "adaptive_ranked",
            DDQRExpansion(holdout_width = 2, acceptance_factor = 0.6, krylov_mode = :lanczos),
            (krylov_rank_tol = basis_tol,),
        ),
        (
            "adaptive_unchecked",
            DDQRExpansion(holdout_width = 2, acceptance_factor = 0.6, krylov_mode = :lanczos),
            (closure_checks = false,),
        ),
        # Per-bond ceil(1.2*d*current_rank), applied to BOTH DD and K/L and root
        # closure frames. The cap is a resource restriction, not a tolerance.
        (
            "fixed4_cap12",
            DDQRExpansion(mode = :fixed, fixed_width = 8, fixed_rank = 4, krylov_mode = :lanczos),
            (work_ceiling_rule = :bond, bond_ceiling_factor = 1.2),
        ),
        (
            "sequential_cap12",
            DDQRExpansion(
                holdout_width = 2,
                acceptance_factor = 0.6,
                batch_blocks = 1,
                krylov_mode = :lanczos,
            ),
            (work_ceiling_rule = :bond, bond_ceiling_factor = 1.2),
        ),
        (
            "batch2_cap12",
            DDQRExpansion(
                holdout_width = 2,
                acceptance_factor = 0.6,
                batch_blocks = 2,
                krylov_mode = :lanczos,
            ),
            (work_ceiling_rule = :bond, bond_ceiling_factor = 1.2),
        ),
        (
            "batch3_cap12",
            DDQRExpansion(
                holdout_width = 2,
                acceptance_factor = 0.6,
                batch_blocks = 3,
                krylov_mode = :lanczos,
            ),
            (work_ceiling_rule = :bond, bond_ceiling_factor = 1.2),
        ),
        (
            "sequential_svd_cap12",
            DDQRExpansion(
                holdout_width = 2,
                acceptance_factor = 0.6,
                batch_blocks = 1,
                cap_selection = :svd,
                krylov_mode = :lanczos,
            ),
            (work_ceiling_rule = :bond, bond_ceiling_factor = 1.2),
        ),
        (
            "batch2_svd_cap12",
            DDQRExpansion(
                holdout_width = 2,
                acceptance_factor = 0.6,
                batch_blocks = 2,
                cap_selection = :svd,
                krylov_mode = :lanczos,
            ),
            (work_ceiling_rule = :bond, bond_ceiling_factor = 1.2),
        ),
        (
            "batch3_svd_cap12",
            DDQRExpansion(
                holdout_width = 2,
                acceptance_factor = 0.6,
                batch_blocks = 3,
                cap_selection = :svd,
                krylov_mode = :lanczos,
            ),
            (work_ceiling_rule = :bond, bond_ceiling_factor = 1.2),
        ),
        (
            "batch2_uncapped",
            DDQRExpansion(
                holdout_width = 2,
                acceptance_factor = 0.6,
                batch_blocks = 2,
                krylov_mode = :lanczos,
            ),
            NamedTuple(),
        ),
        (
            "batch3_uncapped",
            DDQRExpansion(
                holdout_width = 2,
                acceptance_factor = 0.6,
                batch_blocks = 3,
                krylov_mode = :lanczos,
            ),
            NamedTuple(),
        ),
        (
            "fixed4_cap15",
            DDQRExpansion(mode = :fixed, fixed_width = 8, fixed_rank = 4, krylov_mode = :lanczos),
            (work_ceiling_rule = :bond, bond_ceiling_factor = 1.5),
        ),
        (
            "batch2_cap15",
            DDQRExpansion(
                holdout_width = 2,
                acceptance_factor = 0.6,
                batch_blocks = 2,
                krylov_mode = :lanczos,
            ),
            (work_ceiling_rule = :bond, bond_ceiling_factor = 1.5),
        ),
        (
            "T5_fixed4_k8",
            DDQRExpansion(
                mode = :fixed,
                fixed_width = 8,
                fixed_rank = 4,
                krylov_mode = :lanczos,
                krylov_fixed_depth = 8,
            ),
            (work_ceiling_rule = :bond, bond_ceiling_factor = 1.2),
        ),
        (
            "T5_fixed4_k12",
            DDQRExpansion(
                mode = :fixed,
                fixed_width = 8,
                fixed_rank = 4,
                krylov_mode = :lanczos,
                krylov_fixed_depth = 12,
            ),
            (work_ceiling_rule = :bond, bond_ceiling_factor = 1.2),
        ),
        (
            "T5_fixed8_k8",
            DDQRExpansion(
                mode = :fixed,
                fixed_width = 12,
                fixed_rank = 8,
                krylov_mode = :lanczos,
                krylov_fixed_depth = 8,
            ),
            (work_ceiling_rule = :bond, bond_ceiling_factor = 1.2),
        ),
        (
            "T5_batch2_fixedK8",
            DDQRExpansion(
                holdout_width = 2,
                acceptance_factor = 0.6,
                batch_blocks = 2,
                cap_selection = :svd,
                krylov_mode = :lanczos,
                krylov_fixed_depth = 8,
            ),
            (work_ceiling_rule = :bond, bond_ceiling_factor = 1.2),
        ),
        (
            "T5_batch2_arnoldi",
            DDQRExpansion(
                holdout_width = 2,
                acceptance_factor = 0.6,
                batch_blocks = 2,
                cap_selection = :svd,
                krylov_mode = :arnoldi,
            ),
            (work_ceiling_rule = :bond, bond_ceiling_factor = 1.2),
        ),
        (
            "T5_fixed4_adaptK",
            DDQRExpansion(mode = :fixed, fixed_width = 8, fixed_rank = 4, krylov_mode = :lanczos),
            (work_ceiling_rule = :bond, bond_ceiling_factor = 1.2),
        ),
        # Fair A/B comparison: exact same initial DD sketch and rank cap as
        # T5_fixed4_adaptK, only the optional local Krylov-directed DD changes.
        (
            "T5_fixed4_adaptK_ddA",
            DDQRExpansion(mode = :fixed, fixed_width = 8, fixed_rank = 4, krylov_mode = :lanczos),
            (
                work_ceiling_rule = :bond,
                bond_ceiling_factor = 1.2,
                extra_dd_mode = :interleaved,
                extra_dd_width = 8,
                extra_dd_holdout = 4,
                extra_dd_max_checks = 2,
            ),
        ),
        (
            "T5_fixed4_adaptK_ddB",
            DDQRExpansion(mode = :fixed, fixed_width = 8, fixed_rank = 4, krylov_mode = :lanczos),
            (
                work_ceiling_rule = :bond,
                bond_ceiling_factor = 1.2,
                extra_dd_mode = :post,
                extra_dd_width = 8,
                extra_dd_holdout = 4,
                extra_dd_max_checks = 2,
            ),
        ),
        (
            "T5_uncapped",
            DDQRExpansion(
                holdout_width = 2,
                acceptance_factor = 0.6,
                batch_blocks = 2,
                cap_selection = :svd,
                krylov_mode = :lanczos,
            ),
            NamedTuple(),
        ),
        # New genuine rank adaptation: thin block SVD ranks the DD candidates,
        # a holdout selects the MINIMUM accepted k, and a second holdout confirms.
        (
            "T5_rank_adaptK",
            DDQRExpansion(
                mode = :rank_adaptive,
                rank_initial_width = 4,
                holdout_width = 4,
                acceptance_factor = 0.7,
                krylov_mode = :lanczos,
            ),
            (work_ceiling_rule = :bond, bond_ceiling_factor = 1.2),
        ),
        (
            "T5_rank_fixedK8",
            DDQRExpansion(
                mode = :rank_adaptive,
                rank_initial_width = 4,
                holdout_width = 4,
                acceptance_factor = 0.7,
                krylov_mode = :lanczos,
                krylov_fixed_depth = 8,
            ),
            (work_ceiling_rule = :bond, bond_ceiling_factor = 1.2),
        ),
        (
            "T5_rank_arnoldi",
            DDQRExpansion(
                mode = :rank_adaptive,
                rank_initial_width = 4,
                holdout_width = 4,
                acceptance_factor = 0.7,
                krylov_mode = :arnoldi,
            ),
            (work_ceiling_rule = :bond, bond_ceiling_factor = 1.2),
        ),
        (
            "T5_oldstrict_adaptK",
            DDQRExpansion(mode = :fixed, fixed_width = 8, fixed_rank = 4, krylov_mode = :lanczos),
            (work_ceiling_rule = :bond_strict, bond_ceiling_factor = 1.2),
        ),
        (
            "T5_split_fixed8",
            DDQRExpansion(
                mode = :fixed,
                fixed_width = 8,
                fixed_rank = 4,
                krylov_mode = :lanczos,
                krylov_fixed_depth = 8,
            ),
            (work_ceiling_rule = :frame_root, bond_ceiling_factor = 2.0, root_ceiling_factor = 1.2),
        ),
        (
            "T5_split_adaptK",
            DDQRExpansion(mode = :fixed, fixed_width = 8, fixed_rank = 4, krylov_mode = :lanczos),
            (work_ceiling_rule = :frame_root, bond_ceiling_factor = 2.0, root_ceiling_factor = 1.2),
        ),
        (
            "T5_split_rank",
            DDQRExpansion(
                mode = :rank_adaptive,
                rank_initial_width = 4,
                holdout_width = 4,
                acceptance_factor = 0.7,
                krylov_mode = :lanczos,
            ),
            (work_ceiling_rule = :frame_root, bond_ceiling_factor = 2.0, root_ceiling_factor = 1.2),
        ),
        (
            "T5_split_batched",
            DDQRExpansion(
                holdout_width = 2,
                batch_blocks = 2,
                cap_selection = :svd,
                acceptance_factor = 0.6,
                krylov_mode = :lanczos,
            ),
            (work_ceiling_rule = :frame_root, bond_ceiling_factor = 2.0, root_ceiling_factor = 1.2),
        ),
        # Same old 1.2*d*Db cap and adaptive enriched Lanczos for ALL.
        # Fast pilot uses two independent 2-column holdouts, maximizing
        # scalability at chi >=500 (max 64 trial columns).
        (
            "RSVD_pilot2_Kadapt",
            DDQRExpansion(
                mode = :rank_fast,
                rank_initial_width = 2,
                holdout_width = 2,
                acceptance_factor = 0.6,
                krylov_mode = :lanczos,
            ),
            (work_ceiling_rule = :bond, bond_ceiling_factor = 1.2),
        ),
        (
            "RSVD_pilot3_Kadapt",
            DDQRExpansion(
                mode = :rank_fast,
                rank_initial_width = 3,
                holdout_width = 2,
                acceptance_factor = 0.6,
                krylov_mode = :lanczos,
            ),
            (work_ceiling_rule = :bond, bond_ceiling_factor = 1.2),
        ),
        (
            "RSVD_pilot4_Kadapt",
            DDQRExpansion(
                mode = :rank_fast,
                rank_initial_width = 4,
                holdout_width = 2,
                acceptance_factor = 0.6,
                krylov_mode = :lanczos,
            ),
            (work_ceiling_rule = :bond, bond_ceiling_factor = 1.2),
        ),
        (
            "RSVD_pilot3_margin8",
            DDQRExpansion(
                mode = :rank_fast,
                rank_initial_width = 3,
                holdout_width = 2,
                acceptance_factor = 0.8,
                krylov_mode = :lanczos,
            ),
            (work_ceiling_rule = :bond, bond_ceiling_factor = 1.2),
        ),
        (
            "RSVD_rank_standard_Kadapt",
            DDQRExpansion(
                mode = :rank_adaptive,
                rank_initial_width = 4,
                holdout_width = 4,
                acceptance_factor = 0.7,
                krylov_mode = :lanczos,
            ),
            (work_ceiling_rule = :bond, bond_ceiling_factor = 1.2),
        ),
        (
            "RSVD_fixed8rank4_Kadapt",
            DDQRExpansion(mode = :fixed, fixed_width = 8, fixed_rank = 4, krylov_mode = :lanczos),
            (work_ceiling_rule = :bond, bond_ceiling_factor = 1.2),
        ),
        (
            "RSVD_fixed12rank8_Kadapt",
            DDQRExpansion(mode = :fixed, fixed_width = 12, fixed_rank = 8, krylov_mode = :lanczos),
            (work_ceiling_rule = :bond, bond_ceiling_factor = 1.2),
        ),
        (
            "RSVD_sequential_Kadapt",
            DDQRExpansion(
                mode = :adaptive,
                holdout_width = 2,
                acceptance_factor = 0.6,
                batch_blocks = 1,
                krylov_mode = :lanczos,
            ),
            (work_ceiling_rule = :bond, bond_ceiling_factor = 1.2),
        ),
        (
            "RSVD_batched_Kadapt",
            DDQRExpansion(
                mode = :adaptive,
                holdout_width = 2,
                acceptance_factor = 0.6,
                batch_blocks = 2,
                cap_selection = :svd,
                krylov_mode = :lanczos,
            ),
            (work_ceiling_rule = :bond, bond_ceiling_factor = 1.2),
        ),
        # One-pass tail rank selection: same MPO sketch as fixed width 8/12.
        # Crucially, this is NOT an independently validated DD residual.
        # The checked order2 global Krylov closure and ED state error decide
        # whether this cheaper selection is physically acceptable.
        (
            "RSVD_tail12_factor01",
            DDQRExpansion(
                mode = :rank_tail,
                fixed_width = 12,
                tail_factor = 0.10,
                krylov_mode = :lanczos,
            ),
            (work_ceiling_rule = :bond, bond_ceiling_factor = 1.2),
        ),
        (
            "RSVD_tail12_factor005",
            DDQRExpansion(
                mode = :rank_tail,
                fixed_width = 12,
                tail_factor = 0.05,
                krylov_mode = :lanczos,
            ),
            (work_ceiling_rule = :bond, bond_ceiling_factor = 1.2),
        ),
        (
            "RSVD_tail8_factor01",
            DDQRExpansion(
                mode = :rank_tail,
                fixed_width = 8,
                tail_factor = 0.10,
                krylov_mode = :lanczos,
            ),
            (work_ceiling_rule = :bond, bond_ceiling_factor = 1.2),
        ),
    ]
end
