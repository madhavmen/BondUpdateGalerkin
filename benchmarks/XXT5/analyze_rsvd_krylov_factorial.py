"""Analysis of the AUDIT_SET=krylov_factorial runs of xx_large_rsvd_audit.jl.

Usage: python benchmarks/XXT5/analyze_rsvd_krylov_factorial.py OUTPUT_DIR RUN_DIR [RUN_DIR ...]

Each arm is one RSVD option crossed with one adaptive local Krylov solver ("rsvd+krylov").
For spin 1/2 the bond dimension each arm carries is compared, bond by bond, with the EXACT chi of the
free-fermion state under the same per-bond truncation share the step uses (truncate_recursive!: each
cut may discard a Schmidt norm of tol_trunc / (sqrt(root - 1) + sqrt(L - root)), root = L/2 + 1).
For spin 1 and 3/2 the TDVP2 arm at the same tolerance stands in for the needed chi.

Writes OUTPUT_DIR/factorial_summary.csv (one row per run setting and arm, seeds averaged),
OUTPUT_DIR/factorial_report.md and figures chi_vs_time.png and error_vs_wall_factorial.png.
"""

import heapq
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

TRUNC_FACTOR = 1e-3            # tol_trunc = 1e-3 * tol_basis in the driver
TIME_STEP = 0.1
TIMERS = ["t_cbe", "t_kry", "t_frame", "t_env", "t_root", "t_trunc", "t_closure"]
# Colour follows the RSVD option (categorical slots of the reference palette, fixed order);
# the marker and line style follow the Krylov solver.
RSVD_COLOURS = {
    "fixed8rank4": "#2a78d6", "fixed12rank8": "#eb6834", "rank_adaptive": "#1baf7a",
    "rank_fast3": "#eda100", "rank_tail12": "#e87ba4", "rank_tail8": "#008300", "tdvp2": "#6250d6",
}
KRYLOV_STYLE = {"lanczos": ("o", "-"), "arnoldi": ("s", "--"), "lanczos_ranked": ("^", ":"), "tdvp2": ("D", "-")}


def split_arm(policy):
    if policy == "tdvp2":
        return "tdvp2", "tdvp2"
    rsvd, krylov = policy.split("+")
    return rsvd, krylov


def schmidt_weights(nus, floor=1e-20, count_limit=400000):
    """Largest Schmidt weights of a Slater determinant from its entanglement modes, best first."""
    probabilities = sorted(max(nu, 1 - nu) for nu in nus)      # most mixed mode first
    ratios = [(1 - p) / p for p in probabilities]                # descending
    base = float(np.prod(probabilities))
    weights, heap = [], [(-base, -1)]
    while heap and len(weights) < count_limit:
        negative_weight, last = heapq.heappop(heap)
        weights.append(-negative_weight)
        for mode in range(last + 1, len(ratios)):
            child = -negative_weight * ratios[mode]
            if child < floor:
                break
            heapq.heappush(heap, (-child, mode))
    return np.array(weights)


def exact_chi_profile(chain_length, up_sites, time, cut_weight):
    """Exact free-fermion chi on every bond: fewest Schmidt states with discarded weight <= cut_weight."""
    hopping = np.zeros((chain_length, chain_length))
    index = np.arange(chain_length - 1)
    hopping[index, index + 1] = hopping[index + 1, index] = 0.5
    energies, vectors = np.linalg.eigh(hopping)
    propagator = (vectors * np.exp(-1j * energies * time)) @ vectors.T
    correlation = propagator[:, up_sites] @ propagator[:, up_sites].conj().T
    profile = []
    for bond in range(1, chain_length):
        nus = np.clip(np.linalg.eigvalsh(correlation[:bond, :bond]), 0, 1)
        discarded = 1 - np.cumsum(schmidt_weights(nus))
        within = np.nonzero(discarded <= cut_weight)[0]
        profile.append(int(within[0] + 1) if len(within) else np.nan)
    return np.array(profile, dtype=float)


def cut_weight(chain_length, basis_tol):
    root = chain_length // 2 + 1
    divisor = np.sqrt(root - 1) + np.sqrt(chain_length - root)
    return (TRUNC_FACTOR * basis_tol / divisor) ** 2


def load(run_dirs):
    return pd.concat([pd.read_csv(Path(run) / "steps.csv") for run in run_dirs], ignore_index=True)


def add_chi_comparison(steps):
    """Per sampled row: centre-bond chi, exact centre chi (spin 1/2) and the worst bond-by-bond ratio."""
    sampled = steps[steps.bond_profile.notna() & (steps.bond_profile != "")].copy()
    exact_cache = {}
    centre_chi, exact_centre, worst_ratio = [], [], []
    for row in sampled.itertuples():
        profile = np.array([int(value) for value in str(row.bond_profile).split(";")], dtype=float)
        centre = len(profile) // 2
        centre_chi.append(profile[centre])
        if row.twoS != 1:
            exact_centre.append(np.nan)
            worst_ratio.append(np.nan)
            continue
        up = list(range(0, row.L, 2)) if row.initial == "neel" else list(range(row.L // 2))
        key = (row.initial, row.L, row.basis_tol, row.step)
        if key not in exact_cache:
            exact_cache[key] = exact_chi_profile(row.L, up, row.step * TIME_STEP, cut_weight(row.L, row.basis_tol))
        exact = exact_cache[key]
        exact_centre.append(exact[centre])
        worst_ratio.append(np.nanmax(profile / exact))
    sampled["chi_centre"] = centre_chi
    sampled["chi_exact_centre"] = exact_centre
    sampled["chi_ratio_worst_bond"] = worst_ratio
    return sampled


def summarise(steps, sampled):
    keys = ["initial", "twoS", "L", "basis_tol", "policy", "seed"]
    per_seed = steps.groupby(keys).agg(
        wall_total=("wall_s", "sum"),
        h1_matvecs=("h1", "sum"),
        h1_closure=("h1_closure", "sum"),
        krylov_directions=("n_krylov", "sum"),
        dd_kept=("dd_selected", "sum"),
        sketch_cols=("dd_probes", "sum"),
        holdout_cols=("dd_holdout", "sum"),
        unmet_steps=("budget_status", lambda status: int((status == "budget_unmet").sum())),
        cap_violations=("cap_violations", "sum"),
        **{timer: (timer, "sum") for timer in TIMERS},
    )
    errors = sampled.groupby(keys).agg(
        sz_error_max=("sz_error", "max"),
        chi_centre_T=("chi_centre", "last"),
        chi_exact_centre_T=("chi_exact_centre", "last"),
        chi_ratio_worst=("chi_ratio_worst_bond", "max"),
    )
    final = sampled.sort_values("step").groupby(keys).tail(1).set_index(keys)
    per_seed = per_seed.join(errors)
    per_seed["sz_error_T"] = final["sz_error"]
    for timer in TIMERS:
        per_seed[timer + "_share"] = per_seed[timer] / per_seed["wall_total"]
    summary = per_seed.reset_index().groupby(["initial", "twoS", "L", "basis_tol", "policy"]).mean(numeric_only=True)
    summary = summary.drop(columns="seed").reset_index()
    summary[["rsvd", "krylov"]] = summary.policy.apply(lambda policy: pd.Series(split_arm(policy)))
    return summary


def report(summary):
    lines = ["# RSVD option x adaptive Krylov solver", ""]
    lines += [
        "chi_centre_T: centre-bond chi at T; chi_exact: exact free-fermion chi under the same per-bond cut "
        "(spin 1/2); chi_ratio_worst: largest chi_arm / chi_exact over all bonds and sampled times.",
        "",
    ]
    columns = ["rsvd", "krylov", "sz_error_max", "sz_error_T", "chi_centre_T", "chi_exact_centre_T", "chi_ratio_worst",
               "wall_total", "t_kry_share", "t_closure_share", "t_cbe_share", "krylov_directions", "dd_kept",
               "h1_matvecs", "h1_closure", "unmet_steps", "cap_violations"]
    for (initial, spin_twice, chain_length, basis_tol), setting in summary.groupby(["initial", "twoS", "L", "basis_tol"]):
        lines += [f"## {initial}, d = {spin_twice + 1}, L = {chain_length}, tol_basis = {basis_tol:g}", ""]
        table = setting.sort_values(["krylov", "rsvd"])[columns]
        lines += markdown_table(table)
        lines.append("")
    return "\n".join(lines) + "\n"


def markdown_table(table):
    """A plain Markdown table, numbers to 3 significant digits (no tabulate dependency)."""
    def cell(value):
        return f"{value:.3g}" if isinstance(value, (float, np.floating)) else str(value)
    rows = ["| " + " | ".join(table.columns) + " |", "|" + "---|" * len(table.columns)]
    rows += ["| " + " | ".join(cell(value) for value in record) + " |" for record in table.itertuples(index=False)]
    return rows


def plot_chi(sampled, path):
    settings = sorted(sampled.groupby(["initial", "twoS", "L", "basis_tol"]).groups)
    figure, axes = plt.subplots(len(settings), 3, figsize=(15, 4 * len(settings)), squeeze=False)
    for row_axes, setting in zip(axes, settings):
        initial, spin_twice, chain_length, basis_tol = setting
        subset = sampled[(sampled.initial == initial) & (sampled.twoS == spin_twice) & (sampled.L == chain_length)
                         & (sampled.basis_tol == basis_tol) & (sampled.seed == 0)]
        for axis, krylov in zip(row_axes, ["lanczos", "arnoldi", "lanczos_ranked"]):
            for policy, curve in subset.groupby("policy"):
                rsvd, solver = split_arm(policy)
                if solver not in (krylov, "tdvp2"):
                    continue
                marker, linestyle = KRYLOV_STYLE[solver]
                axis.plot(curve.step * TIME_STEP, curve.chi_centre, color=RSVD_COLOURS[rsvd], marker=marker,
                          markersize=5, linewidth=1.5, linestyle=linestyle, label=rsvd)
            exact = subset.groupby("step").chi_exact_centre.first()
            if exact.notna().any():
                axis.plot(exact.index * TIME_STEP, exact.values, color="#1a1a19", linewidth=2.5, label="exact chi")
            axis.set_yscale("log")
            axis.set_title(f"{krylov}: {initial}, d={spin_twice + 1}, L={chain_length}, tol {basis_tol:g}", fontsize=9)
            axis.set_xlabel("t")
            axis.set_ylabel("centre-bond chi")
            axis.grid(alpha=0.25, linewidth=0.6)
        row_axes[-1].legend(fontsize=7, loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    figure.tight_layout()
    figure.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(figure)


def plot_error_vs_wall(summary, path):
    settings = sorted(summary.groupby(["initial", "twoS", "L", "basis_tol"]).groups)
    figure, axes = plt.subplots(1, len(settings), figsize=(5.5 * len(settings), 4.5), squeeze=False)
    for axis, setting in zip(axes[0], settings):
        initial, spin_twice, chain_length, basis_tol = setting
        subset = summary[(summary.initial == initial) & (summary.twoS == spin_twice) & (summary.L == chain_length)
                         & (summary.basis_tol == basis_tol)]
        for record in subset.itertuples():
            marker, _ = KRYLOV_STYLE[record.krylov]
            axis.scatter(record.wall_total, record.sz_error_max, color=RSVD_COLOURS[record.rsvd], marker=marker, s=55,
                         edgecolors="white", linewidths=1.5, label=f"{record.rsvd}+{record.krylov}")
        axis.set_xscale("log")
        axis.set_yscale("log")
        axis.set_title(f"{initial}, d={spin_twice + 1}, L={chain_length}, tol {basis_tol:g}", fontsize=9)
        axis.set_xlabel("wall time over T (s)")
        axis.set_ylabel("worst sampled max |S^z error|")
        axis.grid(alpha=0.25, linewidth=0.6)
    axes[0][-1].legend(fontsize=6, loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    figure.tight_layout()
    figure.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(figure)


def main():
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    output_dir = Path(sys.argv[1])
    output_dir.mkdir(parents=True, exist_ok=True)
    steps = load(sys.argv[2:])
    sampled = add_chi_comparison(steps)
    summary = summarise(steps, sampled)
    summary.to_csv(output_dir / "factorial_summary.csv", index=False)
    (output_dir / "factorial_report.md").write_text(report(summary), encoding="utf-8")
    plot_chi(sampled, output_dir / "chi_vs_time.png")
    plot_error_vs_wall(summary, output_dir / "error_vs_wall_factorial.png")
    print((output_dir / "factorial_report.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
