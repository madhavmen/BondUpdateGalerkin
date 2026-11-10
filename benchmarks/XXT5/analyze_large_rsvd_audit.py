"""Analysis of xx_large_rsvd_audit.jl runs: does each claim of experiments/oldcap-fast-rsvd-xx hold?

Usage: python benchmarks/XXT5/analyze_large_rsvd_audit.py OUTPUT_DIR RUN_DIR [RUN_DIR ...]
Each RUN_DIR holds the steps.csv and bonds.csv of one (L, tol_basis, seed) job.
Writes OUTPUT_DIR/summary.csv (one row per L, tol_basis, policy; seeds averaged),
OUTPUT_DIR/claims.md (one verdict per claim and setting) and three figures.

Claims checked (from the branch's commit messages and code comments):
  C1  :rank_fast needs fewer DD sketch columns (discovery + holdout) than :rank_adaptive at no
      worse accuracy.
  C2  :rank_fast never draws more than 64 trial columns per bond, however large chi grows.
  C3  :rank_tail draws exactly the fixed-width sketch (12 or 8 columns), no holdout, keeps fewer
      directions than the fixed rank, and the checked global closure keeps the state accurate.
  C4  every BUG arm stays inside the per-bond cap min(maxdim, ceil(1.2 d chi_old)).
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

PILOT_POLICIES = ["RSVD_pilot2_Kadapt", "RSVD_pilot3_Kadapt", "RSVD_pilot4_Kadapt", "RSVD_pilot3_margin8"]
STANDARD_POLICY = "RSVD_rank_standard_Kadapt"
TAIL_POLICIES = {"RSVD_tail12_factor01": 12, "RSVD_tail12_factor005": 12, "RSVD_tail8_factor01": 8}
FIXED_POLICIES = {"RSVD_fixed12rank8_Kadapt": (12, 8), "RSVD_fixed8rank4_Kadapt": (8, 4)}
TAIL_FIXED_PARTNER = {
    "RSVD_tail12_factor01": "RSVD_fixed12rank8_Kadapt",
    "RSVD_tail12_factor005": "RSVD_fixed12rank8_Kadapt",
    "RSVD_tail8_factor01": "RSVD_fixed8rank4_Kadapt",
}
PROBE_CEILING = 64
# An arm is "no worse" when its worst sampled S^z error is within this factor of the reference arm's;
# randomised sketches make single-seed errors scatter by about this much.
ACCURACY_SLACK = 1.5

# Colour follows the family (categorical slots of the reference palette, fixed order);
# the marker separates the variants inside a family.
FAMILY_STYLE = {
    "rank_fast": "#2a78d6",
    "rank_adaptive": "#eb6834",
    "rank_tail": "#1baf7a",
    "fixed": "#eda100",
    "adaptive": "#e87ba4",
    "tdvp2": "#6250d6",
}
MARKERS = ["o", "s", "^", "D"]


def family_of(policy):
    if policy in PILOT_POLICIES:
        return "rank_fast"
    if policy == STANDARD_POLICY:
        return "rank_adaptive"
    if policy in TAIL_POLICIES:
        return "rank_tail"
    if policy in FIXED_POLICIES:
        return "fixed"
    if policy == "tdvp2":
        return "tdvp2"
    return "adaptive"


def load(run_dirs):
    steps = pd.concat([pd.read_csv(Path(run) / "steps.csv") for run in run_dirs], ignore_index=True)
    bonds = pd.concat([pd.read_csv(Path(run) / "bonds.csv") for run in run_dirs], ignore_index=True)
    return steps, bonds


def summarise(steps, bonds):
    """One row per (L, basis_tol, policy, seed), then the mean over seeds (maxima for worst cases)."""
    keys = ["initial", "twoS", "L", "basis_tol", "policy", "seed"]
    per_seed = steps.groupby(keys).agg(
        sz_error_max=("sz_error", "max"),
        current_error_max=("current_error", "max"),
        energy_drift_max=("energy_drift", "max"),
        D_max=("D_max", "max"),
        wall_total=("wall_s", "sum"),
        h1=("h1", "sum"),
        h1_closure=("h1_closure", "sum"),
        root=("root", "sum"),
        stream_cols=("stream_cols", "sum"),
        dd_probes=("dd_probes", "sum"),
        dd_holdout=("dd_holdout", "sum"),
        dd_selected=("dd_selected", "sum"),
        n_krylov=("n_krylov", "sum"),
        unmet_steps=("budget_status", lambda status: int((status == "budget_unmet").sum())),
        cap_violations=("cap_violations", "sum"),
        steps=("step", "max"),
    )
    final = steps.sort_values("step").groupby(keys).tail(1).set_index(keys)
    per_seed["sz_error_final"] = final["sz_error"]
    per_seed["D_final"] = final["D_max"]
    per_seed["sketch_cols"] = per_seed["dd_probes"] + per_seed["dd_holdout"]
    entry_max = bonds.groupby(keys).agg(
        max_entry_probes=("n_probes", "max"),
        max_entry_holdout=("n_holdout", "max"),
        max_chi_old=("chi_old", "max"),
    )
    per_seed = per_seed.join(entry_max)
    worst_columns = ["sz_error_max", "current_error_max", "energy_drift_max", "D_max", "cap_violations",
                     "max_entry_probes", "max_entry_holdout", "max_chi_old", "unmet_steps"]
    grouped = per_seed.reset_index().groupby(["initial", "twoS", "L", "basis_tol", "policy"])
    summary = grouped.mean(numeric_only=True).drop(columns="seed")
    summary[worst_columns] = grouped[worst_columns].max()
    summary["seeds"] = grouped.size()
    summary = summary.reset_index()
    summary["family"] = summary["policy"].map(family_of)
    return summary


def verdict(passed):
    return "PASS" if passed else "FAIL"


def claims_report(summary, bonds):
    lines = ["# Claims of experiments/oldcap-fast-rsvd-xx at larger L", ""]
    lines += [f"Accuracy is 'no worse' within a factor {ACCURACY_SLACK} of the reference arm's worst sampled S^z error.", ""]
    for (initial, spin_twice, chain_length, basis_tol), setting in summary.groupby(["initial", "twoS", "L", "basis_tol"]):
        row = {policy: record for policy, record in setting.set_index("policy").iterrows()}
        lines += [f"## {initial}, d = {spin_twice + 1}, L = {chain_length}, tol_basis = {basis_tol:g}", ""]
        lines += ["| claim | arm | measured | verdict |", "|---|---|---|---|"]
        standard = row.get(STANDARD_POLICY)
        for policy in PILOT_POLICIES:
            if policy not in row or standard is None:
                continue
            pilot = row[policy]
            column_ratio = pilot["sketch_cols"] / standard["sketch_cols"]
            error_ratio = pilot["sz_error_max"] / standard["sz_error_max"]
            wall_ratio = pilot["wall_total"] / standard["wall_total"]
            lines.append(
                f"| C1 fewer sketch columns than rank_adaptive, no worse | {policy} | columns x{column_ratio:.2f}, "
                f"S^z error x{error_ratio:.2f}, wall x{wall_ratio:.2f} | "
                f"{verdict(column_ratio < 1 and error_ratio <= ACCURACY_SLACK)} |"
            )
            lines.append(
                f"| C2 at most {PROBE_CEILING} trial columns per bond | {policy} | max {pilot['max_entry_probes']:.0f} "
                f"(max chi_old {pilot['max_chi_old']:.0f}; rank_adaptive max {standard['max_entry_probes']:.0f}) | "
                f"{verdict(pilot['max_entry_probes'] <= PROBE_CEILING)} |"
            )
        for policy, width in TAIL_POLICIES.items():
            partner_name = TAIL_FIXED_PARTNER[policy]
            if policy not in row or partner_name not in row:
                continue
            tail, partner = row[policy], row[partner_name]
            tail_entries = bonds[(bonds.initial == initial) & (bonds.twoS == spin_twice) & (bonds.L == chain_length)
                                 & (bonds.basis_tol == basis_tol) & (bonds.policy == policy)]
            sketched = tail_entries[tail_entries.n_probes > 0]
            single_sketch = bool((sketched.n_probes <= width).all()) and int(tail_entries.n_holdout.sum()) == 0
            fewer_directions = tail["dd_selected"] < partner["dd_selected"]
            error_ratio = tail["sz_error_max"] / partner["sz_error_max"]
            lines.append(
                f"| C3 one {width}-column sketch, no holdout | {policy} | per-bond probes max {tail['max_entry_probes']:.0f}, "
                f"holdout {tail_entries.n_holdout.sum():.0f} | {verdict(single_sketch)} |"
            )
            lines.append(
                f"| C3 fewer kept directions than {partner_name}, accuracy kept | {policy} | selected "
                f"{tail['dd_selected']:.0f} vs {partner['dd_selected']:.0f}, S^z error x{error_ratio:.2f}, "
                f"unmet steps {tail['unmet_steps']:.0f} | {verdict(fewer_directions and error_ratio <= ACCURACY_SLACK)} |"
            )
        bug_arms = setting[setting.policy != "tdvp2"]
        violations = int(bug_arms["cap_violations"].sum())
        lines.append(f"| C4 per-bond cap 1.2 d chi_old | all BUG arms | {violations} frames over the cap | {verdict(violations == 0)} |")
        lines.append("")
    return "\n".join(lines) + "\n"


def style_of(summary):
    """Marker per policy: the order of first appearance inside its family."""
    styles = {}
    for family, members in summary.groupby("family"):
        for index, policy in enumerate(sorted(members.policy.unique())):
            styles[policy] = (FAMILY_STYLE[family], MARKERS[index % len(MARKERS)])
    return styles


def plot_accuracy_vs_cost(summary, cost_column, cost_label, path):
    panels = sorted(summary.groupby(["initial", "twoS", "L"]).groups)
    figure, axes = plt.subplots(1, len(panels), figsize=(5.5 * len(panels), 4.5), squeeze=False)
    styles = style_of(summary)
    for axis, (initial, spin_twice, chain_length) in zip(axes[0], panels):
        subset = summary[(summary.initial == initial) & (summary.twoS == spin_twice) & (summary.L == chain_length)]
        if cost_column != "wall_total":
            subset = subset[subset.policy != "tdvp2"]           # TDVP2 draws no DD sketch
        for policy, curve in subset.groupby("policy"):
            colour, marker = styles[policy]
            curve = curve.sort_values("basis_tol")
            axis.plot(curve[cost_column], curve["sz_error_max"], color=colour, marker=marker,
                      markersize=7, linewidth=1.5, label=policy)
        axis.set_xscale("log")
        axis.set_yscale("log")
        axis.set_title(f"{initial}, d = {spin_twice + 1}, L = {chain_length}")
        axis.set_xlabel(cost_label)
        axis.set_ylabel("worst sampled max |S^z error|")
        axis.grid(alpha=0.25, linewidth=0.6)
    axes[0][-1].legend(fontsize=7, loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    figure.tight_layout()
    figure.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(figure)


def plot_probes_vs_chi(bonds, path):
    shown = ["RSVD_pilot3_Kadapt", STANDARD_POLICY, "RSVD_tail12_factor01", "RSVD_fixed12rank8_Kadapt"]
    panels = sorted(bonds.groupby(["initial", "twoS", "L"]).groups)
    figure, axes = plt.subplots(1, len(panels), figsize=(5.5 * len(panels), 4.5), squeeze=False)
    for axis, (initial, spin_twice, chain_length) in zip(axes[0], panels):
        subset = bonds[(bonds.initial == initial) & (bonds.twoS == spin_twice) & (bonds.L == chain_length)
                       & (bonds.n_probes > 0)]
        for policy in shown:
            entries = subset[subset.policy == policy]
            if entries.empty:
                continue
            # Largest per-bond draw at each chi_old: the quantity the 64-column ceiling bounds.
            worst = entries.groupby("chi_old")["n_probes"].max()
            axis.plot(worst.index, worst.values, color=FAMILY_STYLE[family_of(policy)], marker="o",
                      markersize=4, linewidth=1.2, label=policy)
        axis.axhline(PROBE_CEILING, color="#6b6b6b", linewidth=1, linestyle="--")
        axis.text(axis.get_xlim()[0], PROBE_CEILING * 1.03, f" {PROBE_CEILING}-column ceiling", fontsize=7, color="#6b6b6b")
        axis.set_title(f"{initial}, d = {spin_twice + 1}, L = {chain_length}")
        axis.set_xlabel("bond dimension before the step, chi_old")
        axis.set_ylabel("largest DD sketch per bond-log entry")
        axis.grid(alpha=0.25, linewidth=0.6)
    axes[0][-1].legend(fontsize=7, loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    figure.tight_layout()
    figure.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(figure)


def main():
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    output_dir = Path(sys.argv[1])
    output_dir.mkdir(parents=True, exist_ok=True)
    steps, bonds = load(sys.argv[2:])
    summary = summarise(steps, bonds)
    summary.to_csv(output_dir / "summary.csv", index=False)
    (output_dir / "claims.md").write_text(claims_report(summary, bonds), encoding="utf-8")
    plot_accuracy_vs_cost(summary, "wall_total", "wall time over all steps (s)", output_dir / "error_vs_wall.png")
    plot_accuracy_vs_cost(summary, "sketch_cols", "DD sketch columns (discovery + holdout)",
                          output_dir / "error_vs_sketch_columns.png")
    plot_probes_vs_chi(bonds, output_dir / "probes_vs_chi.png")
    print((output_dir / "claims.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
