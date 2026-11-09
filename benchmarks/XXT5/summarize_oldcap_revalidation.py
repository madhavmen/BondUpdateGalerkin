#!/usr/bin/env python3
"""Summarize REAL Julia U(1) XX T=5 BUG2 runs; never synthesize measurements.

Exact cap: ceil(1.2 * d * D_b_old), applied to all DD/K-L and root frames.
Checks paired error & work for fixed4 depth8 vs fixed4 adaptive Lanczos.
Usage: python benchmarks/XXT5/summarize_oldcap_revalidation.py DIR
"""
import csv
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path

out = Path(sys.argv[1])
p = out / "trajectories.csv"
assert p.exists(), f"Missing numerical output {p}"
with p.open(newline="") as f:
    rows = list(csv.DictReader(f))
assert rows, "Empty Julia numerical output"
fixed = "T5_fixed4_k8"
adaptive = "T5_fixed4_adaptK"
required = {fixed, adaptive}
methods = set(r["method"] for r in rows)
assert required <= methods, f"Need both compared schemes, found {methods}"

def number(row, col):
    v = float(row[col])
    if not math.isfinite(v):
        raise AssertionError(f"Nonfinite {col}: {row}")
    return v

series = defaultdict(list)
for row in rows:
    key = (
        int(row["d"]), int(row["L"]), row["interaction"],
        float(row["basis_tol"]), int(row["seed"]), row["method"]
    )
    series[key].append(row)
assert len(series) > 0

per_seed = []
for key, records in sorted(series.items()):
    d, L, interaction, tol, seed, method = key
    assert records, key
    steps = sorted(int(r["step"]) for r in records)
    assert steps == list(range(1, len(steps) + 1)), f"Missing steps {key}"
    assert len(steps) == 50, f"Not a complete T=5 run: {key}"
    assert math.isclose(float(records[-1]["t"]), 5., abs_tol=1e-9), key
    for record in records:
        for name in ("h1", "h1_closure", "root", "dd_probes", "mpo_stream_cols", "D_final"):
            number(record, name)
    final = number(records[-1], "state_error")
    if method == "tdvp2":
        assert seed == 0
    per_seed.append(dict(
        d=d, L=L, interaction=interaction, basis_tol=tol, seed=seed,
        method=method, final_state_error=final,
        applications=sum(number(r, "h1") + number(r, "h1_closure") for r in records),
        h1=sum(number(r, "h1") for r in records),
        closure=sum(number(r, "h1_closure") for r in records),
        root=sum(number(r, "root") for r in records),
        dd_cols=sum(number(r, "mpo_stream_cols") for r in records),
        dd_probe=sum(number(r, "dd_probes")+number(r,"dd_holdout") for r in records),
        max_final_bond=max(number(r,"D_final") for r in records),
        max_aug_bond=max(number(r,"D_stage") for r in records),
        max_root_core=max(number(r,"root_core_elements") for r in records),
        wall_seconds=sum(number(r,"step_sec") for r in records),
        nonmet_steps=sum(r["budget_status"] != "met" for r in records),
    ))
def save_csv(dest, dataset):
    with (out / dest).open("w", newline="") as f:
        writer = csv.DictWriter(f, dataset[0].keys())
        writer.writeheader()
        writer.writerows(dataset)
save_csv("per_seed_oldcap.csv",per_seed)

kmap = {(v["d"],v["L"],v["interaction"],v["basis_tol"],v["seed"],v["method"]):v
         for v in per_seed}
group = defaultdict(list)
for val in per_seed:
    if val["method"] not in required:
        continue
    d, L, interaction, tol, seed = [val[n] for n in
         ("d", "L", "interaction", "basis_tol", "seed")]
    one = kmap[(d,L,interaction,tol,seed,fixed)]
    two = kmap[(d,L,interaction,tol,seed,adaptive)]
    ediff = abs(one["final_state_error"]-two["final_state_error"])
    # Matched achieved error means differences smaller than 1% or 1e-6
    # (whichever is larger), plus both variants reach the same final time.
    matched = ediff <= max(1.e-6,0.01*one["final_state_error"])
    item = dict(d=d,L=L,interaction=interaction,basis_tol=tol,seed=seed,
        matched_error=matched,
        fixed_error=one["final_state_error"],
        adaptive_error=two["final_state_error"],
        absolute_error_difference=ediff,
        fixed_KL_app=one["applications"],
        adaptive_KL_app=two["applications"],
        KL_saving_fraction=1-two["applications"]/one["applications"],
        fixed_DD_cols=one["dd_cols"],adaptive_DD_cols=two["dd_cols"],
        fixed_root_app=one["root"],adaptive_root_app=two["root"],
        fixed_max_bond=one["max_final_bond"],adaptive_max_bond=two["max_final_bond"],
        fixed_max_aug=one["max_aug_bond"],adaptive_max_aug=two["max_aug_bond"],
        fixed_nonmet=one["nonmet_steps"],adaptive_nonmet=two["nonmet_steps"],
        fixed_wall_s=one["wall_seconds"],adaptive_wall_s=two["wall_seconds"])
    group[(d,L,interaction,tol)].append(item)
paired = [r for data in group.values() for r in data]
save_csv("paired_oldcap.csv",paired)

avg = statistics.mean
summary = []
for (d,L,interaction,tol), vals in sorted(group.items()):
    summary.append(dict(
        d=d,L=L,interaction=interaction,basis_tol=tol,seeds=len(vals),
        all_seeds_matched=all(x["matched_error"] for x in vals),
        fixed_KL_app=avg(x["fixed_KL_app"] for x in vals),
        adaptive_KL_app=avg(x["adaptive_KL_app"] for x in vals),
        saving_percent=100*(1-avg(x["adaptive_KL_app"] for x in vals)/
                              avg(x["fixed_KL_app"] for x in vals)),
        fixed_error=avg(x["fixed_error"] for x in vals),
        adaptive_error=avg(x["adaptive_error"] for x in vals),
        max_abs_error_difference=max(x["absolute_error_difference"] for x in vals),
        fixed_DD_cols=avg(x["fixed_DD_cols"] for x in vals),
        adaptive_DD_cols=avg(x["adaptive_DD_cols"] for x in vals),
        fixed_root_app=avg(x["fixed_root_app"] for x in vals),
        adaptive_root_app=avg(x["adaptive_root_app"] for x in vals),
        fixed_max_bond=max(x["fixed_max_bond"] for x in vals),
        adaptive_max_bond=max(x["adaptive_max_bond"] for x in vals),
        fixed_max_aug=max(x["fixed_max_aug"] for x in vals),
        adaptive_max_aug=max(x["adaptive_max_aug"] for x in vals),
        fixed_wall_s=avg(x["fixed_wall_s"] for x in vals),
        adaptive_wall_s=avg(x["adaptive_wall_s"] for x in vals),
        fixed_nonmet=max(x["fixed_nonmet"] for x in vals),
        adaptive_nonmet=max(x["adaptive_nonmet"] for x in vals),
    ))
save_csv("summary_oldcap.csv",summary)
text = [
    "# T=5 Julia BUG2 revalidation: original physical-dimension-scaled cap",
    "",
    "Each bond has a fixed working-rank MAXIMUM over this step:",
    "`min(maxdim,ceil(1.2*d*D_b_old))`. All BUG methods: order2, checked",
    "global closure, exact U(1) sector Hamiltonian benchmark, DD fixed",
    "sketch width8, retained rank4. Only Krylov stopping differs.",
    "",
    "| d | L | XX | tol_basis | fixed8 H | adaptive H | saving | paired error match | fixed error | adaptive error |",
    "|---:|---:|---|---:|---:|---:|---:|---|---:|---:|",
]
for v in summary:
    text.append(
        f'| {v["d"]} | {v["L"]} | {v["interaction"]} | {v["basis_tol"]:.1e}'
        f' | {v["fixed_KL_app"]:.1f} | {v["adaptive_KL_app"]:.1f}'
        f' | {v["saving_percent"]:.1f}% | {v["all_seeds_matched"]}'
        f' | {v["fixed_error"]:.6g} | {v["adaptive_error"]:.6g} |'
    )
text += [
    "",
    "**Counts:** one-site Hamiltonian applications plus logged closure one-site",
    "applications, reported separately from DD columns and root H actions.",
    "This metric is NOT an unweighted sum of MPO and local operator work.",
    "Savings in H applications do not prove CPU time speedup.",
    "",
    f"Compared {len(paired)} independent seed/tolerance/model pairs.",
    f"Matched pairs: {sum(p['matched_error'] for p in paired)}/{len(paired)}.",
    "",
    "Results generated from full Julia trajectories, not synthetic data.",
]
(out / "report_oldcap.md").write_text("\n".join(text)+"\n")
print("\n".join(text))
