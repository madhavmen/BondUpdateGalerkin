#!/usr/bin/env python3
"""Plot the Julia XX CBE-BUG stopping benchmark. Requires numpy, pandas, matplotlib.
Usage: python benchmarks/QMB/ClosedSystem/plot_xx_adaptive.py results/xx-adaptive
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

out = Path(sys.argv[1] if len(sys.argv) > 1 else "results/xx-adaptive")
tr = pd.read_csv(out / "trajectories.csv")
fr = pd.read_csv(out / "frame_ranks.csv")
pr = pd.read_csv(out / "profiles.csv")

for case in sorted(tr.interaction.unique()):
    sub = tr[tr.interaction == case]
    # Final and augmented frames shown separately: augmented rank determines BUG root work.
    fig, ax = plt.subplots(figsize=(7.8, 4.5))
    for method in sorted(sub.method.unique()):
        x = sub[(sub.method == method) & (sub.seed == 0)]
        ax.plot(x.t, x.D_final, label=method + " final", linewidth=1.8)
        if method != "tdvp2":
            ax.plot(x.t, x.D_stage, ls="--", alpha=.7, label=method + " frame")
    ax.set(xlabel="Time", ylabel="Maximum bond rank", title=case+": augmented vs final")
    ax.grid(alpha=.2)
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(out / ("bond_growth_" + case + ".png"), dpi=200)
    plt.close(fig)

    # Spatial frame growth: one image per method, so sector fronts are not hidden
    # by a maximum over bond cuts.
    for method in sorted(sub.method.unique()):
        if method == "tdvp2":
            continue
        f = fr[(fr.interaction == case) & (fr.method == method) & (fr.seed == 0)]
        mat = f.pivot(index="bond", columns="t", values="D_frame")
        fig, ax = plt.subplots(figsize=(7.8, 4.0))
        im = ax.imshow(mat, origin="lower", aspect="auto",
                       extent=(mat.columns.min(),mat.columns.max(),
                               mat.index.min()-.5,mat.index.max()+.5),
                       interpolation="nearest")
        fig.colorbar(im, ax=ax, label="Augmented frame rank")
        ax.set(xlabel="Time", ylabel="Bond cut", title=case+" / "+method)
        fig.tight_layout()
        fig.savefig(out / ("frames_" + case + "_" + method + ".png"), dpi=200)
        plt.close(fig)

    for method in sorted(sub.method.unique()):
        p = pr[(pr.interaction == case) & (pr.method == method) & (pr.seed == 0)]
        mat = p.pivot(index="site", columns="t", values="sz")
        fig, ax = plt.subplots(figsize=(7.8, 4.0))
        im = ax.imshow(mat, origin="upper", aspect="auto", cmap="coolwarm",
                       vmin=-.5, vmax=.5,
                       extent=(mat.columns.min(),mat.columns.max(),
                               mat.index.max()+.5,mat.index.min()-.5))
        fig.colorbar(im, ax=ax, label="S^z")
        ax.set(xlabel="Time", ylabel="Site", title=case+" / "+method)
        fig.tight_layout()
        fig.savefig(out / ("lightcone_" + case + "_" + method + ".png"), dpi=200)
        plt.close(fig)

# A fair Pareto plot must use the achieved observable error, not merely the
# selected tolerance or number of half-sweeps.
fig, ax = plt.subplots(figsize=(7.8, 4.5))
last = tr.sort_values("step").groupby(["interaction","method","seed"]).tail(1)
for (case, method), group in last.groupby(["interaction","method"]):
    xx = np.maximum(group.max_sz_error.to_numpy(dtype=float),1e-15)
    yy = group.t_step.to_numpy(dtype=float)
    mask = np.isfinite(xx) & np.isfinite(yy) & (yy > 0)
    if np.any(mask):
        ax.scatter(xx[mask], yy[mask],label=case+" "+method)
ax.set(xscale="log",yscale="log",xlabel="Maximum site magnetization error",
       ylabel="Last step wall time (s)",title="Error vs measured step cost")
ax.legend(fontsize=7)
fig.tight_layout()
fig.savefig(out / "accuracy_vs_step_cost.png", dpi=200)
plt.close(fig)
