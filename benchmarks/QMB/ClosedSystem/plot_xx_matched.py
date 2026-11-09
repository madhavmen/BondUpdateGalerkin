"""Plot matched-accuracy work and frame-growth diagnostics from the higher-d XX Julia runner.

Usage:
  python benchmarks/QMB/ClosedSystem/plot_xx_matched.py results/xx-higher-matched

The expensive x-axes are kept separate: one-site K/L and closure applications,
zero-site root actions, DD MPO sketch columns, and wall time. These operations
have different costs, so no uncalibrated weighted 'total applications' is used.
"""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

out = Path(sys.argv[1] if len(sys.argv)>1 else "results/xx-higher-matched")
d = pd.read_csv(out/"trajectories.csv")
cols = ["d","L","interaction","basis_tol","method","seed"]
records = []
for key, runs in d.groupby(cols):
    runs = runs.sort_values("step")
    end = runs.iloc[-1]
    records.append(dict(zip(cols,key)) | {
        "state_error":end.state_error,
        "profile_error":end.profile_error,
        "h1":int((runs.h1+runs.h1_closure).sum()),
        "root_actions":int((runs.root+runs.root_frontier).sum()),
        "sketch_columns":int(runs.mpo_stream_cols.sum()),
        "sketch_calls":int(runs.mpo_stream_calls.sum()),
        "wall_sec":float(runs.step_sec.sum()),
        "max_aug":int(runs.D_stage.max()),
        "max_final":int(runs.D_final.max()),
        "max_root_elements":int(runs.root_core_elements.max()),
        "unmet_steps":int((~runs.budget_status.isin(["met","uncertified","reference"])).sum())
    })
s = pd.DataFrame(records)
s.to_csv(out/"run_summary.csv",index=False)

# Each tolerance qualifies only if ALL tested sketch seeds meet the target.
rows=[]
for key,block in s.groupby(["d","L","interaction","method"]):
    for target in (1e-2,3e-3,1e-3,3e-4,1e-4,3e-5,1e-5):
        candidate=[]
        for tol,group in block.groupby("basis_tol"):
            if group.state_error.max()<=target:
                candidate.append({"basis_tol":tol,
                    "h1":group.h1.mean(),"root":group.root_actions.mean(),
                    "sketch":group.sketch_columns.mean(),
                    "wall_sec":group.wall_sec.mean(),
                    "worst_error":group.state_error.max(),
                    "mean_aug":group.max_aug.mean(),
                    "mean_final":group.max_final.mean(),
                    "nseeds":group.seed.nunique()})
        if candidate:
            # Select K/L-efficient configuration at the matched target;
            # also report independently best measured wall time.
            best=min(candidate,key=lambda a:(a["h1"],a["wall_sec"]))
            fastest=min(candidate,key=lambda a:a["wall_sec"])
            rows.append(dict(zip(["d","L","interaction","method"],key)) |
                {"target":target,"qualified":True,**best,
                 "fastest_wall":fastest["wall_sec"]})
        else:
            rows.append(dict(zip(["d","L","interaction","method"],key)) |
                        {"target":target,"qualified":False})
pd.DataFrame(rows).to_csv(out/"matched_error_targets.csv",index=False)

dd=sorted(d.d.unique()); interactions=sorted(d.interaction.unique())
colors=plt.rcParams["axes.prop_cycle"].by_key()["color"]
for cost,filename,xlabel in [
    ("h1","error_vs_KL_applications.png","Total one-site K/L + closure H applications"),
    ("wall_sec","error_vs_walltime.png","Total measured wall time (seconds)"),
    ("sketch_columns","error_vs_DD_probe_columns.png","DD MPO sketch columns"),
]:
    fig,axs=plt.subplots(len(dd),len(interactions),figsize=(12,7),
                         squeeze=False,layout="constrained")
    for i,dim in enumerate(dd):
      for j,interaction in enumerate(interactions):
        ax=axs[i,j]
        z=s[(s.d==dim)&(s.interaction==interaction)]
        methods=sorted(z.method.unique())
        for m,method in enumerate(methods):
            if cost!="wall_sec" and method=="tdvp2":
                continue # unavailable H/sketch counts for TDVP2
            group=z[z.method==method].groupby("basis_tol").agg(
                cost=(cost,"mean"),error=("state_error","mean"),
                low=("state_error","min"),high=("state_error","max")
            ).sort_values("cost")
            ax.plot(group.cost,np.maximum(group.error,1e-15),"o-",
                    ms=3,label=method,color=colors[m%len(colors)])
            ax.vlines(group.cost,np.maximum(group.low,1e-15),
                np.maximum(group.high,1e-15),
                color=colors[m%len(colors)],alpha=.35)
        ax.set(title=f"d={dim}, {interaction}-range",xlabel=xlabel,
               ylabel="State error sqrt(1-F)",yscale="log")
        ax.grid(alpha=.2)
    handles,labels=axs[0,0].get_legend_handles_labels()
    if handles:fig.legend(handles,labels,loc="outside lower center",ncol=4,fontsize=7)
    fig.savefig(out/filename,dpi=190)
    plt.close(fig)

fig,axs=plt.subplots(len(dd),len(interactions),figsize=(12,7),
                     squeeze=False,layout="constrained")
for i,dim in enumerate(dd):
  for j,interaction in enumerate(interactions):
    ax=axs[i,j]
    sub=d[(d.d==dim)&(d.interaction==interaction)&(d.seed==0)]
    tol=max(sub.basis_tol.unique())
    sub=sub[np.isclose(sub.basis_tol,tol)]
    for method in ("adaptive_lanczos","fixed4_depth8","tdvp2"):
        z=sub[sub.method==method].sort_values("t")
        if z.empty:continue
        ax.plot(z.t,z.D_final,label=method+": final")
        if method!="tdvp2":
            ax.plot(z.t,z.D_stage,"--",label=method+": augmented")
            if method=="adaptive_lanczos":
                ax.plot(z.t,z.D_root_left,":",label="root left W")
                ax.plot(z.t,z.D_root_right,"-.",label="root right Z")
    ax.set(title=f"d={dim}, {interaction}-range, tol={tol:g}",
           xlabel="Time",ylabel="Bond dimension")
    ax.grid(alpha=.2)
handles,labels=axs[0,0].get_legend_handles_labels()
if handles:fig.legend(handles,labels,loc="outside lower center",ncol=3,fontsize=7)
fig.savefig(out/"root_augmented_final_bond_growth.png",dpi=190)
plt.close(fig)
print("Wrote matched-error, operator-work and root-rank outputs to",out)
