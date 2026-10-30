"""WP0 inventory of BUG-Julia b155550 -> CSV with one disposition per file.

usage: python -I inventory.py PLAN.md TREE OUT.csv BUG_JULIA_CHECKOUT
TREE holds src/ tests/ test/ examples/ benchmarks/ of b155550; scripts/ and notebooks/ are listed
from the checkout's git tree.
"""
import csv
import re
import subprocess
import sys
from pathlib import Path

plan, tree, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
sec8 = plan.read_text(encoding="utf-8")
sec8 = sec8[sec8.index("## 8. Explicit"):sec8.index("## 9.")]

# src: rows of plan section 8 (8.1-8.4)
SRCMAP = {}
for line in sec8.splitlines():
    if not line.startswith("| `"):
        continue
    legacy, new, treat = [c.strip() for c in line.split("|")[1:4]]
    for b in re.findall(r"`([\w/]+\.jl)`", legacy):
        SRCMAP[Path(b).name] = (new, treat)

def src_kind(rel):
    if "/baselines/" in rel:
        return "baseline"
    if rel.endswith(("kagome.jl", "breathing_kagome.jl", "square_cylinder.jl",
                     "triangular_cylinder.jl", "haldane_shastry.jl")):
        return "study"
    if "/models/" in rel or "/qtt/" in rel or rel.endswith("quantum_jumps.jl"):
        return "domain"
    return "core"

# tests: by tested behaviour, mirroring the new src/
T = "test/"
TESTMAP = {
    "BondUpdateBUG/test_symmetric_mps.jl": T + "BUG/Core/TensorNetworks/test_symmetric_mps.jl",
    "BondUpdateBUG/test_frame.jl": T + "BUG/Core/Sweeps/test_frames.jl",
    "BondUpdateBUG/test_expv.jl": T + "BUG/Core/LocalSolvers/test_krylov.jl",
    "BondUpdateBUG/test_gates.jl": T + "BUG/Core/TensorNetworks/test_mpo.jl (apply_gate) + QMB/Tooling/test_gates.jl",
    "BondUpdateBUG/test_sectors.jl": T + "BUG/Core/TensorNetworks/test_sectors.jl",
    "BondUpdateBUG/test_fusion_cache.jl": T + "BUG/Core/TensorNetworks/test_sectors.jl",
    "BondUpdateBUG/test_z2_ising.jl": T + "QMB/ClosedSystem/test_z2_ising.jl",
    "sweeps/test_adaptive_api.jl": T + "BUG/test_solve_options.jl",
    "sweeps/test_adaptive_rsvd.jl": T + "BUG/Randomised/test_cbe.jl",
    "sweeps/test_adaptive_search.jl": T + "BUG/Randomised/test_cbe.jl",
    "sweeps/test_dd_cbe.jl": T + "BUG/Randomised/test_dd_cbe.jl",
    "sweeps/test_dd_cbe_controller.jl": T + "BUG/Randomised/test_dd_cbe.jl",
    "sweeps/test_hq_kernels.jl": T + "BUG/Randomised/test_dd_cbe.jl",
    "sweeps/test_adaptive_krylov.jl": T + "BUG/Core/RankAdaptation/test_krylov_enrichment.jl",
    "sweeps/test_krylov_closure.jl": T + "BUG/Core/RankAdaptation/test_krylov_enrichment.jl",
    "sweeps/test_krylov_closure_steps.jl": T + "BUG/Core/RankAdaptation/test_krylov_enrichment.jl",
    "sweeps/test_adaptive_compression.jl": T + "BUG/Core/RankAdaptation/test_truncation.jl",
    "sweeps/test_adaptive_expv.jl": T + "BUG/Core/LocalSolvers/test_krylov.jl",
    "sweeps/test_krylov_recycle.jl": T + "BUG/Core/LocalSolvers/test_krylov_recycle.jl",
    "sweeps/test_hq_root.jl": T + "BUG/Core/LocalSolvers/test_root_methods.jl",
    "sweeps/test_adaptive_diagnostics.jl": T + "BUG/Core/test_diagnostics.jl",
    "sweeps/test_env_inplace.jl": T + "BUG/Core/Environments/test_environment_stacks.jl",
    "sweeps/test_parallel_kls.jl": T + "BUG/Core/Sweeps/test_parallel_kls.jl",
    "sweeps/test_adaptive_parallel.jl": T + "BUG/TimeEvolution/test_parallel.jl",
    "sweeps/test_sweeps.jl": T + "BUG/TimeEvolution/test_bug_step.jl (BUG parts) + Baselines (DMRG/TDVP parts)",
    "sweeps/test_imaginary_time.jl": T + "BUG/ITE/test_ite_step.jl",
    "sweeps/test_adaptive_dmrg.jl": T + "Baselines/test_dmrg.jl",
    "sweeps/test_tdvp2_order.jl": T + "Baselines/test_tdvp2.jl",
    "sweeps/test_tdvp2_truncation.jl": T + "Baselines/test_tdvp2.jl",
    "sweeps/test_pair_mpo.jl": T + "QMB/Tooling/test_pair_mpo.jl",
    "sweeps/test_tfim_mpo.jl": T + "QMB/ClosedSystem/test_tfim.jl",
    "sweeps/test_spin_xx.jl": T + "QMB/ClosedSystem/test_xx.jl",
    "sweeps/test_spin_jw.jl": T + "QMB/ClosedSystem/test_xx.jl",
    "sweeps/test_cbe_bug_exact_refs.jl": T + "QMB/ClosedSystem/test_exact_refs.jl",
    "sweeps/test_symmetry_parity.jl": T + "QMB/ClosedSystem/test_symmetry_parity.jl",
    "sweeps/test_physics.jl": T + "QMB/ClosedSystem/test_heisenberg.jl (Haldane-Shastry arm -> studies/)",
    "sweeps/test_oat.jl": T + "QMB/ClosedSystem/test_oat.jl",
    "sweeps/test_xx_efficiency.jl": T + "QMB/ClosedSystem/test_xx.jl",
    "sweeps/test_square_cylinder.jl": "studies/ (with square_cylinder.jl)",
    "sweeps/test_kagome.jl": "studies/ (with kagome.jl)",
    "sweeps/test_liouville.jl": T + "QMB/OpenSystem/MPDO/test_liouville.jl",
    "sweeps/test_dissipative_t1.jl": T + "QMB/OpenSystem/MCWF/test_no_jump.jl",
    "sweeps/test_dissipative_t2.jl": T + "QMB/OpenSystem/MCWF/test_no_jump.jl",
    "sweeps/test_dissipative_t2_g12.jl": T + "QMB/OpenSystem/MCWF/test_no_jump.jl",
    "sweeps/test_dissipative_u18.jl": T + "QMB/OpenSystem/MCWF/test_no_jump.jl",
    "sweeps/test_dissipative_t3_a1.jl": T + "QMB/OpenSystem/MCWF/test_models.jl",
    "sweeps/test_dissipative_t3_a2.jl": T + "QMB/OpenSystem/MCWF/test_models.jl",
    "sweeps/test_dissipative_t4.jl": T + "QMB/OpenSystem/MCWF/test_quantum_jumps.jl",
    "sweeps/test_dissipative_t4_b.jl": T + "QMB/OpenSystem/MCWF/test_quantum_jumps.jl",
    "sweeps/test_dissipative_t4_conv.jl": T + "QMB/OpenSystem/MCWF/test_quantum_jumps.jl",
    "sweeps/test_dissipative_t5.jl": T + "QMB/OpenSystem/MCWF/test_quantum_jumps.jl",
    "sweeps/test_dissipative_t6.jl": T + "QMB/OpenSystem/MCWF/test_quantum_jumps.jl",
    "sweeps/test_qtt_task1.jl": T + "QTTS/ (Tooling, Elementwise and BUG-on-QTT gates, split by tested file)",
    "sweeps/test_qtt_task2.jl": T + "QTTS/ + BUG/TimeEvolution/test_stage_solvers.jl",
    "sweeps/runtests.jl": T + "runtests.jl",
    "runtests.jl": T + "runtests.jl",
    "submit_full_test_suite.slurm": "not copied (CI runs the suite)",
}
COMMON_DROP = {"test_analytic_reference.jl", "test_dense_reference.jl", "test_exact_sparse.jl"}

EXAMPLES = {
    "README.md": ("examples/README.md", "example"),
    "common.jl": ("examples/QMB/common.jl", "example"),
    "adaptive_quickstart.jl": ("examples/QMB/xx_quickstart.jl", "example"),
    "xx_domain_wall.jl": ("examples/QMB/xx_domain_wall.jl", "example"),
    "tfim_z2.jl": ("examples/QMB/tfim_z2.jl", "example"),
    "oat_z2.jl": ("examples/QMB/oat_z2.jl", "example"),
    "observables_demo.jl": ("examples/QMB/observables_demo.jl", "example"),
    "plot_examples.py": ("archive tag", "obsolete"),
}

def bench(rel):
    name = Path(rel).name
    parts = Path(rel).parts
    sub = parts[1] if len(parts) > 2 else ""
    if name in ("exact_sparse.jl", "hubbard_aa_exact.jl") or rel.endswith("open_system/sparse_lindblad.jl"):
        return "core", "test/common/" + name, "Move (independent reference included by tests)"
    if sub == "xx_efficiency":
        return "study", "benchmarks/QMB/ClosedSystem/xx_efficiency/", "Curate"
    if sub == "open_system":
        return "study", "benchmarks/QMB/OpenSystem/MPDO/", "Curate"
    if sub == "qtt":
        return "study", "benchmarks/QTTS/", "Curate"
    if sub in ("ground_state", "imaginary_time"):
        return "study", "benchmarks/QMB/ClosedSystem/ or studies/", "Curate"
    if sub == "spectral":
        return "study", "studies/", "Archive tag unless promoted"
    if name.startswith("dissipative_") or name in ("lindblad_common.jl", "fused_lindblad.jl"):
        return "study", "benchmarks/QMB/OpenSystem/ or studies/", "Curate"
    if re.search(r"(_diag|_probe|_check|_smoke)\d*\.jl$", name):
        return "obsolete", "archive tag", "Not copied (one-off probe)"
    if name.endswith(".py"):
        return "study", "studies/ or archive tag", "Curate with its driver"
    if name.endswith((".ps1", ".sh", ".slurm", ".txt")):
        return "obsolete", "archive tag", "Not copied (local/cluster wrapper or retired note)"
    if "profile" in name or "microbench" in name or name == "alloc_attribution.jl":
        return "study", "benchmarks/kernels/", "Curate"
    return "unknown", "WP8 review", "Unclassified research driver: keep, move to studies/ or archive, decided in WP8"

def scripts(rel):
    name = Path(rel).name
    if name in ("install_deps.jl",):
        return "obsolete", "not copied", "Superseded by Project.toml + README install"
    if name.endswith(".sbatch"):
        return "study", "studies/runners/", "Only if a kept study needs it"
    if name == "smoke_examples.jl":
        return "core", "test/examples (WP7 examples check)", "Port"
    return "obsolete", "archive tag", "Not copied (diagnostic probe)"

def refs(path):
    try:
        s = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""
    mods = [m for m in ("BondUpdateBUG", "RSVDCBEBondUpdate", "Telum", "LurCGT") if m in s]
    return " ".join(mods)

rows = []
root = tree
for p in sorted((root / "src").rglob("*.jl")):
    rel = p.relative_to(root / "src").as_posix()
    if rel == "BUGJulia.jl":
        rows.append(("src/" + rel, "src", "core", "src/BondUpdateGalerkin.jl", "Replaced (module root; the precompile workload is rebuilt with PrecompileTools later)", refs(p)))
        continue
    new, treat = SRCMAP.get(Path(rel).name, ("review", "MISSING FROM SECTION 8"))
    rows.append(("src/" + rel, "src", src_kind(rel), new.replace("`", ""), treat, refs(p)))
for d in ("tests", "test"):
    for p in sorted((root / d).rglob("*")):
        if not p.is_file():
            continue
        rel = p.relative_to(root / d).as_posix()
        if d == "test":
            tgt, kind = "test/BUG/Core/TensorNetworks/ (Telum kernel checks)", "core"
        elif rel.startswith("common/"):
            tgt = "test/common/" + Path(rel).name
            kind = "core"
        else:
            tgt = TESTMAP.get(rel, "review")
            kind = "study" if tgt.startswith("studies") else "core"
        rows.append((f"{d}/{rel}", d, kind, tgt, "Move, grouped by tested behaviour", refs(p)))
for p in sorted((root / "examples").rglob("*")):
    if p.is_file():
        tgt, kind = EXAMPLES.get(p.name, ("review", "unknown"))
        rows.append(("examples/" + p.name, "examples", kind, tgt, "Verify and port (WP7)", refs(p)))
for p in sorted((root / "benchmarks").rglob("*")):
    if p.is_file():
        rel = p.relative_to(root).as_posix()
        kind, tgt, treat = bench(rel)
        rows.append((rel, "benchmarks", kind, tgt, treat, refs(p)))
git = ["git", "-C", sys.argv[4], "ls-tree", "-r", "--name-only", "b155550"]
for rel in subprocess.run(git + ["scripts", "notebooks"], capture_output=True, text=True).stdout.split():
    if rel.startswith("scripts/"):
        kind, tgt, treat = scripts(rel)
    else:
        kind, tgt, treat = "study", "docs/ tutorials (WP7)", "Rewrite against the new API; one source format"
    rows.append((rel, rel.split("/")[0], kind, tgt, treat, ""))

with out.open("w", newline="", encoding="utf-8") as f:
    w = csv.writer(f, lineterminator="\n")
    w.writerow(["path", "area", "kind", "target", "treatment", "references"])
    w.writerows(rows)

from collections import Counter
print(len(rows), "rows")
print(Counter((r[1], r[2]) for r in rows))
for r in rows:
    if r[3] == "review" or "MISSING" in r[4]:
        print("UNMAPPED", r[0])
