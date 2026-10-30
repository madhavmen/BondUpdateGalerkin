"""Check plan section 8 against BUG-Julia b155550: file coverage and named identifiers.

usage: python -I check_map.py PLAN.md TREE
"""
import re
import sys
from pathlib import Path

plan, tree = Path(sys.argv[1]), Path(sys.argv[2])
text = plan.read_text(encoding="utf-8")
sec8 = text[text.index("## 8. Explicit"):text.index("## 9.")]
src = tree / "src"
files = sorted(p.relative_to(src).as_posix() for p in src.rglob("*.jl"))

# 1. coverage: every legacy src file must be named in section 8
print("== src files not named in section 8")
for f in files:
    if Path(f).name not in sec8:
        print("  ", f)

# 2. identifiers: each backticked name in a row must be defined in that row's legacy files
DEF = (r"(?m)^\s*(?:(?:function\s+|(?:mutable\s+)?struct\s+|abstract\s+type\s+|const\s+|macro\s+)"
       r"(?:\w+\.)?{n}(?![\w!])"
       r"|(?:\w+\.)?{n}(?:\{{[^}}]*\}})?\([^\n]*\)(?:\s*::\s*\S+)?(?:\s*where[^=\n]*)?\s*=(?!=))")
name_to_path = {}
for f in files:
    name_to_path.setdefault(Path(f).name, []).append(f)

print("== identifiers named in a row but not defined in its legacy files")
for line in sec8.splitlines():
    if not line.startswith("| `"):
        continue
    cells = line.split("|")
    legacy, new = cells[1], cells[2]
    lfiles = []
    for b in re.findall(r"`([\w/]+\.jl)`", legacy):
        lfiles += name_to_path.get(Path(b).name, [])
    if not lfiles:
        continue
    body = "\n".join((src / f).read_text(encoding="utf-8", errors="replace") for f in lfiles)
    for ident in re.findall(r"`([A-Za-z_][\w!]*)`", new + legacy):
        if ident.endswith(".jl") or "/" in ident:
            continue
        if not re.search(DEF.format(n=re.escape(ident)), body):
            print(f"   {ident:28s} not defined in {', '.join(lfiles)}")

# 3. definitions of explicitly moved names that also live elsewhere (duplicates)
print("== section 8 identifiers defined in more than one src file")
idents = set(re.findall(r"`([A-Za-z_][\w!]*)`", sec8)) - {"init"}
for ident in sorted(idents):
    hits = [f for f in files
            if re.search(DEF.format(n=re.escape(ident)),
                         (src / f).read_text(encoding="utf-8", errors="replace"))]
    if len(hits) > 1:
        print(f"   {ident:28s} {', '.join(hits)}")
