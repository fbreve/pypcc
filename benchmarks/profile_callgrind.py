"""Build both kernels and collect Callgrind instruction/branch counts on Linux."""
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "callgrind-results"
OUT.mkdir(exist_ok=True)
SOURCE = ROOT / "pcc_step.pyx"
ORIGINAL = SOURCE.read_bytes()
BASE = "10313ab01207d04ccf3b556bd7701d941b87c020"

def run(args):
    subprocess.run(args, cwd=ROOT, check=True)

try:
    for variant in ("baseline", "fused"):
        source = subprocess.run(["git", "show", f"{BASE}:pcc_step.pyx"], cwd=ROOT, check=True, capture_output=True).stdout if variant == "baseline" else ORIGINAL
        SOURCE.write_bytes(source)
        run([sys.executable, "setup.py", "build_ext", "--inplace"])
        raw = OUT / f"{variant}.callgrind"
        run(["valgrind", "--tool=callgrind", "--branch-sim=yes",
             "--cache-sim=no", f"--callgrind-out-file={raw}",
             sys.executable, "benchmarks/profile_kernel_workload.py"])
finally:
    SOURCE.write_bytes(ORIGINAL)



def function_self_counts(path):
    """Aggregate exclusive event costs by Callgrind function ID.

    Call records and their cost lines are deliberately excluded: including
    them would double-count work already attributed to called functions.
    """
    events = []
    names = {}
    counts = {}
    current = None
    in_call = False
    for line in path.read_text().splitlines():
        if line.startswith("events: "):
            events = line.split()[1:]
        elif line.startswith("fn="):
            token = line[3:].strip()
            if token.startswith("("):
                number, _, name = token.partition(")")
                current = number[1:]
                if name.strip():
                    names[current] = name.strip()
            else:
                current = token
            in_call = False
        elif line.startswith("calls="):
            in_call = True
        elif line.startswith(("fl=", "fi=", "ob=", "cob=", "cfi=")):
            # Source/object metadata does not change the active function.
            pass
        elif line and (line[0].isdigit() or line[0] in "+-*") and current is not None:
            if in_call:
                in_call = False
                continue
            values = line.split()
            if len(values) >= 2:
                try:
                    numbers = [int(x) for x in values[1:] if x.lstrip("+-").isdigit()]
                    numbers += [0] * (len(events) - len(numbers))
                except ValueError:
                    continue
                dest = counts.setdefault(current, [0] * len(events))
                for i, value in enumerate(numbers):
                    dest[i] += value
    return [
        {"function": names.get(fid, fid),
         **dict(zip(events, values))}
        for fid, values in counts.items()
        if "pcc_step" in names.get(fid, "").lower()
    ]

summary = {}
for variant in ("baseline", "fused"):
    lines = (OUT / f"{variant}.callgrind").read_text().splitlines()
    events = next(line.split()[1:] for line in lines if line.startswith("events: "))
    totals = next(line.split()[1:] for line in lines if line.startswith("summary: "))
    summary[variant] = dict(zip(events, map(int, totals)))
    summary[variant]["cython_functions"] = function_self_counts(OUT / f"{variant}.callgrind")
    summary[variant]["attribution_note"] = "Provisional: Callgrind compressed event columns (*) and delta encoding are not fully decoded; do not interpret function counts as exact."
    annotated = subprocess.run(
        ["callgrind_annotate", "--inclusive=yes", "--threshold=0",
         str(OUT / f"{variant}.callgrind")],
        cwd=ROOT, check=True, text=True, capture_output=True)
    (OUT / f"{variant}.annotated.txt").write_text(annotated.stdout)
    kernel_lines = [line for line in annotated.stdout.splitlines()
                    if "pcc_step" in line.lower() or "__pyx" in line.lower()]
    (OUT / f"{variant}.annotation-head.txt").write_text(
        "\\n".join(annotated.stdout.splitlines()[:120]))
    summary[variant]["kernel_lines"] = kernel_lines[:20]
    print(variant, "kernel attribution", *kernel_lines[:10], sep="\\n")
print(json.dumps(summary, indent=2))
(OUT / "summary.json").write_text(json.dumps(summary, indent=2))
