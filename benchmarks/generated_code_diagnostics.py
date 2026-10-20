"""Capture Cython generated C and compiler metadata for kernel investigation."""
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = "10313ab01207d04ccf3b556bd7701d941b87c020"
SOURCE = ROOT / "pcc_step.pyx"
ORIGINAL = SOURCE.read_bytes()
OUT = ROOT / "generated-code-diagnostics"
OUT.mkdir(exist_ok=True)

def command(args):
    p = subprocess.run(args, cwd=ROOT, text=True, capture_output=True)
    if p.returncode:
        raise RuntimeError(f"{args}: {p.stdout}\n{p.stderr}")
    return p.stdout.strip()

metadata = {"platform": platform.platform(), "python": sys.version}
try:
    import Cython
    metadata["cython"] = Cython.__version__
    for variant in ("baseline", "fused"):
        if variant == "baseline":
            source = subprocess.run(["git", "show", f"{BASE}:pcc_step.pyx"], cwd=ROOT, check=True, capture_output=True).stdout
        else:
            source = ORIGINAL
        SOURCE.write_bytes(source)
        command([sys.executable, "-m", "cython", "-3", "pcc_step.pyx", "-o", str(OUT / f"pcc_step_{variant}.c")])
        metadata[variant] = {"source_sha256": hashlib.sha256(source).hexdigest(), "generated_c_sha256": hashlib.sha256((OUT / f"pcc_step_{variant}.c").read_bytes()).hexdigest()}
        print(variant, metadata[variant])
finally:
    SOURCE.write_bytes(ORIGINAL)
(OUT / "metadata.json").write_text(json.dumps(metadata, indent=2))
print("Generated C files saved for source comparison.")
