"""Compare compiler-generated Cython kernels with matched timing on Windows.

Run separately with compiler selected by workflow (MSVC or MinGW GCC).
Reuses the same five-block ABBA methodology and stores compiler metadata.
"""
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
print("Python:", sys.version)
print("Platform:", platform.platform())
print("CC:", os.environ.get("CC", "(default distutils compiler)"))
for cmd in (["cl"], ["gcc", "--version"]):
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
        print(cmd[0], "exit", result.returncode,
              (result.stdout or result.stderr).splitlines()[:2])
    except (OSError, subprocess.TimeoutExpired) as exc:
        print(cmd[0], "unavailable:", exc)
subprocess.run([sys.executable, "benchmarks/fused_matched.py"],
               cwd=ROOT, check=True)
path = ROOT / "fused-matched-result.json"
result = json.loads(path.read_text())
result["compiler_environment"] = {
    "cc": os.environ.get("CC"),
    "platform": platform.platform(),
    "python": sys.version,
}
path.write_text(json.dumps(result, indent=2))
