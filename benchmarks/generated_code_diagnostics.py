"""Capture Cython generated C and compiler metadata for kernel investigation."""
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import sysconfig
import re
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = "10313ab01207d04ccf3b556bd7701d941b87c020"
SOURCE = ROOT / "pcc_step.pyx"
ORIGINAL = SOURCE.read_bytes()

def normalized(data):
    return data.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
OUT = ROOT / "generated-code-diagnostics"
OUT.mkdir(exist_ok=True)

def command(args):
    p = subprocess.run(args, cwd=ROOT, text=True, capture_output=True)
    if p.returncode:
        raise RuntimeError(f"{args}: {p.stdout}\n{p.stderr}")
    return p.stdout.strip()

metadata = {"platform": platform.platform(), "python": sys.version, "numpy": np.__version__}
try:
    import Cython
    metadata["cython"] = Cython.__version__
    for variant in ("baseline", "fused"):
        if variant == "baseline":
            source = subprocess.run(["git", "show", f"{BASE}:pcc_step.pyx"], cwd=ROOT, check=True, capture_output=True).stdout
        else:
            source = ORIGINAL
        source = normalized(source)
        SOURCE.write_bytes(source)
        command([sys.executable, "-m", "cython", "-3", "pcc_step.pyx", "-o", str(OUT / f"pcc_step_{variant}.c")])
        generated = (OUT / f"pcc_step_{variant}.c").read_bytes()
        # Cython can emit platform-dependent preprocessor directives and paths.
        # Record normalized text hashes as well as raw bytes for comparison.
        normalized_generated = normalized(generated)
        markers = re.findall(rb'/\* "pcc_step\.pyx":(\d+)', generated)
        metadata[variant] = {
            "source_sha256": hashlib.sha256(source).hexdigest(),
            "generated_c_sha256": hashlib.sha256(generated).hexdigest(),
            "generated_c_lf_sha256": hashlib.sha256(normalized_generated).hexdigest(),
            "generated_c_size": len(generated),
            "source_marker_count": len(markers),
            "source_marker_sha256": hashlib.sha256(b",".join(markers)).hexdigest(),
        }
        print(variant, metadata[variant])
        cfile = OUT / ("pcc_step_" + variant + ".c")
        asmfile = OUT / ("pcc_step_" + variant + (".asm" if os.name == "nt" else ".s"))
        if os.name == "nt":
            args = ["cl", "/nologo", "/O2", "/c", "/FAcs",
                    "/Fa" + str(asmfile), "/Fo" + str(OUT / (variant + ".obj")),
                    "/I" + str(Path(sysconfig.get_paths()["include"])),
                    "/I" + np.get_include(), str(cfile)]
        else:
            args = ["gcc", "-O3", "-S", "-fverbose-asm", "-fPIC",
                    "-I" + sysconfig.get_paths()["include"],
                    "-I" + np.get_include(), str(cfile), "-o", str(asmfile)]
        command(args)
        metadata[variant]["assembly_file"] = asmfile.name
finally:
    SOURCE.write_bytes(ORIGINAL)
(OUT / "metadata.json").write_text(json.dumps(metadata, indent=2))
print("Generated C files saved for source comparison.")
