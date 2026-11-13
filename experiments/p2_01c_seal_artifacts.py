"""P2-01C: verify candidate runtime pins and archive EXACT native bytes, no PCC fits.

Consumes the independently verified P2-01B native smoke JSON and its two
already-compiled/imported extensions. Does not call propagation or use data.
This is an ephemeral CI artifact record, NOT a P2 execution preflight or GO.
"""
import argparse
import hashlib
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
import sysconfig

ROOT = Path(__file__).resolve().parents[1]
LOCK = ROOT / "experiments/p2_01c_candidate_py312_requirements.txt"
P1_SHA = "f0111f153ade3d72191c107ace039e1835e5c27db2cf4b3df8d610b41b2f4166"
MODULES = {"production_pcc_step": "pcc_step",
           "isolated_pcc_step_live_research": "pcc_step_live_research"}
REQUIRED = {"cython", "cloudpickle", "joblib", "llvmlite", "narwhals",
            "numba", "numpy", "packaging", "scikit-learn", "scipy",
            "setuptools", "threadpoolctl", "wheel"}


class SealBlocked(ValueError):
    pass


def require(ok, reason):
    if not ok:
        raise SealBlocked(reason)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def lock_versions(text):
    pinned = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        match = re.fullmatch(r"([a-zA-Z0-9_.-]+)==([a-zA-Z0-9_.+!-]+)", line)
        require(match is not None, "Non-exact or invalid candidate pin: " + line)
        name, version = match.groups()
        canonical = re.sub(r"[-_.]+", "-", name).lower()
        require(canonical not in pinned, "Duplicate candidate pin: " + canonical)
        pinned[canonical] = version
    require(set(pinned) == REQUIRED, "Candidate lock has missing/extra packages")
    return pinned


def validate_versions(expected, actual):
    for package, version in expected.items():
        require(actual.get(package) == version,
                "Installed version drift: " + package + " expected " + version +
                ", got " + str(actual.get(package)))


def validate_native_report(report):
    require(report.get("result") == "NATIVE_ARTIFACTS_BUILT_AND_IMPORTED_NO_PCC_FIT",
            "Not an accepted P2-01B build/import report")
    require(report.get("P1_approved_scientific_sha256") == P1_SHA,
            "Approved P1 SHA is not the expected pinned protocol")
    require(report.get("P2_execution_authorized") is False and
            report.get("P2_preflight_sha256") is None and
            report.get("PCC_fits_executed") == 0 and
            report.get("real_datasets_loaded") == 0 and
            report.get("label_masks_drawn") == 0,
            "P2-01C accepts only zero-fit, non-execution build evidence")
    require(set(report.get("native_Cython_modules", {})) == set(MODULES),
            "Both production and isolated compiled modules are required")
    require(report.get("generated_isolated_Cython_pyx_LF_canonical_sha256") ==
            "40db83e8df030229c15b3660b6bea03f0b45e250a09872bfd40ebaa69d4f5abf",
            "Isolated Cython source canonical SHA drift")


def verify_native_file(path, reported, *, suffix):
    path = Path(path).resolve()
    require(path.is_file() and path.suffix.lower() in (".so", ".pyd"),
            "Imported native extension missing")
    require(path.name.endswith(suffix), "Unexpected native extension ABI suffix")
    require(path.name == reported.get("binary_filename"),
            "Native filename changed since P2-01B report")
    require(path.stat().st_size == reported.get("bytes"), "Native byte count drift")
    require(sha256(path) == reported.get("sha256"), "Native binary bytes drift")
    return path


def run_command(*command):
    result = subprocess.run(command, capture_output=True, text=True,
                            errors="replace", check=False)
    require(result.returncode == 0, "Command failed: " + " ".join(command) +
            " / " + result.stderr[-300:])
    return result.stdout.strip()


def compiler_metadata():
    # Record, do NOT claim the compiler/SDK/ABI are definitively locked.
    observed = {}
    for cmd in ("gcc", "clang", "cc", "cl"):
        resolved = shutil.which(cmd)
        if not resolved:
            continue
        info = {"executable": resolved}
        if cmd != "cl":
            result = subprocess.run([resolved, "--version"], capture_output=True,
                                    text=True, errors="replace", check=False)
            info["version_banner"] = (result.stdout or result.stderr).splitlines()[:2]
        else:
            result = subprocess.run([resolved], capture_output=True,
                                    text=True, errors="replace", check=False)
            info["version_banner"] = (result.stdout or result.stderr).splitlines()[:2]
        observed[cmd] = info
    return {"found": observed,
            "configured_cc": sysconfig.get_config_var("CC"),
            "configured_cflags": sysconfig.get_config_var("CFLAGS"),
            "configured_linker": sysconfig.get_config_var("LDSHARED"),
            "environment_cc": os.getenv("CC"),
            "environment_cflags": os.getenv("CFLAGS"),
            "visual_studio_env_version": os.getenv("VSCMD_VER"),
            "definitive_abi_toolchain_lock": False}


def seal(native_report_path, output):
    require(sys.version_info[:2] == (3, 12),
            "This candidate pin targets Python 3.12 only")
    pins = lock_versions(LOCK.read_text(encoding="utf-8"))
    actual = {name: importlib.metadata.version(name) for name in pins}
    validate_versions(pins, actual)
    report_path = Path(native_report_path).resolve()
    report = json.loads(report_path.read_text(encoding="utf-8"))
    validate_native_report(report)
    require(not output.exists() or not any(output.iterdir()),
            "Never overwrite an existing artifact seal")
    output.mkdir(parents=True, exist_ok=True)
    native_dir = output / "native"
    native_dir.mkdir()
    suffix = sysconfig.get_config_var("EXT_SUFFIX")
    require(isinstance(suffix, str) and suffix.endswith((".so", ".pyd")),
            "Unknown extension ABI suffix")
    sealed = {}
    for label, module_name in MODULES.items():
        mod = importlib.import_module(module_name)   # import only, never propagate
        original = Path(mod.__file__).resolve()
        require(original.is_relative_to(ROOT.resolve()),
                "Extension was not built in isolated repository checkout")
        source = verify_native_file(original, report["native_Cython_modules"][label],
                                    suffix=suffix)
        dest = native_dir / source.name
        require(not dest.exists(), "Native extension destination collision")
        shutil.copy2(source, dest)
        require(sha256(dest) == sha256(source),
                "Copied compiled artifact differs from imported bytes")
        sealed[label] = {"relative_path": str(dest.relative_to(output)).replace("\\", "/"),
                         "sha256": sha256(dest), "bytes": dest.stat().st_size}
    # Snapshot all installed distributions; only listed candidate pins are approved
    # for this technical smoke. The others and their wheel hashes are not sealed.
    freeze = run_command(sys.executable, "-m", "pip", "freeze", "--all")
    (output / "pip-freeze-all.txt").write_text(freeze + "\n", encoding="utf-8")
    (output / "p2-01b-native.json").write_bytes(report_path.read_bytes())
    (output / "candidate-requirements.txt").write_bytes(LOCK.read_bytes())
    audit = {
        "schema": "pypcc-P2-01C-candidate-native-byte-retention/v1",
        "status": "CANDIDATE_PY312_PACKAGE_PINS_AND_EXACT_NATIVE_BYTES_CAPTURED_NO_FITS",
        "P1_scientific_sha256": P1_SHA,
        "P2_01B_report_sha256": sha256(report_path),
        "candidate_lock_sha256": sha256(LOCK),
        "candidate_locked_versions": pins,
        "verified_installed_versions": actual,
        "python": sys.version,
        "python_abi": {"SOABI": sysconfig.get_config_var("SOABI"),
                       "EXT_SUFFIX": suffix,
                       "implementation": platform.python_implementation()},
        "os": {"system": platform.system(), "release": platform.release(),
               "version": platform.version(), "machine": platform.machine()},
        "toolchain_observation_not_final_lock": compiler_metadata(),
        "git_checkout_commit": run_command("git", "-C", str(ROOT), "rev-parse", "HEAD"),
        "gha_context": {key: os.getenv(key) for key in
                        ("GITHUB_RUN_ID", "GITHUB_RUN_ATTEMPT", "GITHUB_SHA",
                         "GITHUB_HEAD_REF", "GITHUB_EVENT_NAME", "RUNNER_OS")},
        "actual_imported_native_artifacts_retained": sealed,
        "numba_PCC_kernel_signatures_locked": False,
        "exact_wheel_and_complete_compiler_lock": False,
        "P2_01_complete_gate_pass": False,
        "P2_06_observer_parity_pass": False,
        "P2_execution_preflight_sha256": None,
        "P2_execution_authorized": False,
        "real_datasets_loaded": 0,
        "graphs_or_masks_generated": 0,
        "PCC_fits_executed": 0,
    }
    (output / "p2-01c-candidate-seal.json").write_text(
        json.dumps(audit, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native-report", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = seal(args.native_report, args.output_dir)
    print(json.dumps(result, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
