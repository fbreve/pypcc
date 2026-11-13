"""P2-01E exact previously archived 13-wheel candidate replay, NO PCC fits.

Input: P2-01D original CI artifact (#264 run 43447245129), containing the
13 exact candidate wheel bytes AND an SHA/METADATA inventory. Download with
GitHub Actions, never retrieve replacement wheels from PyPI in this workflow.

Output: hash-pinned offline pip requirements for this platform; then verify
the installed 13 packages and original P2-01C native binary import/ABI.
The original P2-01C *installed* wheel byte identities were NOT preserved.
This is a NEW candidate wheelhouse, not retrospective proof of the old build.
Final compiler/OS/SDK/native ABI lock and human P2 GO remain false.
"""
import argparse
from email.parser import Parser
import hashlib
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
import sysconfig
import zipfile

from experiments.p2_06_synthetic_parity_exact_artifacts import (
    P1_SHA, ARTIFACT_IDS, validate_prior_archive, restore_native_binaries,
    sha256 as file_sha256,
)

ROOT = Path(__file__).resolve().parents[1]
PINS_FILE = ROOT / "experiments/p2_01c_candidate_py312_requirements.txt"
SOURCE_RUN_ID = 43447245129
OLD_NATIVE_RUN = 43434870932
ARCHIVE_IDS = {"Linux": 14049265732, "Windows": 14047753965}
ARCHIVE_SHA256 = {
    "Linux": "cb79a5637face6afd3a72ea2b05d4b3dc3ce6a52b3bc4ebab61dc8c14fbfc96d",
    "Windows": "216f92e5858d29a771ea2ed7bfca217899a7125dd04ac7a721ba7609cf69b501",
}
PATCH_VERSION = {"Linux": "3.12.16", "Windows": "3.12.10"}
NATIVE_EXTENSION_NAMES = {"production_pcc_step", "isolated_pcc_step_live_research"}


class ReplayBlocked(ValueError):
    pass


def require(ok, explanation):
    if not ok:
        raise ReplayBlocked(explanation)


def pins_from_text(content):
    pinned = {}
    for line in content.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        match = re.fullmatch(r"([a-zA-Z0-9_.-]+)==([a-zA-Z0-9_.+!-]+)", line)
        require(match is not None, "Candidate package version is not exactly pinned")
        name, version = match.groups()
        name = re.sub(r"[-_.]+", "-", name).lower()
        require(name not in pinned, "Duplicate candidate dependency")
        pinned[name] = version
    require(len(pinned) == 13, "Candidate wheelhouse requires exactly 13 package pins")
    return pinned


def runtime_patch_verified():
    system = platform.system()
    require(system in ARCHIVE_IDS, "Only Linux/Windows frozen wheel candidate")
    require(platform.python_implementation() == "CPython" and
            platform.python_version() == PATCH_VERSION[system] and
            sys.version_info[:2] == (3, 12) and
            platform.machine().lower() in ("x86_64", "amd64"),
            "Candidate Python patch/platform/CPython architecture mismatch")
    return system


def sha(data):
    return hashlib.sha256(data).hexdigest()


def trusted_archive_record(report, system):
    require(report.get("schema") ==
            "pypcc-P2-01D-original-native-forensics-new-wheel-candidates/v1" and
            report.get("status") ==
            "ARCHIVED_NATIVE_LINKAGE_OBSERVED_NEW_PLATFORM_WHEELS_HASHED_NOT_FINAL_P2_LOCK",
            "P2-01D wheelhouse archive report identity invalid")
    require(report.get("original_P2_01C_artifact_id") == ARTIFACT_IDS[system] and
            report.get("historical_compiler_log_facts", {}).get("original_P2_01C_run_id") ==
            OLD_NATIVE_RUN and
            report.get("historical_compiler_log_facts", {}).get(
                "original_wheel_SHA256_verified") is False and
            report.get("candidate_wheels_downloaded_now_NOT_original_P2_01C_install_proof")
            is True,
            "Candidate wheel inventory falsely claims to be original native build input")
    require(report.get("P2_execution_preflight_sha256") is None and
            report.get("P2_execution_authorized") is False and
            report.get("real_PCC_fits_executed") == 0 and
            report.get("original_dependency_wheel_bytes_sealed") is False and
            report.get("P2_01_entire_gate_pass") is False,
            "Input archive falsely claims final P2 approval")
    require(report.get("runtime", {}).get("os") == system and
            report["runtime"]["python"] == PATCH_VERSION[system],
            "Candidate archived wheel platform/Python does not match current runtime")


def inspect_wheel(path, expected):
    require(path.is_file() and path.suffix.lower() == ".whl" and
            path.name == expected.get("filename") and
            path.stat().st_size == expected.get("bytes") and
            file_sha256(path) == expected.get("sha256"),
            "P2-01D wheel bytes/name/SHA differ from exact archived candidate")
    require(re.fullmatch("[0-9a-f]{64}", expected.get("sha256", "")) is not None,
            "Invalid archived wheel checksum")
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        require(len(names) == len(set(names)), "Duplicate wheel ZIP member names")
        require(all(not n.startswith(("/", "\\")) and
                    "\\" not in n and ".." not in n.split("/") for n in names),
                "Unsafe wheel member name")
        metadata = [name for name in names
                    if name.endswith(".dist-info/METADATA") and name.count("/") == 1]
        require(len(metadata) == 1, "Missing or ambiguous wheel METADATA")
        meta = Parser().parsestr(archive.read(metadata[0]).decode("utf-8"))
    return meta


def prepare(wheelhouse, destination):
    system = runtime_patch_verified()
    manifest_path = wheelhouse / "p2-01d-native-and-wheels.json"
    require(manifest_path.is_file(), "Archived P2-01D inventory is missing")
    report = json.loads(manifest_path.read_text(encoding="utf-8"))
    trusted_archive_record(report, system)
    pins = pins_from_text(PINS_FILE.read_text(encoding="utf-8"))
    observed = report.get("wheel_candidate_evidence")
    require(type(observed) is dict and set(observed) == set(pins),
            "Archived wheelhouse has missing/extra distributions")
    wheel_dir = wheelhouse / "p2-01d-candidate-wheels"
    require(wheel_dir.is_dir() and
            set(p.name for p in wheel_dir.iterdir()) ==
            {entry["filename"] for entry in observed.values()},
            "Extra/missing wheel byte files in exact archive")
    locked = []
    verified = {}
    for name, version in sorted(pins.items()):
        entry = observed[name]
        require(entry.get("version") == version, "Candidate version drift: " + name)
        path = wheel_dir / entry["filename"]
        meta = inspect_wheel(path, entry)
        normalized = lambda x: re.sub(r"[-_.]+", "-", x).lower()
        require(normalized(meta["Name"]) == name and
                meta["Version"] == version,
                "Archived wheel embedded METADATA version/name mismatch: " + name)
        locked.append(name + "==" + version + " --hash=sha256:" + entry["sha256"])
        verified[name] = {"wheel_filename": path.name,
                          "sha256": entry["sha256"],
                          "bytes": entry["bytes"], "version": version}
    destination.write_text("\n".join(locked) + "\n", encoding="utf-8")
    return {"platform": system, "requirements_sha256": file_sha256(destination),
            "candidate_archived_wheels_sha256": verified}


def validate_installed(wheelhouse, pinned_requirements, native_dir):
    system = runtime_patch_verified()
    manifest_path = wheelhouse / "p2-01d-native-and-wheels.json"
    report = json.loads(manifest_path.read_text(encoding="utf-8"))
    trusted_archive_record(report, system)
    source = prepare(wheelhouse, pinned_requirements)
    require(source["platform"] == system, "Wheelhouse platform drift")
    pins = pins_from_text(PINS_FILE.read_text(encoding="utf-8"))
    observed = {name: importlib.metadata.version(name) for name in pins}
    require(observed == pins and
            observed == report["runtime"]["candidate_installed_versions"],
            "Exact 13 installed candidate version pins differ from archived report")
    # Independently rerun original packaging-compatible wheel inventory check.
    from experiments.p2_01d_archived_binary_wheel_forensics import (
        candidate_wheel_inventory, inspect_binary,
    )
    wheel_dir = wheelhouse / "p2-01d-candidate-wheels"
    inventory = candidate_wheel_inventory(wheel_dir, pins)
    require(all(inventory[name]["sha256"] == source["candidate_archived_wheels_sha256"]
                [name]["sha256"] for name in pins),
            "Archived wheel is not compatible with this interpreter platform/ABI")
    sealed, original = validate_prior_archive(native_dir)
    require(file_sha256(native_dir / "p2-01c-candidate-seal.json") ==
            report["P2_01C_seal_sha256"] and
            sealed["candidate_locked_versions"] == pins,
            "Old native artifact/CPython/pin report mismatch")
    restore_native_binaries(native_dir, original)  # import exact old bytes; NO FIT
    linkage = {}
    for label, details in original.items():
        require(label in NATIVE_EXTENSION_NAMES, "Unexpected native module")
        native = ROOT / details["binary_filename"]
        require(file_sha256(native) == details["sha256"],
                "Original native import bytes differed after loading")
        linkage[label] = {
            "binary_sha256": details["sha256"],
            "binary_bytes": details["bytes"],
            "static_linkage": inspect_binary(native),
        }
    require(set(linkage) == NATIVE_EXTENSION_NAMES, "Native module count not two")
    if system == "Linux":
        os_lib_note = "ELF DT_NEEDED imported names; resolved ELF shared-library bytes not pinned"
    else:
        os_lib_note = "PE DLL imported names; resolved OS DLL and SDK DLL bytes not pinned"
    return {
        "schema": "pypcc-P2-01E-archived-wheel-replay-original-native-import/v1",
        "status": "EXACT_P2_01D_CANDIDATE_WHEELS_REPLAYED_P2_01C_NATIVE_BYTES_IMPORTED_NO_FITS",
        "P1_scientific_sha256": P1_SHA,
        "source_PR_P2_01D": 264,
        "source_actions_run_P2_01D": SOURCE_RUN_ID,
        "source_artifact_id": ARCHIVE_IDS[system],
        "source_artifact_GitHub_SHA256_record_not_rehashed": ARCHIVE_SHA256[system],
        "original_P2_01C_source_actions_run": OLD_NATIVE_RUN,
        "original_P2_01C_source_artifact_id": ARTIFACT_IDS[system],
        "original_native_seal_report_sha256": report["P2_01C_seal_sha256"],
        "candidate_offline_sha_requirements_sha256": source["requirements_sha256"],
        "exact_archived_candidate_wheel_inventory": inventory,
        "installed_candidate_versions": observed,
        "original_native_imports_and_dependency_names": linkage,
        "python": {"version": platform.python_version(),
                   "implementation": platform.python_implementation(),
                   "SOABI": sysconfig.get_config_var("SOABI"),
                   "EXT_SUFFIX": sysconfig.get_config_var("EXT_SUFFIX")},
        "os": {"system": system, "release": platform.release(),
               "version": platform.version(),
               "machine": platform.machine(),
               "runner_image": os.environ.get("ImageOS"),
               "runner_image_version": os.environ.get("ImageVersion")},
        "bootstrap_pip_version_not_part_of_candidate_wheelhouse": subprocess.run(
            [sys.executable, "-m", "pip", "--version"], capture_output=True,
            text=True, check=True).stdout.strip(),
        "real_wheel_archive_older_P2_01C_identity_NOT_proved": True,
        "wheel_bytes_for_THIS_new_candidate_replay_verified": True,
        "no_network_dependency_wheel_install_in_this_workflow": True,
        "reused_old_native_binaries_not_recompiled": True,
        "resolved_OS_or_SDK_DSO_bytes_NOT_sealed": True,
        "compiler_or_SDK_executable_bytes_NOT_sealed": True,
        "native_linkage_limitations": os_lib_note,
        "Numba_PCC_native_kernel_IR_frozen_for_P2": False,
        "P2_01_final_execution_environment_lock_pass": False,
        "P2_07_full_resources_gate_pass": False,
        "P2_08_signed_technical_manifest_pass": False,
        "real_source_datasets_loaded": 0,
        "graph_mask_constructed": 0,
        "PCC_propagations_executed": 0,
        "PCC_real_data_fits_executed": 0,
        "P2_execution_preflight_sha256": None,
        "P2_execution_authorized": False,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--wheelhouse", type=Path, required=True)
    p.add_argument("--requirements-out", type=Path, required=True)
    p.add_argument("--original-native-artifact", type=Path)
    p.add_argument("--output", type=Path)
    args = p.parse_args()
    if args.original_native_artifact is None:
        require(args.output is None, "Prepare mode cannot finalize a runtime report")
        obj = prepare(args.wheelhouse.resolve(), args.requirements_out.resolve())
        print(json.dumps({"status": "PINNED_OFFLINE_SHA_REQUIREMENTS_READY",
                          "n_packages": len(obj["candidate_archived_wheels_sha256"]),
                          "python": platform.python_version(),
                          "P2_execution_authorized": False}, sort_keys=True))
    else:
        require(args.output is not None, "Verify mode requires a report path")
        obj = validate_installed(args.wheelhouse.resolve(),
                                 args.requirements_out.resolve(),
                                 args.original_native_artifact.resolve())
        args.output.write_text(json.dumps(obj, sort_keys=True, indent=2)+"\n",
                               encoding="utf-8")
        print(json.dumps({
            "status": obj["status"],
            "packages": len(obj["exact_archived_candidate_wheel_inventory"]),
            "native_import_count": len(obj["original_native_imports_and_dependency_names"]),
            "PCC_real_data_fits_executed": 0,
            "P2_execution_authorized": False}, sort_keys=True))


if __name__ == "__main__":
    main()
