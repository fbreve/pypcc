"""P2-06 candidate: run immutable original synthetic parity tests on SEALED Cython binaries.

No Cython rebuild. No UCI/public real dataset, real graph, real label mask, or
scientific P2 execution. Tiny 8-node synthetic fixtures in *pinned original tests*
do run PCC, so this must never be reported as 'zero PCC propagation'.
Tests verify same-backend baseline/observer exact state, milestones and RNG;
cross-backend equality is neither required nor tested.
"""
import argparse
import hashlib
import importlib
import importlib.metadata
import json
from pathlib import Path
import platform
import re
import struct
import sys
import sysconfig
import unittest

from experiments.p2_01_native_build_identity import git_blob_sha

ROOT = Path(__file__).resolve().parents[1]
P1_SHA = "f0111f153ade3d72191c107ace039e1835e5c27db2cf4b3df8d610b41b2f4166"
P2_01C_HEAD = "fa71adb7c4defa54de9d233de4b0bc84544d1c24"
TEST_BLOBS = {
    "test_live_numpy_milestones.py": "2dd7b87e54ee5acd458cf56b20adf4516c4a8e60",
    "test_live_numba_milestones.py": "ef56733245bdc7226edd6c3567de27343cdd3734",
    "test_live_cython_milestones.py": "19cce9e695a0351fe135233ba6c069b6bfba95db",
}
MODULES = {"production_pcc_step": "pcc_step",
           "isolated_pcc_step_live_research": "pcc_step_live_research"}
ARTIFACT_IDS = {"Linux": 14041882611, "Windows": 14043931061}


class ParityBlocked(ValueError):
    pass


def require(condition, why):
    if not condition:
        raise ParityBlocked(why)


def sha256(path):
    dig = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for data in iter(lambda: handle.read(1024 * 1024), b""):
            dig.update(data)
    return dig.hexdigest()


def native_header(path):
    """Inspect binary *architecture*, not claim complete linker ABI."""
    raw = Path(path).read_bytes()
    if raw[:4] == b"\\x7fELF":
        require(len(raw) >= 20, "Truncated ELF")
        bits = {1: 32, 2: 64}.get(raw[4])
        require(bits is not None, "Unknown ELF class")
        endianness = {1: "<", 2: ">"}.get(raw[5])
        require(endianness is not None, "Unknown ELF endianness")
        machine = struct.unpack(endianness + "H", raw[18:20])[0]
        require(bits == 64 and machine == 62, "Unexpected Linux native architecture")
        return {"format": "ELF", "bits": bits, "machine": machine,
                "endianness": "little" if endianness == "<" else "big"}
    if raw[:2] == b"MZ":
        require(len(raw) >= 64, "Truncated PE/DOS header")
        offset = struct.unpack("<I", raw[0x3c:0x40])[0]
        require(offset + 6 <= len(raw) and raw[offset:offset+4] == b"PE\\x00\\x00",
                "Missing PE signature")
        machine = struct.unpack("<H", raw[offset+4:offset+6])[0]
        require(machine == 0x8664, "Unexpected Windows native architecture")
        return {"format": "PE", "bits": 64, "machine": machine}
    raise ParityBlocked("Unknown native binary format")


def verify_git_test_sources():
    hashes = {}
    for filename, expected in TEST_BLOBS.items():
        path = ROOT / "tests" / filename
        require(path.is_file(), "Missing pinned original research test: " + filename)
        data = path.read_bytes()
        # CRLF checkout normalization only, matching Git's canonical blob.
        normalized = data.replace(b"\\r\\n", b"\\n")
        require(git_blob_sha(normalized) == expected,
                "Research test bytes changed: " + filename)
        hashes[filename] = expected
    return hashes


def validate_prior_archive(seal_dir):
    archive = json.loads((seal_dir / "p2-01c-candidate-seal.json").read_text(encoding="utf-8"))
    smoke_path = seal_dir / "p2-01b-native.json"
    smoke = json.loads(smoke_path.read_text(encoding="utf-8"))
    require(archive.get("status") ==
            "CANDIDATE_PY312_PACKAGE_PINS_AND_EXACT_NATIVE_BYTES_CAPTURED_NO_FITS",
            "Not the accepted native-byte archive status")
    require(archive.get("git_checkout_commit") == P2_01C_HEAD,
            "Artifact not built from the approved P2-01C final source head")
    require(archive.get("P1_scientific_sha256") == P1_SHA ==
            smoke.get("P1_approved_scientific_sha256"),
            "Frozen scientific P1 SHA changed")
    require(archive.get("P2_01B_report_sha256") == sha256(smoke_path) and
            archive.get("P2_execution_authorized") is False and
            archive.get("P2_execution_preflight_sha256") is None and
            archive.get("PCC_fits_executed") == 0 and
            archive.get("P2_01_complete_gate_pass") is False,
            "Prior P2-01C archive integrity/approval mismatch")
    require(platform.system() in ARTIFACT_IDS and
            archive.get("os", {}).get("system") == platform.system(),
            "Native artifact source OS mismatch")
    require(platform.machine().lower() in ("x86_64", "amd64"),
            "This P2 synthetic candidate is restricted to x86_64")
    require(platform.python_implementation() == "CPython" and
            platform.python_version() == smoke.get("observed_runtime", {}).get("python_version"),
            "Exact Python patch version differs from archived native runtime")
    require(sysconfig.get_config_var("EXT_SUFFIX") ==
            archive.get("python_abi", {}).get("EXT_SUFFIX"),
            "CPython ABI extension suffix drift")
    require(sys.version_info[:2] == (3, 12), "Not Python 3.12")
    require(archive.get("numba_PCC_kernel_signatures_locked") is False and
            archive.get("P2_06_observer_parity_pass") is False,
            "Prior evidence must not claim unverified native parity")
    versions = archive.get("candidate_locked_versions", {})
    require(len(versions) >= 13, "Incomplete candidate package pins")
    for name, expected in versions.items():
        require(importlib.metadata.version(name) == expected,
                "Installed candidate package drift: " + name)
    sealed = archive.get("actual_imported_native_artifacts_retained", {})
    require(set(sealed) == set(MODULES), "Expected exactly two compiled binaries")
    verified = {}
    for label, name in MODULES.items():
        entry = sealed[label]
        relative = Path(entry["relative_path"])
        require(not relative.is_absolute() and ".." not in relative.parts and
                relative.parts[0] == "native", "Unsafe archived native relative path")
        binary = (seal_dir / relative).resolve()
        require(binary.is_relative_to(seal_dir.resolve()), "Archive path escape")
        require(binary.is_file() and binary.suffix.lower() in (".so", ".pyd"),
                "Missing archived native module")
        require(binary.name.startswith(name + "."),
                "Native extension base name mismatch")
        require(binary.name.endswith(sysconfig.get_config_var("EXT_SUFFIX")),
                "Native binary incompatible CPython extension suffix")
        require(binary.stat().st_size == entry["bytes"] and
                sha256(binary) == entry["sha256"],
                "Archived native bytes differ from sealed record")
        require(smoke["native_Cython_modules"][label]["sha256"] == entry["sha256"],
                "P2-01B and P2-01C native hash disagreement")
        verified[label] = {"binary_filename": binary.name, "sha256": entry["sha256"],
                           "bytes": entry["bytes"], "header": native_header(binary)}
    return archive, verified


def restore_native_binaries(seal_dir, verified):
    # Copy only previously sealed bytes, no compilation and no production file edits.
    for label, entry in verified.items():
        source = seal_dir / "native" / entry["binary_filename"]
        dest = ROOT / entry["binary_filename"]
        require(not dest.exists(), "Unsealed compiled binary already present: " + str(dest))
        dest.write_bytes(source.read_bytes())
        require(sha256(dest) == entry["sha256"], "Restored imported native bytes drift")
        module = importlib.import_module(MODULES[label])
        loaded = Path(module.__file__).resolve()
        require(loaded == dest.resolve() and sha256(loaded) == entry["sha256"],
                "Imported native extension differs from sealed P2-01C artifact")


def dispatcher_evidence(name, obj):
    require(hasattr(obj, "nopython_signatures"), "Missing compiled Numba dispatcher " + name)
    signatures = tuple(obj.signatures)
    require(signatures and obj.nopython_signatures,
            "No nopython compiled Numba kernel specialization for " + name)
    out = []
    for signature in signatures:
        llvm = obj.inspect_llvm(signature)
        require(isinstance(llvm, str) and llvm, "Missing Numba LLVM for " + name)
        out.append({"signature": str(signature),
                    "LLVM_IR_SHA256_observed_not_locked":
                    hashlib.sha256(llvm.encode("utf-8")).hexdigest()})
    return out


def execute_parity(seal_dir):
    test_pins = verify_git_test_sources()
    archive, verified = validate_prior_archive(seal_dir)
    restore_native_binaries(seal_dir, verified)
    # Each suite is the exact original test file from the author's PR.
    suite = unittest.TestLoader().discover(
        start_dir=str(ROOT / "tests"), pattern="test_live_*_milestones.py")
    expected_count = suite.countTestCases()
    require(expected_count >= 20, "Missing a synthetic backend parity suite")
    results = unittest.TextTestRunner(stream=sys.stderr, verbosity=2).run(suite)
    require(results.wasSuccessful() and results.testsRun == expected_count,
            "Original synthetic same-backend parity tests did not all pass")
    # Record *actual* compiled PCC specializations after synthetic parity,
    # unlike the old scalar-only smoke. These signatures are not an execution GO.
    from experiments.live_numba_milestones import _observe_compiled
    from pcc_numba import _pcc_step_numba_sequential, _pcc_calc_mmpot_sequential
    dispatchers = {
        "research_observer_compiled": _observe_compiled,
        "production_step_compiled": _pcc_step_numba_sequential,
        "production_rowmax_compiled": _pcc_calc_mmpot_sequential,
    }
    signature_records = {}
    for label, dispatcher in dispatchers.items():
        # rowmax may be compiled only with early_stop=True, provided by tests.
        signature_records[label] = dispatcher_evidence(label, dispatcher)
    for label, entry in verified.items():
        module = importlib.import_module(MODULES[label])
        require(sha256(module.__file__) == entry["sha256"],
                "Imported native binary modified during synthetic tests")
    return {
        "schema": "pypcc-P2-06-synthetic-pinned-backend-parity/v1",
        "result": "THREE_ORIGINAL_SYNTHETIC_BACKEND_TEST_SUITES_PASS",
        "P1_scientific_sha256": P1_SHA,
        "source_test_git_blobs": test_pins,
        "source_observer_commit_heads": {"numpy": "6d908e78878c4922674277e3313dbfe7f7924146",
                                         "numba": "4c8e6df4e01d7a4dfc4efc31eb73db04d6e212e5",
                                         "cython": "af34f50ebabb0a7204d3f30679adaf3f00169742"},
        "prior_P2_01C_commit": P2_01C_HEAD,
        "prior_P2_01C_artifact_id": ARTIFACT_IDS[platform.system()],
        "prior_P2_01C_artifact_report_sha256":
        sha256(seal_dir / "p2-01c-candidate-seal.json"),
        "exact_native_binary_used": verified,
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "candidate_dependency_versions_observed": archive["candidate_locked_versions"],
        "original_synthetic_suite_test_count": results.testsRun,
        "test_failures": len(results.failures),
        "test_errors": len(results.errors),
        "numba_synthetic_real_kernel_compilation_observed": signature_records,
        "P2_06_synthetic_parity_candidate_pass": True,
        "P2_06_full_execution_gate_pass": False,
        "Numba_native_IR_and_ABI_frozen_for_final_P2": False,
        "real_source_datasets_loaded": 0,
        "real_experiment_graphs_or_masks_created": 0,
        "real_PCC_fits_executed": 0,
        "synthetic_PCC_executed_in_tests": True,
        "P2_execution_preflight_sha256": None,
        "P2_execution_authorized": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sealed-artifact-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = execute_parity(args.sealed_artifact_dir.resolve())
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n",
                           encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
