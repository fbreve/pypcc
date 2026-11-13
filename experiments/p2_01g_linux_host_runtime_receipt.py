"""P2-01G Linux candidate runtime receipt — safe imports only, ZERO PCC fits.

The original P2-01F prospective native Cython binaries and original P2-01D
13 wheels are byte-verified, installed OFFLINE, and imported without calling
their propagation functions. Capture which file-backed ELF/DSO objects are
actually mapped in THIS process, their exact current file bytes, CPython ABI,
runner/image label, os-release and candidate machine resource capacity.

A file-map snapshot is NOT the complete transitive DSO/toolchain/SDK/OS image
closure: dlopen of lazy dependencies or Numba JIT may map more objects later.
NEVER elevate the current receipt or its SHA to a final P2 execution gate.
No source UCI data download, graph/mask creation, synthetic or real PCC fits.
"""
import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
import sysconfig

from experiments import p2_07c_consolidated_execution_envelope_no_fit as prior
from experiments import p2_01e_replay_archived_wheelhouse as wheel_replay

ROOT = Path(__file__).resolve().parents[1]
P1_SHA = prior.P1_SHA
POLICY_SHA = "b2b7a969cf6349ec4c315cb2a91b72bf9a82868d4304a0575af246e9d85afc04"
CANDIDATE_SOURCE_BUILD_RUN = 43488066633
NEW_NATIVE_ARTIFACT_ID = 14068871078
CANDIDATE_WHEEL_ARTIFACT_ID = 14049265732
P2_01F_ARCHIVE_SHA256 = "2607b6c9b9657f96194f4ee56ec61aa24caf7ab8f8e760aa9fb30eab31eecd08"
P2_01D_ARCHIVE_SHA256 = "cb79a5637face6afd3a72ea2b05d4b3dc3ce6a52b3bc4ebab61dc8c14fbfc96d"


class RuntimeReceiptBlocked(ValueError):
    pass


def require(condition, reason):
    if not condition:
        raise RuntimeReceiptBlocked(reason)


def sha_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for data in iter(lambda: f.read(1024*1024), b""):
            h.update(data)
    return h.hexdigest()


def file_identity(file):
    path = Path(file).resolve(strict=True)
    require(path.is_file(), "Mapped native file must be regular and still present")
    size = path.stat().st_size
    require(0 < size <= 512*1024**2,
            "Mapped native file too large or missing")
    return {"path":str(path), "bytes":size, "sha256":sha_file(path)}


def mapped_file_paths(lines):
    """Linux /proc/<pid>/maps: retain only real file paths; never treat VDSO
    or deleted images as proven content-addressed disk artifacts."""
    files = set()
    deleted = []
    for line in lines:
        parts = line.split(None, 5)
        if len(parts) != 6:
            continue
        candidate = parts[5].strip()
        if not candidate.startswith("/"):
            continue
        if candidate.endswith(" (deleted)"):
            deleted.append(candidate)
            continue
        files.add(candidate)
    return sorted(files), sorted(set(deleted))


def map_identity():
    p = Path("/proc/self/maps")
    require(p.is_file(), "Only Linux process dynamic loader evidence supported")
    paths,deleted = mapped_file_paths(p.read_text(encoding="utf-8").splitlines())
    require(not deleted, "A file-backed mapped ELF was deleted while loaded")
    require(0 < len(paths) <= 512, "Mapped file list empty or suspiciously large")
    evidence = []
    for file in paths:
        obj=Path(file)
        require(obj.is_file(), "Native mapped file missing from host: "+file)
        evidence.append(file_identity(obj))
    require(any("libc.so" in x["path"] for x in evidence),
            "Could not find the actually loaded GNU libc binary file")
    require(any("pcc_step.cpython-312" in x["path"] for x in evidence) and
            any("pcc_step_live_research.cpython-312" in x["path"] for x in evidence),
            "Both sealed new Cython modules must be actually loaded in process")
    return evidence


def import_extension(module_name, exact_path):
    path = Path(exact_path).resolve(strict=True)
    require(path.suffix == ".so" and path.name.startswith(
            module_name + ".cpython-312-"), "Exact CPython Cython ELF identity unexpected")
    spec = importlib.util.spec_from_file_location(module_name, path)
    require(spec is not None and spec.loader is not None,
            "Cannot load original compiled candidate native module")
    m = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = m
    spec.loader.exec_module(m)  # NO kernel function called
    require(Path(m.__file__).resolve() == path,
            "Python imported some other native module instead")
    return {"module_name":module_name,"native_file_sha256":sha_file(path)}


def validate_receipt(report):
    require(report.get("schema") ==
            "pypcc-P2-01G-linux-exact-native-runtime-DSO-receipt/v1" and
            report.get("status") ==
            "LINUX_CANDIDATE_EXACT_RUNTIME_MAPPING_AND_RESOURCES_OBSERVED_NOT_FULL_P2_LOCK" and
            report.get("P1_scientific_sha256") == P1_SHA and
            report.get("prospective_native_P2_01F_build_run") == CANDIDATE_SOURCE_BUILD_RUN and
            report.get("P2_01F_archive_artifact_id") == NEW_NATIVE_ARTIFACT_ID and
            report.get("P2_01D_candidate_wheel_artifact_id") == CANDIDATE_WHEEL_ARTIFACT_ID and
            report.get("P2_01D_later_candidate_wheels_not_original_P2_01C") is True and
            report.get("Python",{}).get("version") == "3.12.16" and
            report.get("Python",{}).get("system") == "Linux" and
            report.get("Python",{}).get("machine").lower() in ("x86_64","amd64"),
            "Candidate runtime source, interpreter or OS differs")
    require(report.get("P2_01G_host_observation_not_an_immutable_OS_image") is True and
            report.get("P2_01G_complete_lazy_dynamic_dependency_closure_hashed") is False and
            report.get("P2_01_final_execution_toolchain_lock_pass") is False and
            report.get("P2_07_full_resources_gate_pass") is False and
            report.get("P2_execution_preflight_sha256") is None and
            report.get("P2_execution_authorized") is False and
            report.get("real_PCC_fits_executed") == 0 and
            report.get("synthetic_PCC_fits_executed") == 0 and
            report.get("official_UCI_sources_loaded") == 0 and
            report.get("graph_or_mask_created") == 0,
            "Receipt claims forbidden P2 GO or a scientific PCC fit")
    libs = report.get("exact_file_backed_mapped_native_and_executable_hashes")
    require(isinstance(libs,list) and 1 <= len(libs) <= 512 and
            len({x.get("path") for x in libs}) == len(libs) and
            all(re.fullmatch("[0-9a-f]{64}",x.get("sha256","")) is not None and
                type(x.get("bytes")) is int and 0 < x["bytes"]<=512*1024**2
                for x in libs),
            "Invalid actually loaded native library mapping digest list")
    modules = report.get("exact_new_native_Cython_modules_imported_without_fit")
    require(set(modules or {}) ==
            {"production_pcc_step","isolated_pcc_step_live_research"} and
            modules["production_pcc_step"]["module_name"]=="pcc_step" and
            modules["isolated_pcc_step_live_research"]["module_name"]==
            "pcc_step_live_research" and
            all(re.fullmatch("[0-9a-f]{64}",x.get("native_file_sha256",""))
                for x in modules.values()),
            "Missing exact new Cython import evidence")
    budgets = report.get("host_observed_candidate_resource_capacity",{})
    require(type(budgets.get("physical_RAM_bytes")) is int and
            budgets["physical_RAM_bytes"]>0 and
            type(budgets.get("free_space_on_work_dir_bytes")) is int and
            budgets["free_space_on_work_dir_bytes"]>0 and
            budgets.get("candidate_limits_from_P2_07C_have_not_been_approved") is True,
            "No physical host RAM or available scratch disk capacity receipt")
    require(len(report.get("candidate_exact_13_wheel_distribution_versions",{}))==13 and
            report.get("candidate_policy_canonical_sha256")==POLICY_SHA and
            report.get("environment_source_artifact_digest_is_metadata_not_a_recheck_of_ZIP") is True,
            "Package inventory or policy SHA differs")
    return True


def verify_loaded_file_bytes(receipt):
    """Proof against in-place file drift when reexamining a receipt on THIS host;
    this cannot prove the entire OS image cannot change later."""
    validate_receipt(receipt)
    for entry in receipt["exact_file_backed_mapped_native_and_executable_hashes"]:
        identity = file_identity(entry["path"])
        require(identity["sha256"] == entry["sha256"] and
                identity["bytes"] == entry["bytes"],
                "A loaded native/ELF image changed after the recorded receipt")
    py=receipt["Python"]["executable_file"]
    require(file_identity(py["path"]) == py,
            "CPython executable was replaced since receipt")
    require(file_identity("/etc/os-release")["sha256"] ==
            receipt["OS_source_release"]["etc_os_release_sha256"],
            "Linux system OS release source changed")
    return True


def capture(new_native_root, wheelhouse, hash_lock):
    p1=prior.read_approved_P1()
    policy=prior.load(prior.POLICY)
    proposed=prior.validate_policy(policy,p1)
    require(prior.sha256(prior.canonical(policy))==POLICY_SHA,
            "P2-07C policy differs; not allowed to silently amend resource caps")
    require(platform.system()=="Linux" and
            platform.python_version()=="3.12.16" and
            platform.machine().lower() in ("x86_64","amd64"),
            "P2-01G requires exact Linux CPython 3.12.16 x86_64")
    replay=wheel_replay.prepare(wheelhouse,hash_lock)
    require(replay["platform"]=="Linux" and
            len(replay["candidate_archived_wheels_sha256"])==13,
            "Exact later candidate wheel bytes must be available")
    import importlib.metadata
    expected={x:entry["version"] for x,entry in
              replay["candidate_archived_wheels_sha256"].items()}
    observed={x:importlib.metadata.version(x) for x in expected}
    require(expected==observed,"Installed candidate package versions drifted")
    root=new_native_root/"p2-01f-new-native-candidate"
    candidate=prior.validate_new_native(
        prior.load(root/"p2-01f-prospective-build-ledger.json"),root,"Linux")
    require(candidate["candidate_installed_versions"]==expected,
            "Newly compiled native candidate package versions not archived wheels")
    exact_modules={}
    for label,info in candidate["new_native_byte_identity_verified"].items():
        module_name = ("pcc_step" if label=="production_pcc_step"
                       else "pcc_step_live_research")
        path=root/"new-exact-native-seal"/"native"/info["filename"]
        require(sha_file(path)==info["sha256"],"Original new Cython native bytes changed")
        exact_modules[label]=import_extension(module_name,path)
        require(exact_modules[label]["native_file_sha256"]==info["sha256"],
                "Native extension changed on import")
    # Deliberately import, but NEVER invoke propagation/JIT. Loaded DSO inventory
    # is about THIS import-only process and is incomplete for future PCC fits.
    import numpy, numba, scipy, sklearn, llvmlite
    del numpy,numba,scipy,sklearn,llvmlite
    mapped=map_identity()
    files={x["path"]:x for x in mapped}
    require(all(any(Path(k).name==info["filename"] and
                    entry["sha256"]==info["sha256"]
                    for k,entry in files.items())
                for info in candidate["new_native_byte_identity_verified"].values()),
            "Exactly imported native modules not present in the actual ELF mapping")
    physical=os.sysconf("SC_PHYS_PAGES")*os.sysconf("SC_PAGE_SIZE")
    scratch=os.statvfs(ROOT)
    free=scratch.f_bavail*scratch.f_frsize
    python_info={
        "version":platform.python_version(),
        "system":platform.system(),
        "machine":platform.machine(),
        "implementation":platform.python_implementation(),
        "SOABI":sysconfig.get_config_var("SOABI"),
        "EXT_SUFFIX":sysconfig.get_config_var("EXT_SUFFIX"),
        "executable_file":file_identity(sys.executable),
    }
    report={
        "schema":"pypcc-P2-01G-linux-exact-native-runtime-DSO-receipt/v1",
        "status":"LINUX_CANDIDATE_EXACT_RUNTIME_MAPPING_AND_RESOURCES_OBSERVED_NOT_FULL_P2_LOCK",
        "P1_scientific_sha256":P1_SHA,
        "candidate_policy_canonical_sha256":POLICY_SHA,
        "prospective_native_P2_01F_build_run":CANDIDATE_SOURCE_BUILD_RUN,
        "P2_01F_archive_artifact_id":NEW_NATIVE_ARTIFACT_ID,
        "P2_01D_candidate_wheel_artifact_id":CANDIDATE_WHEEL_ARTIFACT_ID,
        "archived_P2_01F_native_artifact_GitHub_ZIP_SHA256_metadata":
            P2_01F_ARCHIVE_SHA256,
        "archived_P2_01D_wheel_artifact_GitHub_ZIP_SHA256_metadata":
            P2_01D_ARCHIVE_SHA256,
        "environment_source_artifact_digest_is_metadata_not_a_recheck_of_ZIP":True,
        "P2_01D_later_candidate_wheels_not_original_P2_01C":True,
        "candidate_exact_13_wheel_distribution_versions":expected,
        "candidate_wheel_exact_archived_sha256":{
            k:v["sha256"] for k,v in
            replay["candidate_archived_wheels_sha256"].items()},
        "actual_new_Cython_binary_provenance_SHA256":
            candidate["new_native_byte_identity_verified"],
        "exact_new_native_Cython_modules_imported_without_fit":exact_modules,
        "exact_file_backed_mapped_native_and_executable_hashes":mapped,
        "number_of_file_backed_mapped_objects_observed":len(mapped),
        "Python":python_info,
        "OS_source_release":{
            "etc_os_release_sha256":sha_file("/etc/os-release"),
            "kernel_release":platform.release(),
            "kernel_version":platform.version(),
            "libc_version_banner":platform.libc_ver(),
            "GITHUB_ImageOS_label_NOT_content_addressed":os.getenv("ImageOS"),
            "GITHUB_ImageVersion_label_NOT_content_addressed":os.getenv("ImageVersion"),
        },
        "host_observed_candidate_resource_capacity":{
            "physical_RAM_bytes":physical,
            "free_space_on_work_dir_bytes":free,
            "candidate_RAM_threshold_bytes":proposed["min_host_installed_ram_bytes"],
            "candidate_disk_threshold_bytes":proposed["min_free_disk_before_start_bytes"],
            "meets_DRAFT_RAM_requirement":
                physical>=proposed["min_host_installed_ram_bytes"],
            "meets_DRAFT_disk_requirement":
                free>=proposed["min_free_disk_before_start_bytes"],
            "candidate_limits_from_P2_07C_have_not_been_approved":True,
        },
        "P2_01G_host_observation_not_an_immutable_OS_image":True,
        "P2_01G_complete_lazy_dynamic_dependency_closure_hashed":False,
        "P2_01G_kernel_or_compiler_header_closure_hashed":False,
        "P2_01G_NumBa_PCC_native_JIT_specialization_hashed":False,
        "P2_01_final_execution_toolchain_lock_pass":False,
        "P2_07_full_resources_gate_pass":False,
        "P2_execution_preflight_sha256":None,
        "P2_execution_authorized":False,
        "official_UCI_sources_loaded":0,
        "graph_or_mask_created":0,
        "synthetic_PCC_fits_executed":0,
        "real_PCC_fits_executed":0,
    }
    validate_receipt(report)
    verify_loaded_file_bytes(report)
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native-artifact",type=Path,required=True)
    parser.add_argument("--wheelhouse",type=Path,required=True)
    parser.add_argument("--requirements-out",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    report=capture(args.native_artifact.resolve(),
                   args.wheelhouse.resolve(),
                   args.requirements_out.resolve())
    args.output.write_text(json.dumps(report,sort_keys=True,indent=2)+"\n",
                           encoding="utf-8")
    print(json.dumps({
        "status":report["status"],
        "file_backed_loaded_ELFs_SHA256_count":report[
            "number_of_file_backed_mapped_objects_observed"],
        "new_original_Cython_imports_no_fit":len(
            report["exact_new_native_Cython_modules_imported_without_fit"]),
        "installed_offline_packages":len(
            report["candidate_exact_13_wheel_distribution_versions"]),
        "candidate_host_RAM_threshold_met":report[
            "host_observed_candidate_resource_capacity"]["meets_DRAFT_RAM_requirement"],
        "candidate_disk_threshold_met":report[
            "host_observed_candidate_resource_capacity"]["meets_DRAFT_disk_requirement"],
        "P2_01_final_execution_toolchain_lock_pass":False,
        "P2_execution_preflight_sha256":None,
        "P2_execution_authorized":False,
        "real_PCC_fits_executed":0,
    },sort_keys=True))


if __name__=="__main__":
    main()
