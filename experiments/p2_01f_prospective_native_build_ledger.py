"""P2-01F: prospectively BUILD new native modules using archived wheel bytes.

Capture actual compiler and linker commands, selected executable hashes,
candidate library bytes and exact newly compiled Cython extensions.
No original P2-01C native binaries are downloaded, changed, or relabeled.
No UCI data, PCC graph/mask, particle propagation or model fit is used.
A partial toolchain ledger does NOT mean complete OS/compiler/SDK closure.
"""
import argparse
import hashlib
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

from experiments.p2_01e_replay_archived_wheelhouse import (
    prepare, runtime_patch_verified, pins_from_text, PINS_FILE,
)
from experiments import p2_01_native_build_identity as native_identity
from experiments import p2_01c_seal_artifacts as native_seal
from experiments.p2_01d_archived_binary_wheel_forensics import inspect_binary

ROOT = Path(__file__).resolve().parents[1]
P1_SHA = "f0111f153ade3d72191c107ace039e1835e5c27db2cf4b3df8d610b41b2f4166"
SOURCE_OBSERVER_HEAD = "af34f50ebabb0a7204d3f30679adaf3f00169742"
BUILD_TIMEOUT = 240


class NativeBuildBlocked(ValueError):
    pass


def require(ok, why):
    if not ok:
        raise NativeBuildBlocked(why)


def sha256_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for piece in iter(lambda: f.read(1024 * 1024), b""):
            h.update(piece)
    return h.hexdigest()


def binary_file(path):
    path = Path(path)
    require(path.is_file() and path.stat().st_size > 0, "Missing compiler/SDK file")
    return {"path": str(path), "resolved_path": str(path.resolve()),
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size}


def run_capture(argv):
    p = subprocess.run(argv, cwd=str(ROOT), capture_output=True, text=True,
                       errors="replace", timeout=BUILD_TIMEOUT, check=False,
                       env={**os.environ, "PIP_NO_INDEX": "1"})
    output = p.stdout + "\n--- stderr ---\n" + p.stderr
    require(p.returncode == 0,
            "Prospective source build failed: " + " ".join(argv) + " : " + output[-2800:])
    return output


def find_compile_and_link(log, platform_system, module):
    """Fail-closed: require actual logged compile AND link commands per module."""
    candidates = [line.strip() for line in log.splitlines()
                  if module in line and (("gcc " in line or "clang " in line)
                                         if platform_system == "Linux"
                                         else ("cl.exe" in line or "link.exe" in line))]
    if platform_system == "Linux":
        compile_commands = [x for x in candidates if re.search(r"\s-c\s", x)
                            and ".c" in x and ".o" in x]
        link_commands = [x for x in candidates if " -shared " in x
                         and ".so" in x]
    elif platform_system == "Windows":
        compile_commands = [x for x in candidates if "cl.exe" in x
                            and "/Tc" in x and "/Fo" in x]
        link_commands = [x for x in candidates if "link.exe" in x
                         and "/DLL" in x and "/OUT:" in x]
    else:
        raise NativeBuildBlocked("Unsupported compile platform")
    require(len(compile_commands) >= 1 and len(link_commands) >= 1,
            "Actual native compiler/link command missing for " + module)
    return {"compile_commands": compile_commands,
            "link_commands": link_commands}


def command_tool_hashes(platform_system, commands):
    """Hash compiler/linker executables actually named in recorded build commands."""
    merged = "\n".join(x for pair in commands.values()
                       for kind in ("compile_commands", "link_commands")
                       for x in pair[kind])
    if platform_system == "Linux":
        require("gcc " in merged, "GCC driver not found in executed build log")
        executable = shutil.which("gcc")
        require(executable, "GCC driver missing from current build PATH")
        files = {"gcc_driver": binary_file(executable)}
        for program in ("as", "ld"):
            path = shutil.which(program)
            if path:
                files[program] = binary_file(path)
        # Driver subprocesses: capture installed paths where available.
        for name in ("cc1", "collect2"):
            p = subprocess.run([executable, "-print-prog-name="+name],
                               capture_output=True, text=True, check=False)
            location = p.stdout.strip()
            if p.returncode == 0 and location:
                if not Path(location).is_file():
                    location = shutil.which(location) or ""
                if location and Path(location).is_file():
                    files[name] = binary_file(location)
        version = subprocess.run([executable, "--version"],capture_output=True,
                                 text=True,check=False)
        return {
            "observed_compile_tool_paths": files,
            "compiler_version_banner": version.stdout.splitlines()[:2],
            "compile_and_link_logged_by_build": True,
            "actual_linker_child_process_attested_by_process_tracing": False,
        }
    require(platform_system == "Windows", "Unsupported compiler system")
    # Distutils prints quoted full MSVC paths at the start of each command.
    files = {}
    paths = re.findall(r'(?im)^\s*"([^"\r\n]+\\(?:cl|link)\.exe)"', merged)
    for raw in paths:
        stem = Path(raw).stem.lower()
        if stem in ("cl", "link"):
            candidate = binary_file(raw)
            require(stem not in files or files[stem]["sha256"] ==
                    candidate["sha256"], "Multiple distinct invoked MSVC tool bytes")
            files[stem] = candidate
    require(set(files) == {"cl", "link"},
            "Could not hash both actually logged MSVC compiler and linker executables")
    proc = subprocess.run([files["cl"]["path"]],capture_output=True,
                          text=True,errors="replace",check=False)
    return {
        "observed_compile_tool_paths": files,
        "compiler_version_banner": (proc.stdout + proc.stderr).splitlines()[:3],
        "compile_and_link_logged_by_build": True,
        "actual_linker_child_process_attested_by_process_tracing": False,
    }


def observed_linker_library_candidates(platform_system, commands):
    """Inventory actually supplied library search dirs; selected libs are NOT proven."""
    link = "\n".join(pair for entry in commands.values()
                     for pair in entry["link_commands"])
    if platform_system == "Windows":
        roots = re.findall(r'(?i)"/LIBPATH:([^"]+)"|/LIBPATH:([^\s"]+)', link)
        roots = [Path(a or b) for a, b in roots]
        requested = ("python312.lib", "ucrt.lib", "vcruntime.lib",
                     "msvcrt.lib", "kernel32.lib", "oldnames.lib")
        found = {}
        for name in requested:
            matches = [root / name for root in roots if (root/name).is_file()]
            found[name] = [binary_file(x) for x in matches]
        require(any(found["python312.lib"]), "Invoked MSVC link paths lack Python import lib")
        require(any(found["ucrt.lib"]), "Invoked MSVC link paths lack SDK UCRT")
        require(any(found["kernel32.lib"]), "Invoked MSVC link paths lack SDK kernel32")
        sdk_files = [x for key in ("ucrt.lib","kernel32.lib") for x in found[key]]
        return {"observed_link_search_directories": [str(x) for x in roots],
                "hashes_of_candidate_named_link_input_files_in_those_dirs": found,
                "sdk_representative_library_files_sha256_verified": bool(sdk_files),
                "actual_linker_selected_libraries_fully_traced": False,
                "all_SDK_include_headers_and_compiler_DLLs_hashed": False}
    # Linux: linker driver/search paths and system library dependencies are
    # recorded separately through inspect_binary; no speculative resolved DSOs.
    return {"observed_link_search_command_lines":
            [x for entry in commands.values() for x in entry["link_commands"]],
            "resolved_runtime_ELF_dependency_file_bytes_hashed": False,
            "actual_linker_selected_libraries_fully_traced": False,
            "all_system_headers_and_compiler_DSO_bytes_hashed": False}


def prospective_build(wheelhouse, lock_path, output):
    system = runtime_patch_verified()
    require(output != ROOT and not output.exists(),
            "Prospective build output must be new, never overwrite old seal")
    # Byte-only previous candidate identity, NOT previous P2-01C original wheel identity.
    old_input = prepare(wheelhouse, lock_path)
    require(old_input["platform"] == system, "Candidate wheel platform mismatch")
    expected = pins_from_text(PINS_FILE.read_text(encoding="utf-8"))
    installed = {name: importlib.metadata.version(name) for name in expected}
    require(installed == expected, "Offline wheel installation missing/mismatched")
    subprocess.run([sys.executable, "-m", "pip", "check"], check=True,
                   capture_output=True, text=True)
    proof = native_identity.verify_copied_observers()
    require(set(proof) == {"numpy", "numba", "cython", "cython_generator"},
            "Original independent observer source identity incomplete")
    output.mkdir(parents=True)
    production_log = run_capture([
        sys.executable, "-m", "pip", "install", "--no-index", "--no-deps",
        "--no-build-isolation", "--verbose", "--editable", "."])
    isolated_log = run_capture([sys.executable, "-m", "experiments.build_live_cython"])
    commands = {
        "new_production_pcc_step": find_compile_and_link(
            production_log, system, "pcc_step"),
        "new_isolated_pcc_step_live_research": find_compile_and_link(
            isolated_log, system, "pcc_step_live_research"),
    }
    (output/"production-build-command-log.txt").write_text(
        production_log,encoding="utf-8")
    (output/"isolated-observer-build-command-log.txt").write_text(
        isolated_log,encoding="utf-8")
    tools = command_tool_hashes(system, commands)
    sdk = observed_linker_library_candidates(system, commands)
    smoke = native_identity.record()  # compile scalar-only JIT, NEVER PCC kernel
    require(smoke.get("PCC_fits_executed") == 0 and
            smoke.get("real_datasets_loaded") == 0 and
            smoke.get("P2_execution_authorized") is False,
            "Native smoke unexpectedly performed scientific fit")
    native_report = output/"new-native-build-import-smoke.json"
    native_report.write_text(json.dumps(smoke, indent=2, sort_keys=True)+"\n",
                             encoding="utf-8")
    copied = native_seal.seal(native_report, output/"new-exact-native-seal")
    native = {}
    for name, info in copied["actual_imported_native_artifacts_retained"].items():
        path = output/"new-exact-native-seal"/info["relative_path"]
        require(sha256_file(path) == info["sha256"], "New native file seal drift")
        native[name] = {"sha256": info["sha256"], "bytes": info["bytes"],
                        "relative_path": info["relative_path"],
                        "ELF_or_PE_static_dependency_names": inspect_binary(path)}
    require(len(native) == 2, "Both newly compiled Cython modules are mandatory")
    checkout = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"],
                              capture_output=True, text=True, check=True).stdout.strip()
    record = {
        "schema": "pypcc-P2-01F-prospective-native-toolchain-byte-ledger/v1",
        "status": "NEW_NATIVE_BINARIES_BUILT_COMPILER_BYTES_RECORDED_PARTIAL_SDK_LEDGER",
        "P1_scientific_sha256": P1_SHA,
        "actual_checkout_git_commit": checkout,
        "original_production_source_and_P1_audited_in_CI": True,
        "original_observer_sources_git_sha1": proof,
        "candidate_wheels_were_replayed_from_P2_01D_not_original_P2_01C": True,
        "candidate_offline_requirements_SHA256": sha256_file(lock_path),
        "candidate_installed_package_versions": installed,
        "candidate_platform": {"system": system, "python": platform.python_version(),
                               "SOABI": sysconfig.get_config_var("SOABI"),
                               "extension_suffix": sysconfig.get_config_var("EXT_SUFFIX"),
                               "runner_image": os.environ.get("ImageOS"),
                               "runner_image_version": os.environ.get("ImageVersion")},
        "actual_recorded_compile_and_link_commands": commands,
        "logged_compiler_and_linker_executable_byte_hashes": tools,
        "observed_linker_libraries_or_search_paths": sdk,
        "new_Cython_production_and_research_binaries": native,
        "new_native_exact_source_smoke_proved": True,
        "new_Cython_binaries_not_claimed_to_be_old_P2_01C_bytes": True,
        "complete_compiler_SDK_system_include_and_binary_link_closure_sealed": False,
        "P2_01_final_execution_toolchain_lock_pass": False,
        "P2_06_full_execution_gate_pass": False,
        "P2_07_full_resources_gate_pass": False,
        "P2_08_signed_technical_manifest_pass": False,
        "original_UCI_datasets_loaded": 0,
        "graphs_or_masks_generated": 0,
        "synthetic_PCC_fits_executed": 0,
        "real_PCC_fits_executed": 0,
        "P2_execution_preflight_sha256": None,
        "P2_execution_authorized": False,
    }
    (output/"p2-01f-prospective-build-ledger.json").write_text(
        json.dumps(record,sort_keys=True,indent=2)+"\n",encoding="utf-8")
    return record


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--wheelhouse",type=Path,required=True)
    p.add_argument("--hash-requirements",type=Path,required=True)
    p.add_argument("--output-dir",type=Path,required=True)
    args = p.parse_args()
    report = prospective_build(args.wheelhouse.resolve(),
                               args.hash_requirements.resolve(),
                               args.output_dir.resolve())
    print(json.dumps({"status": report["status"],
                      "new_native_count": len(report["new_Cython_production_and_research_binaries"]),
                      "compiler_tools_hashed":
                      len(report["logged_compiler_and_linker_executable_byte_hashes"]
                          ["observed_compile_tool_paths"]),
                      "P2_execution_authorized": False,
                      "real_PCC_fits_executed": 0},sort_keys=True))


if __name__ == "__main__":
    main()
