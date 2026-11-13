"""P2-07B: isolated-process CPU/wall and peak working-set pilot, synthetic only.

Run the EXACT unmodified full-fit research benchmark scripts from #253/#255/#256,
each of which pairs original same-backend production with an observer on a
deterministic 500-node synthetic padded graph and checks final state. Reuse
the exact two P2-01C native binaries (no compilation) and validate the seal.

No official UCI download, graph-from-real-data, mask-from-real-label, or actual
P2/P3 PCC experiment. DOES execute synthetic PCC fits; never claim zero fits.
Synthetic CPU/RSS numbers CANNOT authorize 72 real-data fits or bound their RSS.
"""
import argparse
import ctypes
import hashlib
import importlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
import time

from experiments.p2_06_synthetic_parity_exact_artifacts import (
    ROOT, P1_SHA, validate_prior_archive, restore_native_binaries,
)
from experiments.p2_01_native_build_identity import git_blob_sha

BENCHMARKS = {
    "numpy": {
        "commit": "6d908e78878c4922674277e3313dbfe7f7924146",
        "file": "experiments/bench_live_numpy_fullfit.py",
        "git_blob": "cc35e3e3107b2f24b419a6d04efe35ebd8b2d4d5",
        "observer": "experiments/live_numpy_milestones.py",
        "observer_git_blob": "d642bd0f329eceb4f3dc3111b5f8c91f0e085cac",
    },
    "numba": {
        "commit": "4c8e6df4e01d7a4dfc4efc31eb73db04d6e212e5",
        "file": "experiments/bench_live_numba_fullfit.py",
        "git_blob": "241a0500ac4ed33d8c348ac29bb74d62cb98bf4b",
        "observer": "experiments/live_numba_milestones.py",
        "observer_git_blob": "e10942e96be623d47d5537d45bd365f366bba548",
    },
    "cython": {
        "commit": "af34f50ebabb0a7204d3f30679adaf3f00169742",
        "file": "experiments/bench_live_cython_fullfit.py",
        "git_blob": "72a17badcdb78d3353591fa56dfa3ecc835aa819",
        "observer": "experiments/live_cython_milestones.py",
        "observer_git_blob": "281daf0e66a567e587d0bd3e21dc34329a234b95",
    },
}
NODES = 500
ITERATIONS = 1000
REPEATS = 2
LABELED_PER_CLASS = 3
COMPILED_BATCH = 10
MAX_CHILD_SECONDS = 180


class SyntheticBudgetBlocked(ValueError):
    pass


def require(ok, message):
    if not ok:
        raise SyntheticBudgetBlocked(message)


def current_git_blob_at_commit(commit, name):
    proc = subprocess.run(
        ["git", "-C", str(ROOT), "rev-parse", commit + ":" + name],
        capture_output=True, text=True, check=False)
    require(proc.returncode == 0, "Original source commit unavailable: " + name)
    return proc.stdout.strip()


def validate_original_benchmark_sources():
    for backend, info in BENCHMARKS.items():
        for filename, expected in (
            (info["file"], info["git_blob"]),
            (info["observer"], info["observer_git_blob"]),
        ):
            require(current_git_blob_at_commit(info["commit"], filename) == expected,
                    "Original benchmark/observer not bound to original Git source: " + filename)
            candidate = ROOT / filename
            require(candidate.is_file(), "Original benchmark/observer missing: " + filename)
            source = candidate.read_bytes().replace(bytes((13, 10)), bytes((10,)))
            require(git_blob_sha(source) == expected,
                    "Ephemeral benchmark/observer differs from original Git blob: " + filename)
    return BENCHMARKS


def peak_working_set_bytes():
    """Peak process RSS on Linux; peak working set on Windows. NOT same semantics."""
    system = platform.system()
    if system == "Linux":
        import resource
        # Linux ru_maxrss in KiB; NOT the same definition as Windows working set.
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
        require(peak > 0, "Linux getrusage peak resident set not available")
        return {"peak_memory_bytes": int(peak),
                "measurement": "Linux getrusage(RUSAGE_SELF).ru_maxrss * 1024",
                "not_identical_to_windows_peak_working_set": True}
    if system == "Windows":
        from ctypes import wintypes

        class ProcessMemoryCounters(ctypes.Structure):
            _fields_ = [
                ("cb", wintypes.DWORD),
                ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        psapi = ctypes.WinDLL("psapi", use_last_error=True)
        kernel.GetCurrentProcess.argtypes = []
        kernel.GetCurrentProcess.restype = wintypes.HANDLE
        psapi.GetProcessMemoryInfo.argtypes = [
            wintypes.HANDLE, ctypes.POINTER(ProcessMemoryCounters), wintypes.DWORD]
        psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
        mem = ProcessMemoryCounters()
        mem.cb = ctypes.sizeof(mem)
        ok = psapi.GetProcessMemoryInfo(
            kernel.GetCurrentProcess(), ctypes.byref(mem), mem.cb)
        require(ok and mem.PeakWorkingSetSize > 0,
                "Windows GetProcessMemoryInfo PeakWorkingSetSize failed")
        return {"peak_memory_bytes": int(mem.PeakWorkingSetSize),
                "measurement": "Windows GetProcessMemoryInfo PeakWorkingSetSize",
                "not_identical_to_linux_peak_RSS": True}
    raise SyntheticBudgetBlocked("Only Linux/Windows process memory measurement supported")


def verify_original_result(backend, record):
    """Check known original benchmark contract; reject incomplete parity/timing."""
    require(backend in BENCHMARKS and type(record) is dict,
            "Unexpected synthetic backend/result")
    require(record.get("nodes") == NODES and
            record.get("particles") == 2*LABELED_PER_CLASS and
            record.get("repeats", record.get("paired_repeats", REPEATS)) == REPEATS and
            record.get("compilation_and_graph_excluded", True) is True,
            "Original benchmark parameters or source changed")
    require(len(record.get("milestones", [])) >= 7 and
            record["milestones"][0] == 0 and
            record["milestones"][-1] == ITERATIONS,
            "Synthetic observation milestones not complete")
    if backend == "numpy":
        require(record.get("iterations") == ITERATIONS and
                record.get("snapshot_bytes_per_observed_fit", 0) > 0,
                "NumPy observation did not capture synthetic snapshots")
        samples = record.get("samples_cpu_ns", {})
        require(len(samples.get("baseline", [])) == REPEATS and
                len(samples.get("observed", [])) == REPEATS and
                all(x > 0 for x in samples["baseline"]+samples["observed"]),
                "NumPy original paired CPU samples missing")
        value = record.get("observed_over_baseline_cpu_ratio")
        base = record.get("median_baseline_cpu_s")
        obs = record.get("median_observed_cpu_s")
    else:
        batch_field = ("fits_per_batch" if backend == "numba"
                       else "fits_per_sample")
        require(record.get("snapshot_bytes_per_fit", 0) > 0 and
                record.get(batch_field, 0) == COMPILED_BATCH,
                "Compiled observer snapshots or original paired batch missing")
        pairs = record.get("pairs", [])
        require(len(pairs) == REPEATS and
                all(entry.get("baseline", {}).get("cpu_ns", 0) > 0 and
                    entry.get("observed", {}).get("cpu_ns", 0) > 0
                    for entry in pairs),
                "Compiled original baseline/observer CPU pair is missing")
        value = record.get("observed_over_baseline_ratio") if backend == "numba" else (
            1.0 + record.get("overhead_percent_ratio_of_medians", -100)/100.0)
        base = record.get("median_baseline_cpu_s_per_fit")
        obs = record.get("median_observed_cpu_s_per_fit")
        if backend == "numba":
            require(record.get("nopython_observer_verified") is True,
                    "Numba compiled observer not confirmed nopython")
        else:
            require(record.get("mode") == "no-stop" and
                    record.get("observed_effective_iterations") == ITERATIONS,
                    "Cython benchmark not approved early_stop=False path")
    require(type(base) in (int, float) and base > 0 and
            type(obs) in (int, float) and obs > 0 and
            type(value) in (int, float) and value > 0,
            "Original benchmark reported invalid or clock-zero measurements")
    return {"median_baseline_CPU_seconds_per_complete_synthetic_fit": base,
            "median_observer_CPU_seconds_per_complete_synthetic_fit": obs,
            "observed_over_baseline_CPU_ratio": value,
            "original_paired_benchmark_checked_within_own_backend": True}


def run_child(backend, output):
    """Each backend starts in its own OS process; memory includes warmup/JIT."""
    require(backend in BENCHMARKS, "Unknown benchmark backend")
    before_cpu_ns = time.process_time_ns()
    before_wall_ns = time.perf_counter_ns()
    module = importlib.import_module(BENCHMARKS[backend]["file"][:-3].replace("/", "."))
    if backend == "numpy":
        original = module.benchmark(
            nodes=NODES, iterations=ITERATIONS, repeats=REPEATS,
            labels_per_class=LABELED_PER_CLASS)
    elif backend == "numba":
        original = module.benchmark(
            nodes=NODES, iterations=ITERATIONS, repeats=REPEATS,
            fits_per_sample=COMPILED_BATCH, labels_per_class=LABELED_PER_CLASS)
    else:
        original = module.benchmark(
            mode="no-stop", nodes=NODES, iterations=ITERATIONS,
            per_class=LABELED_PER_CLASS, repeats=REPEATS,
            fits_per_sample=COMPILED_BATCH)
    checked = verify_original_result(backend, original)
    memory = peak_working_set_bytes()
    cpu_ns = time.process_time_ns() - before_cpu_ns
    wall_ns = time.perf_counter_ns() - before_wall_ns
    report = {
        "schema": "pypcc-P2-07B-single-isolated-synthetic-benchmark/v1",
        "backend": backend,
        "original_benchmark_git_blob": BENCHMARKS[backend]["git_blob"],
        "original_observer_git_blob": BENCHMARKS[backend]["observer_git_blob"],
        "python": platform.python_version(),
        "os": platform.system(),
        "synthetic_graph_only": True,
        "synthetic_nodes": NODES,
        "synthetic_labeled_particle_count": 2*LABELED_PER_CLASS,
        "synthetic_pcc_iterations_per_fit": ITERATIONS,
        "original_benchmark": original,
        "verified_paired_cpu": checked,
        "child_import_warmup_and_paired_benchmark_wall_seconds": wall_ns/1e9,
        "child_import_warmup_and_paired_benchmark_cpu_seconds": cpu_ns/1e9,
        "child_process_peak_resident_or_working_set": memory,
        "peak_covers_imports_and_warmups_including_JIT": True,
        "peak_is_NOT_backend_fit_incremental_RAM": True,
        "metrics_not_valid_for_50k_real_PCC_fits": True,
        "no_UCI_source_downloaded_or_parsed": True,
        "synthetic_PCC_fits_did_execute": True,
        "real_PCC_fits_executed": 0,
        "P2_execution_preflight_sha256": None,
        "P2_execution_authorized": False,
    }
    output.write_text(json.dumps(report, sort_keys=True, indent=2)+"\n",
                      encoding="utf-8")
    print(json.dumps({"backend": backend, **checked,
                      "peak_process_bytes": memory["peak_memory_bytes"],
                      "synthetic_only": True, "P2_execution_authorized": False},
                     sort_keys=True))


def parent_run(seal_dir):
    require(platform.system() in ("Linux", "Windows"),
            "Only Linux/Windows original native builds accepted")
    bench_pins = validate_original_benchmark_sources()
    original_seal, binaries = validate_prior_archive(seal_dir.resolve())
    restore_native_binaries(seal_dir.resolve(), binaries)
    output = []
    with tempfile.TemporaryDirectory(prefix="p2-07b-synthetic-") as folder:
        for backend in BENCHMARKS:
            destination = Path(folder) / (backend + ".json")
            cmd = [sys.executable, "-m",
                   "experiments.p2_07b_isolated_synthetic_resource_pilot",
                   "--child", backend, "--output", str(destination)]
            start = time.perf_counter_ns()
            try:
                run = subprocess.run(
                    cmd, cwd=str(ROOT), capture_output=True, text=True,
                    check=False, timeout=MAX_CHILD_SECONDS)
            except subprocess.TimeoutExpired as exc:
                raise SyntheticBudgetBlocked(
                    "Synthetic child exceeded fixed time budget: " + backend) from exc
            require(run.returncode == 0 and destination.is_file(),
                    "Synthetic child failed: " + backend + " stderr=" + run.stderr[-3000:])
            child = json.loads(destination.read_text(encoding="utf-8"))
            require(child["backend"] == backend and
                    child["real_PCC_fits_executed"] == 0 and
                    child["P2_execution_authorized"] is False and
                    child["synthetic_PCC_fits_did_execute"] is True and
                    child["no_UCI_source_downloaded_or_parsed"] is True,
                    "Synthetic child falsely claimed real scientific execution")
            output.append({
                **child,
                "parent_measured_child_process_wall_seconds": (
                    time.perf_counter_ns()-start)/1e9,
                "child_stdout": run.stdout[-1000:],
            })
    require(len(output) == 3, "Three distinct original benchmark subprocesses required")
    return {
        "schema": "pypcc-P2-07B-original-three-backend-synthetic-CPU-RSS-pilot/v1",
        "status": "THREE_SYNTHETIC_BACKEND_PROCESS_MEASUREMENTS_NO_REAL_P2_GO",
        "P1_scientific_sha256": P1_SHA,
        "candidate_python_patch_and_OS": {
            "python": platform.python_version(), "os": platform.system(),
            "machine": platform.machine(),
        },
        "original_P2_01C_archived_artifact_seal_SHA256": (
            hashlib.sha256((seal_dir / "p2-01c-candidate-seal.json").read_bytes()).hexdigest()),
        "original_Cython_prod_and_research_binary_SHA256": {
            k: v["sha256"] for k, v in binaries.items()},
        "original_three_fullfit_benchmark_and_observer_pins": bench_pins,
        "samples": output,
        "synthetic_topology": "500-node fixed synthetic bidirectional padded ring, active degree 10, width 16",
        "synthetic_labels": "two synthetic classes, 3 labeled per class, 6 particles",
        "original_milestones_are_synthetic_schedule_not_P1_50k_schedule": True,
        "no_real_UCI_data_loaded": True,
        "all_synthetic_observer_pair_parity_pass": True,
        "memory_measurement_not_equivalent_across_Windows_Linux": True,
        "NOT_a_reliable_upper_bound_on_P2_50k_runtime_or_RAM": True,
        "P2_07_real_fit_CPU_wall_budget_verified": False,
        "P2_07_real_fit_peak_RSS_budget_verified": False,
        "P2_07_full_resources_gate_pass": False,
        "P2_01_final_execution_toolchain_lock_pass": False,
        "P2_08_execution_approved": False,
        "real_PCC_fits_executed": 0,
        "P2_execution_preflight_sha256": None,
        "P2_execution_authorized": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sealed-artifact-dir", type=Path)
    parser.add_argument("--child", choices=tuple(BENCHMARKS))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.child:
        require(args.sealed_artifact_dir is None, "A child may not update native seal")
        run_child(args.child, args.output)
        return
    require(args.sealed_artifact_dir is not None, "Original sealed native artifacts required")
    report = parent_run(args.sealed_artifact_dir.resolve())
    args.output.write_text(json.dumps(report, sort_keys=True, indent=2)+"\n",
                           encoding="utf-8")
    print(json.dumps({
        "status": report["status"],
        "synthetic_backends": [x["backend"] for x in report["samples"]],
        "synthetic_peak_process_bytes": {
            x["backend"]: x["child_process_peak_resident_or_working_set"]["peak_memory_bytes"]
            for x in report["samples"]},
        "real_PCC_fits_executed": 0,
        "P2_07_full_resources_gate_pass": False,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
