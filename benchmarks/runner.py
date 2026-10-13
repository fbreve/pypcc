"""Steady-state timing at the existing propagation boundary, without loop probes."""
import importlib
import importlib.metadata
import importlib.machinery
import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from pcc import ParticleCompetitionAndCooperation as PCC
from .fixtures import array_hash, file_hash


PARAMETERS = dict(p_grd=.5, delta_v=.1, deltap=1., dexp=2., es_chk=2000)
TIMINGS = ("initialization_and_dispatch_s", "propagation_s", "postprocessing_s", "fit_predict_s")
BACKENDS = {"numpy": ("pcc_numpy", "pcc_propagate_numpy"),
            "numba": ("pcc_numba", "pcc_propagate_numba"),
            "cython": ("pcc_step", "pcc_propagate")}


def expected_backend(name):
    module_name, function_name = BACKENDS[name]
    try:
        module = importlib.import_module(module_name)
    except ImportError as exc:
        raise RuntimeError(f"requested backend {name} unavailable; install/build it first") from exc
    if name == "cython" and not any(str(module.__file__).endswith(suffix)
                                    for suffix in importlib.machinery.EXTENSION_SUFFIXES):
        raise RuntimeError("Cython must be a prebuilt native extension")
    return getattr(module, function_name)


def verify_backend(model, name):
    expected = expected_backend(name)
    actual = model._get_backend_fn()
    if actual is not expected:
        raise RuntimeError(f"backend fallback rejected: requested {name}, received {actual}")
    return actual


def signatures():
    module = importlib.import_module("pcc_numba")
    return tuple(module._pcc_propagate_numba_sequential.signatures)


def seed_backend(name, seed):
    np.random.seed(seed)
    if name == "numba":
        import numba
        # Compile once, separately from propagation warm-up and measured fits.
        global _numba_seed
        if _numba_seed is None:
            @numba.njit
            def seed_rng(value):
                np.random.seed(value)
            _numba_seed = seed_rng
        _numba_seed(seed)


_numba_seed = None


def measure(arrays, case, backend_name, iterations, early_stop, seed, parameters=None,
            steady_state=True):
    """Return timings and invariants; every call creates a fresh model and state."""
    parameters = dict(PARAMETERS if parameters is None else parameters)
    model = PCC(impl=backend_name, update_mode="sequential")
    # Graph copying and backend verification are outside the fit timer.
    model.set_graph(arrays["neighbors"], arrays["degrees"], k_nn=case.k_nn)
    backend = verify_backend(model, backend_name)
    seed_backend(backend_name, seed)
    before = signatures() if backend_name == "numba" else None
    if steady_state and backend_name == "numba" and not before:
        raise RuntimeError("Numba must be warmed before steady-state measurement")
    result = {}

    def timed_backend(*args, **kwargs):
        kwargs["return_info"] = True
        entered = time.perf_counter()
        info = backend(*args, **kwargs)
        exited = time.perf_counter()
        result.update(info)
        result.update(initialization_and_dispatch_s=entered-start,
                      propagation_s=exited-entered, propagation_end=exited)

    # Wrap only this model's dispatch. The actual loop and stopping logic remain intact.
    model._get_backend_fn = lambda: timed_backend
    start = time.perf_counter()
    predictions = model.fit_predict(arrays["labels"], max_iter=iterations,
                                    early_stop=early_stop, **parameters)
    end = time.perf_counter()
    if steady_state and backend_name == "numba" and signatures() != before:
        raise RuntimeError("unexpected Numba compilation during a measured fit")
    result["postprocessing_s"] = end-result.pop("propagation_end")
    result["fit_predict_s"] = end-start
    count = result["effective_iterations"]
    if not 0 < count <= iterations or (not early_stop and count != iterations):
        raise RuntimeError("invalid effective iteration count")
    expected_reason = "early_stop" if early_stop and result["stalled_checks"] > result["stop_max"] else "max_iter"
    if result["stop_reason"] != expected_reason:
        raise RuntimeError("inconsistent stopping metrics")
    if expected_reason == "max_iter" and count != iterations:
        raise RuntimeError("capped execution did not reach max_iter")
    if early_stop and result["checks"] != (count+9)//10:
        raise RuntimeError("invalid stopping check count")
    if expected_reason == "early_stop" and (count-1) % 10:
        raise RuntimeError("early stop outside the production check schedule")
    known = arrays["labels"] != -1
    np.testing.assert_array_equal(predictions[known], arrays["truth"][known])
    dom = model.node.dominance
    if not np.isfinite(dom).all() or np.any(dom < -1e-12) or np.any(dom > 1.+1e-12):
        raise RuntimeError("invalid dominance values")
    np.testing.assert_allclose(dom.sum(axis=1), 1., rtol=0., atol=1e-10)
    if np.any(model.part.curnode < 0) or np.any(model.part.curnode >= case.nodes):
        raise RuntimeError("invalid particle positions")
    if model.part.dist_table.dtype != np.uint8:
        raise RuntimeError("invalid distance table dtype")
    if model.part.dist_table.shape != (case.nodes, len(model.part.homenode)):
        raise RuntimeError("invalid distance table shape")
    np.testing.assert_array_equal(model.part.dist_table[model.part.homenode,
                                                       np.arange(len(model.part.homenode))], 0)
    if not np.isfinite(model.part.strength).all() or np.any(model.part.strength < -1e-12) or np.any(model.part.strength > 1.+1e-12):
        raise RuntimeError("invalid particle strengths")
    np.testing.assert_array_equal(dom[known], np.eye(model.c)[model.mapped_labels[known]])
    state = dict(predictions=predictions, dominance=dom, positions=model.part.curnode,
                 strengths=model.part.strength, distances=model.part.dist_table, owndeg=model.owndeg)
    result.update(backend=backend_name, backend_identity=f"{backend.__module__}.{backend.__name__}",
                  mode="early_stopping" if early_stop else "fixed", max_iter=iterations,
                  stopped_before_cap=bool(count < iterations),
                  propagation_seed_requested=seed,
                  propagation_seed_effective=123456789 if backend_name == "cython" else seed,
                  propagation_s_per_iteration=result["propagation_s"]/count,
                  accuracy_unlabeled=float(np.mean(predictions[~known] == arrays["truth"][~known])),
                  state_hashes={key: array_hash(value) for key, value in state.items()})
    return result


def summarize(rows):
    """Summarize repeated timings only within an identical fixture/backend/mode."""
    groups = {}
    for row in rows:
        key = (row["case_id"], row["fixture_sha256"], row["backend"], row["mode"], row["max_iter"])
        groups.setdefault(key, []).append(row)
    summaries = []
    for key, selected in groups.items():
        summary = dict(zip(("case_id", "fixture_sha256", "backend", "mode", "max_iter"), key))
        summary["repetitions"] = len(selected)
        for field in (*TIMINGS, "propagation_s_per_iteration", "effective_iterations"):
            values = np.asarray([row[field] for row in selected])
            q1, q3 = np.percentile(values, [25, 75], method="linear")
            summary[field] = dict(median=float(np.median(values)), q1=float(q1), q3=float(q3),
                                  iqr=float(q3-q1), minimum=float(values.min()), maximum=float(values.max()))
        summary["cap_fraction"] = float(np.mean([row["reached_max_iter"] for row in selected]))
        summary["stop_reasons"] = {reason: sum(row["stop_reason"] == reason for row in selected)
                                   for reason in ("early_stop", "max_iter")}
        summaries.append(summary)
    return summaries


def _git(*arguments):
    try:
        return subprocess.check_output(["git", *arguments], cwd=Path(__file__).resolve().parents[1],
                                       stderr=subprocess.DEVNULL, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _hardware():
    cpu, ram = platform.processor() or None, None
    try:
        if sys.platform == "win32":
            import ctypes
            import winreg
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
                                r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as key:
                cpu = winreg.QueryValueEx(key, "ProcessorNameString")[0].strip()
            class MemoryStatus(ctypes.Structure):
                _fields_ = [("length", ctypes.c_ulong), ("load", ctypes.c_ulong)] + [
                    (name, ctypes.c_ulonglong) for name in
                    ("total", "available", "page_total", "page_available", "virtual_total", "virtual_available", "extended")]
            status = MemoryStatus()
            status.length = ctypes.sizeof(status)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
                ram = status.total
        else:
            ram = os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
            if sys.platform.startswith("linux"):
                for line in Path("/proc/cpuinfo").read_text().splitlines():
                    if line.startswith("model name"):
                        cpu = line.partition(":")[2].strip()
                        break
    except (OSError, ValueError, AttributeError, ImportError):
        pass
    return dict(cpu_model=cpu, ram_bytes=ram, logical_cpus=os.cpu_count())


def environment_manifest(config, backends, build_info=None):
    versions = {}
    for name in ("numpy", "scipy", "scikit-learn", "numba", "llvmlite", "Cython", "setuptools", "pcc_step"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    extension = None
    if "cython" in backends:
        module = importlib.import_module("pcc_step")
        extension = dict(path=module.__file__, sha256=file_hash(module.__file__))
    status = _git("status", "--porcelain")
    root = Path(__file__).resolve().parents[1]
    source_hashes = {path.name: file_hash(path) for path in root.glob("pcc*.py")}
    source_hashes["pcc_step.pyx"] = file_hash(root / "pcc_step.pyx")
    source_hashes.update({str(path.relative_to(root)).replace("\\", "/"): file_hash(path)
                          for path in (root / "benchmarks").glob("*.py")})
    for relative in ("experiments/patience_data.py", "docs/benchmarks/stopping_patience/protocol.json",
                     "docs/benchmarks/stopping_patience/inventory.json"):
        source_hashes[relative] = file_hash(root / relative)
    return dict(schema=1, timestamp_utc=datetime.now(timezone.utc).isoformat(),
                repository=dict(commit=_git("rev-parse", "HEAD"), dirty=None if status is None else bool(status),
                                status=status, source_hashes=source_hashes),
                command=sys.argv, configuration=config,
                environment=dict(**_hardware(), os=platform.platform(), python=sys.version,
                                 executable=sys.executable, implementation=platform.python_implementation(),
                                 python_build_compiler=platform.python_compiler(), versions=versions,
                                 thread_settings={key: os.environ.get(key) for key in
                                                  ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                                                   "NUMBA_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS")},
                                 affinity=sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
                                 power_policy="not controlled"),
                cython_extension=extension, build_provenance=build_info,
                build_provenance_note="User-supplied build record; Python compiler string is not extension compiler provenance.",
                timing_boundaries={
                    "initialization_and_dispatch_s": "fit entry to backend entry, including state/buffer initialization and dispatch",
                    "propagation_s": "backend entry/exit including backend validation and final return_info construction",
                    "postprocessing_s": "backend exit to fit return, including predictions and own-degree normalization",
                    "fit_predict_s": "complete fit; excludes prepared graph, model construction, graph copying and RNG seeding"},
                warmup_policy="two fresh fits of 2 iterations per case/backend/mode; first call recorded separately; not compiler-only latency",
                rng_policy={"numpy": "NumPy global RandomState seeded before every fit",
                            "numba": "independent Numba NumPy RNG seeded through njit before every fit",
                            "cython": "existing internal xorshift32 resets to 123456789; requested seed cannot change it"},
                statistics="within-case median and IQR, NumPy percentile method=linear; no pooling across cases/environments")


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
