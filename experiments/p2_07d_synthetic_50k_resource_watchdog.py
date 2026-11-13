"""P2-07D: 50k-iteration 9-case SYNTHETIC resource-bound pilot, NO UCI sources.

Three generated ring+hub undirected padded graphs with *only dimensions* from
previously verified P2-07A (310/900/440 nodes, dense widths 22/20/24,
features 6/7/6 and 3/2/2 artificial classes). Same fixed P1 50k iteration
limit and eight milestone indices, using original unchanged observers and
newly compiled exact P2-01F Cython binaries. Baseline vs observer complete
mutable state compared within each backend only; native RNG compared where
the original paired benchmarks expose it.

A PARENT process independently polls every synthetic child for CPU, peak
resident/working set, elapsed time and scratch bytes, aborting on a proposed
threshold. Child itself also reports process lifetime peak and fit timings.
Nine cases execute exactly ONCE; failures are append-only and never retried.

THIS DOES RUN SYNTHETIC PCC FITS, NOT REAL/APPROVED SCIENTIFIC PCC FITS.
Synthetic 50k success cannot establish an upper bound on real UCI graph CPU,
memory, wall time or exports; never grant P2/P3 execution approval.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import struct
import subprocess
import sys
import tempfile
import time

from experiments import p2_07c_consolidated_execution_envelope_no_fit as prior

ROOT = Path(__file__).resolve().parents[1]
POLICY = prior.POLICY
P1_SHA = prior.P1_SHA
MILESTONES = (0, 1, 10, 100, 1000, 5000, 25000, 50000)
ITERATIONS = 50000
SEED = 71  # artificial fixed pilot seed; NOT any of 72 P1 propagation seeds
CASES = ((212, 310, 6, 3, 22),
         (850, 900, 7, 2, 20),
         (292, 440, 6, 2, 24))
BACKENDS = ("numpy", "numba", "cython")
ORIGINAL_OBSERVERS = {
    "numpy": ("experiments/live_numpy_milestones.py",
              "6d908e78878c4922674277e3313dbfe7f7924146",
              "d642bd0f329eceb4f3dc3111b5f8c91f0e085cac"),
    "numba": ("experiments/live_numba_milestones.py",
              "4c8e6df4e01d7a4dfc4efc31eb73db04d6e212e5",
              "e10942e96be623d47d5537d45bd365f366bba548"),
    "cython": ("experiments/live_cython_milestones.py",
               "af34f50ebabb0a7204d3f30679adaf3f00169742",
               "281daf0e66a567e587d0bd3e21dc34329a234b95"),
}
NATIVE_BUILD_RUN = 43488066633
SYNTHETIC_ONLY_CHILD_CAP_SECONDS = 240
POLL_SECONDS = .1


class SyntheticGuardBlocked(ValueError):
    pass


def require(ok, reason):
    if not ok:
        raise SyntheticGuardBlocked(reason)


def sha256(raw):
    return hashlib.sha256(raw).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value,sort_keys=True,indent=2)+"\n",
                   encoding="utf-8")
    os.replace(tmp, path)


def original_observers_verified():
    checked = {}
    for backend, (source, commit, blob) in ORIGINAL_OBSERVERS.items():
        p = ROOT/source
        require(p.is_file(), "Missing original observer source " + source)
        content = p.read_bytes().replace(b"\r\n", b"\n")
        actual = hashlib.sha1(
            b"blob "+str(len(content)).encode("ascii")+b"\0"+content).hexdigest()
        require(actual == blob, "Original research observer was changed: "+backend)
        reference = subprocess.run(["git","-C",str(ROOT),"rev-parse",commit+":"+source],
                                   capture_output=True,text=True,check=False)
        require(reference.returncode == 0 and reference.stdout.strip() == blob,
                "Original research observer commit provenance missing")
        checked[backend] = {"Git_source_commit":commit,"source_git_blob_sha1":blob}
    return checked


def archive_and_policy_checked(native_artifact_root):
    p1 = prior.read_approved_P1()
    policy = prior.load(POLICY)
    bounds = prior.validate_policy(policy,p1)
    require(prior.sha256(prior.canonical(policy)) ==
            "b2b7a969cf6349ec4c315cb2a91b72bf9a82868d4304a0575af246e9d85afc04",
            "P2-07C proposed resource envelope was edited")
    roster = [(x["uci_id"], x["original_n_rows"], x["n_features"],
               len(x["class_support"])) for x in p1["selected_datasets"]]
    require(tuple((a,b,c,d) for a,b,c,d,w in CASES) == tuple(roster),
            "Original frozen P1 dataset dimensions / class counts changed")
    system = platform.system()
    require(system in ("Linux","Windows") and
            platform.python_version() ==
            ("3.12.16" if system == "Linux" else "3.12.10"),
            "Unexpected future execution candidate Python patch/OS")
    folder = native_artifact_root.resolve()
    report_path = folder/"p2-01f-new-native-candidate"/"p2-01f-prospective-build-ledger.json"
    native = prior.validate_new_native(
        prior.load(report_path),
        folder/"p2-01f-new-native-candidate",system)
    require(native["candidate_installed_versions"]["numpy"]=="2.5.4" and
            native["candidate_installed_versions"]["numba"]=="0.68.1" and
            native["candidate_installed_versions"]["cython"]=="3.3.1",
            "Offline native package candidate differs")
    import importlib.metadata
    for name, version in native["candidate_installed_versions"].items():
        require(importlib.metadata.version(name)==version,
                "Installed exact P2-01F candidate dependency drift: "+name)
    evidence = {}
    for name, data in native["new_native_byte_identity_verified"].items():
        file = (folder/"p2-01f-new-native-candidate"/"new-exact-native-seal"/
                data["filename"])
        require(file.is_file() and prior.file_sha(file)==data["sha256"],
                "Prospectively built new module archive content drift")
        target = ROOT / file.name
        require(not target.exists(), "Do not overwrite an existing native module")
        shutil.copy2(file,target)
        require(prior.file_sha(target)==data["sha256"],
                "Exact newly built native bytes changed on restore")
        evidence[name] = {"sha256":data["sha256"],"bytes":data["bytes"],
                          "filename":data["filename"]}
    require(set(evidence)=={"production_pcc_step",
                            "isolated_pcc_step_live_research"},
            "Both new candidate native modules are mandatory")
    return bounds,evidence,original_observers_verified()


def synthetic_graph(nodes,width):
    """Deterministic *artificial* symmetric ring+hub, degree10..width, -1 pad."""
    import numpy as np
    require(type(nodes) is int and nodes>=100 and
            type(width) is int and 10<width<nodes//2,
            "Unexpected synthetic ring geometry")
    adjacency = [set((i+j)%nodes for j in (-5,-4,-3,-2,-1,1,2,3,4,5))
                 for i in range(nodes)]
    # Fixed artificial hub 0: 10 regular ring neighbors plus (width-10)
    # extra spokes. Not the real per-node neighbor distribution!
    for other in range(6,6+width-10):
        adjacency[0].add(other)
        adjacency[other].add(0)
    dense = np.full((nodes,width),-1,dtype=np.int64,order="C")
    degree = np.empty(nodes,dtype=np.int64)
    for row,neighbors in enumerate(adjacency):
        active = sorted(neighbors)
        degree[row] = len(active)
        require(10<=len(active)<=width and row not in neighbors and
                all(row in adjacency[x] for x in active),
                "Synthetic graph is not symmetric/degree bounded")
        dense[row,:len(active)] = active
    require(dense.flags.c_contiguous and
            int(degree.min())==10 and int(degree.max())==width and
            bool((dense[np.arange(nodes)[:,None],
                        np.arange(width)[None,:]>=degree[:,None]]==-1).all()),
            "Synthetic graph dense active/padding semantics drift")
    return dense,degree


def synthetic_features_and_labels(nodes,features,classes):
    import numpy as np
    require(classes in (2,3) and nodes//classes>=20 and
            features in (6,7),"Unexpected synthetic feature/class geometry")
    row = np.arange(nodes,dtype=np.float64)[:,None]
    col = np.arange(1,features+1,dtype=np.float64)[None,:]
    features_X = np.ascontiguousarray(
        np.sin(row*col*.013) + np.cos(row*.017+col*.11),dtype=np.float64)
    labels = np.full(nodes,-1,dtype=np.int64)
    for cls in range(classes):
        start = (nodes*cls)//classes
        labels[np.array([start+1,start+3,start+5],dtype=np.int64)] = cls
    require(int(np.count_nonzero(labels>=0))==3*classes and
            all(int(np.count_nonzero(labels==cls))==3 for cls in range(classes)),
            "Synthetic source label counts must be 3 per fake class")
    return features_X,labels


def full_state_sha(model):
    import numpy as np
    h=hashlib.sha256()
    for array in (
        model.node.dominance, model.node.label,
        model.part.homenode, model.part.curnode,
        model.part.label, model.part.strength,
        model.part.dist_table, model.owndeg,
    ):
        a=np.ascontiguousarray(array)
        h.update(str(a.shape).encode("ascii"))
        h.update(a.dtype.str.encode("ascii"))
        h.update(a.tobytes())
    return h.hexdigest()


def child_process_peak():
    """Whole process peak, same OS-specific methodology as prior P2-07B."""
    if platform.system()=="Linux":
        import resource
        return {"peak_bytes":int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024),
                "metric":"Linux RUSAGE_SELF ru_maxrss (KiB times 1024)"}
    import ctypes
    from ctypes import wintypes
    class Counters(ctypes.Structure):
        _fields_=[("cb",wintypes.DWORD),("PageFaultCount",wintypes.DWORD),
                  ("PeakWorkingSetSize",ctypes.c_size_t),("WorkingSetSize",ctypes.c_size_t),
                  ("QuotaPeakPagedPoolUsage",ctypes.c_size_t),
                  ("QuotaPagedPoolUsage",ctypes.c_size_t),
                  ("QuotaPeakNonPagedPoolUsage",ctypes.c_size_t),
                  ("QuotaNonPagedPoolUsage",ctypes.c_size_t),
                  ("PagefileUsage",ctypes.c_size_t),
                  ("PeakPagefileUsage",ctypes.c_size_t)]
    kernel=ctypes.WinDLL("kernel32",use_last_error=True)
    psapi=ctypes.WinDLL("psapi",use_last_error=True)
    kernel.GetCurrentProcess.restype=wintypes.HANDLE
    psapi.GetProcessMemoryInfo.argtypes=[
        wintypes.HANDLE,ctypes.POINTER(Counters),wintypes.DWORD]
    psapi.GetProcessMemoryInfo.restype=wintypes.BOOL
    mem=Counters()
    mem.cb=ctypes.sizeof(mem)
    require(bool(psapi.GetProcessMemoryInfo(
        kernel.GetCurrentProcess(),ctypes.byref(mem),mem.cb)),
        "Cannot sample whole-process Windows peak working set")
    return {"peak_bytes":int(mem.PeakWorkingSetSize),
            "metric":"Windows GetProcessMemoryInfo PeakWorkingSetSize"}


def run_synthetic_pair(uci_id,backend,bounds):
    import importlib
    import numpy as np
    from pcc import ParticleCompetitionAndCooperation
    spec = next((x for x in CASES if x[0]==uci_id),None)
    require(spec is not None and backend in BACKENDS,"Unapproved synthetic workload")
    _,nodes,features,classes,width = spec
    neib_list,neib_qt = synthetic_graph(nodes,width)
    X,labels = synthetic_features_and_labels(nodes,features,classes)
    import pcc_numpy,pcc_numba,pcc_step
    originals = {"numpy":pcc_numpy.pcc_propagate_numpy,
                 "numba":pcc_numba.pcc_propagate_numba,
                 "cython":pcc_step.pcc_propagate}
    selected = originals[backend]
    states={}
    total_cpu_before = time.process_time_ns()
    total_wall_before = time.perf_counter_ns()
    for observed in (False,True):
        model=ParticleCompetitionAndCooperation(
            impl=backend,n_jobs=1,update_mode="sequential")
        require(model._get_backend_fn() is selected,
                "Original strict backend was not resolved (fallback prohibited)")
        # Install only generated synthetic graph; bypass build_graph intentionally.
        # set_graph would replace unused -1 padding with zero. This separate
        # pilot installs the identical frozen dense representation directly.
        model.neib_list=neib_list
        model.neib_qt=neib_qt
        model.k_nn=10
        model.data=X
        captures=[]
        if observed:
            if backend=="numpy":
                from experiments.live_numpy_milestones import observe_numpy_sequential
                def dispatch(*a,**kw):
                    captures.append(observe_numpy_sequential(
                        *a,**kw,milestones=MILESTONES))
                    return None
            elif backend=="numba":
                from experiments.live_numba_milestones import observe_numba_sequential
                def dispatch(*a,**kw):
                    captures.append(observe_numba_sequential(
                        *a,**kw,milestones=MILESTONES))
                    return None
            else:
                from experiments.live_cython_milestones import observe_cython_sequential
                def dispatch(*a,**kw):
                    kw["experimental_rowmax_cache"]=False
                    kw["experimental_canonical_rowmax_init"]=False
                    captures.append(observe_cython_sequential(
                        *a,**kw,milestones=MILESTONES))
                    return None
            model._get_backend_fn=lambda: dispatch
        elif backend=="cython":
            # Original Cython P2-07B same-backend no-stop wrapper, unchanged flags.
            def original_dispatch(*a,**kw):
                kw["experimental_rowmax_cache"]=False
                kw["experimental_canonical_rowmax_init"]=False
                return selected(*a,**kw)
            model._get_backend_fn=lambda: original_dispatch
        cpu_start=time.process_time_ns()
        wall_start=time.perf_counter_ns()
        prediction=model.fit_predict(
            labels.copy(),max_iter=ITERATIONS,early_stop=False,seed=SEED,
            p_grd=.5,delta_v=.1,deltap=1,dexp=2,es_chk=2000)
        cpu_s=(time.process_time_ns()-cpu_start)/1e9
        wall_s=(time.perf_counter_ns()-wall_start)/1e9
        require(cpu_s < bounds["max_one_fit_process_cpu_seconds"] and
                wall_s < bounds["max_one_fit_wall_seconds"],
                "Synthetic one-fit CPU/wall cap would be exceeded")
        if observed:
            require(len(captures)==1 and
                    tuple(captures[0].observed_iterations)==MILESTONES and
                    len(captures[0].milestones)==len(MILESTONES),
                    "Exact P1 milestone observer did not capture all boundaries")
            snapshot_bytes = int(captures[0].bytes_copied)
            require(0<snapshot_bytes<=bounds["max_one_fit_snapshot_array_bytes"],
                    "Original observer snapshot array cap exceeded")
            if backend=="numba":
                from experiments.live_numba_milestones import _observe_compiled
                require(bool(_observe_compiled.nopython_signatures),
                        "Numba original observer missing nopython specialization")
        else:
            snapshot_bytes=0
        rng_fingerprint=None
        if backend=="numpy":
            from experiments.bench_live_numpy_fullfit import rng_digest
            rng_fingerprint=rng_digest()
        elif backend=="numba":
            from experiments.bench_live_numba_fullfit import _next_native_draw
            rng_fingerprint=struct.pack("<d",_next_native_draw()).hex()
        states["observed" if observed else "original"]={
            "prediction_sha256":sha256(np.asarray(
                prediction,dtype="<i8").tobytes(order="C")),
            "full_mutable_state_sha256":full_state_sha(model),
            "native_rng_next_draw_or_numpy_state_fingerprint":rng_fingerprint,
            "cpu_seconds":cpu_s,"wall_seconds":wall_s,
            "snapshot_arrays_bytes":snapshot_bytes,
            "recorded_milestones":list(MILESTONES) if observed else [],
        }
    for key in ("prediction_sha256","full_mutable_state_sha256",
                "native_rng_next_draw_or_numpy_state_fingerprint"):
        require(states["original"][key]==states["observed"][key],
                "50k synthetic observer changed original same-backend "+key)
    peak=child_process_peak()
    require(0<peak["peak_bytes"]<=bounds["max_one_fit_peak_process_resident_bytes"],
            "50k synthetic child exceeded candidate process peak RAM bound")
    return {
        "schema":"pypcc-P2-07D-single-50k-synthetic-resource-case/v1",
        "uci_dimension_reference_only":uci_id,
        "backend":backend,
        "synthetic_graph_not_UCI_source":True,
        "features_are_synthetic_only":True,
        "topology":"deterministic symmetric ring plus one hub; NOT original UCI kNN",
        "original_graph_and_masks_loaded":False,
        "synthetic_nodes":nodes,"synthetic_features":features,
        "synthetic_classes":classes,"particles":classes*3,
        "dense_neighbor_width":width,"dense_neighbor_padding":-1,
        "max_iteration_count":ITERATIONS,"early_stop":False,
        "seed":SEED,"P1_exact_eight_milestones":list(MILESTONES),
        "per_fit_original_and_observer_evidence":states,
        "same_backend_complete_state_and_prediction_parity_pass":True,
        "Cython_native_rng_state_pair_digest_explicitly_exposed":False,
        "whole_process_peak_working_set_or_RSS":peak,
        "entire_child_CPU_seconds":(time.process_time_ns()-total_cpu_before)/1e9,
        "entire_child_wall_seconds":(time.perf_counter_ns()-total_wall_before)/1e9,
        "one_case_per_backend_not_72_production_replications":True,
        "synthetic_PCC_fits_executed":2,
        "real_PCC_fits_executed":0,
        "P2_07_full_resources_gate_pass":False,
        "P2_execution_authorized":False,
    }


def linux_child_metrics(pid):
    status=Path("/proc")/str(pid)/"status"
    stat=Path("/proc")/str(pid)/"stat"
    if not status.exists() or not stat.exists():
        return None
    try:
        fields={}
        for line in status.read_text(encoding="utf-8").splitlines():
            if line.startswith(("VmHWM:","VmRSS:")):
                key,value=line.split(":",1)
                fields[key]=int(value.split()[0])*1024
        stat_line=stat.read_text(encoding="utf-8")
        tail=stat_line[stat_line.rfind(")")+2:].split()
        cpu=(int(tail[11])+int(tail[12]))/os.sysconf("SC_CLK_TCK")
        return {"peak_bytes":max(fields.values(),default=0),"cpu_s":cpu,
                "source":"Linux /proc/<pid>/status VmHWM and stat utime+stime"}
    except (OSError,ValueError,IndexError):
        return None


def windows_child_metrics(process):
    import ctypes
    from ctypes import wintypes
    class Counters(ctypes.Structure):
        _fields_=[("cb",wintypes.DWORD),("PageFaultCount",wintypes.DWORD),
                  ("PeakWorkingSetSize",ctypes.c_size_t),("WorkingSetSize",ctypes.c_size_t),
                  ("QuotaPeakPagedPoolUsage",ctypes.c_size_t),
                  ("QuotaPagedPoolUsage",ctypes.c_size_t),
                  ("QuotaPeakNonPagedPoolUsage",ctypes.c_size_t),
                  ("QuotaNonPagedPoolUsage",ctypes.c_size_t),
                  ("PagefileUsage",ctypes.c_size_t),
                  ("PeakPagefileUsage",ctypes.c_size_t)]
    psapi=ctypes.WinDLL("psapi",use_last_error=True)
    kernel=ctypes.WinDLL("kernel32",use_last_error=True)
    psapi.GetProcessMemoryInfo.argtypes=[
        wintypes.HANDLE,ctypes.POINTER(Counters),wintypes.DWORD]
    psapi.GetProcessMemoryInfo.restype=wintypes.BOOL
    kernel.GetProcessTimes.argtypes=[
        wintypes.HANDLE,
        ctypes.POINTER(wintypes.FILETIME),ctypes.POINTER(wintypes.FILETIME),
        ctypes.POINTER(wintypes.FILETIME),ctypes.POINTER(wintypes.FILETIME)]
    kernel.GetProcessTimes.restype=wintypes.BOOL
    mem=Counters()
    mem.cb=ctypes.sizeof(mem)
    handle=wintypes.HANDLE(process._handle)
    a,b,c,d=(wintypes.FILETIME() for _ in range(4))
    if not psapi.GetProcessMemoryInfo(handle,ctypes.byref(mem),mem.cb):
        return None
    if not kernel.GetProcessTimes(
        handle,ctypes.byref(a),ctypes.byref(b),ctypes.byref(c),ctypes.byref(d)):
        return None
    def ticks(v):
        return (v.dwHighDateTime<<32)+v.dwLowDateTime
    return {"peak_bytes":int(mem.PeakWorkingSetSize),
            "cpu_s":(ticks(c)+ticks(d))/1e7,
            "source":"Windows GetProcessMemoryInfo PeakWorkingSetSize/GetProcessTimes"}


def measure_child(process):
    return (linux_child_metrics(process.pid) if platform.system()=="Linux"
            else windows_child_metrics(process))


def scratch_bytes(directory):
    return sum(x.stat().st_size for x in Path(directory).rglob("*") if x.is_file())


def run_monitored_child(uci_id,backend,bounds,directory):
    """Parent OS watchdog; child is synthetic-only, never a real P1 seed/graph."""
    directory=Path(directory)
    directory.mkdir(parents=True,exist_ok=False)
    out=directory/"case.json"
    command=[sys.executable,"-m",
             "experiments.p2_07d_synthetic_50k_resource_watchdog",
             "--child","--uci-id",str(uci_id),"--backend",backend,
             "--output",str(out)]
    start=time.perf_counter()
    process=subprocess.Popen(command,cwd=ROOT,stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE,text=True,errors="replace")
    max_measured_peak=0
    max_cpu=0.0
    max_disk=0
    samples=0
    failure=None
    while process.poll() is None:
        elapsed=time.perf_counter()-start
        metrics=measure_child(process)
        if metrics is not None:
            max_measured_peak=max(max_measured_peak,metrics["peak_bytes"])
            max_cpu=max(max_cpu,metrics["cpu_s"])
            samples+=1
        max_disk=max(max_disk,scratch_bytes(directory))
        if elapsed>min(SYNTHETIC_ONLY_CHILD_CAP_SECONDS,
                       2*bounds["max_one_fit_wall_seconds"]):
            failure="Synthetic-only child exceeded separate bounded pilot watchdog"
        elif max_measured_peak>bounds["max_one_fit_peak_process_resident_bytes"]:
            failure="Observed whole process peak working set/RSS exceeded candidate cap"
        elif max_cpu>bounds["max_one_fit_process_cpu_seconds"]:
            failure="Observed synthetic child process CPU exceeded candidate cap"
        elif max_disk>bounds["max_one_fit_scratch_bytes"]:
            failure="Synthetic scratch directory exceeded candidate cap"
        if failure:
            process.kill()
            break
        time.sleep(POLL_SECONDS)
    stdout,stderr=process.communicate()
    wall=time.perf_counter()-start
    if process.returncode != 0 or failure:
        raise SyntheticGuardBlocked(
            (failure or "Synthetic-only child exited with nonzero status")+
            "; rc="+str(process.returncode)+"; tail="+stderr[-2000:])
    require(len(stdout.encode("utf-8"))<=bounds["max_one_fit_text_log_bytes"] and
            len(stderr.encode("utf-8"))<=bounds["max_one_fit_text_log_bytes"],
            "Synthetic process emitted an unexpectedly large stdout/stderr log")
    require(out.is_file() and
            out.stat().st_size<=bounds["max_one_fit_aggregate_json_bytes"],
            "Synthetic child case summary absent or too large")
    result=json.loads(out.read_text(encoding="utf-8"))
    peak=result["whole_process_peak_working_set_or_RSS"]["peak_bytes"]
    require(result.get("same_backend_complete_state_and_prediction_parity_pass") is True
            and result.get("synthetic_PCC_fits_executed")==2
            and result.get("real_PCC_fits_executed")==0
            and result.get("P2_execution_authorized") is False
            and result.get("synthetic_nodes")==next(x[1] for x in CASES if x[0]==uci_id)
            and result.get("backend")==backend,
            "Child produced no valid bounded artificial 50k case")
    require(0<peak<=bounds["max_one_fit_peak_process_resident_bytes"] and
            max(max_cpu,result["entire_child_CPU_seconds"]) <=
            bounds["max_one_fit_process_cpu_seconds"] and
            wall<=min(SYNTHETIC_ONLY_CHILD_CAP_SECONDS,
                      2*bounds["max_one_fit_wall_seconds"]) and
            scratch_bytes(directory)<=bounds["max_one_fit_scratch_bytes"],
            "Final synthetic resource cap check failed")
    return {
        "uci_dimension_reference_only":uci_id,"backend":backend,
        "synthetic_graph_nodes":result["synthetic_nodes"],
        "synthetic_particles":result["particles"],
        "dense_width":result["dense_neighbor_width"],
        "complete_synthetic_50k_fit_original_cpu_s":
            result["per_fit_original_and_observer_evidence"]["original"]["cpu_seconds"],
        "complete_synthetic_50k_fit_observed_cpu_s":
            result["per_fit_original_and_observer_evidence"]["observed"]["cpu_seconds"],
        "complete_synthetic_50k_fit_original_wall_s":
            result["per_fit_original_and_observer_evidence"]["original"]["wall_seconds"],
        "complete_synthetic_50k_fit_observed_wall_s":
            result["per_fit_original_and_observer_evidence"]["observed"]["wall_seconds"],
        "observed_eight_snapshots_array_bytes":
            result["per_fit_original_and_observer_evidence"]["observed"][
                "snapshot_arrays_bytes"],
        "whole_process_peak_bytes":peak,
        "OS_parent_max_observed_peak_bytes":max_measured_peak,
        "OS_parent_max_observed_CPU_s":max_cpu,
        "OS_parent_sample_count":samples,
        "entire_child_process_cpu_s":result["entire_child_CPU_seconds"],
        "parent_observed_entire_child_wall_s":wall,
        "peak_scratch_directory_bytes":max(max_disk,scratch_bytes(directory)),
        "same_backend_full_state_and_prediction_parity":True,
        "NumPy_or_Numba_next_RNG_check":
            backend in ("numpy","numba"),
        "synthetic_fit_count_original_plus_observed":2,
        "real_PCC_fits_executed":0,
        "status":"SYNTHETIC_PASS_WITHIN_PROPOSED_CAPS_NOT_REAL_GO",
    }


def append_evidence(path, row):
    """Append-only attempt ledger; original IDs are never silently overwritten."""
    with Path(path).open("a",encoding="utf-8") as f:
        f.write(json.dumps(row,sort_keys=True)+"\n")
        f.flush()
        os.fsync(f.fileno())


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--child",action="store_true")
    p.add_argument("--uci-id",type=int)
    p.add_argument("--backend",choices=BACKENDS)
    p.add_argument("--native-artifact",type=Path)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--scratch-root",type=Path)
    args=p.parse_args()
    if args.child:
        require(args.uci_id is not None and args.backend in BACKENDS and
                args.native_artifact is None and args.scratch_root is None,
                "Synthetic child cannot touch original CI/native archive or UCI data")
        bounds=prior.validate_policy(prior.load(POLICY),prior.read_approved_P1())
        result=run_synthetic_pair(args.uci_id,args.backend,bounds)
        atomic_json(args.output,result)
        print(json.dumps({"status":"SYNTHETIC_50K_PARITY_AND_RESOURCES_PASS",
                          "backend":args.backend,
                          "uci_dimension_reference_only":args.uci_id,
                          "P2_execution_authorized":False},sort_keys=True))
        return
    require(args.native_artifact is not None and
            args.scratch_root is not None and
            args.uci_id is None and args.backend is None,
            "Parent mode only accepts sealed new native artifact/scratch root")
    bounds,native,observers=archive_and_policy_checked(args.native_artifact)
    scratch=args.scratch_root.resolve()
    require(not scratch.exists(),"Prospective synthetic-only scratch must be new")
    scratch.mkdir(parents=True)
    ledger=scratch/"append-only-50k-synthetic-attempts.ndjson"
    rows=[]
    seen=set()
    for uci_id,*_ in CASES:
        for backend in BACKENDS:
            key=(uci_id,backend)
            require(key not in seen,"Repeated synthetic case would violate no-retry rule")
            seen.add(key)
            append_evidence(ledger,{"uci_id":uci_id,"backend":backend,
                                    "status":"STARTED_SYNTHETIC_ONLY"})
            try:
                item=run_monitored_child(uci_id,backend,bounds,
                        scratch/("synthetic_"+str(uci_id)+"_"+backend))
            except Exception as exc:
                append_evidence(ledger,{"uci_id":uci_id,"backend":backend,
                                        "status":"FAILED_NO_RETRY",
                                        "error_type":type(exc).__name__})
                raise
            rows.append(item)
            append_evidence(ledger,{
                "uci_id":uci_id,"backend":backend,"status":"SYNTHETIC_PASS_NO_REAL_GO",
                "synthetic_case_digest":sha256(prior.canonical(item))})
    require(len(rows)==9 and len(seen)==9 and
            all(x["real_PCC_fits_executed"]==0 for x in rows),
            "Nine distinct 50k synthetic-only cases mandatory")
    report={
        "schema":"pypcc-P2-07D-synthetic-50k-case-resource-enforcement/v1",
        "status":"NINE_THREE_BACKEND_50K_SYNTHETIC_RESOURCE_CASES_PASS_NO_REAL_DATA_GO",
        "P1_scientific_sha256":P1_SHA,
        "candidate_P2_07C_policy_sha256":prior.sha256(prior.canonical(
            prior.load(POLICY))),
        "new_binary_provenance_actions_run":NATIVE_BUILD_RUN,
        "new_native_binary_bytes":native,
        "original_unmodified_observer_sources":observers,
        "platform":{"os":platform.system(),
                    "python":platform.python_version()},
        "artificial_workloads":[
            {"uci_dimension_reference_only":uci,"nodes":n,"n_features":d,
             "fake_class_count":c,"synthetic_particles":c*3,
             "dense_width":w,"topology":"artificial 10-neighbor ring plus hub"}
            for uci,n,d,c,w in CASES],
        "synthetic_50k_original_vs_observed_results":rows,
        "full_P1_milestones_exactly":list(MILESTONES),
        "one_synthetic_case_at_a_time":True,
        "synthetic_original_plus_observed_fit_pairs_per_platform":9,
        "synthetic_PCC_fits_executed":18,
        "real_UCI_data_loaded":False,
        "real_source_graph_or_masks_loaded":False,
        "real_PCC_fits_executed":0,
        "candidate_wall_RSS_CPU_disk_abort_logic_exercised_on_success_paths":True,
        "termination_test_on_forced_RSS_CPU_disk_overruns_not_yet_proven":True,
        "OS_polling_can_miss_sub_100ms_spikes":True,
        "Numba_JIT_and_cache_execution_host_not_fully_immutable":True,
        "synthetic_results_not_a_real_UCI_graph_resource_upper_bound":True,
        "P2_07_full_resources_gate_pass":False,
        "P2_01_final_execution_toolchain_lock_pass":False,
        "P2_08_technical_manifest_sha256":None,
        "P2_execution_preflight_sha256":None,
        "P2_execution_authorized":False,
        "source_data_or_particle_state_exported":False,
        "append_only_attempt_ledger_sha256":prior.file_sha(ledger),
    }
    atomic_json(args.output,report)
    print(json.dumps({
        "status":report["status"],
        "n_synthetic_50k_cases":len(rows),
        "n_synthetic_fits":18,"n_real_fits":0,
        "platform":platform.system(),
        "max_process_peak_bytes":max(x["whole_process_peak_bytes"] for x in rows),
        "max_observed_fit_wall_s":max(
            x["complete_synthetic_50k_fit_observed_wall_s"] for x in rows),
        "P2_07_full_resources_gate_pass":False,
        "P2_execution_authorized":False,
    },sort_keys=True))


if __name__=="__main__":
    main()
