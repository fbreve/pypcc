"""P2-07P: original byte-locked OCI, three *synthetic* 50k native full fits.

This is an actual end-to-end exercise of original build_graph -> fit_predict
with fake deterministic 900x7 features and 6 fake class labels, once each
on original NumPy, Numba and Cython (50,000 iterations, no early stop).
Each fit runs in its OWN fresh offline, nonroot, read-only Linux OCI process
with enforced proposed (NOT approved!) 2 GiB memory and bounded wall time.

Never downloads/reads an original UCI dataset, uses real P1 masks or seeds,
captures scientific milestones, or executes ANY of the 72 original fits.
This measures an unobserved SYNTHETIC end-to-end full-fit workload, not
a resource bound/authorization for actual P1 data/checkpoint execution.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
import time
import uuid

from experiments import p2_01k_original_OCI_backend_resolution_import_only as k
from experiments import p2_01l_frozen_72case_backend_guard_no_fit as guard
from experiments import p2_07m_exact_oci_original_graph_preparation_no_fit as m
from experiments import p2_07c_consolidated_execution_envelope_no_fit as policy

SCHEMA="pypcc-P2-07P-existing-exact-original-OCI-synthetic-900n-three-backend-50k-fullfit/v1"
WORKER="pypcc-P2-07P-original-OCI-one-synthetic-50k-fullfit/v1"
STATUS="THREE_ORIGINAL_NATIVE_50K_SYNTHETIC_GRAPH_BUILDS_AND_FITS_ONLY"
P1_SHA=k.P1_SHA
IMAGE_ID=k.IMAGE_ID
IMAGE_SHA=k.IMAGE_SHA
ROOT=m.ROOT
IMAGE_TAG=m.IMAGE_TAG
PYTHON_IN_OCI=m.PYTHON_IN_OCI
BACKENDS=("numpy","numba","cython")
NODES=900
FEATURES=7
CLASSES=2
KNOWN_LABELS=6
KNN=10
ITERATIONS=50000
FAKE_SEED=71  # original P2-07D artificial fixture seed, NEVER a frozen P1 seed
MAX_MEMORY_BYTES=2*1024**3  # candidate, deliberately NOT P2-07C approval
MAX_SYNTHETIC_WALL_SECONDS=300  # separate hard CI safety cap, NOT P1 real fit
TMPFS_MAX=128*1024**2
NO_GO={
    "P2_01_final_execution_toolchain_lock_pass":False,
    "P2_07_full_resources_gate_pass":False,
    "P2_execution_preflight_sha256":None,
    "P2_execution_authorized":False,
    "real_PCC_fits_executed":0,
}

class SyntheticFullfitBlocked(ValueError):
    pass

def require(ok, message):
    if not ok:
        raise SyntheticFullfitBlocked(message)

def canonical(x):
    return json.dumps(x,sort_keys=True,separators=(",",":"),
                      ensure_ascii=True,allow_nan=False).encode("ascii")

def digest(x):
    return hashlib.sha256(canonical(x)).hexdigest()

def original_P1_locked_NO_GO():
    m.verify_source_blobs()
    p1=policy.read_approved_P1()
    require(policy.sha256(policy.canonical(p1))==P1_SHA and
            p1["fit"]["max_iter"]==ITERATIONS and
            p1["fit"]["early_stop"] is False and
            p1["fit"]["total_proposed_fits"]==72 and
            p1["approvals_and_execution"]["P2_execution_approved"] is False,
            "Frozen original P1 or scientific authorization drift")
    return True

def fake_only_features_and_labels():
    """Return ONLY 900 artificial data rows from pre-existing P2-07D fixture."""
    import numpy as np
    from experiments import p2_07d_synthetic_50k_resource_watchdog as prior
    require((850,NODES,FEATURES,CLASSES,20) in prior.CASES and
            prior.SEED==FAKE_SEED and prior.ITERATIONS==ITERATIONS,
            "The prior 50k synthetic fixture dimensions/seed changed")
    features,labels=prior.synthetic_features_and_labels(NODES,FEATURES,CLASSES)
    require(features.shape==(NODES,FEATURES) and
            features.dtype==np.float64 and
            labels.shape==(NODES,) and
            labels.dtype==np.int64 and
            int(np.count_nonzero(labels!=-1))==KNOWN_LABELS and
            int(np.count_nonzero(labels==0))==3 and
            int(np.count_nonzero(labels==1))==3 and
            np.isfinite(features).all(),
            "Synthetic 900-row fixture unexpectedly contains real/invalid labels")
    return features,labels

def graph_checked(model):
    """Validate actual ORIGINAL production build_graph on synthetic features."""
    import numpy as np
    graph=model.neib_list
    count=model.neib_qt
    require(type(graph) is np.ndarray and graph.dtype==np.int64 and
            type(count) is np.ndarray and count.dtype==np.int64 and
            graph.shape[0]==NODES and count.shape==(NODES,) and
            graph.flags.c_contiguous and
            int(count.min())>=KNN and
            int(count.max())==graph.shape[1] and
            int(count.max())<NODES and
            not np.any(count==0),
            "Original synthetic graph did not use valid dense padded int64 kNN")
    idx=np.arange(graph.shape[1])[None,:]
    require(np.all(graph[idx>=count[:,None]]==-1),
            "Original dense -1 padding was lost")
    # Verify EVERY active edge, not just a sample; this is an undirected graph.
    degrees=[set(map(int,graph[i,:int(count[i])])) for i in range(NODES)]
    require(all(i not in neighbors and
                len(neighbors)==int(count[i]) and
                all(0<=j<NODES and i in degrees[j] for j in neighbors)
                for i,neighbors in enumerate(degrees)),
            "Original synthetic graph directed, has loops/duplicates or is asymmetric")
    shape=(int(graph.shape[0]),int(graph.shape[1]))
    h=hashlib.sha256(graph.tobytes(order="C")+count.tobytes(order="C")).hexdigest()
    return shape,h

def original_native_function_locked(backend):
    require(backend in BACKENDS,"Only three original native production backends")
    selected=k.original_three_backends()
    import pcc_step
    require(Path(pcc_step.__file__).resolve().parent==
            Path("/opt/p2-native") and
            m.sha_file(pcc_step.__file__)==k.lock.NEW_NATIVE["production_pcc_step"],
            "Exact original OCI Cython library not present on PYTHONPATH")
    import pcc_step_live_research
    require(Path(pcc_step_live_research.__file__).resolve().parent==
            Path("/opt/p2-native") and
            m.sha_file(pcc_step_live_research.__file__)==
            k.lock.NEW_NATIVE["isolated_pcc_step_live_research"],
            "Exact original OCI research Cython milestone extension changed")
    require(selected["pcc"]._HAS_CYTHON is True and
            selected["cython"] is pcc_step.pcc_propagate and
            selected["numpy"] is not selected["cython"] and
            selected["numba"] is not selected["cython"],
            "An actual production backend silently resolved via fallback")
    return selected[backend]

def cgroup_peak():
    """Asymmetric memory metrics: process VmHWM and cgroup memory.peak."""
    status=Path("/proc/self/status").read_text(encoding="ascii")
    match=re.search(r"(?m)^VmHWM:\s+(\d+)\s+kB$",status)
    require(match is not None,"Linux original OCI VmHWM unavailable")
    mem=Path("/sys/fs/cgroup/memory.peak")
    maxf=Path("/sys/fs/cgroup/memory.max")
    require(mem.is_file() and maxf.is_file() and
            maxf.read_text().strip()==str(MAX_MEMORY_BYTES),
            "Exact original OCI cgroup v2 memory quota missing or not enforced")
    peak=int(mem.read_text().strip())
    require(0<peak<=MAX_MEMORY_BYTES,"Actual cgroup peak exceeded 2 GiB candidate cap")
    return int(match.group(1))*1024,peak

def scratch_bytes():
    """Bounded scan of the disposable container tmpfs, including Numba cache."""
    total=0
    visited=0
    for p in Path("/tmp").rglob("*"):
        visited+=1
        require(visited<=10000,"Unexpected number of temporary files")
        if p.is_file():
            total+=p.stat().st_size
    require(0<=total<=TMPFS_MAX,"Synthetic fullfit tmpfs scratch over proposed fixture cap")
    return total

def one_original_fullfit_synthetic_worker(backend):
    require(backend in BACKENDS and
            os.environ.get("P207P_SYNTHETIC_ONLY")=="1" and
            os.environ.get("PYTHONPATH")==k.CORRECT_PATH and
            platform.system()=="Linux" and
            platform.machine().lower() in ("x86_64","amd64") and
            platform.python_version()=="3.12.16",
            "Only separate original Linux OCI 3.12.16 with corrected native path, synthetic-only")
    original_P1_locked_NO_GO()
    from experiments import p2_01h_import_only_oci_selfcheck as inside
    image_manifest=json.loads(inside.MANIFEST.read_text(encoding="utf-8"))
    require(image_manifest.get("P1_scientific_sha256")==P1_SHA and
            image_manifest.get("P2_execution_authorized") is False,
            "Original image manifest must refuse science execution")
    original=original_native_function_locked(backend)

    from pcc import ParticleCompetitionAndCooperation
    import numpy as np
    from time import perf_counter,process_time
    x,labels=fake_only_features_and_labels()
    model=ParticleCompetitionAndCooperation(impl=backend,n_jobs=1,
                                            update_mode="sequential")
    guard.assert_selected_exact(
        next(name for name,b in guard.BACKENDS.items() if b==backend),
        model,model._get_backend_fn(),original)
    graph_wall_start=perf_counter()
    graph_cpu_start=process_time()
    model.build_graph(x,k_nn=KNN,n_jobs=1)  # ORIGINAL production route, fake X
    graph_wall=perf_counter()-graph_wall_start
    graph_cpu=process_time()-graph_cpu_start
    shape,graph_sha=graph_checked(model)
    require(model._get_backend_fn() is original and
            model.part is None and model.node is None,
            "Graph only phase unexpectedly generated particles or backend fallback")

    fit_start_wall=perf_counter()
    fit_start_cpu=process_time()
    # ONE complete original fit on artificial data, no checkpoints (important!).
    prediction=model.fit_predict(
        labels.copy(),p_grd=0.5,delta_v=0.1,deltap=1.0,dexp=2.0,
        max_iter=ITERATIONS,early_stop=False,es_chk=2000,seed=FAKE_SEED)
    fit_wall=perf_counter()-fit_start_wall
    fit_cpu=process_time()-fit_start_cpu

    require(model.max_iter==ITERATIONS and model.early_stop is False and
            model.impl==backend and model.update_mode=="sequential" and
            model._get_backend_fn() is original and
            model.part is not None and model.node is not None and
            model.part.amount==KNOWN_LABELS and
            type(prediction) is np.ndarray and
            prediction.shape==(NODES,) and
            np.isin(prediction,[0,1]).all() and
            np.all(np.isfinite(model.node.dominance)) and
            0<fit_wall<MAX_SYNTHETIC_WALL_SECONDS and
            0<fit_cpu<MAX_SYNTHETIC_WALL_SECONDS,
            "Original 50k synthetic fit incomplete, failed invariants or safety cap")

    proc_peak,cgroup_mem_peak=cgroup_peak()
    temp_bytes=scratch_bytes()
    return {
        "schema":WORKER,
        "backend":backend,
        "data_kind":"DETERMINISTIC_ARTIFICIAL_ONLY_NO_UCI",
        "image_config_id":IMAGE_ID,
        "P1_scientific_sha256":P1_SHA,
        "synthetic_nodes":NODES,
        "synthetic_features":FEATURES,
        "synthetic_classes":CLASSES,
        "synthetic_known_labels":KNOWN_LABELS,
        "synthetic_k_nn":KNN,
        "synthetic_graph_shape":list(shape),
        "synthetic_graph_sha256":graph_sha,
        "synthetic_full_fit_requested_iterations":ITERATIONS,
        "synthetic_early_stop_enabled":False,
        "synthetic_original_seed_separate_from_P1":FAKE_SEED,
        "synthetic_50k_full_fits_completed":1,
        "real_UCI_PCC_fits_completed":0,
        "original_checkpoint_observer_invocations":0,
        "selected_function_identity_verified":True,
        "synthetic_original_graph_builder_invoked":True,
        "graph_wall_seconds":round(graph_wall,6),
        "graph_cpu_seconds":round(graph_cpu,6),
        "fit_wall_seconds":round(fit_wall,6),
        "fit_cpu_seconds":round(fit_cpu,6),
        "process_peak_rss_bytes":proc_peak,
        "cgroup_v2_peak_bytes":cgroup_mem_peak,
        "tmpfs_file_payload_bytes_end":temp_bytes,
        "candidate_2GiB_memory_limit_enforced":True,
        "synthetic_5min_failclosed_wall_limit_enforced":True,
        "actual_P1_resource_upper_bounds_proven":False,
        "P2_execution_authorized":False,
    }

def validate_worker(row,backend):
    require(type(row) is dict and backend in BACKENDS and
            row.get("schema")==WORKER and row.get("backend")==backend and
            row.get("data_kind")=="DETERMINISTIC_ARTIFICIAL_ONLY_NO_UCI" and
            row.get("image_config_id")==IMAGE_ID and
            row.get("P1_scientific_sha256")==P1_SHA,
            "Original OCI synthetic worker science/data provenance missing")
    expected={
        "synthetic_nodes":NODES,
        "synthetic_features":FEATURES,
        "synthetic_classes":CLASSES,
        "synthetic_known_labels":KNOWN_LABELS,
        "synthetic_k_nn":KNN,
        "synthetic_full_fit_requested_iterations":ITERATIONS,
        "synthetic_early_stop_enabled":False,
        "synthetic_original_seed_separate_from_P1":FAKE_SEED,
        "synthetic_50k_full_fits_completed":1,
        "real_UCI_PCC_fits_completed":0,
        "original_checkpoint_observer_invocations":0,
        "selected_function_identity_verified":True,
        "synthetic_original_graph_builder_invoked":True,
        "candidate_2GiB_memory_limit_enforced":True,
        "synthetic_5min_failclosed_wall_limit_enforced":True,
        "actual_P1_resource_upper_bounds_proven":False,
        "P2_execution_authorized":False,
    }
    require(all(row.get(key)==val for key,val in expected.items()) and
            type(row.get("synthetic_graph_shape")) is list and
            len(row["synthetic_graph_shape"])==2 and
            row["synthetic_graph_shape"][0]==NODES and
            type(row["synthetic_graph_shape"][1]) is int and
            KNN<=row["synthetic_graph_shape"][1]<NODES and
            re.fullmatch("[0-9a-f]{64}",row.get("synthetic_graph_sha256","")) is not None,
            "Synthetic full fit case count, graph seal or no-go flags invalid")
    for key in ("graph_wall_seconds","graph_cpu_seconds",
                "fit_wall_seconds","fit_cpu_seconds"):
        v=row.get(key)
        require(type(v) in (float,int) and 0<v<MAX_SYNTHETIC_WALL_SECONDS,
                "Whole original native synthetic runtime missing or breached: "+key)
    for key in ("process_peak_rss_bytes","cgroup_v2_peak_bytes"):
        v=row.get(key)
        require(type(v) is int and 0<v<=MAX_MEMORY_BYTES,
                "Real OCI whole-process/cgroup peak memory invalid")
    require(type(row.get("tmpfs_file_payload_bytes_end")) is int and
            0<=row["tmpfs_file_payload_bytes_end"]<=TMPFS_MAX,
            "Original OCI scratch metric missing/invalid")
    return True

def docker_args(backend,name):
    require(backend in BACKENDS and
            re.fullmatch(r"pypcc-p207p-[0-9a-f]{12}",name) is not None,
            "Only three explicitly synthetic original OCI workload types accepted")
    return [
        "docker","run","--rm","--name",name,
        "--platform","linux/amd64","--network","none","--read-only",
        "--user","65534:65534","--cap-drop","ALL",
        "--security-opt","no-new-privileges","--pids-limit","128",
        "--memory","2g","--memory-swap","2g","--cpus","1",
        "--tmpfs","/tmp:rw,nosuid,nodev,noexec,mode=1777,size=128m",
        "--env","P207P_SYNTHETIC_ONLY=1",
        "--env",f"PYTHONPATH={k.CORRECT_PATH}",
        "--env","PYTHONDONTWRITEBYTECODE=1","--env","PYTHONNOUSERSITE=1",
        "--env","OPENBLAS_NUM_THREADS=1","--env","OMP_NUM_THREADS=1",
        "--env","MKL_NUM_THREADS=1","--env","NUMEXPR_NUM_THREADS=1",
        "--env","NUMBA_CACHE_DIR=/tmp/numba-cache",
        "--entrypoint",PYTHON_IN_OCI,
        "--mount",f"type=bind,src={ROOT.resolve()},dst=/workspace,readonly",
        IMAGE_TAG,"-m",
        "experiments.p2_07p_exact_oci_synthetic_fullfit_resource_probe",
        "--worker","--backend",backend,
    ]

def run_one(backend):
    name="pypcc-p207p-"+uuid.uuid4().hex[:12]
    try:
        p=subprocess.run(docker_args(backend,name),
                         stdout=subprocess.PIPE,stderr=subprocess.PIPE,
                         text=True,timeout=MAX_SYNTHETIC_WALL_SECONDS,
                         check=False)
        if p.returncode!=0:
            # Deliberately do not forward arbitrary stdout/stderr, even though
            # fake dataset only; keep error categories bounded and source-safe.
            raise SyntheticFullfitBlocked(
                "Original "+backend+" 50k synthetic OCI worker failed closed (returncode)")
        require(len(p.stdout.encode("utf-8"))<8192 and
                len(p.stderr.encode("utf-8"))<8192,
                "Unexpected large original OCI synthetic worker output")
        try:
            row=json.loads(p.stdout)
        except ValueError as e:
            raise SyntheticFullfitBlocked("Original OCI synthetic worker JSON invalid") from e
        validate_worker(row,backend)
        return row
    finally:
        cleanup=subprocess.run(["docker","rm","-f",name],
                               stdout=subprocess.PIPE,stderr=subprocess.PIPE,
                               text=True,check=False,timeout=12)
        require(cleanup.returncode in (0,1),
                "Original synthetic 50k OCI worker cleanup failed")

def assemble(rows):
    require(type(rows) is list and len(rows)==len(BACKENDS) and
            all(validate_worker(row,backend)
                for row,backend in zip(rows,BACKENDS)) and
            len({row["synthetic_graph_sha256"] for row in rows})==1 and
            len({tuple(row["synthetic_graph_shape"]) for row in rows})==1,
            "Three exact original synthetic backends must use the SAME synthetic graph")
    return validate_report({
        "schema":SCHEMA,
        "status":STATUS,
        "P1_scientific_sha256":P1_SHA,
        "exact_original_OCI_image_archive_sha256":IMAGE_SHA,
        "exact_original_OCI_image_config_ID":IMAGE_ID,
        "original_900n_synthetic_fullfits_three_backend_rows":rows,
        "synthetic_50k_full_fits_executed":3,
        "real_UCI_PCC_fits_executed":0,
        "original_checkpoint_observers_executed":0,
        "original_source_UCI_fetched_or_transformed":False,
        "actual_P1_real_knn_graph_used":False,
        "source_real_labels_masks_or_P1_seeds_created":False,
        "real_full_fit_instrumented_50k_resource_upper_bound_proven":False,
        "resource_observations_are_synthetic_and_not_applicable_as_real_upper_bounds":True,
        "proposed_2GiB_container_is_NOT_approved_scientific_envelope":True,
        **NO_GO
    })

def validate_report(report):
    require(type(report) is dict and
            report.get("schema")==SCHEMA and
            report.get("status")==STATUS and
            report.get("P1_scientific_sha256")==P1_SHA and
            report.get("exact_original_OCI_image_archive_sha256")==IMAGE_SHA and
            report.get("exact_original_OCI_image_config_ID")==IMAGE_ID and
            report.get("synthetic_50k_full_fits_executed")==3 and
            report.get("real_UCI_PCC_fits_executed")==0 and
            report.get("original_checkpoint_observers_executed")==0 and
            report.get("original_source_UCI_fetched_or_transformed") is False and
            report.get("actual_P1_real_knn_graph_used") is False and
            report.get("source_real_labels_masks_or_P1_seeds_created") is False and
            report.get("real_full_fit_instrumented_50k_resource_upper_bound_proven") is False and
            report.get("resource_observations_are_synthetic_and_not_applicable_as_real_upper_bounds") is True and
            report.get("proposed_2GiB_container_is_NOT_approved_scientific_envelope") is True and
            all(report.get(key)==v for key,v in NO_GO.items()),
            "Synthetic original-OCI fullfit report falsely claims scientific execution or approval")
    rows=report.get("original_900n_synthetic_fullfits_three_backend_rows")
    require(type(rows) is list and len(rows)==3 and
            all(validate_worker(row,b) for row,b in zip(rows,BACKENDS)) and
            len({row["synthetic_graph_sha256"] for row in rows})==1 and
            len({tuple(row["synthetic_graph_shape"]) for row in rows})==1,
            "Native original fullfit synthetic graph mismatch or missing backend")
    return report

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--worker",action="store_true")
    ap.add_argument("--backend",choices=BACKENDS)
    ap.add_argument("--original-OCI-archive",type=Path)
    ap.add_argument("--output",type=Path)
    args=ap.parse_args()
    if args.worker:
        require(args.backend in BACKENDS and
                args.original_OCI_archive is None and args.output is None,
                "Synthetic-only original OCI worker must not accept source data")
        row=one_original_fullfit_synthetic_worker(args.backend)
        validate_worker(row,args.backend)
        print(canonical(row).decode("ascii"))
        return
    require(args.backend is None and
            args.original_OCI_archive is not None and
            args.output is not None and
            not args.output.exists() and
            platform.system()=="Linux",
            "Original OCI host only: exact archive and absent source-free report path")
    original_P1_locked_NO_GO()
    from experiments import p2_07e_locked_oci_cgroup_resource_supervisor as sup
    loaded=sup.verify_original_image(args.original_OCI_archive.resolve())
    require(loaded["loaded_exact_original_image_ID"]==IMAGE_ID,
            "Exact original OCI config seal failed")
    report=assemble([run_one(backend) for backend in BACKENDS])
    with args.output.open("xb") as f:
        f.write(json.dumps(report,indent=2,sort_keys=True).encode("utf-8")+b"\n")
        f.flush()
        os.fsync(f.fileno())
    require(validate_report(json.loads(args.output.read_text(encoding="utf-8"))),
            "Synthetic 50k report changed after disk write")
    print(json.dumps({
        "status":STATUS,
        "three_original_native_backend_50k_artificial_fits_completed":3,
        "source_free_review_SHA256_NOT_EXECUTION":digest(report),
        "P2_execution_preflight_sha256":None,
        "P2_execution_authorized":False,
        "real_PCC_fits_executed":0,
    },sort_keys=True))

if __name__=="__main__":
    try:
        main()
    except BaseException as exc:
        # Raw traceback from a subprocess could disclose sensitive paths;
        # only expose error category and vetted prefix from author exceptions.
        msg=(str(exc)[:130] if isinstance(exc,SyntheticFullfitBlocked)
             else type(exc).__name__)
        print("P207P_SAFE_FAILURE|"+msg,file=sys.stderr)
        sys.exit(1)
