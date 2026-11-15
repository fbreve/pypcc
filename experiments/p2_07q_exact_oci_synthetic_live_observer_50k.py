"""P2-07Q: exact archived OCI, original 8 live observers, 50k FAKE-only paired fits.

Reuses P2-07P's existing *exact* OCI and its deterministic fake 900x7 data,
this time running baseline and ORIGINAL research live checkpoint observer for
each of NumPy, Numba and Cython. Historical P2-06 observers and RNG helpers
are copied byte-for-byte from already-verified past Git blobs, never rebuilt.
No real UCI source/graph/mask/seed and NO P1 scientific execution authorization.

All actual fits are FAKE and confined to 3 offline/read-only/nonroot processes
of an existing identical OCI, with strictly bounded memory/time/temporary files.
All seven/eight milestone state arrays are kept ONLY in ephemeral process RAM:
no source, scientific or per-node data saved as artifacts. The report contains
small synthetic-only aggregate metrics and parity booleans.
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

from experiments import p2_07p_exact_oci_synthetic_fullfit_resource_probe as p

ROOT=p.ROOT
SCHEMA="pypcc-P2-07Q-original-OCI-3native-8milestone-50k-FAKE-paired-fits/v1"
WORKER="pypcc-P2-07Q-original-OCI-synthetic-50k-paired-observer-worker/v1"
STATUS="THREE_NATIVE_50K_SYNTHETIC_OBSERVERS_PARITY_AND_RESOURCE_MEASURED_NOT_REAL"
BACKENDS=p.BACKENDS
MILESTONES=(0,1,10,100,1000,5000,25000,50000)
MAX_SNAPSHOT_BYTES=64*1024**2
MAX_SYNTHETIC_PAIR_WALL_SECONDS=300
OCI_SHA=p.IMAGE_SHA
OCI_ID=p.IMAGE_ID
P1_SHA=p.P1_SHA
# These blobs were independently confirmed in the historical PR trees.
OBSERVER_BLOBS={
    "experiments/live_numpy_milestones.py":"d642bd0f329eceb4f3dc3111b5f8c91f0e085cac",
    "experiments/live_numba_milestones.py":"e10942e96be623d47d5537d45bd365f366bba548",
    "experiments/live_cython_milestones.py":"281daf0e66a567e587d0bd3e21dc34329a234b95",
    "experiments/bench_live_numpy_fullfit.py":"cc35e3e3107b2f24b419a6d04efe35ebd8b2d4d5",
    "experiments/bench_live_numba_fullfit.py":"241a0500ac4ed33d8c348ac29bb74d62cb98bf4b",
}
NO_GO=p.NO_GO.copy()

class ObserverBlocked(ValueError):
    pass

def require(ok,msg):
    if not ok:
        raise ObserverBlocked(msg)

def canonical(obj):
    return json.dumps(obj,sort_keys=True,separators=(",",":"),
                      allow_nan=False,ensure_ascii=True).encode("ascii")

def digest(obj):
    return hashlib.sha256(canonical(obj)).hexdigest()

def source_locked():
    p.original_P1_locked_NO_GO()
    for name,blob in OBSERVER_BLOBS.items():
        path=ROOT/name
        require(path.is_file() and p.m.git_blob(path)==blob,
                "Original historically audited observer or RNG helper Git blob drifted: "+name)
    return True

def flatten_state(model):
    """Same original P2-07D complete bytewise mutable model fingerprint."""
    from experiments.p2_07d_synthetic_50k_resource_watchdog import full_state_sha
    return full_state_sha(model)

def dispatch_for(backend, original, observed, capture):
    """Research-only per-instance override, original production source untouched."""
    require(backend in BACKENDS and callable(original) and
            isinstance(capture,list) and len(capture)==0,
            "Only exact original three backend paired synthetic observation paths")
    if observed:
        if backend=="numpy":
            from experiments.live_numpy_milestones import observe_numpy_sequential
            observer=observe_numpy_sequential
        elif backend=="numba":
            from experiments.live_numba_milestones import observe_numba_sequential
            observer=observe_numba_sequential
        else:
            from experiments.live_cython_milestones import observe_cython_sequential
            observer=observe_cython_sequential
        def dispatch(*args,**kwargs):
            require(not capture,"Observed synthetic observer called more than once")
            if backend=="cython":
                kwargs["experimental_rowmax_cache"]=False
                kwargs["experimental_canonical_rowmax_init"]=False
            result=observer(*args,**kwargs,milestones=MILESTONES,
                            max_observations=8,max_bytes=MAX_SNAPSHOT_BYTES)
            capture.append(result)
            return None  # same model finalization as original PCC
        return dispatch

    def dispatch(*args,**kwargs):
        require(not capture,"Baseline synthetic native propagator called twice")
        if backend=="cython":
            kwargs["experimental_rowmax_cache"]=False
            kwargs["experimental_canonical_rowmax_init"]=False
        result=original(*args,**kwargs,return_info=True)
        capture.append(result)
        return None  # baseline unchanged, return_info is read-only
    return dispatch

def assert_milestones(report,model,backend):
    """Check full native milestone array capture while keeping arrays private."""
    import numpy as np
    require(tuple(report.observed_iterations)==MILESTONES and
            len(report.milestones)==len(MILESTONES) and
            0<int(report.bytes_copied)<=MAX_SNAPSHOT_BYTES and
            report.info["effective_iterations"]==p.ITERATIONS and
            report.info["reached_max_iter"] is True and
            report.info["stop_reason"]=="max_iter",
            "P1 eight planned artificial checkpoints incomplete or wrong native iteration count")
    expected={
        "positions":(p.KNOWN_LABELS,),
        "strengths":(p.KNOWN_LABELS,),
        "distances":(p.NODES,p.KNOWN_LABELS),
        "dominance":(p.NODES,p.CLASSES),
        "owndeg":(p.NODES,p.CLASSES),
    }
    if backend=="cython":
        expected["rowmax"]=(p.NODES,)
    for milestone,iteration in zip(report.milestones,MILESTONES):
        require(milestone.iteration==iteration and
                set(milestone.arrays)==set(expected),
                "Original checkpoint array names or iteration drifted")
        for name,shape in expected.items():
            array=milestone.arrays[name]
            require(type(array) is np.ndarray and
                    array.shape==shape and array.flags.writeable is False,
                    "Checkpoint contains wrong-shaped or mutable array")
    # The final propagation writes node.owndeg unnormalized, and the
    # PRODUCTION model normalizes it *after* returning from the observer.
    # Therefore compare 4 invariant arrays only, not final owndeg.
    last=report.milestones[-1].arrays
    require(np.array_equal(last["positions"],model.part.curnode) and
            np.array_equal(last["strengths"],model.part.strength) and
            np.array_equal(last["distances"],model.part.dist_table) and
            np.array_equal(last["dominance"],model.node.dominance),
            "Observed last completed-iteration state differs from final native model")
    if backend=="numba":
        from experiments.live_numba_milestones import _observe_compiled
        require(bool(_observe_compiled.nopython_signatures),
                "Numba checkpoint observer failed to compile in nopython mode")
    if backend=="cython":
        from experiments.live_cython_milestones import xorshift32_next
        require(report.cached_rowmax_enabled is False and
                report.next_rng_uint32==xorshift32_next(report.final_rng_state_uint32) and
                type(report.final_rng_state_uint32) is int,
                "Original isolated Cython observer's native next-RNG proof malformed")
    return int(report.bytes_copied)

def native_rng_digest(backend):
    if backend=="numpy":
        from experiments.bench_live_numpy_fullfit import rng_digest
        return rng_digest()
    if backend=="numba":
        from experiments.bench_live_numba_fullfit import _next_native_draw
        import struct
        return struct.pack("<d",float(_next_native_draw())).hex()
    # Original Cython binary does not expose its private native RNG state.
    return None

def observed_one(backend,original,features,labels,observed):
    from pcc import ParticleCompetitionAndCooperation
    from time import process_time,perf_counter
    import numpy as np
    model=ParticleCompetitionAndCooperation(impl=backend,n_jobs=1,
                                            update_mode="sequential")
    selected=model._get_backend_fn()
    require(selected is original,
            "Original backend identity mismatch before instance-only observer override")
    graph_wall0=perf_counter()
    graph_cpu0=process_time()
    model.build_graph(features,k_nn=p.KNN,n_jobs=1)
    graph_wall=perf_counter()-graph_wall0
    graph_cpu=process_time()-graph_cpu0
    shape,graph_hash=p.graph_checked(model)
    require(model._get_backend_fn() is original,
            "Original backend identity switched after building fake kNN")
    captures=[]
    model._get_backend_fn=lambda:dispatch  # See below; only per-instance override.
    dispatch=dispatch_for(backend,original,observed,captures)
    # The patched method now returns this strictly controlled dispatch function;
    # the original function's identity was verified immediately above.
    time0=perf_counter()
    cpu0=process_time()
    prediction=model.fit_predict(
        labels.copy(),p_grd=0.5,delta_v=0.1,deltap=1.0,dexp=2.0,
        max_iter=p.ITERATIONS,early_stop=False,es_chk=2000,seed=p.FAKE_SEED)
    elapsed_wall=perf_counter()-time0
    elapsed_cpu=process_time()-cpu0
    require(len(captures)==1 and type(prediction) is np.ndarray and
            prediction.shape==(p.NODES,) and
            np.isin(prediction,[0,1]).all() and
            model.max_iter==p.ITERATIONS and model.early_stop is False and
            0<elapsed_wall<MAX_SYNTHETIC_PAIR_WALL_SECONDS and
            0<elapsed_cpu<MAX_SYNTHETIC_PAIR_WALL_SECONDS,
            "Original 50k synthetic baseline/observer failed or short-circuited")
    info=captures[0].info if observed else captures[0]
    require(info["effective_iterations"]==p.ITERATIONS and
            info["reached_max_iter"] is True and
            info["stop_reason"]=="max_iter",
            "Original 50k full synthetic baseline/observer reported wrong iteration count")
    snapshot_bytes=assert_milestones(captures[0],model,backend) if observed else 0
    h=hashlib.sha256(np.asarray(prediction,dtype="<i8").tobytes(order="C")).hexdigest()
    return {
        "mode":"observed" if observed else "original",
        "full_state_sha256":flatten_state(model),
        "final_prediction_sha256":h,
        "native_rng_next_or_numpy_state":native_rng_digest(backend),
        "graph_shape":list(shape),
        "synthetic_graph_sha256":graph_hash,
        "graph_cpu_seconds":round(graph_cpu,6),
        "graph_wall_seconds":round(graph_wall,6),
        "fit_cpu_seconds":round(elapsed_cpu,6),
        "fit_wall_seconds":round(elapsed_wall,6),
        "snapshot_array_bytes":snapshot_bytes,
        "milestones":list(MILESTONES) if observed else [],
        "internal_effective_iterations":int(info["effective_iterations"]),
        "PCC_fits_synthetic":1,
        "PCC_fits_real":0,
    }

def worker(backend):
    require(backend in BACKENDS and
            os.environ.get("P207Q_FAKE_ONLY")=="1" and
            os.environ.get("PYTHONPATH")==p.k.CORRECT_PATH and
            platform.system()=="Linux" and
            platform.python_version()=="3.12.16" and
            platform.machine().lower() in ("x86_64","amd64"),
            "Only exact original OCI Linux Python 3.12.16 fake-only paired observer worker")
    source_locked()
    original=p.original_native_function_locked(backend)
    features,labels=p.fake_only_features_and_labels()
    results=[observed_one(backend,original,features,labels,observed=flag)
             for flag in (False,True)]
    ref,obs=results
    require(ref["full_state_sha256"]==obs["full_state_sha256"] and
            ref["final_prediction_sha256"]==obs["final_prediction_sha256"] and
            ref["graph_shape"]==obs["graph_shape"] and
            ref["synthetic_graph_sha256"]==obs["synthetic_graph_sha256"] and
            (backend=="cython" or
             ref["native_rng_next_or_numpy_state"]==obs["native_rng_next_or_numpy_state"]),
            "Original full 50k FAKE checkpoint observation modified native baseline state, predictions or RNG")
    process_peak,cgroup_peak=p.cgroup_peak()
    tmpfs=p.scratch_bytes()
    output={
        "schema":WORKER,"backend":backend,
        "data_kind":"DETERMINISTIC_ARTIFICIAL_ONLY_NO_UCI",
        "original_Cython_native_sha256":p.k.lock.NEW_NATIVE["production_pcc_step"],
        "original_isolated_Cython_observer_native_sha256":
            p.k.lock.NEW_NATIVE["isolated_pcc_step_live_research"],
        "P1_scientific_sha256":P1_SHA,
        "exact_original_OCI_config_id":OCI_ID,
        "synthetic_nodes":p.NODES,"synthetic_classes":p.CLASSES,
        "synthetic_fake_label_count":p.KNOWN_LABELS,
        "original_native_fullfit_and_historical_observer":[
            # Avoid exporting even artificial full model/prediction/RNG
            # fingerprints; aggregate parity shown as strict booleans only.
            {key:result[key] for key in (
                "mode","graph_shape","synthetic_graph_sha256",
                "graph_cpu_seconds","graph_wall_seconds",
                "fit_cpu_seconds","fit_wall_seconds","snapshot_array_bytes",
                "milestones","internal_effective_iterations",
                "PCC_fits_synthetic","PCC_fits_real",
            )} for result in results
        ],
        "same_backend_full_mutable_state_parity_verified":True,
        "same_backend_prediction_parity_verified":True,
        "numpy_or_numba_next_rng_parity_verified":backend!="cython",
        "cython_baseline_native_rng_exposed_and_compared":False,
        "exact_eight_observations_captured_in_full_propagation":True,
        "actual_synthetic_fits_completed":2,
        "real_original_P1_fits_completed":0,
        "synthetic_pair_process_peak_RSS_bytes":process_peak,
        "synthetic_pair_cgroup_v2_peak_bytes":cgroup_peak,
        "synthetic_pair_tmpfs_file_payload_bytes":tmpfs,
        "real_P1_50k_checkpoint_resource_upper_bounds_proven":False,
        "P2_execution_authorized":False,
    }
    validate_worker(output,backend)
    return output

def validate_worker(row,backend):
    require(type(row) is dict and backend in BACKENDS and
            row.get("schema")==WORKER and row.get("backend")==backend and
            row.get("data_kind")=="DETERMINISTIC_ARTIFICIAL_ONLY_NO_UCI" and
            row.get("P1_scientific_sha256")==P1_SHA and
            row.get("exact_original_OCI_config_id")==OCI_ID and
            row.get("original_Cython_native_sha256")==
            p.k.lock.NEW_NATIVE["production_pcc_step"] and
            row.get("original_isolated_Cython_observer_native_sha256")==
            p.k.lock.NEW_NATIVE["isolated_pcc_step_live_research"] and
            row.get("synthetic_nodes")==p.NODES and
            row.get("synthetic_classes")==p.CLASSES and
            row.get("synthetic_fake_label_count")==p.KNOWN_LABELS,
            "Original OCI 50k paired observation source/native identity drifted")
    expected={
        "same_backend_full_mutable_state_parity_verified":True,
        "same_backend_prediction_parity_verified":True,
        "numpy_or_numba_next_rng_parity_verified":backend!="cython",
        "cython_baseline_native_rng_exposed_and_compared":False,
        "exact_eight_observations_captured_in_full_propagation":True,
        "actual_synthetic_fits_completed":2,
        "real_original_P1_fits_completed":0,
        "real_P1_50k_checkpoint_resource_upper_bounds_proven":False,
        "P2_execution_authorized":False,
    }
    require(all(row.get(key)==val for key,val in expected.items()),
            "Original 50k paired artificial observer parity/resource status false")
    results=row.get("original_native_fullfit_and_historical_observer")
    require(type(results) is list and len(results)==2 and
            [x.get("mode") for x in results]==["original","observed"],
            "Missing baseline/observer sequence")
    for part in results:
        expected_items={
            "mode","graph_shape","synthetic_graph_sha256",
            "graph_cpu_seconds","graph_wall_seconds",
            "fit_cpu_seconds","fit_wall_seconds","snapshot_array_bytes",
            "milestones","internal_effective_iterations",
            "PCC_fits_synthetic","PCC_fits_real",
        }
        require(type(part) is dict and set(part)==expected_items and
                part["graph_shape"][0]==p.NODES and
                isinstance(part["graph_shape"][1],int) and
                p.KNN<=part["graph_shape"][1]<p.NODES and
                re.fullmatch("[0-9a-f]{64}",part["synthetic_graph_sha256"]) and
                part["internal_effective_iterations"]==p.ITERATIONS and
                part["PCC_fits_synthetic"]==1 and
                part["PCC_fits_real"]==0,
                "Original synthetic fullfit metrics absent/incorrect or unexpected export")
        for field in ("graph_cpu_seconds","graph_wall_seconds",
                      "fit_cpu_seconds","fit_wall_seconds"):
            require(type(part[field]) in (float,int) and
                    0<float(part[field])<MAX_SYNTHETIC_PAIR_WALL_SECONDS,
                    "Original fullfit synthetic CPU/wall invalid: "+field)
    normal,observed=results
    require(normal["snapshot_array_bytes"]==0 and normal["milestones"]==[] and
            observed["milestones"]==list(MILESTONES) and
            type(observed["snapshot_array_bytes"]) is int and
            0<observed["snapshot_array_bytes"]<=MAX_SNAPSHOT_BYTES and
            normal["graph_shape"]==observed["graph_shape"] and
            normal["synthetic_graph_sha256"]==observed["synthetic_graph_sha256"],
            "Original eight synthetic checkpoints incomplete or baseline graph changed")
    for key in ("synthetic_pair_process_peak_RSS_bytes",
                "synthetic_pair_cgroup_v2_peak_bytes"):
        v=row.get(key)
        require(type(v) is int and 0<v<=p.MAX_MEMORY_BYTES,
                "Original OCI paired observer process/cgroup memory invalid")
    v=row.get("synthetic_pair_tmpfs_file_payload_bytes")
    require(type(v) is int and 0<=v<=p.TMPFS_MAX,
            "Original OCI synthetic observer tmpfs limit exceeded")
    return True

def docker_args(backend,name):
    require(backend in BACKENDS and
            re.fullmatch(r"pypcc-p207q-[0-9a-f]{12}",name) is not None,
            "Only original 3 artificial backend types and safe worker names")
    original_name=name.replace("p207q","p207p",1)
    cmd=p.docker_args(backend,original_name)
    cmd[cmd.index("--name")+1]=name
    cmd[cmd.index("P207P_SYNTHETIC_ONLY=1")]="P207Q_FAKE_ONLY=1"
    cmd[cmd.index("experiments.p2_07p_exact_oci_synthetic_fullfit_resource_probe")]=(
        "experiments.p2_07q_exact_oci_synthetic_live_observer_50k")
    return cmd

def run_one(backend):
    name="pypcc-p207q-"+uuid.uuid4().hex[:12]
    try:
        result=subprocess.run(docker_args(backend,name),text=True,
                              stdout=subprocess.PIPE,stderr=subprocess.PIPE,
                              timeout=MAX_SYNTHETIC_PAIR_WALL_SECONDS,check=False)
        require(result.returncode==0 and
                len(result.stdout.encode())<11000 and
                len(result.stderr.encode())<11000,
                "Original OCI paired 50k fake observer failed closed: "+backend)
        try:
            data=json.loads(result.stdout)
        except ValueError as ex:
            raise ObserverBlocked("Original OCI fake observer did not emit JSON") from ex
        validate_worker(data,backend)
        return data
    finally:
        clean=subprocess.run(["docker","rm","-f",name],
                             stdout=subprocess.PIPE,stderr=subprocess.PIPE,
                             timeout=15,check=False)
        require(clean.returncode in (0,1),
                "Original OCI synthetic observer cleanup failed")

def assemble(rows):
    require(type(rows) is list and len(rows)==3 and
            all(validate_worker(row,backend) for row,backend in zip(rows,BACKENDS)) and
            len({row["original_native_fullfit_and_historical_observer"][0]["synthetic_graph_sha256"]
                 for row in rows})==1 and
            len({tuple(row["original_native_fullfit_and_historical_observer"][0]["graph_shape"])
                 for row in rows})==1,
            "Synthetic 50k original observer report must cover same exact fake graph and 3 native backends")
    report={
        "schema":SCHEMA,
        "status":STATUS,
        "P1_scientific_sha256":P1_SHA,
        "original_OCI_archive_sha256":OCI_SHA,
        "original_OCI_config_ID":OCI_ID,
        "historical_original_observers_Git_blob_SHA1":OBSERVER_BLOBS,
        "original_three_native_backend_50k_synthetic_parity_and_resource":rows,
        "synthetic_50k_complete_original_and_observed_fits_executed":6,
        "real_original_P1_PCC_fits_executed":0,
        "original_UCI_data_graph_or_P1_masks_seeds_accessed":False,
        "all_eight_milestones_real_iteration_observation_for_fake_input":True,
        "full_real_UCI_graph_mask_checkpoint_50k_CPU_RSS_disk_upper_bound_proven":False,
        "original_Cython_baseline_rng_state_observable":False,
        "native_OS_compiler_SDK_full_byte_closure_proven":False,
        **NO_GO,
    }
    return validate_report(report)

def validate_report(report):
    require(type(report) is dict and
            report.get("schema")==SCHEMA and
            report.get("status")==STATUS and
            report.get("P1_scientific_sha256")==P1_SHA and
            report.get("original_OCI_archive_sha256")==OCI_SHA and
            report.get("original_OCI_config_ID")==OCI_ID and
            report.get("historical_original_observers_Git_blob_SHA1")==OBSERVER_BLOBS and
            report.get("synthetic_50k_complete_original_and_observed_fits_executed")==6 and
            report.get("real_original_P1_PCC_fits_executed")==0 and
            report.get("original_UCI_data_graph_or_P1_masks_seeds_accessed") is False and
            report.get("all_eight_milestones_real_iteration_observation_for_fake_input") is True and
            report.get("full_real_UCI_graph_mask_checkpoint_50k_CPU_RSS_disk_upper_bound_proven") is False and
            report.get("original_Cython_baseline_rng_state_observable") is False and
            report.get("native_OS_compiler_SDK_full_byte_closure_proven") is False and
            all(report.get(key)==expected for key,expected in NO_GO.items()),
            "Synthetic-only observed original OCI report claims real authorization or wrong frozen scientific identity")
    rows=report.get("original_three_native_backend_50k_synthetic_parity_and_resource")
    require(type(rows) is list and len(rows)==3 and
            all(validate_worker(x,backend) for x,backend in zip(rows,BACKENDS)) and
            len({x["original_native_fullfit_and_historical_observer"][0]["synthetic_graph_sha256"]
                 for x in rows})==1,
            "Three identical fake original graph backends with exact 8 checkpoints mandatory")
    return report

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker",action="store_true")
    parser.add_argument("--backend",choices=BACKENDS)
    parser.add_argument("--existing-original-OCI",type=Path)
    parser.add_argument("--output",type=Path)
    args=parser.parse_args()
    if args.worker:
        require(args.backend in BACKENDS and args.existing_original_OCI is None and
                args.output is None, "Only one research fake observer in OCI child")
        output=worker(args.backend)
        print(canonical(output).decode("ascii"))
        return
    require(args.backend is None and args.existing_original_OCI is not None and
            args.output is not None and not args.output.exists() and
            platform.system()=="Linux",
            "Host must explicitly provide old original immutable OCI and new report file")
    source_locked()
    from experiments import p2_07e_locked_oci_cgroup_resource_supervisor as sup
    loaded=sup.verify_original_image(args.existing_original_OCI.resolve())
    require(loaded["loaded_exact_original_image_ID"]==OCI_ID,
            "Original OCI image ID altered")
    report=assemble([run_one(backend) for backend in BACKENDS])
    with args.output.open("xb") as fd:
        fd.write(json.dumps(report,sort_keys=True,indent=2).encode("ascii")+b"\n")
        fd.flush()
        os.fsync(fd.fileno())
    require(validate_report(json.loads(args.output.read_text(encoding="utf-8"))),
            "Persisted original OCI fake observer evidence failed STOP gate")
    print(json.dumps({
        "status":STATUS,
        "six_original_backend_fake_50k_fits":6,
        "review_only_sha256_NOT_P2_execution":digest(report),
        "P2_execution_preflight_sha256":None,
        "P2_execution_authorized":False,
        "real_PCC_fits_executed":0
    },sort_keys=True))

if __name__=="__main__":
    try:
        main()
    except BaseException as ex:
        label=(str(ex)[:150] if isinstance(ex,ObserverBlocked) else type(ex).__name__)
        print("P207Q_SAFE_FAILURE|"+label,file=sys.stderr)
        sys.exit(1)
