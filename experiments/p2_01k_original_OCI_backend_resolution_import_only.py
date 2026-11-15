"""P2-01K: exact archived OCI *backend resolution* import-only (NO PCC).

Tests one independently isolated container using the historical graph-only
PYTHONPATH=/workspace and one using the intended future three-backend
PYTHONPATH=/workspace:/opt/p2-native:/opt/p2-source-helpers.

Verifies Cython fallback risk and makes the exact 3 original backends
importable without running propagation, JIT, scientific data, masks, graph,
model fit or updating native binaries. The original OCI and master are never
changed. This is a prospective runtime integration finding, not P2 GO.
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
import uuid

from experiments import p2_07m_exact_oci_original_graph_preparation_no_fit as m
from experiments import p2_01i_verify_exact_exported_runtime_image as lock

SCHEMA="pypcc-P2-01K-EXACT-OCI-original-production-backend-resolution-import-only/v1"
WORKER="pypcc-P2-01K-exact-original-OCI-backend-import-only-worker/v1"
STATUS="CYTHON_FALLBACK_PATH_GAP_DETECTED_THREE_BACKENDS_EXACTLY_RESOLVED_WITH_PATH_FIX_NO_FITS"
IMAGE_ID=lock.EXPECTED_IMAGE_ID
IMAGE_SHA=lock.EXPECTED_IMAGE_SHA
P1_SHA=lock.P1_SHA
ROOT=m.ROOT
FALLBACK_PATH="/workspace"
CORRECT_PATH="/workspace:/opt/p2-native:/opt/p2-source-helpers"
MODES=("missing_native_path","correct_three_backends")

class BackendLockBlocked(ValueError):
    pass

def require(ok,msg):
    if not ok:
        raise BackendLockBlocked(msg)

def canonical(data):
    return json.dumps(data,sort_keys=True,separators=(",",":"),
                      ensure_ascii=True,allow_nan=False).encode("ascii")

def digest(data):
    return hashlib.sha256(canonical(data)).hexdigest()

def file_sha(path):
    return m.sha_file(path)

def original_three_backends():
    """Only import/resolve constructor functions; NEVER call a propagator."""
    import pcc
    from pcc import ParticleCompetitionAndCooperation
    import pcc_numpy
    import pcc_numba
    require(Path(pcc.__file__).resolve()==(ROOT/"pcc.py").resolve() and
            Path(pcc_numpy.__file__).resolve()==(ROOT/"pcc_numpy.py").resolve() and
            Path(pcc_numba.__file__).resolve()==(ROOT/"pcc_numba.py").resolve() and
            pcc._HAS_NUMBA is True,
            "Wrong production source or original Numba backend import failed")
    def fn(name):
        # Instantiation only, no graph creation and no fit.
        model=ParticleCompetitionAndCooperation(impl=name,n_jobs=1,
                                                update_mode="sequential")
        require(model.neib_list is None and model.part is None and model.node is None,
                "Backend function resolution unexpectedly constructed scientific state")
        return model._get_backend_fn()

    np_fn=fn("numpy")
    numba_fn=fn("numba")
    cy_fn=fn("cython")
    require(np_fn is pcc_numpy.pcc_propagate_numpy and
            numba_fn is pcc_numba.pcc_propagate_numba,
            "Original NumPy or Numba backend silently changed")
    return {"pcc":pcc, "cython":cy_fn,"numpy":np_fn,"numba":numba_fn}

def worker(mode):
    require(mode in MODES and
            platform.system()=="Linux" and
            platform.python_version()=="3.12.16" and
            platform.machine().lower() in ("amd64","x86_64"),
            "Exact original Linux OCI and the two import-only modes required")
    m.verify_source_blobs()
    from experiments import p2_07c_consolidated_execution_envelope_no_fit as p1
    from experiments import p2_01h_import_only_oci_selfcheck as inside
    frozen=p1.read_approved_P1()
    require(frozen["approvals_and_execution"]["P2_execution_approved"] is False and
            frozen["fit"]["max_iter"]==50000 and
            frozen["fit"]["total_proposed_fits"]==72,
            "Original P1 approved exploratory design/execution gates drifted")
    img=json.loads(inside.MANIFEST.read_text(encoding="utf-8"))
    require(img.get("P1_scientific_sha256")==P1_SHA and
            img.get("P2_execution_authorized") is False and
            img.get("P2_01_final_execution_toolchain_lock_pass") is False,
            "Image manifest must be original unapproved P2 candidate")
    exact=img.get("P2_01F_exact_new_native")
    require(type(exact) is dict and
            set(exact)=={"production_pcc_step","isolated_pcc_step_live_research"},
            "Image must contain both original byte-pinned Cython binaries")
    # Docker caller overrides PYTHONPATH but not installed interpreter trees.
    expected_path=FALLBACK_PATH if mode==MODES[0] else CORRECT_PATH
    require(os.environ.get("PYTHONPATH")==expected_path,
            "Original OCI PYTHONPATH does not match isolated comparison")
    import importlib.util
    native_info=exact["production_pcc_step"]
    native=Path("/opt/p2-native")/native_info["filename"]
    observer_info=exact["isolated_pcc_step_live_research"]
    observer=Path("/opt/p2-native")/observer_info["filename"]
    require(native.is_file() and observer.is_file() and
            file_sha(native)==lock.NEW_NATIVE["production_pcc_step"] and
            file_sha(observer)==lock.NEW_NATIVE["isolated_pcc_step_live_research"],
            "Frozen native modules physically present but byte-altered")

    # Need path *and* actual production function identity, not mere existence
    # in the archived OCI.
    loaded=original_three_backends()
    pcc=loaded["pcc"]
    if mode==MODES[0]:
        require(importlib.util.find_spec("pcc_step") is None and
                pcc._HAS_CYTHON is False and
                loaded["cython"] is loaded["numba"],
                "Old /workspace-only config did NOT exhibit expected silent Cython->Numba fallback")
        return {
            "schema":WORKER,
            "mode":mode,
            "P1_scientific_sha256":P1_SHA,
            "existing_exact_OCI_config_id":IMAGE_ID,
            "original_Cython_ELF_stored_in_image_SHA256":file_sha(native),
            "original_observer_ELF_stored_in_image_SHA256":file_sha(observer),
            "cython_module_available_on_PYTHONPATH":False,
            "explicit_cython_request_resolved_actual_backend":"numba",
            "numpy_backend_correct":True,
            "numba_backend_correct":True,
            "cython_backend_correct":False,
            "Cython_live_observer_imported":False,
            "model_graphs_built":0,
            "Numba_JIT_specializations_compiled":0,
            "PCC_fits_and_kernel_propagations_executed":0,
            "P2_execution_authorized":False,
        }

    import importlib
    pyx=importlib.import_module("pcc_step")
    live=importlib.import_module("pcc_step_live_research")
    require(pcc._HAS_CYTHON is True and
            Path(pyx.__file__).resolve()==native.resolve() and
            Path(live.__file__).resolve()==observer.resolve() and
            file_sha(pyx.__file__)==lock.NEW_NATIVE["production_pcc_step"] and
            file_sha(live.__file__)==lock.NEW_NATIVE["isolated_pcc_step_live_research"] and
            loaded["cython"] is pyx.pcc_propagate and
            loaded["cython"] is not loaded["numba"],
            "Cython backend fallback was not eliminated or exact native observer import failed")
    return {
        "schema":WORKER,
        "mode":mode,
        "P1_scientific_sha256":P1_SHA,
        "existing_exact_OCI_config_id":IMAGE_ID,
        "original_Cython_ELF_stored_in_image_SHA256":file_sha(native),
        "original_observer_ELF_stored_in_image_SHA256":file_sha(observer),
        "cython_module_available_on_PYTHONPATH":True,
        "explicit_cython_request_resolved_actual_backend":"cython",
        "numpy_backend_correct":True,
        "numba_backend_correct":True,
        "cython_backend_correct":True,
        "Cython_live_observer_imported":True,
        "model_graphs_built":0,
        "Numba_JIT_specializations_compiled":0,
        "PCC_fits_and_kernel_propagations_executed":0,
        "P2_execution_authorized":False,
    }

def validate_worker(row,mode):
    require(type(row) is dict and mode in MODES and
            row.get("schema")==WORKER and row.get("mode")==mode and
            row.get("P1_scientific_sha256")==P1_SHA and
            row.get("existing_exact_OCI_config_id")==IMAGE_ID and
            row.get("original_Cython_ELF_stored_in_image_SHA256")==
                lock.NEW_NATIVE["production_pcc_step"] and
            row.get("original_observer_ELF_stored_in_image_SHA256")==
                lock.NEW_NATIVE["isolated_pcc_step_live_research"] and
            row.get("numpy_backend_correct") is True and
            row.get("numba_backend_correct") is True and
            row.get("model_graphs_built")==0 and
            row.get("Numba_JIT_specializations_compiled")==0 and
            row.get("PCC_fits_and_kernel_propagations_executed")==0 and
            row.get("P2_execution_authorized") is False,
            "Original runtime did not respect exact immutable no-fit import-only contract")
    corrected=mode==MODES[1]
    require(row.get("cython_module_available_on_PYTHONPATH") is corrected and
            row.get("explicit_cython_request_resolved_actual_backend")==
                ("cython" if corrected else "numba") and
            row.get("cython_backend_correct") is corrected and
            row.get("Cython_live_observer_imported") is corrected,
            "Original OCI backend fallback diagnostic or 3-backend fix was falsified")
    return True

def docker_args(mode,name):
    require(mode in MODES and
            re.fullmatch(r"pypcc-p201k-[0-9a-f]{12}",name),
            "Only exact original no-fit OCI backend resolution modes accepted")
    pth=FALLBACK_PATH if mode==MODES[0] else CORRECT_PATH
    return ["docker","run","--rm","--name",name,
            "--platform","linux/amd64","--network","none","--read-only",
            "--user","65534:65534","--cap-drop","ALL",
            "--security-opt","no-new-privileges","--pids-limit","128",
            "--memory","2g","--memory-swap","2g","--cpus","1",
            "--tmpfs","/tmp:rw,nosuid,nodev,noexec,mode=1777,size=32m",
            "--env","PYTHONDONTWRITEBYTECODE=1",
            "--env","PYTHONNOUSERSITE=1",
            "--env","OPENBLAS_NUM_THREADS=1",
            "--env","OMP_NUM_THREADS=1",
            "--env","NUMBA_CACHE_DIR=/tmp/numba-cache",
            "--env",f"PYTHONPATH={pth}",
            "--entrypoint",m.PYTHON_IN_OCI,
            "--mount",f"type=bind,src={ROOT.resolve()},dst=/workspace,readonly",
            m.IMAGE_TAG,"-m",
            "experiments.p2_01k_original_OCI_backend_resolution_import_only",
            "--worker-mode",mode]

def one(mode):
    name="pypcc-p201k-"+uuid.uuid4().hex[:12]
    try:
        result=subprocess.run(docker_args(mode,name),text=True,
                              stdout=subprocess.PIPE,stderr=subprocess.PIPE,
                              timeout=90,check=False)
        if result.returncode != 0:
            markers=[s for s in result.stderr.splitlines()
                     if s.startswith("P201K_SAFE_ERROR|") and len(s)<220]
            for x in markers[-1:]:
                print(x,file=sys.stderr)
            raise BackendLockBlocked("Original OCI import-only backend resolution failed: "+mode)
        require(len(result.stdout.encode("utf-8"))<=8192 and
                len(result.stderr.encode("utf-8"))<=8192,
                "Non-source-safe oversized backend import report")
        try:
            row=json.loads(result.stdout)
        except ValueError as e:
            raise BackendLockBlocked("Original backend import-only JSON not parseable") from e
        validate_worker(row,mode)
        return row
    finally:
        clean=subprocess.run(["docker","rm","-f",name],
                             stdout=subprocess.PIPE,stderr=subprocess.PIPE,
                             check=False,timeout=12)
        require(clean.returncode in (0,1),
                "Original exact OCI import-only container cleanup failed")

def assemble(rows):
    require(type(rows) is list and len(rows)==2 and
            all(validate_worker(row,mode) for row,mode in zip(rows,MODES)),
            "Both original OCI import modes required to demonstrate fallback and exact fix")
    report={
        "schema":SCHEMA,
        "status":STATUS,
        "P1_scientific_sha256":P1_SHA,
        "exact_original_OCI_export_SHA256":IMAGE_SHA,
        "exact_original_OCI_config_ID":IMAGE_ID,
        "preexisting_image_default_PYTHONPATH":"/opt/p2-source-helpers",
        "prior_graph_only_import_mount_override_excludes_native_path":True,
        "only_working_future_backend_resolution_PYTHONPATH":CORRECT_PATH,
        "source_free_baseline_AND_corrected_backend_resolution":rows,
        "unchanged_original_Cython_and_observer_ELF_bytes_proven":True,
        "actual_P1_50000_iteration_Cython_and_NumPy_and_Numba_full_fit_observer_dispatch_proven":False,
        "actual_full_original_PCC_50k_CPU_wall_RSS_disk_resource_bounds_proven":False,
        "complete_compiler_SDK_OS_source_build_closure_proven":False,
        "image_durable_off_GHA_archive_copy_proven":False,
        "P2_01_final_execution_toolchain_lock_pass":False,
        "P2_07_full_resources_gate_pass":False,
        "P2_execution_preflight_sha256":None,
        "P2_execution_authorized":False,
        "real_PCC_fits_executed":0,
    }
    validate_report(report)
    return report

def validate_report(r):
    require(type(r) is dict and r.get("schema")==SCHEMA and
            r.get("status")==STATUS and
            r.get("P1_scientific_sha256")==P1_SHA and
            r.get("exact_original_OCI_export_SHA256")==IMAGE_SHA and
            r.get("exact_original_OCI_config_ID")==IMAGE_ID and
            r.get("preexisting_image_default_PYTHONPATH")=="/opt/p2-source-helpers" and
            r.get("prior_graph_only_import_mount_override_excludes_native_path") is True and
            r.get("only_working_future_backend_resolution_PYTHONPATH")==CORRECT_PATH,
            "Immutable original OCI runtime paths/identities drifted")
    rows=r.get("source_free_baseline_AND_corrected_backend_resolution")
    require(type(rows) is list and len(rows)==2 and
            all(validate_worker(row,mode) for row,mode in zip(rows,MODES)),
            "Baseline Cython fallback and corrected triple backend imports required")
    for key,expected in (
        ("unchanged_original_Cython_and_observer_ELF_bytes_proven",True),
        ("actual_P1_50000_iteration_Cython_and_NumPy_and_Numba_full_fit_observer_dispatch_proven",False),
        ("actual_full_original_PCC_50k_CPU_wall_RSS_disk_resource_bounds_proven",False),
        ("complete_compiler_SDK_OS_source_build_closure_proven",False),
        ("image_durable_off_GHA_archive_copy_proven",False),
        ("P2_01_final_execution_toolchain_lock_pass",False),
        ("P2_07_full_resources_gate_pass",False),
        ("P2_execution_preflight_sha256",None),
        ("P2_execution_authorized",False),
        ("real_PCC_fits_executed",0),
    ):
        require(r.get(key)==expected,
                "Forbidden backend, toolchain or scientific execution-GO claim: "+key)
    return True

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker-mode",choices=MODES)
    parser.add_argument("--existing-archived-OCI",type=Path)
    parser.add_argument("--output",type=Path)
    opts=parser.parse_args()
    if opts.worker_mode:
        require(opts.existing_archived_OCI is None and opts.output is None,
                "Import worker accepts no scientific data or external artifacts")
        result=worker(opts.worker_mode)
        validate_worker(result,opts.worker_mode)
        print(canonical(result).decode("ascii"))
        return
    require(opts.existing_archived_OCI is not None and
            opts.output is not None and
            not opts.output.exists() and
            platform.system()=="Linux",
            "Host requires previous original OCI bytes and new receipt; NO FIT")
    m.verify_source_blobs()
    from experiments import p2_07e_locked_oci_cgroup_resource_supervisor as guard
    old=guard.verify_original_image(opts.existing_archived_OCI.resolve())
    require(old["loaded_exact_original_image_ID"]==IMAGE_ID,
            "Original byte-locked OCI did not load unchanged")
    report=assemble([one(mode) for mode in MODES])
    with opts.output.open("xb") as f:
        f.write(json.dumps(report,sort_keys=True,indent=2).encode("utf-8")+b"\n")
        f.flush()
        os.fsync(f.fileno())
    require(validate_report(json.loads(opts.output.read_text(encoding="utf-8"))),
            "Persisted no-fit backend-resolve evidence invalid")
    print(json.dumps({
        "status":STATUS,
        "actual_Cython_fallback_before_path_correction":"numba",
        "corrected_three_exact_backend_functions_imported":True,
        "review_only_SHA256_NOT_EXECUTION":digest(report),
        "P2_execution_preflight_sha256":None,
        "P2_execution_authorized":False,
        "real_PCC_fits_executed":0,
    },sort_keys=True))

if __name__=="__main__":
    try:
        main()
    except BaseException as e:
        import traceback
        frames=traceback.extract_tb(e.__traceback__)
        last=frames[-1] if frames else None
        name=Path(last.filename).name if last else "unknown"
        line=last.lineno if last else 0
        text=(str(e) if isinstance(e,BackendLockBlocked)
              else type(e).__name__)
        print("P201K_SAFE_ERROR|"+name+"|"+str(line)+"|"+text[:120],
              file=sys.stderr)
        sys.exit(1)
