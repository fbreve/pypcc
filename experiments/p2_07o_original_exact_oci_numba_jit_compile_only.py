"""P2-07O — original binary OCI Numba specialization, COMPILE ONLY.

Does NOT call the compiled CPUDispatcher, the model, pcc_step, fit_predict,
any propagator, graph builder, or UCI source. The exact original Numba
*signature* is compiled from its immutable production source under the
P2-01I/J byte-locked original OCI, without allocating scientific input
matrices or generating labels/masks/seeds.

This observes Numba's nopython JIT compile CPU/wall, process VmHWM, and
mapped file-backed ELF/DSO identity. It is not a full 50k PCC-fit peak,
not complete source/toolchain closure, and not P2 execution authorization.
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

from experiments import p2_07m_exact_oci_original_graph_preparation_no_fit as m

SCHEMA="pypcc-P2-07O-exact-original-OCI-Numba-nopython-JIT-compile-only/v1"
WORKER_SCHEMA="pypcc-P2-07O-exact-original-OCI-Numba-JIT-worker/v1"
STATUS="ORIGINAL_NUMBA_SOURCE_SIGNATURE_COMPILED_NO_KERNEL_CALLS_NO_FITS"
IMAGE_ID=m.IMAGE_ID
IMAGE_SHA=m.IMAGE_SHA
P1_SHA=m.P1_SHA
ROOT=m.ROOT
IMAGE_TAG=m.IMAGE_TAG
PYTHON_IN_OCI=m.PYTHON_IN_OCI
CANDIDATE_CAP=2*1024**3

# Original pcc.py passes these 27 argument dtypes and memory layouts to
# _pcc_propagate_numba_sequential when c < 8 (P1 datasets have 2/3 classes).
# Numba specializes by element type, dimensions and layout, not row counts.
# This remains a declaration, not actual propagation or a data-size bound.
TYPE_SPEC=(
    "i64_mat_C", "i64_vec_C", "i64_vec_C", "f64", "f64", "i64",
    "f64_vec_C", "i64_vec_C", "i64_vec_C", "f64_vec_C",
    "u8_mat_F", "f64_mat_F", "f64_mat_F", "f64", "f64",
    "f64_vec_C", "f64_vec_C", "f64_vec_C", "f64_vec_C",
    "f64_vec_C", "f64_vec_C", "f64_vec_C", "i64", "bool",
    "i64", "i64", "bool",
)

class CompileOnlyBlocked(ValueError):
    pass

def require(ok, reason):
    if not ok:
        raise CompileOnlyBlocked(reason)

def canonical(x):
    return json.dumps(x,sort_keys=True,ensure_ascii=True,allow_nan=False,
                      separators=(",",":")).encode("ascii")

def digest(x):
    return hashlib.sha256(canonical(x)).hexdigest()

def signature_types():
    """Build Numba *type objects only*: no particle-state arrays allocated."""
    import numba
    t=numba.types
    kinds={
        "i64":t.int64,
        "f64":t.float64,
        "bool":t.boolean,
        "i64_vec_C":t.Array(t.int64,1,"C"),
        "i64_mat_C":t.Array(t.int64,2,"C"),
        "f64_vec_C":t.Array(t.float64,1,"C"),
        "f64_mat_F":t.Array(t.float64,2,"F"),
        "u8_mat_F":t.Array(t.uint8,2,"F"),
    }
    return tuple(kinds[name] for name in TYPE_SPEC)

def kernel_import_and_compile_only():
    """Worker only: no invocation of any Numba CPUDispatcher."""
    require(platform.system()=="Linux" and
            platform.machine().lower() in ("x86_64","amd64") and
            platform.python_version()=="3.12.16",
            "Only original Linux 3.12.16 OCI runtime may compile")
    m.verify_source_blobs()
    import importlib.metadata
    for package,version in (("numba","0.68.1"),):
        # Need exact lock check; accepted only if matches recorded OCI
        require(importlib.metadata.version(package)==version,
                "Unexpected candidate Numba version")
    import numba
    import pcc_numba
    from experiments import p2_01h_import_only_oci_selfcheck as selfcheck
    from experiments import p2_07c_consolidated_execution_envelope_no_fit as c
    from pathlib import Path

    require(Path(pcc_numba.__file__).resolve()==(ROOT/"pcc_numba.py").resolve(),
            "Wrong original Python Numba production source imported")
    approved=c.read_approved_P1()
    require(approved["fit"]["max_iter"]==50000 and
            approved["fit"]["early_stop"] is False and
            len(approved["selected_datasets"])==3 and
            approved["approvals_and_execution"]["P2_execution_approved"] is False,
            "P1 original scientific protocol drifted")
    original_manifest=json.loads(selfcheck.MANIFEST.read_text(encoding="utf-8"))
    require(original_manifest["P1_scientific_sha256"]==P1_SHA and
            original_manifest["P2_execution_authorized"] is False,
            "Original exact image manifest incorrectly authorizes execution")

    target=pcc_numba._pcc_propagate_numba_sequential
    nested=pcc_numba._pcc_step_numba_sequential
    seed=pcc_numba._seed_numba_rng
    require(type(target).__name__=="CPUDispatcher" and
            type(nested).__name__=="CPUDispatcher" and
            type(seed).__name__=="CPUDispatcher" and
            len(target.signatures)==0 and
            len(nested.signatures)==0 and
            len(seed.signatures)==0 and
            len(TYPE_SPEC)==27,
            "Fresh original nopython dispatchers expected before compilation")

    original_maps=selfcheck.loaded_file_backed_ELFs()
    started=time.monotonic()
    started_cpu=time.process_time()
    sig=signature_types()
    # This is Numba's compile API, NOT the CPUDispatcher call API.
    target.compile(sig)
    # Original pcc_propagate_numba always invokes this seed dispatcher
    # if a scientific P1 seed is supplied. Compile its signature ONLY.
    seed.compile((numba.types.int64,))
    wall=round(time.monotonic()-started,6)
    cpu=round(time.process_time()-started_cpu,6)
    require(target.signatures==[sig] and
            len(nested.signatures)>=1 and
            seed.signatures==[(numba.types.int64,)] and
            target.nopython_signatures and
            nested.nopython_signatures and
            seed.nopython_signatures,
            "Original CPUDispatcher did not compile nopython signatures")
    later_maps=selfcheck.loaded_file_backed_ELFs()
    original_by_path={x["path"]:x for x in original_maps}
    later_by_path={x["path"]:x for x in later_maps}
    require(all(path in later_by_path and later_by_path[path]==val
                for path,val in original_by_path.items()),
            "Original file-backed native DSO changed/disappeared during JIT")
    delta=[later_by_path[k] for k in sorted(set(later_by_path)-set(original_by_path))]
    # Digest all mapped file names and original exact bytes, but do not
    # disclose arbitrary source rows or journal per-array details.
    rss=m.vm_kib("VmRSS")*1024
    peak=max(rss,m.vm_kib("VmHWM")*1024)
    row={
        "schema":WORKER_SCHEMA,
        "status":STATUS,
        "P1_scientific_sha256":P1_SHA,
        "exact_existing_image_config_id":IMAGE_ID,
        "original_production_numba_git_blob":m.PRODUCTION_BLOBS["pcc_numba.py"],
        "original_source_only_argument_types_spec_SHA256":digest(TYPE_SPEC),
        "original_propagation_JIT_compiled_nopython_dispatcher_signatures":1,
        "original_nested_step_JIT_nopython_signatures_at_least_one":True,
        "original_seed_JIT_nopython_signatures":1,
        "JIT_signature_compilation_wall_seconds":wall,
        "JIT_signature_compilation_process_CPU_seconds":cpu,
        "JIT_process_peak_resident_bytes":peak,
        "JIT_process_RSS_after_compile_bytes":rss,
        "imported_file_backed_objects_before_JIT":len(original_maps),
        "imported_file_backed_objects_after_JIT":len(later_maps),
        "imported_file_backed_objects_after_JIT_manifest_SHA256":digest(later_maps),
        "new_file_backed_objects_mapped_during_JIT_count":len(delta),
        "new_file_backed_objects_mapped_during_JIT_manifest_SHA256":digest(delta),
        "this_IS_NOT_actual_PCC_50k_fit_peak_RSS_or_CPU":True,
        "original_scientific_matrices_graphs_masks_or_seeds_used":0,
        "Numba_dispatchers_invoked_for_propagation_or_RNG":0,
        "PCC_fits_or_propagations_executed":0,
        "P2_execution_authorized":False,
    }
    validate_worker(row)
    print(canonical(row).decode("ascii"))

def validate_worker(row):
    require(type(row) is dict and row.get("schema")==WORKER_SCHEMA and
            row.get("status")==STATUS and
            row.get("P1_scientific_sha256")==P1_SHA and
            row.get("exact_existing_image_config_id")==IMAGE_ID and
            row.get("original_production_numba_git_blob")==
                m.PRODUCTION_BLOBS["pcc_numba.py"] and
            row.get("original_source_only_argument_types_spec_SHA256")==digest(TYPE_SPEC) and
            row.get("original_propagation_JIT_compiled_nopython_dispatcher_signatures")==1 and
            row.get("original_nested_step_JIT_nopython_signatures_at_least_one") is True and
            row.get("original_seed_JIT_nopython_signatures")==1 and
            row.get("this_IS_NOT_actual_PCC_50k_fit_peak_RSS_or_CPU") is True and
            row.get("original_scientific_matrices_graphs_masks_or_seeds_used")==0 and
            row.get("Numba_dispatchers_invoked_for_propagation_or_RNG")==0 and
            row.get("PCC_fits_or_propagations_executed")==0 and
            row.get("P2_execution_authorized") is False,
            "Wrong exact-OCI Numba compile-only scientific STOP contract")
    for name in ("JIT_signature_compilation_wall_seconds",
                 "JIT_signature_compilation_process_CPU_seconds"):
        x=row.get(name)
        require(type(x) in (int,float) and 0<=x<240,
                "Original JIT compile timing invalid: "+name)
    rss=row.get("JIT_process_RSS_after_compile_bytes")
    peak=row.get("JIT_process_peak_resident_bytes")
    require(type(rss) is int and type(peak) is int and 0<rss<=peak<2**40,
            "Original exact-OCI Numba process peak memory sample invalid")
    before=row.get("imported_file_backed_objects_before_JIT")
    after=row.get("imported_file_backed_objects_after_JIT")
    extra=row.get("new_file_backed_objects_mapped_during_JIT_count")
    require(type(before) is int and type(after) is int and
            type(extra) is int and 5<=before<=after<=512 and
            0<=extra<=after-before,
            "JIT native library inventory unexpected")
    for key in ("imported_file_backed_objects_after_JIT_manifest_SHA256",
                "new_file_backed_objects_mapped_during_JIT_manifest_SHA256"):
        x=row.get(key)
        require(type(x) is str and len(x)==64 and
                set(x)<=set("0123456789abcdef"),
                "Native DSO manifest digest invalid")
    return True

def docker_args(name):
    require(re.fullmatch(r"pypcc-p207o-[0-9a-f]{12}",name) is not None,
            "Unsafe exact-OCI Numba compile-only process name")
    return [
        "docker","run","--rm","--name",name,
        "--platform","linux/amd64","--network","none","--read-only",
        "--user","65534:65534","--cap-drop","ALL",
        "--security-opt","no-new-privileges","--pids-limit","128",
        "--memory","2g","--memory-swap","2g","--cpus","1",
        "--tmpfs","/tmp:rw,nosuid,nodev,noexec,mode=1777,size=128m",
        "--env","PYTHONNOUSERSITE=1",
        "--env","PYTHONDONTWRITEBYTECODE=1",
        "--env","PYTHONPATH=/workspace",
        "--env","NUMBA_CACHE_DIR=/tmp/numba-cache",
        "--env","OPENBLAS_NUM_THREADS=1",
        "--env","OMP_NUM_THREADS=1",
        "--env","MKL_NUM_THREADS=1",
        "--entrypoint",PYTHON_IN_OCI,
        "--mount",f"type=bind,src={ROOT.resolve()},dst=/workspace,readonly",
        IMAGE_TAG,"-m",
        "experiments.p2_07o_original_exact_oci_numba_jit_compile_only",
        "--worker",
    ]

def run_compile_only(*, runner=subprocess.run):
    name="pypcc-p207o-"+uuid.uuid4().hex[:12]
    command=docker_args(name)
    try:
        output=runner(command,stdout=subprocess.PIPE,stderr=subprocess.PIPE,
                      timeout=240,text=True,check=False)
        if output.returncode != 0:
            # Log only sanitizer-produced error records. Arbitrary C compiler
            # messages and source prints are not exported.
            safe=[x for x in output.stderr.splitlines()
                  if x.startswith("P207O_SAFE_ERROR|") and len(x)<220]
            for line in safe[-2:]:
                print(line,file=sys.stderr)
            raise CompileOnlyBlocked(
                "Original exact-OCI Numba compile-only subprocess failed")
        require(len(output.stdout.encode("utf-8"))<10000 and
                len(output.stderr.encode("utf-8"))<10000,
                "Original JIT compile-only worker emitted excess output")
        try:
            row=json.loads(output.stdout)
        except ValueError as exc:
            raise CompileOnlyBlocked(
                "Original JIT compile-only worker did not emit source-free JSON") from exc
        validate_worker(row)
        return row
    finally:
        # Enforce cleanup after success, failed JIT, wall timeout or OOM.
        cleanup=subprocess.run(["docker","rm","-f",name],
                               stdout=subprocess.PIPE,stderr=subprocess.PIPE,
                               timeout=15,check=False)
        require(cleanup.returncode in (0,1),
                "Original JIT compile-only container removal failed")

def assemble(row):
    validate_worker(row)
    report={
        "schema":SCHEMA,
        "status":STATUS,
        "P1_scientific_sha256":P1_SHA,
        "exact_existing_OCI_archive_SHA256":IMAGE_SHA,
        "exact_existing_OCI_config_ID":IMAGE_ID,
        "original_source_commit_sha":m.PRODUCTION_BLOBS["pcc_numba.py"],
        "candidate_2GiB_process_RSS_limit_NOT_APPROVED":CANDIDATE_CAP,
        "original_OCI_Numba_compile_only_measurement":row,
        "type_signature_is_production_layout_not_real_graph_dimensions":True,
        "real_frozen_P1_masks_and_seeds_not_used":True,
        "JIT_compilation_only_is_NOT_actual_50000_iteration_full_fit_resource_bound":True,
        "further_JIT_runtime_data_dependent_specialization_not_excluded":True,
        "complete_runtime_OS_compiler_SDK_build_byte_closure_proven":False,
        "full_50k_original_PCC_fit_CPU_wall_RSS_disk_resource_bound_proven":False,
        "P2_01_final_execution_toolchain_lock_pass":False,
        "P2_07_full_resources_gate_pass":False,
        "P2_execution_preflight_sha256":None,
        "P2_execution_authorized":False,
        "real_PCC_fits_executed":0,
    }
    validate_report(report)
    return report

def validate_report(report):
    require(type(report) is dict and report.get("schema")==SCHEMA and
            report.get("status")==STATUS and
            report.get("P1_scientific_sha256")==P1_SHA and
            report.get("exact_existing_OCI_archive_SHA256")==IMAGE_SHA and
            report.get("exact_existing_OCI_config_ID")==IMAGE_ID and
            report.get("original_source_commit_sha")==m.PRODUCTION_BLOBS["pcc_numba.py"] and
            report.get("candidate_2GiB_process_RSS_limit_NOT_APPROVED")==CANDIDATE_CAP and
            validate_worker(report.get("original_OCI_Numba_compile_only_measurement")),
            "Original OCI no-execution specialized Numba resource audit tampered")
    for k,x in (
        ("type_signature_is_production_layout_not_real_graph_dimensions",True),
        ("real_frozen_P1_masks_and_seeds_not_used",True),
        ("JIT_compilation_only_is_NOT_actual_50000_iteration_full_fit_resource_bound",True),
        ("further_JIT_runtime_data_dependent_specialization_not_excluded",True),
        ("complete_runtime_OS_compiler_SDK_build_byte_closure_proven",False),
        ("full_50k_original_PCC_fit_CPU_wall_RSS_disk_resource_bound_proven",False),
        ("P2_01_final_execution_toolchain_lock_pass",False),
        ("P2_07_full_resources_gate_pass",False),
        ("P2_execution_preflight_sha256",None),
        ("P2_execution_authorized",False),
        ("real_PCC_fits_executed",0),
    ):
        require(report.get(k)==x,"Forbidden original P2 full-fit/authorization claim: "+k)
    return True

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--worker",action="store_true")
    p.add_argument("--exact-original-OCI",type=Path)
    p.add_argument("--output",type=Path)
    a=p.parse_args()
    if a.worker:
        require(a.exact_original_OCI is None and a.output is None,
                "Exact OCI JIT worker accepts no input files or scientific parameters")
        kernel_import_and_compile_only()
        return
    require(a.exact_original_OCI is not None and a.output is not None and
            not a.output.exists() and platform.system()=="Linux",
            "External host only loads original archived OCI, never overwrites receipt")
    m.verify_source_blobs()
    from experiments import p2_07e_locked_oci_cgroup_resource_supervisor as earlier
    exact=earlier.verify_original_image(a.exact_original_OCI.resolve())
    require(exact["loaded_exact_original_image_ID"]==IMAGE_ID,
            "OCI bytes/config were not original P2-01I")
    report=assemble(run_compile_only())
    with a.output.open("xb") as f:
        f.write(json.dumps(report,sort_keys=True,indent=2).encode("utf-8")+b"\n")
        f.flush()
        os.fsync(f.fileno())
    require(validate_report(json.loads(a.output.read_text(encoding="utf-8"))),
            "Persisted original OCI JIT compile-only result invalid")
    print(json.dumps({
        "status":STATUS,
        "original_OCI_Numba_compile_only_process_peak_RSS_bytes":
            report["original_OCI_Numba_compile_only_measurement"]["JIT_process_peak_resident_bytes"],
        "JIT_compilation_CPU_seconds":
            report["original_OCI_Numba_compile_only_measurement"][
                "JIT_signature_compilation_process_CPU_seconds"],
        "review_only_SHA256_NOT_EXECUTION":digest(report),
        "P2_execution_preflight_sha256":None,
        "P2_execution_authorized":False,
        "real_PCC_fits_executed":0,
    },sort_keys=True))

if __name__=="__main__":
    try:
        main()
    except BaseException as exc:
        # No raw tracebacks or arbitrary exception strings: avoid forwarding
        # any possible source content or personal CI infrastructure details.
        import traceback
        frames=traceback.extract_tb(exc.__traceback__)
        last=frames[-1] if frames else None
        name=Path(last.filename).name if last else "unknown"
        line=last.lineno if last else 0
        message=(str(exc) if isinstance(exc,CompileOnlyBlocked)
                 else type(exc).__name__)
        print("P207O_SAFE_ERROR|"+name+"|"+str(line)+"|"+message[:130],
              file=sys.stderr)
        raise SystemExit(1)
