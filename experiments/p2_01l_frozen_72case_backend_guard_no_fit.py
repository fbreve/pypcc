"""P2-01L — fail-closed backend entrypoint guard for the frozen 72-case study.

RESEARCH PREFLIGHT ONLY: permits resolving the authentic production
NumPy/Numba/Cython function for one P1 case, but never executes that function,
calls fit_predict, constructs a graph, creates masks, or starts any real fit.

This module is suitable for explicit use *before* future scientific work,
but is NOT such a worker and does not grant execution authorization. It
intentionally refuses the historical silent Cython->Numba fallback.
"""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
import uuid

from experiments import p2_01k_original_OCI_backend_resolution_import_only as k
from experiments import p2_07c_consolidated_execution_envelope_no_fit as c
from experiments import p2_07m_exact_oci_original_graph_preparation_no_fit as m

ROOT=m.ROOT
SCHEMA="pypcc-P2-01L-real-original-OCI-P1-72case-backend-guard-NO-FIT/v1"
STATUS="ALL_72_FROZEN_BACKEND_IDENTITIES_RESOLVED_NO_FALLBACK_NO_FITS"
BACKENDS=dict(zip(c.BACKENDS,("numpy","numba","cython")))
P1_SHA=k.P1_SHA
IMAGE_ID=k.IMAGE_ID
IMAGE_SHA=k.IMAGE_SHA
NO_GO={
    "P2_01_final_execution_toolchain_lock_pass":False,
    "P2_07_full_resources_gate_pass":False,
    "P2_execution_preflight_sha256":None,
    "P2_execution_authorized":False,
    "real_PCC_fits_executed":0,
}

class BackendRejected(ValueError):
    pass

def require(condition,why):
    if not condition:
        raise BackendRejected(why)

def canonical(obj):
    return json.dumps(obj,sort_keys=True,ensure_ascii=True,
                      allow_nan=False,separators=(",",":")).encode("ascii")

def sha(obj):
    return hashlib.sha256(canonical(obj)).hexdigest()

def case_lookup(protocol,uci_id,replicate,backend):
    """Validate an exact frozen case ID, WITHOUT touching the model or RNG."""
    require(type(protocol) is dict and
            protocol.get("schema")=="pypcc-P1-scientific-protocol-frozen/v1" and
            protocol.get("status")=="SCIENTIFICALLY_APPROVED_EXPLORATORY_P1_ONLY" and
            protocol.get("approvals_and_execution",{}).get("P2_execution_approved") is False,
            "Original P1 scientific protocol must remain review-only")
    require(type(uci_id) is int and type(replicate) is int and
            type(backend) is str and backend in BACKENDS and
            0<=replicate<8 and
            uci_id in (212,850,292) and
            protocol["fit"]["backend_order"]==list(c.BACKENDS) and
            protocol["fit"]["total_proposed_fits"]==72 and
            protocol["fit"]["max_iter"]==50000 and
            protocol["fit"]["early_stop"] is False,
            "Unapproved backend, case, run length or scientific fit settings")
    case={"uci_id":uci_id,"replicate":replicate,"backend":backend,
          "status":"NOT_RUN_NO_APPROVAL"}
    require(case in c.case_order(protocol),
            "Case ID not in immutable P1 administrative order")
    return case

def assert_selected_exact(requested,model,selected,expected):
    """Strict identity check; do not trust impl name or a callable's __name__."""
    require(requested in BACKENDS and
            type(model.impl) is str and
            model.impl==BACKENDS[requested] and
            model.n_jobs==1 and
            model.update_mode=="sequential" and
            selected is expected and
            callable(selected),
            "BACKEND_IDENTITY_MISMATCH_OR_SILENT_FALLBACK")
    return True

def source_bytes_locked():
    """Original Python blobs, not arbitrary code in import search path."""
    require(m.verify_source_blobs(),"Original Python source changed")
    return True

def assert_case_backend_exact(protocol,uci_id,replicate,backend, *,
                              enforce_original_OCI=True):
    """Returns the selected production function ONLY after strict verification.

    No graph, mask, RNG or propagation is performed. This is an admission
    prerequisite for a FUTURE executor; never authorization on its own.
    """
    case=case_lookup(protocol,uci_id,replicate,backend)
    source_bytes_locked()
    import pcc, pcc_numpy, pcc_numba
    from pcc import ParticleCompetitionAndCooperation
    import pcc_graph
    origins={
        "pcc.py":pcc,
        "pcc_numpy.py":pcc_numpy,
        "pcc_numba.py":pcc_numba,
        "pcc_graph.py":pcc_graph,
    }
    require(all(Path(module.__file__).resolve()==(ROOT/name).resolve()
                for name,module in origins.items()),
            "Actual imported production modules differ from frozen source")
    require(pcc._HAS_NUMBA is True and
            pcc.pcc_propagate_numba is pcc_numba.pcc_propagate_numba and
            pcc.pcc_step_numpy is pcc_numpy.pcc_step_numpy,
            "Original NumPy/Numba imports or production resolver changed")
    name=BACKENDS[backend]
    model=ParticleCompetitionAndCooperation(impl=name,n_jobs=1,
                                            update_mode="sequential")
    require(type(model) is ParticleCompetitionAndCooperation and
            model.neib_list is None and model.part is None and model.node is None,
            "Import-only model must be uninitialized before any scientific fit")
    expected={
        "numpy":pcc_numpy.pcc_propagate_numpy,
        "numba":pcc_numba.pcc_propagate_numba,
    }
    if name=="cython":
        require(pcc._HAS_CYTHON is True,
                "BACKEND_IDENTITY_MISMATCH_OR_SILENT_FALLBACK")
        import pcc_step
        require(callable(getattr(pcc_step,"pcc_propagate",None)),
                "Original Cython propagation function missing")
        if enforce_original_OCI:
            require(Path(pcc_step.__file__).resolve().parent==
                    Path("/opt/p2-native") and
                    m.sha_file(Path(pcc_step.__file__))==
                    k.lock.NEW_NATIVE["production_pcc_step"],
                    "Original Cython production .so is not byte-locked in exact OCI")
        expected["cython"]=pcc_step.pcc_propagate
    selected=model._get_backend_fn()    # resolve, NEVER invoke returned function
    assert_selected_exact(backend,model,selected,expected[name])
    if name=="cython" and enforce_original_OCI:
        import pcc_step_live_research
        require(Path(pcc_step_live_research.__file__).resolve().parent==
                Path("/opt/p2-native") and
                m.sha_file(Path(pcc_step_live_research.__file__))==
                k.lock.NEW_NATIVE["isolated_pcc_step_live_research"],
                "Actual original Cython checkpoint observer .so differs from OCI")
    return selected

def container_args(mode,name):
    require(mode in ("baseline_must_reject","corrected_72_case") and
            re.fullmatch(r"pypcc-p201l-[0-9a-f]{12}",name) is not None,
            "Only original OCI import-only guard audit modes permitted")
    # Reuse the previously validated P2-01K confinement policy verbatim.
    # P2-01K enforces its own container-name prefix; replace only that
    # validated --name token after constructing the same Docker policy.
    k_name=name.replace("p201l","p201k",1)
    cmd=k.docker_args(
        k.MODES[0] if mode=="baseline_must_reject" else k.MODES[1],k_name)
    cmd[cmd.index("--name")+1]=name
    index=cmd.index("experiments.p2_01k_original_OCI_backend_resolution_import_only")
    cmd[index]="experiments.p2_01l_frozen_72case_backend_guard_no_fit"
    cmd[-2:]=["--worker-mode",mode]
    return cmd

def worker(mode):
    require(mode in ("baseline_must_reject","corrected_72_case") and
            platform.system()=="Linux" and
            platform.python_version()=="3.12.16" and
            platform.machine().lower() in ("amd64","x86_64"),
            "Original exact Linux OCI only; no scientific source inputs")
    expected=k.FALLBACK_PATH if mode=="baseline_must_reject" else k.CORRECT_PATH
    require(os.environ.get("PYTHONPATH")==expected,
            "Original candidate OCI import path differs from intended mode")
    protocol=c.read_approved_P1()
    require(c.sha256(c.canonical(protocol))==P1_SHA,
            "Frozen P1 exact SHA mismatch")
    source_bytes_locked()
    if mode=="baseline_must_reject":
        # This is a STRICT expected rejection, not an arbitrary subprocess
        # failure, unavailable file, missing package or synthetic PCC outcome.
        case=case_lookup(protocol,212,0,c.BACKENDS[2])
        try:
            assert_case_backend_exact(protocol,212,0,c.BACKENDS[2])
        except BackendRejected as exc:
            require(str(exc)=="BACKEND_IDENTITY_MISMATCH_OR_SILENT_FALLBACK",
                    "Baseline rejection reason differs from exact Cython fallback")
        else:
            raise BackendRejected("Baseline dangerous Cython fallback was admitted")
        return {"mode":mode,"original_case":case,
                "blocked_by":"BACKEND_IDENTITY_MISMATCH_OR_SILENT_FALLBACK",
                "case_dispatch_invocations":0,"PCC_fits_executed":0}

    counts=Counter()
    for item in c.case_order(protocol):
        selected=assert_case_backend_exact(
            protocol,item["uci_id"],item["replicate"],item["backend"])
        require(callable(selected),"Original production model resolver failed")
        counts[item["backend"]]+=1
    require(dict(counts)=={name:24 for name in c.BACKENDS},
            "Original P1 backend count or 72-case roster drift")
    return {"mode":mode,"cases_frozen":72,"backend_cases":dict(counts),
            "fallback_accepted":False,"original_Cython_native_sha256":
                k.lock.NEW_NATIVE["production_pcc_step"],
            "original_Cython_observer_sha256":
                k.lock.NEW_NATIVE["isolated_pcc_step_live_research"],
            "model_graphs_built":0,"Numba_JIT_compilations":0,
            "PCC_propagation_function_calls":0,"PCC_fits_executed":0}

def validate_worker(row,mode):
    require(type(row) is dict and
            mode in ("baseline_must_reject","corrected_72_case") and
            row.get("mode")==mode and row.get("PCC_fits_executed")==0,
            "Invalid original no-fit backend guard worker output")
    if mode=="baseline_must_reject":
        require(row.get("original_case")=={
            "uci_id":212,"replicate":0,"backend":c.BACKENDS[2],
            "status":"NOT_RUN_NO_APPROVAL"} and
            row.get("blocked_by")=="BACKEND_IDENTITY_MISMATCH_OR_SILENT_FALLBACK" and
            row.get("case_dispatch_invocations")==0,
            "Expected original Cython fallback did not fail closed")
    else:
        require(row.get("cases_frozen")==72 and
                row.get("backend_cases")=={name:24 for name in c.BACKENDS} and
                row.get("fallback_accepted") is False and
                row.get("original_Cython_native_sha256")==
                    k.lock.NEW_NATIVE["production_pcc_step"] and
                row.get("original_Cython_observer_sha256")==
                    k.lock.NEW_NATIVE["isolated_pcc_step_live_research"] and
                row.get("model_graphs_built")==0 and
                row.get("Numba_JIT_compilations")==0 and
                row.get("PCC_propagation_function_calls")==0,
                "72 original backend identities not proven with no code execution")
    return True

def run_one(mode):
    name="pypcc-p201l-"+uuid.uuid4().hex[:12]
    try:
        outcome=subprocess.run(container_args(mode,name),text=True,
                               stdout=subprocess.PIPE,stderr=subprocess.PIPE,
                               timeout=105,check=False)
        require(outcome.returncode==0 and
                len(outcome.stdout.encode())<=6000 and
                len(outcome.stderr.encode())<=6000,
                "One exact OCI import-only worker failed closed")
        try:
            row=json.loads(outcome.stdout)
        except ValueError as exc:
            raise BackendRejected("Exact OCI import-only worker returned invalid JSON") from exc
        validate_worker(row,mode)
        return row
    finally:
        cleaned=subprocess.run(["docker","rm","-f",name],
                               stdout=subprocess.PIPE,stderr=subprocess.PIPE,
                               timeout=15,check=False)
        require(cleaned.returncode in (0,1),
                "Exact OCI import-only worker cleanup failed")

def assemble(rows):
    require(type(rows) is list and len(rows)==2 and
            validate_worker(rows[0],"baseline_must_reject") and
            validate_worker(rows[1],"corrected_72_case"),
            "Both expected negative and positive 72-case admission records required")
    report={"schema":SCHEMA,"status":STATUS,
            "P1_scientific_sha256":P1_SHA,
            "exact_existing_OCI_export_SHA256":IMAGE_SHA,
            "exact_existing_OCI_config_id":IMAGE_ID,
            "guard_is_reusable_future_worker_prerequisite_not_execution_authorization":True,
            "original_Cython_fallback_is_fatal":True,
            "frozen_cases_visited_without_fits":72,
            "original_OCI_import_only_backend_guard_evidence":rows,
            "real_PCC_propagations_in_current_step":0,
            "original_50k_full_fit_resource_limits_proven":False,
            **NO_GO}
    validate_report(report)
    return report

def validate_report(r):
    require(type(r) is dict and r.get("schema")==SCHEMA and
            r.get("status")==STATUS and
            r.get("P1_scientific_sha256")==P1_SHA and
            r.get("exact_existing_OCI_export_SHA256")==IMAGE_SHA and
            r.get("exact_existing_OCI_config_id")==IMAGE_ID and
            r.get("guard_is_reusable_future_worker_prerequisite_not_execution_authorization") is True and
            r.get("original_Cython_fallback_is_fatal") is True and
            r.get("frozen_cases_visited_without_fits")==72 and
            r.get("real_PCC_propagations_in_current_step")==0 and
            r.get("original_50k_full_fit_resource_limits_proven") is False and
            all(r.get(key)==value for key,value in NO_GO.items()),
            "Original scientific STOP or no-fallback preflight status falsified")
    rows=r.get("original_OCI_import_only_backend_guard_evidence")
    require(type(rows) is list and len(rows)==2 and
            validate_worker(rows[0],"baseline_must_reject") and
            validate_worker(rows[1],"corrected_72_case"),
            "Original OCI source-free backend admission records incomplete")
    return True

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker-mode",choices=("baseline_must_reject","corrected_72_case"))
    parser.add_argument("--original-OCI-archive",type=Path)
    parser.add_argument("--output",type=Path)
    args=parser.parse_args()
    if args.worker_mode:
        require(args.original_OCI_archive is None and args.output is None,
                "Worker does not accept any scientific data files")
        result=worker(args.worker_mode)
        validate_worker(result,args.worker_mode)
        print(canonical(result).decode("ascii"))
        return
    require(args.original_OCI_archive is not None and args.output is not None
            and not args.output.exists() and platform.system()=="Linux",
            "Only Linux host with original OCI archive and fresh safe output path")
    source_bytes_locked()
    from experiments import p2_07e_locked_oci_cgroup_resource_supervisor as supervisor
    loaded=supervisor.verify_original_image(args.original_OCI_archive.resolve())
    require(loaded["loaded_exact_original_image_ID"]==IMAGE_ID,
            "Original historical image config ID wrong")
    report=assemble([run_one("baseline_must_reject"),
                     run_one("corrected_72_case")])
    with args.output.open("xb") as file:
        file.write(json.dumps(report,sort_keys=True,indent=2).encode("utf-8")+b"\n")
        file.flush()
        os.fsync(file.fileno())
    print(json.dumps({"status":STATUS,"review_only_sha256_not_P2_execution":sha(report),
                      "frozen_72_cases_nonexecuting_backend_checks":72,
                      **NO_GO},sort_keys=True))

if __name__=="__main__":
    try:
        main()
    except BaseException as err:
        print("P201L_SAFE_ERROR|"+type(err).__name__,
              file=sys.stderr)
        sys.exit(1)
