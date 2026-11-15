"""P2-07L — measure ORIGINAL graph preparation, no PCC fit or propagation.

Precisely the existing frozen P2-04 graph-construction path is used in three
separate short-lived Linux processes. Each downloads only the approved
cryptographically pinned UCI source, performs original feature transformation
and build_graph(), then hashes the dense padded undirected graph. No masks,
no seeds and no fit_predict/propagation. Outputs aggregate timings, process
high-water memory and graph hashes only; no original rows, labels or edges.

This is a GRAPH-PREPARATION measurement, NOT evidence that the proposed
2GiB / 1800s / 3600CPU-s ceilings cover original PCC 50k fits.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

from experiments import p2_07c_consolidated_execution_envelope_no_fit as c
from experiments import p2_07_metadata_only_resource_budget as prior

SCHEMA="pypcc-P2-07L-original-frozen-real-graph-preparation-resource-no-fit/v1"
STATUS="THREE_REAL_ORIGINAL_GRAPHS_PREPARED_HASH_VERIFIED_ZERO_PCC_FITS"
SOURCES=(212,850,292)
DIMENSIONS={212:(310,6,22),850:(900,7,20),292:(440,6,24)}
GRAPH_SHA=prior.GRAPH_DIGESTS
MAX_CAPTURE_BYTES=8*1024
CHILD_TIMEOUT_SECONDS=150
CANDIDATE_PREP_CEILING_BYTES=2*1024**3
P1_SHA=c.P1_SHA

class PrepBlocked(ValueError):
    pass

def require(ok,msg):
    if not ok: raise PrepBlocked(msg)

def canonical(x):
    return json.dumps(x,sort_keys=True,ensure_ascii=True,allow_nan=False,
                      separators=(",",":")).encode("ascii")

def digest(x):
    return hashlib.sha256(canonical(x)).hexdigest()

def _vm_kib(key):
    fields={}
    for line in Path("/proc/self/status").read_text(encoding="ascii").splitlines():
        if ":" in line:
            a,b=line.split(":",1)
            fields[a]=b.strip()
    value=fields.get(key,"")
    require(value.endswith(" kB") and value[:-3].strip().isdigit(),
            "Linux kernel process memory data missing: "+key)
    return int(value[:-3].strip())

def inspect_child(row):
    require(type(row) is dict and
            row.get("uci_id") in SOURCES and
            row.get("schema")=="pypcc-P2-07L-real-graph-prep-child/v1",
            "Unknown child source/format")
    n,f,width=DIMENSIONS[row["uci_id"]]
    require(row.get("original_source_rows")==n and
            row.get("original_feature_count")==f and
            row.get("actual_dense_neighbor_width")==width and
            row.get("exact_original_graph_SHA256")==GRAPH_SHA[row["uci_id"]] and
            row.get("model_graph_was_undirected") is True and
            row.get("model_padded_dense_minus_one") is True and
            row.get("graph_arrays_bytes")==n*(width+1)*8 and
            row.get("PCC_propagations_executed")==0 and
            row.get("real_PCC_fits_executed")==0 and
            row.get("real_masks_or_seeds_generated")==0 and
            row.get("rows_labels_graph_edges_or_masks_exported") is False and
            row.get("P2_execution_authorized") is False,
            "Graph/preparation/scientific contract diverged")
    for key in ("graph_preparation_wall_seconds",
                "graph_preparation_process_cpu_seconds",
                "total_source_download_parse_transform_graph_wall_seconds"):
        val=row.get(key)
        require(type(val) in (float,int) and 0<=val<CHILD_TIMEOUT_SECONDS,
                "Invalid process timing: "+key)
    require(type(row.get("original_process_vm_hwm_bytes")) is int and
            0<row["original_process_vm_hwm_bytes"]<2**40 and
            type(row.get("original_process_vm_rss_after_graph_bytes")) is int and
            0<row["original_process_vm_rss_after_graph_bytes"] <=
            row["original_process_vm_hwm_bytes"] and
            row.get("peak_rss_is_NOT_a_full_50k_PCC_fit_bound") is True,
            "Invalid high-water physical process memory evidence")
    return True

def graph_only_child(uci_id):
    require(platform.system()=="Linux" and sys.version_info[:2]==(3,12) and
            uci_id in SOURCES,
            "Graph-only process must run Linux candidate Py3.12 and frozen UCI ID")
    # Import the ORIGINAL already-vetted source and graph builder only in child.
    from experiments import p2_02a_official_uci_byte_integrity as official
    from experiments import p2_02b_03_ordered_features_no_fit as prepared
    from experiments import p2_04_05_graph_mask_no_fit as graph
    import importlib.metadata
    for name,version in graph.EXPECTED_VERSIONS.items():
        require(importlib.metadata.version(name)==version,
                "Unpinned graph-only preparation package: "+name)
    p1,policy,_,_ = __import__(
        "experiments.p2_07f_sealed_72case_no_fit_administrative_dryrun",
        fromlist=["frozen_inputs"]).frozen_inputs()
    require(p1["fit"]["max_iter"]==50000 and
            p1["fit"]["early_stop"] is False and
            policy["approved_execution_host_and_resource_limits"] is False,
            "Original P1 or resource approval changed")
    spec=next(x for x in p1["selected_datasets"] if x["uci_id"]==uci_id)
    graph.verify_build_graph_route()
    prepared.parser_sources_checked()
    started=time.monotonic()
    outer,_=official.download_official(spec)
    member=prepared.load_member_bytes(outer,spec)
    cols,rows,_,indices,_=prepared.parse_ordered(spec,member)
    names=(list(prepared.WHOLESALE_X) if uci_id==292
           else [cols[i] for i in indices])
    X,labels,_=prepared.numeric_rows_and_labels(spec,cols,rows,indices)
    # labels are parsed by original source code but never used or exported;
    # no masks, sampled original labeled indices or scientific seeds exist.
    require(len(labels)==spec["original_n_rows"],"Original source row count drift")
    raw_sha=prepared.array_fingerprint(X,names)[
        "shape_dtype_feature_order_and_numeric_bytes_sha256"]
    require(raw_sha==graph.EXPECTED_RAW_SHA[uci_id],
            "Original source X drifted before measuring graph resources")
    Z,transform=prepared.approved_transform(X,wholesale=(uci_id==292))
    transformed_sha=prepared.array_fingerprint(Z,names)[
        "shape_dtype_feature_order_and_numeric_bytes_sha256"]
    require(transformed_sha==graph.EXPECTED_TRANSFORM_SHA[uci_id] and
            transform["labels_used_to_select_or_scale_features"] is False,
            "Original approved preprocessing drifted")
    graph_start=time.monotonic()
    cpu_start=time.process_time()
    evidence=graph.graph_through_model(Z)
    graph_cpu=time.process_time()-cpu_start
    graph_wall=time.monotonic()-graph_start
    require(evidence["exact_full_dense_graph_SHA256"]==GRAPH_SHA[uci_id] and
            evidence["graph_storage_bytes"]==
            len(Z)*(evidence["maximum_degree"]+1)*8 and
            evidence["dense_unused_padding_all_minus_one"] and
            evidence["edges_without_reciprocals"]==0,
            "Original actual source graph/undirected dense padding differs")
    row={
        "schema":"pypcc-P2-07L-real-graph-prep-child/v1",
        "uci_id":uci_id,
        "original_source_rows":len(Z),
        "original_feature_count":Z.shape[1],
        "actual_dense_neighbor_width":evidence["maximum_degree"],
        "exact_original_graph_SHA256":evidence["exact_full_dense_graph_SHA256"],
        "graph_arrays_bytes":evidence["graph_storage_bytes"],
        "graph_preparation_wall_seconds":round(graph_wall,6),
        "graph_preparation_process_cpu_seconds":round(graph_cpu,6),
        "total_source_download_parse_transform_graph_wall_seconds":
            round(time.monotonic()-started,6),
        "original_process_vm_hwm_bytes":_vm_kib("VmHWM")*1024,
        "original_process_vm_rss_after_graph_bytes":_vm_kib("VmRSS")*1024,
        "model_graph_was_undirected":True,
        "model_padded_dense_minus_one":True,
        "peak_rss_is_NOT_a_full_50k_PCC_fit_bound":True,
        "real_masks_or_seeds_generated":0,
        "PCC_propagations_executed":0,
        "real_PCC_fits_executed":0,
        "rows_labels_graph_edges_or_masks_exported":False,
        "P2_execution_authorized":False,
    }
    inspect_child(row)
    print(canonical(row).decode("ascii"))

def run_child(uci_id, *, runner=subprocess.run):
    require(uci_id in SOURCES,"Unexpected original UCI source")
    env=dict(os.environ)
    env.update({"OPENBLAS_NUM_THREADS":"1","OMP_NUM_THREADS":"1",
                "MKL_NUM_THREADS":"1","NUMEXPR_NUM_THREADS":"1"})
    command=[sys.executable,"-m",
             "experiments.p2_07l_original_graph_preparation_resource_no_fit",
             "--child","--uci-id",str(uci_id)]
    result=runner(command,capture_output=True,text=True,timeout=CHILD_TIMEOUT_SECONDS,
                  env=env,check=False)
    if result.returncode != 0:
        # The child suppresses raw exception messages and dataset values.
        # Only an exception class, source filename, function and line are safe.
        diagnostic = [
            line for line in result.stderr.splitlines()
            if line.startswith("P207L_CHILD_FAILURE|") and len(line) < 250
        ]
        if diagnostic:
            # The child message is already sanitized (no exception text/data).
            print(diagnostic[-1],file=sys.stderr)
        raise PrepBlocked("Original graph-only subprocess failed: " +
                          (diagnostic[-1] if diagnostic else "NO_SAFE_DIAGNOSTIC"))
    require(len(result.stdout.encode("utf-8"))<MAX_CAPTURE_BYTES and
            len(result.stderr.encode("utf-8"))<MAX_CAPTURE_BYTES,
            "Original graph-only subprocess emitted unbounded logs")
    try:
        row=json.loads(result.stdout)
    except (ValueError,TypeError) as exc:
        raise PrepBlocked("Original graph prep emitted malformed JSON") from exc
    inspect_child(row)
    require(row["uci_id"]==uci_id,"Isolated graph source was replaced/reordered")
    return row

def final_report(rows):
    require(type(rows) is list and len(rows)==3 and
            [x.get("uci_id") for x in rows]==list(SOURCES) and
            all(inspect_child(x) for x in rows),
            "Missing/replaced/duplicate original graph preparation process")
    p1=c.read_approved_P1()
    policy=c.load(c.POLICY)
    c.validate_policy(policy,p1)
    require(c.sha256(c.canonical(policy))==
            "b2b7a969cf6349ec4c315cb2a91b72bf9a82868d4304a0575af246e9d85afc04",
            "Original P2-07C candidate cap not matched")
    report={
        "schema":SCHEMA,
        "status":STATUS,
        "P1_scientific_sha256":P1_SHA,
        "original_unapproved_P2_07C_policy_SHA256":
            "b2b7a969cf6349ec4c315cb2a91b72bf9a82868d4304a0575af246e9d85afc04",
        "original_graph_source_hashes":{str(x["uci_id"]):x["exact_original_graph_SHA256"] for x in rows},
        "observed_original_graph_only_processes":rows,
        "isolated_approved_original_graph_preparations":3,
        "original_72_scientific_PCC_fit_count":0,
        "actual_original_full_50k_PCC_fit_resource_upper_bound_proven":False,
        "observed_graph_only_peak_RSS_below_candidate_2GiB":
            all(x["original_process_vm_hwm_bytes"] < CANDIDATE_PREP_CEILING_BYTES for x in rows),
        "candidate_2GiB_limit_is_NOT_approved_or_full_fit_feasible":True,
        "selected_final_source_reproducible_host_approved":False,
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
            report.get("original_unapproved_P2_07C_policy_SHA256")==
            "b2b7a969cf6349ec4c315cb2a91b72bf9a82868d4304a0575af246e9d85afc04",
            "P1/P2-07C source drift")
    rows=report.get("observed_original_graph_only_processes")
    require(type(rows) is list and len(rows)==3 and
            [row.get("uci_id") for row in rows]==list(SOURCES) and
            all(inspect_child(row) for row in rows),
            "Unexpected original source graph preparation report")
    require(report.get("original_graph_source_hashes")==
            {str(x["uci_id"]):GRAPH_SHA[x["uci_id"]] for x in rows} and
            report.get("observed_graph_only_peak_RSS_below_candidate_2GiB") is
            all(x["original_process_vm_hwm_bytes"]<CANDIDATE_PREP_CEILING_BYTES
                for x in rows),
            "Graph original digest or observed resource claim invalid")
    for key,expected in (
        ("isolated_approved_original_graph_preparations",3),
        ("original_72_scientific_PCC_fit_count",0),
        ("actual_original_full_50k_PCC_fit_resource_upper_bound_proven",False),
        ("candidate_2GiB_limit_is_NOT_approved_or_full_fit_feasible",True),
        ("selected_final_source_reproducible_host_approved",False),
        ("P2_01_final_execution_toolchain_lock_pass",False),
        ("P2_07_full_resources_gate_pass",False),
        ("P2_execution_preflight_sha256",None),
        ("P2_execution_authorized",False),
        ("real_PCC_fits_executed",0),
    ):
        require(report.get(key)==expected,"False PCC scientific approval/resource claim: "+key)
    return True

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--child",action="store_true")
    parser.add_argument("--uci-id",type=int)
    parser.add_argument("--output",type=Path)
    args=parser.parse_args()
    if args.child:
        require(args.uci_id in SOURCES and args.output is None,
                "Child may only prepare one approved original graph")
        graph_only_child(args.uci_id)
        return
    require(platform.system()=="Linux" and
            args.uci_id is None and args.output is not None and
            not args.output.exists(),
            "Linux source-only audit cannot replace a previous report")
    rows=[run_child(i) for i in SOURCES]
    report=final_report(rows)
    with args.output.open("xb") as dest:
        dest.write(json.dumps(report,sort_keys=True,indent=2).encode("utf-8")+b"\n")
        dest.flush()
        os.fsync(dest.fileno())
    require(validate_report(json.loads(args.output.read_text(encoding="utf-8"))),
            "Persistent source-only graph resource report invalid")
    print(json.dumps({
        "status":STATUS,
        "real_graph_only_process_peak_RSS_bytes":
            {str(r["uci_id"]):r["original_process_vm_hwm_bytes"] for r in rows},
        "real_graph_only_CPU_seconds":
            {str(r["uci_id"]):r["graph_preparation_process_cpu_seconds"] for r in rows},
        "P2_execution_preflight_sha256":None,
        "P2_execution_authorized":False,
        "real_PCC_fits_executed":0,
    },sort_keys=True))

if __name__=="__main__":
    try:
        main()
    except BaseException as exc:
        # No error string or traceback is printed: either might contain
        # original source-record values. CI gets only sanitized location.
        import traceback
        frames=traceback.extract_tb(exc.__traceback__)
        frame=frames[-1] if frames else None
        filename=Path(frame.filename).name if frame else "unknown"
        function=frame.name if frame else "unknown"
        lineno=frame.lineno if frame else 0
        print("P207L_CHILD_FAILURE|"+type(exc).__name__+"|"+
              filename+"|"+function+"|"+str(lineno),file=sys.stderr)
        raise SystemExit(1)
