"""P2-07M: three ORIGINAL graph preparations in EXACT pre-existing OCI, NO PCC fits.

Source acquisition/preprocessing happens transiently in an ephemeral CI host using
approved P1 source and transform hashes. The verified float64 Z matrices are
read-only mounted into the exact P2-01I OCI image; offline/unprivileged worker
performs only original build_graph, verifies exact full dense graph SHA, and
returns aggregate CPU/RSS. Input arrays are never uploaded or retained.

The OCI is a pinned *candidate binary substrate*, NOT a reproducible native
toolchain or authorized scientific worker. No masks, RNG, particles or PCC
fit_predict() are invoked. Memory/time are GRAPH-PREPARATION observations.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import time
import uuid

from experiments import p2_01i_verify_exact_exported_runtime_image as locked

SCHEMA="pypcc-P2-07M-exact-existing-OCI-real-source-graph-prep-no-fit/v1"
STATUS="THREE_ORIGINAL_GRAPH_ONLY_EXACT_OCI_HASHES_MATCH_ZERO_FITS"
P1_SHA=locked.P1_SHA
IMAGE_ID=locked.EXPECTED_IMAGE_ID
IMAGE_SHA=locked.EXPECTED_IMAGE_SHA
IDS=(212,850,292)
DIMENSIONS={212:(310,6,22),850:(900,7,20),292:(440,6,24)}
GRAPH_HASHES={
    212:"434ec3bb811222964304ae163afa1bf3eb25f767139107b5bb0f05dc67db27f2",
    850:"b0e260ae198037e89e5e8b0691da15b4c7bfc4cf17e41adb1fe8209a7a975384",
    292:"63e502a5d3f981a20c8c1a80884aa740a25fa7ea7f13335ad7554ad6e7824ca6",
}
X_HASHES={
    212:"420ff646d5edd234f907a27b71b00f5d29e4732959b58f34f32315740247cf21",
    850:"6b6d9a0520fee1a7625467a1ce6618f738d4eb815a4c4a786603329196294cf2",
    292:"9a4b2e4a49e2dd4d96f9cb99381d8849e9cce7571124761505938fc24c8acbe7",
}
PRODUCTION_BLOBS={"pcc.py":"01606aaee860c3b9bc84710e7362a2ef81046d45",
                  "pcc_graph.py":"3e8943462555b4feafbd162125b51e41f29d7ab7",
                  "pcc_numpy.py":"edf09a4ac5487327d8bd2b63415b5dd6564209f5",
                  "pcc_numba.py":"86d21d57ef535bbdb8566f10ace38cb54f93f741",
                  "pcc_validation.py":"b98510431c2281463988321fb7f1b6b159a5f77d"}
CANDIDATE_CAP=2*1024**3
ROOT=Path(__file__).resolve().parents[1]
PYTHON_IN_OCI="/opt/hostedtoolcache/Python/3.12.16/x64/bin/python3.12"
IMAGE_TAG="pypcc-p2-01h-candidate:immutable"


class RuntimeGraphBlocked(ValueError):
    pass


def require(ok, msg):
    if not ok:
        raise RuntimeGraphBlocked(msg)


def canonical(x):
    return json.dumps(x,sort_keys=True,ensure_ascii=True,
                      allow_nan=False,separators=(",",":")).encode("ascii")


def sha_bytes(data):
    return hashlib.sha256(data).hexdigest()


def sha_file(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as fd:
        for chunk in iter(lambda:fd.read(1024**2),b""):
            h.update(chunk)
    return h.hexdigest()


def git_blob(path):
    raw=Path(path).read_bytes()
    # GitHub Windows checkout may expand canonical LF blobs to CRLF.
    # Normalize only that well-defined text checkout conversion, not
    # source tokens/whitespace or the expected immutable Git object hash.
    normalized=raw.replace(b"\r\n",b"\n")
    return hashlib.sha1(b"blob "+str(len(normalized)).encode("ascii")+
                        b"\0"+normalized).hexdigest()


def verify_source_blobs():
    for name,expected in PRODUCTION_BLOBS.items():
        require(git_blob(ROOT/name)==expected, "Original production Git blob drifted: "+name)
    return True


def prepare_input(uci_id, folder):
    """Host-only source preparation: no graph, mask, seed or fit."""
    require(uci_id in IDS and folder.is_dir(), "Only frozen P1 graph input")
    import numpy as np
    from experiments import p2_02a_official_uci_byte_integrity as official
    from experiments import p2_02b_03_ordered_features_no_fit as prepared
    from experiments import p2_04_05_graph_mask_no_fit as graph
    from experiments import p2_07c_consolidated_execution_envelope_no_fit as policy

    p1=policy.read_approved_P1()
    envelope=policy.load(policy.POLICY)
    policy.validate_policy(envelope,p1)
    require(envelope["approved_execution_host_and_resource_limits"] is False,
            "Candidate resource approval must be false")
    for name,version in graph.EXPECTED_VERSIONS.items():
        import importlib.metadata
        require(importlib.metadata.version(name)==version,
                "Graph input host did not use frozen P2 numerical dependencies")
    prepared.parser_sources_checked()
    p1_ref=graph.original_reference_source_verified()
    require(p1_ref==graph.EXPECTED_P1_REFERENCE_BLOB,
            "Approved reference source drift")
    spec=next(x for x in p1["selected_datasets"] if x["uci_id"]==uci_id)
    archive,_=official.download_official(spec)
    require(sha_bytes(archive)==spec["zip_sha256"], "Official UCI ZIP SHA drift")
    member=prepared.load_member_bytes(archive,spec)
    columns,rows,_,indices,_=prepared.parse_ordered(spec,member)
    names=(list(prepared.WHOLESALE_X) if uci_id==292
           else [columns[i] for i in indices])
    X,labels,_=prepared.numeric_rows_and_labels(spec,columns,rows,indices)
    require(len(labels)==spec["original_n_rows"], "Approved row count drift")
    raw=prepared.array_fingerprint(X,names)
    require(raw["shape_dtype_feature_order_and_numeric_bytes_sha256"]==
            graph.EXPECTED_RAW_SHA[uci_id], "Raw P1 feature matrix SHA drift")
    Z,meta=prepared.approved_transform(X,wholesale=(uci_id==292))
    fingerprint=prepared.array_fingerprint(Z,names)
    observed_z_sha=fingerprint["shape_dtype_feature_order_and_numeric_bytes_sha256"]
    require(observed_z_sha==X_HASHES[uci_id],
            "P1 transformed matrix exact SHA drift on host uci="+str(uci_id)+
            " observed_sha256="+observed_z_sha)
    require(meta["labels_used_to_select_or_scale_features"] is False,
            "Original transformation must never consume labels")
    require(Z.flags.c_contiguous,
            "Original transformed float64 must remain C-contiguous")

    numpy_path=folder/("Z-%d.npy"%uci_id)
    metadata_path=folder/("Z-%d.json"%uci_id)
    require(not numpy_path.exists() and not metadata_path.exists(),
            "Original matrix file collision")
    with numpy_path.open("xb") as file:
        np.save(file,Z,allow_pickle=False)
    os.chmod(numpy_path,0o444)
    dimensions=DIMENSIONS[uci_id]
    require(Z.shape==(dimensions[0],dimensions[1]),
            "Unexpected original dimension")
    receipt={
        "schema":"pypcc-P2-07M-transient-original-Z-only/v1",
        "uci_id":uci_id,
        "P1_scientific_sha256":P1_SHA,
        "X_transformed_exact_SHA256":X_HASHES[uci_id],
        "source_zip_sha256":spec["zip_sha256"],
        "npy_file_sha256":sha_file(numpy_path),
        "shape":[dimensions[0],dimensions[1]],
        "ordered_predictor_names":names,
        "P2_execution_authorized":False,
        "no_masks_seeds_or_PCC_fits":True,
    }
    with metadata_path.open("xb") as file:
        file.write(canonical(receipt))
    os.chmod(metadata_path,0o444)
    return receipt


def vm_kib(key):
    data={}
    for line in Path("/proc/self/status").read_text(encoding="ascii").splitlines():
        if ":" in line:
            a,b=line.split(":",1)
            data[a]=b.strip()
    val=data.get(key,"")
    require(val.endswith(" kB") and val[:-3].strip().isdigit(),
            "Missing Linux kernel VmRSS/VmHWM counter")
    return int(val[:-3].strip())


def validate_worker_result(row):
    require(type(row) is dict and
            row.get("schema")=="pypcc-P2-07M-exact-OCI-worker-graph-only/v1" and
            row.get("uci_id") in IDS and
            row.get("P1_scientific_sha256")==P1_SHA and
            row.get("loaded_exact_existing_OCI_config_id")==IMAGE_ID and
            row.get("original_graph_sha256")==GRAPH_HASHES[row["uci_id"]] and
            row.get("original_transformed_X_sha256")==X_HASHES[row["uci_id"]] and
            row.get("original_graph_dense_minus_one_padding") is True and
            row.get("original_graph_undirected") is True and
            row.get("original_graph_width")==DIMENSIONS[row["uci_id"]][2] and
            row.get("original_graph_array_bytes")==
              8*DIMENSIONS[row["uci_id"]][0]*(DIMENSIONS[row["uci_id"]][2]+1) and
            row.get("real_original_PCC_fits_executed")==0 and
            row.get("synthetic_PCC_fits_executed")==0 and
            row.get("PCC_propagations_executed")==0 and
            row.get("original_masks_or_PCC_seeds_generated")==0 and
            row.get("real_UCI_source_rows_or_graph_edges_exported") is False and
            row.get("P2_execution_authorized") is False,
            "Runtime original graph and no-execution evidence invalid")
    for item in ("graph_build_wall_seconds","graph_build_cpu_seconds"):
        require(type(row.get(item)) in (float,int) and
                0 <= row[item] < 120, "Invalid isolated OCI graph CPU/wall")
    require(type(row.get("process_peak_resident_bytes")) is int and
            0<row["process_peak_resident_bytes"]<2**40 and
            type(row.get("process_RSS_after_graph_bytes")) is int and
            0<row["process_RSS_after_graph_bytes"] <=row["process_peak_resident_bytes"],
            "Invalid actual OCI resident highwater")
    return True


def worker(uci_id, folder):
    require(platform.system()=="Linux" and
            platform.python_version()=="3.12.16" and
            uci_id in IDS and
            str(folder)=="/inputs",
            "Original OCI worker allowed only Linux/Python3.12.16 and frozen inputs")
    import numpy as np
    from experiments import p2_04_05_graph_mask_no_fit as graph
    from experiments import p2_02b_03_ordered_features_no_fit as prepared
    from experiments import p2_01h_import_only_oci_selfcheck as selfcheck
    # Image itself includes manifest and exact native versions; this uses
    # imports only and no source graph or PCC propagation in selfcheck.
    # Full runtime selfcheck is not called again (contains mapped ELF paths).
    manifest=json.loads(selfcheck.MANIFEST.read_text(encoding="utf-8"))
    require(manifest["P1_scientific_sha256"]==P1_SHA and
            manifest["P2_execution_authorized"] is False,
            "Original OCI runtime manifest wrong/forged")
    import importlib.metadata
    for name,version in graph.EXPECTED_VERSIONS.items():
        if name=="openpyxl":
            continue # Host's approved parser used openpyxl; none used in OCI.
        require(importlib.metadata.version(name)==version,
                "OCI numerical graph dependency mismatch")
    metadata=json.loads((folder/("Z-%d.json"%uci_id)).read_text(encoding="ascii"))
    require(metadata.get("schema")=="pypcc-P2-07M-transient-original-Z-only/v1" and
            metadata.get("uci_id")==uci_id and
            metadata.get("P1_scientific_sha256")==P1_SHA and
            metadata.get("X_transformed_exact_SHA256")==X_HASHES[uci_id] and
            metadata.get("P2_execution_authorized") is False and
            metadata.get("no_masks_seeds_or_PCC_fits") is True,
            "Original transformed input envelope invalid")
    file=folder/("Z-%d.npy"%uci_id)
    require(file.is_file() and sha_file(file)==metadata["npy_file_sha256"],
            "Original temporary transformed X bytes drifted")
    Z=np.load(file,allow_pickle=False)
    dimensions=DIMENSIONS[uci_id]
    require(type(Z) is np.ndarray and Z.shape==(dimensions[0],dimensions[1]) and
            Z.dtype==np.dtype("float64") and Z.flags.c_contiguous and
            np.isfinite(Z).all(), "Original transformed matrix dtype/shape invalid")
    fp=prepared.array_fingerprint(Z,metadata["ordered_predictor_names"])
    require(fp["shape_dtype_feature_order_and_numeric_bytes_sha256"]==X_HASHES[uci_id],
            "Original ordered transformed-X bytes differ inside exact OCI")
    graph.verify_build_graph_route()
    clock=time.monotonic()
    cpu=time.process_time()
    graph_result=graph.graph_through_model(Z)
    elapsed=round(time.monotonic()-clock,6)
    used_cpu=round(time.process_time()-cpu,6)
    require(graph_result["exact_full_dense_graph_SHA256"]==GRAPH_HASHES[uci_id] and
            graph_result["dense_unused_padding_all_minus_one"] and
            graph_result["edges_without_reciprocals"]==0 and
            graph_result["maximum_degree"]==dimensions[2],
            "ORIGINAL unchanged full graph SHA, symmetry, width or padding mismatch")
    rss=vm_kib("VmRSS")*1024
    peak=max(rss,vm_kib("VmHWM")*1024)
    report={
        "schema":"pypcc-P2-07M-exact-OCI-worker-graph-only/v1",
        "uci_id":uci_id,
        "P1_scientific_sha256":P1_SHA,
        "loaded_exact_existing_OCI_config_id":IMAGE_ID,
        "original_transformed_X_sha256":X_HASHES[uci_id],
        "original_graph_sha256":GRAPH_HASHES[uci_id],
        "original_graph_width":graph_result["maximum_degree"],
        "original_graph_array_bytes":graph_result["graph_storage_bytes"],
        "original_graph_dense_minus_one_padding":True,
        "original_graph_undirected":True,
        "graph_build_wall_seconds":elapsed,
        "graph_build_cpu_seconds":used_cpu,
        "process_peak_resident_bytes":peak,
        "process_RSS_after_graph_bytes":rss,
        "real_original_PCC_fits_executed":0,
        "synthetic_PCC_fits_executed":0,
        "PCC_propagations_executed":0,
        "original_masks_or_PCC_seeds_generated":0,
        "real_UCI_source_rows_or_graph_edges_exported":False,
        "P2_execution_authorized":False,
    }
    validate_worker_result(report)
    print(canonical(report).decode("ascii"))


def validated_docker_command(uci_id, name, inputs):
    require(uci_id in IDS and
            re.fullmatch(r"pypcc-p207m-[0-9a-f]{12}",name) and
            inputs.is_absolute() and inputs.is_dir(),
            "Unsafe non-frozen OCI graph-only command")
    return [
        "docker","run","--rm","--name",name,
        "--platform","linux/amd64","--network","none","--read-only",
        "--user","65534:65534","--cap-drop","ALL",
        "--security-opt","no-new-privileges","--pids-limit","128",
        "--memory","2g","--memory-swap","2g","--cpus","1",
        "--tmpfs","/tmp:rw,nosuid,nodev,noexec,size=32m",
        "--env","PYTHONDONTWRITEBYTECODE=1","--env","PYTHONNOUSERSITE=1",
        "--env","PYTHONPATH=/workspace",
        # Numba's import-time cache locator must not write in /workspace,
        # which is intentionally mounted read-only. No PCC JIT is run.
        "--env","NUMBA_CACHE_DIR=/tmp/numba-cache",
        "--env","OPENBLAS_NUM_THREADS=1","--env","OMP_NUM_THREADS=1",
        "--env","MKL_NUM_THREADS=1","--env","NUMEXPR_NUM_THREADS=1",
        "--entrypoint",PYTHON_IN_OCI,
        "--mount",f"type=bind,src={ROOT.resolve()},dst=/workspace,readonly",
        "--mount",f"type=bind,src={inputs},dst=/inputs,readonly",
        IMAGE_TAG,"-m",
        "experiments.p2_07m_exact_oci_original_graph_preparation_no_fit",
        "--worker","--uci-id",str(uci_id),"--inputs","/inputs"
    ]


def run_original_graph(uci_id, inputs, *, run=subprocess.run):
    name="pypcc-p207m-"+uuid.uuid4().hex[:12]
    command=validated_docker_command(uci_id,name,inputs)
    try:
        result=run(command,stdout=subprocess.PIPE,stderr=subprocess.PIPE,
                   text=True,timeout=120,check=False)
        if result.returncode != 0:
            # Worker emits an authored source-safe P207M failure marker.
            # Never echo arbitrary raw stderr (it might contain source data).
            safe=[line[:220] for line in result.stderr.splitlines()
                  if line.startswith("P207M_FAILED_SOURCE_SAFE|") and
                  len(line)<240]
            if safe:
                print(safe[-1],file=sys.stderr)
            # Limited fixed categories identify Docker/Python loader failures
            # that occur before the worker's own error handler is installed.
            classes=("OCI runtime exec failed","Error while finding module specification",
                     "No module named","permission denied","Permission denied",
                     "Unable to find image","exec format error")
            safe_class=next((label for label in classes
                             if label in result.stderr), "NO_SAFE_WORKER_DIAGNOSTIC")
            raise RuntimeGraphBlocked(
                "Original OCI source-free worker exit nonzero; category="+safe_class)
        require(len(result.stdout.encode("utf-8"))<8192 and
                len(result.stderr.encode("utf-8"))<8192,
                "Original OCI graph-only worker emitted unbounded output")
        try:
            row=json.loads(result.stdout)
        except ValueError as exc:
            raise RuntimeGraphBlocked("Original OCI graph worker emitted invalid JSON") from exc
        validate_worker_result(row)
        require(row["uci_id"]==uci_id,"OCI graph order or source swapped")
        return row
    finally:
        # Forced remove even after timeout/exit; no detached worker survives.
        cleanup=subprocess.run(["docker","rm","-f",name],
                               stdout=subprocess.PIPE,stderr=subprocess.PIPE,
                               timeout=10,check=False)
        require(cleanup.returncode in (0,1),
                "Unable to clean up exact OCI graph-only container")


def validate_report(report):
    require(type(report) is dict and
            report.get("schema")==SCHEMA and
            report.get("status")==STATUS and
            report.get("P1_scientific_sha256")==P1_SHA and
            report.get("verified_original_OCI_archive_SHA256")==IMAGE_SHA and
            report.get("verified_original_OCI_config_id")==IMAGE_ID and
            report.get("original_P2_07L_source_only_report_SHA256")==
              "1511cccf2f177b17e3f9147dbb379767b44a2dce0f524814dcfa540c94f2dbba",
            "Frozen source, original OCI or prior graph-only evidence drift")
    rows=report.get("exact_OCI_three_original_graph_only_resources")
    require(type(rows) is list and len(rows)==3 and
            [x.get("uci_id") for x in rows]==list(IDS) and
            all(validate_worker_result(x) for x in rows) and
            report.get("graph_only_all_original_hashes_match") is True and
            report.get("observed_peak_below_unapproved_candidate_2GiB_for_graph_only") is
              all(x["process_peak_resident_bytes"]<CANDIDATE_CAP for x in rows),
            "Original OCI graph worker evidence incomplete or falsely budgeted")
    for key,value in (
        ("original_UCI_PCC_fits_executed",0),
        ("original_masks_or_scientific_PCC_seeds_created",0),
        ("scientific_PCC_kernel_or_particle_propagations_executed",0),
        ("source_dataset_matrices_or_graph_neighbors_persisted_in_artifacts",False),
        ("complete_host_compiler_SDK_system_byte_closure_proven",False),
        ("actual_full_original_50k_fit_CPU_wall_RSS_disk_ceiling_proven",False),
        ("P2_01_final_execution_toolchain_lock_pass",False),
        ("P2_07_full_resources_gate_pass",False),
        ("P2_execution_preflight_sha256",None),
        ("P2_execution_authorized",False),
        ("real_PCC_fits_executed",0),
    ):
        require(report.get(key)==value,
                "False original scientific result/resource or approval status: "+key)
    return True


def assemble(rows):
    require(type(rows) is list and len(rows)==3 and
            [x.get("uci_id") for x in rows]==list(IDS) and
            all(validate_worker_result(x) for x in rows),
            "Original three source graphs not verified in existing OCI")
    report={
        "schema":SCHEMA,
        "status":STATUS,
        "P1_scientific_sha256":P1_SHA,
        "verified_original_OCI_archive_SHA256":IMAGE_SHA,
        "verified_original_OCI_config_id":IMAGE_ID,
        "original_P2_07L_source_only_report_SHA256":
            "1511cccf2f177b17e3f9147dbb379767b44a2dce0f524814dcfa540c94f2dbba",
        "exact_OCI_three_original_graph_only_resources":rows,
        "graph_only_all_original_hashes_match":True,
        "observed_peak_below_unapproved_candidate_2GiB_for_graph_only":
            all(x["process_peak_resident_bytes"]<CANDIDATE_CAP for x in rows),
        "original_UCI_PCC_fits_executed":0,
        "original_masks_or_scientific_PCC_seeds_created":0,
        "scientific_PCC_kernel_or_particle_propagations_executed":0,
        "source_dataset_matrices_or_graph_neighbors_persisted_in_artifacts":False,
        "complete_host_compiler_SDK_system_byte_closure_proven":False,
        "actual_full_original_50k_fit_CPU_wall_RSS_disk_ceiling_proven":False,
        "P2_01_final_execution_toolchain_lock_pass":False,
        "P2_07_full_resources_gate_pass":False,
        "P2_execution_preflight_sha256":None,
        "P2_execution_authorized":False,
        "real_PCC_fits_executed":0,
    }
    validate_report(report)
    return report


def verify_staged_inputs(folder):
    """No network: check three exact pre-verified temporary .npy/JSON pairs."""
    require(folder == (ROOT/"p2-07m-transient-original-Z-no-artifact").resolve()
            and folder.is_dir() and not folder.is_symlink(),
            "Only the one pre-verified ephemeral source directory is allowed")
    expected={("Z-%d.%s"%(uci,ext)) for uci in IDS
              for ext in ("npy","json")}
    require({x.name for x in folder.iterdir()}==expected and
            all(x.is_file() and not x.is_symlink() for x in folder.iterdir()),
            "Original temporary input files missing or unexpected")
    import numpy as np
    from experiments import p2_02b_03_ordered_features_no_fit as prepared
    from experiments import p2_07c_consolidated_execution_envelope_no_fit as c
    p1=c.read_approved_P1()
    for uci in IDS:
        doc=json.loads((folder/("Z-%d.json"%uci)).read_text(encoding="ascii"))
        src=next(x for x in p1["selected_datasets"] if x["uci_id"]==uci)
        require(doc.get("schema")=="pypcc-P2-07M-transient-original-Z-only/v1" and
                doc.get("uci_id")==uci and
                doc.get("P1_scientific_sha256")==P1_SHA and
                doc.get("source_zip_sha256")==src["zip_sha256"] and
                doc.get("X_transformed_exact_SHA256")==X_HASHES[uci] and
                doc.get("shape")==list(DIMENSIONS[uci][:2]) and
                doc.get("P2_execution_authorized") is False and
                doc.get("no_masks_seeds_or_PCC_fits") is True,
                "Original temporary Z/P1 metadata differs")
        source=folder/("Z-%d.npy"%uci)
        require(sha_file(source)==doc.get("npy_file_sha256"),
                "Preverified temporary original matrix SHA drift")
        Z=np.load(source,allow_pickle=False,mmap_mode="r")
        require(Z.shape==DIMENSIONS[uci][:2] and
                Z.dtype==np.dtype("float64") and
                np.isfinite(Z).all() and
                prepared.array_fingerprint(
                    Z,doc["ordered_predictor_names"])[
                    "shape_dtype_feature_order_and_numeric_bytes_sha256"]==
                    X_HASHES[uci],
                "One of three preverified transformed matrices changed")
    return True


def host_main(oci_artifact, output, inputs):
    require(platform.system()=="Linux" and
            platform.machine().lower() in ("x86_64","amd64") and
            sys.version_info[:2]==(3,12) and
            not output.exists(),
            "Only distinct Linux Py3.12 original OCI review without overwrite")
    verify_source_blobs()
    # Reuse precisely the three matrices verified once in the earlier CI
    # step; do not repeat official HTTPS requests, which can fail/rate-limit.
    verify_staged_inputs(inputs)
    rows=[]
    try:
        from experiments import p2_07e_locked_oci_cgroup_resource_supervisor as previous
        host=previous.verify_original_image(oci_artifact)
        require(host["loaded_exact_original_image_ID"]==IMAGE_ID,
                "Wrong previous exact OCI binary loaded")
        for original_id in IDS:
            rows.append(run_original_graph(original_id,inputs))
    finally:
        # Always delete the ephemeral real source arrays even if OCI fails.
        shutil.rmtree(inputs)
    require(not inputs.exists(),"Original transformed source arrays were not destroyed")
    report=assemble(rows)
    with output.open("xb") as f:
        f.write(json.dumps(report,indent=2,sort_keys=True).encode("utf-8")+b"\n")
        f.flush()
        os.fsync(f.fileno())
    require(validate_report(json.loads(output.read_text(encoding="utf-8"))),
            "Persisted no-fit exact OCI original graph-only report invalid")
    print(json.dumps({
        "status":STATUS,
        "measured_OCI_full_process_peak_RSS_bytes":
            {str(x["uci_id"]):x["process_peak_resident_bytes"] for x in rows},
        "measured_OCI_graph_build_CPU_seconds":
            {str(x["uci_id"]):x["graph_build_cpu_seconds"] for x in rows},
        "source_free_review_sha256_NOT_EXECUTION_AUTHORIZATION":sha_bytes(canonical(report)),
        "P2_execution_preflight_sha256":None,
        "P2_execution_authorized":False,
        "real_PCC_fits_executed":0,
    },sort_keys=True))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker",action="store_true")
    parser.add_argument("--uci-id",type=int)
    parser.add_argument("--inputs",type=Path)
    parser.add_argument("--existing-original-oci",type=Path)
    parser.add_argument("--preverified-ephemeral-inputs",type=Path)
    parser.add_argument("--output",type=Path)
    args=parser.parse_args()
    if args.worker:
        require(args.inputs is not None and
                args.existing_original_oci is None and
                args.preverified_ephemeral_inputs is None and args.output is None,
                "Worker must be isolated graph-only with preverified temporary X")
        worker(args.uci_id,args.inputs)
    else:
        require(args.uci_id is None and args.inputs is None and
                args.existing_original_oci is not None and
                args.preverified_ephemeral_inputs is not None and
                args.output is not None,
                "Host requires exact original OCI, already verified ephemeral inputs and review receipt")
        host_main(args.existing_original_oci.resolve(),args.output.resolve(),
                  args.preverified_ephemeral_inputs.resolve())


if __name__=="__main__":
    try:
        main()
    except BaseException as exc:
        # Never log original row values, file content or raw tracebacks.
        import traceback
        frames=traceback.extract_tb(exc.__traceback__)
        final=frames[-1] if frames else None
        filename=Path(final.filename).name if final else "unknown"
        line=final.lineno if final else 0
        # These controlled failure messages never carry source row values.
        if isinstance(exc,RuntimeGraphBlocked):
            message=str(exc)[:160]
        else:
            message=type(exc).__name__
        print("P207M_FAILED_SOURCE_SAFE|"+filename+"|"+str(line)+"|"+message,
              file=sys.stderr)
        sys.exit(1)
