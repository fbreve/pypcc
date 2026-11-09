"""Aggregate preregistered native inverse-mapping parity and CPU/memory gates."""
import json
import math
from pathlib import Path
import statistics
import sys

BASE="15695279884a82d0c71b4cb395187a04a1c8a698"
CASES={(150,3,False),(569,2,False),(1023,10,False),
       (1024,10,False),(1797,10,False),(30000,2,False),
       (30000,7,False),(30000,8,False),(30000,10,False),
       (30000,50,False),(100000,2,False),(100000,10,False),
       (100000,50,False),(100000,50,True)}
DATASETS={"Iris","Wine","Breast Cancer","Digits"}
BACKENDS={"numpy","numba","cython"}
REAL={(d,s,b,e) for d in DATASETS for s in (38027,38053)
      for b in BACKENDS for e in (False,True)}
TIMERS={"fit_predict_s","initialization_and_dispatch_s",
        "propagation_s","postprocessing_s"}


def is_sha(v):
    return isinstance(v,str) and len(v)==64 and all(x in "0123456789abcdef" for x in v)


def validate(doc):
    assert doc["schema"]==1 and doc["base_commit"]==BASE
    assert doc["os"] in ("linux","win32")
    assert is_sha(doc["original_pcc_sha256"])
    assert len(doc["micro"])==14 and len(doc["real"])==48
    micro={(r["nodes"],r["classes"],r["invalid"]):r for r in doc["micro"]}
    assert len(micro)==14 and set(micro)==CASES
    for (n,c,invalid),r in micro.items():
        assert type(invalid) is bool
        assert r["output_c_contiguous"] is True and r["output_writable"] is True
        assert all(is_sha(r[k]) for k in
                   ("internal_sha","source_labels_sha","decoded_sha"))
        assert len(r["timings_ms"])==25 and len(r["traced_peak_bytes"])==5
        for pair in r["timings_ms"]+r["traced_peak_bytes"]:
            assert len(pair)==2
            assert all(type(v) in (int,float) and math.isfinite(v) and v>0
                       for v in pair)
    native={(r["dataset"],r["label_seed"],r["backend"],r["early_stop"]):r
            for r in doc["real"]}
    assert len(native)==48 and set(native)==REAL
    for (_,_,b,_),r in native.items():
        assert is_sha(r["label_sha"])
        assert len(r["graph_sha"])==2 and all(is_sha(x) for x in r["graph_sha"])
        assert set(r["state_hashes"])=={
            "predictions","dominance","positions","strengths","distances","owndeg"}
        assert all(is_sha(v) for v in r["state_hashes"].values())
        assert r["backend_identity"]=={
            "numpy":"pcc_numpy.pcc_propagate_numpy",
            "numba":"pcc_numba.pcc_propagate_numba",
            "cython":"pcc_step.pcc_propagate"}[b]
        assert r["layout"]==("C" if b=="numba" and r["dataset"]=="Digits" else "F")
        if b=="numpy": assert is_sha(r["rng_sha"])
        else: assert r["rng_sha"] is None
        assert 0<r["info"]["effective_iterations"]<=1000
        assert r["info"]["stop_reason"] in ("max_iter","early_stop")
        for key in ("reference_timings","candidate_timings"):
            tm=r[key]
            assert set(tm)==TIMERS
            assert all(type(v) in (int,float) and math.isfinite(v) and v>0
                       for v in tm.values())


def main(root):
    paths=list(root.glob("**/inverse_label_result.json"))
    assert len(paths)==2,f"two complete independent host artifacts required: {paths}"
    docs=[json.loads(p.read_text(encoding="utf-8")) for p in paths]
    assert {doc["os"] for doc in docs}=={"linux","win32"}
    for doc in docs:validate(doc)
    assert len({doc["original_pcc_sha256"] for doc in docs})==1
    fixture_hash={}
    real_hash={}
    rows=[]
    go=True
    print("| OS | Nodes×classes | Invalid input | Inverse CPU reduction | Peak excess |")
    print("|---|---|---|---:|---:|")
    for doc in docs:
        for r in doc["micro"]:
            n,c,invalid=r["nodes"],r["classes"],r["invalid"]
            k=(n,c,invalid)
            h=(r["internal_sha"],r["source_labels_sha"],r["decoded_sha"])
            if k in fixture_hash:
                assert fixture_hash[k]==h,"cross-host micro source mismatch"
            fixture_hash[k]=h
            saving=100*(1-statistics.median([v/u for u,v in r["timings_ms"]]))
            excess=statistics.median([v-u for u,v in r["traced_peak_bytes"]])
            rows.append(dict(os=doc["os"],nodes=n,classes=c,invalid=invalid,
                             inverse_cpu_saving_pct=saving,
                             median_traced_excess_bytes=excess))
            print(f"| {doc['os']} | {n}×{c} | {invalid} | {saving:+.2f}% | {excess:+.0f} B |")
            if excess>24*n+524288:
                go=False
            if (n,c,invalid) in {
                (30000,10,False),(30000,50,False),(100000,50,False)
            }:
                if saving<25:go=False
            elif saving < -10:
                go=False
        for r in doc["real"]:
            key=(r["dataset"],r["label_seed"])
            h=(r["label_sha"],tuple(r["graph_sha"]))
            if key in real_hash:
                assert real_hash[key]==h,"cross-OS real data fingerprint"
            real_hash[key]=h
        for dataset in DATASETS:
            for backend in BACKENDS:
                for early in (False,True):
                    subset=[r for r in doc["real"]
                            if (r["dataset"],r["backend"],r["early_stop"])==
                            (dataset,backend,early)]
                    assert len(subset)==2
                    percent=100*(1-statistics.median([
                        r["candidate_timings"]["fit_predict_s"]/
                        r["reference_timings"]["fit_predict_s"] for r in subset]))
                    if percent < -10: go=False
    summary=dict(schema=1,completeness="PASS",decision="GO" if go else "NO-GO",
                 pairs=dict(inverse_cpu=700,tracemalloc_peak=140,full_native=96),
                 rows=sorted(rows,key=lambda x:(x["os"],x["nodes"],x["classes"],x["invalid"])))
    Path("inverse_label_summary.json").write_text(
        json.dumps(summary,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    print("Strict two-OS completeness PASS; preregistered candidate:",summary["decision"])
    # A prospectively valid NO-GO is a successful scientific job.


if __name__=="__main__":
    main(Path(sys.argv[1]))
