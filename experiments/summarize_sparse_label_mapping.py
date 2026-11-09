"""Prospectively fixed two-OS sparse label mapping exactness/CPU gates."""
import json
import math
from pathlib import Path
import statistics
import sys

BASE="a2fd27adf53155f378f12eeeb7cf67ec0625e195"
CASES={
    (150,3,"sparse"), (569,2,"sparse"), (1023,10,"sparse"),
    (1024,10,"sparse"), (1797,10,"sparse"), (30000,10,"sparse"),
    (30000,50,"sparse"), (100000,50,"sparse"), (100000,2,"sparse"),
    (30000,50,"half"), (30000,10,"half"), (100000,50,"ninety"),
}
DATASETS={"Iris","Wine","Breast Cancer","Digits"}
BACKENDS={"numpy","numba","cython"}
EXPECTED_NATIVE={(d,seed,b,e) for d in DATASETS for seed in (37027,37053)
                 for b in BACKENDS for e in (False,True)}
TIMERS=("fit_predict_s","initialization_and_dispatch_s",
        "propagation_s","postprocessing_s")


def sha(value):
    return (isinstance(value,str) and len(value)==64 and
            all(ch in "0123456789abcdef" for ch in value))


def require_report(doc):
    assert doc["schema"]==1 and doc["base_commit"]==BASE
    assert doc["os"] in ("linux","win32"),doc["os"]
    assert sha(doc["original_pcc_sha256"])
    assert len(doc["micro"])==12 and len(doc["real"])==48
    micro={(r["nodes"],r["classes"],r["density"]):r for r in doc["micro"]}
    assert len(micro)==12 and set(micro)==CASES
    for (n,c,d),r in micro.items():
        assert all(sha(r[k]) for k in ("labels_sha","uniques_sha","mapped_sha"))
        expected_homes=3*c if d=="sparse" else (n//2 if d=="half" else int(n*0.9))
        assert r["homes"]==expected_homes
        assert len(r["timings_ms"])==25 and len(r["traced_peak_bytes"])==5
        for pair in r["timings_ms"]+r["traced_peak_bytes"]:
            assert len(pair)==2
            assert all(isinstance(v,(float,int)) and math.isfinite(v) and v>0 for v in pair)
    native={(r["dataset"],r["label_seed"],r["backend"],r["early_stop"]):r
            for r in doc["real"]}
    assert len(native)==48 and set(native)==EXPECTED_NATIVE
    for (_,_,b,_),r in native.items():
        assert sha(r["label_sha"])
        assert len(r["graph_sha"])==2 and all(sha(v) for v in r["graph_sha"])
        assert set(r["state_hashes"])=={
            "predictions","dominance","positions","strengths","distances","owndeg"}
        assert all(sha(v) for v in r["state_hashes"].values())
        assert r["backend_identity"]=={
            "numpy":"pcc_numpy.pcc_propagate_numpy",
            "numba":"pcc_numba.pcc_propagate_numba",
            "cython":"pcc_step.pcc_propagate"}[b]
        assert r["layout"]==("C" if b=="numba" and r["dataset"]=="Digits" else "F")
        if b=="numpy": assert sha(r["rng_sha"])
        else: assert r["rng_sha"] is None
        assert 0 < r["info"]["effective_iterations"] <= 1000
        assert r["info"]["stop_reason"] in ("max_iter","early_stop")
        for key in ("reference_timings","candidate_timings"):
            timed=r[key]
            assert set(timed)==set(TIMERS)
            assert all(isinstance(v,(int,float)) and math.isfinite(v) and v>0
                       for v in timed.values())


def run(root):
    paths=list(root.glob("**/label_mapping_result.json"))
    assert len(paths)==2,f"two distinct OS reports required: {paths}"
    docs=[json.loads(p.read_text(encoding="utf-8")) for p in paths]
    assert {d["os"] for d in docs}=={"linux","win32"}
    for doc in docs: require_report(doc)
    assert len({d["original_pcc_sha256"] for d in docs})==1
    case_hashes={}
    native_hashes={}
    rows=[]
    go=True
    print("| OS | Nodes x classes | Labeled | Mapping CPU saving | Tracked peak delta |")
    print("|---|---|---|---:|---:|")
    for doc in docs:
        os=doc["os"]
        for r in doc["micro"]:
            n,c,d=r["nodes"],r["classes"],r["density"]
            k=(n,c,d)
            source=(r["labels_sha"],r["uniques_sha"],r["mapped_sha"])
            if k in case_hashes: assert case_hashes[k]==source,"cross-OS mapping fixture"
            case_hashes[k]=source
            cpu=100*(1-statistics.median([v/u for u,v in r["timings_ms"]]))
            memory=statistics.median([v-u for u,v in r["traced_peak_bytes"]])
            rows.append(dict(os=os,nodes=n,classes=c,density=d,cpu_saving_pct=cpu,
                             median_tracemalloc_extra_bytes=memory,
                             allocation_cap_bytes=24*n+524288))
            print(f"| {os} | {n} x {c} | {d} | {cpu:+.2f}% | {memory:+.0f} B |")
            if memory>24*n+524288: go=False
            if d=="sparse":
                if (n,c) in ((30000,10),(30000,50),(100000,50)):
                    if cpu<25: go=False
                elif (n,c) in ((150,3),(569,2),(1023,10),(1024,10),
                               (1797,10),(100000,2)) and cpu< -10:
                    go=False
            elif cpu < -10:
                go=False
        for r in doc["real"]:
            k=(r["dataset"],r["label_seed"])
            h=(r["label_sha"],tuple(r["graph_sha"]))
            if k in native_hashes: assert native_hashes[k]==h,"cross-OS real fixture"
            native_hashes[k]=h
        for ds in DATASETS:
            for backend in BACKENDS:
                for early_stop in (False,True):
                    cohort=[r for r in doc["real"] if (r["dataset"],r["backend"],r["early_stop"])
                            ==(ds,backend,early_stop)]
                    assert len(cohort)==2
                    saving=100*(1-statistics.median([
                        r["candidate_timings"]["fit_predict_s"] /
                        r["reference_timings"]["fit_predict_s"] for r in cohort]))
                    if saving < -10: go=False
    summary=dict(schema=1,scientific_completeness="PASS",
                 promotion="GO" if go else "NO-GO",
                 matched_pairs=dict(mapping_cpu=600,mapping_traced_peaks=120,
                                    full_native_fit=96),
                 rows=sorted(rows,key=lambda r:(
                     r["os"],r["nodes"],r["classes"],r["density"])))
    Path("label_mapping_summary.json").write_text(
        json.dumps(summary,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    print("COMPLETE scientific comparison PASS; preregistered promotion:",summary["promotion"])
    # Negative scientific findings do not cause CI infrastructure to fail.


if __name__=="__main__":
    run(Path(sys.argv[1]))
