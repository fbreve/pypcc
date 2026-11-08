"""Validate complete original-master study; negative performance is scientific NO-GO."""
import json
import math
from pathlib import Path
import statistics
import sys

SIZES={(150,3),(1797,10),(30000,10),(30000,50),(100000,50),(100000,2)}
DATASETS={"Iris","Wine","Breast Cancer","Digits"}
BACKENDS={"numpy","numba","cython"}
EXPECTED_REAL={(d,s,b,e) for d in DATASETS for s in (35027,35053)
               for b in BACKENDS for e in (False,True)}
BASE="7e589e2c9097a507d636b55c6269c0e3223e8343"
TIMERS=("fit_predict_s","initialization_and_dispatch_s",
        "propagation_s","postprocessing_s")


def valid_sha(v):
    return isinstance(v,str) and len(v)==64 and all(x in "0123456789abcdef" for x in v)


def check(doc):
    assert doc["schema"]==1 and doc["base_commit"]==BASE
    assert doc["os"] in ("linux","win32")
    assert valid_sha(doc["git_blob_sha256"])
    assert len(doc["micro"])==12 and len(doc["real"])==48
    micro={(r["nodes"],r["classes"],r["backend"]):r for r in doc["micro"]}
    assert len(micro)==12
    assert set(micro)=={(n,c,b) for n,c in SIZES for b in ("numpy","numba")}
    for (n,c,b),r in micro.items():
        assert valid_sha(r["mapped_sha"])
        assert len(r["array_hashes"])==3
        assert all(valid_sha(x) for x in r["array_hashes"])
        assert len(r["timings_ms"])==20
        assert len(r["traced_peaks_bytes"])==4
        expected_order="C" if b=="numba" and c>=8 else "F"
        assert r["layout"][1 if expected_order=="C" else 2]
        assert r["layout"][4]  # owndeg remains Fortran ordered
        for entry in r["timings_ms"]+r["traced_peaks_bytes"]:
            assert len(entry)==2
            assert all(isinstance(x,(float,int)) and math.isfinite(x) and x>0 for x in entry)
    real={(r["dataset"],r["label_seed"],r["backend"],r["early_stop"]):r
          for r in doc["real"]}
    assert len(real)==48 and set(real)==EXPECTED_REAL
    for (_,_,backend,_),r in real.items():
        assert valid_sha(r["label_sha"])
        assert len(r["graph_sha"])==2 and all(valid_sha(x) for x in r["graph_sha"])
        assert set(r["state_hashes"])=={"predictions","dominance","positions",
                                       "strengths","distances","owndeg"}
        assert all(valid_sha(x) for x in r["state_hashes"].values())
        assert r["backend_identity"]=={
            "numpy":"pcc_numpy.pcc_propagate_numpy",
            "numba":"pcc_numba.pcc_propagate_numba",
            "cython":"pcc_step.pcc_propagate"}[backend]
        assert r["layout"]==("C" if backend=="numba" and r["dataset"]=="Digits" else "F")
        if backend=="numpy": assert valid_sha(r["rng_sha"])
        else: assert r["rng_sha"] is None
        info=r["info"]
        assert 0 < info["effective_iterations"] <= 1000
        assert info["stop_reason"] in ("max_iter","early_stop")
        for label in ("reference_timings","candidate_timings"):
            timing=r[label]
            assert set(timing)==set(TIMERS)
            assert all(isinstance(x,(int,float)) and math.isfinite(x) and x>0
                       for x in timing.values())


def main(root):
    paths=list(root.glob("**/nodes_onehot_result.json"))
    assert len(paths)==2,f"need 2 independent OS result files, received {paths}"
    reports=[json.loads(p.read_text(encoding="utf-8")) for p in paths]
    assert {d["os"] for d in reports}=={"linux","win32"}
    for d in reports:check(d)
    assert len({d["git_blob_sha256"] for d in reports})==1
    fixture_hashes={}
    rows=[]
    gate=True
    print("| OS | Nodes x classes | Backend/layout | CPU saving | Peak saving |")
    print("|---|---|---|---:|---:|")
    for doc in reports:
        os=doc["os"]
        for r in doc["micro"]:
            k=(r["nodes"],r["classes"],r["backend"])
            h=(r["mapped_sha"],tuple(r["array_hashes"]))
            if k in fixture_hashes: assert fixture_hashes[k]==h,"cross-host fixture mismatch"
            fixture_hashes[k]=h
            cpu=100*(1-statistics.median([b/a for a,b in r["timings_ms"]]))
            mem=100*(1-statistics.median([b/a for a,b in r["traced_peaks_bytes"]]))
            rows.append(dict(os=os,nodes=r["nodes"],classes=r["classes"],
                             backend=r["backend"],cpu_saving_pct=cpu,peak_saving_pct=mem))
            print(f"| {os} | {r['nodes']} x {r['classes']} | {r['backend']} / "
                  f"{'C' if r['layout'][1] else 'F'} | {cpu:+.2f}% | {mem:+.2f}% |")
            if mem < -10: gate=False
            if r["nodes"] in (150,1797) and cpu < -10: gate=False
            if r["nodes"]==30000 and cpu < 15: gate=False
    real_fixture={}
    for doc in reports:
        os=doc["os"]
        for r in doc["real"]:
            k=(r["dataset"],r["label_seed"])
            h=(r["label_sha"],tuple(r["graph_sha"]))
            if k in real_fixture: assert real_fixture[k]==h,"cross-host native fixture mismatch"
            real_fixture[k]=h
        for ds in DATASETS:
            for backend in BACKENDS:
                for early in (False,True):
                    subset=[r for r in doc["real"] if
                            (r["dataset"],r["backend"],r["early_stop"])==(ds,backend,early)]
                    assert len(subset)==2
                    reduction=100*(1-statistics.median(
                        [r["candidate_timings"]["fit_predict_s"] /
                         r["reference_timings"]["fit_predict_s"] for r in subset]))
                    if reduction < -10: gate=False
    summary=dict(schema=1,scientific_completeness="PASS",
                 candidate_promotion="GO" if gate else "NO-GO",
                 pairs=dict(method_cpu=480,method_tracemalloc=96,native=96),
                 rows=sorted(rows,key=lambda r:(r["os"],r["nodes"],r["classes"],r["backend"])))
    Path("nodes_onehot_summary.json").write_text(
        json.dumps(summary,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    print("Scientific completeness PASS; frozen candidate decision:",summary["candidate_promotion"])
    # Observed NO-GO is a valid experiment, not a CI infrastructure failure.


if __name__=="__main__":
    main(Path(sys.argv[1]))
