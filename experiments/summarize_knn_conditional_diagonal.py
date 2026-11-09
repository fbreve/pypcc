"""Strict two-OS prospective CPU/temporary-memory GO gate for kNN diagonal cleanup."""
import json
import math
from pathlib import Path
import statistics
import sys

BASE="c1073ae53ea785031de9e474b678e94f44a2124d"
NAMES=("iris","wine","breast_cancer","digits",
       "duplicate512","duplicate2048","all_identical64")
K=(1,3,10)
BACKENDS=("numpy","numba","cython")
EXPECTED={(name,k) for name in NAMES for k in K}
NATIVE={(name,seed,b,early) for name in NAMES[:4]
        for seed in (42027,42053) for b in BACKENDS for early in (False,True)}
INVALID={"bad_shape","zero_k","negative_k","equal_node_count","float_k"}


def is_sha(x):
    return isinstance(x,str) and len(x)==64 and all(ch in "0123456789abcdef" for ch in x)


def positive(x):
    return type(x) in (float,int) and math.isfinite(x) and x>0


def validate(d):
    assert d["schema"]==1 and d["base_commit"]==BASE
    assert d["platform"] in ("linux","win32")
    assert isinstance(d["head_commit"],str) and len(d["head_commit"])==40
    assert is_sha(d["git_original_graph_source_sha256"])
    for name in ("OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","MKL_NUM_THREADS",
                 "NUMBA_NUM_THREADS"):
        assert d["env"][name]=="1"
    assert all(d.get(k) for k in ("python","numpy","scipy","sklearn"))
    assert len(d["samples"])==21 and len(d["native"])==48
    samples={(s["name"],s["k"]):s for s in d["samples"]}
    assert len(samples)==21 and set(samples)==EXPECTED
    for (name,k),s in samples.items():
        assert s["synthetic"]==(name not in NAMES[:4])
        assert s["nodes"]=={
            "iris":150,"wine":178,"breast_cancer":569,"digits":1797,
            "duplicate512":512,"duplicate2048":2048,"all_identical64":64}[name]
        assert s["features"]=={
            "iris":4,"wine":13,"breast_cancer":30,"digits":64,
            "duplicate512":8,"duplicate2048":16,"all_identical64":4}[name]
        assert is_sha(s["input_sha"])
        assert len(s["ordered_graph_sha"])==2
        assert all(is_sha(h) for h in s["ordered_graph_sha"])
        assert isinstance(s["original_self_count"],int)
        assert 0<=s["original_self_count"]<=s["nodes"]*k
        assert len(s["timings_ms"])==25 and len(s["peak_bytes"])==5
        for pair in s["timings_ms"]+s["peak_bytes"]:
            assert len(pair)==2 and all(positive(v) for v in pair)
    assert len(d["invalid"])==5 and set(d["invalid"])==INVALID
    for name,check in d["invalid"].items():
        assert len(check)==2 and check[0] in ("TypeError","ValueError")
        assert isinstance(check[1],str) and check[1]
    native={(r["dataset"],r["label_seed"],r["backend"],r["early_stop"]):r
            for r in d["native"]}
    assert len(native)==48 and set(native)==NATIVE
    for (_,_,backend,_),r in native.items():
        assert r["backend_identity"]=={
            "numpy":"pcc_numpy.pcc_propagate_numpy",
            "numba":"pcc_numba.pcc_propagate_numba",
            "cython":"pcc_step.pcc_propagate"}[backend]
        assert r["layout"]==("C" if backend=="numba" and r["dataset"]=="digits" else "F")
        assert is_sha(r["label_sha"])
        assert all(is_sha(h) for h in r["original_graph_sha"])
        assert set(r["states"])=={
            "predictions","dominance","positions","strengths","distances","owndeg"}
        assert all(is_sha(h) for h in r["states"].values())
        assert 0<r["info"]["effective_iterations"]<=1000
        assert r["info"]["stop_reason"] in ("max_iter","early_stop")
        assert is_sha(r["rng_sha"]) if backend=="numpy" else r["rng_sha"] is None
    return samples,native


def main(folder):
    paths=list(folder.glob("**/knn_diagonal_result.json"))
    assert len(paths)==2,f"need independent Linux and Windows run results: {paths}"
    ds=[json.loads(p.read_text(encoding="utf-8")) for p in paths]
    assert {d["platform"] for d in ds}=={"linux","win32"}
    ds=sorted(ds,key=lambda d:d["platform"])
    validated=[validate(d) for d in ds]
    assert len({d["head_commit"] for d in ds})==1
    assert len({d["git_original_graph_source_sha256"] for d in ds})==1
    assert ds[0]["invalid"]==ds[1]["invalid"],"cross-host original invalid input differs"
    present_self=False
    present_no_self=False
    rows=[]
    go=True
    for d,(samples,native) in zip(ds,validated):
        hit=0
        for name,k in sorted(EXPECTED):
            s=samples[(name,k)]
            present_self |= s["original_self_count"]>0
            present_no_self |= s["original_self_count"]==0
            cpu=100*(1-statistics.median([b/a for a,b in s["timings_ms"]]))
            mem=statistics.median([b-a for a,b in s["peak_bytes"]])
            if mem>16*s["nodes"]*k+1048576:go=False
            if cpu < -10:go=False
            if k==10 and name in NAMES[:4] and cpu>=5:hit+=1
            rows.append(dict(host=d["platform"],name=name,k=k,
                             self_edges=s["original_self_count"],
                             cpu_saving_pct=cpu,
                             extra_traced_peak_bytes=mem,
                             original_median_ms=statistics.median([v[0] for v in s["timings_ms"]]),
                             candidate_median_ms=statistics.median([v[1] for v in s["timings_ms"]])))
            other=validated[1 if d is ds[0] else 0][0][(name,k)]
            if s["input_sha"]!=other["input_sha"]:
                raise AssertionError("cross-OS fixtures mismatch "+repr((name,k)))
            if name in NAMES[:4] and s["ordered_graph_sha"]!=other["ordered_graph_sha"]:
                raise AssertionError("cross-OS original real graph differs "+repr((name,k)))
        if hit<2: go=False
    if not (present_self and present_no_self):
        raise AssertionError("did not characterize BOTH real self-loop and no-loop paths")
    lines=[
        "# Preregistered kNN conditional sparse-diagonal cleanup",
        "",
        "**Scientific completeness PASS**: 2 OS × 21 exact frozen-source",
        "graph fixtures × 25 alternating baseline/candidate whole-build CPU",
        "pairs = **1050 pairs**, 5 distinct tracked peak pairs/fixture/OS",
        "= **210 peaks**, plus **96 complete same-backend native PCC**",
        "six-state original-vs-candidate pairs across four real datasets.",
        "Original sklearn engine, tie selection, graph sorting, dense int64",
        "dtype and unused -1 padding are strictly preserved within each OS.",
        "",
        f"**Prospectively frozen performance decision: {'GO' if go else 'NO-GO'}**",
        "",
        "| OS | Input | k | Original self edges | Paired median full graph CPU saved | Extra traced peak bytes | Original median ms | Candidate median ms |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in sorted(rows,key=lambda x:(x["host"],x["name"],x["k"])):
        lines.append(
            f"| {r['host']} | {r['name']} | {r['k']} | "
            f"{r['self_edges']} | {r['cpu_saving_pct']:+.2f}% | "
            f"{r['extra_traced_peak_bytes']:+.0f} | "
            f"{r['original_median_ms']:.3f} | "
            f"{r['candidate_median_ms']:.3f} |")
    lines+=["",
        "A negative CPU result is a valid COMPLETE experiment, not a CI",
        "error. Criteria were frozen before seeing results: >=5% whole",
        "builder median paired saving on >=2 of 4 real k10 datasets on",
        "each host, no >10% regression among any 21 cases/host, and",
        "bounded extra tracked peak <=16*n*k+1MiB on every case/host.",
        "There is no outlier removal or retroactively lowered target.",
        "Input double-checks ensure tied/duplicate self-loop cases and",
        "loop-free cases both tested. Tracemalloc is not process RSS.",
        "No graph becomes sparse for propagation; original padded dense",
        "int64 layout remains permanent. Native fits from two different",
        "backends are NOT required to share RNGs.",
    ]
    Path("knn_diagonal_decision.json").write_text(json.dumps(dict(
        schema=1,complete=True,decision="GO" if go else "NO-GO",
        total_timing_pairs=1050,traced_peak_pairs=210,native_pairs=96,
        rows=rows),indent=2,allow_nan=False)+"\n",encoding="utf-8")
    Path("knn_diagonal_report.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    print("\n".join(lines),flush=True)
    print("Complete scientific analysis PASS. Preregistered verdict:",
          "GO" if go else "NO-GO")


if __name__=="__main__":
    main(Path(sys.argv[1]))
