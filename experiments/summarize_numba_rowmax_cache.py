"""Strict mandatory Linux/Windows source-paired Numba rowmax CPU-for-memory decision."""
import json
import math
from pathlib import Path
import statistics
import sys

BASE="d25d5446b7fa51cc0264a496b7eea87dbb11f1fc"
DATASETS=("iris","wine","breast_cancer","digits")
SEEDS=(62011,62137)
MODES=("monitored","fixed")
VARIANTS=("original","candidate")
STATES={"predictions","dominance","positions","strengths","distances","owndeg"}
CASES={(ds,s) for ds in DATASETS for s in SEEDS}


def chk(ok,reason):
    if not ok:
        raise AssertionError(reason)


def sha(s):
    return isinstance(s,str) and len(s)==64 and all(ch in "0123456789abcdef" for ch in s)


def num(s):
    return type(s) in (int,float) and math.isfinite(s) and s>0


def validate_fit(x,mode,cap,peaks=False):
    chk(x["iteration_cap"]==cap,"native Numba cap different")
    chk(x["mode"]==mode,"wrong original monitored/fixed path")
    chk(type(x["effective_iterations"]) is int
        and 0<x["effective_iterations"]<=cap,"invalid native iteration count")
    chk(x["stop_reason"] in ("max_iter","early_stop"),"native stop metrics missing")
    chk(type(x["stop_max"]) is int and x["stop_max"]>=1,"no original patience")
    chk(type(x["checks"]) is int and x["checks"]>=0,"missing monitoring checkpoints")
    chk(type(x["stalled_checks"]) is int and x["stalled_checks"]>=0,
        "missing plateau monitoring counter")
    if mode=="fixed":
        chk(x["effective_iterations"]==cap and x["stop_reason"]=="max_iter",
            "fixed Numba path stopped early")
        chk(x["checks"]==0,"fixed Numba must not invoke potential monitor")
    elif x["effective_iterations"]<cap:
        chk(x["stop_reason"]=="early_stop",
            "native Numba premature exit without real early stop")
    chk(num(x["elapsed_ns"]) and num(x["propagation_elapsed_ns"]),
        "missing full fit or real native propagation time")
    chk(x["propagation_elapsed_ns"]<=x["elapsed_ns"]*1.01,
        "native propagation time exceeds whole model fit")
    chk(set(x["state_hashes"])==STATES and
        all(sha(h) for h in x["state_hashes"].values()),
        "not all six original native states hashed")
    chk(sha(x["mapped_labels_hash"]) and
        x["predicted_hash"]==x["state_hashes"]["predictions"],
        "missing original predictions/label mapping provenance")
    chk(type(x["rng_next"]) is float and math.isfinite(x["rng_next"])
        and 0<=x["rng_next"]<1,"native Numba RNG next value missing")
    chk(x["peak_bytes"] is None if not peaks else num(x["peak_bytes"]),
        "invalid separate tracked memory evidence")


def one(doc):
    chk(doc["schema"]==1 and doc["base"]==BASE,"wrong production source")
    chk(doc["platform"] in ("linux","win32"),"wrong OS")
    chk(len(doc["branch_head"])==40 and sha(doc["source_sha256"])
        and sha(doc["candidate_sha256"]),"missing original Git source identity")
    chk(len(doc["cases"])==8,"missing any original real dataset/mask")
    chk(all(doc["threads"].get(v)=="1" for v in (
        "OPENBLAS_NUM_THREADS","OMP_NUM_THREADS",
        "MKL_NUM_THREADS","NUMBA_NUM_THREADS")),
        "native CPU BLAS/Numba threads not controlled")
    seen={}
    for case in doc["cases"]:
        key=(case["dataset"],case["seed"])
        chk(key in CASES and key not in seen,"wrong fixture or duplicate")
        ds,seed=key
        chk(case["case_id"]==f"{ds}-labels{seed}","wrong fixture case id")
        chk(case["nodes"]=={"iris":150,"wine":178,"breast_cancer":569,"digits":1797}[ds],
            "wrong number of original nodes")
        chk(case["classes"]=={"iris":3,"wine":3,"breast_cancer":2,"digits":10}[ds],
            "wrong original class count")
        chk(case["graph_reused"]==(seed==SEEDS[1]),"wrong one-time kNN reuse")
        chk(sha(case["input_sha"]) and sha(case["labels_sha"])
            and sha(case["fixture_sha"]) and
            len(case["graph_sha"])==2 and all(sha(v) for v in case["graph_sha"]),
            "missing graph/dataset/mask fingerprints")
        chk(len(case["samples"])==12 and len(case["peaks"])==6,
            "incomplete full CPU or separately traced memory pairs")
        chk(len(case["warmups"])==4,
            "both native Numba compiled monitored/fixed paths not warmed")
        chk(case["audited_checkpoints"]==201,
            "missing exact original 201 potential checks for every case")
        chk(set(case["verified_state_hashes"])==STATES and
            all(sha(v) for v in case["verified_state_hashes"].values())
            and type(case["verified_rng_next"]) is float,
            "missing checked cached full native short-run parity")
        warms=set()
        for w in case["warmups"]:
            identity=(w["variant"],w["mode"])
            chk(identity in {(v,m) for v in VARIANTS for m in MODES}
                and identity not in warms,"wrong duplicate native warmup mode")
            validate_fit(w,w["mode"],2)
            warms.add(identity)
        chk(len(warms)==4,"missing a whole compiled Numba specialization")
        measurements=set()
        reference_by_mode={}
        for pair in case["samples"]:
            mode=pair["mode"]
            rep=pair["repetition"]
            ident=(mode,rep)
            chk(mode in MODES and type(rep) is int and 0<=rep<6
                and ident not in measurements,"duplicate or missing CPU pair")
            chk(pair["order"]==(["candidate","original"] if rep%2==0
                                else ["original","candidate"]),
                "counterbalanced original/candidate order changed")
            for variant in VARIANTS:
                validate_fit(pair[variant],mode,30000)
            a,b=pair["original"],pair["candidate"]
            for k in ("state_hashes","rng_next","mapped_labels_hash",
                      "effective_iterations","stop_reason","checks",
                      "stalled_checks","stop_max"):
                chk(a[k]==b[k],"Numba source-paired state/stop/RNG differs: "+k)
            fingerprint=(a["state_hashes"],a["rng_next"],
                         a["effective_iterations"],a["stop_reason"],
                         a["stalled_checks"],a["checks"])
            chk(mode not in reference_by_mode
                or fingerprint==reference_by_mode[mode],
                "original native baseline not deterministic across six repeats")
            reference_by_mode[mode]=fingerprint
            measurements.add(ident)
        chk(measurements=={(mode,i) for mode in MODES for i in range(6)},
            "missing original CPU repetition")
        traced=set()
        for pair in case["peaks"]:
            mode=pair["mode"]
            rep=pair["repetition"]
            key2=(mode,rep)
            chk(key2 not in traced and mode in MODES and 0<=rep<3,
                "duplicate/invalid tracked memory pair")
            for variant in VARIANTS:
                validate_fit(pair[variant],mode,30000,peaks=True)
            for field in ("state_hashes","rng_next","effective_iterations",
                          "stop_reason","stalled_checks","checks"):
                chk(pair["original"][field]==pair["candidate"][field],
                    "traced Numba source parity differs: "+field)
            traced.add(key2)
        chk(traced=={(mode,i) for mode in MODES for i in range(3)},
            "missing original 3 traced peak pairs per mode")
        seen[key]=case
    chk(set(seen)==CASES,"not all eight precommitted original fixture masks")
    return seen


def summarize(docs,verified):
    out=[
        "# Numba exact row-maximum cache versus original: prospectively frozen CPU trial",
        "",
        "**Scientific completeness PASSED:** frozen current production source;",
        "two independent OS, 192 complete alternating source-paired FULL",
        "fit_predict CPU timing pairs; 96 separately traced full-fit peak pairs;",
        "16 independent 201-check exact cached-vs-original potential audits;",
        "32 original/candidate 2-step Numba warmed pairs (both compiled modes),",
        "and strict six-array state, native stop counters and next Numba RNG",
        "parity across ALL original vs candidate pairs. Both variants are",
        "compiled from the original Git source with audited unique transforms;",
        "NO production Python/Numba/Cython package source files were changed.",
        "",
        "## Frozen GO/NO-GO CPU-for-memory decision",
        "",
        "| Host | Dataset | Mode | Actual early-stopped masks | Median FULL-fit paired CPU saved | Median original full fit ms | Median candidate full fit ms | Median additional traced peak bytes |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    rows=[]
    go=True
    for doc, cases in zip(docs,verified):
        host=doc["platform"]
        for ds in DATASETS:
            for mode in MODES:
                observed=[row for seed in SEEDS for row in cases[(ds,seed)]["samples"]
                          if row["mode"]==mode]
                trace=[row for seed in SEEDS for row in cases[(ds,seed)]["peaks"]
                       if row["mode"]==mode]
                chk(len(observed)==12 and len(trace)==6,
                    "incomplete across seed/repeat independent CPU data")
                save=100*(1-statistics.median(
                    x["candidate"]["elapsed_ns"]/x["original"]["elapsed_ns"]
                    for x in observed))
                peak=statistics.median(
                    x["candidate"]["peak_bytes"]-x["original"]["peak_bytes"]
                    for x in trace)
                original_ms=statistics.median(x["original"]["elapsed_ns"]/1e6 for x in observed)
                candidate_ms=statistics.median(x["candidate"]["elapsed_ns"]/1e6 for x in observed)
                actual_stops=sum(cases[(ds,seed)]["samples"][0 if mode=="monitored" else 6]
                    ["original"]["stop_reason"]=="early_stop" for seed in SEEDS)
                budget=8*{"iris":150,"wine":178,"breast_cancer":569,"digits":1797}[ds]+2*1024*1024
                if peak>budget:go=False
                if mode=="monitored":
                    if ds=="digits" and save<8:go=False
                    if ds=="breast_cancer" and save<5:go=False
                    if ds in ("iris","wine") and save< -5:go=False
                elif save< -5:go=False
                rows.append(dict(host=host,dataset=ds,mode=mode,
                                 stopped_masks=actual_stops,cpu_saved_pct=save,
                                 peak_delta_bytes=peak,
                                 original_median_ms=original_ms,
                                 candidate_median_ms=candidate_ms))
                out.append(
                    f"| {host} | {ds} | {mode} | {actual_stops}/2 | "
                    f"{save:+.2f}% | {original_ms:.3f} | {candidate_ms:.3f} | "
                    f"{peak:+.0f} |")
    out.extend([
        "",
        f"**Preregistered verdict: {'GO' if go else 'NO-GO'}**.",
        "",
        "GO requires monitored whole-fit >=8% saving on Digits AND >=5%",
        "on Breast Cancer on EACH OS, no monitored Iris/Wine median",
        "regression worse than 5%, and NO fixed-path median regression",
        "worse than 5% on any dataset/OS. Traced extra peak must fit",
        "<=8*nodes+2MiB. No thresholds were adjusted after results.",
        "Scientific complete NO-GO is a valid finding and MUST NOT fail",
        "the CI aggregate. CPU savings, not RAM compression, is the goal.",
        "",
        "## Limitations",
        "",
        "- Results are matched original and separately JIT-compiled candidate",
        "  full model fits on each host (not a production merge). Native",
        "  JIT is excluded from timings by real two-mode warmups.",
        "- Memory uses separate Python tracemalloc peaks, not process RSS",
        "  or a claim that native allocations have been measured exhaustively.",
        "- The candidate writes an n_nodes float64 cache for early-stop only;",
        "  fixed native propagation and its original step are unmodified.",
        "  Graphs intentionally remain CPU-optimized padded dense int64.",
        "- Python model construction, set_graph and one-time kNN preparation",
        "  are outside the fit_predict timing boundary.",
        "- Stop metadata, floating checkpoint exactness, all six native",
        "  final arrays, and next Numba RNG outputs are separately compared.",
        "- Censored 30k cases may stop at higher caps; no alternate stopping",
        "  criterion or changed plateau threshold is implied.",
        "- Separate GitHub Windows/Linux runners and native binaries vary;",
        "  never pool raw durations or claim guaranteed speedups.",
        "- Even scientific GO requires independently reviewed minimal",
        "  master-based production source and four-platform regression.",
    ])
    return "\n".join(out)+"\n",go,rows


def main(root):
    files=list(Path(root).glob("**/numba_rowmax_results.json"))
    chk(len(files)==2,f"Require both complete original OS artifacts; found {files}")
    docs=[json.loads(p.read_text(encoding="utf-8")) for p in files]
    chk({d["platform"] for d in docs}=={"linux","win32"},"wrong OS set")
    docs.sort(key=lambda d:d["platform"])
    verified=[one(d) for d in docs]
    for key in CASES:
        a,b=(q[key] for q in verified)
        for name in ("input_sha","labels_sha","graph_sha","fixture_sha"):
            chk(a[name]==b[name],"cross-OS standard input differs: "+name)
    for name in ("base","branch_head","source_sha256","candidate_sha256"):
        chk(len({json.dumps(d[name],sort_keys=True) for d in docs})==1,
            "different candidate/original Git source across hosts: "+name)
    report,go,rows=summarize(docs,verified)
    Path("numba_rowmax_decision.json").write_text(
        json.dumps(dict(schema=1,complete=True,verdict="GO" if go else "NO-GO",
                        full_cpu_pairs=192,tracked_peak_pairs=96,
                        bitwise_potential_verification_fits=16,
                        checked_potential_values=3216,warmup_pairs=32,
                        rows=rows),indent=2,allow_nan=False)+"\n",encoding="utf-8")
    Path("numba_rowmax_report.md").write_text(report,encoding="utf-8")
    print(report,flush=True)
    print("STRICT BOTH-HOST ORIGINAL VS CACHED NUMBA COMPLETENESS PASS; "
          "frozen decision:",("GO" if go else "NO-GO"),flush=True)


if __name__=="__main__":
    main(sys.argv[1])
