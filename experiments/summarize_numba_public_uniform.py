"""Strict two-OS complete source-exact Numba original MT19937 pointer CPU trial.

Never confuse expected semantic equality with CPU GO. All original/candidate
real full-fit seeded native arrays, native RNG endpoint, stop information,
memory and source/graph proofs must be present independent of performance.
"""
import json
import math
from pathlib import Path
import statistics
import sys

BASE="b4c100bd22eeff1979a1a581bc73682b8c3705b8"
DATASETS=("iris","wine","breast_cancer","digits")
SEEDS=(76013,76117)
MODES=("monitored","fixed")
RNG_K=(1,2,3,10,17,64,255,1000,2147483649)
STATES={"predictions","dominance","positions","strengths","distances","owndeg"}
FIELDS=("state_hashes","rng_next","mapped_labels_hash",
        "effective_iterations","stop_reason","checks",
        "stalled_checks","stop_max")
THREADS=("OPENBLAS_NUM_THREADS","OMP_NUM_THREADS",
         "MKL_NUM_THREADS","NUMBA_NUM_THREADS")


def must(ok,reason):
    if not ok:raise AssertionError(reason)


def sha(x):
    return isinstance(x,str) and len(x)==64 and all(c in "0123456789abcdef" for c in x)


def finitepos(x):
    return type(x) in (int,float) and math.isfinite(x) and x>0


def exact(a,b,where):
    for key in FIELDS:
        must(a[key]==b[key],"source pair Numba MT state/stop/full prediction differs "+
             repr((where,key)))


def validate_fit(x,mode,cap,tracked=False):
    must(x["cap"]==cap and x["mode"]==mode and
         x["stop_reason"] in ("max_iter","early_stop") and
         type(x["effective_iterations"]) is int and
         0<=x["effective_iterations"]<=cap and
         x["effective_iterations"]==(cap if mode=="fixed" else
                                     x["effective_iterations"]),
         "native stopping path not original cap/mode")
    must((cap==0 and x["effective_iterations"]==0) or
         (cap>0 and x["effective_iterations"]>0),
         "missing original native iteration count")
    must(type(x["checks"]) is int and
         type(x["stalled_checks"]) is int and
         type(x["stop_max"]) is int,
         "actual native original info metrics missing")
    must((x["effective_iterations"]<cap)==
         (x["stop_reason"]=="early_stop") if mode=="monitored" else
         x["stop_reason"]=="max_iter",
         "Numba original stop status changed")
    must(set(x["state_hashes"])==STATES and all(
        sha(s) for s in x["state_hashes"].values()),
        "missing six original final arrays")
    must(sha(x["mapped_labels_hash"]) and
         type(x["rng_next"]) is float and 0<=x["rng_next"]<1,
         "original mapped labels or next Numba random draw missing")
    must(x["native_dominance_layout"] in ("C","F"),
         "C/F layout of original Numba not recorded")
    must(type(x["cpu_ns"]) is int and x["cpu_ns"]>=0 and
         finitepos(x["wall_ns"]),
         "full original native fit timing missing")
    if tracked:
        must(type(x["peak_bytes"]) is int and x["peak_bytes"]>0,
             "missing independently observed full-fit tracemalloc")
    else:
        must(x["peak_bytes"] is None,"CPU timing contaminated by memory tracer")


def validate_batch(b,mode):
    must(b["sample_count"]==8 and b["all_same"] is True,
         "eight full fit results not identical")
    must(finitepos(b["total_cpu_ns"]) and
         finitepos(b["total_wall_ns"]),
         "Windows-resolved eight original full fits CPU/wall missing")
    must(type(b["mean_fit_cpu_ns"]) is float and
         b["mean_fit_cpu_ns"]==b["total_cpu_ns"]/8,
         "eight real fitted CPU time does not agree with full batch")
    must(type(b["mean_fit_wall_ns"]) is float and
         b["mean_fit_wall_ns"]==b["total_wall_ns"]/8,
         "eight real fitted wall time does not agree with full batch")
    must(len(b["each_fit_full_state_hashes"])==8 and
         all(row==b["final"]["state_hashes"]
             for row in b["each_fit_full_state_hashes"]),
         "not all eight native original full predictions/arrays equal")
    x=b["final"]
    must(x["cap"]==30000 and x["mode"]==mode and
         0<x["effective_iterations"]<=30000,
         "full original native 30k monitored/fixed batch missing")
    must(x["native_dominance_layout"] in ("C","F") and
         set(x["state_hashes"])==STATES and
         all(sha(v) for v in x["state_hashes"].values()),
         "missing complete native array states after eight fits")
    must(sha(x["mapped_labels_hash"]) and
         type(x["rng_next"]) is float and 0<=x["rng_next"]<1,
         "missing exact original final Numba MT random state")
    must(type(x["checks"]) is int and
         type(x["stalled_checks"]) is int and
         type(x["stop_max"]) is int and
         x["stop_reason"] in ("early_stop","max_iter"),
         "missing true native stopping metadata")
    must((x["effective_iterations"]<30000)==(
         x["stop_reason"]=="early_stop") if mode=="monitored" else
         (x["effective_iterations"]==30000 and
          x["stop_reason"]=="max_iter"),
         "true source Numba stop status wrong")


def validate_doc(d):
    must(d["schema"]==1 and d["base"]==BASE and
         d["platform"] in ("linux","win32") and
         len(d["head"])==40 and
         sha(d["source_sha256"]) and sha(d["candidate_sha256"]) and
         d["source_sha256"]!=d["candidate_sha256"],
         "scientific branch not original frozen production source")
    must(d["full_fit_batch_size"]==8 and
         all(d["threading"].get(x)=="1" for x in THREADS),
         "sample CPU count/thread cap not preregistered")
    rng=d["rng_k_cases"]
    must(len(rng)==len(RNG_K) and
         [x["k"] for x in rng]==list(RNG_K) and
         all(sha(x["samples_hash"]) and
             type(x["next_numba_rng"]) is float and
             0<=x["next_numba_rng"]<1 for x in rng),
         "MT19937 exact uniform/random/randint, k=1 and large-int controls missing")
    edges=d["uniform_edge_cases"]
    frozen=("0.0","0.1","1.0","1.23","3.7","100000.0",
            "1e-300","1e+307","inf","nan")
    must(len(edges)==10 and
         [(v["index"],v["total_repr"]) for v in edges]==
         list(enumerate(frozen)) and
         all(sha(v["stream_bit_sha"]) and
             type(v["next_numba_rng"]) is float and
             0<=v["next_numba_rng"]<1 for v in edges),
         "missing ten exact bit64 Numba uniform vs random finite/Inf/NaN and RNG edge controls")
    must(len(d["cases"])==8,"missing any four real original graph datasets/seeds")
    cases={}
    for case in d["cases"]:
        ds,seed=case["dataset"],case["seed"]
        key=(ds,seed)
        must(ds in DATASETS and seed in SEEDS and key not in cases and
             case["case_id"]==f"{ds}-labels{seed}",
             "duplicate or wrong case/dataset/seed")
        must(case["nodes"]=={"iris":150,"wine":178,"breast_cancer":569,
                             "digits":1797}[ds] and
             case["classes"]=={"iris":3,"wine":3,"breast_cancer":2,
                               "digits":10}[ds],
             "unexpected real native dataset")
        must(case["graph_reused"]==(seed==SEEDS[1]),
             "original kNN preprocessing not reused correctly")
        must(all(sha(case[x]) for x in ("fixture_sha","labels_sha","input_sha"))
             and len(case["graph_sha"])==2 and
             all(sha(x) for x in case["graph_sha"]),
             "original symmetric dense graph or labels SHA incomplete")
        must(len(case["samples"])==12 and len(case["peaks"])==4 and
             len(case["warmups"])==4 and
             len(case["policy_cases"])==8 and len(case["no_seed"])==2,
             "missing source-paired original full CPU/memory/branch RNG cases")
        seen=set()
        per_mode_ref={}
        for trial in case["samples"]:
            mode,rep=trial["mode"],trial["rep"]
            must(mode in MODES and type(rep) is int and 0<=rep<6 and
                 (mode,rep) not in seen,
                 "wrong or duplicate balanced eight-fit CPU sample")
            must(trial["order"]==(
                ["candidate","original"] if rep%2==0
                else ["original","candidate"]),
                "source cpu timing order not interleaved")
            for var in ("original","candidate"):
                validate_batch(trial[var],mode)
                must(trial[var]["final"]["native_dominance_layout"]==
                     ("C" if case["classes"]>=8 else "F"),
                     "not actual C/F layout in original/candidate")
                result=trial[var]["final"]
                if mode not in per_mode_ref:
                    per_mode_ref[mode]={v:result[v] for v in FIELDS}
                for field in FIELDS:
                    must(result[field]==per_mode_ref[mode][field],
                         "original repeated seeded MT and all six states drift "+
                         repr((key,mode,rep,field)))
            exact(trial["original"]["final"],trial["candidate"]["final"],
                  (key,mode,rep))
            seen.add((mode,rep))
        must(seen=={(mode,i) for mode in MODES for i in range(6)},
             "incomplete original/candidate CPU cohorts")
        seen=set()
        for trial in case["peaks"]:
            mode,rep=trial["mode"],trial["rep"]
            must(mode in MODES and type(rep) is int and 0<=rep<2
                 and (mode,rep) not in seen,
                 "wrong memory tracer independent source pair")
            for variant in ("original","candidate"):
                validate_fit(trial[variant],mode,30000,tracked=True)
            exact(trial["original"],trial["candidate"],
                  (key,"memory",mode,rep))
            seen.add((mode,rep))
        must(seen=={(mode,i) for mode in MODES for i in range(2)},
             "missing independent memory traces")
        seen=set()
        for trial in case["warmups"]:
            mode,variant=trial["mode"],trial["variant"]
            must(mode in MODES and variant in ("original","candidate") and
                 (mode,variant) not in seen,
                 "duplicate or missing original native 2-step JIT warmup")
            validate_fit(trial,mode,2)
            must(trial["effective_iterations"]==2,
                 "native Numba JIT mode not really warmed")
            seen.add((mode,variant))
        must(len(seen)==4,"not all original Numba modes/wrappers JIT compiled")
        seen=set()
        for trial in case["policy_cases"]:
            cap,grd,mode=trial["cap"],trial["p_grd"],trial["mode"]
            key2=(mode,cap,grd)
            must(mode in MODES and cap in (2,1001) and
                 grd in (0.,1.) and key2 not in seen,
                 "not all original random-only/greedy-only edge policies tested")
            for var in ("original","candidate"):
                validate_fit(trial[var],mode,cap)
            exact(trial["original"],trial["candidate"],
                  (key,"native-extremes",key2))
            seen.add(key2)
        must(len(seen)==8,"edge policy source-paired cases incomplete")
        seen=set()
        for trial in case["no_seed"]:
            cap=trial["cap"]
            must(cap in (0,5) and cap not in seen,
                 "missing 0-iteration/5-step unseeded continuation")
            for var in ("original","candidate"):
                validate_fit(trial[var],"fixed",cap)
            exact(trial["original"],trial["candidate"],
                  (key,"no-explicit-seed",cap))
            seen.add(cap)
        must(seen=={0,5},"no-seed native RNG boundaries missing")
        cases[key]=case
    must(set(cases)=={(d,s) for d in DATASETS for s in SEEDS},
         "missing exact all independent data/label masks")
    return cases


def report(docs,verified):
    lines=[
        "# Exact public Numba uniform(0,total) versus random()*total: full-fit CPU",
        "",
        "SCIENCE COMPLETENESS: 192 interleaved source CPU batch pairs",
        "(8 original 30k-cap full seeded fits/variant/batch; 3,072",
        "timed native full fits), 64 original/candidate separately",
        "traced peak memory pairs, 64 native JIT 2-step warmups,",
        "128 p_grd=0/1 forced boundary source pairs, 32 native",
        "0-or-5 iteration no-explicit-seed pairs, and two-OS 9-k",
        "Numba original np.random.random/uniform/randint MT19937 stream",
        "controls, with 17 draws+next sample per k.",
        "",
        "Real Iris/Wine/Breast Cancer/Digits undirected union kNN10,",
        "two independent class-balanced labeled masks (3/class),",
        "genuine original Numba C/F class memory layout. Original master",
        "source vs exactly ONE greedy uniform call expression after particle",
        "guards; both original np.random.randint(k) branches unchanged.",
        "All six exact final states, next original Numba RNG, mapped labels",
        "and original native stop/counters match in every source pair.",
        "Only public Numba-supported NumPy RNG functions are involved.",
        "",
        "| OS | Real dataset | Stop mode | Paired 8-fit native CPU saved | Original per-fit CPU ms | Candidate per-fit CPU ms | Maximum Python traced peak extra bytes |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    rows=[]
    go=True
    for d,cases in zip(docs,verified):
        for ds in DATASETS:
            n={"iris":150,"wine":178,"breast_cancer":569,"digits":1797}[ds]
            for mode in MODES:
                trials=[p for seed in SEEDS for p in cases[(ds,seed)]["samples"]
                        if p["mode"]==mode]
                peaks=[p for seed in SEEDS for p in cases[(ds,seed)]["peaks"]
                       if p["mode"]==mode]
                must(len(trials)==12 and len(peaks)==4,
                     "two seed×six CPU batch and memory cohorts missing")
                saved=100*(1-statistics.median(
                    p["candidate"]["total_cpu_ns"]/
                    p["original"]["total_cpu_ns"] for p in trials))
                old_ms=statistics.median(
                    p["original"]["mean_fit_cpu_ns"] for p in trials)/1e6
                new_ms=statistics.median(
                    p["candidate"]["mean_fit_cpu_ns"] for p in trials)/1e6
                extra=max(p["candidate"]["peak_bytes"]-
                          p["original"]["peak_bytes"] for p in peaks)
                if saved< -5 or extra>8*n+1048576:
                    go=False
                if ds in ("breast_cancer","digits") and saved<5:
                    go=False
                rows.append(dict(os=d["platform"],dataset=ds,mode=mode,
                                 cpu_saved_pct=saved,original_fit_cpu_ms=old_ms,
                                 candidate_fit_cpu_ms=new_ms,
                                 max_python_peak_extra_bytes=extra))
                lines.append(
                    f"| {d['platform']} | {ds} | {mode} | "
                    f"{saved:+.2f}% | {old_ms:.3f} | {new_ms:.3f} | {extra:+d} |")
    label="GO" if go else "NO-GO"
    lines.extend([
        "",
        f"**Precommitted scientific verdict: {label}.**",
        "",
        "GO requires >=5% savings in BOTH Breast Cancer and Digits",
        "in BOTH native stop modes on EACH OS, <=5% slowdown for every",
        "other case, and extra tracemalloc peak <=8*n+1MiB. The extra",
        "traced memory is a BUDGET to permit saving CPU, not a demand",
        "to reduce RAM. Same exact Numba seeded MT19937 path is mandatory.",
        "Raw CPU batches are technical timing repetitions, not independent",
        "label-mask trials, and OS hosted hardware cannot be pooled.",
        "Callgrind instruction % are NOT equivalent to CPU-time percentages;",
        "source edit uses only public np.random.random and uniform APIs;",
        "even a scientific GO would still require",
        "independent production compatibility design and 4-platform CI.",
        "This SCIENCE branch is DRAFT/UNMERGED in all outcomes, without",
        "changes to actual master or the padded dense int64 kNN graph.",
    ])
    return "\n".join(lines)+"\n",label,rows


def main(path):
    docs_file=list(Path(path).glob("**/numba_public_uniform_results.json"))
    must(len(docs_file)==2,"require complete original Linux and Windows raw corpora")
    docs=[json.loads(p.read_text(encoding="utf-8")) for p in docs_file]
    must({doc["platform"] for doc in docs}=={"linux","win32"},
         "both separate Numba OS and native CPU measurements required")
    docs.sort(key=lambda x:x["platform"])
    verified=[validate_doc(doc) for doc in docs]
    for field in ("base","head","source_sha256","candidate_sha256"):
        must(len({doc[field] for doc in docs})==1,
             "source identity drift between native original host results "+field)
    for k in RNG_K:
        row1=docs[0]["rng_k_cases"][RNG_K.index(k)]
        row2=docs[1]["rng_k_cases"][RNG_K.index(k)]
        must(row1["samples_hash"]==row2["samples_hash"] and
             row1["next_numba_rng"]==row2["next_numba_rng"],
             "cross-OS original MT19937 random/uniform/randint sequence drift")
    for i in range(10):
        left=docs[0]["uniform_edge_cases"][i]
        right=docs[1]["uniform_edge_cases"][i]
        must(left==right,
             "cross-host Numba float64 uniform bit-pattern/RNG mismatch in case "+str(i))
    for k in {(d,s) for d in DATASETS for s in SEEDS}:
        a=verified[0][k]
        b=verified[1][k]
        for field in ("fixture_sha","labels_sha","graph_sha","input_sha"):
            must(a[field]==b[field],
                 "cross-host original graph/label/corpus inconsistency "+field)
        for mode in MODES:
            rows_a=[p for p in a["samples"] if p["mode"]==mode]
            rows_b=[p for p in b["samples"] if p["mode"]==mode]
            exact(rows_a[0]["original"]["final"],
                  rows_b[0]["original"]["final"],
                  (k,"cross-OS",mode))
    report_text,decision,rows=report(docs,verified)
    Path("numba_public_uniform_report.md").write_text(report_text,encoding="utf-8")
    Path("numba_public_uniform_decision.json").write_text(
        json.dumps(dict(schema=1,completeness="PASS",decision=decision,
                        native_cpu_batch_pairs=192,
                        native_cpu_fits=3072,
                        separately_traced_peak_pairs=64,
                        native_jit_warmups=64,
                        edge_policy_pairs=128,
                        no_explicit_seed_pairs=32,
                        numba_rng_stream_k_values=len(RNG_K),
                        float64_uniform_edge_cases=10,
                        rows=rows),indent=2,allow_nan=False)+"\n",
        encoding="utf-8")
    print(report_text,flush=True)
    print("STRICT COMPLETE original-vs-public-NUMBA uniform expression"
          " on both OS: "+decision,flush=True)


if __name__=="__main__":
    main(sys.argv[1])
