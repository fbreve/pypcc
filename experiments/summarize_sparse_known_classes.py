"""Strict both-OS original Git pcc.py class-unique source and CPU study."""
import json
import math
from pathlib import Path
import statistics
import sys

BASE="b4c100bd22eeff1979a1a581bc73682b8c3705b8"
MICRO=((1023,2,"sparse"),(1024,2,"sparse"),(1797,10,"sparse"),
       (30000,10,"sparse"),(30000,50,"sparse"),
       (100000,10,"sparse"),(100000,50,"sparse"),
       (300000,10,"sparse"),(300000,50,"sparse"),
       (100000,10,"dense"),(100000,10,"unlabeled"),
       (100000,10,"single"))
DATA=("iris","wine","breast_cancer","digits")
SEEDS=(78013,78117)
BACKENDS=("numpy","numba","cython")
STATES={"predictions","dominance","positions","strengths","distances","owndeg"}
INFO={"effective_iterations","stop_reason","checks","stalled_checks","stop_max"}
THREADS=("OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","MKL_NUM_THREADS",
         "NUMBA_NUM_THREADS")
COUNTS={"numpy":4,"numba":32,"cython":64}

def must(ok,message):
    if not ok:raise AssertionError(message)

def sha(value):
    return isinstance(value,str) and len(value)==64 and all(
        x in "0123456789abcdef" for x in value)

def positive(value):
    return type(value) in (int,float) and math.isfinite(value) and value>0

def check_result(x,backend,classes,cap,mode):
    must(x["native_backend"]==backend and x["cap"]==cap and
         x["monitored"] is mode,"incorrect real native PCC mode/backend")
    must(set(x["state_hashes"])==STATES and all(
        sha(v) for v in x["state_hashes"].values()) and
        sha(x["mapped_labels_sha"]) and
        sha(x["sorted_known_classes_sha"]),"full native six-state or classes not present")
    expected="C" if backend=="numba" and classes>=8 else "F"
    must(x["native_dominance_layout"]==expected,
         "not genuine NumPy/Numba/Cython dominance C/F layout")
    must(type(x["n_particles"]) is int and x["n_particles"]>0,
         "not a true labeled original PCC fit")
    info=x["info"]
    must(set(info)==INFO and info["stop_reason"] in ("max_iter","early_stop")
         and type(info["effective_iterations"]) is int
         and 0<=info["effective_iterations"]<=cap
         and all(type(info[k]) is int for k in
                 ("checks","stalled_checks","stop_max")),
         "invalid original native stop/check counters")
    if cap==0 or not mode:
        must(info["effective_iterations"]==cap and
             info["stop_reason"]=="max_iter",
             "original fixed or zero-step early stopping changed")
    else:
        must(info["effective_iterations"]>0,
             "real monitored original propagation ran no iterations")
    if backend in ("numpy","numba"):
        must(type(x["numba_or_numpy_next_rng"]) is float
             and 0<=x["numba_or_numpy_next_rng"]<1,
             "missing true native next RNG sample")
    else:
        must(x["numba_or_numpy_next_rng"] is None,
             "Cython native xorshift incorrectly treated as NumPy RNG")

def check_batch(b,backend,classes,cap,mode,count):
    must(b["count"]==count and positive(b["wall_ns"])
         and positive(b["cpu_ns"]) and
         b["cpu_per_fit_ns"]==b["cpu_ns"]/count and
         b["wall_per_fit_ns"]==b["wall_ns"]/count,
         "invalid original 4/32/64/128 Windows-resolved full CPU batch")
    check_result(b["last"],backend,classes,cap,mode)
    must(len(b["all_predictions_sha"])==count
         and all(x==b["last"]["state_hashes"]["predictions"]
                 for x in b["all_predictions_sha"]),
         "some repeated complete original PCC predictions differ")

def check_pair(p,backend,classes,cap,mode,count):
    for variant in ("original","candidate"):
        check_batch(p[variant],backend,classes,cap,mode,count)
    must(p["original"]["last"]==p["candidate"]["last"] and
         p["original"]["all_predictions_sha"]==
         p["candidate"]["all_predictions_sha"],
         "original/candidate native six arrays, RNG, label map or stop differ")

def verify(d):
    must(d["schema"]==1 and d["base"]==BASE
         and d["platform"] in ("linux","win32") and len(d["head"])==40
         and sha(d["source_sha256"]) and sha(d["candidate_sha256"])
         and d["source_sha256"]!=d["candidate_sha256"]
         and all(d["threads"][k]=="1" for k in THREADS),
         "unfrozen original PCC source, runner or CPU threads")
    micro={}
    must(len(d["micro_cases"])==12,"not all twelve original class-density micro cases")
    for row in d["micro_cases"]:
        key=(row["n"],row["classes"],row["kind"])
        must(key in MICRO and key not in micro and
             sha(row["input_sha"]) and sha(row["outputs_sha"]),
             "duplicate or nonfrozen labeled-class microphase")
        reps=(16384 if row["n"]<=1797 else 4096 if row["n"]==30000 else 2048 if row["n"]==100000 else 1024)
        must(row["repetitions"]==reps and
             len(row["timings"])==25 and
             len(row["peak_pairs"])==5,
             "original separate full CPU and tracemalloc source-pair samples missing")
        for rep,sample in enumerate(row["timings"]):
            must(sample["rep"]==rep and sample["order"]==(
                ["candidate","original"] if rep%2==0 else
                ["original","candidate"]),
                "nonalternating original/candidate micro CPU order")
            for variant in ("original","candidate"):
                x=sample[variant]
                must(positive(x["total_cpu_ns"]) and
                     x["per_class_discovery_cpu_ns"]==
                     x["total_cpu_ns"]/reps and
                     x["classes_sha"]==row["outputs_sha"] and
                     type(x["count"]) is int,
                     "incorrect exact original/candidate np.unique phase")
        for rep,sample in enumerate(row["peak_pairs"]):
            must(sample["rep"]==rep and
                 all(type(sample[k]) is int and sample[k]>=0
                     for k in ("original","candidate")) and
                 sample["candidate"]-sample["original"]<=
                 16*row["n"]+1048576,
                 "traced added RAM exceeds prespecified CPU-for-memory budget")
        micro[key]=row
    must(set(micro)==set(MICRO),"original boundary/dense/sparse cases absent")
    real={}
    must(len(d["real_pairs"])==96,
         "missing 96 real three-backend native source CPU pairs for this OS")
    for row in d["real_pairs"]:
        key=(row["dataset"],row["seed"],row["backend"],
             row["monitored"],row["rep"])
        must(row["kind"]=="real" and
             row["dataset"] in DATA and row["seed"] in SEEDS and
             row["backend"] in BACKENDS and
             row["monitored"] in (True,False) and
             row["rep"] in (0,1) and row["cap"]==1000 and
             key not in real and
             sha(row["fixture_sha"]) and sha(row["labels_sha"]) and
             len(row["graph_sha"])==2 and
             all(sha(x) for x in row["graph_sha"]),
             "not original real sklearn undirected kNN10 PCC input")
        must(row["order"]==(
            ["candidate","original"] if row["rep"]%2==0
            else ["original","candidate"]),
            "unbalanced original/candidate full real CPU sample")
        check_pair(row,row["backend"],row["classes"],1000,
                   row["monitored"],COUNTS[row["backend"]])
        real[key]=row
    must(set(real)=={
        (ds,seed,backend,mode,i) for ds in DATA for seed in SEEDS
        for backend in BACKENDS for mode in (False,True) for i in (0,1)},
         "missing independent real native masked full-fit evidence")
    ring={}
    must(len(d["ring_pairs"])==48,
         "missing 48 original ring 0/5-iteration full initialization cohorts")
    for row in d["ring_pairs"]:
        key=(row["nodes"],row["classes"],row["backend"],
             row["cap"],row["rep"])
        must(row["kind"]=="ring" and row["nodes"] in (30000,100000)
             and row["classes"] in (2,10,50)
             and row["backend"] in ("numpy","numba")
             and row["cap"] in (0,5) and row["monitored"] is False
             and row["rep"] in (0,1) and key not in ring
             and len(row["graph_sha"])==2
             and all(sha(x) for x in row["graph_sha"])
             and sha(row["labels_sha"]),
             "wrong original ordered dense int64 synthetic ring")
        must(row["order"]==(
            ["candidate","original"] if row["rep"]%2==0
            else ["original","candidate"]),
            "ring CPU original/candidate source order not alternated")
        check_pair(row,row["backend"],row["classes"],
                   row["cap"],False,32)
        ring[key]=row
    must(set(ring)=={
        (n,c,b,cap,i) for n in (30000,100000) for c in (2,10,50)
        for b in ("numpy","numba") for cap in (0,5) for i in (0,1)},
         "missing at least one original native ring model CPU source pair")
    dense={}
    must(len(d["dense_controls"])==2,
         "full 50%-labeled dense original Numba controls missing")
    for row in d["dense_controls"]:
        n=row["n"]
        must(n in (1024,1797) and n not in dense and
             row["kind"]=="dense" and row["count"]==128 and
             sha(row["labels_sha"]) and len(row["graph_sha"])==2
             and all(sha(x) for x in row["graph_sha"]),
             "not original densely labeled safe small full fit")
        check_pair(row,"numba",2,0,False,128)
        dense[n]=row
    must(set(dense)=={1024,1797},"missing dense original control")
    no={}
    must(len(d["no_class_controls"])==2,
         "zero/single-class no-propagation semantic controls absent")
    for row in d["no_class_controls"]:
        kind=row["kind"]
        must(kind in ("unlabeled","single") and kind not in no and
             row["n"]==100000 and sha(row["labels_sha"])
             and len(row["both"])==2
             and {x["variant"] for x in row["both"]}==
             {"original","candidate"},
             "wrong original single-class/no-class input")
        for trial in row["both"]:
            must(trial["returned_same_input"] is True
                 and trial["node_none"] is True
                 and trial["particle_none"] is True
                 and "requires at least 2 labeled classes" in trial["warning"],
                 "original returned object identity or public warning changed")
        must(row["both"][0]["warning"]==row["both"][1]["warning"],
             "candidate changed public no-fit warning")
        no[kind]=row
    return dict(micro=micro,real=real,ring=ring,dense=dense,no=no)

def median_saved(trials):
    return 100*(1-statistics.median(
        t["candidate"]["cpu_ns"]/t["original"]["cpu_ns"]
        for t in trials))

def evaluate(docs,data):
    out=[
        "# Exact production PCC sparse class discovery original/candidate CPU",
        "",
        "Original full Git pcc.py, exactly one class discovery source hunk.",
        "Twelve density/size controls, source-compiled 600 true batched",
        "class timings and 120 separate Python tracked-peak comparisons.",
        "Real original NumPy/Numba/Cython source-paired full fit, exact",
        "native state and next RNG, independent seeded masks, zero/5-step",
        "30k/100k ring models with identical padded dense neighbor arrays.",
        "",
        "| OS | Nodes | Classes | Known-label density | Old micro CPU ms | Candidate micro CPU ms | CPU saved | Max added traced bytes |",
        "|---|---:|---:|---|---:|---:|---:|---:|",
    ]
    micro=[]
    go=True
    for d,v in zip(docs,data):
        for key in MICRO:
            row=v["micro"][key]
            times=row["timings"]
            saved=100*(1-statistics.median(
                t["candidate"]["total_cpu_ns"]/t["original"]["total_cpu_ns"]
                for t in times))
            original=statistics.median(
                t["original"]["per_class_discovery_cpu_ns"]
                for t in times)/1e6
            candidate=statistics.median(
                t["candidate"]["per_class_discovery_cpu_ns"]
                for t in times)/1e6
            extra=max(x["candidate"]-x["original"]
                      for x in row["peak_pairs"])
            if key[0]>=30000 and key[2]=="sparse" and saved<40:go=False
            if key[2]=="dense" and saved< -50:go=False
            micro.append(dict(os=d["platform"],nodes=key[0],
                              classes=key[1],density=key[2],
                              original_ms=original,candidate_ms=candidate,
                              cpu_saved_pct=saved,peak_extra_bytes=extra))
            out.append(
                f"| {d['platform']} | {key[0]} | {key[1]} | {key[2]} | "
                f"{original:.4f} | {candidate:.4f} | {saved:+.2f}% | {extra:+d} |")
    out+=["","## Real full original PCC after isolated JIT compilation",
          "",
          "| OS | Dataset | Backend | Stop | Full fit CPU saved |",
          "|---|---|---|---|---:|"]
    real=[]
    for d,v in zip(docs,data):
        for ds in DATA:
            for backend in BACKENDS:
                for mode in (False,True):
                    trials=[v["real"][(ds,seed,backend,mode,i)]
                            for seed in SEEDS for i in (0,1)]
                    saved=median_saved(trials)
                    if saved < -5:go=False
                    real.append(dict(os=d["platform"],dataset=ds,
                                     backend=backend,monitored=mode,
                                     cpu_saved_pct=saved,
                                     original_ms=statistics.median(
                                         t["original"]["cpu_per_fit_ns"]
                                         for t in trials)/1e6,
                                     candidate_ms=statistics.median(
                                         t["candidate"]["cpu_per_fit_ns"]
                                         for t in trials)/1e6))
                    out.append(
                        f"| {d['platform']} | {ds} | {backend} | "
                        f"{'monitored' if mode else 'fixed'} | {saved:+.2f}% |")
    out+=["","## Real original 30k/100k-node ring initialized full PCC",
          "","| OS | Nodes | Classes | Backend | Iterations | Complete CPU saved |",
          "|---|---:|---:|---|---:|---:|"]
    rings=[]
    for d,v in zip(docs,data):
        for n in (30000,100000):
            for c in (2,10,50):
                for backend in ("numpy","numba"):
                    for cap in (0,5):
                        trials=[v["ring"][(n,c,backend,cap,i)]
                                for i in (0,1)]
                        saved=median_saved(trials)
                        if saved< -5:go=False
                        if n==100000 and c==10 and saved<3:go=False
                        rings.append(dict(os=d["platform"],nodes=n,classes=c,
                                          backend=backend,iterations=cap,
                                          cpu_saved_pct=saved,
                                          original_fit_ms=statistics.median(
                                              x["original"]["cpu_per_fit_ns"]
                                              for x in trials)/1e6,
                                          candidate_fit_ms=statistics.median(
                                              x["candidate"]["cpu_per_fit_ns"]
                                              for x in trials)/1e6))
                        out.append(
                            f"| {d['platform']} | {n} | {c} | "
                            f"{backend} | {cap} | {saved:+.2f}% |")
        for control in v["dense"].values():
            if control["candidate"]["cpu_ns"]>1.05*control["original"]["cpu_ns"]:
                go=False
    verdict="GO" if go else "NO-GO"
    out +=[
        "",
        f"**Frozen scientific performance verdict: {verdict}.**",
        "All original source, exact native states, RNG, stopping,",
        "class labels, output object identity and CPU/memory observations",
        "passed the independent completeness and source guards.",
        "Frozen CPU GO: >=40% per micro sparse 30k/100k/300k both OS;",
        ">=3% full fit CPU at 100k×10 0 and 5 iterations for each",
        "NumPy and Numba both OS; no full backend/mode worse by >5%;",
        "no dense isolated source phase worse by >50%; bounded Python",
        "tracked transient memory <=16*n+1MiB; exact states mandatory.",
        "",
        "Micro speedups are NOT whole fit speedups. New random generator",
        "code, graph layout, stop criteria and numeric precision were",
        "NOT changed. True CPU process time on Windows is quantized;",
        "CPU samples were batched before any data according to protocol.",
        "Scientific branch remains permanently DRAFT/UNMERGED.",
    ]
    return "\n".join(out)+"\n",verdict,micro,real,rings

def main(folder):
    paths=list(Path(folder).glob("**/sparse_known_classes_results.json"))
    must(len(paths)==2,"two independent complete native OS artifacts mandatory")
    docs=[json.loads(f.read_text(encoding="utf-8")) for f in paths]
    must({d["platform"] for d in docs}=={"linux","win32"},
         "expected one original Linux and one original Windows dataset")
    docs.sort(key=lambda d:d["platform"])
    validated=[verify(d) for d in docs]
    for key in ("base","head","source_sha256","candidate_sha256"):
        must(len({d[key] for d in docs})==1,
             "different original Git source/hash between native OS: "+key)
    for key in MICRO:
        a,b=(v["micro"][key] for v in validated)
        must(a["input_sha"]==b["input_sha"] and
             a["outputs_sha"]==b["outputs_sha"],
             "native OS class discovery source result not equal")
    for key in validated[0]["real"]:
        a,b=(v["real"][key] for v in validated)
        must(a["fixture_sha"]==b["fixture_sha"]
             and a["labels_sha"]==b["labels_sha"]
             and a["graph_sha"]==b["graph_sha"]
             and a["original"]["last"]==b["original"]["last"],
             "cross-OS original sklearn native graph/state/RNG mismatch")
    for key in validated[0]["ring"]:
        a,b=(v["ring"][key] for v in validated)
        must(a["labels_sha"]==b["labels_sha"]
             and a["graph_sha"]==b["graph_sha"]
             and a["original"]["last"]==b["original"]["last"],
             "cross-OS original seeded ring/native state not equal")
    report,decision,micro,real,ring=evaluate(docs,validated)
    Path("sparse_known_classes_report.md").write_text(report,encoding="utf-8")
    Path("sparse_known_classes_decision.json").write_text(
        json.dumps(dict(schema=1,complete=True,decision=decision,
                        total_micro_cpu_pairs=600,
                        total_micro_traced_peak_pairs=120,
                        total_real_native_full_fit_pairs=192,
                        total_ring_full_fit_pairs=96,
                        total_dense_fit_controls=4,
                        total_no_class_identity_cases=4,
                        micro=micro,real=real,ring=ring),
                   indent=2,allow_nan=False)+"\n",encoding="utf-8")
    print(report,flush=True)
    print("STRICT SOURCE-EXACT TWO-OS COMPLETE: 600+120 isolated class "
          "phase CPU+memory pairs; 192 actual all-backend real full PCC "
          "native pairs; 96 original 30k/100k ring pairs; 4 dense "
          "model controls; 4 no-class identity cases. Decision:",
          decision,flush=True)

if __name__=="__main__":
    main(sys.argv[1])
