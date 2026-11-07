"""Strict complete 2-OS current-master performance baseline validation and report.

Pure stdlib offline review of the repo's existing benchmarks.run artifacts.
Scientific choices were preregistered before any timing results were collected.
"""
import argparse
import json
from pathlib import Path
import statistics
from collections import Counter,defaultdict
from itertools import permutations

DATASETS=("wine","breast_cancer","digits")
SEEDS=(25013,25117)
BACKENDS=("numpy","numba","cython")
HOSTS=("linux","windows")
PHASES=("initialization_and_dispatch_s","propagation_s",
        "postprocessing_s","fit_predict_s")
EXPECTED_STATES={
    "predictions","dominance","positions","strengths","distances","owndeg"}
NCLASSES={"wine":3,"breast_cancer":2,"digits":10}
EXPECTED_KEYS={
    (ds,seed,backend,rep) for ds in DATASETS for seed in SEEDS
    for backend in BACKENDS for rep in range(6)}
BASELINE_MASTER="8298ece05c4edf4bbdc871d0aa5b13d9b8ed4128"


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def lines(path):
    return [
        json.loads(line) for line in
        Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()]


def is_hash(value):
    return isinstance(value,str) and len(value)==64 and all(
        c in "0123456789abcdef" for c in value.lower())


def ensure(ok,message):
    if not ok:
        raise AssertionError(message)


def verify_host(root):
    root=Path(root)
    host=root.name.removeprefix("baseline-")
    ensure(host in HOSTS,f"Unexpected host: {host}")
    manifest=read_json(root/"records/manifest.json")
    prep=read_json(root/"records/preparation.json")
    measured=lines(root/"records/measurements.jsonl")
    warmup=lines(root/"records/warmups.jsonl")
    frozen=read_json(root/"dry-run.json")
    complete=read_json(root/"records/completion.json")
    memory=read_json(root/"memory.json")
    invocation=read_json(root/"invocation.json")
    native_summary=read_json(root/"records/summary.json")
    freeze=(root/"pip-freeze.txt").read_text(encoding="utf-8")
    log=(root/"execution.log").read_text(
        encoding="utf-8",errors="replace")
    ensure(len(log)>0 and len(freeze)>0,f"Empty real run logs or freeze: {host}")
    c=manifest.get("configuration",{})
    w=c.get("workload",{})
    expected_case_ids=[
        f"{d}-labels{s}" for d in DATASETS for s in SEEDS]
    for plan in (frozen,w):
        ensure(
            plan.get("case_count")==6 and
            plan.get("preparation_groups")==3 and
            plan.get("measured_executions")==108 and
            plan.get("warmup_executions")==36 and
            plan.get("measured_iteration_cap")==1080000 and
            plan.get("particle_update_cap")==16200000 and
            plan.get("backends")==list(BACKENDS) and
            plan.get("order_policy")=="permutations" and
            plan.get("order_seed")==1901 and
            plan.get("mode")=="fixed" and
            plan.get("repetitions")==6 and
            plan.get("requires_expensive_opt_in") is False and
            [z.get("case_id") for z in plan.get("cases",[])]==
                expected_case_ids,
            f"Frozen work or six-plan mismatch: {host}")
    ensure(
        c.get("mode")=="fixed" and
        c.get("propagation_seed")==0 and
        c.get("order_policy")=="permutations" and
        c.get("order_seed")==1901 and
        c.get("parameters")==dict(p_grd=.5,delta_v=.1,deltap=1.,
                                  dexp=2.,es_chk=2000) and
        c.get("session_id")==f"master-native-{host}",
        f"Runtime configuration changed: {host}")
    r=manifest.get("repository",{})
    ensure(is_hash(r.get("commit")),
           f"Missing current benchmark branch commit: {host}")
    # The current branch adds only benchmark/CI/docs, so this commit is
    # expected to differ from the pinned production ancestor.
    ensure(r.get("commit")!=BASELINE_MASTER,
           f"Not executing the benchmark branch under review: {host}")
    source=r.get("source_hashes",{})
    ensure(all(is_hash(source.get(p))
               for p in ("pcc.py","pcc_numba.py","pcc_numpy.py",
                         "pcc_graph.py","pcc_step.pyx",
                         "benchmarks/run.py","benchmarks/runner.py")),
           f"Production source hash inventory incomplete: {host}")
    ext=manifest.get("cython_extension",{})
    ensure(is_hash(ext.get("sha256")) and
           str(ext.get("path","")).lower().endswith((".pyd",".so")),
           f"Real native Cython extension not identified: {host}")
    versions=manifest.get("environment",{}).get("versions",{})
    ensure(all(isinstance(versions.get(pkg),str)
               and versions[pkg] for pkg in
               ("numpy","scipy","scikit-learn","numba","Cython","pcc_step")),
           f"Missing native dependency identities: {host}")
    ensure(
        complete.get("status")=="complete" and
        complete.get("measured_executions")==108 and
        len(complete.get("fixtures",[]))==6 and
        not complete.get("skipped") and
        len(measured)==108 and len(warmup)==36 and
        len(native_summary)==18,
        f"Truncated benchmark artifacts: {host}")
    ensure(
        prep.get("graph_builds")==3 and
        len(prep.get("fixtures",[]))==6,
        f"Graph was not prepared once per dataset: {host}")
    entries=prep["fixtures"]
    ensure(
        [x["case_id"] for x in entries]==expected_case_ids and
        [bool(x["reused"]) for x in entries]==
            [False,True,False,True,False,True] and
        sum(not x["reused"] for x in entries)==3,
        f"Incorrect graph-cache reuse: {host}")
    ensure(
        len({x["case_id"] for x in complete["fixtures"]})==6 and
        all(is_hash(x.get("fixture_sha256"))
            for x in complete["fixtures"]),
        f"Broken benchmark source fixture hashes: {host}")
    fixtures={
        x["case_id"]:x["fixture_sha256"]
        for x in complete["fixtures"]}
    ensure(
        len(fixtures)==6 and
        set(fixtures)==set(expected_case_ids),
        f"Incorrect expected six fixtures: {host}")
    ensure(
        memory.get("host_label")==host and
        memory.get("exit_code")==0 and
        memory.get("n_samples",0)>0 and
        memory.get("sampled_peak_rss_bytes",0)>0 and
        memory.get("sampled_peak_rss_bytes",0)>=
            memory.get("sampled_first_rss_bytes",0) and
        memory.get("sample_interval_seconds")==.05 and
        memory.get("subprocess_wall_seconds",0)>0 and
        invocation.get("host_label")==host and
        invocation.get("baseline_master_sha")==BASELINE_MASTER,
        f"Missing external process memory provenance: {host}")
    expected_native={
        "numpy":"pcc_numpy.pcc_propagate_numpy",
        "numba":"pcc_numba.pcc_propagate_numba",
        "cython":"pcc_step.pcc_propagate",
    }
    group=defaultdict(list)
    bycase=defaultdict(list)
    state_ref={}
    accuracy_ref={}
    keys=set()
    for row in measured:
        cid=row.get("case_id")
        ds,sep,seed_str=cid.rpartition("-labels")
        ensure(bool(sep) and ds in DATASETS and
               seed_str.isdigit() and
               int(seed_str) in SEEDS,
               f"Unplanned fixture: {host}/{cid}")
        seed=int(seed_str)
        b=row.get("backend")
        repeat=row.get("repeat")
        key=(ds,seed,b,repeat)
        ensure(
            b in BACKENDS and isinstance(repeat,int)
            and 0<=repeat<6 and key not in keys,
            f"Duplicate, missing or invalid measurement: {host}/{key}")
        keys.add(key)
        ensure(
            row.get("fixture_sha256")==fixtures[cid] and
            row.get("labeled_nodes")==3*NCLASSES[ds] and
            row.get("backend_identity")==expected_native[b] and
            row.get("max_iter")==10000 and
            row.get("effective_iterations")==10000 and
            row.get("mode")=="fixed" and
            row.get("stop_reason")=="max_iter" and
            row.get("stopped_before_cap") is False and
            row.get("reached_max_iter") is True and
            row.get("propagation_seed_requested")==0 and
            row.get("propagation_seed_effective")==
                (123456789 if b=="cython" else 0) and
            0<=row.get("accuracy_unlabeled",-1)<=1 and
            row.get("order_index") in (0,1,2) and
            row.get("session_id")==f"master-native-{host}",
            f"Wrong backend, mode, timing or identity: {host}/{key}")
        ensure(
            all(isinstance(row.get(phase), (float,int)) and
                0<row[phase]<600 for phase in PHASES) and
            abs(sum(row[phase] for phase in PHASES[:-1])-
                row["fit_predict_s"])<1e-6 and
            abs(row["propagation_s_per_iteration"]-
                row["propagation_s"]/10000)<1e-10,
            f"Invalid phase accounting: {host}/{key}")
        hashes=row.get("state_hashes",{})
        ensure(
            set(hashes)==EXPECTED_STATES and
            all(is_hash(v) for v in hashes.values()),
            f"Missing complete native final-state hashes: {host}/{key}")
        identity=(ds,seed,b)
        if identity not in state_ref:
            state_ref[identity]=hashes
            accuracy_ref[identity]=row["accuracy_unlabeled"]
        else:
            ensure(
                hashes==state_ref[identity] and
                row["accuracy_unlabeled"]==accuracy_ref[identity],
                f"Nonrepeatable same-backend state/accuracy: {host}/{identity}")
        group[identity].append(row)
        bycase[(ds,seed,repeat)].append(row)
    ensure(keys==EXPECTED_KEYS,f"Incomplete cases/repetitions: {host}")
    full_perms=set(permutations(BACKENDS))
    for ds in DATASETS:
        for seed in SEEDS:
            shown=[]
            for rep in range(6):
                rr=bycase[ds,seed,rep]
                ensure(len(rr)==3,
                       f"Missing full three-backend repetition: {host}/{ds}/{seed}/{rep}")
                order=tuple(rr[0]["execution_order"])
                ensure(
                    order in full_perms and
                    all(tuple(x["execution_order"])==order for x in rr) and
                    [(x["order_index"],x["backend"])
                     for x in rr]==list(enumerate(order)),
                    f"Execution order is not the announced permutation: {host}/{ds}/{seed}/{rep}")
                shown.append(order)
            ensure(
                set(shown)==full_perms,
                f"Unbalanced 6-permutation block: {host}/{ds}/{seed}")
    wu=Counter()
    for row in warmup:
        k=(row.get("case_id"),row.get("backend"),row.get("call"))
        ensure(row.get("call") in ("first","warm") and
               row.get("case_id") in fixtures and
               row.get("backend") in BACKENDS and
               row.get("fixture_sha256")==fixtures[row["case_id"]] and
               row.get("effective_iterations")==2 and
               row.get("mode")=="fixed",
               f"Malformed warmup metadata: {host}/{k}")
        wu[k]+=1
    ensure(set(wu)=={(cid,b,stage) for cid in expected_case_ids
                    for b in BACKENDS for stage in ("first","warm")}
           and all(v==1 for v in wu.values()),
           f"Missing/excess native backend warmup: {host}")
    for summary in native_summary:
        cid=summary.get("case_id")
        b=summary.get("backend")
        ensure(
            cid in fixtures and b in BACKENDS and
            summary.get("fixture_sha256")==fixtures[cid] and
            summary.get("repetitions")==6 and
            summary.get("mode")=="fixed" and
            summary.get("max_iter")==10000 and
            summary.get("stop_reasons",{}).get("max_iter")==6 and
            summary.get("cap_fraction")==1.,
            f"Incorrect existing harness per-fixture summary: {host}")
    return dict(host=host,config=c,manifest=manifest,
                measured=measured,group=group,
                environment=manifest["environment"],memory=memory,
                prep=prep,fixtures=fixtures,source=source)


def describe(verified):
    lines=[
        "# Unmodified production PCC: first native cross-OS current-master baseline",
        "",
        "**Every frozen condition passed:** two host processes (Linux and Windows);",
        "each 6 dataset×seed fixtures × 3 original sequential backends × 6",
        "balanced order repetitions = **108 timed 10k-iteration full fits",
        "per OS / 216 measured total**, plus 36 original two-step warmups",
        "per OS. No numerical or kernel source modified. Values are hosted",
        "CI observations, not local hardware or cross-host controlled runs.",
        "",
        "## Provenance and sampled process RSS",
        "",
        "| Host | CPU | Python | Cython extension SHA256 | Total wall (s) | 50-ms-sampled max RSS (MiB) |",
        "|---|---|---|---|---:|---:|",
    ]
    for v in verified:
        env=v["environment"]
        mem=v["memory"]
        lines.append(
            f"| {v['host']} | {env.get('cpu_model','unknown')} | "
            f"{env.get('python','?').split()[0]} | "
            f"`{v['manifest']['cython_extension']['sha256']}` | "
            f"{mem['subprocess_wall_seconds']:.2f} | "
            f"{mem['sampled_peak_rss_bytes']/1048576:.1f} |")
    lines.extend(["",
        "RSS is from an external 50-ms whole-subprocess memory sample,",
        "including data loading, graph preparation, JIT warmup and all fits;",
        "it is **not a per-fit or precise allocation peak**. Host hardware,",
        "threadpool and OS differ: DO NOT compare Linux versus Windows speed",
        "as if hardware was controlled.",
        "",
        "## Propagation and complete fit medians (ms)",
        "",
        "| Host | Dataset | Backend | Propagation median (ms) | Q1–Q3 (ms) | Full fit median (ms) | Accuracy ranges across 2 masks |",
        "|---|---|---|---:|---|---:|---|",
    ])
    def quartiles(values):
        s=sorted(values)
        # Existing benchmark uses NumPy percentile(method='linear');
        # with 12 observations, index is 2.75 and 8.25 respectively.
        def pct(p):
            q=(len(s)-1)*p
            i=int(q)
            fract=q-i
            return s[i]*(1-fract)+(s[min(i+1,len(s)-1)]*fract)
        return pct(.25),pct(.75)
    for v in verified:
        for ds in DATASETS:
            for b in BACKENDS:
                rows=[r for r in v["measured"]
                      if r["case_id"].startswith(ds+"-labels")
                      and r["backend"]==b]
                ensure(len(rows)==12,"Unexpected benchmark group size")
                vals=[1000*r["propagation_s"] for r in rows]
                total=[1000*r["fit_predict_s"] for r in rows]
                a,bq=quartiles(vals)
                acc=sorted(set(round(100*r["accuracy_unlabeled"],4)
                               for r in rows))
                lines.append(
                    f"| {v['host']} | {ds} | {b} | "
                    f"{statistics.median(vals):.3f} | "
                    f"{a:.3f}–{bq:.3f} | "
                    f"{statistics.median(total):.3f} | "
                    f"{min(acc):.2f}–{max(acc):.2f}% |")
    lines.extend(["",
        "## Matched native propagation comparisons: Numba ÷ Cython",
        "",
        "| Host | Dataset | Median paired Numba/Cython ratio | Range across 12 matched fixture/repetition pairs |",
        "|---|---|---:|---|",
    ])
    for v in verified:
        for ds in DATASETS:
            ratios=[]
            for seed in SEEDS:
                for rep in range(6):
                    pair={x["backend"]:x["propagation_s"]
                          for x in v["measured"]
                          if x["case_id"]==f"{ds}-labels{seed}"
                          and x["repeat"]==rep}
                    ensure(set(pair)==set(BACKENDS),
                           "Cannot pair all actual backend permutations")
                    ratios.append(pair["numba"]/pair["cython"])
            lines.append(
                f"| {v['host']} | {ds} | "
                f"{statistics.median(ratios):.3f}× | "
                f"{min(ratios):.3f}–{max(ratios):.3f}× |")
    lines.extend(["",
        "## Preparation and instrumentation caveats",
        "",
        "| Host | Dataset | Original one-time loading+standardization+label selection (ms) | One-time original kNN graph build (ms) |",
        "|---|---|---:|---:|",
    ])
    for v in verified:
        for ds in DATASETS:
            first=next(r for r in v["prep"]["fixtures"]
                       if r["case_id"]==f"{ds}-labels{SEEDS[0]}")
            lines.append(
                f"| {v['host']} | {ds} | "
                f"{1000*(first['dataset_load_s']+first['preprocessing_and_labels_s']):.3f} | "
                f"{1000*first['graph_build_s']:.3f} |")
    lines.extend(["",
        "- All 6 repeated original state-array SHA digests (per fixture/backend),",
        "  native 10k iteration counts, stop reasons and unlabeled accuracy",
        "  matched exactly. NumPy, Numba, Cython **do not share RNG streams**",
        "  and are not required to produce identical cross-backend predictions.",
        "- Graph construction is measured **once per dataset** and intentionally",
        "  excluded from the full fit timer; total end-to-end with a new graph",
        "  must account for preparation, not reuse this timer as a direct",
        "  graph-inclusive measurement.",
        "- Cython's row-max stopping cache is already integrated into master",
        "  via #178/#179; fixed no-stop 10k timing **does not measure it**.",
        "  A separate frozen early-stop pilot is needed if monitoring dominates.",
        "- The six technical repetitions and two label masks per dataset",
        "  do not give independent samples of machine performance; hosted",
        "  runners and compilation/build chains vary. No backend-universal",
        "  winner or production optimization is established by this pilot.",
        "- Only archived measured results may support a future targeted",
        "  **new independent candidate-vs-this-master build**; this baseline",
        "  did not modify production or propose a default change.",
    ])
    return "\n".join(lines)+"\n"


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--roots",nargs=2,required=True)
    p.add_argument("--output",required=True)
    a=p.parse_args()
    verified=[verify_host(v) for v in a.roots]
    ensure([v["host"] for v in verified]==list(HOSTS),
           "Expect independent linux, windows artifacts in that order")
    ensure(len({v["manifest"]["repository"]["commit"] for v in verified})==1,
           "Both OS jobs did not execute exactly the same Git revision")
    ensure(len({json.dumps(v["source"],sort_keys=True) for v in verified})==1,
           "Cython/PCC source hashes differ across hosts")
    for ds in DATASETS:
        for seed in SEEDS:
            cid=f"{ds}-labels{seed}"
            ensure(len({v["fixtures"][cid] for v in verified})==1,
                   f"Frozen graph/label fixture differs across OS: {cid}")
    s=describe(verified)
    Path(a.output).write_text(s,encoding="utf-8")
    print(s,flush=True)


if __name__=="__main__":
    main()
