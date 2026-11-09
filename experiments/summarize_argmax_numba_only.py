"""Strict both-OS held-out Numba-only argmax whole-fit scientific decision."""
import json
from pathlib import Path
import statistics
import sys

BASE = "bcf0fed83e64a753bf8ab01dba5124749931ebc2"
REAL = ("wine", "breast_cancer", "digits")
CASES = (*REAL, "ring_40000_20", "ring_80000_50", "ring_120000_50")
BACKENDS = ("numpy", "numba", "cython")
STATE = ("predictions", "dominance", "positions", "strengths",
         "distances", "owndeg", "node_labels", "mapped_labels")
MEMORY_CASES = (("ring_120000_50", "numba", 0, False),
                ("ring_120000_50", "numba", 5, False),
                ("digits", "numba", 0, False),
                ("digits", "numba", 5, False),
                ("digits", "numba", 1000, True))


def require(test, msg):
    if not test:
        raise AssertionError(msg)


def expected():
    values=set()
    for case in CASES:
        for backend in BACKENDS:
            for steps in (0, 5):
                values.add((case, backend, steps, False))
            if case in REAL and backend != "numpy":
                values.update({(case, backend, 1000, False),
                               (case, backend, 1000, True)})
    require(len(values) == 48, "precommitted 48 cohorts")
    return values


def cohort_key(item):
    return item["case"], item["backend"], item["steps"], item["early_stop"]


def memory_key(item):
    return cohort_key(item)


def med(values):
    return statistics.median(values)


def rows_by_host(d):
    require(d["source_base"] == BASE and d["exact_production_source"], "pinned real original source")
    require(d["pairs"] == 6 and d["micro_functional_checks"] == 200,
            "prospective pairs/independent argmax semantics")
    # Source-derived dispatcher identity must be exercised, not just present.
    require(d.get("dispatch_branch_shapes") == {
        "numba": [40, 10], "cython": [20, 10], "numpy": [20, 10]},
        "actual Numba-only optimization branch never executed")
    require(tuple(d["cases"]) == CASES, "changed corpus")
    parsed={}
    for r in d["cohorts"]:
        key=cohort_key(r)
        require(key not in parsed, f"duplicate cohort {key}")
        parsed[key]=r
        require(len(r["label_shas"]) == 2 and r["label_shas"][0] != r["label_shas"][1],
                f"independent original masks {key}")
        expected_batch=256 if key[0] in REAL else {
            "ring_40000_20":32, "ring_80000_50":16,
            "ring_120000_50":8}[key[0]]
        require(r["batch_fits"] == expected_batch, f"frozen real batch length {key}")
        require(len(r["pairs"]) == 6, f"missing completed time pairs {key}")
        for idx,p in enumerate(r["pairs"]):
            require(p["rep"] == idx, f"wrong paired sequence {key}")
            for timing in ("original_cpu_ns","candidate_cpu_ns",
                           "original_wall_ns","candidate_wall_ns"):
                require(type(p[timing]) is int and p[timing] > 0, f"invalid/quantized zero CPU {key}")
            require(len(p["state_shas"]) == expected_batch and
                    len(p["layout"]) == expected_batch, f"missing real full fits {key}")
            for state in p["state_shas"]:
                require(set(state) == set(STATE) and
                        all(isinstance(h,str) and len(h)==64 for h in state.values()),
                        f"invalid native state sha {key}")
            require(len(p["native_replay"]) == 2, f"native stop/RNG replay missing {key}")
            for replay in p["native_replay"]:
                require(type(replay["stop_info"]) is dict and
                        "effective_iterations" in replay["stop_info"] and
                        "stop_reason" in replay["stop_info"],
                        f"wrong native stop replay {key}")
                require(replay["stop_info"]["effective_iterations"] <= key[2],
                        f"native stop beyond max iter {key}")
    require(set(parsed) == expected(), f"frozen cohort mismatch {expected()^set(parsed)}")
    memory={}
    for x in d["memory"]:
        key=memory_key(x)
        require(key in MEMORY_CASES and key not in memory, "memory unexpected/duplicate")
        require(len(x["samples"]) == 3, f"memory repetitions {key}")
        for idx,row in enumerate(x["samples"]):
            require(row["rep"]==idx and
                    type(row["original_peak_bytes"]) is int and row["original_peak_bytes"] >0 and
                    type(row["candidate_peak_bytes"]) is int and row["candidate_peak_bytes"] >0,
                    "memory peak missing")
        memory[key]=x
    require(set(memory) == set(MEMORY_CASES), "missing full-fit memory controls")
    return parsed,memory


def delta(original,candidate):
    return (original-candidate)*100/original


def scores(rows):
    result={}
    for key,item in rows.items():
        result[key]=dict(cpu=med([delta(p["original_cpu_ns"],p["candidate_cpu_ns"])
                                   for p in item["pairs"]]),
                         wall=med([delta(p["original_wall_ns"],p["candidate_wall_ns"])
                                    for p in item["pairs"]]),
                         original_cpu_ms=med([p["original_cpu_ns"]/item["batch_fits"]/1e6
                                             for p in item["pairs"]]),
                         candidate_cpu_ms=med([p["candidate_cpu_ns"]/item["batch_fits"]/1e6
                                              for p in item["pairs"]]))
    return result


def decision(linux, windows, ml, mw):
    checks=[]
    for osname, data, memory in (("Linux",linux,ml),("Windows",windows,mw)):
        for case in ("ring_80000_50", "ring_120000_50"):
            for iterations in (0, 5):
                key=(case, "numba", iterations, False)
                checks.append(dict(os=osname,group="primary-numba-heldout",
                                   cohort=key,savings_percent=data[key]["cpu"],
                                   required=8.,passed=data[key]["cpu"] >= 8.))
        for iterations in (0, 5):
            key=("digits", "numba", iterations, False)
            checks.append(dict(os=osname,group="real-digits-wall-heldout",
                               cohort=key,savings_percent=data[key]["wall"],
                               required=4.,passed=data[key]["wall"] >= 4.))
        for key, score in data.items():
            if key[0] in REAL:
                checks.append(dict(os=osname,group="no-real-regression",
                                   cohort=key,savings_percent=score["wall"],
                                   required=-8.,passed=score["wall"] >= -8.))
            elif key not in [("ring_80000_50", "numba", i, False)
                             for i in (0,5)] + [
                                 ("ring_120000_50", "numba", i, False)
                                 for i in (0,5)]:
                checks.append(dict(os=osname,group="no-other-synthetic-regression",
                                   cohort=key,savings_percent=score["cpu"],
                                   required=-8.,passed=score["cpu"] >= -8.))
        for key, item in memory.items():
            for sample in item["samples"]:
                o=sample["original_peak_bytes"]
                candidate=sample["candidate_peak_bytes"]
                checks.append(dict(os=osname,group="no-memory-regression",
                                   cohort=key,sample=sample["rep"],
                                   orig_bytes=o,candidate_bytes=candidate,
                                   cap_bytes=1.05*o + 1024*1024,
                                   passed=candidate <= 1.05*o + 1024*1024))
    outcome="GO" if all(x["passed"] for x in checks) else "NO-GO"
    return outcome, checks


def render(hosts, outcome, checks):
    lines=["# Exact original PyPCC Numba-only argmax copy avoidance held-out full-fit experiment",
           "",
           "**SCIENCE COMPLETE**: this is a frozen original-source A/B analysis,"
           " not permission to merge the research branch.",
           "All real source guards, full arrays/output/graph, original native"
           " stop/next-RNG replay, previous-fit ownership, and rollback are"
           " mandatory for each successful host job. The independent strict"
           " aggregator additionally requires complete paired trials and"
           " identical cross-host source and inputs.",
           "",
           f"## Frozen independent performance decision: **{outcome}**",
           "",
           "CPU is per-process and may be quantized on Windows. Per-cohort"
           " values below are median of SIX counterbalanced paired BATCH"
           " savings; no timings from different OS are pooled, and no pair"
           " is removed. Positive = candidate is faster.",
           ""]
    for label,cohorts in hosts:
        lines.extend([f"### {label}","",
                      "| Corpus | Backend | Iterations | Stop? | Original CPU ms/fit |"
                      " Candidate CPU ms/fit | Paired CPU saving | Paired wall saving |",
                      "|---|---|---:|---|---:|---:|---:|---:|"])
        for k in sorted(cohorts):
            s=cohorts[k]
            lines.append(f"| {k[0]} | {k[1]} | {k[2]} | {k[3]} | "
                         f"{s['original_cpu_ms']:.4f} | {s['candidate_cpu_ms']:.4f} | "
                         f"{s['cpu']:+.2f}% | {s['wall']:+.2f}% |")
        lines.append("")
    for group in ("primary-numba-heldout","real-digits-wall-heldout","no-real-regression",
                  "no-other-synthetic-regression","no-memory-regression"):
        selected=[x for x in checks if x["group"]==group]
        fail=[x for x in selected if not x["passed"]]
        lines.extend([f"### {group}: {len(selected)-len(fail)}/{len(selected)} pass",""])
        for x in fail:
            lines.append(f"- FAILED {x['os']} {x['cohort']} "
                         f"savings={x.get('savings_percent', 'n/a')} "
                         f"peak={x.get('candidate_bytes','n/a')} "
                         f"required={x.get('required',x.get('cap_bytes'))}")
        lines.append("")
    lines.extend([
        "## Interpretation",
        "",
        "The baseline original creates an advanced-indexed dominance submatrix"
        " before rowwise argmax; the candidate computes argmax over the FULL"
        " original matrix and selects only the unlabeled results **only when the actual backend is Numba**. This is"
        " a backend-identity-dispatched public NumPy change. Cython and NumPy retain the exact original expression. It may scan labeled rows"
        " unnecessarily; any isolated microphase speedup cannot establish a"
        " meaningful complete-fit CPU improvement.",
        "",
        f"**Frozen verdict {outcome}.** " + (
            "A distinct minimal production PR and its own full four-platform CI"
            " are required; this branch remains unmerged." if outcome=="GO"
            else "Keep the current production source unchanged and close the"
            " scientific branch without merge, preserving negative evidence."),
        "",
        "Measured memory peaks are limited to Python/NumPy tracemalloc; this is"
        " NOT an OS RSS/native heap measurement. Original padded dense int64"
        " neighbors and all original graph/particle algorithms stay unchanged."
    ])
    return "\n".join(lines)+"\n"


def main():
    root=Path(sys.argv[1]); output=Path(sys.argv[2])
    paths=sorted(root.rglob("argmax-numba-only-native-*.json"))
    require(len(paths)==2,f"need exact 2 original native host artifacts, found {paths}")
    docs=[json.loads(p.read_text(encoding="utf-8")) for p in paths]
    by_os={d["os"]:d for d in docs}
    require(set(by_os)=={"linux","win32"},"missing Linux/Windows")
    l,ml=rows_by_host(by_os["linux"])
    w,mw=rows_by_host(by_os["win32"])
    for sha in ("original_sha","candidate_sha"):
        require(by_os["linux"][sha]==by_os["win32"][sha],f"Git-source derived {sha} inconsistent")
    for k in expected():
        a,b=l[k],w[k]
        require(a["graph_sha"]==b["graph_sha"] and
                a["degree_sha"]==b["degree_sha"] and
                a["label_shas"]==b["label_shas"],f"input/ordered graph differ {k}")
        for idx in range(6):
            require(a["pairs"][idx]["state_shas"]==b["pairs"][idx]["state_shas"],
                    f"cross-host source-native arrays differ {k}")
            for j in range(2):
                require(a["pairs"][idx]["native_replay"][j]["stop_info"]==
                        b["pairs"][idx]["native_replay"][j]["stop_info"],
                        f"cross-host native stop metrics differ {k}")
    linux=scores(l); windows=scores(w)
    status,checks=decision(linux,windows,ml,mw)
    output.mkdir(parents=True,exist_ok=True)
    (output/"argmax_numba_only_scientific_report.md").write_text(
        render([("Linux",linux),("Windows",windows)],status,checks),
        encoding="utf-8")
    (output/"argmax_numba_only_scientific_decision.json").write_text(
        json.dumps(dict(base=BASE,verdict=status,source_exact=True,
                        native_parity_and_ownership=True,hosts=["linux","win32"],
                        cohorts_per_os=48,checks=checks),indent=2)+"\n",
        encoding="utf-8")
    print((output/"argmax_fullrows_scientific_report.md").read_text(encoding="utf-8"))
    print(f"STRICT BOTH-OS NUMBA-ONLY PCC ARGMAX SCIENCE COMPLETE: {status}")


if __name__=="__main__":
    main()
