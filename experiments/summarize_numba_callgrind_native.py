"""Validate actual isolated Linux Callgrind Ir totals and Numba JIT coverage.

Ir = dynamically executed INSTRUCTIONS under Valgrind, not measured runtime
cycles, energy, native IPC, or a claim of CPU-time allocation.
"""
from collections import defaultdict
import json
import re
from pathlib import Path
import sys

BASE="b4c100bd22eeff1979a1a581bc73682b8c3705b8"
DATASETS=("wine","digits")
SEEDS=(73013,73117)


def check(cond,reason):
    if not cond:
        raise AssertionError(reason)


def read_ir(path):
    contents=Path(path).read_text(encoding="utf-8",errors="replace").splitlines()
    mapped={}
    events=None
    total=None
    for line in contents:
        value=line.strip()
        if value.startswith(("fn=","cfn=")):
            found=re.fullmatch(r"(?:c?fn)=\((\d+)\)\s+(.+)",value)
            if found:
                mapped[found.group(1)]=found.group(2)
    counters=defaultdict(int)
    current=None
    pending_call=False
    position=re.compile(r"^(?:\d+|[+\-*]\d*|\*)$")
    for line in contents:
        value=line.strip()
        if not value or value.startswith("#"):
            continue
        if value.startswith("events:"):
            events=value.partition(":")[2].split()
        elif value.startswith("totals:"):
            check(total is None,"multiple original instruction totals")
            total=int(value.partition(":")[2].strip())
        elif value.startswith("fn="):
            tail=value[3:]
            compressed=re.fullmatch(r"\((\d+)\)(?:\s+(.*))?",tail)
            current=(mapped.get(compressed.group(1),
                                compressed.group(2) or "unresolved-id:"+compressed.group(1))
                     if compressed else tail)
            pending_call=False
        elif value.startswith("calls="):
            pending_call=True
        elif value.startswith(("cfn=","cob=","cfi=","fl=","fi=","fe=","ob=")):
            continue
        elif current is not None:
            parts=value.split()
            if (len(parts)>=2 and position.fullmatch(parts[0]) and
                    parts[1].isdigit()):
                if pending_call:
                    pending_call=False
                else:
                    counters[current]+=int(parts[1])
    check(events==["Ir"] and total is not None and total>0,
          "missing real isolated Callgrind Ir event/totals")
    accounted=sum(counters.values())
    check(accounted==total,
          "exclusive instruction count != actual Callgrind global totals "
          +repr((accounted,total)))
    check(total>1000000,"instrumentation apparently did not include a real kernel")
    top=sorted(counters.items(),key=lambda p:(-p[1],p[0]))
    groups={
        "numba_step":("pcc_step_numba_sequential",),
        "numba_propagate":("pcc_propagate_numba_sequential",),
        "numba_potential":("pcc_calc_mmpot_sequential",),
        "numba_rng_seed":("seed_numba_rng",),
    }
    collected={}
    for grp,needles in groups.items():
        counted=sum(count for name,count in top
                    if any(needle in name.lower() for needle in needles))
        collected[grp]=counted
    visible=collected["numba_step"]>0 or collected["numba_propagate"]>0
    return dict(events=events,instructions=total,
        attributed_percent=round(100*sum(collected.values())/total,4),
        original_native_attribution=collected,
        native_jit_functions_visible=visible,
        top_functions=[dict(name=name,instructions=count,
                            percentage=round(100*count/total,4))
                       for name,count in top[:50]],
        function_count=len(top))


def main(path):
    root=Path(path)
    artifact_files=list(root.glob("**/numba_callgrind_case_*.json"))
    check(len(artifact_files)==4,
          f"Expected exactly 4 original cases and 4 native Callgrind files; got {len(artifact_files)}")
    observations=[]
    observed=set()
    canonical=None
    for case_file in artifact_files:
        case=json.loads(case_file.read_text(encoding="utf-8"))
        ds,seed=case["dataset"],case["seed"]
        key=(ds,seed)
        check(ds in DATASETS and seed in SEEDS and key not in observed,
              "missing or duplicate real sklearn source/label mask")
        observed.add(key)
        check(case["schema"]==1 and case["base"]==BASE and
              case["case_id"]==f"{ds}-labels{seed}" and
              case["numba_jit_profiling"] is True and
              case["isolated_profile"] is True,
              "invalid real pinned original native source")
        check(case["original_graph_dense_int64"] is True and
              len(case["graph_sha"])==2 and
              all(len(x)==64 for x in case["graph_sha"]) and
              len(case["source_sha"])==64,
              "graph source/dense numpy backing and source hash missing")
        if canonical is None:canonical=case["source_sha"]
        check(case["source_sha"]==canonical,
              "nonfrozen production Numba source between cases")
        check(case["native_layout"]==("C" if ds=="digits" else "F"),
              "incorrect genuine Numba class memory layout")
        check(len(case["state_hashes"])==6 and all(
            len(s)==64 for s in case["state_hashes"].values()),
            "six final true Numba state SHA missing")
        check(isinstance(case["rng_next"],float)
              and 0<=case["rng_next"]<1,
              "Numba native RNG after fit missing")
        stop=case["original_stop"]
        check(stop["effective_iterations"]==1000 and
              stop["stop_reason"]=="max_iter" and
              stop["checks"]==0 and
              stop["stalled_checks"]==0,
              "not exact frozen 1000-step original fixed propagation")
        check(case["original_compiled_step_signatures"],
              "no compiled Numba native step signed proof")
        raw_files=list(case_file.parent.glob(
            f"callgrind.out.{case['callgrind_pid']}"))
        check(len(raw_files)==1,
              "native Callgrind profile was not archived for actual source PID")
        profile=read_ir(raw_files[0])
        observations.append(dict(
            dataset=ds,seed=seed,fixture_sha=case["fixture_sha"],
            graph_sha=case["graph_sha"],
            callgrind_file=str(raw_files[0]),pid=case["callgrind_pid"],
            **profile))
    check(observed=={(ds,s) for ds in DATASETS for s in SEEDS},
          "incomplete original source Numba profile set")
    for ds in DATASETS:
        rows=[x for x in observations if x["dataset"]==ds]
        check(rows[0]["graph_sha"]==rows[1]["graph_sha"],
              "original canonical graph differs across label seeds")
    resolved=all(x["native_jit_functions_visible"] for x in observations)
    verdict="NATIVE-JIT-ATTRIBUTED" if resolved else "PROFILE-UNRESOLVED"
    lines=[
        "# Native original Numba instruction-level Callgrind profile",
        "",
        "Original frozen post-#216 production Numba JIT source, 4 real",
        "offline sklearn Wine/Digits cases, original symmetric dense int64",
        "kNN10, 3 labeled homes per class, seeded full 1000 fixed steps.",
        "Compiled real Numba signatures were warmed BEFORE Callgrind",
        "instrumentation, and all six final states and Numba RNG samples",
        "were compared to independent exact source replay AFTER profiling.",
        "",
        "**IMPORTANT: Callgrind Ir counts are dynamic INSTRUCTION counts,",
        "not time, CPU cycles, memory bandwidth, cache misses, or energy.**",
        "NUMBA_ENABLE_PROFILING/DEBUGINFO may alter native JIT optimization;",
        "do NOT directly infer percentage CPU time from instruction shares.",
        "",
        "| Dataset | Seed | Isolated native instruction Ir | Numba step Ir | Propagation Ir | Potential Ir | Native JIT symbols resolved |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for r in sorted(observations,key=lambda r:(r["dataset"],r["seed"])):
        c=r["original_native_attribution"]
        lines.append(
            f"| {r['dataset']} | {r['seed']} | {r['instructions']:,} | "
            f"{c['numba_step']:,} | {c['numba_propagate']:,} | "
            f"{c['numba_potential']:,} | "
            f"{'YES' if r['native_jit_functions_visible'] else 'NO'} |")
        lines.extend([
            "",
            f"### {r['dataset']}, labeled seed {r['seed']}",
            "",
            "Largest exclusive original dynamic instruction consumers:",
            "",
        ])
        for f in r["top_functions"][:12]:
            lines.append(
                f"- {f['percentage']:.2f}%  — {f['instructions']:,} Ir — "
                f"\`{f['name'][:140]}\`")
    lines.extend([
        "",
        f"**Strict diagnostic verdict: {verdict}**. All four original",
        "instruction profiles and exact native state/RNG/graph/stop/source",
        "evidence were present. If PROFILE-UNRESOLVED, the JIT native",
        "functions could not be reliably attributed; this is a profiling",
        "visibility limitation, NOT a claim that compiled code has no cost.",
        "No code from this scientific PR should be merged.",
    ])
    Path("numba_callgrind_profile_report.md").write_text(
        "\n".join(lines)+"\n",encoding="utf-8")
    Path("numba_callgrind_profile_decision.json").write_text(
        json.dumps(dict(schema=1,complete=True,verdict=verdict,
                        count=4,cases=observations),
                   indent=2,allow_nan=False)+"\n",encoding="utf-8")
    print("\n".join(lines),flush=True)
    print("NUMBA_ACTUAL_LINUX_CALLGRIND_STRICT_COMPLETE",verdict,flush=True)


if __name__=="__main__":
    main(sys.argv[1])
