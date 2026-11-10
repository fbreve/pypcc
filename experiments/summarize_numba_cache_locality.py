"""Independently verify native Numba D1/LL SIMULATED Callgrind events.

Ir/Dr/Dw are dynamic Valgrind instruction/memory-reference counts.
D1/LL misses are simulated against a hypothetical frozen cache hierarchy,
NOT actual hardware cache misses/cycles/CPU time. Linux cache profiling only;
Windows independently verifies every genuine original Numba state/RNG/stop.
"""
from collections import defaultdict
import json
from pathlib import Path
import re
import sys

BASE="4c1c410d8eb04547421f72ee1d3ca2eeb5763ff2"
DATASETS=("wine","digits")
SEEDS=(89021,89119)
POLICIES=(0.0,0.5,1.0)
STATES={"predictions","dominance","positions","strengths",
        "distances","owndeg","node_labels","mapped_labels"}
PROTECTED={"pcc.py","pcc_numba.py","pcc_graph.py","pcc_numpy.py",
           "pcc_validation.py","pcc_step.pyx","benchmarks/fixtures.py"}
METRICS=("Ir","Dr","Dw","I1mr","D1mr","D1mw","ILmr","DLmr","DLmw")
MEMORY=("Dr","Dw","D1mr","D1mw","DLmr","DLmw")


def check(cond,msg):
    if not cond:
        raise AssertionError(msg)


def sha(x):
    return isinstance(x,str) and len(x)==64 and all(c in "0123456789abcdef" for c in x)


def parse_profile(path,native_entries):
    lines=Path(path).read_text(encoding="utf-8",errors="replace").splitlines()
    mapped={}
    for line in lines:
        val=line.strip()
        z=re.fullmatch(r"(?:c?fn)=\((\d+)\)\s+(.+)",val)
        if z:
            mapped[z.group(1)]=z.group(2)
    events=None;totals=None;exclusive=defaultdict(lambda: defaultdict(int))
    fn=None;pending_call=False
    position=re.compile(r"^(?:\d+|[+\-*]\d*|\*)$")
    counts=0
    for line in lines:
        val=line.strip()
        if not val or val.startswith("#"):
            continue
        if val.startswith("events:"):
            ev=val.partition(":")[2].split()
            check(events is None and set(ev)==set(METRICS) and
                  len(ev)==len(METRICS),"actual Valgrind cache-sim events missing/duplicated: "+str(ev))
            events=ev
        elif val.startswith("totals:"):
            check(events is not None and totals is None,"multiple actual cache totals")
            raw=val.partition(":")[2].split()
            check(len(raw)==len(events) and all(x.isdigit() for x in raw),
                  "unparsable actual Valgrind total cache events")
            totals=dict(zip(events,map(int,raw)))
        elif val.startswith("fn="):
            tail=val[3:]
            z=re.fullmatch(r"\((\d+)\)(?:\s+(.*))?",tail)
            if z:
                fn=mapped.get(z.group(1),z.group(2) or "unresolved-id:"+z.group(1))
            else:
                fn=tail
            pending_call=False
        elif val.startswith("calls="):
            pending_call=True
        elif val.startswith(("cfn=","cob=","cfi=","fl=","fi=","fe=","ob=",
                             "cfl=","cfe=","jump=","jcnd=","jcc=")):
            continue
        elif fn is not None and events is not None:
            fields=val.split()
            if len(fields)==len(events)+1 and position.fullmatch(fields[0]):
                if pending_call:
                    pending_call=False
                    continue
                check(all(x.isdigit() for x in fields[1:]),
                      "unrecognized original Callgrind cache event cost encoding: "+val[:140])
                for metric,raw in zip(events,fields[1:]):
                    exclusive[fn][metric]+=int(raw)
                counts+=1
    check(totals is not None and all(totals[k]>0 for k in ("Ir","Dr","Dw")),
          "real native code/memory operations not included in isolated profile")
    check(counts>100 and len(exclusive)>5,"no native profiled functions/rows")
    sums={key:sum(c[key] for c in exclusive.values()) for key in events}
    check(sums==totals,
          "exclusive cost sum disagrees with raw Valgrind totals: "+
          str({k:(sums[k],totals[k]) for k in events if sums[k]!=totals[k]}))

    # Valgrind may report the native Numba code as anonymous 0x entry names.
    # Resolve solely SAME-PROCESS LLVM pointer recorded before collection.
    addresses={}
    for name,entries in native_entries.items():
        for entry in entries:
            addr=int(entry["address"],16)
            check(addr not in addresses or addresses[addr]==name,
                  "nonunique LLVM pointer")
            addresses[addr]=name
    resolved=defaultdict(lambda:defaultdict(int))
    for name,events_by_name in exclusive.items():
        marker=re.fullmatch(r"0x[0-9a-fA-F]+",name)
        label=addresses.get(int(name,16),name) if marker else name
        for key,amount in events_by_name.items():
            resolved[label][key]+=amount
    step=resolved.get("_pcc_step_numba_sequential",{})
    visible=step.get("Dr",0)>0 or step.get("Dw",0)>0
    top=sorted(resolved.items(),key=lambda item:-item[1]["Dr"])
    top_rows=[{"name":name,"counts":dict(counts)} for name,counts in top[:30]]
    return dict(
        event_order=events,raw_totals=totals,exclusive_totals=sums,
        independent_original_native_JIT_pointer_count=len(addresses),
        native_step_identified=visible,step_memory={k:step.get(k,0) for k in MEMORY},
        other_memory={k:totals[k]-step.get(k,0) for k in MEMORY},
        top_read_functions=top_rows,profile_function_count=len(resolved),
        raw_cost_rows=counts)


def inspect(d):
    check(d["schema"]==1 and d["base"]==BASE and
          d["steps"]==1000 and d["numba_jit_profiling"] is True and
          d["original_dense_padded_int64"] is True and
          d["owndeg_f_layout"] is True and
          d["distances_uint8"] is True,
          "not true original Numba source dense/uint8 source provenance")
    check(d["dataset"] in DATASETS and d["seed"] in SEEDS and
          d["policy"] in POLICIES and
          d["case_id"]==f"{d['dataset']}-k10-labels{d['seed']}-p{d['policy']}",
          "unexpected original data/graph/policy condition")
    check(d["cache_simulation_model"]==dict(
        D1_bytes=32768,D1_ways=8,LL_bytes=8388608,LL_ways=16,line_bytes=64),
        "cache simulation model drift")
    check(set(d["source_hashes"])==PROTECTED and
          all(sha(x) for x in d["source_hashes"].values()) and
          all(sha(x) for x in d["graph_sha"]+[d["labels_sha"],d["fixture_sha"]]),
          "original source SHA and exact original graph/label SHA missing")
    check(set(d["state_hashes"])==STATES and
          all(sha(x) for x in d["state_hashes"].values()),
          "eight true original output state hashes absent")
    check(d["native_layout"]==("C" if d["dataset"]=="digits" else "F"),
          "wrong original native C/F class layout")
    check(type(d["rng_next"]) is float and 0<=d["rng_next"]<1,
          "true Numba seeded MT19937 next draw absent")
    info=d["original_stop"]
    check(info["effective_iterations"]==1000 and
          info["stop_reason"]=="max_iter" and
          info["checks"]==0 and info["stalled_checks"]==0,
          "not 1000 exact original native loop steps")


def make_report(data):
    lines=[
        "# Source-original Numba dynamic memory references and simulated-cache misses",
        "",
        "**Complete scientific diagnostic; no source optimization and no CPU GO/NO-GO.**",
        "Linux Valgrind Callgrind modeled a **32 KiB 8-way L1D**,",
        "**8 MiB 16-way LL**, both 64-byte lines. The original Numba",
        "public 1000-step full fit was profiled after JIT compilation",
        "and graph preparation, with the exact original source and native RNG.",
        "An independent real original Windows run reproduced all graph SHA,",
        "eight array states, stop and Numba MT19937 endpoints on 12 cases.",
        "",
        "**Crucial limit:** Valgrind's D1mr/D1mw/DLmr/DLmw counts",
        "are **SIMULATED** cache misses, NOT real CPU hardware counters,",
        "native cache misses, CPU cycles, time, measured bandwidth,",
        "energy, cache bottleneck share or evidence of a faster kernel.",
        "Opt-in Numba debug symbols/profiling may modify generated machine code.",
        "Dr and Dw are dynamic native memory-reference events as seen",
        "under Valgrind; they also include Python runtime activity during",
        "the original public fit. Step-specific attribution relies",
        "ONLY on exact same-process LLVM JIT pointers.",
        "",
        "| Dataset | Seed | p_grd | Total Dr | Total Dw | D1 read misses (simulated) | LL read misses (simulated) | Native step Dr | Native step D1 read misses |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for key in sorted(data):
        a=data[key]["cache"]
        e=a["raw_totals"];j=a["step_memory"]
        ds,seed,p=key
        lines.append(f"| {ds} | {seed} | {p:.1f} | {e['Dr']:,} | "
                     f"{e['Dw']:,} | {e['D1mr']:,} | {e['DLmr']:,} | "
                     f"{j['Dr']:,} | {j['D1mr']:,} |")
    lines += ["","## Kernel attribution and simulated cache fractions","",
              "| Dataset | Seed | p_grd | Native JIT step found | Step of total Dr | Step D1 read miss / step Dr | Step LL read miss / step Dr |",
              "|---|---:|---:|---|---:|---:|---:|"]
    for key in sorted(data):
        a=data[key]["cache"];e=a["raw_totals"];j=a["step_memory"]
        ds,seed,p=key
        def rate(a,b):return 100*a/b if b else 0
        lines.append(f"| {ds} | {seed} | {p:.1f} | "
                     f"{'yes' if a['native_step_identified'] else 'NO — unresolved'} | "
                     f"{rate(j['Dr'],e['Dr']):.2f}% | "
                     f"{rate(j['D1mr'],j['Dr']):.4f}% | "
                     f"{rate(j['DLmr'],j['Dr']):.4f}% |")
    lines += ["","## Interpretation","",
              "The simulation measures references to **all memory** in",
              "the native compiled step, including original random-state",
              "helpers and other operations, NOT exclusively reads of",
              "the dominance or distance table. Separate source-line",
              "attribution would be needed before naming particular arrays",
              "as causal cache consumers; no attribution is invented.",
              "Differences across p_grd reflect different original",
              "particle trajectories, not an instruction or cache",
              "speedup from changing a single branch.",
              "",
              "Past source-changing NO-GOs #112/#214/#229/#230/#234",
              "remain closed. The padded dense int64 graph is a deliberate",
              "speed trade-off and must not be replaced on memory alone.",
              "Any candidate demands separate two-OS full-fit process CPU",
              "and exact-state/RNG/stop preregistered A/B.",
              "Raw GitHub Actions artifacts are finite-retention evidence."]
    return "\n".join(lines)+"\n"


def main(path):
    root=Path(path)
    linux_files=sorted(root.rglob("numba_cache_locality_linux_*.json"))
    windows_files=sorted(root.rglob("numba_cache_locality_windows_*.json"))
    check(len(linux_files)==len(windows_files)==12,
          f"require exact 12 Linux profiles and 12 genuine Windows original checks: {len(linux_files)} {len(windows_files)}")
    def load(files):
        rows={}
        for f in files:
            d=json.loads(f.read_text(encoding="utf-8"))
            inspect(d)
            key=(d["dataset"],d["seed"],d["policy"])
            check(key not in rows,"duplicated original source case "+str(key))
            rows[key]=(d,f)
        return rows
    linux=load(linux_files);windows=load(windows_files)
    expected={(ds,seed,p) for ds in DATASETS
              for seed in SEEDS for p in POLICIES}
    check(set(linux)==set(windows)==expected,
          "one or more original Linux/Windows policy/seed cases missing")
    outputs={}
    signatures=None
    for key in sorted(expected):
        d,filepath=linux[key]
        win,_=windows[key]
        check(d["linux_actual_valgrind"] is True and
              d["windows_uninstrumented_reference"] is False and
              d["callgrind_pid"]>0 and
              win["windows_uninstrumented_reference"] is True and
              win["linux_actual_valgrind"] is False and
              win["callgrind_pid"] is None,
              "Windows reference incorrectly claimed Linux simulated cache profile")
        for field in ("base","source_hashes","graph_sha","labels_sha","fixture_sha",
                      "state_hashes","rng_next","original_stop","native_layout"):
            check(d[field]==win[field],"cross-host exact original "+field+" differs in "+str(key))
        if signatures is None:
            signatures=d["source_hashes"]
        check(d["source_hashes"]==signatures,
              "original production source changed between cases")
        native=d["native_jit_exact_entry_addresses"]
        check("_pcc_step_numba_sequential" in native and
              native["_pcc_step_numba_sequential"] and
              d["original_compiled_step_signatures"],
              "real same-process exact Numba step pointer unavailable")
        raw=list(filepath.parent.glob(f"callgrind.out.{d['callgrind_pid']}"))
        check(len(raw)==1,"original matching raw Valgrind CPU PID archive missing")
        cache=parse_profile(raw[0],native)
        outputs[key]=dict(cache=cache,linux=d,windows=win)
    all_visible=all(row["cache"]["native_step_identified"] for row in outputs.values())
    verdict="NATIVE_SIMULATED_CACHE_ATTRIBUTED" if all_visible else "PROFILE_UNRESOLVED"
    report=make_report(outputs)
    Path("numba_cache_locality_report.md").write_text(report,encoding="utf-8")
    decision=dict(schema=1,base=BASE,verdict=verdict,
                  profiling_only=True,simulated_not_hardware_counters=True,
                  original_source_unchanged=True,
                  profiles_linux=12,source_replay_windows=12,
                  eight_states_rng_stop_cross_host_exact=True,
                  observations={str(key):value["cache"] for key,value in outputs.items()})
    Path("numba_cache_locality_decision.json").write_text(
        json.dumps(decision,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    print(report,flush=True)
    print("STRICT ORIGINAL NUMBA SIMULATED CACHE TWO-HOST SOURCE PARITY COMPLETE",
          verdict,flush=True)


if __name__=="__main__":
    main(sys.argv[1])
