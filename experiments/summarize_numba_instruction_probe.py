"""Exact original Numba instruction-address feasibility from 3 Linux raw profiles.

This is NOT source-line or array-object profiling. Dynamic Dr/Dw are
instrumented Valgrind references; D1/LL misses are simulated, NOT hardware.
Every raw Callgrind exclusive cost must reconcile EXACTLY to its 9 totals.
"""
from collections import defaultdict
import json
from pathlib import Path
import re
import sys

BASE="4c1c410d8eb04547421f72ee1d3ca2eeb5763ff2"
CASES={("digits",89021,0.0),("digits",89021,0.5),
       ("wine",89021,0.5)}
EVENTS=("Ir","Dr","Dw","I1mr","D1mr","D1mw","ILmr","DLmr","DLmw")
STATES={"predictions","dominance","positions","strengths",
        "distances","owndeg","node_labels","mapped_labels"}
SOURCE={"pcc.py","pcc_numba.py","pcc_graph.py","pcc_numpy.py",
        "pcc_validation.py","pcc_step.pyx","benchmarks/fixtures.py"}
CACHE=dict(D1_bytes=32768,D1_ways=8,LL_bytes=8388608,LL_ways=16,line_bytes=64)
NONHARDWARE=("D1mr","D1mw","DLmr","DLmw")
POS=re.compile(r"(?:0x[0-9a-fA-F]+|\d+|[+\-](?:0x[0-9a-fA-F]+|\d+)|\*)\Z")


def must(v,m):
    if not v:
        raise AssertionError(m)


def hash64(s):
    return isinstance(s,str) and re.fullmatch("[0-9a-f]{64}",s) is not None


def source_valid(d):
    must(d["schema"]==1 and d["base"]==BASE and
         d["steps"]==1000 and d["numba_jit_profiling"] is True and
         d["original_dense_padded_int64"] is True and
         d["owndeg_f_layout"] is True and d["distances_uint8"] is True,
         "not original source 1000-step Numba profile")
    key=(d["dataset"],d["seed"],d["policy"])
    must(key in CASES,"unexpected dataset/label seed/movement policy")
    must(d["cache_simulation_model"]==CACHE,"simulated-cache model drift")
    must(set(d["source_hashes"])==SOURCE and
         all(hash64(v) for v in d["source_hashes"].values()),
         "original guarded production source missing")
    must(len(d["graph_sha"])==2 and
         all(hash64(h) for h in d["graph_sha"]+
             [d["labels_sha"],d["fixture_sha"]]),
         "real original undirected graph/labels evidence missing")
    must(set(d["state_hashes"])==STATES and
         all(hash64(v) for v in d["state_hashes"].values()),
         "all eight original full state fingerprints required")
    must(d["native_layout"]==("C" if d["dataset"]=="digits" else "F"),
         "wrong actual original C/F dominance")
    must(type(d["rng_next"]) is float and 0<=d["rng_next"]<1,
         "true Numba MT19937 endpoint invalid")
    stop=d["original_stop"]
    must(stop["effective_iterations"]==1000 and
         stop["stop_reason"]=="max_iter" and
         stop["checks"]==0 and stop["stalled_checks"]==0,
         "original real native fixed 1000 iteration control missing")
    return key


def native_step_pointer(d):
    functions=d["native_jit_exact_entry_addresses"]
    must("_pcc_step_numba_sequential" in functions and
         len(functions["_pcc_step_numba_sequential"])==1,
         "original same-process JIT step address missing")
    ptr=int(functions["_pcc_step_numba_sequential"][0]["address"],16)
    must(ptr>0,"original compiled source step pointer is zero")
    return ptr


def read_profile(path,pointer):
    lines=Path(path).read_text(encoding="utf-8",errors="replace").splitlines()
    maps={}
    for raw in lines:
        m=re.fullmatch(r"(?:c?fn)=\((\d+)\)\s+(.+)",raw.strip())
        if m:
            if m.group(1) in maps:
                must(maps[m.group(1)]==m.group(2),
                     "ambiguous original callgrind fn ID")
            maps[m.group(1)]=m.group(2)
    functions=defaultdict(lambda:defaultdict(int))
    instructions=defaultdict(lambda:defaultdict(int))
    current=None;pending=False;total=None;events=None;position=None
    row_count=0;named=0;unmapped=0;last_addr=None
    for raw in lines:
        s=raw.strip()
        if not s or s.startswith("#"):
            continue
        if s.startswith("positions:"):
            must(position is None,"multiple original Callgrind position schemas")
            position=s.partition(":")[2].split()
        elif s.startswith("events:"):
            ev=s.partition(":")[2].split()
            must(events is None and tuple(ev)==EVENTS,
                 "incomplete or reordered original nine cache events")
            events=ev
        elif s.startswith("totals:"):
            vals=s.partition(":")[2].split()
            must(total is None and events is not None and
                 len(vals)==len(EVENTS) and all(v.isdigit() for v in vals),
                 "invalid original Callgrind raw totals")
            total=dict(zip(EVENTS,map(int,vals)))
        elif s.startswith("fn="):
            tail=s[3:]
            m=re.fullmatch(r"\((\d+)\)(?:\s+(.*))?",tail)
            current=(maps.get(m.group(1),m.group(2) or
                     "unresolved-fn:"+m.group(1)) if m else tail)
            pending=False
            last_addr=None
        elif s.startswith("calls="):
            pending=True
        elif s.startswith(("cfn=","cob=","cfi=","fl=","fi=","fe=",
                           "ob=","cfl=","cfe=","jump=","jcnd=","jcc=")):
            continue
        elif current is not None and events is not None and position is not None:
            parts=s.split()
            if not (parts and POS.fullmatch(parts[0])):
                continue
            must(2<=len(parts)<=len(EVENTS)+1 and
                 all(v.isdigit() for v in parts[1:]),
                 "unsupported Callgrind instruction/native cost row "+s[:150])
            if pending:
                pending=False
                continue  # CALL EDGE, never count callee-inclusive costs twice.
            # The frozen workflow uses --compress-pos=no, so the recorded
            # instruction position should be absolute. Never manufacture
            # an address if Callgrind emits a compressed/relative position.
            token=parts[0]
            if token[0] in "+-*":
                unmapped+=1
                addr=None
            else:
                addr=int(token,16) if token.startswith("0x") else int(token)
                if addr==0:
                    unmapped+=1
                    addr=None
                else:
                    named+=1
            cost=dict(zip(EVENTS,map(int,parts[1:])))
            for metric,value in cost.items():
                functions[current][metric]+=value
                # Callgrind also names unrelated functions e.g.
                # "0x0000000004950c20'2" (disambiguation suffix).
                # NEVER coerce such a noncanonical name into a real
                # compiled JIT pointer; only a FULL hexadecimal match is
                # eligible for native-step attribution.
                if (re.fullmatch(r"0x[0-9a-fA-F]+",current) and
                        int(current,16)==pointer):
                    if addr is not None:
                        instructions[addr][metric]+=value
                    elif value:
                        # Preserve un-attributed JIT cost; this must block
                        # claims of complete instruction visibility.
                        instructions[None][metric]+=value
            row_count+=1
    must(position==["instr"],"native dump did NOT retain instruction positions")
    must(total is not None and len(functions)>5 and row_count>100,
         "incomplete original Callgrind memory profile")
    summation={metric:sum(f[metric] for f in functions.values())
               for metric in EVENTS}
    must(summation==total,
         "original per-function exclusive events do not match 9 raw totals "+
         str({k:(summation[k],total[k]) for k in EVENTS
              if summation[k]!=total[k]}))
    label="0x"+format(pointer,"x")
    # Explicit JIT fn labels may have leading zeroes; canonicalize pointers.
    step_fns=[value for name,value in functions.items()
              if re.fullmatch(r"0x[0-9A-Fa-f]+",name)
              and int(name,16)==pointer]
    must(len(step_fns)==1 and step_fns[0]["Dr"]>0,
         "true same-process compiled native JIT step could not be resolved")
    step=step_fns[0]
    instr_summary={k:sum(d[k] for d in instructions.values())
                   for k in EVENTS}
    visible=(None not in instructions and len(instructions)>1 and
             all(instr_summary[k]==step[k] for k in EVENTS) and
             all(addr>=pointer and addr-pointer<1048576 for addr in instructions))
    top=sorted(((a,v) for a,v in instructions.items() if a is not None),
               key=lambda pair:(-pair[1]["D1mr"],pair[0]))[:10]
    return dict(raw_totals=total,exclusive_sums=summation,
                original_native_pointer=hex(pointer),
                original_step_events={k:step[k] for k in EVENTS},
                positions=position,native_cost_rows=row_count,
                absolute_cost_rows=named,unresolved_cost_rows=unmapped,
                native_instruction_addresses=len(instructions)-
                (1 if None in instructions else 0),
                all_step_instr_events_reconcile=instr_summary==dict(step),
                complete_llvm_entry_relative_addresses=visible,
                top10_simulated_D1_reads=[
                    dict(instruction=hex(addr),offset=hex(addr-pointer),
                         D1mr=v["D1mr"],Dr=v["Dr"],DLmr=v["DLmr"])
                    for addr,v in top if addr>=pointer],
                top10_D1mr_share_of_native_step_pct=(
                    100*sum(v["D1mr"] for addr,v in top
                            if addr>=pointer)/step["D1mr"]
                    if visible and step["D1mr"] else None),
                note="Callgrind simulated D1/LL; no source line or array identity")


def extract_profiles(path,prefix):
    files=sorted(Path(path).rglob(prefix+"*.json"))
    required=3 if prefix=="numba_instr_probe_" else 12
    must(len(files)==required,
         "missing exact raw profiles "+prefix+": "+str(len(files)))
    indexed={}
    for file in files:
        d=json.loads(file.read_text(encoding="utf-8"))
        key=(d["dataset"],d["seed"],d["policy"])
        if key not in CASES:
            must(prefix!="numba_instr_probe_","unexpected new case")
            continue  # Frozen #236 has nine OTHER original cases.
        key=source_valid(d)
        must(key not in indexed,"duplicate fixed original graph/policy")
        indexed[key]=(d,file)
    must(set(indexed)==CASES,"missing required original 3-case subset")
    return indexed


def render(result,verdict):
    lines=[
        "# Numba JIT instruction-position attribution: genuine original source",
        "",
        "**"+verdict+"** — scientific feasibility ONLY, no production CPU GO.",
        "",
        "The entire #236 original corpus had source line 0 for its exact",
        "Numba JIT step; this targeted new 3-case original-source Callgrind",
        "probe requests machine INSTRUCTION positions instead.",
        "Both memory read/write event counts and D1/LL misses are derived",
        "from the same hypothetical 32KiB L1D, 8MiB LL cache simulator.",
        "Misses are NOT physical CPU hardware events or CPU speedups.",
        "Real original full-fit graph, eight states, Numba MT19937 and",
        "stopping are replayed bytewise against original #236 Linux/Windows.",
        "",
        "| Original input | p_grd | Step Dr | Simulated step D1 read misses | Distinct step instructions | Top 10 instruction share of step simulated D1 misses |",
        "|---|---:|---:|---:|---:|---:|"]
    for (ds,seed,policy),d in sorted(result.items()):
        prof=d["instruction"]
        share=prof["top10_D1mr_share_of_native_step_pct"]
        ratio="unresolved" if share is None else f"{share:.2f}%"
        lines.append(f"| {ds} kNN10, seed {seed} | {policy:.1f} | "
                     f"{prof['original_step_events']['Dr']:,} | "
                     f"{prof['original_step_events']['D1mr']:,} | "
                     f"{prof['native_instruction_addresses']:,} | {ratio} |")
    lines+=["","## Top simulated D1 read-miss instruction addresses",""]
    for key,d in sorted(result.items()):
        ins=d["instruction"]
        lines.append(f"### {key}")
        if not ins["complete_llvm_entry_relative_addresses"]:
            lines.append("Instruction addresses unresolved; no address attribution claimed.")
        else:
            lines.extend(
                f"- offset **{x['offset']}**: D1mr={x['D1mr']:,}, "
                f"Dr={x['Dr']:,}, DLmr={x['DLmr']:,} (all simulated caches)"
                for x in ins["top10_simulated_D1_reads"])
        lines.append("")
    lines.extend([
        "**Interpretation boundary:** even a high fraction attributed to",
        "a JIT instruction does NOT identify dominance, dist_table,",
        "neib_list or owndeg. One instruction can serve multiple buffers,",
        "and the original JIT debug/profiling switches may affect codegen.",
        "No proposed memory layout change follows from these counts.",
        "Prior CPU-negative layout/neighbor changes #100/#104/#214/#229",
        "remain binding; dense padded int64 neighbor representation is",
        "intentionally retained for CPU speed. Raw Actions artifacts",
        "expire under GitHub retention; report preserves only results."
    ])
    return "\n".join(lines)+"\n"


def main(folder):
    root=Path(folder)
    new=extract_profiles(root,"numba_instr_probe_")
    old_linux=extract_profiles(root,"numba_cache_locality_linux_")
    old_win=extract_profiles(root,"numba_cache_locality_windows_")
    scores={}
    source_hashes=None
    for key,(current,file) in new.items():
        must(current["linux_actual_valgrind"] is True and
             current["windows_uninstrumented_reference"] is False,
             "new genuine Linux Valgrind required")
        for previous,label in ((old_linux[key][0],"Linux #236"),
                               (old_win[key][0],"Windows #236")):
            for field in ("base","source_hashes","graph_sha",
                          "labels_sha","fixture_sha","state_hashes",
                          "rng_next","original_stop","native_layout",
                          "owndeg_f_layout","distances_uint8"):
                must(current[field]==previous[field],
                     "new original native state/seed/source diverged from "+label+
                     " for "+str(key)+" field "+field)
        if source_hashes is None:
            source_hashes=current["source_hashes"]
        must(current["source_hashes"]==source_hashes,
             "mixed original production source across new cases")
        outputs=sorted(file.parent.glob("callgrind.out.*"))
        must(len(outputs)==1,"original Callgrind raw PID profile missing")
        profile=read_profile(outputs[0],native_step_pointer(current))
        scores[key]=dict(instruction=profile,
                         source_sha=current["source_hashes"],
                         state=current["state_hashes"])
    visible=all(s["instruction"]["complete_llvm_entry_relative_addresses"]
                for s in scores.values())
    verdict="INSTRUCTION_ADDRESS_VISIBLE" if visible else "SOURCE_INSTR_UNRESOLVED"
    report=render(scores,verdict)
    Path("numba_jit_instruction_probe_report.md").write_text(
        report,encoding="utf-8")
    Path("numba_jit_instruction_probe_decision.json").write_text(
        json.dumps(dict(schema=1,base=BASE,verdict=verdict,
                        cases=3,all_nine_original_raw_totals_exact=True,
                        full_state_mt_rng_stop_source_verified_with_236=True,
                        no_production_performance_go=True,
                        simulated_not_hardware=True,
                        results={str(k):v["instruction"]
                                 for k,v in scores.items()}),
                   indent=2,allow_nan=False)+"\n",encoding="utf-8")
    print(report,flush=True)
    print("STRICT ORIGINAL NUMBA INSTRUCTION PROBE COMPLETE",verdict,flush=True)


if __name__=="__main__":
    main(sys.argv[1])
