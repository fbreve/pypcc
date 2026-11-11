"""Independent STRICT two-host verdict for the frozen Cython rowmax study.

Scientific CI completion and CPU optimization GO are separate decisions.
All failed GO checks remain mandatory negative results, NOT parser omissions.
"""
import json
from pathlib import Path
from statistics import median
import sys

from experiments.cython_canonical_rowmax_init_probe import (
    BASE, COHORTS, MASKS, SEQ, BATCH, BLOCKS_PER_PROCESS, cohort_id
)


def require(value, message):
    if not value:
        raise AssertionError("STRICT_FROZEN_CYTHON_AUDIT " + message)


def read_one(directory,pattern):
    found=list(Path(directory).rglob(pattern))
    require(len(found)==1, "expected one file "+pattern+" found "+str(len(found)))
    return json.loads(found[0].read_text(encoding="utf-8"))


def check(root):
    root=Path(root)
    results={}
    hashes={}
    summaries={}
    verdicts=[]
    for host in ("ubuntu-latest","windows-latest"):
        p=root/host
        completed=read_one(p,"complete.json")
        manifest=read_one(p,"native_source_manifest.json")
        require(completed["base"]==manifest["base"]==BASE and
                completed["host"]==("linux" if host.startswith("ubuntu") else "windows") and
                completed["runs"]==len(COHORTS)*len(MASKS)*len(SEQ) and
                completed["source_order"]==list(SEQ) and
                completed["masks"]==list(MASKS),
                "incomplete source-exact native host experiment "+host)
        require(set(manifest["candidate_only_changes"])=={"pcc.py","pcc_step.pyx"} and
                manifest["distinct_original_native_binary"] and
                manifest["no_production_file_changes"],
                "production source/provenance guard absent")
        for filename,oldsha in manifest["files"]["original"].items():
            newsha=manifest["files"]["candidate"][filename]
            require((oldsha!=newsha)==(filename in ("pcc.py","pcc_step.pyx")),
                    "source differences are not the precise two permitted files")
        hashes[host]=manifest["files"]
        rows=list((p/"measurements").glob("*.json"))
        require(len(rows)==completed["runs"],"wrong actual source-native case count "+host)
        results[host]={}
        for cohort in COHORTS:
            cid=cohort_id(cohort)
            for mask in MASKS:
                key=cid+"_mask"+str(mask)
                seq=[]
                for round_no, variant in enumerate(SEQ):
                    file=p/"measurements"/(cid+"_mask"+str(mask)+"_round"+
                        str(round_no)+"_"+variant+".json")
                    require(file.is_file(),"missing frozen native source "+str(file))
                    x=json.loads(file.read_text(encoding="utf-8"))
                    require(x["schema"]==1 and x["base"]==BASE and
                            x["variant"]==variant and x["cohort"]==cid and
                            x["mask_seed"]==mask and x["iterations"]==cohort[1] and
                            x["early_stop"]==cohort[2] and
                            x["host"]==completed["host"] and
                            x["compiled_real_cython_backend"] is True and
                            x["new_fast_path_only_in_candidate_highlevel_cached"] is True and
                            x["previous_fit_ownership_and_invalid_rollback"] is True and
                            x["batch_size"]==BATCH and
                            x["source_hashes"]["pcc.py"]==manifest["files"][variant]["pcc.py"] and
                            x["source_hashes"]["pcc_step.pyx"]==manifest["files"][variant]["pcc_step.pyx"],
                            "original native source/fixture mismatch "+key)
                    require(len(x["cpu_batches"])==BLOCKS_PER_PROCESS,
                            "missing original native paired CPU blocks "+key)
                    for i,block in enumerate(x["cpu_batches"]):
                        require(block["block"]==i and block["fit_count"]==BATCH and
                                block["cpu_ns"]>0 and block["wall_ns"]>0 and
                                block["last_state_hashes"]==x["state"],
                                "missing or non-equivalent whole original Cython fit")
                    require(set(x["state"])=={
                            "predictions","dominance","node_labels",
                            "part_curnode","part_label","part_strength","dist_table",
                            "owndeg","mapped_labels","part_homenode"
                        } and all(len(h)==64 for h in x["state"].values()),
                        "source-native output SHA missing")
                    require(x["direct_canonical_cache_verifier"] is not None and
                            x["direct_noncanonical_default_cache_verifier"] is not None,
                            "canonical and low-level arbitrary cache verification missing")
                    require(x["stop"]["effective_iterations"]<=cohort[1] and
                            x["stop"]["checks"] == (
                                (x["stop"]["effective_iterations"]+9)//10 if cohort[2] else 0),
                            "actual native stopping invalid")
                    seq.append(x)
                baseline=seq[0]
                for ix,x in enumerate(seq[1:],1):
                    for field in ("inputs","state","stop","direct_canonical_cache_verifier",
                                  "direct_noncanonical_default_cache_verifier","c","n_nodes","k_nn"):
                        require(x[field]==baseline[field],
                                "not byte-equivalent original/candidate full state "+
                                field+" at "+host+":"+key+" round "+str(ix))
                # Frozen order: O, C, C, O. Three original/candidate
                # block comparisons in each counterbalanced half.
                pairs=[]
                for original_i,candidate_i in ((0,1),(3,2)):
                    for j in range(BLOCKS_PER_PROCESS):
                        orig=seq[original_i]["cpu_batches"][j]
                        cand=seq[candidate_i]["cpu_batches"][j]
                        pairs.append({
                            "cpu_saving_percent":
                                100.0*(orig["cpu_ns"]-cand["cpu_ns"])/orig["cpu_ns"],
                            "wall_saving_percent":
                                100.0*(orig["wall_ns"]-cand["wall_ns"])/orig["wall_ns"],
                            "orig_cpu_ns":orig["cpu_ns"],"candidate_cpu_ns":cand["cpu_ns"],
                            "orig_wall_ns":orig["wall_ns"],"candidate_wall_ns":cand["wall_ns"]
                        })
                require(len(pairs)==6,"missing six true native original-candidate pairs")
                cpu=median(p["cpu_saving_percent"] for p in pairs)
                wall=median(p["wall_saving_percent"] for p in pairs)
                orig_memory=median(seq[i]["python_traced_peak_bytes"] for i in (0,3))
                candidate_memory=median(seq[i]["python_traced_peak_bytes"] for i in (1,2))
                require(orig_memory>0 and candidate_memory>0 and
                        candidate_memory-orig_memory<=8*baseline["n_nodes"]+1048576,
                        "full-fit Python peak memory gate FAIL at "+host+":"+key)
                v={
                    "host":host, "cohort":cid,"mask_seed":mask,
                    "n_nodes":baseline["n_nodes"],"classes":baseline["c"],
                    "cpu_saving_percent":cpu, "wall_saving_percent":wall,
                    "python_traced_peak_extra_bytes":candidate_memory-orig_memory,
                    "actual_exact_original_and_candidate_terminal_state":True,
                    "actual_original_native_stop_and_cython_seed_replay":True,
                    "six_full_fit_cpu_pairs":pairs,
                }
                primary=cohort[0]=="ring100k50" and cohort[1] in (0,5)
                v["primary_100k50_cpu_gate_pass"]=bool(not primary or cpu>=5.0)
                v["secondary_no_cpu_regression_pass"]=bool(primary or cpu>=-5.0)
                v["whole_fit_wall_no_regression_pass"]=bool(wall>=-5.0)
                results[host][key]=v
                for kind in ("primary_100k50_cpu_gate_pass",
                             "secondary_no_cpu_regression_pass",
                             "whole_fit_wall_no_regression_pass"):
                    if not v[kind]:
                        verdicts.append(host+" / "+key+" / "+kind+" / "+
                                        str(round(cpu if kind!="whole_fit_wall_no_regression_pass"
                                                  else wall,3))+"%")
                summaries[(host,key)]=v
    # Frozen input/source/results are identical independent of host; CPU is not pooled.
    require(hashes["ubuntu-latest"]==hashes["windows-latest"],
            "different source files compiled on Windows/Linux")
    for cohort in COHORTS:
        cid=cohort_id(cohort)
        for mask in MASKS:
            k=cid+"_mask"+str(mask)
            left=read_one(root/"ubuntu-latest"/"measurements",k+"_round0_original.json")
            right=read_one(root/"windows-latest"/"measurements",k+"_round0_original.json")
            for field in ("inputs","state","stop","direct_canonical_cache_verifier",
                          "direct_noncanonical_default_cache_verifier"):
                require(left[field]==right[field],
                        "true original Cython state/graph/stop mismatch across OS "+k)
    decision="CPU_GO" if not verdicts else "CPU_NO_GO"
    report = {
        "schema":1,"original_master":BASE,"decision":decision,
        "complete_original_candidate_native_source_state_stop_parity":True,
        "native_linux_windows_full_fit_whole_cpu":True,
        "source_identity_cross_os":True,
        "frozen_primary_threshold_percent":5.0,
        "frozen_secondary_no_regression_threshold_percent":-5.0,
        "frozen_wall_no_regression_threshold_percent":-5.0,
        "failures":verdicts,
        "measurements":[summaries[k] for k in sorted(summaries)],
        "research_branch_must_never_merge":True
    }
    Path("cython_canonical_rowmax_cpu_decision.json").write_text(
        json.dumps(report,indent=2)+"\n",encoding="utf-8")
    lines=[
        "# Frozen Cython canonical-rowmax initialization exact-source 2-OS trial","",
        "**"+decision+"** — true original and candidate native Cython source, "+
        "six paired batches per cohort, both Linux and Windows.",
        "", "| Host | Cohort | Mask | Full-fit CPU saving | Full-fit wall saving |",
        "|---|---|---:|---:|---:|",
    ]
    for k in sorted(summaries):
        row=summaries[k]
        lines.append("| "+row["host"]+" | "+row["cohort"]+" | "+str(row["mask_seed"])+
                     " | "+format(row["cpu_saving_percent"],".2f")+"% | "+
                     format(row["wall_saving_percent"],".2f")+"% |")
    lines+=["","## Prospectively frozen gate outcomes", "",
            "Complete state, graph, actual Cython stopping, seeded replay and "
            "separate no-opt-in low-level arbitrary-matrix cache verification: PASS.",
            "CPU/wall/memory gate failures: "+str(len(verdicts)), ""]
    lines.extend("* "+failure for failure in verdicts)
    lines+=["","Native Cython CPU timing, not Valgrind simulated cache. "
            "No particle algorithm/RNG/stopping changes. Direct low-level arbitrary "
            "dominance retains original initializer by default. Do NOT merge the "
            "experiment; even CPU_GO would need separate minimal production PR."]
    Path("cython_canonical_rowmax_cpu_report.md").write_text(
        "\n".join(lines)+"\n",encoding="utf-8")
    print("\n".join(lines),flush=True)
    return report


if __name__=="__main__":
    check(sys.argv[1])
