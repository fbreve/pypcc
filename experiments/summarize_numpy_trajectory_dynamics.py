"""Require all eight preregistered native trajectory triples, without omissions."""
import argparse
import json
from pathlib import Path

DATASETS=("iris","wine","breast_cancer","digits")
SEEDS=(18013,18109)
FLAGS=("final_six_arrays_exact","final_stop_info_exact",
       "final_full_rng_exact","final_next16_exact",
       "true_native_prefix_exact","first_shadow_stop_exact")


def enforce(files):
    cases={}
    datasets_seen=set()
    for filename in files:
        document=json.loads(Path(filename).read_text(encoding="utf-8"))
        dataset=document.get("dataset")
        if dataset not in DATASETS or dataset in datasets_seen:
            raise AssertionError(f"Wrong/repeated dataset report: {dataset}")
        datasets_seen.add(dataset)
        if document.get("cap")!=6001 or document.get("window")!=100:
            raise AssertionError("Pilot cohort/horizon/window changed")
        if document.get("seeds")!=list(SEEDS):
            raise AssertionError("Pilot seeds changed")
        rows=document.get("cases",[])
        if len(rows)!=2:
            raise AssertionError(f"Wrong number of seed results: {dataset}")
        for case in rows:
            key=(dataset,case["seed"])
            if key in cases or key not in {
                    (d,s) for d in DATASETS for s in SEEDS}:
                raise AssertionError(f"Duplicate/unregistered case: {key}")
            if any(case.get(flag) is not True for flag in FLAGS):
                raise AssertionError(f"Exact state/stop/RNG violation: {key}")
            info=case["native_stop_info"]
            stop=case["stop_iteration"]
            if stop!=info["effective_iterations"] or not 0<stop<=6001:
                raise AssertionError(f"Invalid true stop metadata: {key}")
            if case["stop_reason"] not in ("early_stop","max_iter"):
                raise AssertionError(f"Unknown stop reason: {key}")
            if (case["stop_reason"]=="max_iter" and
                    (stop!=6001 or
                     case["classification_disagreement_after_continuation"]!=0)):
                raise AssertionError(f"Nonstopping run unexpectedly diverged: {key}")
            metrics=case["dynamics"]
            if (metrics["native_callbacks"]!=6002 or
                    metrics["full_windows"]!=60 or
                    len(metrics["windows"])!=60 or
                    len(metrics["particle_max_streaks"])!=case["n_particles"]):
                raise AssertionError(f"Missing every-iteration native trace: {key}")
            if any(w["start"]!=100*i or w["end"]!=100*(i+1)
                   for i,w in enumerate(metrics["windows"])):
                raise AssertionError(f"Wrong disjoint window ranges: {key}")
            if (not 0<=metrics["quiet_but_hidden_windows"]<=
                    metrics["quiet_endpoint_windows"]<=60):
                raise AssertionError(f"Invalid hidden-reversal accounting: {key}")
            if (not 0<=metrics["zero_motion_steps"]<=6001 or
                    not 0<=metrics["particles_never_changed_position"]<=
                    case["n_particles"] or
                    not 0<=metrics["longest_stationary_steps"]<=6001 or
                    metrics["longest_stationary_steps"]!=max(
                        metrics["particle_max_streaks"],default=0) or
                    any(not 0<=x<=6001
                        for x in metrics["particle_max_streaks"]) or
                    not 0<=metrics["particles_final_stationary_ge_100"]<=
                        case["n_particles"]):
                raise AssertionError(f"Invalid movement accounting: {key}")
            if (case["stop_reason"]=="early_stop" and
                    metrics["shadow_first_stop_iteration"]!=stop):
                raise AssertionError(f"Native stop not recovered from trace: {key}")
            if (case["stop_reason"]=="max_iter" and
                    metrics["shadow_first_stop_iteration"] is not None):
                raise AssertionError(f"False potential trigger: {key}")
            if not (0<=case["stop_unlabeled_accuracy"]<=1 and
                    0<=case["continued_unlabeled_accuracy"]<=1):
                raise AssertionError(f"Invalid accuracy: {key}")
            if any(x<=0 for x in case["times_seconds"].values()):
                raise AssertionError(f"Nonpositive timing: {key}")
            cases[key]=case
    expected={(d,s) for d in DATASETS for s in SEEDS}
    if set(cases)!=expected:
        raise AssertionError(
            f"Incomplete preregistered pilot: missing={sorted(expected-set(cases))}")
    return cases


def write_summary(cases):
    lines=[
        "# True NumPy PCC per-iteration trajectory pilot — 8/8 triples validated",
        "",
        "Linux/Python3.12; 4 reciprocal union-10NN datasets × 2 fresh seeds;",
        "8 ordinary no-stop + 8 real potential-stop + 8 observed no-stop",
        "full native fits, cap 6001. The 6002 physical per-iteration records",
        "come from ONE uninterrupted observed fit per case, not prefix replays.",
        "",
        "| Dataset | Seed | Potential stop | Endpoint-quiet windows (of 60) | Quiet with hidden reversals | Reverted node-window events | Never-moved particles | Longest stationary streak | Steps with no particle moving | Changed unlabeled outputs after continuation | Accuracy stop → continue |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for key,c in sorted(cases.items()):
        d=c["dynamics"]
        reason="early" if c["stop_reason"]=="early_stop" else "max"
        lines.append(
            f"| {key[0]} | {key[1]} | {reason} at {c['stop_iteration']} | "
            f"{d['quiet_endpoint_windows']} | {d['quiet_but_hidden_windows']} | "
            f"{d['reverted_node_window_events']} | "
            f"{d['particles_never_changed_position']} / {c['n_particles']} | "
            f"{d['longest_stationary_steps']} | "
            f"{d['zero_motion_steps']} / 6001 | "
            f"{c['classification_disagreement_after_continuation']} | "
            f"{c['stop_unlabeled_accuracy']:.4f} → "
            f"{c['continued_unlabeled_accuracy']:.4f} |")
    lines.extend([
        "",
        "All 8 observed and unobserved 6001-step runs agree on six exact final",
        "arrays, native stopping metadata, complete MT19937 state, and next 16 draws.",
        "At the actual stop time, all six raw native arrays and RNG equal",
        "the uninterrupted observed prefix. The shadow potential check fires",
        "at the same actual iteration (or never), with no new stopping policy.",
        "",
        "**Definitions:** Endpoint-quiet means all labels match at the start/end",
        "of a *fixed* 100-step window; hidden reversal means some node changed",
        "within and reverted by the end. A never-moved particle may still have",
        "attempted and rejected moves—position-only data cannot distinguish.",
        "Counts across 60 disjoint windows are node-window events, not unique",
        "nodes over the full trajectory. The 6001st transition is included in",
        "step totals but not the 60 complete windows.",
        "",
        "No causal claim or accuracy advantage is inferred from this small",
        "descriptive pilot. These are **aggressive es_chk=10** stop settings,",
        "not production recommendations. Draft dependent PR only.",
    ])
    return "\n".join(lines)+"\n"


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--files",nargs="+",required=True)
    p.add_argument("--output",required=True)
    args=p.parse_args()
    cases=enforce(args.files)
    output=write_summary(cases)
    Path(args.output).write_text(output,encoding="utf-8")
    print(output,flush=True)


if __name__=="__main__":
    main()
