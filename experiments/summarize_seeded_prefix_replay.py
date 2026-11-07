"""Require all frozen seeded-prefix replay cases and preserve caveats."""
import argparse
import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean

from experiments.audit_seeded_prefix_replay import (
    DATASETS, BACKENDS, SEEDS, MILESTONES, ARRAYS,
)

DIGEST = re.compile(r"^[0-9a-f]{64}$")


def validate(paths):
    if len(paths) != len(DATASETS) * len(BACKENDS):
        raise AssertionError("Need all nine dataset/backend CSV artifacts")
    rows = {}
    graphs, labeled = defaultdict(set), defaultdict(set)
    for path in paths:
        with open(path, encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                dataset, backend = row["dataset"],row["backend"]
                seed,iteration = int(row["seed"]),int(row["iteration"])
                if (dataset not in DATASETS or backend not in BACKENDS or
                        seed not in SEEDS or iteration not in MILESTONES):
                    raise AssertionError(f"Unexpected replay cohort {row}")
                key=(dataset,backend,seed,iteration)
                if key in rows:
                    raise AssertionError(f"Duplicate replay case {key}")
                if (row["actual_backend"] != BACKENDS[backend] or
                        row["reciprocal_graph"] != "True" or
                        row["requested_early_stop"] != "False" or
                        int(row["reported_iterations"]) != iteration or
                        row["reported_stop_reason"] != "max_iter" or
                        int(row["reported_checks"]) != 0 or
                        row["terminal_original_model_parity"] != "True"):
                    raise AssertionError(f"Graph/backend/no-stop/parity gate failed: {key}")
                classes=int(row["classes"])
                if not (int(row["particles"]) == 2*classes and
                        int(row["n_nodes"]) > 2*classes):
                    raise AssertionError(f"Wrong label cohort: {key}")
                accuracy=float(row["accuracy_unlabeled_retrospective"])
                potential=float(row["mean_max_dominance"])
                margin=float(row["mean_margin"])
                if not (0 <= accuracy <= 1 and math.isfinite(potential)
                        and math.isfinite(margin) and
                        float(row["elapsed_seconds_exploratory"])>0):
                    raise AssertionError(f"Nonfinite report values: {key}")
                hashes=json.loads(row["state_digests"])
                if (set(hashes) != set(ARRAYS)
                        or not all(isinstance(v,str) and DIGEST.fullmatch(v)
                                   for v in hashes.values())):
                    raise AssertionError(f"Invalid full-state digests: {key}")
                changes=row["class_changes_since_previous_sample"]
                if iteration==MILESTONES[0]:
                    if changes not in ("", "None"):
                        raise AssertionError("First sample has no prior sample")
                elif not changes.isdecimal() or int(changes)>int(row["n_nodes"]):
                    raise AssertionError(f"Invalid interval flip count: {key}")
                graphs[dataset].add(row["graph_digest"])
                labeled[(dataset,seed)].add(row["labels_digest"])
                rows[key]=row
    expected={
        (dataset,backend,seed,m)
        for dataset in DATASETS for backend in BACKENDS
        for seed in SEEDS for m in MILESTONES}
    if set(rows)!=expected or len(rows)!=90:
        raise AssertionError(
            f"Incomplete 90-case cohort: seen={len(rows)} "
            f"missing={len(expected-set(rows))}")
    for dataset in DATASETS:
        if len(graphs[dataset])!=1:
            raise AssertionError("Different graph across seed/backend replays")
        for seed in SEEDS:
            if len(labeled[(dataset,seed)])!=1:
                raise AssertionError("Different labeled sample across backends")
    # All samples are obtained from fresh independent fits. This test checks
    # reproducibility against the original high-level model at the final cap.
    # It does NOT prove an uninterrupted single-fit trajectory at every prefix.
    return rows


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--files", required=True, nargs="+")
    p.add_argument("--output", required=True)
    args=p.parse_args(argv)
    rows=validate(args.files)
    lines=[
        "# Clean-master seeded prefix replay: exploratory checkpoint audit",
        "",
        "**Coverage:** 90/90 checkpoint records; **18/18 exact six-array "
        "terminal parity checks against the ordinary unmodified "
        "`ParticleCompetitionAndCooperation.fit_predict` API PASS**. "
        "Three canonical undirected union-10NN datasets, three actual "
        "production backends, two fixed seeds, five milestones each. "
        "Each checkpoint is a **separate seeded fit**, not an observation "
        "from one uninterrupted run. All 108 fits retained in the CI audit.",
        "",
        "## Retrospective results at the 2,001-step cap",
        "",
        "| Dataset | Backend | Mean initially-unlabeled accuracy (2 seeds) | "
        "Observed label changes across the four sampled intervals | "
        "Intervals with any observed changes / total |",
        "|---|---|---:|---:|---:|",
    ]
    summaries={}
    for ds in DATASETS:
        for backend in BACKENDS:
            accuracy=[]
            flips=0
            changed_intervals=0
            for seed in SEEDS:
                accuracy.append(float(rows[(ds,backend,seed,MILESTONES[-1])][
                    "accuracy_unlabeled_retrospective"]))
                for cap in MILESTONES[1:]:
                    changes=int(rows[(ds,backend,seed,cap)][
                        "class_changes_since_previous_sample"])
                    flips+=changes
                    changed_intervals+=int(changes>0)
            summaries[(ds,backend)]=(mean(accuracy),flips,changed_intervals)
            lines.append(
                f"| {ds} | {backend} | {mean(accuracy):.4f} | "
                f"{flips} | {changed_intervals}/8 |")
    lines.extend([
        "",
        "## What the observations do and do not demonstrate",
        "",
        "- Each result has exact matching end state between a backend-info "
        "instrumented high-level dispatch and a fresh ordinary PCC model "
        "for the same last-cap seed; all other milestone fits use the "
        "same actual backend with `return_info=True`, no early stop, "
        "and requested/effective iteration counts equal.",
        "- Checkpoint differences are net changes **between the sampled "
        "iteration values**; a label may flip twice and return to its "
        "earlier class between snapshots without being counted. A zero "
        "is *not proof of convergence*, especially given previous "
        "independent long-horizon Iris flip evidence.",
        "- The sampling approach is an **independent seeded replay**, "
        "not PR #18's efficient continuous-run snapshot design and "
        "does not provide hidden Numba/Cython RNG state or same-fit "
        "step-by-step fingerprints. It costs `sum(milestones)` update "
        "iterations per seed/backend, plus one duplicated final fit. "
        "It cannot validate safety of potential/P10 stopping criteria "
        "without distinct prospective design.",
        "- Do not interpret accuracy differences between NumPy, Numba "
        "and Cython as matched stochastic paths: their RNG algorithms "
        "can differ despite a shared integer seed. Graph and labeled "
        "vertices are matched, not the native walk sequence.",
        "",
        "**Integration decision:** diagnostic code and CI only; no "
        "production backend edits. Keep historical PR #18 unmerged "
        "until its continuous native instrumentation is deliberately "
        "reimplemented and proven against current master. Do not "
        "discard its historical reports as if they were invalid, but "
        "do not reuse old test results as new parity evidence.",
        "",
    ])
    Path(args.output).write_text("\n".join(lines),encoding="utf-8")
    print("\n".join(lines[:21]),flush=True)


if __name__=="__main__":
    main()
