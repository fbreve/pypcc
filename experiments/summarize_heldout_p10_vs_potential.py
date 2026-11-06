"""Summarize the frozen 48-case held-out comparison without excluding outliers.

Only Python stdlib is required; exact eight matrix CSVs are mandatory.
The report is descriptive, never a statistical-equivalence declaration.
"""
import argparse
import csv
from pathlib import Path
from statistics import mean, median

from experiments.audit_heldout_p10_vs_potential import (
    DATASETS, SEEDS, SUPERVISION, POLICIES,
)


def read_cases(paths):
    cases = {}
    seen = set()
    if len(paths) != len(DATASETS) * len(SUPERVISION):
        raise AssertionError("Require all eight dataset/supervision artifacts")
    for path in paths:
        with open(path, newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                dataset = row["dataset"]
                labeled = int(row["labeled_per_class"])
                seed = int(row["seed"])
                policy = row["policy"]
                if (dataset not in DATASETS or labeled not in SUPERVISION
                        or seed not in SEEDS or policy not in POLICIES):
                    raise AssertionError(f"Unexpected condition: {row}")
                key = (dataset, labeled, seed, policy)
                if key in seen:
                    raise AssertionError(f"Duplicate row {key}")
                seen.add(key)
                if row["candidate_production_10k_no_stop_parity"] != "True":
                    raise AssertionError(f"Candidate parity FAILED in {key}")
                if abs(float(row["reciprocal_arc_fraction"]) - 1.0) > 1e-12:
                    raise AssertionError(f"Directed adjacency in {key}")
                cases.setdefault((dataset, labeled, seed), {})[policy] = row
    expected = {(ds, n, s) for ds in DATASETS for n in SUPERVISION for s in SEEDS}
    if set(cases) != expected:
        raise AssertionError(f"Missing case(s): {expected - set(cases)}")
    for case, policy_rows in cases.items():
        if set(policy_rows) != set(POLICIES):
            raise AssertionError(f"Missing policies {case}")
        a, b = policy_rows["no_stop_100k"], policy_rows["no_stop_200k"]
        if int(a["effective_iterations"]) != 100000 or (
                int(b["effective_iterations"]) != 200000):
            raise AssertionError(f"Reference cap mismatch in {case}")
        delta = float(policy_rows["experimental_label_p10"][
            "accuracy_p10_minus_potential_pp"])
        raw_delta = 100 * (
            float(policy_rows["experimental_label_p10"]["accuracy"]) -
            float(policy_rows["potential_default"]["accuracy"]))
        if abs(delta - raw_delta) > 1e-8:
            raise AssertionError(f"Inconsistent accuracy difference {case}")
    if len(cases) != 48 or len(seen) != 192:
        raise AssertionError("Expected exactly 48 cases, 192 policy rows")
    return cases


def summarize(group):
    pairs = []
    for policy_rows in group:
        potential = policy_rows["potential_default"]
        p10 = policy_rows["experimental_label_p10"]
        reference = policy_rows["no_stop_200k"]
        pairs.append(dict(
            direct_delta=100*(float(p10["accuracy"]) - float(potential["accuracy"])),
            potential_delta=100*(float(potential["accuracy"]) - float(reference["accuracy"])),
            p10_delta=100*(float(p10["accuracy"]) - float(reference["accuracy"])),
            potential_stop=potential["stopped_early"] == "True",
            p10_stop=p10["stopped_early"] == "True",
            potential_saved=float(potential["saved_iterations_fraction_vs_200k"]),
            p10_saved=float(p10["saved_iterations_fraction_vs_200k"]),
            potential_time_saved=float(potential["saved_wall_time_fraction_vs_200k"]),
            p10_time_saved=float(p10["saved_wall_time_fraction_vs_200k"]),
            potential_agree=float(potential["agreement_with_200k"]),
            p10_agree=float(p10["agreement_with_200k"]),
            ref_flips=int(reference["reference_100k_200k_prediction_flips"]),
        ))
    deltas = [p["direct_delta"] for p in pairs]
    return dict(
        n=len(pairs), mean_delta=mean(deltas), median_delta=median(deltas),
        min_delta=min(deltas), max_delta=max(deltas),
        p10_better=sum(x > 1e-9 for x in deltas),
        potential_better=sum(x < -1e-9 for x in deltas),
        ties=sum(abs(x) <= 1e-9 for x in deltas),
        potential_stop=sum(p["potential_stop"] for p in pairs),
        p10_stop=sum(p["p10_stop"] for p in pairs),
        potential_saved=mean(p["potential_saved"] for p in pairs),
        p10_saved=mean(p["p10_saved"] for p in pairs),
        potential_time_saved=mean(p["potential_time_saved"] for p in pairs),
        p10_time_saved=mean(p["p10_time_saved"] for p in pairs),
        potential_agree=mean(p["potential_agree"] for p in pairs),
        p10_agree=mean(p["p10_agree"] for p in pairs),
        potential_worst_ref=min(p["potential_delta"] for p in pairs),
        p10_worst_ref=min(p["p10_delta"] for p in pairs),
        p10_large_ref_loss=sum(p["p10_delta"] < -2 for p in pairs),
        potential_large_ref_loss=sum(p["potential_delta"] < -2 for p in pairs),
        p10_large_pair_loss=sum(p["direct_delta"] < -2 for p in pairs),
        potential_large_pair_loss=sum(p["direct_delta"] > 2 for p in pairs),
        drift_cases=sum(p["ref_flips"] > 0 for p in pairs),
        total_ref_flips=sum(p["ref_flips"] for p in pairs),
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--files", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    cases = read_cases(args.files)
    lines = [
        "# Preregistered held-out potential vs P10: generated descriptive report",
        "",
        "**Data integrity:** 8/8 dataset/supervision artifacts, 48/48 cases, "
        "192/192 output rows, no dropped seeds, all candidate 10k parity gates PASS.",
        "",
        "Scope: four canonical undirected kNN datasets, 2 or 5 labels/class, "
        "six previously unused seeds (5407,5501,5609,5701,5801,5903), "
        "production NumPy potential vs experimental NumPy label P10, "
        "200k reference cap. **A 200k reference is not convergence or optimality.**",
        "",
        "## Paired accuracy and stopping by dataset × supervision",
        "",
        "| Dataset | Labels/class | Cases | P10 better / potential better / tied | "
        "Mean P10−potential (pp) | Median (pp) | Worst P10−potential (pp) | "
        "Potential stops | P10 stops |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for dataset in DATASETS:
        for labeled in SUPERVISION:
            subset = [cases[(dataset,labeled,s)] for s in SEEDS]
            x = summarize(subset)
            lines.append(
                f"| {dataset} | {labeled} | {x['n']} | "
                f"{x['p10_better']}/{x['potential_better']}/{x['ties']} | "
                f"{x['mean_delta']:+.3f} | {x['median_delta']:+.3f} | "
                f"{x['min_delta']:+.3f} | {x['potential_stop']}/{x['n']} | "
                f"{x['p10_stop']}/{x['n']} |"
            )
    overall = summarize(list(cases.values()))
    lines.extend([
        "",
        "## Overall descriptive totals (not an inferential test)",
        "",
        f"- Matched cases: **{overall['n']}**; P10 better **{overall['p10_better']}**, "
        f"potential better **{overall['potential_better']}**, "
        f"ties **{overall['ties']}**.",
        f"- P10−potential accuracy: mean **{overall['mean_delta']:+.3f} pp**, "
        f"median **{overall['median_delta']:+.3f} pp**, "
        f"range **[{overall['min_delta']:+.3f}, {overall['max_delta']:+.3f}] pp**.",
        f"- Production potential: early **{overall['potential_stop']}/{overall['n']}**, "
        f"average iteration saving **{100*overall['potential_saved']:.2f}%**, "
        f"mean wall-time saving **{100*overall['potential_time_saved']:.2f}%**, "
        f"mean 200k-label agreement **{100*overall['potential_agree']:.2f}%**.",
        f"- Experimental P10: early **{overall['p10_stop']}/{overall['n']}**, "
        f"average iteration saving **{100*overall['p10_saved']:.2f}%**, "
        f"mean wall-time saving **{100*overall['p10_time_saved']:.2f}%**, "
        f"mean 200k-label agreement **{100*overall['p10_agree']:.2f}%**.",
        f"- Worst accuracy loss vs own 200k no-stop: "
        f"potential **{overall['potential_worst_ref']:+.3f} pp**, "
        f"P10 **{overall['p10_worst_ref']:+.3f} pp**.",
        f"- Cases with >2 pp loss vs 200k: potential "
        f"**{overall['potential_large_ref_loss']}**, "
        f"P10 **{overall['p10_large_ref_loss']}**.",
        f"- Direct paired >2 pp loss: P10 relative to potential "
        f"**{overall['p10_large_pair_loss']}**, "
        f"potential relative to P10 **{overall['potential_large_pair_loss']}**.",
        f"- 100k→200k no-stop label drift: "
        f"**{overall['drift_cases']}/{overall['n']}** cases, "
        f"**{overall['total_ref_flips']}** changed predictions in total "
        "(node counts differ by dataset/label density).",
        "",
        "## Largest paired accuracy differences (no exclusions)",
        "",
        "| Dataset | Labels/class | Seed | Potential accuracy | P10 accuracy | "
        "P10−potential (pp) | Potential steps | P10 steps | 100k→200k flips |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    ordered = sorted(
        cases.items(),
        key=lambda item: abs(
            float(item[1]["experimental_label_p10"]["accuracy"]) -
            float(item[1]["potential_default"]["accuracy"])),
        reverse=True,
    )
    for (dataset, labeled, seed), policies in ordered[:12]:
        p = policies["potential_default"]
        q = policies["experimental_label_p10"]
        lines.append(
            f"| {dataset} | {labeled} | {seed} | "
            f"{100*float(p['accuracy']):.3f}% | "
            f"{100*float(q['accuracy']):.3f}% | "
            f"{100*(float(q['accuracy'])-float(p['accuracy'])):+.3f} | "
            f"{p['effective_iterations']} | {q['effective_iterations']} | "
            f"{q['reference_100k_200k_prediction_flips']} |"
        )
    lines.extend([
        "",
        "## Methodological limits and decision",
        "",
        "This is a **fixed 48-case descriptive, stratified study**, not an "
        "equivalence or significance test; six seeds per stratum and four "
        "datasets do not support universal conclusions. Signed deltas can "
        "cancel large losses; inspect outliers and per-stratum results. "
        "200k no-stop labels may drift later and can be less accurate than "
        "early-stopped labels. Accuracy uses true labels on initially unlabeled "
        "nodes, while final-label agreement measures only reference fidelity. "
        "Wall-times are one-off CI timings and not calibrated benchmarks. "
        "The two supervision densities use different selected labels, not "
        "paired causal treatment units. **Do not merge a production stopping "
        "change or adjust any stopping thresholds based on these results.**",
        "",
    ])
    Path(args.output).write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines[:27]), flush=True)


if __name__ == "__main__":
    main()
