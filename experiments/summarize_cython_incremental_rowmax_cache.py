"""Audit all frozen Cython row-max cache A/B results; exclude nothing."""
import argparse
import csv
from pathlib import Path
from statistics import mean, median

from experiments.audit_cython_incremental_rowmax_cache import (
    DATASETS, SEEDS, POLICIES, VERIFY_CAP,
)

VARIANTS = ("legacy", "cache")


def validate(paths):
    if len(paths) != len(DATASETS):
        raise AssertionError("Require all four dataset matrix artifacts")
    rows = {}
    for path in paths:
        with open(path, newline="", encoding="utf-8") as file:
            for row in csv.DictReader(file):
                dataset = row["dataset"]
                seed = int(row["seed"])
                policy = row["policy"]
                variant = row["variant"]
                if (dataset not in DATASETS or seed not in SEEDS or
                        policy not in [item[0] for item in POLICIES] or
                        variant not in VARIANTS):
                    raise AssertionError(f"Unexpected experimental run {row}")
                key = (dataset, seed, policy, variant)
                if key in rows:
                    raise AssertionError(f"Duplicate {key}")
                if (row["all_state_arrays_bitwise_equal"] != "True" or
                        row["all_propagation_info_equal"] != "True" or
                        row["verified_metric_every_check"] != "True" or
                        int(row["verified_metric_checks"]) !=
                        (VERIFY_CAP + 9) // 10 or
                        row["dominance_fortran_contiguous"] != "True" or
                        abs(float(row["reciprocal_arc_fraction"]) - 1) > 1e-12):
                    raise AssertionError(f"Hard state/metric parity or graph failure {key}")
                rows[key] = row
    expected = {
        (ds,seed,policy[0],variant)
        for ds in DATASETS for seed in SEEDS for policy in POLICIES
        for variant in VARIANTS}
    if set(rows) != expected:
        raise AssertionError(f"Missing rows {expected - set(rows)}")
    if len(rows) != 64:
        raise AssertionError("Require all 64 timed fits")

    pairs = {}
    for dataset in DATASETS:
        for seed in SEEDS:
            for name, cap, es_chk in POLICIES:
                a = rows[(dataset, seed, name, "legacy")]
                b = rows[(dataset, seed, name, "cache")]
                if (int(a["cap"]) != cap or int(b["cap"]) != cap or
                        int(a["es_chk"]) != es_chk or
                        int(b["es_chk"]) != es_chk):
                    raise AssertionError("Wrong frozen cap or controls")
                for f in (
                    "effective_iterations", "stop_reason", "stalled_checks",
                    "stop_max", "checks", "accuracy",
                ):
                    if a[f] != b[f]:
                        raise AssertionError(f"Metadata/accuracy mismatch {f}")
                if int(a["order_index"]) == int(b["order_index"]):
                    raise AssertionError("Timing-order isolation failed")
                if (name == "forced_no_stop" and
                        int(a["effective_iterations"]) != cap):
                    raise AssertionError("Forced fixed workload stopped early")
                original = float(a["elapsed_seconds"])
                cached = float(b["elapsed_seconds"])
                if not (original > 0 and cached > 0):
                    raise AssertionError("Nonpositive fit time")
                savings = 100*(1-cached/original)
                for item in (a,b):
                    if abs(float(item["time_saved_percent"]) - savings) > 1e-8:
                        raise AssertionError("Inconsistent timing calculation")
                pairs[(dataset, seed, name)] = savings
    if len(pairs) != 32:
        raise AssertionError("Expected 32 exact paired comparisons")
    return pairs


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--files", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    args=parser.parse_args(argv)
    pairs=validate(args.files)
    summary={}
    text=[
        "# Cython cached row maxima: complete preregistered A/B result",
        "",
        "**Hard parity audit:** 4/4 datasets, 4/4 predefined seeds, "
        "2 stopping policies; **64/64 timed fits and 32/32 exact six-array "
        "state/full-propagation-info parity checks pass**. Before the timed "
        "comparison, 16 independent 2,000-step cached runs verified the "
        "cached mean maximum against the legacy full matrix scan at **all "
        "200 potential checkpoints each** (3,200 checks total); no excluded "
        "seeds/cases.",
        "",
        "## Whole-fit time saved by cache vs original metric",
        "",
        "| Dataset | Regime | Mean saving | Median | Min | Max |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for ds in DATASETS:
        for policy,cap,_ in POLICIES:
            values=[pairs[(ds,s,policy)] for s in SEEDS]
            summary[(ds,policy)] = dict(
                mean=mean(values),median=median(values),
                min=min(values),max=max(values))
            v=summary[(ds,policy)]
            text.append(f"| {ds} | {policy} | {v['mean']:+.2f}% | "
                        f"{v['median']:+.2f}% | {v['min']:+.2f}% | "
                        f"{v['max']:+.2f}% |")
    target = summary[("digits","forced_no_stop")]["median"]
    worst = min(
        summary[(ds,"forced_no_stop")]["median"] for ds in DATASETS)
    passes = target >= 10 and worst >= -5
    text.extend([
        "",
        "## Prespecified exploratory performance decision",
        "",
        f"- Median full-fit saving, **Digits, 20k forced monitoring:** "
        f"**{target:+.2f}%**, threshold >= +10%.",
        f"- Worst 20k per-dataset median: **{worst:+.2f}%**, "
        "minimum allowed -5%.",
        f"- Decision: **{'PASS: merits independent replication, no master merge' if passes else 'FAIL: discard candidate as tested'}**.",
        "",
        "## Caveats",
        "",
        "This is not a true O(1) global floating-point accumulator. "
        "Node maxima are incrementally refreshed from the **unchanged "
        "existing particle movement maximum**, and the global mean is "
        "formed by a canonical-order O(N) reduction at the same 10-step "
        "monitoring schedule. This prevents changes in floating-point "
        "summation order on the tested finite states. Verification mode "
        "independently compares **every visited check** with the full "
        "O(N*C) scan, outside the timed benchmark.",
        "",
        "Timings are one run per mode/seed on GitHub shared runners. "
        "Negative savings must be retained. The two policies include "
        "forced identical 20k work and real default potential stopping "
        "with 100k cap, both requiring exact stop metadata equality. "
        "No backend default changes or master merge are authorized "
        "without independent replication across platforms/layouts/"
        "edge cases, plus a full code safety review.",
        "",
    ])
    Path(args.output).write_text("\n".join(text), encoding="utf-8")
    print("\n".join(text[:22]),flush=True)


if __name__=="__main__":
    main()
