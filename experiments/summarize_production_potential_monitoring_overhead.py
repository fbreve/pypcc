"""Validate all 96 timed fits and report descriptive monitoring overhead."""
import argparse
import csv
from pathlib import Path
from statistics import mean, median

from experiments.audit_production_potential_monitoring_overhead import (
    DATASETS, SEEDS, BACKENDS, CAP,
)

EXPECTED_MODES = ("metric_off", "metric_on_forced_no_stop")


def read_complete_cohort(paths):
    if len(paths) != len(DATASETS):
        raise AssertionError("Exactly four dataset artifacts are required")
    fits = {}
    for path in paths:
        with open(path, encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                dataset, seed, backend, mode = (
                    row["dataset"], int(row["seed"]),
                    row["backend"], row["mode"],
                )
                if (dataset not in DATASETS or seed not in SEEDS
                        or backend not in BACKENDS
                        or mode not in EXPECTED_MODES):
                    raise AssertionError(f"Unexpected run: {row}")
                key = (dataset, seed, backend, mode)
                if key in fits:
                    raise AssertionError(f"Duplicate run: {key}")
                if (int(row["cap"]) != CAP or
                        int(row["effective_iterations"]) != CAP or
                        row["all_six_state_arrays_bitwise_equal"] != "True" or
                        row["can_not_stop_early"] != "True" or
                        abs(float(row["reciprocal_arc_fraction"]) - 1) > 1e-12):
                    raise AssertionError(f"Failed no-stop/graph/state gate: {key}")
                fits[key] = row
    expected = {
        (d, s, b, m)
        for d in DATASETS for s in SEEDS for b in BACKENDS
        for m in EXPECTED_MODES
    }
    if set(fits) != expected:
        raise AssertionError(
            f"Missing {len(expected - set(fits))} experiment rows")
    if len(fits) != 96:
        raise AssertionError("Expected exactly 96 records")

    pairs = {}
    for d in DATASETS:
        for s in SEEDS:
            for b in BACKENDS:
                base = fits[(d, s, b, EXPECTED_MODES[0])]
                monitor = fits[(d, s, b, EXPECTED_MODES[1])]
                if int(base["order_index"]) == int(monitor["order_index"]):
                    raise AssertionError("Paired modes must run separately")
                off, on = (float(base["elapsed_seconds"]),
                           float(monitor["elapsed_seconds"]))
                if not 0 < off and not 0 < on:
                    raise AssertionError("Invalid wall time")
                overhead = 100*(on/off - 1)
                share = 100*(on-off)/on
                for row in (base, monitor):
                    if (abs(float(row["overhead_percent"]) - overhead) > 1e-8
                            or abs(float(row["monitor_cost_share_percent"])
                                   - share) > 1e-8):
                        raise AssertionError("Corrupted pair timing fields")
                pairs[(d,s,b)] = (overhead, share)
    if len(pairs) != 48:
        raise AssertionError("Expected 48 exact-state monitored pairs")
    return pairs


def summary(values):
    return (mean(values), median(values), min(values), max(values))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--files", required=True, nargs="+")
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    pairs = read_complete_cohort(args.files)

    lines = [
        "# Measured production potential-monitoring overhead — full cohort",
        "",
        "**Complete output integrity:** 4/4 datasets, 4 seeds, 3 actual "
        "backends, 2 modes = **96/96 fits**, and **48/48 exact six-array "
        "full-state equality gates PASS**. No backend fallback, invalid graph "
        "or early termination. All potential monitoring used an artificially "
        "large stop counter to force the full **20k iterations**; "
        "this is a controlled overhead measurement, **not an altered stopping policy**.",
        "",
        "## Paired overhead by backend",
        "",
        "| Backend | Matched pairs | Mean on/off overhead | Median overhead | "
        "Minimum | Maximum | Mean fraction of monitored time attributable to checks |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for b in BACKENDS:
        values = [v[0] for (d,s,be),v in pairs.items() if be==b]
        shares = [v[1] for (d,s,be),v in pairs.items() if be==b]
        avg,med,lo,hi = summary(values)
        lines.append(f"| {b} | {len(values)} | {avg:+.2f}% | "
                     f"{med:+.2f}% | {lo:+.2f}% | {hi:+.2f}% | "
                     f"{mean(shares):+.2f}% |")
    lines.extend(["", "## Detailed dataset/backend sensitivity", "",
                  "| Dataset | Backend | Mean overhead | Median | Range |",
                  "|---|---|---:|---:|---:|"])
    for d in DATASETS:
        for b in BACKENDS:
            values = [pairs[(d,s,b)][0] for s in SEEDS]
            avg,med,lo,hi=summary(values)
            lines.append(f"| {d} | {b} | {avg:+.2f}% | {med:+.2f}% | "
                         f"[{lo:+.2f}, {hi:+.2f}]% |")

    lines.extend([
        "", "## Interpretation", "",
        "The benchmark forced both conditions through identical **20,000** "
        "production particle-update steps and verified exact equality of "
        "predictions, particle positions, particle strength, particle distance "
        "tables, node dominance and normalized ownership. Thus the measured "
        "difference cannot be attributed to stop-induced changes in the "
        "stochastic trajectory on these inputs. It includes the cost of "
        "existing potential checks and their effects on cache/loop behavior.",
        "",
        "These are **single paired CI timings across four fixed seeds** and "
        "are susceptible to host contention/noise; negative observed overheads "
        "must remain in the record. They do not quantify default end-to-end "
        "stopping benefit or guarantee identical acceleration on other CPUs.",
        "",
        "This is **diagnostic only**. No change to backend code, numerical "
        "metric, monitoring schedule, thresholds or production defaults is "
        "warranted before a separate experimental optimization and strict "
        "stop-iteration/full-state regression parity suite.",
        "",
    ])
    Path(args.output).write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines[:15]), flush=True)


if __name__ == "__main__":
    main()
