"""Recompute descriptive fixed-campaign tables from raw files, without fitting PCC."""
import argparse
import csv
import itertools
import json
import math
from pathlib import Path
import statistics

BACKENDS = ("numpy", "numba", "cython")
DATASETS = ("wine", "digits", "iris", "breast_cancer", "balancescale",
            "germancredit", "diabetes", "sonar", "glass", "vehicle", "ionosphere",
            "pendigits", "satimage", "spambase", "pageblocks", "banknote",
            "bloodtransfusion", "yeast", "segment", "qsarbiodeg")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def quartile(values, fraction):
    ordered = sorted(values)
    index = (len(ordered) - 1) * fraction
    low = int(index)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (index - low)


def describe(values):
    return {"count": len(values), "median": statistics.median(values),
            "q1": quartile(values, .25), "q3": quartile(values, .75),
            "iqr": quartile(values, .75) - quartile(values, .25),
            "min": min(values), "max": max(values)}


def audit_run(root, datasets, seeds, permutations=False, original=None):
    root = Path(root)
    cases = {f"{dataset}-labels{seed}" for dataset in datasets for seed in seeds}
    completion = read_json(root / "completion.json")
    if completion["status"] != "complete" or completion["measured_executions"] != len(cases) * 18:
        raise ValueError("Incomplete run")
    fixtures = {item["case_id"]: item["fixture_sha256"] for item in completion["fixtures"]}
    case_indices = {item["case_id"]: index for index, item in enumerate(completion["fixtures"])}
    if set(fixtures) != cases or len(completion["fixtures"]) != len(cases):
        raise ValueError("Fixture inventory differs")
    for case in cases:
        fixture = read_json(root / "fixtures" / case / "fixture.json")
        if fixture["fixture_sha256"] != fixtures[case]:
            raise ValueError("Fixture hash differs")
        fingerprints = fixture["source"]["frozen_fingerprints"]
        if not fingerprints or any(not item["matches"] or item["actual"] != item["expected"]
                                   for item in fingerprints.values()):
            raise ValueError("Frozen fingerprint mismatch")
        if set(fixture["labeled_class_counts"].values()) != {3}:
            raise ValueError("Label coverage differs")
    indexed = {}
    warm_count = 0
    for file, budget, repetitions in [("measurements.jsonl", 10000, range(6)),
                                      ("warmups.jsonl", 2, ("first", "warm"))]:
        rows = [json.loads(line) for line in (root / file).read_text(encoding="utf-8").splitlines()]
        keys = [(r["case_id"], r["backend"], r["repeat"] if budget == 10000 else r["call"])
                for r in rows]
        expected = set(itertools.product(cases, BACKENDS, repetitions))
        if len(keys) != len(expected) or set(keys) != expected:
            raise ValueError(f"Missing or duplicate records: {file}")
        states = {}
        for row in rows:
            case, backend = row["case_id"], row["backend"]
            if (row["mode"] != "fixed" or row["effective_iterations"] != budget
                    or row["max_iter"] != budget or row["checks"] != 0
                    or row["fixture_sha256"] != fixtures[case]
                    or not row["backend_identity"].startswith({"numpy": "pcc_numpy.",
                            "numba": "pcc_numba.", "cython": "pcc_step."}[backend])):
                raise ValueError("Protocol/backend identity differs")
            for timer in ("initialization_and_dispatch_s", "propagation_s",
                          "postprocessing_s", "fit_predict_s"):
                if not math.isfinite(row[timer]) or row[timer] <= 0:
                    raise ValueError("Invalid timing")
            state = (row["state_hashes"], row["accuracy_unlabeled"])
            if states.setdefault((case, backend), state) != state:
                raise ValueError("Within-backend reproducibility failure")
            if budget == 10000:
                key = (case, backend, row["repeat"])
                indexed[key] = row
                if original is not None:
                    reference = original[key]
                    if (row["fixture_sha256"] != reference["fixture_sha256"]
                            or state != (reference["state_hashes"], reference["accuracy_unlabeled"])):
                        raise ValueError("Follow-up differs from original fixture/state")
        if budget == 2:
            warm_count = len(rows)
    for case in cases:
        orders = []
        for repeat in range(6):
            rows = [indexed[case, backend, repeat] for backend in BACKENDS]
            order = tuple(r["backend"] for r in sorted(rows, key=lambda r: r["order_index"]))
            if sorted(r["order_index"] for r in rows) != [0, 1, 2]:
                raise ValueError("Invalid execution positions")
            if permutations and any(tuple(r["execution_order"]) != order for r in rows):
                raise ValueError("Recorded execution order differs")
            orders.append(order)
        if permutations:
            if set(orders) != set(itertools.permutations(BACKENDS)):
                raise ValueError("Permutation schedule differs")
        elif orders != [BACKENDS[(case_indices[case] + r) % 3:] +
                        BACKENDS[:(case_indices[case] + r) % 3] for r in range(6)]:
            raise ValueError("Original rotation schedule differs")
    tables = []
    for dataset in datasets:
        subset = {key: row for key, row in indexed.items() if key[0].rsplit("-labels", 1)[0] == dataset}
        for backend in BACKENDS:
            stats = describe([r["propagation_s"] for key, r in subset.items() if key[1] == backend])
            tables.append({"dataset": dataset, "backend": backend, **stats})
        for numerator, denominator in [("numpy", "numba"), ("numpy", "cython"), ("numba", "cython")]:
            ratios = [row["propagation_s"] / indexed[case, denominator, repeat]["propagation_s"]
                      for (case, backend, repeat), row in subset.items() if backend == numerator]
            tables.append({"dataset": dataset, "backend": f"{numerator}/{denominator}", **describe(ratios)})
    return indexed, {"measurements": len(indexed), "warmups": warm_count,
                     "fixtures": len(cases), "tables": tables, "outliers_excluded": 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--primary", type=Path, required=True)
    parser.add_argument("--sensitivity", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    primary, results = audit_run(args.primary / "fixed-10000", DATASETS, range(10, 20))
    reports = {"primary": results}
    for session in range(1, 4):
        name = f"session-{session}"
        _, reports[name] = audit_run(args.sensitivity / name,
                                    ("wine", "digits", "satimage", "pageblocks"), (10, 11),
                                    permutations=True, original=primary)
    args.output.mkdir(parents=True)
    (args.output / "analysis.json").write_text(json.dumps(reports, indent=2), encoding="utf-8")
    for name, report in reports.items():
        with (args.output / f"{name}.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=report["tables"][0].keys())
            writer.writeheader()
            writer.writerows(report["tables"])
    print(json.dumps({name: {key: value for key, value in report.items() if key != "tables"}
                      for name, report in reports.items()}, indent=2))


if __name__ == "__main__":
    main()
