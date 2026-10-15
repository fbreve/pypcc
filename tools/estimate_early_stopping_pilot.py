"""Read historical timing artifacts; never load datasets or execute PCC."""
import argparse
import json
from pathlib import Path

DATASETS = ("wine", "digits", "germancredit", "satimage")
BACKENDS = ("numpy", "numba", "cython")


def estimate(root):
    cases = {f"{name}-labels{seed}" for name in DATASETS for seed in (10, 11)}
    rows = json.loads((root / "summary.json").read_text(encoding="utf-8"))
    selected = [r for r in rows if r["case_id"] in cases]
    expected = {(case, backend) for case in cases for backend in BACKENDS}
    if len(selected) != 24 or {(r["case_id"], r["backend"]) for r in selected} != expected:
        raise ValueError("Require all 24 historical fixture/backend groups")
    if any(r["mode"] != "fixed" or r["max_iter"] != 10000 for r in selected):
        raise ValueError("Require the original 10,000-iteration fixed baseline")
    groups = []
    for row in selected:
        groups.append(dict(case_id=row["case_id"], backend=row["backend"],
            propagation_cap_s={field: row["propagation_s"][field] * 50 * 3
                               for field in ("q1", "median", "q3", "maximum")},
            fit_overhead_s=3 * sum(row[field]["median"] for field in
                                  ("initialization_and_dispatch_s", "postprocessing_s"))))
    warmups = [json.loads(line) for line in (root / "warmups.jsonl").read_text().splitlines()]
    warmups = [r for r in warmups if r["case_id"] in cases]
    if len(warmups) != 48:
        raise ValueError("Require all 48 historical warm-up records")
    preparation = json.loads((root / "preparation.json").read_text())["fixtures"]
    prep_s = sum(r[field] for r in preparation if r["case_id"] in cases for field in
                 ("dataset_load_s", "preprocessing_and_labels_s", "graph_build_s"))
    helper_s = json.loads((root / "seed_helper.json").read_text())["first_call_s"]
    overhead = sum(g["fit_overhead_s"] for g in groups)
    warm_s = sum(r["fit_predict_s"] for r in warmups)
    totals = {field: sum(g["propagation_cap_s"][field] for g in groups) + overhead + warm_s + prep_s + helper_s
              for field in ("q1", "median", "q3", "maximum")}
    return dict(status="cap_work_projection", baseline=str(root), groups=groups,
                measured_fits=72, warmups=48, graph_builds=4,
                measured_iteration_cap=36000000, particle_update_cap=567000000,
                warmup_iterations=96, warmup_particle_updates=1512,
                preparation_s=prep_s, historical_warmup_fit_s=warm_s,
                historical_seed_helper_s=helper_s, fit_overhead_s=overhead,
                projected_accounted_s=totals,
                limitations="Cap-work linear extrapolation, not a runtime bound or confidence interval. No early-stop savings assumed. Excludes startup, copies, hashing, artifact I/O and stopping-check overhead; state distributions and machine load can change. Historical warm-ups are not new measurements.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(estimate(args.baseline), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
