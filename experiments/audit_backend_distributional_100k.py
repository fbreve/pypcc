"""Held-out, predeclared 100k baseline under production default stochastic movement.

See backend_distributional_100k_protocol.md. Reports data rather than assuming
same-seed cross-backend trajectories or declaring equivalence.
"""
import argparse
import csv

from experiments.audit_canonical_production_backend_stop import (
    FIELDS, BACKENDS, compare, warm_numba,
)

SEEDS = (4603, 4703, 4801, 4903, 5003, 5101)
DATASETS = ("iris", "wine", "breast_cancer", "digits")
CAP = 100000


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=DATASETS, required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    warm_numba(args.dataset)
    all_rows = []
    for seed in SEEDS:
        rows = compare(args.dataset, seed, CAP)
        if len(rows) != 2 * len(BACKENDS):
            raise AssertionError("Missing backend/policy results")
        for row in rows:
            if row["iterations"] <= 0 or row["iterations"] > CAP:
                raise AssertionError("Invalid propagation iterations")
            if row["policy"] == "production_no_stop" and row["iterations"] != CAP:
                raise AssertionError("No-stop reference did not reach full cap")
            if not 0 <= row["accuracy"] <= 1:
                raise AssertionError("Invalid accuracy")
            if not 0 <= row["agreement_with_own_no_stop"] <= 1:
                raise AssertionError("Invalid own-backend agreement")
        all_rows.extend(rows)
        print(f"{args.dataset} seed={seed}: "
              + ", ".join(
                  f"{r['backend']}/{r['policy']}={r['iterations']}"
                  for r in rows), flush=True)
    with open(args.output, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"{args.dataset}: saved {len(all_rows)} predeclared rows", flush=True)


if __name__ == "__main__":
    main()
