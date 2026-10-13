"""Cache-only frozen input audit; constructs graphs but never propagates particles."""
import argparse
from importlib.metadata import version
import json
from pathlib import Path

from .fixtures import Case, REAL_DATASETS, prepare_fixture
from .historical import ENTRIES, FROZEN, INVENTORY, fingerprint


def audit(names, cache):
    rows = []
    for name in names:
        try:
            arrays, metadata = prepare_fixture(Case(name, *REAL_DATASETS[name][:3], label_seed=10),
                                               historical_cache=cache)
            checks = {kind: dict(expected=FROZEN[name][key],
                                 actual=fingerprint(*(arrays[field] for field in fields)))
                      for kind, key, fields in (
                          ("raw", "raw_data_fingerprint", ("raw", "truth")),
                          ("processed", "processed_data_fingerprint", ("x", "truth")),
                          ("ordered_graph", "graph_fingerprint", ("neighbors", "degrees")))}
            rows.append(dict(dataset=name, status="match", checks=checks))
        except (ValueError, FileNotFoundError, OSError) as exc:
            rows.append(dict(dataset=name, status="failed", error=str(exc)))
    return dict(schema=1, status="match" if all(r["status"] == "match" for r in rows) else "failed",
                purpose="Input fingerprint validation only; no downloads, propagation or performance measurements",
                environment={pkg: version(pkg) for pkg in ("numpy", "scipy", "scikit-learn")},
                frozen_environment={key: INVENTORY["environment"][key] for key in ("numpy", "scipy", "sklearn")},
                datasets=rows)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", choices=ENTRIES, default=list(ENTRIES))
    parser.add_argument("--historical-cache", type=Path)
    parser.add_argument("--output", type=Path, help="optional new report file; existing files refused")
    args = parser.parse_args(argv)
    # Reserve output before any potentially costly input preparation.
    stream = args.output.open("x", encoding="utf-8") if args.output else None
    try:
        report = audit(args.datasets, args.historical_cache)
        if stream:
            json.dump(report, stream, indent=2, allow_nan=False)
        for row in report["datasets"]:
            print(row["dataset"], row["status"], row.get("error", "raw, processed and ordered graph match"))
        if report["status"] != "match":
            raise SystemExit(1)
    finally:
        if stream:
            stream.close()


if __name__ == "__main__":
    main()
