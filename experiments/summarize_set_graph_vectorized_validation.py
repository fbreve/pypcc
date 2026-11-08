"""Strict two-host aggregation; scientific completeness is independent of GO/NO-GO."""
import json
import math
from pathlib import Path
import statistics
import sys

SIZES = {(150, 20), (569, 25), (1797, 35),
         (30000, 35), (30000, 100), (100000, 35)}
BASE = "ababc48986b40f083d5930e922f844aa2ea418b2"


def check(data):
    assert data["schema"] == 1 and data["baseline"] == BASE
    assert data["seed"] == 34007 and data["adverse_count"] == 8
    assert len(data["cases"]) == 6 and len(data["native"]) == 24
    assert {(c["nodes"], c["width"]) for c in data["cases"]} == SIZES
    assert {(v["dataset"], v["source"], v["backend"]) for v in data["native"]} == {
        (d, source, b) for d in ("Iris", "Wine", "Breast Cancer", "Digits")
        for source in ("knn", "edge_index")
        for b in ("numpy", "numba", "cython")}
    for v in data["native"]:
        assert len(v["state_hashes"]) == 6 and len(v["graph_hash"]) == 64
        assert all(len(k) == 64 for k in v["state_hashes"])
    for c in data["cases"]:
        assert len(c["timings_ms"]) == 25 and len(c["traced_peaks_bytes"]) == 5
        for p in c["timings_ms"] + c["traced_peaks_bytes"]:
            assert len(p) == 2 and all(math.isfinite(z) and z > 0 for z in p)
        assert len(c["graph_sha"]) == 64 and len(c["degree_sha"]) == 64


def main(root):
    paths = list(root.glob("**/set_graph_validation_result.json"))
    assert len(paths) == 2, f"Expected exactly 2 OS artifacts, got {paths}"
    reports = [json.loads(p.read_text(encoding="utf-8")) for p in paths]
    assert {r["platform"] for r in reports} == {"linux", "win32"}
    for report in reports:
        check(report)
    assert len({r["original_blob_sha256"] for r in reports}) == 1
    by_os = {r["platform"]: {(c["nodes"], c["width"]): c for c in r["cases"]}
             for r in reports}
    for size in SIZES:
        assert len({by_os[os][size]["graph_sha"] for os in by_os}) == 1
        assert len({by_os[os][size]["degree_sha"] for os in by_os}) == 1

    rows, go = [], True
    print("| OS | Nodes x width | Paired CPU saving | Traced-peak saving |")
    print("|---|---|---:|---:|")
    for os in ("linux", "win32"):
        for n, w in sorted(SIZES):
            c = by_os[os][(n, w)]
            cpu = 100 * (1 - statistics.median([b/a for a, b in c["timings_ms"]]))
            mem = 100 * (1 - statistics.median([b/a for a, b in c["traced_peaks_bytes"]]))
            print(f"| {os} | {n} x {w} | {cpu:+.2f}% | {mem:+.2f}% |")
            rows.append(dict(os=os, nodes=n, width=w, cpu_saving_percent=cpu,
                             tracked_peak_saving_percent=mem))
            if cpu < -5 or mem < -10:
                go = False
            if n == 30000 and cpu < 25:
                go = False
    report = dict(schema=1, scientific_completeness="PASS", promotion="GO" if go else "NO-GO",
                  rows=rows, source_sha256=reports[0]["original_blob_sha256"])
    Path("set_graph_validation_summary.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print("Scientific completeness: PASS; CPU/memory candidate decision:", report["promotion"])
    # A negative benchmark result is an informative result, not a broken runner.


if __name__ == "__main__":
    main(Path(sys.argv[1]))
