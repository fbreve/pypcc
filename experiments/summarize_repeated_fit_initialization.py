"""Independent strict Linux/Windows repeated-fit observational profile aggregator."""
import json
import math
from pathlib import Path
import statistics
import sys

BASE = "b4c100bd22eeff1979a1a581bc73682b8c3705b8"
SEEDS = (82013, 82117)
BACKENDS = ("numpy", "numba", "cython")
CASES = ("wine", "breast_cancer", "digits",
         "ring_30000_2", "ring_30000_10", "ring_30000_50",
         "ring_100000_2", "ring_100000_10", "ring_100000_50")
ARRAYS = {"predictions", "dominance", "positions", "strengths", "distances", "owndeg"}
PHASES = {"particles", "nodes", "labelprop", "native"}


def require(ok, msg):
    if not ok:
        raise AssertionError(msg)


def expected():
    return {(case, backend, steps)
            for case in CASES
            for backend in BACKENDS
            for steps in ((0, 5, 500)
                          if case in CASES[:3] and backend in ("numba", "cython")
                          else (0, 5))}


def inspect(doc):
    require(doc["base"] == BASE and doc["protected_sources_exact"], "source provenance")
    require(doc["batches"] == 4 and doc["fits_per_batch"] == 8, "precommitted samples")
    require(doc["cases"] == list(CASES), "case list mismatch")
    data = {}
    for item in doc["cohorts"]:
        k = (item["case"], item["backend"], item["steps"])
        require(k not in data, "duplicate case")
        data[k] = item
        require(len(item["pairs"]) == 4, f"missing pairs {k}")
        require(len(item["mask_shas"]) == 2 and len(set(item["mask_shas"])) == 2,
                f"independent masks absent {k}")
        sizes = item["dimensions"]
        require(sizes["dist_table_bytes"] > 0 and sizes["dominance_bytes"] > 0,
                f"nonpositive state size {k}")
        for idx, p in enumerate(item["pairs"]):
            require(p["repetition"] == idx, f"wrong repetition {k}")
            for timing in ("control_wall_ns", "traced_wall_ns"):
                require(type(p[timing]) is int and p[timing] > 0, f"invalid wall {k}")
            for timing in ("control_cpu_ns", "traced_cpu_ns"):
                require(type(p[timing]) is int and p[timing] >= 0, f"invalid CPU {k}")
            require(len(p["state_hashes"]) == 8, f"incomplete state {k}")
            for fit in p["state_hashes"]:
                require(set(fit) == ARRAYS and all(len(h) == 64 for h in fit.values()),
                        f"malformed six-state fingerprints {k}")
            require(len(p["replay_next_rng"]) == 2, f"no replay RNG controls {k}")
            require(set(p["traced_phases_ns"]) == PHASES, f"missing observed phases {k}")
            for phase, durations in p["traced_phases_ns"].items():
                require(len(durations) == 8 and
                        all(type(v) is int and v >= 0 for v in durations),
                        f"bad phase durations {k} {phase}")
            require(all(native <= label
                        for native, label in zip(p["traced_phases_ns"]["native"],
                                                 p["traced_phases_ns"]["labelprop"])),
                    f"native outside label propagation {k}")
    require(set(data) == expected(), f"missing/extra full cohorts: {expected() ^ set(data)}")
    require(len(data) == 60, "unexpected case count")
    return data


def median_ms(values):
    return statistics.median(values) / 1e6


def report(a, b):
    lines = ["# Original repeated-fit initialization and ownership profile",
             "",
             "Scientific **observation**, not an optimization or speedup claim.",
             "Unchanged real production source and native backends; original padded"
             " dense int64 graph retained. ONE installed graph and model per arm,"
             " alternating two masks in eight-fit batches; state hashes and replay"
             " RNG checked outside timing.",
             "",
             "Both OS original/traced cohorts complete, 4 paired 8-fit batches each."
             " Source and all six state arrays, C/F layout, independent previous-fit"
             " array objects, Numba/NumPy next seeded RNG replay and invalid-seed"
             " rollback checks passed.",
             "Windows short process CPU uses quantized time; per-phase probes are"
             " WALL only and have measured control-vs-observer overhead. Residual"
             " cannot be identified as an isolated class-mapping or scratch-allocation"
             " time without further experiments.",
             ""]
    for doc, rows in ((a, inspect(a)), (b, inspect(b))):
        lines += [f"## {doc['os']}", "",
                  "| Workload | Backend | Steps | Original fit wall ms | "
                  "Traced fit wall ms | GenParticles ms | GenNodes ms | "
                  "Native ms | Native share of traced fit | Probe wall drift |",
                  "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for k in sorted(rows):
            item = rows[k]
            paired = item["pairs"]
            base = median_ms([p["control_wall_ns"]/8 for p in paired])
            traced = median_ms([p["traced_wall_ns"]/8 for p in paired])
            phases = {key: median_ms([v for p in paired
                                     for v in p["traced_phases_ns"][key]])
                      for key in PHASES}
            native_share = phases["native"] / traced * 100
            overhead = (traced / base - 1) * 100
            lines.append(f"| {k[0]} | {k[1]} | {k[2]} | {base:.4f} | "
                         f"{traced:.4f} | {phases['particles']:.4f} | "
                         f"{phases['nodes']:.4f} | {phases['native']:.4f} | "
                         f"{native_share:.1f}% | {overhead:+.1f}% |")
        lines.append("")
        lines.append("**Interpretation:** phase timings are measured on the traced "
                     "arm, while uninstrumented complete-fit medians are the CPU/wall "
                     "baseline. The non-native remainder includes multiple distinct "
                     "activities and is not a single candidate optimization.")
        lines.append("")
    lines += [
        "## Decision",
        "",
        "**COMPLETE OBSERVATIONAL PROFILE — no performance GO/NO-GO applicable.**",
        "No reuse optimization was tested and no production source changed. "
        "An allocation or state reuse candidate is only justified if its total "
        "fit CPU potential is material in the relevant workload and it can "
        "preserve previous-result array independence and exception rollback. "
        "No scientific PR merge into master."
    ]
    return "\n".join(lines) + "\n"


def main():
    root = Path(sys.argv[1])
    out = Path(sys.argv[2])
    paths = sorted(root.rglob("repeated-fit-*.json"))
    require(len(paths) == 2, f"expected Linux and Windows: {paths}")
    docs = [json.loads(p.read_text(encoding="utf-8")) for p in paths]
    by_os = {d["os"]: d for d in docs}
    require(set(by_os) == {"linux", "win32"}, f"missing hosts {set(by_os)}")
    linux, windows = by_os["linux"], by_os["win32"]
    a, b = inspect(linux), inspect(windows)
    for key in expected():
        require(a[key]["graph_sha"] == b[key]["graph_sha"],
                f"ordered graph differs across OS {key}")
        require(a[key]["degree_sha"] == b[key]["degree_sha"],
                f"degree differs across OS {key}")
        require(a[key]["mask_shas"] == b[key]["mask_shas"],
                f"masks differ across OS {key}")
        for idx in range(4):
            require(a[key]["pairs"][idx]["state_hashes"]
                    == b[key]["pairs"][idx]["state_hashes"],
                    f"cross-host state fingerprints differ {key}")
    out.mkdir(parents=True, exist_ok=True)
    (out / "repeated_fit_initialization_report.md").write_text(
        report(linux, windows), encoding="utf-8")
    (out / "repeated_fit_initialization_decision.json").write_text(
        json.dumps(dict(base=BASE, verdict="OBSERVATION_COMPLETE_NO_GO_APPLICABLE",
                        hosts=["linux", "win32"], cohorts_per_host=60,
                        batches_per_cohort=4, fits_per_batch=8,
                        total_timed_fits=60*4*8*2*2,
                        all_parity_checks=True,
                        production_unchanged=True), indent=2) + "\n",
        encoding="utf-8")
    print((out / "repeated_fit_initialization_report.md").read_text(
        encoding="utf-8"))
    print("STRICT 2-OS SOURCE-EXACT REPEATED-FIT OBSERVATION COMPLETE")


if __name__ == "__main__":
    main()
