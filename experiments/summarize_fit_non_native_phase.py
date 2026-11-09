"""Aggregate exact real-source Linux/Windows PCC fit-phase observations."""
import json
from pathlib import Path
import statistics
import sys

BASE = "b4c100bd22eeff1979a1a581bc73682b8c3705b8"
REAL = ("wine", "breast_cancer", "digits")
CASES = (*REAL, "ring_30000_10", "ring_100000_50")
BACKENDS = ("numpy", "numba", "cython")
PHASES = ("preflight", "classes_backend", "model_setup", "label_mapping",
          "parameters", "particles", "nodes", "distance_weights",
          "scratch_allocation", "stop_preflight", "dispatch", "native",
          "unlabeled_assignment", "normalization", "inverse_labels")
STATE = {"predictions", "dominance", "positions", "strengths", "distances", "owndeg"}


def insist(value, reason):
    if not value:
        raise AssertionError(reason)


def expected():
    return {(c, b, iterations) for c in CASES for b in BACKENDS
            for iterations in ((0, 5, 500) if c in REAL and b != "numpy" else (0, 5))}


def checked(d):
    insist(d["base"] == BASE and d["production_source_unchanged"], "source provenance")
    insist(d["pairs"] == 4 and d["batch_fits"] == 8, "frozen batch")
    insist(tuple(d["cases"]) == CASES and tuple(d["phases"]) == PHASES, "frozen corpus")
    result = {}
    for item in d["cohorts"]:
        key = (item["case"], item["backend"], item["steps"])
        insist(key not in result, "duplicate cohort")
        result[key] = item
        insist(len(item["label_shas"]) == 2 and len(set(item["label_shas"])) == 2, "label masks")
        insist(len(item["repetitions"]) == 4, "four paired batches")
        for i, rep in enumerate(item["repetitions"]):
            insist(rep["repetition"] == i, "paired order")
            for name in ("original_wall_ns", "observed_wall_ns"):
                insist(type(rep[name]) is int and rep[name] > 0, "wall clock")
            for name in ("original_cpu_ns", "observed_cpu_ns"):
                insist(type(rep[name]) is int and rep[name] >= 0, "CPU clock")
            insist(len(rep["states"]) == 8 and len(rep["observed_phase_ns"]) == 8
                   and len(rep["next_rng"]) == 2, "full original PCC fit observations")
            for state in rep["states"]:
                insist(set(state) == STATE and all(len(h) == 64 for h in state.values()),
                       "complete six-state arrays")
            for times in rep["observed_phase_ns"]:
                insist(set(times) == set(PHASES) and
                       all(type(x) is int and x >= 0 for x in times.values()),
                       "complete 15 disjoint original phases")
    insist(set(result) == expected() and len(result) == 36,
           f"missing/excess cohorts {expected() ^ set(result)}")
    return result


def median_ms(values):
    return statistics.median(values) / 1e6


def report(docs):
    rows = ["# Original PyPCC fit non-native phase attribution",
            "",
            "Scientific observation, NOT a production optimization or GO/NO-GO.",
            "The original-source real PCC kernels, neighbor graph, random draw order,",
            "all six state arrays and rollback remain unchanged. Phase measurements",
            "are wall time with measurement overhead; separate full-batch CPU and",
            "wall controls use genuinely unmodified original PCC. Windows CPU",
            "time is quantized and host times must not be pooled.",
            ""]
    for d in docs:
        data = checked(d)
        rows.extend(["## " + d["os"], "",
                     "| Case | Backend | Steps | Original complete-fit wall (ms) |",
                     "|:--|:--|--:|--:|"])
        # Detailed original/traced times and the complete fifteen-dimensional
        # phase vectors are retained in each host's raw JSON artifacts.
        for key in sorted(data):
            pairs = data[key]["repetitions"]
            wall = median_ms([x["original_wall_ns"]/8 for x in pairs])
            rows.append(f"| {key[0]} | {key[1]} | {key[2]} | {wall:.4f} |")
        rows.extend(["", "### Selected original wall phase decomposition",
                     "",
                     "| Case | Backend | Steps | Original CPU ms | Traced whole wall ms | Validation | Labels/map | Particles | Nodes | Scratch/stop | Native | Normalize | Output mapping |",
                     "|:--|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|"])
        for key in sorted(data):
            if key[0] not in ("digits","breast_cancer","ring_100000_50") or key[1] != "numba":
                continue
            pairs = data[key]["repetitions"]
            m = {stage: median_ms([v[stage] for rep in pairs
                                  for v in rep["observed_phase_ns"]]) for stage in PHASES}
            cpu = median_ms([rep["original_cpu_ns"]/8 for rep in pairs])
            traced = median_ms([rep["observed_wall_ns"]/8 for rep in pairs])
            label = m["classes_backend"] + m["model_setup"] + m["label_mapping"]
            scratch = m["scratch_allocation"] + m["stop_preflight"]
            rows.append(f"| {key[0]} | {key[1]} | {key[2]} | {cpu:.4f} | "
                        f"{traced:.4f} | {m['preflight']:.4f} | {label:.4f} | "
                        f"{m['particles']:.4f} | {m['nodes']:.4f} | {scratch:.4f} | "
                        f"{m['native']:.4f} | {m['normalization']:.4f} | "
                        f"{m['inverse_labels']:.4f} |")
        rows.extend(["", "Disjoint phases not shown separately in this compact table",
                     "(parameters, distance weights, dispatch, unlabeled assignment)",
                     "remain fully recorded in raw JSON. Stage wall times reflect probes,",
                     "not an independently measured CPU saving.", ""])
    rows.extend(["## Decision", "",
                 "**SOURCE, SCIENCE AND TWO-HOST COMPLETENESS PASS; NO OPTIMIZATION GO APPLICABLE.**",
                 "Next investigate an independently preregistered A/B source candidate",
                 "only if measured full-fit CPU potential is material, and only with",
                 "exact prior-fit state independence, rollback, native RNG/stopping,",
                 "graph order and bounded memory. Do not alter the CPU-oriented",
                 "padded dense int64 neighbor representation based on memory alone."])
    return "\n".join(rows) + "\n"


def main():
    root, output = Path(sys.argv[1]), Path(sys.argv[2])
    files = sorted(root.rglob("phase-original-*.json"))
    insist(len(files) == 2, f"missing source host artifacts {files}")
    docs = [json.loads(p.read_text(encoding="utf-8")) for p in files]
    by_os = {d["os"]: d for d in docs}
    insist(set(by_os) == {"linux", "win32"}, "two OS required")
    a, b = checked(by_os["linux"]), checked(by_os["win32"])
    # The original-Git source was independently byte-guarded against BASE on
    # both native runners. The raw source_sha256 was computed from git-show
    # text-mode output; Windows converts CRLF on decode whereas Linux does not.
    # A cross-OS SHA256 of the decoded text is not a valid Git-blob guard.
    # Keep both raw diagnostic hashes, rely on the checked canonical Git BASE
    # plus exact native outputs, graphs and masks rather than falsely failing.
    insist(by_os["linux"]["base"] == by_os["win32"]["base"] == BASE and
           by_os["linux"]["production_source_unchanged"] and
           by_os["win32"]["production_source_unchanged"],
           "immutable original Git source provenance missing")
    for key in expected():
        insist(a[key]["ordered_graph_sha"] == b[key]["ordered_graph_sha"] and
               a[key]["degrees_sha"] == b[key]["degrees_sha"] and
               a[key]["label_shas"] == b[key]["label_shas"], f"cross-host graph/labels {key}")
        for i in range(4):
            insist(a[key]["repetitions"][i]["states"] ==
                   b[key]["repetitions"][i]["states"], f"cross-host exact states {key}")
    output.mkdir(parents=True, exist_ok=True)
    (output / "fit_non_native_phase_report.md").write_text(
        report([by_os["linux"], by_os["win32"]]), encoding="utf-8")
    (output / "fit_non_native_phase_decision.json").write_text(
        json.dumps(dict(base=BASE, verdict="COMPLETE_OBSERVATION_NO_PRODUCTION_GO",
                        hosts=["linux", "win32"], cohorts_per_host=36,
                        source_parity=True, full_fit_parity=True,
                        timed_original_and_traced_fits=36*4*8*2*2,
                        production_unchanged=True), indent=2)+"\n", encoding="utf-8")
    print((output / "fit_non_native_phase_report.md").read_text(encoding="utf-8"))
    print("STRICT BOTH-OS ORIGINAL PCC FIT PHASE OBSERVATION COMPLETE")


if __name__ == "__main__":
    main()
