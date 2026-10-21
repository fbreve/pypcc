"""Full-trajectory Cython counters and instrumented/uninstrumented state parity."""
import json
import sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from benchmarks.graph_topology import build
from pcc_step import pcc_propagate

def run(args, scratch, weights, counts=None):
    return pcc_propagate(*args, 1.0, 2.0, *scratch, weights, 300, 0, 10, 1, trace_counts=counts)

def main():
    records = []
    for degree in (2, 8, 32):
        for fraction in (0.1, 0.25, 0.5):
            for layout in ("prefix", "random"):
                for placement in ("stratified", "random"):
                    for seed in (101, 202, 303):
                        args, scratch, weights = build(degree, fraction, 512, layout, placement, seed)
                        other = [a.copy() if isinstance(a, np.ndarray) else a for a in args]
                        other_scratch = [a.copy() for a in scratch]
                        counts = {}
                        run(args, scratch, weights, counts)
                        run(other, other_scratch, weights)
                        for i in (7, 8, 9, 10, 11, 12):
                            if not np.array_equal(args[i], other[i]):
                                raise AssertionError(f"State divergence at index {i}: {degree}, {fraction}, {layout}, {placement}, {seed}")
                        if counts["accepted"] + counts["rejected"] + counts["skipped"] != 512 * 300:
                            raise AssertionError(f"Invalid count total: {counts}")
                        if counts["greedy_selected"] + counts["random_selected"] != counts["accepted"] + counts["rejected"]:
                            raise AssertionError(f"Invalid movement count: {counts}")
                        if counts["labeled_destinations"] + counts["unlabeled_destinations"] != counts["accepted"] + counts["rejected"]:
                            raise AssertionError(f"Invalid destination count: {counts}")
                        records.append(dict(degree=degree, fraction=fraction, layout=layout, placement=placement, seed=seed, **counts))
    Path("full-trajectory-counters.json").write_text(json.dumps(records, indent=2))
    print(f"Validated {len(records)} complete Cython trajectories")
    for record in records:
        print(json.dumps(record, sort_keys=True))

if __name__ == "__main__":
    main()
