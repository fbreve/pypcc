"""Require all eight preregistered phase-4 dense NumPy parity pairs."""
import argparse
import csv
from pathlib import Path

from experiments.numpy_continuous_observer import MAX_BYTES

OSES=("ubuntu-latest","windows-latest")
DATASETS=("iris","digits")
SEEDS=(16007,16111)
FLAGS=("arrays_exact","stop_exact","rng_exact","next16_exact",
       "final_snapshot_exact","dense_schedule_exact")


def check(files):
    found={}
    for name in files:
        with open(name,newline="",encoding="utf-8") as f:
            for row in csv.DictReader(f):
                key=(row["os"],row["dataset"],int(row["seed"]))
                if key in found:
                    raise AssertionError(f"Duplicate dense pair: {key}")
                if key not in {(os,d,s) for os in OSES for d in DATASETS for s in SEEDS}:
                    raise AssertionError(f"Unexpected dense pair: {key}")
                if not all(row[k]=="True" for k in FLAGS):
                    raise AssertionError(f"Strict parity failure: {key}")
                if (int(row["row_count"])!=112 or
                        int(row["effective_iterations"])!=111 or
                        not 0<int(row["used_bytes"])<=int(row["max_bytes"])==MAX_BYTES):
                    raise AssertionError(f"Snapshot or budget mismatch: {key}")
                if not (float(row["baseline_seconds"])>0 and
                        float(row["observed_seconds"])>0):
                    raise AssertionError(f"Invalid timings: {key}")
                found[key]=row
    expected={(os,d,s) for os in OSES for d in DATASETS for s in SEEDS}
    if set(found)!=expected:
        raise AssertionError(f"Missing frozen dense pairs: {sorted(expected-set(found))}")
    return found


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--files",nargs="+",required=True)
    p.add_argument("--output",required=True)
    args=p.parse_args()
    rows=check(args.files)
    lines=["# Phase 4: all-iteration observer, frozen 8/8 exact pairs",
           "","| OS | Dataset | Seed | Rows | Retained MiB | Overhead % |",
           "|---|---|---:|---:|---:|---:|"]
    for key,row in sorted(rows.items()):
        lines.append("| "+ " | ".join((
            key[0],key[1],str(key[2]),row["row_count"],
            f'{int(row["used_bytes"])/(1024*1024):.3f}',
            f'{float(row["overhead_percent"]):+.2f}'))+" |")
    lines+=["","All 8 pairs match six complete arrays, native stop metadata,",
            "full global MT19937 state and next 16 random outputs.",
            "Timing is exploratory and not a preregistered performance gate.",
            "Retained bytes exclude temporary arrays, Python overhead and RSS."]
    Path(args.output).write_text("\n".join(lines)+"\n",encoding="utf-8")
    print("\n".join(lines),flush=True)


if __name__=="__main__":
    main()
