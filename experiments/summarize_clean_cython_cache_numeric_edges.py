"""Verify the complete frozen 6480-case Windows/Linux cache numeric grid."""
import argparse
import csv
from itertools import product
from pathlib import Path

OSES=("ubuntu-latest","windows-latest")
CLASSES=(2,3,5,10,32)
GREEDY=(0.0,0.5,1.0)
DELTA_V=(0.0,1e-12,0.1)
DELTAP=(0.4,1.0)
STOP_MAX=(0,1,2)
LAYOUTS=("C","F")
SEEDS=(311,733)
CAPS=(1,11,65)
FIELDS=("exact_state","exact_full_stop_metadata",
        "verified_cache_at_all_checkpoints","isolated_labeled_node")


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--files",nargs="+",required=True)
    p.add_argument("--output",required=True)
    args=p.parse_args(argv)
    if len(args.files)!=2:
        raise AssertionError("Exactly two independent OS artifacts required")
    rows={}
    for path in args.files:
        with open(path,newline="",encoding="utf-8") as f:
            for row in csv.DictReader(f):
                osname=row["runner_os"]
                case=(
                    int(row["classes"]),float(row["p_grd"]),
                    float(row["delta_v"]),float(row["deltap"]),
                    int(row["stop_max"]),row["layout"],
                    int(row["seed"]),int(row["cap"]))
                key=(osname,case)
                if osname not in OSES or key in rows:
                    raise AssertionError("Unexpected or duplicate OS×case")
                if (any(row[field]!="True" for field in FIELDS)
                        or int(row["checks"])<1
                        or not 0<int(row["effective_iterations"])<=case[-1]):
                    raise AssertionError(f"Invalid strict-state/stop/metric gate {key}")
                rows[key]=row
    expected=set(product(
        CLASSES,GREEDY,DELTA_V,DELTAP,STOP_MAX,LAYOUTS,SEEDS,CAPS))
    if len(expected)!=3240:
        raise AssertionError("Invalid frozen grid size")
    for osname in OSES:
        actual={case for name,case in rows if name==osname}
        if actual!=expected:
            raise AssertionError(
                f"Incomplete {osname}: missing={len(expected-actual)}, "
                f"extra={len(actual-expected)}")
    if len(rows)!=6480:
        raise AssertionError(f"Expected 6480 exact parity cases, got {len(rows)}")
    text=(
        "# Clean-master candidate Cython numerical corner-case parity\n\n"
        "**6,480/6,480 exact-state + full stop-metadata + per-check "
        "potential equivalence cases passed**, with **3,240/3,240 on "
        "Windows** and **3,240/3,240 on Linux**. No omitted case.\n\n"
        "Grid: 2/3/5/10/32 classes; p_grd 0/0.5/1; delta_v "
        "0/1e-12/0.1; deltap 0.4/1; stop_max 0/1/2; C/F dominance "
        "layout; seeds 311/733; caps 1/11/65; one isolated labeled node. "
        "Each candidate cache global potential metric was independently "
        "compared with the original whole-dominance scan at **every "
        "sampled stopping checkpoint** before stop-state comparison.\n\n"
        "The result applies to the V2 specialized Cython source on "
        "the clean-master draft PR, not automatically to arbitrary "
        "graphs or invalid inputs. The production master is unchanged.\n"
    )
    Path(args.output).write_text(text,encoding="utf-8")
    print(text,flush=True)


if __name__=="__main__":
    main()
