"""Frozen direct-Cython numerical/edge-case parity stress on Linux/Windows.

Every cached run self-checks ALL global-potential samples against the original
full dominance scan. The baseline has exactly the same input arrays/RNG seed.
See cython_rowmax_cache_replication_protocol.md.
"""
import argparse
import csv
from itertools import product

import numpy as np
import pcc_step

CLASSES = (2, 3, 5, 10, 32)
GREEDY = (0.0, 0.5, 1.0)
DELTA_V = (0.0, 1e-12, 0.1)
DELTAP = (0.4, 1.0)
STOP_MAX = (0, 1, 2)
LAYOUTS = ("C", "F")
SEEDS = (311, 733)
CAPS = (1, 11, 65)
EXPECTED_CASES = (
    len(CLASSES) * len(GREEDY) * len(DELTA_V) * len(DELTAP) *
    len(STOP_MAX) * len(LAYOUTS) * len(SEEDS) * len(CAPS)
)
FIELDS = (
    "runner_os", "classes", "p_grd", "delta_v", "deltap",
    "stop_max", "layout", "seed", "cap",
    "n_nodes", "isolated_labeled_node", "effective_iterations",
    "checks", "stop_reason", "exact_state",
    "exact_full_stop_metadata", "verified_cache_at_all_checkpoints",
)
MUTABLE = ("positions", "strengths", "distances", "dominance", "ownership")


def make_buffers(c, p_grd, delta_v, deltap, layout):
    n = max(24, c + 4)
    # A reciprocal ring for the first n-1 vertices plus an isolated last
    # vertex. This exercises unchanged cache entries and skipped movements.
    nbr = np.zeros((n, 2), dtype=np.int64)
    degree = np.zeros(n, dtype=np.int64)
    for i in range(n - 1):
        nbr[i, 0] = (i - 1) % (n - 1)
        nbr[i, 1] = (i + 1) % (n - 1)
        degree[i] = 2
    if degree[-1] != 0 or np.any(degree[:-1] != 2):
        raise AssertionError("Not the frozen isolated-node graph")
    labels = np.full(n, -1, dtype=np.int64)
    labels[:c - 1] = np.arange(c - 1, dtype=np.int64)
    labels[-1] = c - 1
    homes = np.flatnonzero(labels >= 0)
    if len(homes) != c or homes[-1] != n - 1:
        raise AssertionError("Missing isolated labeled particle")
    positions = homes.copy()
    pclasses = labels[homes].copy()
    strengths = np.ones(c, dtype=np.float64)
    distances = np.full((n,c), min(255,n-1), dtype=np.uint8, order="F")
    distances[homes,np.arange(c)] = 0
    dominance = np.full((n,c),1.0/c,dtype=np.float64,order=layout)
    dominance[homes,:] = 0.0
    dominance[homes,pclasses] = 1.0
    ownership = np.zeros((n,c),dtype=np.float64,order="F")
    scratch = [
        np.empty(c, dtype=np.float64), np.empty(c,dtype=np.float64),
        np.empty(2,dtype=np.float64), np.empty(2,dtype=np.float64),
        np.empty(2,dtype=np.float64), np.empty(2,dtype=np.float64),
        1.0 / (np.arange(257,dtype=np.float64)+1.0)**2.0,
    ]
    args = (
        nbr, degree, labels, p_grd, delta_v, c,
        np.zeros(c,dtype=np.float64), positions, pclasses,
        strengths, distances, dominance, ownership,
        deltap, 2.0, *scratch,
    )
    state = {
        "positions": positions, "strengths": strengths,
        "distances": distances, "dominance": dominance,
        "ownership": ownership,
    }
    return args, state, n


def propagate(condition, use_cache):
    c,p_grd,delta_v,deltap,stop_max,layout,seed,cap=condition
    args, state, n = make_buffers(c,p_grd,delta_v,deltap,layout)
    info = pcc_step.pcc_propagate(
        *args, cap, 1, 2000, stop_max, return_info=True,
        seed=seed, experimental_rowmax_cache=use_cache,
        experimental_verify_rowmax_cache=use_cache,
    )
    states={k:np.asarray(a).copy() for k,a in state.items()}
    if not all(np.isfinite(a).all() for a in (
            states["dominance"],states["strengths"],states["ownership"])):
        raise AssertionError("Nonfinite direct-Cython output")
    return states, dict(info), n


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runner-os",required=True,
                        choices=("ubuntu-latest","windows-latest"))
    parser.add_argument("--output",required=True)
    args=parser.parse_args(argv)
    cases=product(CLASSES,GREEDY,DELTA_V,DELTAP,STOP_MAX,
                  LAYOUTS,SEEDS,CAPS)
    count=0
    with open(args.output, "w", newline="", encoding="utf-8") as out:
        writer=csv.DictWriter(out,fieldnames=FIELDS)
        writer.writeheader()
        out.flush()
        for condition in cases:
            old_state, old_info,n = propagate(condition,False)
            new_state,new_info,_ = propagate(condition,True)
            state_eq=all(np.array_equal(old_state[key],new_state[key])
                         for key in MUTABLE)
            meta_eq=old_info==new_info
            if not state_eq or not meta_eq:
                broken=[key for key in MUTABLE
                        if not np.array_equal(old_state[key],new_state[key])]
                raise AssertionError(
                    f"{args.runner_os}: state/metric discrepancy {condition} "
                    f"arrays={broken}, original={old_info},cache={new_info}")
            c,p_grd,delta_v,deltap,stop_max,layout,seed,cap = condition
            checks=int(old_info["checks"])
            if not 0<old_info["effective_iterations"]<=cap or not checks:
                raise AssertionError(f"Missing stop metric observation {condition}")
            writer.writerow(dict(
                runner_os=args.runner_os,classes=c,p_grd=p_grd,
                delta_v=delta_v,deltap=deltap,stop_max=stop_max,
                layout=layout,seed=seed,cap=cap,n_nodes=n,
                isolated_labeled_node=True,
                effective_iterations=old_info["effective_iterations"],
                checks=checks,stop_reason=old_info["stop_reason"],
                exact_state=state_eq,
                exact_full_stop_metadata=meta_eq,
                verified_cache_at_all_checkpoints=True,
            ))
            count+=1
            if count%180==0:
                out.flush()
                print(f"{args.runner_os}: {count}/{EXPECTED_CASES} "
                      "exact numeric and stop parity cases PASS",flush=True)
    if count != EXPECTED_CASES:
        raise AssertionError(
            f"Cartesian product incomplete {count} != {EXPECTED_CASES}")
    print(f"{args.runner_os}: all {count} synthetic direct-Cython cases pass",
          flush=True)


if __name__=="__main__":
    main()
