"""Long-duration paired native benchmark with independent compiled binaries.

A more stable 300k-step benchmark than the original 20k (~3-5 ms Breast
Cancer) workload. All full-state and full-stop comparisons remain mandatory.
See cython_cache_long_native_protocol.md (frozen before this run).
"""
import argparse
import csv
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path
from statistics import median

import numpy as np

# This experiment runs from each checkout's CWD but imports its driver from
# the candidate tree. Each worker inserts CWD ahead of that tree so the actual
# compiled pcc_step / pcc_graph come from its independently built checkout.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.audit_clean_master_cython_cache import (  # noqa: E402
    canonical_direct_kernel_input, expected_stop_max, STATE,
)

OSES = ("ubuntu-latest", "windows-latest")
DATASETS = ("digits", "breast_cancer")
SEEDS = (9103, 9209, 9301, 9403)
VARIANTS = ("master", "candidate_off", "candidate_on")
CAP = 300000
ES_CHK = 10000000
BLOCKS = 2
INNER_REPEATS = 4
FIELDS = (
    "runner_os", "dataset", "seed", "block", "repeat", "variant",
    "order_index", "iterations", "checks", "stop_reason", "stop_max",
    "stalled_checks", "elapsed_seconds", "source_commit",
    "extension_sha256", "extension_file", "state_sha256",
    "master_block_median_seconds", "off_block_median_seconds",
    "on_block_median_seconds", "off_saving_percent", "on_saving_percent",
    "all_three_state_digests_equal", "all_three_stop_info_equal",
    "real_native_extension", "reciprocal_graph",
)


def fingerprint(state):
    h = hashlib.sha256()
    for key in STATE:
        arr = np.asarray(state[key])
        h.update(key.encode("ascii"))
        h.update(arr.dtype.str.encode("ascii"))
        h.update(str(arr.shape).encode("ascii"))
        h.update(arr.tobytes(order="C"))
    return h.hexdigest()


def timed_worker(args):
    checkout = Path.cwd().resolve()
    sys.path.insert(0, str(checkout))
    import pcc_step
    extension = Path(pcc_step.__file__).resolve()
    if extension.parent != checkout or extension.suffix.lower() not in (
            ".so", ".pyd"):
        raise AssertionError(f"Expected real checkout-native extension: {extension}")
    git_sha = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=checkout, text=True).strip()
    kwargs = dict(return_info=True, seed=args.seed)
    if args.variant == "candidate_off":
        kwargs["experimental_rowmax_cache"] = False
    if args.variant == "candidate_on":
        # Deliberately omit the keyword to exercise the activated default.
        pass
    times = []
    digests = []
    reference_info = None
    final_state = None
    for repeat in range(INNER_REPEATS):
        kernel_args, state, labels, truth, degrees = (
            canonical_direct_kernel_input(args.dataset, args.seed))
        n = len(labels)
        p = int(np.count_nonzero(labels >= 0))
        stop_max = expected_stop_max(n, p, 10, ES_CHK, degrees)
        if stop_max <= CAP // 10 + 1:
            raise AssertionError("Potential monitor might stop early")
        t0 = time.perf_counter()
        info = pcc_step.pcc_propagate(
            *kernel_args, CAP, 1, ES_CHK, stop_max, **kwargs)
        elapsed = time.perf_counter() - t0
        info = dict(info)
        if (info["effective_iterations"] != CAP
                or info["stop_max"] != stop_max
                or not elapsed > 0):
            raise AssertionError(f"Wrong fixed-work output {args.variant}: {info}")
        predictions = np.argmax(state["dominance"], axis=1)
        predictions[labels >= 0] = labels[labels >= 0]
        state["predictions"] = predictions
        if reference_info is None:
            reference_info = info
        elif reference_info != info:
            raise AssertionError("Repetitions differ in stopping metadata")
        d = fingerprint(state)
        if digests and d != digests[0]:
            raise AssertionError("Repetitions differ in final full state")
        digests.append(d)
        times.append(elapsed)
        if repeat == INNER_REPEATS-1:
            final_state = state
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        str(output.with_suffix(".npz")),
        **{key: np.asarray(value) for key,value in final_state.items()})
    metadata = dict(
        runner_os=args.runner_os,dataset=args.dataset,seed=args.seed,
        variant=args.variant,block=args.block,cap=CAP,es_chk=ES_CHK,
        times=times,state_digests=digests,info=reference_info,
        extension_file=str(extension),extension_sha256=hashlib.sha256(
            extension.read_bytes()).hexdigest(),
        source_commit=git_sha,real_native_extension=True,
        reciprocal_graph=True,
    )
    output.with_suffix(".json").write_text(
        json.dumps(metadata,sort_keys=True,indent=2),encoding="utf-8")
    print(f"{args.variant}/{args.dataset}/{args.seed}/block{args.block}: "
          f"median native {1000*median(times):.3f}ms, "
          f"state+stop digest PASS",flush=True)


def ensure_three_way(records):
    infos = [records[v][1]["info"] for v in VARIANTS]
    if not (infos[0] == infos[1] == infos[2]):
        raise AssertionError("Three independently compiled variants differ in stop info")
    hashes = [records[v][1]["state_digests"] for v in VARIANTS]
    if not(hashes[0] == hashes[1] == hashes[2]):
        raise AssertionError("Three variants differ in some full state repetition")
    with np.load(records["master"][0],allow_pickle=False) as baseline:
        for variant in VARIANTS[1:]:
            with np.load(records[variant][0],allow_pickle=False) as other:
                for key in STATE:
                    if key not in baseline or key not in other or not (
                            np.array_equal(baseline[key],other[key])):
                        raise AssertionError(f"Final native array mismatch: {variant}/{key}")
    baseline=records["master"][1]
    off=records["candidate_off"][1]
    on=records["candidate_on"][1]
    if not (baseline["source_commit"]!=off["source_commit"]
            and off["source_commit"]==on["source_commit"]
            and baseline["extension_sha256"]!=off["extension_sha256"]
            and off["extension_sha256"]==on["extension_sha256"]):
        raise AssertionError("Incorrect independent native source/binary identities")
    return hashes[0]


def orchestrate(args):
    baseline = Path(args.baseline_dir).resolve()
    candidate = Path(args.candidate_dir).resolve()
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True,exist_ok=True)
    script = (candidate/"experiments"/"audit_cython_cache_long_native.py").resolve()
    if not script.is_file() or baseline==candidate:
        raise AssertionError("Missing split native checkouts")
    with open(args.csv,"w",newline="",encoding="utf-8") as f:
        writer = csv.DictWriter(f,fieldnames=FIELDS)
        writer.writeheader()
        f.flush()
        for seed_idx,seed in enumerate(SEEDS):
            for block in range(BLOCKS):
                order=VARIANTS[(seed_idx+block)%3:]+VARIANTS[:(seed_idx+block)%3]
                results={}
                for order_idx,variant in enumerate(order):
                    checkout = baseline if variant=="master" else candidate
                    prefix=(output/f"long_{args.runner_os}_{args.dataset}_"
                            f"{seed}_block{block}_{variant}")
                    call=[sys.executable,str(script),"worker",
                          "--runner-os",args.runner_os,
                          "--dataset",args.dataset,"--seed",str(seed),
                          "--variant",variant,"--block",str(block),
                          "--output",str(prefix)]
                    proc=subprocess.run(
                        call,cwd=checkout,capture_output=True,
                        text=True,check=False)
                    if proc.returncode:
                        raise RuntimeError(
                            f"Longrun worker failed: {call}; stdout={proc.stdout[-3000:]} "
                            f"stderr={proc.stderr[-3000:]}")
                    results[variant]=(
                        prefix.with_suffix(".npz"),
                        json.loads(prefix.with_suffix(".json").read_text(
                            encoding="utf-8")),
                        order_idx)
                hashes=ensure_three_way(results)
                medians={variant:median(results[variant][1]["times"])
                         for variant in VARIANTS}
                saved_off=100*(1-medians["candidate_off"]/medians["master"])
                saved_on=100*(1-medians["candidate_on"]/medians["master"])
                for variant in VARIANTS:
                    _,info,order_idx=results[variant]
                    for rep,(elapsed,digest) in enumerate(
                            zip(info["times"],info["state_digests"])):
                        writer.writerow(dict(
                            runner_os=args.runner_os,dataset=args.dataset,
                            seed=seed,block=block,repeat=rep,variant=variant,
                            order_index=order_idx,
                            iterations=info["info"]["effective_iterations"],
                            checks=info["info"]["checks"],
                            stop_reason=info["info"]["stop_reason"],
                            stop_max=info["info"]["stop_max"],
                            stalled_checks=info["info"]["stalled_checks"],
                            elapsed_seconds=elapsed,
                            source_commit=info["source_commit"],
                            extension_sha256=info["extension_sha256"],
                            extension_file=info["extension_file"],
                            state_sha256=digest,
                            master_block_median_seconds=medians["master"],
                            off_block_median_seconds=medians["candidate_off"],
                            on_block_median_seconds=medians["candidate_on"],
                            off_saving_percent=saved_off,
                            on_saving_percent=saved_on,
                            all_three_state_digests_equal=True,
                            all_three_stop_info_equal=True,
                            real_native_extension=True,
                            reciprocal_graph=True,
                        ))
                f.flush()
                print(f"{args.runner_os}/{args.dataset}/{seed}/block{block}: "
                      f"long 300k medians(ms) master={1000*medians['master']:.2f}, "
                      f"off={1000*medians['candidate_off']:.2f}, "
                      f"on={1000*medians['candidate_on']:.2f}; "
                      f"off={saved_off:+.2f}% on={saved_on:+.2f}% "
                      "exact triplet PASS",flush=True)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    cmd=parser.add_subparsers(dest="command",required=True)
    w=cmd.add_parser("worker")
    w.add_argument("--runner-os",choices=OSES,required=True)
    w.add_argument("--dataset",choices=DATASETS,required=True)
    w.add_argument("--seed",type=int,choices=SEEDS,required=True)
    w.add_argument("--variant",choices=VARIANTS,required=True)
    w.add_argument("--block",type=int,choices=range(BLOCKS),required=True)
    w.add_argument("--output",required=True)
    o=cmd.add_parser("orchestrate")
    o.add_argument("--runner-os",choices=OSES,required=True)
    o.add_argument("--dataset",choices=DATASETS,required=True)
    o.add_argument("--baseline-dir",required=True)
    o.add_argument("--candidate-dir",required=True)
    o.add_argument("--output-dir",required=True)
    o.add_argument("--csv",required=True)
    args=parser.parse_args(argv)
    timed_worker(args) if args.command=="worker" else orchestrate(args)


if __name__=="__main__":
    main()
