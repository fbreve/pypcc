"""Validate every cross-revision Cython artifact and the frozen decision gates."""
import argparse
import csv
from itertools import product
from pathlib import Path
from statistics import mean,median

from experiments.audit_clean_master_cython_cache import (
    DATASETS,OSES,SEEDS,POLICIES,VARIANTS,
)


def validate(paths):
    if len(paths)!=len(DATASETS)*len(OSES):
        raise AssertionError("Require all four OS×dataset raw CSV artifacts")
    rows={}
    for path in paths:
        with open(path,encoding="utf-8",newline="") as file:
            for r in csv.DictReader(file):
                osname,ds,seed,policy,rep,variant=(
                    r["runner_os"],r["dataset"],int(r["seed"]),
                    r["policy"],int(r["repeat"]),r["variant"])
                if (osname not in OSES or ds not in DATASETS
                        or seed not in SEEDS
                        or policy not in [p[0] for p in POLICIES]
                        or variant not in VARIANTS):
                    raise AssertionError(f"Unexpected run {r}")
                key=(osname,ds,seed,policy,rep,variant)
                if key in rows:
                    raise AssertionError(f"Duplicate run: {key}")
                if any(r[field]!="True" for field in (
                        "all_three_states_equal",
                        "all_three_stop_metadata_equal",
                        "real_cython_native_extension",
                        "reciprocal_graph")):
                    raise AssertionError(f"Missing strict parity/native gate {key}")
                if not (r["extension_file"] and
                        len(r["extension_sha256"])==64 and
                        len(r["source_commit"])==40):
                    raise AssertionError(f"Missing commit/native hash {key}")
                rows[key]=r
    expected={
        (osname,ds,seed,policy,rep,variant)
        for osname,ds,seed,(policy,cap,monitored,es_chk,reps),variant
        in product(OSES,DATASETS,SEEDS,POLICIES,VARIANTS)
        for rep in range(reps)
    }
    if set(rows)!=expected or len(rows)!=240:
        raise AssertionError(
            f"Frozen cohort mismatch: rows={len(rows)}, "
            f"missing={len(expected-set(rows))}, extra={len(set(rows)-expected)}")
    samples={}
    master_commits=set()
    candidate_commits=set()
    for osname,ds,seed in product(OSES,DATASETS,SEEDS):
        for policy,cap,monitored,es_chk,reps in POLICIES:
            for rep in range(reps):
                batch={
                    v: rows[(osname,ds,seed,policy,rep,v)]
                    for v in VARIANTS
                }
                if set(int(x["order_index"]) for x in batch.values())!={0,1,2}:
                    raise AssertionError("Missing balanced triplet order")
                for field in (
                    "iterations","stop_reason","checks","stop_max",
                    "stalled_checks",
                ):
                    values={r[field] for r in batch.values()}
                    if len(values)!=1:
                        raise AssertionError(f"Stop metadata diverged in {field}")
                for r in batch.values():
                    if not (0<int(r["iterations"])<=cap):
                        raise AssertionError("Bad iteration metadata")
                    if policy in ("forced_monitor_20k","no_stop_5k") and (
                            int(r["iterations"])!=cap):
                        raise AssertionError("Unexpected fixed workload shortening")
                a,b,c=[batch[v] for v in VARIANTS]
                if b["source_commit"]!=c["source_commit"]:
                    raise AssertionError("Not the same candidate commit")
                if b["extension_sha256"]!=c["extension_sha256"]:
                    raise AssertionError("Opt-in and default-off binaries differ")
                if (a["source_commit"]==b["source_commit"]
                        or a["extension_sha256"]==b["extension_sha256"]):
                    raise AssertionError("Not independent master/candidate builds")
                master_commits.add(a["source_commit"])
                candidate_commits.add(b["source_commit"])
                base=float(a["elapsed_seconds"])
                off=float(b["elapsed_seconds"])
                on=float(c["elapsed_seconds"])
                if min(base,off,on)<=0:
                    raise AssertionError("Invalid elapsed time")
                saving_off=100*(1-off/base)
                saving_on=100*(1-on/base)
                for r in batch.values():
                    if any(abs(float(r[key])-value)>1e-8 for key,value in (
                        ("candidate_off_saving_percent",saving_off),
                        ("candidate_on_saving_percent",saving_on),
                        ("master_seconds",base),
                        ("candidate_off_seconds",off),
                        ("candidate_on_seconds",on),
                    )):
                        raise AssertionError("Mismatch in saved timing calculations")
                samples[(osname,ds,seed,policy,rep)]=dict(
                    off=saving_off,on=saving_on)
    if len(samples)!=80:
        raise AssertionError("Expected exactly 80 triple-mode pairs")
    if len(master_commits)!=1 or len(candidate_commits)!=1:
        raise AssertionError(
            "GitHub baseline/candidate SHAs drifted between jobs")
    return samples, next(iter(master_commits)), next(iter(candidate_commits))


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--files",required=True,nargs="+")
    parser.add_argument("--output",required=True)
    args=parser.parse_args(argv)
    samples,master_sha,candidate_sha=validate(args.files)
    lines=[
        "# Separate binary baseline vs clean-master Cython cache",
        "",
        "## Data-integrity and source isolation",
        "",
        "**240/240 separately invoked native Cython fits; 80/80 strict "
        "master/candidate-off/candidate-on exact array and complete-stop-info "
        "triplets passed.** All raw cases retained. Two independent native "
        "extensions compiled in different checkouts; each run verifies the "
        "loaded extension path and the aggregator checks distinct native "
        "SHA-256 hashes and git commit IDs.",
        f"- Untouched `master` baseline commit: `{master_sha}`.",
        f"- Clean candidate commit: `{candidate_sha}`.",
        "- Conditions: 2 OS × Digits/Breast Cancer × 4 new seeds "
        "8501/8609/8707/8803 × three variants; "
        "three repeated 20k fixed-check comparisons plus "
        "one 100k default-stop and one 5k no-stop case per seed.",
        "",
        "## Native-only propagation speed savings against untouched master",
        "",
        "| OS | Dataset | Regime | Candidate default-off median | "
        "Candidate cache-on median | On mean across raw pairs |",
        "|---|---|---|---:|---:|---:|",
    ]
    medians={}
    for osname,ds in product(OSES,DATASETS):
        for policy,_,_,_,reps in POLICIES:
            off_seed_medians=[]
            on_seed_medians=[]
            raw_on=[]
            for seed in SEEDS:
                offs=[samples[(osname,ds,seed,policy,rep)]["off"]
                      for rep in range(reps)]
                ons=[samples[(osname,ds,seed,policy,rep)]["on"]
                     for rep in range(reps)]
                off_seed_medians.append(median(offs))
                on_seed_medians.append(median(ons))
                raw_on.extend(ons)
            offmed=median(off_seed_medians)
            onmed=median(on_seed_medians)
            medians[(osname,ds,policy)]=(offmed,onmed)
            lines.append(
                f"| {osname} | {ds} | {policy} | "
                f"{offmed:+.2f}% | {onmed:+.2f}% | "
                f"{mean(raw_on):+.2f}% |"
            )
    speed_digits=[
        medians[(o,"digits","forced_monitor_20k")][1] for o in OSES]
    speed_breast=[
        medians[(o,"breast_cancer","forced_monitor_20k")][1] for o in OSES]
    speed_off=[
        medians[(o,d,"forced_monitor_20k")][0]
        for o,d in product(OSES,DATASETS)
    ]
    passes=(
        all(x>=15 for x in speed_digits)
        and all(x>=-5 for x in speed_breast)
        and all(x>=-8 for x in speed_off)
    )
    lines.extend([
        "",
        "## Frozen exploratory gate",
        "",
        f"- Cache on Digits >=+15% (Linux/Windows): "
        f"**{speed_digits[0]:+.2f}% / {speed_digits[1]:+.2f}%**.",
        f"- Cache on Breast Cancer >=-5% (Linux/Windows): "
        f"**{speed_breast[0]:+.2f}% / {speed_breast[1]:+.2f}%**.",
        "- Default-off must not be worse than -8% on any OS×dataset "
        f"(worst observed **{min(speed_off):+.2f}%**).",
        f"- **{'PASS: merits focused production patch review, no automatic merge' if passes else 'FAIL: investigate independent-binary regression before any merge'}**.",
        "",
        "## Interpretation",
        "",
        "This checks actual independently compiled untouched-master and "
        "clean-candidate native extensions, not variants within one binary. "
        "The kernel call alone is timed, so these are propagation savings "
        "rather than whole-model `fit_predict` savings. Repeated runner "
        "measurements mitigate but do not remove host scheduling noise. "
        "Do not infer universal CPU speedups from two GitHub-hosted OSes. "
        "All exact six-state-array, predictions and full stop metadata "
        "comparisons are mandatory and independent of performance. "
        "Do not merge unless a minimal-source review, memory-lifetime "
        "audit and regression checks on the clean PR are satisfactory.",
        "",
    ])
    Path(args.output).write_text("\n".join(lines),encoding="utf-8")
    print("\n".join(lines[:23]),flush=True)


if __name__=="__main__":
    main()
