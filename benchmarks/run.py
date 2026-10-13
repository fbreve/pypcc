"""Run with python -m benchmarks.run; defaults to a read-only workload plan."""
import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
import sys
import time

from .fixtures import (Case, REAL_DATASETS, SYNTHETIC_DEFAULTS, prepare_fixture,
                       prepare_mnist, save_fixture)
from .historical import ENTRIES, load_historical
from .runner import (PARAMETERS, environment_manifest, expected_backend, measure,
                     seed_backend, summarize, write_json)


def execution_order(backends, case_index, repeat):
    offset = (case_index + repeat) % len(backends)
    return backends[offset:] + backends[:offset]


def plan(cases, backends, repeats, iterations):
    rows = []
    for case in cases:
        case.validate()
        rows.append(dict(case_id=case.case_id, nodes=case.nodes, features=case.features,
                         classes=case.classes, labeled_nodes=case.particles, k_nn=case.k_nn,
                         measured_executions=len(backends)*repeats,
                         warmup_executions=2*len(backends), iteration_cap=iterations,
                         measured_iteration_cap=iterations*len(backends)*repeats,
                         particle_update_cap=iterations*case.particles*len(backends)*repeats,
                         graph_pair_feature_upper_work=case.nodes**2*case.features,
                         distance_table_bytes=case.nodes*case.particles,
                         feature_arrays_bytes=2*case.nodes*case.features*8,
                         dominance_bytes=case.nodes*case.classes*8,
                         historical_source=ENTRIES.get(case.name) if not case.synthetic else None))
    updates = sum(row["particle_update_cap"] for row in rows)
    return dict(cases=rows, case_count=len(rows), backends=backends, repetitions=repeats,
                measured_executions=sum(row["measured_executions"] for row in rows),
                warmup_executions=sum(row["warmup_executions"] for row in rows),
                measured_iteration_cap=sum(row["measured_iteration_cap"] for row in rows),
                particle_update_cap=updates,
                requires_expensive_opt_in=any(case.nodes > 2000 or case.nodes**2*case.features > 1000000000
                                             for case in cases) or updates > 20000000,
                estimate_note="Work caps, not predicted wall time. Graph upper work is N^2*features, not an operation count. Early stopping may reduce work; greedy degree/class costs vary. Warmups use 2 iterations each. No loading, graph construction, download or compilation in dry-run.")


def select_cases(args):
    specs = []
    if args.historical and (args.datasets is not None or args.synthetic or args.synthetic_config):
        raise ValueError("--historical selects only the frozen group; select other experiments separately")
    names = (list(ENTRIES) if args.historical else
             args.datasets if args.datasets is not None else
             ["iris", "wine", "breast_cancer", "digits"])
    for name in names:
        nodes, features, classes, _ = REAL_DATASETS[name]
        specs.append(Case(name, nodes, features, classes, k_nn=args.k_nn,
                          labels_per_class=None if args.labeled_fraction is not None else args.labels_per_class,
                          labeled_fraction=args.labeled_fraction, data_seed=args.data_seed))
    if args.synthetic:
        specs.extend(SYNTHETIC_DEFAULTS)
    if args.synthetic_config:
        raw = json.loads(args.synthetic_config.read_text(encoding="utf-8"))
        if not isinstance(raw, list):
            raise ValueError("synthetic config must be a JSON list of Case configurations")
        for item in raw:
            item = dict(item)
            item["synthetic"] = True
            if "labeled_fraction" in item and "labels_per_class" not in item:
                item["labels_per_class"] = None
            specs.append(Case(**item))
    seeds = args.label_seeds if args.label_seeds is not None else ([10] if args.historical else [0])
    cases = [replace(case, label_seed=seed) for case in specs for seed in seeds]
    if not cases or len({case.case_id for case in cases}) != len(cases):
        raise ValueError("require at least one case and unique dataset names/label seeds")
    return cases


def execute(args, cases, workload):
    if workload["requires_expensive_opt_in"] and not args.allow_expensive:
        raise ValueError("workload requires --allow-expensive; review the dry-run before opting in")
    if args.output is None:
        raise ValueError("--output must name a new directory for --execute")
    # Validate all requested backends before preparing any datasets.
    for name in args.backends:
        expected_backend(name)
    build_info = None
    if args.build_info:
        build_info = json.loads(args.build_info.read_text(encoding="utf-8"))
    manifest = environment_manifest(dict(workload=workload, cases=[asdict(case) for case in cases],
                                         mode=args.mode, parameters=PARAMETERS,
                                         historical_cache=str(args.historical_cache) if args.historical_cache else None,
                                         propagation_seed=args.propagation_seed,
                                         mnist_cache=str(args.mnist_cache) if args.mnist_cache else None),
                                    args.backends, build_info)
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output / "manifest.json", manifest)
    rows, warmups, fixtures, skipped = [], [], [], []
    try:
        with (args.output / "measurements.jsonl").open("x", encoding="utf-8") as stream, \
                (args.output / "warmups.jsonl").open("x", encoding="utf-8") as warm_stream:
            if "numba" in args.backends:
                start = time.perf_counter()
                seed_backend("numba", args.propagation_seed)
                write_json(args.output / "seed_helper.json", {"first_call_s": time.perf_counter()-start})
            for case_index, case in enumerate(cases):
                try:
                    arrays, metadata = prepare_fixture(case, args.mnist_cache, args.historical_cache)
                except FileNotFoundError as exc:
                    if case.name.startswith("mnist_") and args.skip_missing_optional:
                        skipped.append(dict(case_id=case.case_id, reason=str(exc)))
                        print(f"Skipped {case.case_id}: {exc}", file=sys.stderr)
                        continue
                    raise
                save_fixture(args.output / "fixtures" / case.case_id, arrays, metadata)
                fixtures.append(dict(case_id=case.case_id, fixture_sha256=metadata["fixture_sha256"]))
                for name in args.backends:
                    for index in range(2):
                        warm = measure(arrays, case, name, 2, args.mode == "early_stopping",
                                       args.propagation_seed, steady_state=False)
                        warm.update(case_id=case.case_id, call="first" if index == 0 else "warm",
                                    fixture_sha256=metadata["fixture_sha256"])
                        warmups.append(warm)
                        warm_stream.write(json.dumps(warm, allow_nan=False)+"\n")
                        warm_stream.flush()
                reference_states = {}
                for repeat in range(args.repeats):
                    for order_index, name in enumerate(execution_order(args.backends, case_index, repeat)):
                        row = measure(arrays, case, name, args.iterations,
                                      args.mode == "early_stopping", args.propagation_seed)
                        identity = (row["state_hashes"], row["effective_iterations"], row["stop_reason"])
                        if name in reference_states and identity != reference_states[name]:
                            raise RuntimeError(f"nonrepeatable state/metrics for {case.case_id}/{name}")
                        reference_states[name] = identity
                        row.update(case_id=case.case_id, repeat=repeat, order_index=order_index,
                                   fixture_sha256=metadata["fixture_sha256"], labeled_nodes=metadata["labeled_nodes"])
                        stream.write(json.dumps(row, allow_nan=False)+"\n")
                        stream.flush()
                        rows.append(row)
                        print(f"{case.case_id} {name} repetition {repeat+1}/{args.repeats}: "
                              f"{row['effective_iterations']} iterations", flush=True)
        write_json(args.output / "summary.json", summarize(rows))
        write_json(args.output / "completion.json", dict(status="complete" if rows else "no_measurements",
                                                        measured_executions=len(rows), fixtures=fixtures, skipped=skipped))
    except Exception as exc:
        write_json(args.output / "failure.json", dict(status="failed", exception=type(exc).__name__,
                                                     message=str(exc), measured_executions=len(rows), skipped=skipped))
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare-mnist", type=Path, metavar="NEW_CACHE", help="explicit download only; no benchmark")
    parser.add_argument("--datasets", nargs="*", choices=REAL_DATASETS,
                        default=None)
    parser.add_argument("--historical", action="store_true", help="exact frozen 20-dataset group; excludes MNIST")
    parser.add_argument("--historical-cache", type=Path)
    parser.add_argument("--prepare-historical", type=Path, metavar="CACHE", help="explicit source download/verification only")
    parser.add_argument("--synthetic", action="store_true", help="include three small representative synthetic cases")
    parser.add_argument("--synthetic-config", type=Path)
    parser.add_argument("--backends", nargs="+", choices=["numpy", "numba", "cython"], default=["numpy", "numba", "cython"])
    parser.add_argument("--mode", choices=["fixed", "early_stopping"], default="fixed")
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=6)
    parser.add_argument("--label-seeds", nargs="+", type=int, default=None)
    parser.add_argument("--data-seed", type=int, default=0)
    parser.add_argument("--propagation-seed", type=int, default=0)
    label_group = parser.add_mutually_exclusive_group()
    label_group.add_argument("--labels-per-class", type=int, default=3)
    label_group.add_argument("--labeled-fraction", type=float)
    parser.add_argument("--k-nn", type=int, default=10)
    parser.add_argument("--mnist-cache", type=Path)
    parser.add_argument("--skip-missing-optional", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--build-info", type=Path, help="optional JSON record of compiler, flags, command and build log hash")
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--execute", action="store_true")
    action.add_argument("--dry-run", action="store_true")
    parser.add_argument("--allow-expensive", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.prepare_historical:
            if args.execute or args.prepare_mnist:
                raise ValueError("preparation and execution must be separate")
            names = list(ENTRIES) if args.historical else (args.datasets or ["iris", "wine", "breast_cancer", "digits"])
            if any(name not in ENTRIES for name in names):
                raise ValueError("historical preparation excludes MNIST")
            for name in names:
                load_historical(name, args.prepare_historical, download=True)
            return
        if args.prepare_mnist:
            if args.execute or args.historical:
                raise ValueError("dataset preparation and benchmark execution must be separate commands")
            prepare_mnist(args.prepare_mnist)
            return
        if not 1 <= args.iterations <= 2147483647 or args.repeats < 1:
            raise ValueError("require positive repeats and iterations <= 2147483647")
        if not 0 <= args.propagation_seed < 2**32:
            raise ValueError("propagation seed must be an unsigned 32-bit integer")
        if len(set(args.backends)) != len(args.backends):
            raise ValueError("backend selections must be unique")
        cases = select_cases(args)
        workload = plan(cases, args.backends, args.repeats, args.iterations)
        workload["mode"] = args.mode
        print(json.dumps(workload, indent=2), flush=True)
        if args.execute:
            execute(args, cases, workload)
    except (ValueError, FileExistsError, FileNotFoundError, RuntimeError, ImportError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
