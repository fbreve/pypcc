"""Source-exact original-vs-candidate Cython whole-fit experiment.

No production source files are changed. Each variant is compiled as a separate
real native extension from the protected pinned Git objects on each host.
"""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import tracemalloc

BASE = "4c1c410d8eb04547421f72ee1d3ca2eeb5763ff2"
SOURCE_FILES = (
    "pcc.py", "pcc_step.pyx", "pcc_graph.py", "pcc_numpy.py",
    "pcc_numba.py", "pcc_validation.py", "pcc_metrics.py",
    "setup.py", "pyproject.toml", "VERSION",
)
MASKS = (89021, 89119)
COHORTS = tuple(
    (name, count, monitored)
    for name, counts, modes in (
        ("ring100k50", (0, 5), (True,)),
        ("ring30k50", (0, 5), (True,)),
        ("digits", (2001,), (True, False)),
        ("wine", (2001,), (True, False)),
        ("breast_cancer", (2001,), (True, False)),
    )
    for count in counts for monitored in modes
)
BATCH = 8
BLOCKS_PER_PROCESS = 3
SEQ = ("original", "candidate", "candidate", "original")


def ensure(test, message):
    if not test:
        raise AssertionError(message)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def git_object(path):
    return subprocess.run(
        ["git", "show", BASE + ":" + path], check=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE
    ).stdout


def exact_change(source, before, after, key):
    ensure(source.count(before) == 1, "source drift: " + key +
           " matches=" + str(source.count(before)))
    return source.replace(before, after, 1)


def produce_candidate_pyx(content):
    initializer = """
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef void _pcc_initialize_rowmax_from_canonical_labels(
    np.int64_t[:] labels, Py_ssize_t n_nodes, double uniform_value,
    double[:] row_max_cache
) noexcept nogil:
    cdef Py_ssize_t i
    for i in range(n_nodes):
        if labels[i] == -1:
            row_max_cache[i] = uniform_value
        else:
            row_max_cache[i] = 1.0

"""
    content = exact_change(
        content, "\ncpdef pcc_propagate(\n",
        "\n" + initializer + "cpdef pcc_propagate(\n",
        "new O(n) canonical initializer insertion"
    )
    content = exact_change(
        content,
        "    bint             experimental_verify_rowmax_cache = False\n):",
        "    bint             experimental_verify_rowmax_cache = False,\n"
        "    bint             experimental_canonical_rowmax_init = False\n):",
        "append default-false low-level opt-in"
    )
    content = exact_change(
        content,
        "    if cache_enabled:\n"
        "        row_max_cache = np.empty(n_nodes, dtype=np.float64)\n"
        "        with nogil:\n"
        "            _pcc_initialize_rowmax_cache(dominance, n_nodes, c, row_max_cache)",
        "    if cache_enabled:\n"
        "        row_max_cache = np.empty(n_nodes, dtype=np.float64)\n"
        "        with nogil:\n"
        "            if experimental_canonical_rowmax_init:\n"
        "                _pcc_initialize_rowmax_from_canonical_labels(\n"
        "                    labels, n_nodes, 1.0 / c, row_max_cache)\n"
        "            else:\n"
        "                _pcc_initialize_rowmax_cache(\n"
        "                    dominance, n_nodes, c, row_max_cache)",
        "single original cache initialization replacement"
    )
    return content


def produce_candidate_py(content):
    addition = """        canonical_init_kwargs = {}
        # Only this exact class just created canonical one-hot/uniform dominance.
        # Direct low-level callers and custom subclass states remain unchanged.
        if (type(self) is ParticleCompetitionAndCooperation and
                self.early_stop and self.update_mode == "sequential" and
                _HAS_CYTHON):
            from pcc_step import pcc_propagate as actual_cython_propagate
            if propagate_fn is actual_cython_propagate:
                canonical_init_kwargs["experimental_canonical_rowmax_init"] = True

"""
    content = exact_change(
        content,
        "        propagate_fn(\n            self.neib_list, self.neib_qt,",
        addition + "        propagate_fn(\n            self.neib_list, self.neib_qt,",
        "canonical high-level real Cython identity"
    )
    content = exact_change(
        content,
        '            **({"seed": seed} if seed is not None else {})\n',
        '            **({"seed": seed} if seed is not None else {}),\n'
        '            **canonical_init_kwargs\n',
        "canonical high-level named keyword"
    )
    return content


def prepare(work):
    work = Path(work).resolve()
    ensure(not (work / "original").exists() and
           not (work / "candidate").exists(),
           "refusing to overwrite compiled native sources")
    # Verify every current production source matches the pinned original Git.
    for file in SOURCE_FILES:
        # Windows Git checkout may contain CRLF while the immutable Git blob
        # uses LF. Compare canonical Git bytes, and separately use Git's own
        # worktree-normalized diff for unstaged local production modifications.
        expected = git_object(file)
        actual = subprocess.run(
            ["git", "show", "HEAD:" + file], check=True,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE
        ).stdout
        ensure(actual == expected, "protected production Git blob drift: " + file)
        clean = subprocess.run(["git", "diff", "--quiet", "HEAD", "--", file],
                               check=False).returncode
        ensure(clean == 0, "protected production Git worktree changed: " + file)
    records = {}
    for variant in ("original", "candidate"):
        folder = work / variant
        folder.mkdir(parents=True)
        records[variant] = {}
        for file in SOURCE_FILES:
            original = git_object(file)
            result = original
            if variant == "candidate":
                if file == "pcc_step.pyx":
                    result = produce_candidate_pyx(original.decode("utf-8")).encode("utf-8")
                elif file == "pcc.py":
                    result = produce_candidate_py(original.decode("utf-8")).encode("utf-8")
            (folder / file).write_bytes(result)
            records[variant][file] = sha(result)
        env = os.environ.copy()
        env["PYTHONPATH"] = str(folder) + os.pathsep + str(Path.cwd())
        subprocess.run(
            [sys.executable, "setup.py", "build_ext", "--inplace"],
            cwd=str(folder), env=env, check=True, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT
        )
        ensure(any(folder.glob("pcc_step*.so")) or
               any(folder.glob("pcc_step*.pyd")),
               "real Cython native extension was not compiled: " + variant)
        print("NATIVE_CYTHON_VARIANT_COMPILED", variant, flush=True)
    ensure(records["original"]["pcc.py"] != records["candidate"]["pcc.py"] and
           records["original"]["pcc_step.pyx"] != records["candidate"]["pcc_step.pyx"],
           "candidate failed to change exactly expected source files")
    for path in SOURCE_FILES:
        if path not in ("pcc.py", "pcc_step.pyx"):
            ensure(records["original"][path] == records["candidate"][path],
                   "forbidden source modification: " + path)
    payload = {
        "schema": 1, "base": BASE, "files": records,
        "candidate_only_changes": ["pcc.py", "pcc_step.pyx"],
        "distinct_original_native_binary": True,
        "no_production_file_changes": True,
    }
    (work / "native_source_manifest.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    return payload


def get_case(case_id):
    return next((cohort for cohort in COHORTS
                 if cohort_id(cohort) == case_id), None)


def cohort_id(cohort):
    name, count, monitored = cohort
    return name + "_it" + str(count) + ("_monitored" if monitored else "_fixed")


def run_measure(opts):
    from benchmarks.fixtures import Case, REAL_DATASETS, prepare_fixture, array_hash
    import numpy as np
    import pcc
    import pcc_step
    from pcc import ParticleCompetitionAndCooperation as PCC

    src = Path.cwd().resolve()
    ensure(Path(pcc.__file__).resolve().parent == src,
           "Python model source not isolated to variant directory")
    ensure(Path(pcc_step.__file__).resolve().parent == src and
           not pcc_step.__file__.endswith(".py"),
           "Cython native extension not isolated to variant directory")
    cohort = get_case(opts.case)
    ensure(cohort is not None and opts.mask in MASKS and opts.variant in ("original", "candidate"),
           "unfrozen case/mask/variant")
    name, steps, monitored = cohort
    if name.startswith("ring"):
        n = 100000 if name == "ring100k50" else 30000
        c = 50
        degree = 16
        indices = np.arange(n, dtype=np.int64)
        offsets = np.concatenate((np.arange(1, 9), -np.arange(1, 9)))
        neighbors = np.ascontiguousarray((indices[:, None] + offsets[None, :]) % n)
        degrees = np.full(n, degree, dtype=np.int64)
        truth = indices % c
        rng = np.random.default_rng(opts.mask)
        labels = np.full(n, -1, dtype=np.int64)
        for cls in range(c):
            labeled = rng.choice(indices[truth == cls], size=3, replace=False)
            labels[labeled] = cls
        k_nn = degree
    else:
        n, features, c, _ = REAL_DATASETS[name]
        case = Case(name, n, features, c, k_nn=10,
                    labels_per_class=3, data_seed=0, label_seed=opts.mask)
        arrays, metadata = prepare_fixture(case)
        labels = arrays["labels"]
        neighbors, degrees = arrays["neighbors"], arrays["degrees"]
        k_nn = 10
    input_hashes = {
        "graph": array_hash(neighbors), "degrees": array_hash(degrees),
        "labels": array_hash(labels)
    }
    ensure(len(np.unique(labels[labels != -1])) == c,
           "original class coverage not preserved")
    ensure(np.all(degrees > 0) and np.all(degrees <= neighbors.shape[1]),
           "invalid undirected original degrees")
    if name.startswith("ring"):
        ensure(np.all(neighbors[:, :8] == ((indices[:, None] + np.arange(1, 9)) % n)),
               "ring forward edges altered")
        ensure(np.all(neighbors[:, 8:] == ((indices[:, None] - np.arange(1, 9)) % n)),
               "ring reciprocal edges altered")

    def fit(model):
        return model.fit_predict(
            labels, p_grd=0.5, delta_v=0.1, deltap=1.0, dexp=2.0,
            max_iter=steps, early_stop=monitored, es_chk=2000, seed=29
        )

    def new_model():
        model = PCC(impl="cython", update_mode="sequential")
        model.set_graph(neighbors, degrees, k_nn=k_nn)
        ensure(model._get_backend_fn() is pcc_step.pcc_propagate,
               "real native Cython backend fallback is forbidden")
        return model

    def model_state(model, prediction):
        return {
            "predictions": array_hash(prediction),
            "dominance": array_hash(model.node.dominance),
            "node_labels": array_hash(model.node.label),
            "part_curnode": array_hash(model.part.curnode),
            "part_label": array_hash(model.part.label),
            "part_strength": array_hash(model.part.strength),
            "dist_table": array_hash(model.part.dist_table),
            "owndeg": array_hash(model.owndeg),
            "mapped_labels": array_hash(model.mapped_labels),
            "part_homenode": array_hash(model.part.homenode),
        }

    # The normal high-level backend is captured read-only for one scientific
    # state/stop smoke, then restored before ANY CPU measurements.
    actual_native = pcc_step.pcc_propagate
    actual_calls = []

    def inspector(*args, **kwargs):
        expected = opts.variant == "candidate" and monitored
        received = kwargs.get("experimental_canonical_rowmax_init", False)
        ensure(received is expected, "actual canonical fast path not exclusively activated")
        kwargs["return_info"] = True
        actual_calls.append(actual_native(*args, **kwargs))
        return actual_calls[-1]

    inspector.__name__ = "pcc_propagate"
    pcc_step.pcc_propagate = inspector
    try:
        validation_model = new_model()
        validation_pred = fit(validation_model)
        ensure(len(actual_calls) == 1, "not one actual source native propagation")
        smoke_hashes = model_state(validation_model, validation_pred)
        original_stop = actual_calls[0]
    finally:
        pcc_step.pcc_propagate = actual_native
    ensure(validation_model._get_backend_fn() is actual_native,
           "original compiled native backend not restored")
    expected_initial = np.where(labels == -1, 1.0 / c, 1.0)
    initial = np.full((n, c), 1.0 / c, dtype=np.float64, order="F")
    known = np.flatnonzero(labels != -1)
    initial[known, :] = 0.0
    initial[known, labels[known]] = 1.0
    ensure(np.array_equal(np.max(initial, axis=1), expected_initial),
           "canonical rowmax formula differs from original float64 initialized dominance")

    # No-opt-in low-level call with explicitly NONCANONICAL dominance.
    # Verifier checks the actual cache against each full row-scan checkpoint.
    raw_dom = validation_model.node.dominance.copy(order="F")
    raw_dom[0, :] = np.linspace(0.01, 0.25, c, dtype=np.float64)
    def direct_cache_verify(dom, canonical_flag):
        kw = dict(return_info=True, seed=29,
                  experimental_rowmax_cache=True,
                  experimental_verify_rowmax_cache=True)
        if opts.variant == "candidate" and canonical_flag:
            kw["experimental_canonical_rowmax_init"] = True
        p = validation_model.part
        result = actual_native(
            validation_model.neib_list, validation_model.neib_qt,
            validation_model.mapped_labels, .5, .1, c, validation_model.zerovec,
            p.curnode.copy(), p.label.copy(), p.strength.copy(),
            p.dist_table.copy(order="F"), dom.copy(order="F"),
            validation_model.owndeg.copy(order="F"), 1.0, 2.0,
            validation_model.buf_dom_row.copy(), validation_model.buf_reduc.copy(),
            validation_model.buf_dom_list.copy(), validation_model.buf_dist_list.copy(),
            validation_model.buf_prob.copy(), validation_model.buf_slices.copy(),
            validation_model.dist_weights.copy(), 21, True, 2000, 1, **kw
        )
        return result
    no_optin_info = direct_cache_verify(raw_dom, False)
    canonical_info = direct_cache_verify(initial, True)
    # Fit independence: a second normal fit must NOT mutate the first state.
    before = validation_model.node.dominance
    before_sha = array_hash(before)
    second_pred = fit(validation_model)
    ensure(array_hash(before) == before_sha and
           before is not validation_model.node.dominance,
           "previous-fit dominance ownership was mutated")
    ensure(model_state(validation_model, second_pred) == smoke_hashes,
           "seeded repeated full fits are not exact source replays")
    old_parts, old_node = validation_model.part, validation_model.node
    bad_labels = labels.astype(np.float64, copy=True)
    bad_labels[0] = float("nan")
    try:
        validation_model.fit_predict(bad_labels, seed=29)
    except ValueError:
        pass
    else:
        raise AssertionError("invalid-input rollback failed to reject nonfinite label")
    ensure(validation_model.part is old_parts and validation_model.node is old_node,
           "invalid-input rollback destroyed fitted state")

    # Normal, unobserved complete public fits; warm outside every timed batch.
    model = new_model()
    _ = fit(model)
    gc.collect()
    samples = []
    for block in range(BLOCKS_PER_PROCESS):
        gc.collect()
        t0 = time.perf_counter_ns()
        cpu0 = time.process_time_ns()
        for repeat in range(BATCH):
            prediction = fit(model)
        cpu1 = time.process_time_ns()
        t1 = time.perf_counter_ns()
        ensure(cpu1 > cpu0 and t1 > t0, "nonpositive full-fit timing interval")
        block_state = model_state(model, prediction)
        ensure(block_state == smoke_hashes,
               "unobserved true public fit changed final state or output")
        samples.append({
            "block": block, "fit_count": BATCH,
            "cpu_ns": cpu1-cpu0, "wall_ns": t1-t0,
            "last_state_hashes": block_state,
        })

    # Separate untimed peak measurement; do not include in CPU timing.
    tracemalloc.start()
    _ = fit(model)
    current_bytes, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    ensure(model_state(model, _) == smoke_hashes,
           "peak-memory trace affected model output")
    result = {
        "schema": 1, "base": BASE, "variant": opts.variant,
        "host": platform.system().lower(),
        "python": platform.python_version(), "cohort": opts.case,
        "mask_seed": opts.mask, "iterations": steps, "early_stop": monitored,
        "c": c, "n_nodes": n, "k_nn": k_nn,
        "source_hashes": {
            "pcc.py": sha(Path("pcc.py").read_bytes()),
            "pcc_step.pyx": sha(Path("pcc_step.pyx").read_bytes())
        },
        "inputs": input_hashes, "state": smoke_hashes,
        "stop": original_stop,
        "direct_noncanonical_default_cache_verifier": no_optin_info,
        "direct_canonical_cache_verifier": canonical_info,
        "previous_fit_ownership_and_invalid_rollback": True,
        "compiled_real_cython_backend": True,
        "new_fast_path_only_in_candidate_highlevel_cached": True,
        "batch_size": BATCH, "cpu_batches": samples,
        "python_traced_peak_bytes": int(peak_bytes),
        "not_native_rss": True,
        "pid": os.getpid(),
    }
    out = Path(opts.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n",
                   encoding="utf-8")
    print("FROZEN_NATIVE_CYTHON_SAMPLED", opts.variant, opts.case, opts.mask,
          "stop", original_stop["stop_reason"], "cpu", [x["cpu_ns"] for x in samples],
          flush=True)


def orchestrate(opts):
    root = Path.cwd().resolve()
    out = Path(opts.out).resolve()
    tree_dir = out / "native_sources"
    manifest = prepare(tree_dir)
    logs = out / "measurements"
    logs.mkdir(parents=True, exist_ok=True)
    count = 0
    for cohort in COHORTS:
        cid = cohort_id(cohort)
        for mask in MASKS:
            for round_no, variant in enumerate(SEQ):
                source = tree_dir / variant
                output = logs / (cid + "_mask" + str(mask) + "_round" +
                                 str(round_no) + "_" + variant + ".json")
                env = os.environ.copy()
                env["PYTHONPATH"] = str(source) + os.pathsep + str(root)
                env.update(OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1",
                           MKL_NUM_THREADS="1", NUMBA_NUM_THREADS="1")
                cmd = [
                    sys.executable, "-m",
                    "experiments.cython_canonical_rowmax_init_probe",
                    "measure", "--variant", variant, "--case", cid,
                    "--mask", str(mask), "--out", str(output)
                ]
                subprocess.run(cmd, cwd=str(source), env=env, check=True)
                count += 1
    ensure(count == len(COHORTS)*len(MASKS)*len(SEQ),
           "lost a preregistered native variant run")
    (out / "complete.json").write_text(json.dumps({
        "base": BASE, "manifest": manifest, "runs": count,
        "cohorts": [cohort_id(c) for c in COHORTS],
        "masks": MASKS, "source_order": SEQ,
        "host": platform.system().lower(),
    }, indent=2) + "\n", encoding="utf-8")
    print("BOTH_CYTHON_VARIANTS_ALL_COHORTS_NATIVE_COMPLETE",count,flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("operation", choices=("orchestrate","measure"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--variant", choices=("original","candidate"))
    ap.add_argument("--case")
    ap.add_argument("--mask", type=int)
    args = ap.parse_args()
    if args.operation == "orchestrate":
        orchestrate(args)
    else:
        run_measure(args)


if __name__ == "__main__":
    main()
