"""Source-exact original PCC non-native fit phase observer.

Scientific-only: full real master pcc.py is loaded twice; only the observed
copy has wall-clock boundary statements inserted. Production modules unchanged.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time
import types
import numpy as np
from numba import njit
from sklearn import datasets
from sklearn.preprocessing import StandardScaler

from benchmarks.fixtures import array_hash
from benchmarks.runner import expected_backend
from pcc import ParticleCompetitionAndCooperation as PCC
from pcc_graph import build_knn_graph

BASE = "b4c100bd22eeff1979a1a581bc73682b8c3705b8"
PROTECTED = ("pcc.py", "pcc_graph.py", "pcc_numba.py", "pcc_numpy.py",
             "pcc_step.pyx", "pcc_validation.py", "setup.py")
MASK_SEEDS = (83013, 83117)
REAL = ("wine", "breast_cancer", "digits")
CASES = (*REAL, "ring_30000_10", "ring_100000_50")
BACKENDS = ("numpy", "numba", "cython")
PHASES = ("preflight", "classes_backend", "model_setup", "label_mapping",
          "parameters", "particles", "nodes", "distance_weights",
          "scratch_allocation", "stop_preflight", "dispatch", "native",
          "unlabeled_assignment", "normalization", "inverse_labels")
STATE = ("predictions", "dominance", "positions", "strengths",
         "distances", "owndeg")
N_PAIRS = 4
FITS_PER_BATCH = 8
PARAMS = dict(p_grd=.5, delta_v=.1, deltap=1., dexp=2.,
              es_chk=2000, early_stop=False, seed=29)


@njit
def next_numba():
    return np.random.random()


def check_original_git_source():
    root = Path(__file__).resolve().parents[1]
    for path in PROTECTED:
        blob = subprocess.check_output(["git", "show", BASE + ":" + path])
        current = (root / path).read_bytes()
        # The Git blob is LF; Windows checkout can expose CRLF.
        assert current.replace(b"\r\n", b"\n") == blob.replace(b"\r\n", b"\n"), path
    changed = subprocess.check_output(
        ["git", "diff", "--name-only", BASE, "HEAD", "--", *PROTECTED],
        text=True).strip()
    assert not changed, f"protected production source changed: {changed}"
    return subprocess.check_output(["git", "show", BASE + ":pcc.py"], text=True)


def instrument(original):
    """Insert clock boundaries without replacing/deleting original operations."""
    extra_method = (
        '    def _science_cut(self, stage):\n'
        '        now = _science_clock.perf_counter_ns()\n'
        '        self._science_times[stage] = now - self._science_last\n'
        '        self._science_last = now\n\n'
    )
    inserts = [
        ("    def fit_predict(self, labels,",
         extra_method),
        ("        if self.neib_list is None or self.neib_qt is None:\n",
         "        self._science_times = {}\n"
         "        self._science_last = _science_clock.perf_counter_ns()\n"),
        ("        prepared_labels = np.ascontiguousarray(labels, dtype=np.int64)\n",
         "        self._science_cut('preflight')\n"),
        ("        previous_state = self.__dict__.copy()\n",
         "        self._science_cut('classes_backend')\n"),
        ("            self.mapped_labels = np.full(self.labels.shape, -1, dtype=np.int64)\n",
         "            self._science_cut('model_setup')\n"),
        ("            self.p_grd = p_grd\n",
         "            self._science_cut('label_mapping')\n"),
        ("            self.part = self.__genParticles()\n",
         "            self._science_cut('parameters')\n"),
        ("            self.node = self.__genNodes()\n",
         "            self._science_cut('particles')\n"),
        ("            # Pre-calculate distance weights lookup table for 0..255 range\n",
         "            self._science_cut('nodes')\n"),
        ("            if seed is None:\n                self.__labelPropagation()\n",
         "            self._science_cut('distance_weights')\n"),
        ("            return final_labels\n",
         "            self._science_cut('inverse_labels')\n"),
        ("        early_stop = self.early_stop\n",
         "        self._science_cut('scratch_allocation')\n"),
        ("        # DIRECT call to the backend propagation function\n",
         "        self._science_cut('stop_preflight')\n"),
        ("        propagate_fn(\n",
         "        self._science_cut('dispatch')\n"),
        ("        unlabeled = node.label == -1\n",
         "        self._science_cut('native')\n"),
        ("        row_sums = np.sum(self.owndeg, axis=1, keepdims=True)\n",
         "        self._science_cut('unlabeled_assignment')\n"),
        ("    def __genParticles(self) -> Particles:\n",
         "        self._science_cut('normalization')\n\n"),
    ]
    # The final normalization marker must be at the end of labelPropagation,
    # just BEFORE the next original method def.
    source = original
    for anchor, added in inserts:
        if source.count(anchor) != 1:
            raise AssertionError(f"non-unique source insertion: {anchor!r}, occurrences={source.count(anchor)}")
        source = source.replace(anchor, added + anchor, 1)
    source = "import time as _science_clock\n" + source
    return source


def traced_class(original):
    source = instrument(original)
    mod = types.ModuleType("_pcc_fit_phase_scientific")
    mod.__file__ = "scientific-original-plus-clocks/pcc.py"
    sys.modules[mod.__name__] = mod
    exec(compile(source, mod.__file__, "exec"), mod.__dict__)
    return mod.ParticleCompetitionAndCooperation


def labels_for(truth, seed):
    rng = np.random.default_rng(seed)
    mask = np.full(len(truth), -1, dtype=np.int64)
    for cls in np.unique(truth):
        eligible = np.flatnonzero(truth == cls)
        assert len(eligible) > 3
        mask[rng.choice(eligible, 3, replace=False)] = cls
    return mask


def fixture(name):
    if name in REAL:
        dataset = getattr(datasets, "load_" + name)()
        truth = np.asarray(dataset.target, dtype=np.int64)
        x = StandardScaler().fit_transform(dataset.data)
        graph, degree = build_knn_graph(x, k_nn=10, n_jobs=1)
        k = 10
    else:
        _, n, classes = name.split("_")
        n, classes = int(n), int(classes)
        truth = np.arange(n, dtype=np.int64) % classes
        x = np.arange(n, dtype=np.int64)
        graph = np.stack(((x-1) % n, (x+1) % n), axis=1)
        degree = np.full(n, 2, dtype=np.int64)
        k = 2
        assert np.array_equal(graph[graph[:, 0], 1], x)
        assert np.array_equal(graph[graph[:, 1], 0], x)
    masks = [labels_for(truth, seed) for seed in MASK_SEEDS]
    assert array_hash(masks[0]) != array_hash(masks[1])
    return graph, degree, masks, k


def model(cls, backend, graph, degree, k):
    obj = cls(impl=backend)
    obj.set_graph(graph, degree, k_nn=k)
    assert obj._get_backend_fn() is expected_backend(backend)
    return obj


def state_refs(m, prediction):
    return (prediction, m.node.dominance, m.part.curnode,
            m.part.strength, m.part.dist_table, m.owndeg)


def sha_state(values):
    return dict(zip(STATE, (array_hash(a) for a in values)))


def run_batch(m, masks, steps, trace):
    retained = []
    phases = []
    start_cpu = time.process_time_ns()
    start_wall = time.perf_counter_ns()
    for i in range(FITS_PER_BATCH):
        ans = m.fit_predict(masks[i % 2], max_iter=steps, **PARAMS)
        retained.append(state_refs(m, ans))
        if trace:
            # Keep reference, not a dictionary copy/hash inside timed region.
            phases.append(m._science_times)
    wall_ns = time.perf_counter_ns() - start_wall
    cpu_ns = time.process_time_ns() - start_cpu
    all_sha = [sha_state(st) for st in retained]
    for left, right in zip(retained, retained[1:]):
        assert all(left[j] is not right[j] for j in (1, 4, 5)), \
            "previously returned fitted arrays not independent"
    if trace:
        for phase in phases:
            assert set(phase) == set(PHASES), (set(PHASES)-set(phase), set(phase)-set(PHASES))
            assert all(type(value) is int and value >= 0 for value in phase.values())
    return dict(cpu_ns=cpu_ns, wall_ns=wall_ns, states=all_sha,
                phase_ns=[dict(p) for p in phases])


def rng_sample(backend):
    if backend == "numba":
        return float(next_numba())
    if backend == "numpy":
        return float(np.random.random())
    return None


def independent_replay(a, b, masks, steps, expected):
    rng = []
    for k, mask in enumerate(masks):
        ap = a.fit_predict(mask, max_iter=steps, **PARAMS)
        ah = sha_state(state_refs(a, ap))
        ar = rng_sample(a.impl)
        bp = b.fit_predict(mask, max_iter=steps, **PARAMS)
        bh = sha_state(state_refs(b, bp))
        br = rng_sample(b.impl)
        assert ah == bh == expected[k], "source or Numba/NumPy RNG parity broken"
        assert ar == br, "native next seeded RNG mismatch"
        assert a.node.dominance.flags.c_contiguous == b.node.dominance.flags.c_contiguous
        assert a.node.dominance.flags.f_contiguous == b.node.dominance.flags.f_contiguous
        rng.append(ar)
    for m in (a, b):
        old = state_refs(m, masks[-1])
        old_sha = sha_state(old)
        try:
            m.fit_predict(masks[-1], seed=-1, max_iter=steps)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid seed accepted")
        current = state_refs(m, old[0])
        assert old_sha == sha_state(current)
        assert all(x is y for x, y in zip(old[1:], current[1:])), \
            "invalid-seed changed fitted array identity"
    return rng


def one_case(cls, name, backend, steps, smoke=False):
    graph, degree, masks, k = fixture(name)
    control = model(PCC, backend, graph, degree, k)
    observed = model(cls, backend, graph, degree, k)
    if backend == "numba":
        next_numba()
        control.fit_predict(masks[0], max_iter=2, **PARAMS)
        observed.fit_predict(masks[0], max_iter=2, **PARAMS)
    repeats = 1 if smoke else N_PAIRS
    rows = []
    for rep in range(repeats):
        if rep % 2 == 0:
            a = run_batch(control, masks, steps, False)
            b = run_batch(observed, masks, steps, True)
        else:
            b = run_batch(observed, masks, steps, True)
            a = run_batch(control, masks, steps, False)
        assert a["states"] == b["states"], "original-versus-instrumented states differ"
        replay_rng = independent_replay(control, observed, masks, steps, a["states"])
        rows.append(dict(repetition=rep, original_cpu_ns=a["cpu_ns"],
                         original_wall_ns=a["wall_ns"],
                         observed_cpu_ns=b["cpu_ns"],
                         observed_wall_ns=b["wall_ns"],
                         states=a["states"], observed_phase_ns=b["phase_ns"],
                         next_rng=replay_rng))
        print(f"{name} {backend} {steps} rep={rep+1}/{repeats} "
              f"orig_cpu_ms={a['cpu_ns']/1e6:.3f} "
              f"trace_cpu_ms={b['cpu_ns']/1e6:.3f}", flush=True)
    return dict(case=name, backend=backend, steps=steps,
                ordered_graph_sha=array_hash(graph), degrees_sha=array_hash(degree),
                label_shas=[array_hash(mask) for mask in masks],
                repetitions=rows)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--output", required=True)
    args = p.parse_args()
    for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
                "NUMBA_NUM_THREADS"):
        assert os.environ.get(key) == "1", f"pin {key}=1"
    original = check_original_git_source()
    cls = traced_class(original)
    assert cls is not PCC
    cases = ("wine",) if args.smoke else CASES
    out = []
    for name in cases:
        for backend in (("numba",) if args.smoke else BACKENDS):
            steps_list = ((0,) if args.smoke else
                          (0, 5, 500) if name in REAL and backend in ("numba", "cython")
                          else (0, 5))
            for steps in steps_list:
                out.append(one_case(cls, name, backend, steps, args.smoke))
    result = dict(schema=1, base=BASE, source_sha256=hashlib.sha256(original.encode()).hexdigest(),
                  os=sys.platform, python=sys.version, platform=platform.platform(),
                  batch_fits=FITS_PER_BATCH, pairs=1 if args.smoke else N_PAIRS,
                  cases=cases, phases=PHASES, cohorts=out,
                  production_source_unchanged=True)
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    print(f"COMPLETE {sys.platform} cohorts={len(out)} "
          f"timed_native_fits={len(out)*FITS_PER_BATCH*result['pairs']*2}", flush=True)


if __name__ == "__main__":
    main()
