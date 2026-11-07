"""Verify experimental NumPy observer contract from a built, installed wheel.

Run OUTSIDE the checkout with `python -I path/to/this-script.py`.
Do not import experiments/* or rely on editable installs.
"""
import argparse
from importlib import metadata
import inspect
import json
from pathlib import Path
import platform
import sys

import numpy as np

import pcc
import pcc_numpy
from pcc import ParticleCompetitionAndCooperation as PCC


SEED = 17011
SNAPSHOTS = (0, 1, 21)


def rng_copy(value):
    return (value[0], np.asarray(value[1], dtype=np.uint32).copy(),
            int(value[2]), int(value[3]), float(value[4]))


def rng_equal(a, b):
    return a[0] == b[0] and np.array_equal(a[1], b[1]) and a[2:] == b[2:]


class NonMutatingObserver:
    """Standalone wheel-client callback; not the experiments collector."""

    def __init__(self, mapped_labels):
        self.labels = np.asarray(mapped_labels, dtype=np.int64).copy()
        self.rows = []
        self.final_info = None

    def wants(self, iteration):
        return iteration in SNAPSHOTS

    def capture(self, iteration, dominance, positions, strengths,
                distances, raw_ownership, rng_state, info):
        assignments = np.argmax(dominance, axis=1).astype(np.int64)
        assignments[self.labels != -1] = self.labels[self.labels != -1]
        state = {
            "assignments": assignments,
            "positions": np.array(positions, copy=True),
            "strengths": np.array(strengths, copy=True),
            "distances": np.array(distances, copy=True),
            "dominance": np.array(dominance, copy=True),
            "raw_ownership": np.array(raw_ownership, copy=True),
        }
        rng = rng_copy(rng_state)
        if self.rows and self.rows[-1]["iteration"] == iteration:
            if info is None:
                raise AssertionError("duplicate non-terminal observation")
            last = self.rows[-1]
            if (not all(np.array_equal(last["state"][key], value)
                        for key, value in state.items())
                    or not rng_equal(last["rng"], rng)):
                raise AssertionError("wheel observer final state unexpectedly mutated")
            if self.final_info is not None:
                raise AssertionError("duplicate final notification")
            last["final_info"] = dict(info)
            self.final_info = dict(info)
            return

        if iteration not in SNAPSHOTS:
            raise AssertionError("unexpected milestone")
        if self.rows and self.rows[-1]["iteration"] >= iteration:
            raise AssertionError("non-monotonic milestone")
        record = {
            "iteration": iteration, "state": state, "rng": rng,
            "final_info": dict(info) if info is not None else None,
        }
        self.rows.append(record)
        if info is not None:
            self.final_info = dict(info)


class WheelAuditedPCC(PCC):
    def __init__(self, record):
        super().__init__(impl="numpy")
        self.record = record
        self.observer = None
        self.native_info = None

    def _get_backend_fn(self):
        native = PCC._get_backend_fn(self)
        if native is not pcc_numpy.pcc_propagate_numpy:
            raise AssertionError("wheel NumPy kernel resolution is incorrect")

        def wrapped(*args, **kwargs):
            if "observer" in kwargs or "return_info" in kwargs:
                raise AssertionError("high-level fit unexpectedly provides observer/info")
            if self.record:
                self.observer = NonMutatingObserver(args[2])
                kwargs["observer"] = self.observer
            self.native_info = dict(native(*args, return_info=True, **kwargs))
            return None

        wrapped.__name__ = native.__name__
        return wrapped


def fit(record):
    # Six-node reciprocal undirected ring; equal degrees, two initial classes.
    graph = np.array([[1, 5], [0, 2], [1, 3],
                      [2, 4], [3, 5], [4, 0]], dtype=np.int64)
    degrees = np.full(6, 2, dtype=np.int64)
    labels = np.array([0, -1, -1, 1, -1, -1], dtype=np.int64)
    model = WheelAuditedPCC(record)
    model.set_graph(graph, degrees, k_nn=2)
    prediction = model.fit_predict(
        labels.copy(), seed=SEED, max_iter=21, early_stop=False,
        es_chk=10, p_grd=.5, delta_v=.1, deltap=1., dexp=2.)
    arrays = {
        "predictions": np.asarray(prediction).copy(),
        "positions": model.part.curnode.copy(),
        "strengths": model.part.strength.copy(),
        "distances": model.part.dist_table.copy(),
        "dominance": model.node.dominance.copy(),
        "ownership": model.owndeg.copy(),
    }
    rng = rng_copy(np.random.get_state())
    after = np.random.random(16)
    if not np.array_equal(prediction[labels >= 0], labels[labels >= 0]):
        raise AssertionError("known vertex labels were modified")
    return dict(model=model, arrays=arrays, info=model.native_info,
                rng=rng, after=after)


def verify(checkout):
    package = metadata.distribution("pcc_step")
    root = checkout.resolve()
    origins = {name: Path(mod.__file__).resolve()
               for name, mod in (("pcc", pcc), ("pcc_numpy", pcc_numpy))}
    checks = {
        "installed_module_origins": all(not path.is_relative_to(root)
                                         for path in origins.values()),
        "wheel_contains_numpy_module": any(
            str(f).replace("\\", "/") == "pcc_numpy.py"
            for f in (package.files or ())),
        "wheel_excludes_experiments": not any(
            str(f).replace("\\", "/").startswith("experiments/")
            for f in (package.files or ())),
        "public_fit_has_no_observer": (
            "observer" not in inspect.signature(PCC.fit_predict).parameters),
    }
    kernel_args = inspect.signature(pcc_numpy.pcc_propagate_numpy).parameters
    checks["private_trailing_observer_default_none"] = bool(
        tuple(kernel_args)[-1] == "observer"
        and kernel_args["observer"].default is None)
    try:
        PCC(impl="numpy").fit_predict(np.array([0, 1]), observer=object())
    except TypeError as exc:
        checks["high_level_rejects_observer"] = "observer" in str(exc)
    else:
        checks["high_level_rejects_observer"] = False
    if not all(checks.values()):
        raise AssertionError(f"Wheel API or location contract failed: {checks}")

    off = fit(False)
    on = fit(True)
    checks["six_final_arrays_exact"] = all(
        np.array_equal(off["arrays"][name], on["arrays"][name])
        for name in off["arrays"])
    checks["native_stop_info_exact"] = off["info"] == on["info"]
    checks["complete_rng_exact"] = rng_equal(off["rng"], on["rng"])
    checks["next16_exact"] = np.array_equal(off["after"], on["after"])
    observer = on["model"].observer
    checks["exact_actual_schedule"] = (
        [r["iteration"] for r in observer.rows] == list(SNAPSHOTS)
        and len(observer.rows) == 3)
    checks["genuine_terminal_info"] = (
        off["info"]["effective_iterations"] == 21
        and off["info"]["stop_reason"] == "max_iter"
        and observer.final_info == off["info"]
        and observer.rows[-1]["final_info"] == off["info"])
    last = observer.rows[-1]
    arrays = on["arrays"]
    mapped = on["model"].mapped_labels
    assignment = np.argmax(arrays["dominance"], axis=1)
    assignment[mapped != -1] = mapped[mapped != -1]
    own = last["state"]["raw_ownership"]
    sums = own.sum(axis=1, keepdims=True)
    sums[sums == 0] = 1.
    checks["terminal_state_rng_exact"] = all((
        np.array_equal(last["state"]["assignments"], assignment),
        np.array_equal(last["state"]["positions"], arrays["positions"]),
        np.array_equal(last["state"]["strengths"], arrays["strengths"]),
        np.array_equal(last["state"]["distances"], arrays["distances"]),
        np.array_equal(last["state"]["dominance"], arrays["dominance"]),
        np.array_equal(own / sums, arrays["ownership"]),
        rng_equal(last["rng"], on["rng"]),
    ))
    if not all(checks.values()):
        raise AssertionError(f"Installed wheel observer parity failure: {checks}")
    return {
        "python": platform.python_version(),
        "platform": platform.system(),
        "package_name": package.metadata["Name"],
        "package_version": package.version,
        "source_checkout": str(root),
        "module_origins": {key: str(value) for key, value in origins.items()},
        "seed": SEED,
        "snapshot_iterations": [r["iteration"] for r in observer.rows],
        "checks": checks,
        "passed": True,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkout", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = verify(args.checkout)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n",
                           encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
