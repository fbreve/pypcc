"""Bounded opt-in *single-run* NumPy PCC trajectory observations.

No imports in production PCC kernels. A collector is used only when a
caller passes observer= to the experimental NumPy propagator keyword.
The recorded RNG state is copied without drawing numbers.
"""
import hashlib
import json

import numpy as np

MAX_OBSERVATIONS = 9
MAX_BYTES = 48 * 1024 * 1024
DEFAULT_MILESTONES = (0, 1, 11, 101, 501, 2001, 6001)
STATE_NAMES = (
    "assignments", "positions", "strengths", "distances",
    "dominance", "raw_ownership",
)


def digest(array):
    array = np.asarray(array)
    h = hashlib.sha256()
    h.update(repr(tuple(array.shape)).encode("ascii"))
    h.update(array.dtype.str.encode("ascii"))
    h.update(array.tobytes(order="C"))
    return h.hexdigest()


def clone_rng(state):
    if not isinstance(state, tuple) or len(state) != 5:
        raise ValueError("NumPy RNG state must use the MT19937 five-tuple")
    return (state[0], np.asarray(state[1], dtype=np.uint32).copy(),
            int(state[2]), int(state[3]), float(state[4]))


def rng_equal(left, right):
    return (
        left[0] == right[0]
        and np.array_equal(left[1], right[1])
        and left[2:] == right[2:]
    )


def rng_digest(state):
    name, keys, pos, has_gauss, cached_gaussian = state
    h = hashlib.sha256()
    h.update(name.encode("ascii"))
    h.update(keys.tobytes(order="C"))
    h.update(f":{pos}:{has_gauss}:{float(cached_gaussian).hex()}".encode("ascii"))
    return h.hexdigest()


class ContinuousNumpyObserver:
    """One-use bounded collector; never takes a random draw or alters a view."""

    def __init__(self, mapped_labels, *, milestones=DEFAULT_MILESTONES,
                 max_observations=MAX_OBSERVATIONS, max_bytes=MAX_BYTES):
        self.labels = np.array(mapped_labels, dtype=np.int64, copy=True)
        if self.labels.ndim != 1:
            raise ValueError("mapped labels must be a 1-D vector")
        schedule = tuple(milestones)
        if (not schedule or any(type(i) is not int or i < 0 for i in schedule)
                or tuple(sorted(set(schedule))) != schedule):
            raise ValueError("milestones must be distinct increasing nonnegative integers")
        if (type(max_observations) is not int or max_observations < 1 or
                type(max_bytes) is not int or max_bytes < 1):
            raise ValueError("invalid positive snapshot budgets")
        if len(schedule) > max_observations:
            raise ValueError("milestones exceed the observation cap")
        self.milestones = frozenset(schedule)
        self.max_observations = max_observations
        self.max_bytes = max_bytes
        self.used_bytes = 0
        self.samples = []
        self.final_info = None

    def wants(self, iteration):
        if type(iteration) is not int or iteration < 0:
            raise ValueError("invalid iteration notification")
        return iteration in self.milestones

    def capture(self, iteration, dominance, positions, strengths,
                distances, raw_ownership, rng_state, info):
        if self.final_info is not None:
            raise AssertionError("observer cannot receive data after finalization")
        if type(iteration) is not int or iteration < 0:
            raise ValueError("iteration must be a nonnegative Python integer")
        is_final = info is not None
        if not is_final and not self.wants(iteration):
            raise AssertionError("NumPy propagator sent an unscheduled snapshot")

        # Bound retained payload *before* copying any row arrays. This is not
        # a process RSS / transient-allocation bound: the post-copy statistics
        # (argmax, sort, hashes) may allocate additional temporary buffers.
        dom_view = np.asarray(dominance)
        own_view = np.asarray(raw_ownership)
        if (dom_view.ndim != 2 or dom_view.shape[0] != self.labels.size
                or own_view.shape != dom_view.shape):
            raise AssertionError("observer state dimensions changed")
        upgrading_final = bool(
            is_final and self.samples
            and self.samples[-1]["iteration"] == iteration)
        if not upgrading_final:
            if self.samples and iteration <= self.samples[-1]["iteration"]:
                raise AssertionError("observer iterations must be strictly increasing")
            if len(self.samples) >= self.max_observations:
                raise ValueError("observer row budget exceeded")
            expected_bytes = (
                self.labels.size * np.dtype(np.int64).itemsize
                + sum(np.asarray(v).nbytes for v in (
                    positions, strengths, distances, dominance, raw_ownership))
                + np.asarray(rng_state[1]).nbytes
            )
            if self.used_bytes + expected_bytes > self.max_bytes:
                raise ValueError("observer byte budget exceeded")

        state = {
            "assignments": np.argmax(dominance, axis=1).astype(np.int64),
            "positions": np.array(positions, copy=True),
            "strengths": np.array(strengths, copy=True),
            "distances": np.array(distances, copy=True),
            "dominance": np.array(dominance, copy=True),
            "raw_ownership": np.array(raw_ownership, copy=True),
        }
        known = self.labels != -1
        state["assignments"][known] = self.labels[known]
        if (state["dominance"].shape[0] != self.labels.size or
                state["raw_ownership"].shape != state["dominance"].shape):
            raise AssertionError("observer state dimensions changed")
        if self.samples and self.samples[-1]["iteration"] == iteration:
            if not is_final:
                raise AssertionError("duplicate non-final snapshot")
            last = self.samples[-1]
            if (not all(np.array_equal(last["state"][k], state[k])
                        for k in STATE_NAMES) or
                    not rng_equal(last["rng"], rng_state)):
                raise AssertionError("finalization changed the already sampled state")
            last["final_info"] = dict(info)
            self.final_info = dict(info)
            return
        rng = clone_rng(rng_state)
        size = sum(v.nbytes for v in state.values()) + rng[1].nbytes
        if size != expected_bytes:
            raise AssertionError("observer retained byte estimate changed")
        if any(v.dtype.kind == "f" and not np.all(np.isfinite(v))
               for v in state.values()):
            raise AssertionError("non-finite state observed")
        top2 = np.sort(state["dominance"],axis=1)[:, -2:]
        previous = self.samples[-1] if self.samples else None
        self.samples.append({
            "iteration": iteration,
            "state": state,
            "rng": rng,
            "rng_digest": rng_digest(rng),
            "array_digests": {k: digest(state[k]) for k in STATE_NAMES},
            "net_assignment_changes": None if previous is None else int(
                np.count_nonzero(
                    state["assignments"] != previous["state"]["assignments"])),
            "mean_max_dominance": float(
                np.mean(np.max(state["dominance"],axis=1))),
            "mean_margin": float(np.mean(top2[:, 1]-top2[:, 0])),
            "final_info": dict(info) if is_final else None,
        })
        self.used_bytes += size
        if is_final:
            self.final_info = dict(info)

    def summarize(self):
        if (self.final_info is None or not self.samples or
                self.samples[0]["iteration"] != 0 or
                self.samples[-1]["final_info"] != self.final_info):
            raise AssertionError("collector missing real initial or final state")
        return [
            {
                "iteration": row["iteration"],
                "final": row["final_info"] is not None,
                "net_assignment_changes": row["net_assignment_changes"],
                "mean_max_dominance": row["mean_max_dominance"],
                "mean_margin": row["mean_margin"],
                "array_digests": row["array_digests"],
                "rng_digest": row["rng_digest"],
            }
            for row in self.samples
        ]

    def summarize_json(self):
        return json.dumps(self.summarize(), sort_keys=True,
                          separators=(",",":"), allow_nan=False)
