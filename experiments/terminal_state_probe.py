"""Opt-in read-only *terminal-state* research probe for current PyPCC.

This is NOT an intra-fit trajectory observer or a C2 checkpoint/restart.
Call it only after a completed fit_predict; it does not call a backend,
read or set RNG, or modify the model. Never use the resulting state to
claim how many iterations were actually executed during early stopping.
"""
from dataclasses import dataclass
import hashlib
from types import MappingProxyType

import numpy as np


@dataclass(frozen=True)
class TerminalState:
    arrays: object
    sha256: object
    total_bytes: int
    iteration_cap: int
    actual_iterations_known: bool = False


# Canonical output is little-endian, C-contiguous even if native source
# arrays are Fortran-order. Float64 bits are never normalized or rounded.
SCHEMA = (
    ("dominance", "node.dominance", "<f8", 2),
    ("node_classes", "node.label", "<i8", 1),
    ("particle_positions", "part.curnode", "<i8", 1),
    ("particle_strengths", "part.strength", "<f8", 1),
    ("particle_homes", "part.homenode", "<i8", 1),
    ("particle_classes", "part.label", "<i8", 1),
    ("mapped_labels", "mapped_labels", "<i8", 1),
    ("normalized_owndeg", "owndeg", "<f8", 2),
)


def _field(model, path):
    current = model
    for name in path.split("."):
        current = getattr(current, name, None)
        if current is None:
            raise ValueError("model does not have a completed fitted state: " + path)
    return current


def capture_terminal_state(model, *, max_bytes=64 * 1024 * 1024,
                           include_distances=False):
    """Copy bounded finished model arrays without using an RNG or running PCC.

    This is **not** a live trajectory checkpoint. All arrays are independent
    read-only copies; sha256 is over little-endian C-order array *values*,
    without a file header. The source model is not mutated and has no
    additional memory cost unless this function is explicitly called.
    """
    if type(max_bytes) is not int or max_bytes <= 0:
        raise ValueError("max_bytes must be a positive integer")
    if type(include_distances) is not bool:
        raise ValueError("include_distances must be bool")
    if _field(model, "part") is None or _field(model, "node") is None:
        raise ValueError("no completed fit")
    n, c = _field(model, "node.dominance").shape
    p = _field(model, "part.curnode").shape[0]
    if not (n > 0 and c >= 2 and p >= 2):
        raise ValueError("invalid fitted shape")
    definitions = [
        (name, source, np.dtype(dtype), (n, c) if rank == 2 else
         (n,) if name in ("node_classes", "mapped_labels") else (p,))
        for name, source, dtype, rank in SCHEMA
    ]
    # The terminal state also contains an N x C normalized owndeg array.
    definitions[-1] = ("normalized_owndeg", "owndeg", np.dtype("<f8"), (n, c))
    if include_distances:
        definitions.append(("particle_distances", "part.dist_table",
                            np.dtype("u1"), (n, p)))
    source_arrays = {}
    total = 0
    for name, source, dtype, shape in definitions:
        value = _field(model, source)
        if not isinstance(value, np.ndarray) or value.shape != shape or value.dtype != dtype:
            raise ValueError(f"{source} has incompatible shape or dtype")
        source_arrays[name] = value
        total += value.nbytes
        if total > max_bytes:
            raise ValueError("terminal snapshot exceeds max_bytes before copying")
    iteration_cap = _field(model, "max_iter")
    if type(iteration_cap) is not int or iteration_cap < 0:
        raise ValueError("invalid iteration cap")

    copied, hashes = {}, {}
    for name, _source, dtype, _shape in definitions:
        value = np.array(source_arrays[name], dtype=dtype, copy=True, order="C")
        value.setflags(write=False)
        copied[name] = value
        hashes[name] = hashlib.sha256(memoryview(value).cast("B")).hexdigest()
    return TerminalState(MappingProxyType(copied),
                         MappingProxyType(hashes), total, iteration_cap)
