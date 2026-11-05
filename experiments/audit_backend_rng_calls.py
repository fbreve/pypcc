"""Isolate RNG sampling-call differences in sequential production PCC backends.

This intentionally models ONLY the random-walk branch's RNG calls
(p_grd=0), not the particle dynamics or an accuracy comparison.
The Cython path is a source-code translation, not a call to its binary.
Experimental; no changes to production kernels.
"""
import argparse
import csv
from pathlib import Path

import numpy as np
from numba import njit


@njit
def numba_uniform(seed, count):
    np.random.seed(seed)
    out = np.empty(count, dtype=np.float64)
    for i in range(count):
        out[i] = np.random.random()
    return out


@njit
def numba_random_walk_indices(seed, degrees):
    np.random.seed(seed)
    indices = np.empty(len(degrees), dtype=np.int64)
    for i in range(len(degrees)):
        np.random.random()  # Bernoulli decision; p_grd=0
        indices[i] = np.random.randint(degrees[i])
    return indices


def numpy_random_walk_indices(seed, degrees):
    rng = np.random.RandomState(seed)
    indices = np.empty(len(degrees), dtype=np.int64)
    for i, degree in enumerate(degrees):
        rng.random_sample()  # Bernoulli decision; p_grd=0
        indices[i] = int(rng.random_sample() * int(degree))
    return indices


def xorshift32_next(state):
    state ^= (state << 13) & 0xffffffff
    state ^= state >> 17
    state ^= (state << 5) & 0xffffffff
    return state & 0xffffffff


def cython_source_random_walk_indices(seed, degrees):
    """Python translation of pcc_step.pyx xorshift+modulo, not Cython binary."""
    state = int(seed) & 0xffffffff
    if state == 0:
        state = 123456789
    indices = np.empty(len(degrees), dtype=np.int64)
    for i, degree in enumerate(degrees):
        state = xorshift32_next(state)  # Bernoulli decision; p_grd=0
        state = xorshift32_next(state)  # modulo index
        indices[i] = state % int(degree)
    return indices


def audit(seed, degrees):
    rng = np.random.RandomState(seed)
    numpy_draws = rng.random_sample(32)
    numba_draws = numba_uniform(seed, 32)
    numpy_indices = numpy_random_walk_indices(seed, degrees)
    numba_indices = numba_random_walk_indices(seed, degrees)
    cython_reference = cython_source_random_walk_indices(seed, degrees)
    for indices in (numpy_indices, numba_indices, cython_reference):
        if np.any(indices < 0) or np.any(indices >= degrees):
            raise AssertionError("Invalid neighbor index")
    return {
        "seed": seed,
        "n_selections": len(degrees),
        "raw_numpy_numba_exact": bool(np.array_equal(numpy_draws, numba_draws)),
        "random_walk_numpy_numba_agreement": float(
            np.mean(numpy_indices == numba_indices)),
        "random_walk_numpy_cython_reference_agreement": float(
            np.mean(numpy_indices == cython_reference)),
        "random_walk_numba_cython_reference_agreement": float(
            np.mean(numba_indices == cython_reference)),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    # Frozen synthetic degrees: do not fit to observed classification results.
    degrees = np.tile(
        np.array([2, 3, 7, 10, 11, 13, 17, 23], dtype=np.int64), 128)
    rows = [audit(seed, degrees) for seed in (4001, 4093)]
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        print(row, flush=True)


if __name__ == "__main__":
    main()
