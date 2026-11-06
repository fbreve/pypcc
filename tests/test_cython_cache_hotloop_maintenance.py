"""Prevent behavioral drift between the legacy and cached Cython hot loops.

We intentionally retain two compiled-specialized loops: a runtime flag inside
the particle loop previously slowed *default-disabled* Digits ~30% on Linux.
This guard requires the cached loop to differ from the original ONLY by the
extra cache argument and the single cache write after the original row maximum.
It has no runtime effect, needs no Cython compiler and works on Windows/Linux.
"""
import unittest
from pathlib import Path


SOURCE = Path(__file__).resolve().parents[1] / "pcc_step.pyx"
LEGACY_NAME = "cdef void pcc_step_sequential("
CACHED_NAME = "cdef void pcc_step_sequential_cached("
FOLLOWING_NAME = "cpdef pcc_step("

# Permitted specialization is intentionally explicit, not a broad
# whitespace/comment stripper that might hide a changed numerical operation.
EXTRA_ARGUMENT = (
    "    unsigned int    *rng_state,\n"
    "    double[:]        row_max_cache\n"
)
ORIGINAL_ARGUMENT = "    unsigned int    *rng_state\n"
CACHE_WRITE = (
    "            if labels[next_node] == -1:\n"
    "                # Existing movement decision already calculated the final row\n"
    "                # maximum in the legacy class-order. No second scan needed.\n"
    "                row_max_cache[next_node] = max_dom\n\n"
)


def extract_hot_loops(source):
    """Return both private function bodies, excluding independent decorators."""
    if (source.count(LEGACY_NAME) != 1 or
            source.count(CACHED_NAME) != 1 or
            source.count(FOLLOWING_NAME) != 1):
        raise AssertionError("Unexpected Cython specialization function count")
    a = source.index(LEGACY_NAME)
    b = source.index(CACHED_NAME)
    c = source.index(FOLLOWING_NAME, b)
    if not (a < b < c):
        raise AssertionError("Legacy and cached Cython function order changed")
    # The 3 decorators before the next function belong to the next function
    # and should not be interpreted as differences in the two bodies.
    marker = "\n@cython.boundscheck(False)\n@cython.wraparound(False)\n@cython.cdivision(True)\n"
    first = source[a:b]
    if marker not in first:
        raise AssertionError("Missing cached Cython decorators")
    first = first[:first.rindex(marker)].rstrip()
    second = source[b:c].rstrip()
    return first, second


def verify_specialization_is_identical(source):
    """Raise if any evolution of the cached particle dynamics is unmirrored."""
    legacy, cached = extract_hot_loops(source)
    if cached.count(EXTRA_ARGUMENT) != 1:
        raise AssertionError("Unexpected cached-kernel argument signature")
    if legacy.count(ORIGINAL_ARGUMENT) != 1:
        raise AssertionError("Unexpected original-kernel argument signature")
    if cached.count(CACHE_WRITE) != 1:
        raise AssertionError("Unexpected cached-kernel node-maximum update")
    normalized = cached.replace(CACHED_NAME, LEGACY_NAME, 1)
    normalized = normalized.replace(EXTRA_ARGUMENT, ORIGINAL_ARGUMENT, 1)
    normalized = normalized.replace(CACHE_WRITE, "", 1).rstrip()
    if normalized != legacy:
        import difflib
        differences = "\n".join(difflib.unified_diff(
            legacy.splitlines(), normalized.splitlines(),
            fromfile="legacy_particle_step",
            tofile="cached_particle_step_without_cache_write",
            lineterm="",
        ))
        raise AssertionError(
            "The cached Cython kernel changed particle dynamics or numerical "
            "operation order without updating its legacy twin:\n" +
            differences)

    # Activation makes the cache default-on for stopping, but must preserve
    # explicit opt-out and the original no-stop path.
    if source.count("experimental_rowmax_cache = True") != 1:
        raise AssertionError("Rowmax cache unexpectedly changed from the activated default-on")
    if source.count("experimental_verify_rowmax_cache = False") != 1:
        raise AssertionError("Verification flag changed from opt-in")
    if source.count("if cache_enabled:\n                pcc_step_sequential_cached(") != 1:
        raise AssertionError("Missing outer-loop specialization dispatch")
    if source.count("else:\n                pcc_step_sequential(") != 1:
        raise AssertionError("Legacy specialization not selected when disabled")


class TestCythonCacheHotLoopMaintenance(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = SOURCE.read_text(encoding="utf-8")

    def test_cached_hot_loop_is_only_original_plus_cache_write(self):
        verify_specialization_is_identical(self.source)

    def test_guard_detects_drift_in_cached_movement_rule(self):
        head, tail = self.source.split(CACHED_NAME, 1)
        original = "if randval < p_grd:"
        if tail.count(original) != 1:
            self.fail("Unexpected greedy-movement comparison in cached function")
        altered = head + CACHED_NAME + tail.replace(
            original, "if randval <= p_grd:", 1)
        with self.assertRaisesRegex(AssertionError, "changed particle dynamics"):
            verify_specialization_is_identical(altered)

    def test_guard_detects_missing_cache_update(self):
        altered = self.source.replace(CACHE_WRITE, "", 1)
        with self.assertRaisesRegex(AssertionError, "node-maximum update"):
            verify_specialization_is_identical(altered)


if __name__ == "__main__":
    unittest.main()
