"""Build an isolated Cython observer from the EXACT frozen production source.

Research only. Never edit pcc_step.pyx, setup.py, packaging metadata or master.
The source-to-source splice is fail-closed against the original Git blob SHA,
uses unique anchors and preserves both specialized compiled particle kernels
byte-for-byte. Compiled output is a distinct module pcc_step_live_research.
"""
from pathlib import Path
import hashlib
import os
import sys

ROOT = Path(__file__).resolve().parents[1]
ORIGINAL = ROOT / "pcc_step.pyx"
EXPECTED_GIT_BLOB = "021c9a2c34f912e0d8382f05d0e21b9713ffd5fc"
MODULE = "pcc_step_live_research"

CAPTURE_HELPER = '''
@cython.boundscheck(False)
@cython.wraparound(False)
cdef void _research_capture(
    Py_ssize_t slot,
    np.int64_t[:] positions, double[:] strengths,
    unsigned char[:, :] distances, double[:, :] dominance,
    double[:, :] owndeg, double[:] row_max_cache, bint cache_enabled,
    unsigned int rng_state,
    np.int64_t[:, :] positions_out, double[:, :] strengths_out,
    unsigned char[:, :, :] distances_out,
    double[:, :, :] dominance_out, double[:, :, :] owndeg_out,
    double[:, :] rowmax_out, np.uint32_t[:] rng_out,
) noexcept nogil:
    cdef Py_ssize_t i, j, n_nodes = dominance.shape[0]
    cdef Py_ssize_t c = dominance.shape[1], n_particles = positions.shape[0]
    cdef double maximum
    for j in range(n_particles):
        positions_out[slot, j] = positions[j]
        strengths_out[slot, j] = strengths[j]
    for i in range(n_nodes):
        maximum = dominance[i, 0]
        for j in range(c):
            dominance_out[slot, i, j] = dominance[i, j]
            owndeg_out[slot, i, j] = owndeg[i, j]
            if dominance[i, j] > maximum:
                maximum = dominance[i, j]
        if cache_enabled:
            rowmax_out[slot, i] = row_max_cache[i]
        else:
            rowmax_out[slot, i] = maximum
        for j in range(n_particles):
            distances_out[slot, i, j] = distances[i, j]
    rng_out[slot] = rng_state

'''

SIGNATURE_ANCHOR = "    bint             experimental_canonical_rowmax_init = False\\n):"
EXTRA_SIGNATURE = """    bint             experimental_canonical_rowmax_init = False,
    object           research_observer = None
):"""
CDEF_ANCHOR = "    cdef int metric_mismatch_iteration = 0"
EXTRA_CDEF = """    cdef int metric_mismatch_iteration = 0
    cdef bint research_active = research_observer is not None
    cdef bint research_stop = False
    cdef Py_ssize_t research_cursor = 0
    cdef np.int64_t[:] research_schedule = None
    cdef np.int64_t[:, :] research_positions = None
    cdef double[:, :] research_strengths = None
    cdef unsigned char[:, :, :] research_distances = None
    cdef double[:, :, :] research_dominance = None
    cdef double[:, :, :] research_owndeg = None
    cdef double[:, :] research_rowmax = None
    cdef np.uint32_t[:] research_rng = None
    cdef np.uint32_t[:] research_final_rng = None
    cdef np.int64_t[:] research_recorded = None
    if research_active:
        research_schedule = research_observer["schedule"]
        research_positions = research_observer["positions"]
        research_strengths = research_observer["strengths"]
        research_distances = research_observer["distances"]
        research_dominance = research_observer["dominance"]
        research_owndeg = research_observer["owndeg"]
        research_rowmax = research_observer["rowmax"]
        research_rng = research_observer["rng"]
        research_final_rng = research_observer["final_rng"]
        research_recorded = research_observer["recorded"]"""

CAPTURE_CALL = """_research_capture(
                    research_cursor, part_curnode, part_strength,
                    dist_table, dominance, owndeg, row_max_cache,
                    cache_enabled, rng_state, research_positions,
                    research_strengths, research_distances,
                    research_dominance, research_owndeg,
                    research_rowmax, research_rng)
                research_cursor += 1"""

LOOP_ANCHOR = "    with nogil:\\n        for it in range(max_iter):"
LOOP_INSERT = """    with nogil:
        if research_active and research_schedule.shape[0] > 0 and research_schedule[0] == 0:
            """ + CAPTURE_CALL + """
        for it in range(max_iter):
            research_stop = False"""

STOP_ANCHOR = "                    if stop_cnt > stop_max:\\n                        break"
STOP_INSERT = """                    if stop_cnt > stop_max:
                        if not research_active:
                            break
                        research_stop = True"""
AFTER_STOP_ANCHOR = """                        research_stop = True

    if metric_mismatch_iteration:"""
AFTER_STOP_INSERT = """                        research_stop = True

            if research_active and research_cursor < research_schedule.shape[0] and research_schedule[research_cursor] == it + 1:
                """ + CAPTURE_CALL + """
            if research_stop:
                break

    if research_active:
        research_final_rng[0] = rng_state
        research_recorded[0] = research_cursor
    if metric_mismatch_iteration:"""


def _replace_once(source, old, new, title):
    if source.count(old) != 1:
        raise RuntimeError(f"Unsafe Cython source drift: {title} anchor count {source.count(old)}")
    return source.replace(old, new, 1)


def source_blob_sha(raw):
    return hashlib.sha1(b"blob " + str(len(raw)).encode("ascii") + b"\\0" + raw).hexdigest()


def generate():
    raw = ORIGINAL.read_bytes()
    if source_blob_sha(raw) != EXPECTED_GIT_BLOB:
        raise RuntimeError("Cython original Git blob changed; manually re-audit before regeneration")
    source = raw.decode("utf-8")
    source = _replace_once(source, "cpdef pcc_propagate(", CAPTURE_HELPER + "cpdef pcc_propagate(", "function entry")
    source = _replace_once(source, SIGNATURE_ANCHOR, EXTRA_SIGNATURE, "signature")
    source = _replace_once(source, CDEF_ANCHOR, EXTRA_CDEF, "declarations")
    source = _replace_once(source, LOOP_ANCHOR, LOOP_INSERT, "zero checkpoint and loop")
    source = _replace_once(source, STOP_ANCHOR, STOP_INSERT, "early stop")
    source = _replace_once(source, AFTER_STOP_ANCHOR, AFTER_STOP_INSERT, "terminal capture/final RNG")

    # Both original native hotloops must remain byte-for-byte unchanged.
    hotloops = raw.decode("utf-8").split("cdef void pcc_step_sequential(", 1)[1]
    hotloops = hotloops.split("cpdef pcc_step(", 1)[0]
    checkloops = source.split("cdef void pcc_step_sequential(", 1)[1]
    checkloops = checkloops.split("cpdef pcc_step(", 1)[0]
    if hotloops != checkloops:
        raise AssertionError("Native particle movement kernel changed")
    scratch = ROOT / ".cache" / "research_live_cython"
    scratch.mkdir(parents=True, exist_ok=True)
    generated = scratch / (MODULE + ".pyx")
    generated.write_text(source, encoding="utf-8")
    return generated


def build():
    from setuptools import Extension, setup
    from Cython.Build import cythonize
    import numpy as np

    generated = generate()
    flags = ["/O2"] if os.name == "nt" else ["-O3"]
    extension = Extension(
        MODULE, [str(generated)], include_dirs=[np.get_include()],
        extra_compile_args=flags,
    )
    setup(
        name=MODULE,
        version="0.0.0",
        ext_modules=cythonize([extension], language_level="3", force=True),
        script_args=[
            "build_ext", "--build-lib", str(ROOT),
            "--build-temp", str(ROOT / ".cache" / "research_live_cython" / "build"),
        ],
    )


if __name__ == "__main__":
    build()
