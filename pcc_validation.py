"""Input checks performed before entering unchecked sequential kernels."""
import numpy as np
from decimal import Decimal


def validate_finite(name, value):
    """Reject nonfinite numeric values without coercing integer/string inputs.

    Object arrays retain their existing conversion rules, including numeric
    strings. Check floating/complex objects and Decimal before integer casts.
    This does not impose ranges, shapes or a new label vocabulary.
    """
    values = np.asarray(value)
    if values.dtype.kind in 'fc':
        if not np.all(np.isfinite(values)):
            raise ValueError(f"{name} must contain only finite numeric values")
    elif values.dtype.kind == 'O':
        for item in values.flat:
            if isinstance(item, Decimal):
                finite = item.is_finite()
            elif isinstance(item, (float, complex, np.inexact)):
                finite = np.isfinite(item)
            else:
                continue
            if not finite:
                raise ValueError(f"{name} must contain only finite numeric values")


def validate_vector(name, value, length):
    if value is None or np.ndim(value) != 1 or len(value) < length:
        raise ValueError(f"{name} must be a vector with at least {length} entries")


def validate_step_inputs(neib_list, neib_qt, labels, c, zerovec,
                         part_curnode, part_label, part_strength, dist_table,
                         dominance, owndeg):
    if np.ndim(neib_list) != 2 or neib_list.shape[0] == 0:
        raise ValueError("neib_list must be a nonempty node-by-neighbor matrix")
    n = neib_list.shape[0]
    if c < 2:
        raise ValueError("sequential kernels require at least two classes")
    validate_vector("neib_qt", neib_qt, n)
    degrees = np.asarray(neib_qt)[:n]
    if np.any(degrees < 0) or np.any(degrees > neib_list.shape[1]):
        raise ValueError("node degrees must fit the neighbor matrix")
    validate_vector("labels", labels, n)
    validate_vector("zerovec", zerovec, c)
    validate_vector("part_curnode", part_curnode, 0)
    particles = len(part_curnode)
    validate_vector("part_label", part_label, particles)
    validate_vector("part_strength", part_strength, particles)
    for name, value, columns in (("dist_table", dist_table, particles),
                                 ("dominance", dominance, c),
                                 ("owndeg", owndeg, c)):
        if (np.ndim(value) != 2 or value.shape[0] < n
                or value.shape[1] < columns):
            raise ValueError(f"{name} must have at least shape ({n}, {columns})")
    # Cython supplies a typed memoryview here; asarray exposes its existing
    # buffer without changing the object ultimately passed to either kernel.
    distances = np.asarray(dist_table)
    if distances.dtype.kind not in 'iu':
        raise ValueError("dist_table must be an integer array with distances in [0, 255]")
    # Production uint8 tables already guarantee the lookup index range. Retain
    # compatible wider integer buffers without coercing, copying or scanning
    # unused padding; only non-uint8 active regions need a range check.
    if distances.dtype != np.uint8:
        active_distances = distances[:n, :particles]
        if np.any(active_distances < 0) or np.any(active_distances > 255):
            raise ValueError("dist_table distances must be in [0, 255]")
    return max(1, int(degrees.max()))


def validate_buffers(c, max_degree, dom_row, reduc, dom_list, dist_list,
                     prob, slices, dist_weights):
    for name, value, length in (("dom_row", dom_row, c), ("reduc", reduc, c),
                                ("dom_list", dom_list, max_degree),
                                ("dist_list", dist_list, max_degree),
                                ("prob", prob, max_degree),
                                ("slices", slices, max_degree),
                                ("dist_weights", dist_weights, 256)):
        validate_vector(name, value, length)


def validate_kernel_scalars(p_grd, delta_v, deltap, dexp):
    """Reject nonfinite direct-call parameters before any state or RNG change."""
    for name, value in (("p_grd", p_grd), ("delta_v", delta_v),
                        ("deltap", deltap), ("dexp", dexp)):
        validate_finite(name, value)


def validate_fractional_buffers(part_strength, dominance, owndeg,
                                dom_row=None, dom_list=None, dist_list=None,
                                prob=None, slices=None):
    """Buffers storing fractional results must not silently truncate them."""
    for name, value in (("part_strength", part_strength),
                        ("dominance", dominance), ("owndeg", owndeg),
                        ("dom_row", dom_row), ("dom_list", dom_list),
                        ("dist_list", dist_list), ("prob", prob),
                        ("slices", slices)):
        if value is not None and np.asarray(value).dtype.kind != "f":
            raise ValueError(f"{name} must have a floating-point dtype")


def validate_mutable_arrays(*named_arrays):
    """Check writable NumPy buffers before the first particle is updated."""
    for name, value in named_arrays:
        if isinstance(value, np.ndarray) and not value.flags.writeable:
            raise ValueError(f"{name} must be writable")


def validate_seed(seed):
    """Normalize an optional unsigned 32-bit seed without changing RNG state."""
    if seed is None:
        return None
    if (isinstance(seed, (bool, np.bool_))
            or not isinstance(seed, (int, np.integer))
            or not 0 <= seed <= 2**32 - 1):
        raise ValueError("seed must be an integer in [0, 2**32 - 1] or None")
    return int(seed)


def validate_propagation_controls(max_iter, es_chk, stop_max):
    """Reject invalid iteration controls before any kernel or RNG mutation."""
    for name, value in (("max_iter", max_iter), ("es_chk", es_chk),
                        ("stop_max", stop_max)):
        if (isinstance(value, (bool, np.bool_))
                or not isinstance(value, (int, np.integer))
                or value < 0):
            raise ValueError(f"{name} must be a nonnegative integer")
