"""Shape checks performed before entering unchecked sequential kernels."""
import numpy as np


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
