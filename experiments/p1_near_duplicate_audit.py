"""Provisional P1 source-quality *near-duplicate* diagnostic, with NO PCC fit.

Distance is RMS standardized numeric Euclidean distance, using population
standard deviation of ALL rows within each source dataset. This is an
exploratory quality screen, NOT the future PCC kNN graph or classifier metric.

Threshold grid is predeclared in source before inspecting any near-duplicate
result; it must not be tuned after seeing source outcomes. All unordered pairs
are counted once. No record-level features, row indices or pair identities
are returned/published. No sample exclusion or dataset selection occurs.
"""
import math
from pathlib import Path

from experiments import p1_offline_candidate_integrity as offline

THRESHOLDS = (1e-6, 1e-4, 1e-3, 1e-2)
BLOCK_ROWS = 128


class NearDuplicateAuditBlocked(ValueError):
    pass


def _require(condition, reason):
    if not condition:
        raise NearDuplicateAuditBlocked(reason)


def analyze_features(matrix, class_labels):
    """Fully descriptive O(n^2) near-neighbor screen with bounded blocks."""
    import numpy as np  # separate, pinned audit-only dependency, never PCC
    x = np.asarray(matrix, dtype=np.float64)
    labels = np.asarray(class_labels, dtype=str)
    _require(x.ndim == 2 and x.shape[0] >= 2 and x.shape[1] >= 1,
             "Expected at least 2 source rows with numeric features")
    n, d = x.shape
    _require(n <= offline.MAX_ROWS and x.size <= offline.MAX_CELLS,
             "Source matrix outside audit budget")
    _require(labels.shape == (n,), "Class vector length mismatch")
    _require(bool(np.isfinite(x).all()), "Nonfinite feature in source")
    sigma = x.std(axis=0, ddof=0)
    _require(bool(np.isfinite(sigma).all()) and bool((sigma > 0).all()),
             "Constant or invalid feature prevents standardized distances")
    z = (x - x.mean(axis=0)) / sigma
    _require(bool(np.isfinite(z).all()), "Nonfinite normalized value")

    thresholds = np.asarray(THRESHOLDS, dtype=np.float64)
    _require(bool((thresholds > 0).all()) and
             list(thresholds) == sorted(set(thresholds)),
             "Distance grid must be ascending positive values")
    pair_totals = [0] * len(thresholds)
    cross_labels = [0] * len(thresholds)
    affected = [set() for _ in thresholds]
    min_rms = float("inf")
    columns = np.arange(n, dtype=np.int64)[None, :]
    for start in range(0, n, BLOCK_ROWS):
        chunk = z[start:start + BLOCK_ROWS]
        row_ids = np.arange(start, start + len(chunk), dtype=np.int64)[:, None]
        delta = chunk[:, None, :] - z[None, :, :]
        dist2 = np.einsum("ijk,ijk->ij", delta, delta, optimize=True) / d
        upper = columns > row_ids
        if bool(upper.any()):
            min_rms = min(min_rms, float(np.sqrt(dist2[upper].min())))
        different = labels[start:start + len(chunk), None] != labels[None, :]
        for i, threshold in enumerate(thresholds):
            selected = upper & (dist2 <= threshold * threshold)
            if not bool(selected.any()):
                continue
            pair_totals[i] += int(np.count_nonzero(selected))
            cross_labels[i] += int(np.count_nonzero(selected & different))
            local_rows, global_cols = np.nonzero(selected)
            affected[i].update(int(start + j) for j in local_rows)
            affected[i].update(int(j) for j in global_cols)

    tests = [
        {
            "rms_std_euclidean_threshold_le": float(t),
            "unordered_pair_count": int(pairs),
            "cross_class_pair_count": int(cross),
            "same_class_pair_count": int(pairs - cross),
            "distinct_rows_involved": len(affected[i]),
        }
        for i, (t, pairs, cross) in enumerate(zip(
            thresholds, pair_totals, cross_labels
        ))
    ]
    return {
        "schema": "pypcc-p1-source-near-duplicate-diagnostic/v1",
        "numpy_version": np.__version__,
        "quality_only_no_PCC_run": True,
        "n_source_rows": int(n),
        "features": int(d),
        "unordered_pairs_inspected": int(n * (n - 1) // 2),
        "method": "RMS Euclidean on per-source-feature population-zscores; every unordered pair counted once",
        "std_ddof": 0,
        "thresholds_predeclared_not_chosen_from_PCC_outcomes": True,
        "thresholds": tests,
        "minimum_distinct_row_rms_standardized_distance": min_rms,
        "source_row_or_pair_identifiers_disclosed": False,
        "near_duplicates_are_not_proven_same_subject_or_image": True,
        "subject_or_batch_identity_verified": False,
        "dataset_selected_for_P1": False,
        "model_fits_executed": 0,
        "execution_authorized": False,
    }


def audit_local_archive(path, *, dataset_id, member):
    """Examine already pinned local UCI bytes; never fetch or make graphs."""
    raw, filename, _, _ = offline.safely_read(path, member)
    fmt = Path(filename).suffix.lower()
    _require(Path(filename).name.lower() ==
             offline.TASKS[dataset_id]["file"].lower(), "Wrong UCI task")
    if fmt == ".arff":
        rows, headers, _ = offline.parse_arff(raw)
    elif fmt == ".xlsx":
        rows, headers, _ = offline.parse_xlsx(raw)
    else:
        raise NearDuplicateAuditBlocked("Unsupported pinned file format")
    expected = offline.TASKS[dataset_id]
    _require(len(rows) == expected["n"] and
             len(headers) == expected["d"] + 1,
             "Source shape changed")
    numeric = [[float(v) for v in row[:-1]] for row in rows]
    labels = [str(row[-1]).strip(" '\"") for row in rows]
    _require(all(v in expected["labels"] for v in labels),
             "Source target class changed")
    return analyze_features(numeric, labels)
