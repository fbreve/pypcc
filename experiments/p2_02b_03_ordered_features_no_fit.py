"""P2-02B/P2-03: frozen UCI rows/schema and approved float64 transforms ONLY.

Strictly no graph, label mask, random seed consumption or PCC propagation.
Every official source outer ZIP and selected member is verified against immutable
P1 first. Original row/feature order is preserved, never sorted. Only feature-
matrix/target-order SHA and aggregate schema metadata are exported, NEVER rows,
patient/grain/customer records, or transformed arrays.

Reuses byte-identical, separately audited P1 source parsers from PR #260:
these are fetched into the *ephemeral CI checkout*, without changing production.
"""
import argparse
from collections import Counter
from decimal import Decimal, InvalidOperation
import hashlib
import io
import json
import math
from pathlib import Path
import subprocess
import sys
import zipfile

import numpy as np
from experiments import p2_02a_official_uci_byte_integrity as source
from experiments import p1_offline_candidate_integrity as p1_offline
from experiments import p1_alternative_domain_source_quality as p1_alternate

ROOT = Path(__file__).resolve().parents[1]
EXPECTED_PARSER_SOURCE_COMMIT = "096a6c127eeeaa6b37171c99824a7a80a040aa72"
PARSER_BLOBS = {
    "experiments/p1_offline_candidate_integrity.py":
        "1a38a36b8737effb71b49f90049dfdf43ae6d9ee",
    "experiments/p1_alternative_domain_source_quality.py":
        "1cea28ad67105923166bb1f8acc993520b370e83",
}
NUMPY_VERSION = "2.5.4"
OPENPYXL_VERSION = "3.1.5"
WHOLESALE_COLUMNS = (
    "Channel", "Region", "Fresh", "Milk", "Grocery", "Frozen",
    "Detergents_Paper", "Delicassen",
)
WHOLESALE_X = WHOLESALE_COLUMNS[2:]


class OrderedSourceBlocked(ValueError):
    pass


def require(ok, reason):
    if not ok:
        raise OrderedSourceBlocked(reason)


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def git_blob_sha(data):
    return hashlib.sha1(b"blob " + str(len(data)).encode("ascii") + b"\0" + data).hexdigest()


def parser_sources_checked():
    # Git object pins are required; installed/dynamic parser versions are disallowed.
    out = subprocess.run(
        ["git", "-C", str(ROOT), "cat-file", "-t", EXPECTED_PARSER_SOURCE_COMMIT],
        capture_output=True, text=True, check=False)
    require(out.returncode == 0 and out.stdout.strip() == "commit",
            "Missing immutable original P1 parser Git commit")
    for path, expected in PARSER_BLOBS.items():
        raw = (ROOT / path).read_bytes()
        require(git_blob_sha(raw.replace(bytes((13, 10)), bytes((10,)))) == expected,
                "Parser file does not match original approved research Git blob: " + path)
        original = subprocess.run(
            ["git", "-C", str(ROOT), "rev-parse",
             EXPECTED_PARSER_SOURCE_COMMIT + ":" + path],
            capture_output=True, text=True, check=False)
        require(original.returncode == 0 and original.stdout.strip() == expected,
                "P1 parser Git object SHA mismatch")
    return PARSER_BLOBS.copy()


def load_member_bytes(outer, spec):
    """Verify P2-02A source byte identity *again*, before any row parsing."""
    zipped, infos = source.safe_zip(outer)
    try:
        if spec["uci_id"] == 850:
            nested, name = source.select_member(zipped, infos, "Raisin_Dataset.zip")
            require(name.rsplit("/", 1)[-1] == "Raisin_Dataset.zip",
                    "Raisin official inner archive name changed")
            inner, inner_infos = source.safe_zip(nested)
            try:
                member, path = source.select_member(
                    inner, inner_infos, spec["file_name"])
            finally:
                inner.close()
        else:
            member, path = source.select_member(zipped, infos, spec["file_name"])
    finally:
        zipped.close()
    require(path.rsplit("/", 1)[-1] == spec["file_name"],
            "Pinned original member filename case changed")
    require(sha256(member) == spec["file_sha256"],
            "Original dataset bytes drift before ordered row parsing")
    return member


def parse_ordered(spec, member):
    """Original format parsers; no record sorting, deduplication or imputation."""
    if spec["uci_id"] == 212:
        rows, columns, classes = p1_offline.parse_arff(member)
        require(len(columns) == 7 and columns[-1].strip().lower() == "class" and
                len(set(x.casefold() for x in columns)) == len(columns),
                "Vertebral source numeric feature/class columns drift")
        require(classes == set(spec["class_support"]),
                "Vertebral declared three-class ARFF target drift")
        return columns, rows, classes, list(range(6)), 6
    if spec["uci_id"] == 850:
        import openpyxl
        with io.BytesIO(member) as payload:
            book = openpyxl.load_workbook(
                payload, read_only=True, data_only=False)
            try:
                require(book.sheetnames ==
                        ["Raisin_Grains_Dataset", "Citation_Request"],
                        "Raisin original worksheet layout/order changed")
                data = book["Raisin_Grains_Dataset"]
                require(data.max_row == 901 and data.max_column == 8,
                        "Raisin worksheet data dimensions changed")
            finally:
                book.close()
        rows, columns, _ = p1_offline.parse_xlsx(member)
        require(len(columns) == 8 and columns[-1] == "Class" and
                len(set(x.casefold() for x in columns)) == 8,
                "Raisin seven numeric columns or Class target drift")
        return columns, rows, None, list(range(7)), 7
    if spec["uci_id"] == 292:
        columns, rows = p1_alternate.parse_wholesale(member)
        require(tuple(columns) == WHOLESALE_COLUMNS, "Wholesale source column order drift")
        require(spec["actual_target"] == "Channel" and
                spec["target_is_derived"] is True and
                spec["excluded_predictors"] == ["Channel", "Region"] and
                list(WHOLESALE_X) == spec["predictive_features"],
                "Wholesale derived Channel target or Region/X exclusions drift")
        return columns, rows, None, list(range(2, 8)), 6
    raise OrderedSourceBlocked("Unexpected unapproved dataset")


def finite_float(token):
    """Finite real source features only; original rows are never dropped."""
    try:
        val = Decimal(str(token).strip())
    except (InvalidOperation, ValueError, TypeError) as exc:
        raise OrderedSourceBlocked("Source contains missing/non-numeric feature") from exc
    require(val.is_finite() and abs(val) <= Decimal("1e100"),
            "Source feature nonfinite or out-of-bounds")
    numerical = float(val)
    require(math.isfinite(numerical), "Feature not representable as finite float64")
    return numerical


def canonical_class(dataset_id, token):
    value = str(token).strip(" '\"")
    if dataset_id == 292:
        try:
            normalized = p1_alternate.normalize_number(value)
        except (InvalidOperation, ValueError) as exc:
            raise OrderedSourceBlocked("Wholesale target is not original numeric code") from exc
        value = str(normalized)
    require(bool(value), "Missing source target class")
    return value


def numeric_rows_and_labels(spec, columns, rows, feature_indices):
    """Preserves complete original source row order; no sorting/label-based X."""
    expected_count = spec["original_n_rows"]
    expected_features = spec["n_features"]
    require(len(rows) == expected_count and len(feature_indices) == expected_features,
            "P1 frozen row or feature cardinality drift")
    labels = []
    raw = np.empty((expected_count, expected_features), dtype=np.float64, order="C")
    require(len(columns) >= expected_features + 1, "Missing source target column")
    target_col = columns.index("Channel") if spec["uci_id"] == 292 else len(columns)-1
    require(target_col not in feature_indices and len(set(feature_indices)) == len(feature_indices),
            "Source target leaked into numeric predictors")
    for i, row in enumerate(rows):
        require(len(row) == len(columns), "Original source row width drift")
        label = canonical_class(spec["uci_id"], row[target_col])
        require(label in spec["class_support"], "Unknown original target code")
        labels.append(label)
        for j, col in enumerate(feature_indices):
            raw[i, j] = finite_float(row[col])
    counts = dict(sorted(Counter(labels).items()))
    require(counts == spec["class_support"],
            "Original source label supports differ from frozen P1 science")
    require(raw.shape == (expected_count, expected_features) and
            raw.dtype == np.dtype("float64") and np.isfinite(raw).all(),
            "Raw features failed the exact source shape/dtype/finite check")
    return raw, tuple(labels), counts


def array_fingerprint(value, column_names):
    """Shape/dtype/ordered feature names + little-endian C-order float64 bytes."""
    require(type(value) is np.ndarray and value.ndim == 2 and
            value.dtype == np.dtype("float64") and np.isfinite(value).all(),
            "Only finite 2D float64 array fingerprint allowed")
    require(len(column_names) == value.shape[1] and
            all(isinstance(x, str) and bool(x) for x in column_names),
            "Ordered predictor names not supplied")
    canonical = np.asarray(value, dtype="<f8", order="C")
    blob = canonical.tobytes(order="C")
    metadata = {"schema": "pypcc-P2-ordered-float64-LE-Crowmajor-v1",
                "shape": list(value.shape), "dtype": "<f8",
                "byte_order": "little", "layout": "C_rowmajor",
                "ordered_predictor_names": list(column_names)}
    prefix = json.dumps(metadata, sort_keys=True,
                        separators=(",", ":"), ensure_ascii=True).encode("ascii")
    return {"representation": metadata, "data_bytes": len(blob),
            "raw_numeric_bytes_sha256": sha256(blob),
            "shape_dtype_feature_order_and_numeric_bytes_sha256":
            sha256(prefix + b"\0" + blob)}


def label_fingerprint(labels, order):
    require(tuple(sorted(set(labels))) == tuple(order),
            "Original class lexicographic order mismatch")
    class_indices = {label: i for i, label in enumerate(order)}
    encoded = np.array([class_indices[x] for x in labels], dtype="<i8")
    return {"original_target_order_sha256": sha256(encoded.tobytes(order="C")),
            "label_encoding": dict((label, i) for i, label in enumerate(order)),
            "label_rows": len(labels),
            "target_order_serialization": "little-endian int64 indices by original row"}


def approved_transform(X, *, wholesale):
    """P1: log1p for six Wholesale spending columns, then ddof=0 z-score."""
    require(X.dtype == np.dtype("float64") and X.ndim == 2 and
            np.isfinite(X).all(), "Invalid raw float64 matrix")
    if wholesale:
        require((X >= 0).all(), "Wholesale annual spending must be nonnegative")
        transformed_input = np.log1p(X)
    else:
        transformed_input = np.array(X, dtype=np.float64, copy=True, order="C")
    require(np.isfinite(transformed_input).all(),
            "Nonfinite values after approved log1p")
    mu = np.mean(transformed_input, axis=0, dtype=np.float64)
    sigma = np.std(transformed_input, axis=0, ddof=0, dtype=np.float64)
    require(np.isfinite(mu).all() and np.isfinite(sigma).all()
            and (sigma > 0.0).all(), "Zero variance / nonfinite z-score parameters")
    Z = np.asarray((transformed_input - mu) / sigma, dtype=np.float64, order="C")
    require(Z.shape == X.shape and np.isfinite(Z).all(),
            "Approved transform changed shape or introduced nonfinite values")
    return Z, {"method": "float64_np_log1p_then_ddof0_population_zscore"
               if wholesale else "float64_ddof0_population_zscore",
               "source_rows_or_columns_excluded": 0,
               "labels_used_to_select_or_scale_features": False,
               "numpy_version": np.__version__,
               "pop_std_ddof": 0,
               "mean_abs_max_after_zscore":
               float(np.abs(np.mean(Z, axis=0, dtype=np.float64)).max()),
               "unit_variance_max_abs_error":
               float(np.abs(np.std(Z, axis=0, ddof=0, dtype=np.float64) - 1.0).max())}


def evaluate_one(spec):
    outer, _ = source.download_official(spec)
    require(sha256(outer) == spec["zip_sha256"],
            "Source archive SHA mismatch before parsing")
    member = load_member_bytes(outer, spec)
    columns, rows, declared, indices, expected_features = parse_ordered(spec, member)
    require(expected_features == spec["n_features"],
            "Dataset parser did not select frozen feature count")
    if spec["uci_id"] == 292:
        names = list(WHOLESALE_X)
    else:
        names = [columns[i] for i in indices]
    X, labels, counts = numeric_rows_and_labels(spec, columns, rows, indices)
    ordered_class_names = tuple(sorted(spec["class_support"]))
    class_evidence = label_fingerprint(labels, ordered_class_names)
    transformed, transform_evidence = approved_transform(
        X, wholesale=(spec["uci_id"] == 292))
    raw_fingerprint = array_fingerprint(X, names)
    transformed_fingerprint = array_fingerprint(transformed, names)
    require(raw_fingerprint["representation"]["shape"] ==
            transformed_fingerprint["representation"]["shape"],
            "Transformation unexpectedly changed source row/column identity")
    return {
        "uci_id": spec["uci_id"],
        "registry_index": spec["registry_index"],
        "source_zip_sha256": spec["zip_sha256"],
        "member_sha256": spec["file_sha256"],
        "source_member_exact_filename": spec["file_name"],
        "original_source_rows": len(rows),
        "original_data_columns": columns,
        "ordered_numeric_predictor_names": names,
        "derived_target": "Channel" if spec["uci_id"] == 292 else "Class",
        "excluded_model_predictors": ["Channel", "Region"] if spec["uci_id"] == 292 else ["Class"],
        "target_class_support": counts,
        "class_order": list(ordered_class_names),
        "original_target_index_order": class_evidence,
        "raw_X": raw_fingerprint,
        "approved_transformed_X": transformed_fingerprint,
        "transformation": transform_evidence,
        "source_original_order_retained_without_sort_or_dedup": True,
        "observation_data_rows_exported": 0,
        "raw_or_transformed_matrix_exported": False,
        "graph_or_masks_generated": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    import importlib.metadata
    require(np.__version__ == NUMPY_VERSION, "Unexpected NumPy runtime version")
    require(importlib.metadata.version("openpyxl") == OPENPYXL_VERSION,
            "Unexpected XLSX parser runtime version")
    checked = parser_sources_checked()
    protocol = json.loads(source.P1_JSON.read_text(encoding="utf-8"))
    manifest = json.loads(source.P1_MANIFEST.read_text(encoding="utf-8"))
    registry = source.validate_registry(protocol, manifest)
    records = [evaluate_one(spec) for spec in registry]
    report = {
        "schema": "pypcc-P2-02B-03-ordered-source-and-features-no-fit/v1",
        "status": "EXACT_SOURCE_SCHEMA_AND_APPROVED_FLOAT64_TRANSFORMS_HASHED_NO_PCC_FITS",
        "P1_scientific_sha256": source.P1_SHA,
        "source_parser_PR": 260,
        "source_parser_exact_original_commit": EXPECTED_PARSER_SOURCE_COMMIT,
        "source_parser_original_git_blobs": checked,
        "numpy_version": np.__version__,
        "openpyxl_version": importlib.metadata.version("openpyxl"),
        "original_dataset_registry_order": list(source.DATASET_IDS),
        "datasets": records,
        "P2_02B_rows_columns_classes_and_order_pass": True,
        "P2_02_complete_gate_pass": True,
        "P2_03_transformed_X_and_hashes_candidate_pass": True,
        "P2_03_complete_execution_environment_gate_pass": False,
        "original_source_rows_preserved": True,
        "ground_truth_only_used_for_class_registry_not_feature_transforms": True,
        "source_raw_values_persisted_or_uploaded": False,
        "graphs_generated": 0,
        "label_masks_generated": 0,
        "PCC_propagations_executed": 0,
        "PCC_real_source_fits_executed": 0,
        "P2_04_P2_05_verified": False,
        "P2_execution_preflight_sha256": None,
        "P2_execution_authorized": False,
    }
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n",
                           encoding="utf-8")
    print(json.dumps({
        "status": report["status"],
        "dataset_ids": report["original_dataset_registry_order"],
        "dataset_rows": [x["original_source_rows"] for x in records],
        "raw_X_sha256": [
            x["raw_X"]["shape_dtype_feature_order_and_numeric_bytes_sha256"]
            for x in records],
        "transformed_X_sha256": [
            x["approved_transformed_X"]["shape_dtype_feature_order_and_numeric_bytes_sha256"]
            for x in records],
        "PCC_real_source_fits_executed": 0,
        "P2_execution_authorized": False,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
