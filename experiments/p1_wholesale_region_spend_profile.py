"""Descriptive, SOURCE-ONLY Wholesale source-quality diagnostics (no PCC fits).

Uses official UCI 292 bytes, the *derived* proposed Channel target, six spending
features and Region only for stratified descriptive diagnostics (never as X).
No inference, significance tests, graphs, label masks, sampling, fitting,
preprocessing choices, rows or identifying records are output.

Statistical conventions are set in source before inspecting the output:
  - quantiles: nearest-rank x[ceil(p*n)-1], p={0.25,0.50,0.75,0.95,0.99};
  - upper Tukey fence: Q3 + 1.5*(Q3-Q1), strict exceedance;
  - Cramer's V = sqrt(chi-square/(n*min(r-1,c-1))) without p-value;
  - all features are raw numeric monetary units; no model transformation.
"""
import argparse
import collections
from decimal import Decimal
import hashlib
import json
from math import ceil, sqrt

from experiments import p1_alternative_domain_source_quality as uci


DATASET_ID = 292
FEATURES = tuple(uci.DATASETS[DATASET_ID]["numeric_features"])
P_QUANTILES = (Decimal("0.25"), Decimal("0.50"), Decimal("0.75"),
               Decimal("0.95"), Decimal("0.99"))


class DiagnosticBlocked(ValueError):
    pass


def _require(ok, reason):
    if not ok:
        raise DiagnosticBlocked(reason)


def nearest_rank(xs, p):
    _require(bool(xs), "No source values")
    return xs[ceil(float(p)*len(xs))-1]


def _decimal(value):
    return Decimal(str(value))


def describe_wholesale_rows(rows, columns):
    """Pure, synthetic-testable source-statistics; never returns individual rows."""
    _require(columns == uci.DATASETS[DATASET_ID]["all_columns"],
             "Original UCI source column order changed")
    _require(len(rows) == 440, "Expected original 440 customers")
    target_i, region_i = columns.index("Channel"), columns.index("Region")
    feature_indices = [columns.index(x) for x in FEATURES]
    _require("Region" not in FEATURES and "Channel" not in FEATURES,
             "Leaking categorical source variables as numeric features")
    contingency = {"1": {"1": 0, "2": 0}, "2": {"1": 0, "2": 0},
                   "3": {"1": 0, "2": 0}}
    numeric_columns = {name: [] for name in FEATURES}
    for row in rows:
        _require(len(row) == len(columns), "Wrong source row width")
        ch = str(uci.normalize_number(row[target_i]))
        region = str(uci.normalize_number(row[region_i]))
        _require(ch in {"1", "2"} and region in {"1", "2", "3"},
                 "Unknown Channel/Region class")
        contingency[region][ch] += 1
        for name, index in zip(FEATURES, feature_indices):
            val = uci.normalize_number(row[index])
            _require(val >= 0, "Annual monetary spending unexpectedly negative")
            numeric_columns[name].append(val)
    _require({k: sum(v[k] for v in contingency.values()) for k in ("1","2")}
             == {"1": 298, "2": 142}, "Original channel support changed")

    n = len(rows)
    col_totals = {"1": 298, "2": 142}
    by_region = {r: sum(sub.values()) for r, sub in contingency.items()}
    chi2 = Decimal(0)
    for region in ("1", "2", "3"):
        for channel in ("1", "2"):
            expected = _decimal(by_region[region])*_decimal(col_totals[channel])/_decimal(n)
            _require(expected > 0, "Empty expected Region x Channel cell")
            observed = _decimal(contingency[region][channel])
            chi2 += (observed-expected)**2/expected
    cramers_v = sqrt(float(chi2)/n)  # min(3-1,2-1)=1

    metrics = {}
    for name, values in numeric_columns.items():
        ordered = sorted(values)
        q = {str(p): nearest_rank(ordered, p) for p in P_QUANTILES}
        q1, q3 = q["0.25"], q["0.75"]
        upper = q3 + Decimal("1.5")*(q3-q1)
        _require(ordered[0] >= 0, "Unexpected negative source values")
        mean = sum(values)/len(values)
        metrics[name] = {
            "units": "original monetary units, raw source",
            "min": float(ordered[0]),
            "max": float(ordered[-1]),
            "mean": float(mean),
            "quantile_method": "nearest_rank_ceil_p_n",
            "q25": float(q1),
            "median_q50": float(q["0.50"]),
            "q75": float(q3),
            "q95": float(q["0.95"]),
            "q99": float(q["0.99"]),
            "strict_tukey_high_outlier_count": sum(v > upper for v in ordered),
            "high_outlier_fence_q75_plus_1_5_iqr": float(upper),
            "max_over_median_if_positive": (
                float(ordered[-1]/q["0.50"]) if q["0.50"] > 0 else None
            ),
            "values_modified_or_excluded": 0,
        }

    return {
        "schema": "pypcc-p1-wholesale-origin-region-channel-diagnostics/v1",
        "phase": "NO_FIT_SOURCE_QUALITY_REVIEW_ONLY",
        "n": n,
        "derived_target": "Channel",
        "raw_numeric_features_used": list(FEATURES),
        "region_is_excluded_from_model_features": True,
        "region_is_used_only_for_group_level_quality_diagnostics": True,
        "region_channel_counts_by_original_UCI_codes": contingency,
        "region_totals": by_region,
        "channel_totals": col_totals,
        "channel_1_share_by_region": {
            r: float(_decimal(contingency[r]["1"])/_decimal(by_region[r]))
            for r in ("1", "2", "3")
        },
        "pearson_chi_square_descriptive_not_significance_test": float(chi2),
        "cramers_v_descriptive_not_causal": cramers_v,
        "features": metrics,
        "no_p_value_or_automatic_exclusion": True,
        "does_not_establish_independent_customers_or_batches": True,
        "does_not_prescribe_scaling_or_PCC_graph": True,
        "source_rows_not_exported": True,
        "P1_dataset_selected": False,
        "P1_scientific_approval": False,
        "P2_execution_preflight_sha256": None,
        "PCC_fits_executed": 0,
        "execution_authorized": False,
    }


def assess_official_source(pinned_record):
    """Only direct UCI official ZIP, verify source pins before source profiling."""
    spec = next((x for x in pinned_record.get("datasets", [])
                 if x.get("id") == 292), None)
    _require(spec is not None, "Wholesale source identity absent from pins")
    _require(pinned_record.get("status") == "REVIEW_ONLY_NOT_SELECTED" and
             pinned_record.get("execution_authorized") is False and
             pinned_record.get("real_PCC_fits_executed") == 0 and
             pinned_record.get("selected_dataset_ids") == [],
             "Prior evidence cannot be treated as experiment approval")
    source, member = uci.read_official_archive(292)
    _require(source["archive_sha256"] == spec["archive_sha256"] and
             source["data_member_sha256"] == spec["member_sha256"],
             "UCI source ZIP/member SHA drift: do not inspect changed source")
    columns, rows = uci.parse_wholesale(member)
    report = describe_wholesale_rows(rows, columns)
    report["source_url"] = source["source_url"]
    report["archive_sha256"] = source["archive_sha256"]
    report["selected_member_sha256"] = source["data_member_sha256"]
    return report


def validate_pinned_diagnostics(result, ref):
    """Strict, non-approval replay of source-derived aggregated fields."""
    _require(ref.get("schema") == "pypcc-p1-wholesale-origin-region-channel-diagnostics/v1",
             "Expected exact source-only diagnostic schema")
    for flag in ("execution_authorized","P1_dataset_selected","P1_scientific_approval"):
        _require(ref.get(flag) is False, "Historical diagnostic cannot authorize P1/P2")
    _require(ref.get("PCC_fits_executed") == 0 and
             ref.get("P2_execution_preflight_sha256") is None,
             "No model execution can be inferred")
    for field in (
        "source_url", "archive_sha256", "selected_member_sha256",
        "n", "derived_target", "raw_numeric_features_used",
        "region_channel_counts_by_original_UCI_codes",
        "region_totals","channel_totals",
        "channel_1_share_by_region",
        "pearson_chi_square_descriptive_not_significance_test",
        "cramers_v_descriptive_not_causal", "features",
    ):
        _require(result.get(field) == ref.get(field),
                 "Source-only diagnostics drift: " + field)
    result["reproduced_pinned_quality_only"] = True
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-quality-manifest", required=True)
    p.add_argument("--expect-diagnostics", default=None)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    with open(args.source_quality_manifest, encoding="utf-8") as f:
        manifest = json.load(f)
    result = assess_official_source(manifest)
    if args.expect_diagnostics:
        with open(args.expect_diagnostics, encoding="utf-8") as f:
            frozen = json.load(f)
        validate_pinned_diagnostics(result, frozen)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(result, f, sort_keys=True, indent=2)
        f.write("\n")
    # Full output contains ONLY aggregate source distributions, no original rows.
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
