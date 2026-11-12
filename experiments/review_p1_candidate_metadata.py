"""Inert, fail-closed P1 *candidate metadata* review; never fetch data or fit.

This audits a manually transcribed UCI-page candidate list against the exact
existing 20-dataset PyPCC inventory. It does NOT verify external source bytes,
publication novelty, source licenses, patient independence, dataset classes or
preprocessing. Passing only makes a candidate list ready for HUMAN REVIEW.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "docs" / "current_p1_candidate_metadata_registry_2026-11-12.json"
HISTORICAL = ROOT / "docs" / "performance_campaign_plan.md"
SCHEMA = "pypcc-p1-candidate-metadata-registry/v0.1"
EXPECTED_SOURCE_BLOB = "35944c493d38046e71790ca4521cc42ce0aa570d"
EXPECTED_IDS = (212, 545, 850)
EXPECTED_METADATA = {
    212: (310, 6, 3, "10.24432/C5K89B"),
    545: (3810, 7, 2, "10.24432/C5MW4Z"),
    850: (900, 7, 2, "10.24432/C5660T"),
}
HISTORICAL_HEADER = "## Frozen 20-dataset inventory"
TABLE_ROW = re.compile(
    r"^\|\s*([^|]+?)\s*\|\s*`([a-z0-9_]+)`\s*\|",
    flags=re.MULTILINE,
)


class CandidateReviewBlocked(ValueError):
    pass


def _require(value, why):
    if not value:
        raise CandidateReviewBlocked(why)


def _git_blob_sha(raw):
    return hashlib.sha1(
        b"blob " + str(len(raw)).encode("ascii") + b"\0" + raw,
    ).hexdigest()


def historical_inventory(raw):
    """Read only the 20 frozen Markdown table rows, not performance results."""
    canonical = raw.replace(b"\r\n", b"\n")
    _require(_git_blob_sha(canonical) == EXPECTED_SOURCE_BLOB,
             "Historical 20-dataset table Git blob changed")
    text = canonical.decode("utf-8")
    _require(text.count(HISTORICAL_HEADER) == 1,
             "Historical benchmark inventory header not unique")
    section = text.split(HISTORICAL_HEADER, 1)[1].split("## Matrix for review", 1)[0]
    rows = [(name.strip(), alias) for name, alias in TABLE_ROW.findall(section)]
    _require(len(rows) == 20 and len(set(rows)) == 20,
             "Historical benchmark must contain 20 distinct source identities")
    _require(len({alias for _, alias in rows}) == 20,
             "Historical aliases must be distinct")
    return rows


def _canonical_sha256(value):
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"),
        ensure_ascii=True, allow_nan=False,
    ).encode("ascii")).hexdigest()


def review(registry, *, historical_raw):
    """Review metadata only; no network, NumPy or PCC imports."""
    _require(type(registry) is dict and registry.get("schema") == SCHEMA,
             "Unexpected candidate metadata schema")
    _require(registry.get("status") == "CANDIDATES_ONLY_NOT_SELECTED",
             "Review refuses any selected or approved candidate status")
    h = registry["historical_registry"]
    _require(h["source"] == "docs/performance_campaign_plan.md" and
             h["source_git_blob_sha"] == EXPECTED_SOURCE_BLOB,
             "Historical inventory provenance mismatch")
    historical = historical_inventory(historical_raw)
    metadata_history = [
        (x["historical_name"], x["cli_alias"]) for x in h["entries"]
    ]
    _require(metadata_history == historical,
             "Recorded historical benchmark inventory differs from source")
    _require(h["not_exhaustive"] is True and
             registry["full_history_exposure_audit_complete"] is False,
             "Cannot claim exhaustive historical exposure clearance")
    _require(registry["p1_draft_pr"] == 258 and
             registry["p1_draft_head"] == "9c8c88aab745e7526a68b2aef958858990f1cfea" and
             registry["p1_draft_checksum_unapproved"] ==
             "fdbf74c1faea06c38b27c935de9167685f372786d277fbdc8726510adc80c854",
             "Unapproved P1 draft reference changed")
    _require(registry["source_base_commit"] ==
             "ea69f690208620d5563c8efece6b2e04369f52c0",
             "Unexpected production base")

    candidates = registry["candidates"]
    _require(type(candidates) is list and len(candidates) == 3 and
             registry["candidate_count_for_review"] == len(candidates),
             "Expected exactly three review-only candidates")
    ids = [c["uci_dataset_id"] for c in candidates]
    _require(ids == list(EXPECTED_IDS),
             "UCI candidate IDs and proposal ordering changed")
    for candidate in candidates:
        uid = candidate["uci_dataset_id"]
        _require(
            (candidate["instances_reported"],
             candidate["numeric_features_reported"],
             candidate["classes_reported"],
             candidate["doi"]) == EXPECTED_METADATA[uid],
            "External metadata altered without review"
        )
        _require(
            candidate["id"].startswith(f"uci-{uid}-") and
            candidate["source_page"].startswith(
                f"https://archive.ics.uci.edu/dataset/{uid}/"
            ),
            "Candidate identity/canonical UCI metadata URL mismatch"
        )
        _require(
            candidate["license_reported"] == "CC BY 4.0" and
            candidate["missing_values_reported"] is False and
            candidate["provenance_review"] == "candidate_metadata_only",
            "Candidate metadata must remain unapproved and source-attributed",
        )
        _require(
            2 <= candidate["classes_reported"] <= 5 and
            candidate["instances_reported"] >=
                candidate["classes_reported"] * 10,
            "Candidate metadata violates provisional class/count bounds",
        )
        _require(
            candidate["minimum_class_count_reported"] is None or
            candidate["minimum_class_count_reported"] >= 10,
            "Reported minority support insufficient",
        )
        _require(
            candidate["name"].strip().lower().replace(" ", "") not in {
                name.strip().lower().replace(" ", "")
                for name, _ in historical
            },
            "Candidate exact name already in 20-dataset campaign",
        )
        _require(
            len(candidate["data_exposure_caveat"]) >= 40 and
            candidate["canonical_data_file_proposed"],
            "Missing dataset caveat or exact file proposal"
        )

    _require(registry["selected_dataset_ids"] == [] and
             registry["dataset_bytes_sha256"] is None and
             registry["source_archive_sha256"] is None,
             "Candidate review must not present a selected/frozen dataset")
    _require(registry["independent_reviewer_approved"] is False and
             registry["license_and_data_integrity_approved"] is False,
             "Candidate metadata cannot assert independent approval")
    _require(registry["run_permission"] == {
        "P1_scientific_protocol_frozen": False,
        "P2_execution_preflight_sha256": None,
        "execution_authorized": False,
        "model_fits_executed": 0,
        "datasets_downloaded": 0,
    }, "Candidate list cannot carry model-fit execution approval")
    _require(
        all(registry[k] is True for k in (
            "no_dataset_loading", "no_PCC_import",
            "no_production_code_changes"
        )),
        "Inert candidate-review constraints changed",
    )
    _require(len(registry["gates_before_selection"]) >= 6,
             "Missing pre-selection scientific review gates")
    _require("NOT an exhaustive" in h["source_scope"],
             "Scope must not overclaim novelty")
    return {
        "schema": "pypcc-p1-candidate-metadata-review/v1",
        "metadata_review": "PASS",
        "candidate_ids": ids,
        "historically_documented_datasets": len(historical),
        "candidates_not_in_frozen_20_by_name": True,
        "novelty_outside_frozen_20_verified": False,
        "candidate_checksum_not_approval": _canonical_sha256(registry),
        "datasets_selected": 0,
        "source_bytes_verified": False,
        "source_license_independently_approved": False,
        "P1_scientific_protocol_frozen": False,
        "P2_execution_preflight_sha256": None,
        "execution_authorized": False,
        "datasets_downloaded": 0,
        "model_fits_executed": 0,
        "review_blockers": registry["gates_before_selection"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, default=REGISTRY)
    parser.add_argument("--historical", type=Path, default=HISTORICAL)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    metadata = json.loads(args.registry.read_text(encoding="utf-8"))
    report = review(metadata, historical_raw=args.historical.read_bytes())
    rendered = json.dumps(report, sort_keys=True, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
