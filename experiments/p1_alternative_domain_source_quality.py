"""Source-only candidate-domain diversity audit; NOT a PCC experiment.

Read UCI 292 Wholesale customers (derived CHANNEL target) and UCI 379
Website Phishing from official HTTPS ZIPs in an ephemeral CI runner.

No model, RNG, graph, labels sampling, fitting, or row/pair data export.
The source task and feature columns are fixed BEFORE looking at PCC outcomes.
Both candidates remain UNSELECTED until scientific review.
"""
import argparse
from collections import Counter
import csv
from decimal import Decimal, InvalidOperation
import hashlib
import io
import json
from pathlib import PurePosixPath
import re
import urllib.parse
import urllib.request
import zipfile


DATASETS = {
    292: {
        "source_url": "https://archive.ics.uci.edu/static/public/292/wholesale%2Bcustomers.zip",
        "source_member_basename": "Wholesale customers data.csv",
        "n": 440,
        "all_columns": [
            "Channel", "Region", "Fresh", "Milk", "Grocery", "Frozen",
            "Detergents_Paper", "Delicassen",
        ],
        "numeric_features": [
            "Fresh", "Milk", "Grocery", "Frozen",
            "Detergents_Paper", "Delicassen",
        ],
        "class_counts": {"1": 298, "2": 142},
        "proposed_target": "Channel",
        "target_interpretation": {"1": "Horeca", "2": "Retail"},
        "excluded_columns": ["Region"],
        "data_domain": "wholesale_spending",
        "target_approval_required": True,
        "caveat": (
            "UCI metadata lists Region as its target. This is an explicitly "
            "derived Channel-target binary task, excluding Region from X."
        ),
    },
    379: {
        "source_url": "https://archive.ics.uci.edu/static/public/379/website%2Bphishing.zip",
        "source_member_basename": "PhishingData.arff",
        "n": 1353,
        "all_columns": [
            "SFH", "popUpWindow", "SSLfinal_State", "Request_URL",
            "URL_of_Anchor", "web_traffic", "URL_Length",
            "age_of_domain", "having_IP_Address", "Result",
        ],
        "numeric_features": [
            "SFH", "popUpWindow", "SSLfinal_State", "Request_URL",
            "URL_of_Anchor", "web_traffic", "URL_Length",
            "age_of_domain", "having_IP_Address",
        ],
        "class_counts": {"-1": 702, "0": 103, "1": 548},
        "proposed_target": "Result",
        "target_interpretation": {
            "-1": "phishing", "0": "suspicious", "1": "legitimate",
        },
        "excluded_columns": [],
        "data_domain": "website_security",
        "target_approval_required": True,
        "caveat": (
            "All nine features are categorical codes -1/0/1. Numeric "
            "encoding does NOT authorize Euclidean continuous-feature "
            "geometry or ordinal-distance interpretation."
        ),
    },
}
MAX_ZIP = 1024 * 1024
MAX_UNCOMPRESSED = 4 * 1024 * 1024
ARFF_ATTR = re.compile(r"^@attribute\s+(?:'([^']+)'|\"([^\"]+)\"|(\S+))\s+(.+)$", re.I)


class CandidateReviewBlocked(ValueError):
    pass


def check(ok, message):
    if not ok:
        raise CandidateReviewBlocked(message)


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def allowed_url(url):
    u = urllib.parse.urlparse(url)
    check(
        u.scheme == "https" and
        u.hostname == "archive.ics.uci.edu" and
        u.port in (None, 443) and
        u.username is None and u.password is None and
        u.path.startswith("/static/public/") and
        u.path.endswith(".zip"),
        "Unexpected external URL/redirect",
    )
    return url


class StrictRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        allowed_url(newurl)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def read_official_archive(dataset_id, *, payload_override=None):
    """Return archive/member bytes and hashes; no extracted files, no uploads."""
    dataset = DATASETS[dataset_id]
    url = allowed_url(dataset["source_url"])
    if payload_override is None:
        req = urllib.request.Request(
            url, headers={"User-Agent": "pypcc-p1-no-fit-domain-source-review/1"},
        )
        with urllib.request.build_opener(StrictRedirect()).open(
                req, timeout=40) as response:
            check(response.status == 200, "Official UCI HTTP status was not 200")
            allowed_url(response.geturl())
            size = response.headers.get("Content-Length")
            check(size is None or int(size) <= MAX_ZIP, "Large HTTP payload")
            raw = response.read(MAX_ZIP + 1)
    else:
        raw = payload_override
    check(len(raw) <= MAX_ZIP, "Official archive exceeds maximum source size")
    try:
        with zipfile.ZipFile(io.BytesIO(raw)) as archive:
            info = archive.infolist()
            check(0 < len(info) <= 40, "Unexpected ZIP entry count")
            check(sum(x.file_size for x in info) <= MAX_UNCOMPRESSED,
                  "ZIP total uncompressed size exceeds limit")
            candidates = []
            for entry in info:
                name = entry.filename
                check(not name.startswith(("/", "\\")) and
                      "\\" not in name and
                      ".." not in PurePosixPath(name).parts,
                      "Unsafe ZIP path")
                check((entry.external_attr >> 16) & 0o170000 != 0o120000,
                      "ZIP symlink rejected")
                check(entry.file_size <= MAX_UNCOMPRESSED, "ZIP member oversized")
                if not entry.is_dir() and (
                    PurePosixPath(name).name.lower() ==
                    dataset["source_member_basename"].lower()
                ):
                    candidates.append(entry)
            check(len(candidates) == 1,
                  "Exactly one expected official ZIP data member required")
            check(archive.testzip() is None, "Failed ZIP CRC integrity test")
            entry = candidates[0]
            member = archive.read(entry)
            check(len(member) == entry.file_size, "Incomplete data member")
            return {
                "source_url": url,
                "archive_sha256": sha256(raw),
                "archive_size_bytes": len(raw),
                "data_member": entry.filename,
                "data_member_sha256": sha256(member),
            }, member
    except (OSError, RuntimeError, zipfile.BadZipFile) as exc:
        raise CandidateReviewBlocked("ZIP structural or CRC failure") from exc


def parse_wholesale(member):
    try:
        records = list(csv.reader(io.StringIO(
            member.decode("utf-8-sig"), newline=""
        )))
    except UnicodeError as exc:
        raise CandidateReviewBlocked("Wholesale CSV is not UTF-8") from exc
    check(bool(records), "Empty Wholesale CSV")
    expected = DATASETS[292]["all_columns"]
    check([h.strip() for h in records[0]] == expected,
          "Wholesale source schema/column order changed")
    return expected, records[1:]


def parse_phishing(member):
    try:
        rawlines = member.decode("utf-8-sig").splitlines()
    except UnicodeError as exc:
        raise CandidateReviewBlocked("Phishing ARFF is not UTF-8") from exc
    attrs, body, has_data = [], [], False
    for raw in rawlines:
        line = raw.strip()
        if not line or line.startswith("%"):
            continue
        if not has_data:
            if line.lower() == "@data":
                has_data = True
            elif line.lower().startswith("@attribute"):
                m = ARFF_ATTR.fullmatch(line)
                check(m is not None, "Invalid ARFF attribute declaration")
                attr_name = m.group(1) or m.group(2) or m.group(3)
                attr_type = m.group(4).strip().lower()
                check(attr_type in {"numeric", "real", "integer"} or
                      (attr_type.startswith("{") and attr_type.endswith("}")),
                      "Unsupported ARFF feature/target type")
                attrs.append(attr_name)
        else:
            check(not line.startswith("{"), "Sparse ARFF is not allowed")
            body.append(line)
    check(has_data, "ARFF data section absent")
    check(attrs == DATASETS[379]["all_columns"],
          "Phishing source feature order or class target changed")
    return attrs, list(csv.reader(body))


def normalize_number(token):
    try:
        value = Decimal(str(token).strip())
    except (ValueError, InvalidOperation) as exc:
        raise CandidateReviewBlocked("Non-numeric source feature or target") from exc
    check(value.is_finite() and abs(value) <= Decimal("1e100"),
          "Nonfinite or implausibly large numeric source value")
    return value.normalize()


def inspect_source(dataset_id, *, payload_override=None):
    check(dataset_id in DATASETS, "Unknown candidate rejected")
    source, raw_member = read_official_archive(
        dataset_id, payload_override=payload_override,
    )
    spec = DATASETS[dataset_id]
    if dataset_id == 292:
        columns, records = parse_wholesale(raw_member)
    else:
        columns, records = parse_phishing(raw_member)
    check(len(records) == spec["n"], "Source row count changed")
    features = spec["numeric_features"]
    check(len(features) == len(set(features)), "Duplicate feature names")
    target_col = columns.index(spec["proposed_target"])
    feature_indices = [columns.index(name) for name in features]
    dropped = [name for name in columns if name not in
               features + [spec["proposed_target"]]]
    check(dropped == spec["excluded_columns"], "Feature leakage/omission drift")
    count = Counter()
    features_only = Counter()
    complete = Counter()
    first_class = {}
    conflicting_patterns = set()
    other_counts = Counter()
    for row in records:
        check(len(row) == len(columns), "Source row has wrong number of columns")
        target = str(normalize_number(row[target_col]))
        check(target in spec["class_counts"], "Unknown class code")
        feature_tuple = tuple(normalize_number(row[i]) for i in feature_indices)
        if dataset_id == 379:
            check(all(value in {-1, 0, 1} for value in feature_tuple),
                  "Categorical web-code other than -1/0/1")
        if dataset_id == 292:
            region = str(normalize_number(row[columns.index("Region")]))
            check(region in {"1", "2", "3"}, "Unknown region code")
            other_counts[region] += 1
        count[target] += 1
        features_only[feature_tuple] += 1
        complete[(feature_tuple, target)] += 1
        if feature_tuple in first_class and first_class[feature_tuple] != target:
            conflicting_patterns.add(feature_tuple)
        first_class.setdefault(feature_tuple, target)
    check(dict(count) == spec["class_counts"], "Published target support drift")
    check(all(n >= 10 for n in count.values()),
          "Minimum 3/class prospective task cannot be supported")
    repeated = sum(n - 1 for n in features_only.values())
    exact = sum(n - 1 for n in complete.values())
    result = {
        "schema": "pypcc-p1-alternate-domains-official-uci-no-fit/v1",
        "uci_id": dataset_id,
        "domain": spec["data_domain"],
        "source": source,
        "rows": len(records),
        "features_used": features,
        "features_excluded": dropped,
        "proposed_target": spec["proposed_target"],
        "class_support_by_uci_code": dict(sorted(count.items())),
        "original_source_numeric_feature_count": len(features),
        "exact_repeated_feature_vector_surplus_rows": repeated,
        "exact_duplicate_full_row_surplus": exact,
        "conflicting_label_feature_patterns": len(conflicting_patterns),
        "distinct_feature_patterns": len(features_only),
        "region_support_if_wholesale": dict(sorted(other_counts.items())) if
            dataset_id == 292 else None,
        "feature_type_warning": spec["caveat"],
        "categorical_features_euclidean_ready": False if dataset_id == 379 else None,
        "target_task_independently_reviewed": False,
        "near_duplicates_or_record_grouping_verified": False,
        "novelty_in_prior_papers_and_local_research_verified": False,
        "datasets_selected": 0,
        "scientific_P1_frozen": False,
        "P2_execution_preflight_sha256": None,
        "PCC_fits_executed": 0,
        "execution_authorized": False,
    }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-id", type=int, choices=sorted(DATASETS),
                        required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    report = inspect_source(args.dataset_id)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(report, fh, sort_keys=True, indent=2)
        fh.write("\n")
    print(json.dumps({
        "uci_id": report["uci_id"],
        "archive_sha256": report["source"]["archive_sha256"],
        "member_sha256": report["source"]["data_member_sha256"],
        "class_support": report["class_support_by_uci_code"],
        "repeated_feature_surplus": report[
            "exact_repeated_feature_vector_surplus_rows"
        ],
        "duplicate_full_row_surplus": report["exact_duplicate_full_row_surplus"],
        "cross_class_feature_collisions": report[
            "conflicting_label_feature_patterns"
        ],
        "PCC_fits_executed": 0,
        "execution_authorized": False,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
