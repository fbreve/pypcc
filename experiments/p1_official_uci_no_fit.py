"""Acquire official UCI candidate ZIPs *only for offline quality review*, NO PCC.

Runs in an ephemeral CI runner. Does not generate labels, construct a graph,
fit a model, or select P1 datasets. Does not upload the raw UCI files.
The official HTTPS URL and the downloaded source bytes are both preserved
in the manifest, but HTTPS alone is not an independent signed checksum.
"""
import argparse
import datetime
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import tempfile
import urllib.parse
import urllib.request
import zipfile

from experiments import p1_offline_candidate_integrity as offline


UCI_URLS = {
    212: "https://archive.ics.uci.edu/static/public/212/vertebral%2Bcolumn.zip",
    545: "https://archive.ics.uci.edu/static/public/545/rice%2Bcammeo%2Band%2Bosmancik.zip",
    850: "https://archive.ics.uci.edu/static/public/850/raisin.zip",
}
TRUSTED_HOSTS = {"archive.ics.uci.edu", "www.archive.ics.uci.edu"}


class ExternalDataBlocked(ValueError):
    pass


def _check_url(uri):
    u = urllib.parse.urlparse(uri)
    if (u.scheme != "https" or u.hostname not in TRUSTED_HOSTS
            or u.port not in (None, 443) or u.username is not None
            or u.password is not None):
        raise ExternalDataBlocked("Non-UCI HTTPS URL or unsafe redirect")
    if not u.path.startswith("/static/public/") or not u.path.endswith(".zip"):
        raise ExternalDataBlocked("Wrong official UCI ZIP URL")
    return uri


class _UciRedirects(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        _check_url(newurl)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def download(dataset_id):
    url = _check_url(UCI_URLS[dataset_id])
    opener = urllib.request.build_opener(_UciRedirects())
    request = urllib.request.Request(
        url, headers={"User-Agent": "pypcc-research-no-fit-dataset-integrity/1.0"}
    )
    with opener.open(request, timeout=35) as response:
        effective = _check_url(response.geturl())
        actual_len = response.headers.get("Content-Length")
        if actual_len is not None and int(actual_len) > offline.MAX_ARCHIVE:
            raise ExternalDataBlocked("UCI archive exceeds byte safety budget")
        raw = response.read(offline.MAX_ARCHIVE + 1)
        if len(raw) > offline.MAX_ARCHIVE:
            raise ExternalDataBlocked("UCI download exceeds byte safety budget")
        if actual_len is not None and len(raw) != int(actual_len):
            raise ExternalDataBlocked("Incomplete UCI download")
        meta = {
            "requested_url": url,
            "effective_url": effective,
            "http_status": response.status,
            "archive_bytes": len(raw),
            "download_sha256": hashlib.sha256(raw).hexdigest(),
            "etag_if_present": response.headers.get("ETag"),
            "last_modified_if_present": response.headers.get("Last-Modified"),
        }
        if response.status != 200:
            raise ExternalDataBlocked(f"Non-200 UCI HTTP status: {response.status}")
    return raw, meta


def _safe_member_list(raw):
    try:
        with zipfile.ZipFile(io.BytesIO(raw)) as z:
            info = z.infolist()
            if len(info) > 40:
                raise ExternalDataBlocked("Too many ZIP members")
            if sum(x.file_size for x in info) > offline.MAX_MEMBER:
                raise ExternalDataBlocked("Combined decompressed ZIP exceeds cap")
            for i in info:
                if (i.filename.startswith(("/", "\\")) or
                        "\\" in i.filename or
                        ".." in PurePosixPath(i.filename).parts or
                        i.file_size > offline.MAX_MEMBER or
                        (i.external_attr >> 16) & 0o170000 == 0o120000):
                    raise ExternalDataBlocked("Unsafe ZIP member metadata")
            return info
    except zipfile.BadZipFile as e:
        raise ExternalDataBlocked(f"UCI download is not a ZIP: {e}") from e


def _select(raw, name, suffix):
    entries = _safe_member_list(raw)
    wanted = [x for x in entries if not x.is_dir()
              and PurePosixPath(x.filename).name.lower() == name.lower()
              and x.filename.lower().endswith(suffix)]
    if len(wanted) != 1:
        raise ExternalDataBlocked(
            f"Expected exactly one named file {name}, found {len(wanted)} "
            f"of {len(entries)} archived entries"
        )
    return wanted[0].filename


def inspect_one(dataset_id):
    raw, source = download(dataset_id)
    suffix = offline.TASKS[dataset_id]["file"]
    zips = []
    try:
        member = _select(raw, suffix, Path(suffix).suffix.lower())
    except ExternalDataBlocked:
        # Some UCI datasets contain an extra dataset-specific ZIP wrapper.
        if dataset_id != 850:
            raise
        inner = _select(raw, "Raisin_Dataset.zip", ".zip")
        with zipfile.ZipFile(io.BytesIO(raw)) as z:
            nested = z.read(inner)
        if len(nested) > offline.MAX_ARCHIVE:
            raise ExternalDataBlocked("Nested Raisin ZIP exceeds audit limit")
        zips.append({
            "member": inner,
            "bytes": len(nested),
            "sha256": hashlib.sha256(nested).hexdigest(),
        })
        raw = nested
        member = _select(raw, suffix, ".xlsx")

    with tempfile.TemporaryDirectory(prefix="uci-no-fit-") as folder:
        path = Path(folder) / "uci-official-candidate.zip"
        path.write_bytes(raw)
        try:
            quality = offline.review(dataset_id, path, member=member)
        except offline.FixtureRejected:
            # Only safe schema metadata; NEVER print feature vectors, rows or
            # other record-level values from patient/grain datasets.
            with zipfile.ZipFile(io.BytesIO(raw)) as z:
                selected_bytes = z.read(member)
            if dataset_id == 212:
                from collections import Counter
                rows, header, declared = offline.parse_arff(selected_bytes)
                print(json.dumps({
                    "schema_diagnostic": "ARFF metadata only",
                    "class_name_counts": dict(Counter(
                        str(row[-1]).strip() for row in rows
                    )),
                    "declared_nominal_values": sorted(declared),
                    "attribute_names": header,
                }, sort_keys=True))
            elif dataset_id == 850:
                from openpyxl import load_workbook
                wb = load_workbook(io.BytesIO(selected_bytes),
                                   read_only=True, data_only=False)
                print(json.dumps({
                    "schema_diagnostic": "XLSX metadata only",
                    "worksheets": [
                        {
                            "title": ws.title, "row_count": ws.max_row,
                            "column_count": ws.max_column,
                            "header_names": list(next(ws.iter_rows(
                                min_row=1, max_row=1, values_only=True
                            ))),
                        } for ws in wb.worksheets
                    ],
                }, sort_keys=True))
                wb.close()
            raise
    return {
        "schema": "pypcc-p1-uci-official-https-no-fit-review/v1",
        "retrieved_at_utc": datetime.datetime.now(
            datetime.timezone.utc
        ).isoformat(),
        "source": source,
        "embedded_archives": zips,
        "quality": quality,
        "official_source_signed_checksum_verified": False,
        "original_publication_exposure_completely_checked": False,
        "subject_or_group_independence_verified": False,
        "dataset_selected_for_P1": False,
        "P1_scientifically_frozen": False,
        "P2_execution_preflight_sha256": None,
        "PCC_fits_executed": 0,
        "execution_authorized": False,
    }


def verify_against_previous_observation(report, prior):
    """Compare public source bytes/results to an earlier audit, NOT to a science seal."""
    if prior.get("schema") != "pypcc-p1-official-uci-https-archive-quality-results/v1":
        raise ExternalDataBlocked("Unexpected prior quality-record schema")
    case = [v for v in prior.get("datasets", [])
            if v.get("uci_id") == report["quality"]["dataset_id"]]
    if len(case) != 1:
        raise ExternalDataBlocked("Missing or ambiguous prior candidate ID")
    expected = case[0]
    actual = report["quality"]
    comparisons = {
        "source_zip_sha256": report["source"]["download_sha256"],
        "selected_member_sha256": actual["source_member_sha256"],
        "observed_rows": actual["rows"],
        "numeric_features": actual["numeric_features"],
        "class_support": actual["class_support"],
        "exact_duplicate_full_rows": actual["exact_duplicate_full_rows"],
        "exact_repeated_feature_rows": actual["exact_repeated_feature_rows"],
        "conflicting_label_feature_patterns":
            actual["conflicting_label_feature_patterns"],
        "url": report["source"]["requested_url"],
    }
    for key, value in comparisons.items():
        if expected.get(key) != value:
            raise ExternalDataBlocked(
                f"Official source or quality result differs from prior audit: {key}"
            )
    report["matches_previous_source_only_audit"] = True
    report["previous_source_only_audit_is_not_scientific_approval"] = True
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-id", type=int, choices=sorted(UCI_URLS),
                        required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expect-record", type=Path, default=None,
                        help="Reject changed source bytes vs earlier no-fit audit")
    args = parser.parse_args()
    report = inspect_one(args.dataset_id)
    if args.expect_record is not None:
        prior = json.loads(args.expect_record.read_text(encoding="utf-8"))
        verify_against_previous_observation(report, prior)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    # Avoid exposing any raw row or dataset, but emit useful quality summary.
    print(json.dumps({
        "dataset_id": args.dataset_id,
        "official_download_sha256": report["source"]["download_sha256"],
        "member_sha256": report["quality"]["source_member_sha256"],
        "class_support": report["quality"]["class_support"],
        "exact_repeated_feature_rows":
            report["quality"]["exact_repeated_feature_rows"],
        "exact_duplicate_full_rows":
            report["quality"]["exact_duplicate_full_rows"],
        "conflicting_label_feature_patterns":
            report["quality"]["conflicting_label_feature_patterns"],
        "PCC_fits_executed": 0,
        "execution_authorized": False,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
