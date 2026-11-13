"""P2-02A: exact official UCI ZIP + selected-member byte identity, no PCC.

Reads the immutable P1 scientific registry, then downloads ONLY its 3 exact
official HTTPS ZIPs. Verifies SHA-256 on outer ZIP and the precisely named
member (including Raisin's ZIP-in-ZIP), with strict archive safety limits.
NO tabular parsing, original rows, label masks, features, graphs or PCC calls.
Nothing containing real source bytes is saved to a GitHub Actions artifact.
"""
import argparse
import datetime as dt
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import urllib.parse
import urllib.request
import zipfile

from experiments import p2_01_source_platform_preflight as provenance

ROOT = Path(__file__).resolve().parents[1]
P1_JSON = ROOT / "docs/current_p1_frozen_scientific_protocol_2026-11-12.json"
P1_MANIFEST = ROOT / "docs/current_p1_scientific_sha_manifest_2026-11-12.json"
P1_SHA = "f0111f153ade3d72191c107ace039e1835e5c27db2cf4b3df8d610b41b2f4166"

# Fail-closed source registry: P1 SHA also cryptographically protects complete
# dataset metadata; these literal identifiers prevent a changed UCI roster.
DATASET_IDS = (212, 850, 292)
EXPECTED_FILE_NAMES = {
    212: "column_3C_weka.arff",
    850: "Raisin_Dataset.xlsx",
    292: "Wholesale customers data.csv",
}
MAX_ARCHIVE_BYTES = 16 * 1024 * 1024
MAX_MEMBER_BYTES = 32 * 1024 * 1024
MAX_ARCHIVE_TOTAL_UNCOMPRESSED = 32 * 1024 * 1024
MAX_ZIP_MEMBERS = 40
TRUSTED_HOST = "archive.ics.uci.edu"


class SourceBlocked(ValueError):
    pass


def require(ok, reason):
    if not ok:
        raise SourceBlocked(reason)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def validate_registry(protocol, manifest):
    require(provenance.canonical_sha(protocol) == P1_SHA ==
            manifest.get("approved_P1_scientific_protocol_sha256"),
            "Frozen P1 science protocol/manifest digest mismatch")
    require(protocol.get("status") ==
            "SCIENTIFICALLY_APPROVED_EXPLORATORY_P1_ONLY" and
            protocol.get("schema") == "pypcc-P1-scientific-protocol-frozen/v1",
            "P1 scientific approval scope mismatch")
    require(protocol.get("approvals_and_execution", {}).get("P2_execution_approved") is False and
            manifest.get("P2_execution_authorized") is False and
            manifest.get("P2_preflight_sha256") is None,
            "P1 never authorizes real scientific PCC execution")
    ds = protocol.get("selected_datasets")
    require(type(ds) is list and len(ds) == len(DATASET_IDS) and
            tuple(x["uci_id"] for x in ds) == DATASET_IDS and
            tuple(x["registry_index"] for x in ds) == (0, 1, 2),
            "The three frozen UCI dataset IDs/order drifted")
    for item in ds:
        dataset_id = item["uci_id"]
        require(item["file_name"] == EXPECTED_FILE_NAMES[dataset_id] and
                item["experiment_type"] == "EXPLORATORY_ONLY",
                "Frozen UCI member/task identity changed")
        expected = "https://archive.ics.uci.edu/static/public/" + str(dataset_id) + "/"
        require(item["official_source_url"].startswith(expected) and
                item["official_source_url"].endswith(".zip") and
                len(item["zip_sha256"]) == len(item["file_sha256"]) == 64,
                "Frozen UCI URL and/or checksum invalid")
        safe_url(item["official_source_url"], expected=item["official_source_url"])
    return ds


def safe_url(url, *, expected):
    """Restrict redirects to the same approved literal URL, no host drift."""
    parsed = urllib.parse.urlsplit(url)
    require(url == expected and parsed.scheme == "https" and
            parsed.hostname == TRUSTED_HOST and parsed.port in (None, 443)
            and parsed.username is None and parsed.password is None and
            parsed.query == "" and parsed.fragment == "" and
            parsed.path.startswith("/static/public/") and parsed.path.endswith(".zip"),
            "Unapproved HTTPS source or redirect")
    return url


class LockedRedirects(urllib.request.HTTPRedirectHandler):
    def __init__(self, expected):
        self.expected = expected
        super().__init__()

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        safe_url(newurl, expected=self.expected)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def download_official(spec):
    url = spec["official_source_url"]
    safe_url(url, expected=url)
    opener = urllib.request.build_opener(LockedRedirects(url))
    req = urllib.request.Request(
        url, headers={"User-Agent": "PyPCC-P2-02A-frozen-source-bytes-no-fit/1.0"})
    try:
        with opener.open(req, timeout=35) as response:
            safe_url(response.geturl(), expected=url)
            require(response.status == 200, "Unexpected UCI HTTP status")
            length = response.headers.get("Content-Length")
            if length is not None:
                require(length.isdecimal() and int(length) <= MAX_ARCHIVE_BYTES,
                        "UCI Content-Length exceeds safety bound")
            raw = response.read(MAX_ARCHIVE_BYTES + 1)
            require(len(raw) <= MAX_ARCHIVE_BYTES, "UCI download too large")
            if length is not None:
                require(int(length) == len(raw), "Truncated UCI HTTP body")
            response_meta = {
                "source_host": TRUSTED_HOST,
                "http_status": response.status,
                "requested_url": url,
                "effective_url": response.geturl(),
                "content_length_if_present": length,
                "ETag_if_present": response.headers.get("ETag"),
                "Last_Modified_if_present": response.headers.get("Last-Modified"),
                "archive_download_bytes": len(raw),
            }
    except (OSError, TimeoutError) as exc:
        raise SourceBlocked("Official UCI HTTPS acquisition failed, no fallback") from exc
    require(sha(raw) == spec["zip_sha256"],
            "UCI official outer ZIP SHA-256 mismatch: FAIL CLOSED")
    return raw, response_meta


def safe_zip(raw):
    require(len(raw) <= MAX_ARCHIVE_BYTES, "Oversized compressed ZIP input")
    try:
        archive = zipfile.ZipFile(io.BytesIO(raw))
        infos = archive.infolist()
        require(0 < len(infos) <= MAX_ZIP_MEMBERS,
                "Unexpected number of ZIP members")
        require(sum(i.file_size for i in infos) <= MAX_ARCHIVE_TOTAL_UNCOMPRESSED,
                "ZIP expanded contents exceed memory budget")
        seen = set()
        for item in infos:
            name = item.filename
            path = PurePosixPath(name)
            require(bool(name) and not name.startswith(("/", "\\")) and
                    "\\" not in name and ":" not in name and
                    ".." not in path.parts and path.parts[0] != "." and
                    item.file_size <= MAX_MEMBER_BYTES and
                    item.compress_size <= MAX_ARCHIVE_BYTES and
                    item.flag_bits & 1 == 0 and
                    ((item.external_attr >> 16) & 0o170000) != 0o120000,
                    "Unsafe ZIP entry, encrypted member, symlink or path")
            name_key = name.casefold()
            require(name_key not in seen, "Ambiguous duplicate ZIP entry")
            seen.add(name_key)
        return archive, infos
    except (zipfile.BadZipFile, zipfile.LargeZipFile) as exc:
        raise SourceBlocked("Malformed ZIP, refusing source") from exc


def select_member(archive, infos, expected_name):
    matching = [
        item for item in infos
        if not item.is_dir() and
        PurePosixPath(item.filename).name.casefold() == expected_name.casefold()
    ]
    require(len(matching) == 1, "Expected exactly one pinned UCI member")
    item = matching[0]
    try:
        content = archive.read(item)
    except (RuntimeError, zipfile.BadZipFile, EOFError) as exc:
        raise SourceBlocked("Selected member CRC/decompression failure") from exc
    require(len(content) == item.file_size and len(content) <= MAX_MEMBER_BYTES,
            "Selected member size mismatch")
    return content, item.filename


def member_from_official_zip(outer, spec):
    # This selection is byte-only, no spreadsheet, ARFF or CSV data parsing.
    outer_zip, members = safe_zip(outer)
    nested = None
    try:
        if spec["uci_id"] == 850:
            # UCI Raisin official outer ZIP has a nested Raisin_Dataset.zip.
            inner_bytes, wrapper = select_member(
                outer_zip, members, "Raisin_Dataset.zip")
            nested = {
                "nested_zip_member": wrapper,
                "nested_zip_sha256_observed": sha(inner_bytes),
                "nested_zip_bytes": len(inner_bytes),
            }
            inner_zip, inner_infos = safe_zip(inner_bytes)
            try:
                selected, member = select_member(
                    inner_zip, inner_infos, spec["file_name"])
            finally:
                inner_zip.close()
        else:
            selected, member = select_member(
                outer_zip, members, spec["file_name"])
    finally:
        outer_zip.close()
    require(sha(selected) == spec["file_sha256"],
            "Pinned selected UCI data member SHA-256 mismatch: FAIL CLOSED")
    return {"selected_member_path": member,
            "selected_member_basename": PurePosixPath(member).name,
            "selected_member_sha256": sha(selected),
            "selected_member_bytes": len(selected),
            "nested_wrapper": nested,
            "actual_data_rows_examined": 0,
            "original_source_data_bytes_in_report": False}


def audit_one(spec):
    outer, http = download_official(spec)
    evidence = member_from_official_zip(outer, spec)
    return {
        "uci_id": spec["uci_id"],
        "registry_index": spec["registry_index"],
        "selected_source_name": spec["source_name"],
        "P1_zip_sha256": spec["zip_sha256"],
        "verified_outer_zip_sha256": sha(outer),
        "P1_member_sha256": spec["file_sha256"],
        "source": http,
        "member": evidence,
        "official_source_authenticated_by_independent_signature": False,
    }


def audit():
    protocol = json.loads(P1_JSON.read_text(encoding="utf-8"))
    manifest = json.loads(P1_MANIFEST.read_text(encoding="utf-8"))
    selected = validate_registry(protocol, manifest)
    output = [audit_one(spec) for spec in selected]
    return {
        "schema": "pypcc-P2-02A-frozen-UCI-byte-provenance/v1",
        "status": "THREE_EXACT_OFFICIAL_ZIPS_AND_MEMBERS_VERIFIED_NO_ROWS_OR_FITS",
        "verified_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "P1_scientific_sha256": P1_SHA,
        "selected_uci_ids_in_approved_order": list(DATASET_IDS),
        "sources": output,
        "source_zip_and_selected_member_integrity_pass": True,
        "P2_02A_exact_raw_byte_identity_pass": True,
        "P2_02B_rows_columns_targets_and_original_order_verified": False,
        "P2_02_entire_gate_pass": False,
        "P2_03_transformed_features_verified": False,
        "real_source_bytes_downloaded_in_ephemeral_runner": True,
        "real_record_level_values_persisted_or_exported": False,
        "real_graphs_generated": 0,
        "real_label_masks_generated": 0,
        "PCC_propagations_executed": 0,
        "real_PCC_fits_executed": 0,
        "P2_execution_preflight_sha256": None,
        "P2_execution_authorized": False,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", required=True, type=Path)
    args = p.parse_args()
    result = audit()
    args.output.write_text(json.dumps(result, sort_keys=True, indent=2) + "\n",
                           encoding="utf-8")
    print(json.dumps({
        "status": result["status"],
        "uci_ids": result["selected_uci_ids_in_approved_order"],
        "all_source_hashes_match": True,
        "real_data_rows_examined": 0,
        "real_PCC_fits_executed": 0,
        "P2_execution_authorized": False,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
