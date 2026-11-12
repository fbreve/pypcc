"""OFFLINE fixture-integrity screening for three UNSELECTED P1 UCI candidates.

This reads ONLY an explicitly supplied local ARFF/CSV/XLSX or UCI ZIP archive.
No network, PCC import, graph, label mask, sampling, optimizer or model fit.
All reports are descriptive and are NOT scientific or execution approval.
For XLSX, optional openpyxl must be installed in the audit environment; its
version and input source SHA must later be frozen independently in P2.
"""
import argparse
import collections
import csv
from decimal import Decimal, InvalidOperation
import hashlib
import io
import json
import math
from pathlib import Path, PurePosixPath
import re
import zipfile


MAX_ARCHIVE = 16 * 1024 * 1024
MAX_MEMBER = 32 * 1024 * 1024
MAX_ROWS = 20000
MAX_CELLS = 200000
TASKS = {
    212: {
        "name": "Vertebral Column 3-class", "n": 310, "d": 6,
        "labels": {"DH": 60, "SL": 150, "NO": 100},
        "file": "column_3c_weka.arff",
    },
    545: {
        "name": "Rice Cammeo Osmancik", "n": 3810, "d": 7,
        "labels": {"Cammeo": None, "Osmancik": None},
        "file": "rice_cammeo_osmancik.arff",
    },
    850: {
        "name": "Raisin Kecimen Besni", "n": 900, "d": 7,
        "labels": {"Kecimen": 450, "Besni": 450},
        "file": "raisin_dataset.xlsx",
    },
}
ATTR = re.compile(r"^@attribute\s+(?:'([^']+)'|\"([^\"]+)\"|(\S+))\s+(.+)$", re.I)


class FixtureRejected(ValueError):
    pass


def require(ok, why):
    if not ok:
        raise FixtureRejected(why)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def safely_read(path, member):
    """Reject oversized files, traversal and ambiguous archives; never extract."""
    path = Path(path)
    require(path.is_file(), "Explicit local fixture path does not exist")
    require(path.stat().st_size <= MAX_ARCHIVE, "Local source/archive is too large")
    raw = path.read_bytes()
    require(len(raw) <= MAX_ARCHIVE, "Archive size changed during inspection")
    outer_sha = sha(raw)
    if path.suffix.lower() != ".zip":
        require(member is None, "Member argument only applies to ZIP inputs")
        require(path.suffix.lower() in {".arff", ".csv", ".xlsx"},
                "Input must be ZIP, ARFF, CSV or XLSX")
        return raw, path.name, outer_sha, outer_sha
    try:
        archive = zipfile.ZipFile(io.BytesIO(raw))
        entries = archive.infolist()
        require(len(entries) <= 40, "Unexpectedly many ZIP entries")
        for entry in entries:
            name = entry.filename
            parts = PurePosixPath(name).parts
            require(
                not name.startswith("/") and not name.startswith("\\") and
                ".." not in parts and "\\" not in name and
                entry.file_size <= MAX_MEMBER and entry.compress_size <= MAX_ARCHIVE,
                "Unsafe ZIP member path or uncompressed size"
            )
            require(not ((entry.external_attr >> 16) & 0o170000) == 0o120000,
                    "ZIP symbolic links are prohibited")
        require(sum(e.file_size for e in entries) <= MAX_MEMBER,\n                "Combined ZIP contents exceed safe audit limit")\n        require(archive.testzip() is None, "Corrupt ZIP member CRC")\n        files = [e for e in entries if not e.is_dir() and
                 e.filename.lower().endswith((".arff", ".csv", ".xlsx"))]
        if member is not None:
            files = [e for e in files if e.filename == member]
        require(len(files) == 1, "Select exactly one data file by --member")
        chosen = files[0]
        require(chosen.file_size <= MAX_MEMBER, "ZIP member exceeds byte limit")
        payload = archive.read(chosen)
        require(len(payload) == chosen.file_size, "ZIP size inconsistency")
        return payload, chosen.filename, outer_sha, sha(payload)
    except (zipfile.BadZipFile, RuntimeError, OSError) as exc:
        raise FixtureRejected(f"ZIP integrity error: {exc}") from exc


def parse_arff(raw):
    try:
        lines = raw.decode("utf-8-sig").splitlines()
    except UnicodeError as exc:
        raise FixtureRejected("ARFF must be UTF-8 for this protocol") from exc
    attrs, data_lines, in_data = [], [], False
    for line in lines:
        t = line.strip()
        if not t or t.startswith("%"):
            continue
        if not in_data:
            if t.lower() == "@data":
                in_data = True
            elif t.lower().startswith("@attribute"):
                match = ATTR.match(t)
                require(match is not None, "Cannot parse ARFF attribute")
                attrs.append(((match.group(1) or match.group(2) or match.group(3)),
                              match.group(4).strip()))
        else:
            require(not t.startswith("{"), "Sparse ARFF not supported")
            data_lines.append(t)
    require(in_data and attrs, "Missing ARFF @data/attributes")
    for name, type_ in attrs[:-1]:
        require(type_.lower() in {"numeric", "real", "integer"},
                f"Non-numeric attribute {name}")
    require(attrs[-1][0].strip().lower() == "class", "ARFF class must be last")
    classes = attrs[-1][1].strip()
    require(classes.startswith("{") and classes.endswith("}"),
            "Nominal ARFF target declaration required")
    permitted = {v.strip(" '\"") for v in classes[1:-1].split(",")}
    require(permitted, "Empty ARFF target declaration")
    return list(csv.reader(data_lines)), [name for name, _ in attrs], permitted


def parse_csv(raw):
    try:
        rows = list(csv.reader(io.StringIO(raw.decode("utf-8-sig"), newline="")))
    except (UnicodeError, csv.Error) as exc:
        raise FixtureRejected("CSV decode/parse error") from exc
    require(rows, "Empty CSV")
    header = [h.strip() for h in rows[0]]
    require(header[-1].lower() == "class" and
            len(set(h.lower() for h in header)) == len(header),
            "CSV must include distinct header columns and last Class target")
    return rows[1:], header, None


def parse_xlsx(raw):
    try:
        from openpyxl import load_workbook
    except ImportError as exc:
        raise FixtureRejected(
            "XLSX requires an explicitly installed optional openpyxl in audit environment"
        ) from exc
    try:
        workbook = load_workbook(io.BytesIO(raw), read_only=True, data_only=False)
        require(len(workbook.sheetnames) == 1, "XLSX must have one unambiguous worksheet")
        sheet = workbook.active
        require(sheet.max_row is None or sheet.max_row <= MAX_ROWS + 1,
                "XLSX sheet too large")
        require(sheet.max_column is None or sheet.max_column <= 10,
                "XLSX has unexpected columns")
        rows = []
        for row in sheet.iter_rows(values_only=False):
            vals = []
            for cell in row:
                require(cell.data_type != "f", "Formulas forbidden in data")
                vals.append("" if cell.value is None else str(cell.value))
            rows.append(vals)
            require(len(rows) <= MAX_ROWS + 1, "XLSX has too many rows")
        workbook.close()
    except (OSError, ValueError, zipfile.BadZipFile) as exc:
        raise FixtureRejected(f"XLSX structural parse error: {exc}") from exc
    require(rows, "Empty XLSX")
    header = [x.strip() for x in rows[0]]
    require(header[-1].lower() == "class", "XLSX last column must be Class")
    return rows[1:], header, None


def review(dataset_id, path, *, member=None):
    require(dataset_id in TASKS, "Unknown UCI candidate: refuse replacement")
    task = TASKS[dataset_id]
    raw, filename, outer_sha, member_sha = safely_read(path, member)
    ext = Path(filename).suffix.lower()
    require(Path(filename).name.lower() == task["file"],
            f"Wrong task or source member: expected {task['file']}")
    if ext == ".arff":
        rows, header, nominal = parse_arff(raw)
    elif ext == ".csv":
        rows, header, nominal = parse_csv(raw)
    elif ext == ".xlsx":
        rows, header, nominal = parse_xlsx(raw)
    else:
        raise FixtureRejected("Unsupported dataset format")
    require(len(header) == task["d"] + 1, "Unexpected feature count")
    require(len(rows) == task["n"], "Unexpected observation count")
    require(len(rows) * len(header) <= MAX_CELLS, "Too many spreadsheet cells")
    counts = collections.Counter()
    feature_rows, complete_rows = collections.Counter(), collections.Counter()
    first_label = {}
    conflicting_feature_rows = set()
    for i, row in enumerate(rows):
        require(len(row) == len(header), f"Wrong row width at row {i + 1}")
        numeric = []
        for v in row[:-1]:
            try:
                d = Decimal(str(v).strip())
            except (InvalidOperation, ValueError) as exc:
                raise FixtureRejected(f"Invalid feature at row {i + 1}") from exc
            require(d.is_finite() and abs(d) <= Decimal("1E100"),
                    f"Nonfinite/out-of-range feature at row {i + 1}")
            numeric.append(d.normalize())
        label = str(row[-1]).strip(" '\"")
        require(label in task["labels"], f"Unexpected target class at row {i + 1}")
        if nominal is not None:
            require(label in nominal, "ARFF class absent from nominal declaration")
        feat = tuple(numeric)
        counts[label] += 1
        feature_rows[feat] += 1
        complete_rows[(feat, label)] += 1
        if feat in first_label and first_label[feat] != label:
            conflicting_feature_rows.add(feat)
        first_label.setdefault(feat, label)
    require(set(counts) == set(task["labels"]), "Missing declared target class")
    for name, expected in task["labels"].items():
        if expected is not None:
            require(counts[name] == expected,
                    f"Published class support mismatch for {name}")
        require(counts[name] >= 10, f"Insufficient class support: {name}")
    return {
        "schema": "pypcc-p1-local-fixture-quality-report/v1",
        "phase": "UNSELECTED_CANDIDATE_OFFLINE_QUALITY_ONLY",
        "dataset_id": dataset_id,
        "task": task["name"],
        "input_file": str(path),
        "archive_sha256": outer_sha,
        "source_member": filename,
        "source_member_sha256": member_sha,
        "rows": len(rows),
        "numeric_features": task["d"],
        "class_support": dict(sorted(counts.items())),
        "exact_repeated_feature_rows": sum(n - 1 for n in feature_rows.values()),
        "exact_duplicate_full_rows": sum(n - 1 for n in complete_rows.values()),
        "conflicting_label_feature_patterns": len(conflicting_feature_rows),
        "near_duplicates_checked": False,
        "independence_of_subjects_verified": False,
        "source_authenticity_independently_verified": False,
        "scientific_novelty_verified": False,
        "datasets_selected": 0,
        "P1_frozen": False,
        "P2_execution_preflight_sha256": None,
        "execution_authorized": False,
        "model_fits_executed": 0,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-id", type=int, required=True, choices=sorted(TASKS))
    parser.add_argument("--file", type=Path, required=True,
                        help="Local explicit input; tool never downloads")
    parser.add_argument("--member", default=None,
                        help="Exact ZIP member name when source contains multiple data files")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    report = review(args.dataset_id, args.file, member=args.member)
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
