"""Read-only PyPI upstream sdist/source provenance review, no package execution."""
from __future__ import annotations

import ast
from datetime import datetime, timezone
from email.parser import Parser
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import re
import tarfile
import urllib.request

EXPECTED_SHA = "cc807593f9c809158d3fc1473b0087090fe84417a84bf5e0294d9cb8a0802c28"
ORIGINAL_SHA = "a0a12cae2607c01eb2f5ffab7a52540fa721234b"
PACKAGE_URL = (
    "https://files.pythonhosted.org/packages/cd/78/"
    "19650d6689218c643b3f597b8019c1a904dbb5e4c50d9c7a71dba520a32d/"
    "pypcc-0.0.6.tar.gz"
)
UPSTREAM_ROOT = (
    "https://raw.githubusercontent.com/caiocarneloz/pycc/"
    + ORIGINAL_SHA + "/"
)
MAX_ARCHIVE_BYTES = 1048576
MAX_SOURCE_BYTES = 524288
MAX_SOURCE_TOTAL = 2097152
MAX_MEMBERS = 100

def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()

def download(url: str, *, limit: int) -> bytes:
    req = urllib.request.Request(
        url, headers={"User-Agent": "pypcc-original-source-license-audit/1.0"}
    )
    with urllib.request.urlopen(req, timeout=35) as response:
        data = response.read(limit + 1)
        if len(data) > limit:
            raise ValueError("download exceeds capped size")
        return data

def safe_member(info: tarfile.TarInfo) -> bool:
    path = PurePosixPath(info.name)
    if path.is_absolute() or ".." in path.parts or str(path) == ".":
        raise ValueError("unsafe archive path: " + repr(info.name))
    if not (info.isfile() or info.isdir()):
        raise ValueError("non-file archive entry (symlink/device): " + repr(info.name))
    if info.isfile() and info.size > MAX_SOURCE_BYTES:
        raise ValueError("archive member over size budget: " + repr(info.name))
    return info.isfile()

def normalized(b: bytes) -> bytes:
    return b.replace(b"\r\n", b"\n").replace(b"\r", b"\n")

def functions(b: bytes) -> list[str] | None:
    try:
        tree = ast.parse(b.decode("utf-8-sig"))
        names = []
        for item in tree.body:
            if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                names.append(item.name)
                if isinstance(item, ast.ClassDef):
                    names += [
                        item.name + "." + child.name for child in item.body
                        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef))
                    ]
        return names[:100]
    except (UnicodeError, SyntaxError):
        return None

def main():
    original = download(PACKAGE_URL, limit=MAX_ARCHIVE_BYTES)
    actual = sha(original)
    if actual != EXPECTED_SHA:
        raise AssertionError("PyPI original upstream sdist SHA256 does not match PyPI")
    files: dict[str, bytes] = {}
    with tarfile.open(fileobj=io.BytesIO(original), mode="r:gz") as tf:
        members = tf.getmembers()
        if not (1 <= len(members) <= MAX_MEMBERS):
            raise AssertionError("suspicious archive member count")
        for m in members:
            if not safe_member(m):
                continue
            member = tf.extractfile(m)
            if member is None:
                raise AssertionError("cannot read regular member")
            data = member.read(MAX_SOURCE_BYTES + 1)
            if len(data) != m.size:
                raise AssertionError("archive file truncated or over size cap")
            if m.name in files:
                raise AssertionError("duplicate archive path")
            files[m.name] = data
    if sum(map(len, files.values())) > MAX_SOURCE_TOTAL:
        raise AssertionError("decompressed total too large")
    basename = {p: PurePosixPath(p).name for p in files}
    license_names = [
        p for p, name in basename.items()
        if re.match(r"^(LICENSE|LICENCE|COPYING|COPYRIGHT|NOTICE)([._-].*)?$", name, re.I)
    ]
    metadata_names = [
        p for p, name in basename.items()
        if name in ("PKG-INFO", "METADATA")
    ]
    setup_names = [
        p for p, name in basename.items()
        if name == "setup.py"
    ]
    py_sources = [p for p in files if p.endswith(".py")]
    pkg_metadata = []
    for p in metadata_names:
        msg = Parser().parsestr(files[p].decode("utf-8", "replace"))
        pkg_metadata.append({
            "path": p,
            "name": msg.get("Name"),
            "version": msg.get("Version"),
            "author": msg.get("Author"),
            "author_email": msg.get("Author-email"),
            "license_field": msg.get("License"),
            "license_expression": msg.get("License-Expression"),
            "license_files": msg.get_all("License-File", []),
            "license_classifiers": [
                x for x in msg.get_all("Classifier", [])
                if "license" in x.lower()
            ],
            "file_sha256": sha(files[p]),
        })
    setup_evidence = []
    for p in setup_names:
        text = files[p].decode("utf-8", "replace").splitlines()
        matches = [
            {"line": i + 1, "text": line.strip()[:250]}
            for i, line in enumerate(text)
            if re.search("licen[cs]|author|url|copyright|readme", line, re.I)
        ]
        setup_evidence.append({"path": p, "relevant_lines": matches[:30]})
    license_details = []
    for p in license_names:
        b = files[p]
        license_details.append({
            "path": p,
            "sha256": sha(b),
            "bytes": len(b),
            "first_20_lines": b.decode("utf-8", "replace").splitlines()[:20],
            "explicit_MIT_grant": b"Permission is hereby granted" in b
            and b"without restriction" in b,
        })
    upstream = {}
    for fn in ("pcc.py", "demo.py", "README.md"):
        raw = download(UPSTREAM_ROOT + fn, limit=MAX_SOURCE_BYTES)
        upstream[fn] = {
            "url": UPSTREAM_ROOT + fn, "sha256": sha(raw),
            "normalized_sha256": sha(normalized(raw)),
            "bytes": len(raw),
        }
    py_comparisons = []
    for path in py_sources:
        data = files[path]
        py_comparisons.append({
            "sdist_path": path,
            "sha256": sha(data),
            "normalized_sha256": sha(normalized(data)),
            "bytes": len(data),
            "top_level_and_method_names": functions(data),
            "original_GitHub_matches": [
                {"path": name, "exact": sha(data) == v["sha256"],
                 "line_ending_normalized":
                    sha(normalized(data)) == v["normalized_sha256"]}
                for name,v in upstream.items() if name.endswith(".py")
            ],
        })
    # No internet fetched bytes are ever executed, imported, or extracted to disk.
    result = {
        "schema": 1,
        "inspected_utc": datetime.now(timezone.utc).isoformat(),
        "source_url": PACKAGE_URL,
        "verified_sdist_sha256": actual,
        "expected_sdist_sha256": EXPECTED_SHA,
        "source_size": len(original),
        "upstream_git_sha": ORIGINAL_SHA,
        "files": [
            {"path": p, "sha256": sha(b), "bytes": len(b)}
            for p, b in sorted(files.items())
        ],
        "publisher_metadata": pkg_metadata,
        "setup_py_license_lines": setup_evidence,
        "license_or_notice_files": license_details,
        "python_source_comparisons": py_comparisons,
        "upstream_git_sources": upstream,
        "no_source_executed": True,
        "no_source_extracted_to_disk": True,
        "conclusions": {
            "sdist_hash_verified": True,
            "separate_actual_license_document_present": bool(license_names),
            "literal_MIT_grant_present_in_license_document": any(
                x["explicit_MIT_grant"] for x in license_details
            ),
            "upstream_source_exactly_matched": any(
                x["exact"]
                for cmp in py_comparisons
                for x in cmp["original_GitHub_matches"]
            ),
            "upstream_source_normalized_matched": any(
                x["line_ending_normalized"]
                for cmp in py_comparisons
                for x in cmp["original_GitHub_matches"]
            ),
            "grant_to_all_fork_contributions_verified": False,
            "authorize_auto_license_change": False,
        },
    }
    Path("upstream_pypi_0_0_6_inspection.json").write_text(
        json.dumps(result, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    md = [
        "# Original pypcc 0.0.6 verified source-distribution inspection",
        "", "- Original SHA256: \`" + actual + "\` (PASS)",
        "- GitHub upstream immutable tree SHA: \`" + ORIGINAL_SHA + "\`",
        "- Source files: \`" + ", ".join(sorted(files)) + "\`",
        "- LICENSE/COPYING/NOTICE-like files: " +
            (", ".join(license_names) or "**none**"),
        "- Package metadata: \`" + json.dumps(pkg_metadata, ensure_ascii=False) + "\`",
        "- Setup license/author lines: \`" +
            json.dumps(setup_evidence, ensure_ascii=False) + "\`",
        "- Source comparison: \`" +
            json.dumps(py_comparisons, ensure_ascii=False) + "\`",
        "- NO source executed, imported, installed or extracted; no license was added.",
        "", "## Legal-readiness limit",
        "Actual source identity/notice evidence above is distinct from legal rights "
        "for each historical GitHub revision, inherited patches and all additional "
        "copyright holders. License choice for the fork requires a separately "
        "reviewed provenance and attribution decision.",
    ]
    for x in license_details:
        md += ["", "## License or notice: " + x["path"],
               "\`\`\`text", *x["first_20_lines"], "\`\`\`"]
    Path("upstream_pypi_0_0_6_inspection.md").write_text(
        "\n".join(md) + "\n", encoding="utf-8"
    )
    print("VERIFIED_UPSTREAM_PYPI_SDIST_SHA256", actual)
    print("UPSTREAM_GIT_SHA", ORIGINAL_SHA)
    print("SOURCE_MEMBERS", [x["path"] for x in result["files"]])
    print("PACKAGE_METADATA", json.dumps(pkg_metadata))
    print("LICENSE_FILE_FINDINGS", json.dumps(license_details))
    print("SETUP_META_LINES", json.dumps(setup_evidence))
    print("MODULE_COMPARISONS", json.dumps(py_comparisons))
    print("CONCLUSIONS", json.dumps(result["conclusions"]))
    print("SUCCESS: ORIGINAL TARBALL INSPECTED WITHOUT RUNNING UNTRUSTED CODE")

if __name__ == "__main__":
    main()
