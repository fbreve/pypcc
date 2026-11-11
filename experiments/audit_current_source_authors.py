"""Read-only line-by-line Git authorship provenance for current core PCC modules.

A Git blame author is evidence of a committed line, NOT a legal conclusion
about copyright ownership, rights or independent creation.
"""
from collections import Counter
import json
from pathlib import Path
import re
import subprocess

BASE = "8d5f5aa9061a51ec5c4cd782b8ca10a0365049da"
FILES = (
    "pcc.py",
    "pcc_numpy.py",
    "pcc_numba.py",
    "pcc_step.pyx",
    "pcc_graph.py",
    "pcc_validation.py",
    "pcc_metrics.py",
)
OTHER = ("setup.py", "pyproject.toml", "example.py", "benchmark.py")
OUT = Path("current_pypcc_code_line_provenance.json")


def git(*argv):
    return subprocess.run(
        ["git", *argv], check=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, encoding="utf-8",
        errors="replace"
    ).stdout


def blame(filepath, *, aggressive_copy_tracking=False):
    # Ordinary Git blame is primary attribution. Copy/move detection is a
    # secondary sensitivity diagnostic only; generic "else:" may falsely
    # match snippets in other files and is not proof of code reuse.
    opts = ("-w", "-M", "-C", "-C") if aggressive_copy_tracking else ()
    stdout = git("blame", *opts, "--line-porcelain",
                 BASE, "--", filepath)
    # --line-porcelain always emits full metadata for each output source line.
    rows = []
    author, source_sha, line_no = None, None, None
    for line in stdout.splitlines():
        header = re.match(r"^([0-9a-f]{40}|\^[0-9a-f]{39}) (\d+) (\d+)(?: (\d+))?$", line)
        if header:
            source_sha = header.group(1)
            line_no = int(header.group(3))
            author = None
        elif line.startswith("author "):
            author = line[7:]
        elif line.startswith("\t"):
            if author is None or source_sha is None or line_no is None:
                raise AssertionError("unexpected git-blame porcelain record")
            s = line[1:]
            rows.append({"author": author, "source_sha": source_sha,
                         "line": line_no, "text": s})
    expected = Path(filepath).read_text(encoding="utf-8").splitlines()
    if len(rows) != len(expected):
        raise AssertionError(f"{filepath}: blame rows {len(rows)} != source {len(expected)}")
    for row, txt in zip(rows, expected):
        if row["text"] != txt:
            raise AssertionError(f"{filepath}:{row['line']} line text mismatch")
    return rows


def main():
    actual = git("rev-parse", BASE).strip()
    if actual != BASE:
        raise AssertionError("frozen original production SHA unavailable")
    changed = git("diff", "--name-only", BASE, "HEAD", "--", *FILES, *OTHER)
    if changed.strip():
        raise AssertionError("research branch modified original production files")
    report = {
        "source_production_sha": BASE,
        "method": "ordinary git blame --line-porcelain <original SHA> -- path",
        "copy_sensitivity": "git blame -w -M -C -C (secondary, may attribute generic copied lines)",
        "source_sha_frozen": True,
        "files": {},
        "total": {},
        "cautions": [
            "Git blame attributes current lines to commit metadata, not copyright.",
            "A changed line can receive a new author even if adapted from earlier code.",
            "A semantic algorithm translation is not an identical source code line.",
            "Historical fork ancestry remains even when no literal lines survive.",
            "Third-party runtime packages are declared dependencies, not vendored sources.",
            "Generated pcc_step.c is tracked separately from authored pcc_step.pyx.",
        ],
    }
    all_code = Counter()
    all_total = Counter()
    all_nonmaintainer = []
    for filepath in FILES:
        rows = blame(filepath)
        copied = blame(filepath, aggressive_copy_tracking=True)
        totals = Counter(r["author"] for r in rows)
        codes = Counter(r["author"] for r in rows
                        if r["text"].strip() and not r["text"].lstrip().startswith("#"))
        candidates = [
            {"line": r["line"], "commit": r["source_sha"],
             "author": r["author"], "content_preview": r["text"][:140]}
            for r in rows if r["author"].casefold().strip()
            not in {"fabricio breve", "fbreve"}
        ]
        all_nonmaintainer.extend({"file": filepath, **r} for r in candidates)
        report["files"][filepath] = {
            "all_lines": len(rows),
            "nonempty_noncomment_lines": sum(codes.values()),
            "authors_all_lines": dict(totals),
            "authors_nonempty_noncomment": dict(codes),
            "nonmaintainer_line_sample": candidates[:70],
            "nonmaintainer_line_count": len(candidates),
            "aggressive_copy_heuristic_authors_nonempty_noncomment": dict(
                Counter(r["author"] for r in copied
                        if r["text"].strip() and not r["text"].lstrip().startswith("#"))
            ),
        }
        all_code.update(codes)
        all_total.update(totals)
    report["total"] = {
        "all_lines": sum(all_total.values()),
        "nonempty_noncomment_lines": sum(all_code.values()),
        "authors_all_lines": dict(all_total),
        "authors_nonempty_noncomment": dict(all_code),
        "nonmaintainer_current_lines": len(all_nonmaintainer),
        "nonmaintainer_sample": all_nonmaintainer[:100],
    }
    report["packaging_and_example"] = {
        filepath: {
            "authors_all_lines": dict(Counter(r["author"] for r in blame(filepath))),
            "all_lines": len(blame(filepath)),
        }
        for filepath in OTHER if Path(filepath).is_file()
    }
    OUT.write_text(json.dumps(report, indent=2, ensure_ascii=False,
                              sort_keys=True)+"\n", encoding="utf-8")
    print(json.dumps({
        "sha": BASE,
        "total": report["total"],
        "source_files": {k: {
            "nonempty_noncomment_lines": v["nonempty_noncomment_lines"],
            "authors": v["authors_nonempty_noncomment"],
            "nonmaintainer_lines": v["nonmaintainer_line_count"],
            "copy_heuristic_authors": v["aggressive_copy_heuristic_authors_nonempty_noncomment"],
        } for k, v in report["files"].items()},
        "authorship_only_not_copyright": True,
    }, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
