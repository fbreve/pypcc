"""Offline packaging and byte-integrity verification; never runs PCC or downloads.

Each --input is a preserved result directory with artifact_sha256.json.
Original files, including outliers and historical analysis scripts, stay untouched.
"""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import subprocess
import zipfile


def digest(data):
    return hashlib.sha256(data).hexdigest()


def safe_name(name):
    path = PurePosixPath(name)
    if not name or path.is_absolute() or ".." in path.parts or "\\" in name or ":" in name:
        raise ValueError(f"Unsafe archive path: {name}")
    return name


def verify_archive(path):
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)):
            raise ValueError("Duplicate archive members")
        for name in names:
            safe_name(name)
        hashes = json.loads(archive.read("checksums.json"))
        if set(names) != set(hashes) | {"checksums.json"}:
            raise ValueError("Archive member/checksum inventory differs")
        for name, expected in hashes.items():
            if digest(archive.read(name)) != expected:
                raise ValueError(f"Checksum mismatch: {name}")
        return {"status": "verified", "files": len(names),
                "bytes": Path(path).stat().st_size,
                "sha256": digest(Path(path).read_bytes())}


def build_archive(output, inputs, commits=(), repository=Path(".")):
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    payload = {}
    for root in map(Path, inputs):
        hashes = json.loads((root / "artifact_sha256.json").read_text(encoding="utf-8"))
        actual = {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()}
        if actual != set(hashes) | {"artifact_sha256.json"}:
            raise ValueError(f"Source inventory differs: {root}")
        for name in sorted(actual):
            safe_name(name)
            data = (root / name).read_bytes()
            if name in hashes and digest(data) != hashes[name]:
                raise ValueError(f"Source checksum mismatch: {root / name}")
            member = safe_name(f"results/{root.name}/{name}")
            if member in payload:
                raise ValueError(f"Duplicate result root: {root.name}")
            payload[member] = data
    for commit in commits:
        # git archive reads committed trees only; no checkout or metadata writes.
        resolved = subprocess.check_output(
            ["git", "rev-parse", "--verify", f"{commit}^{{commit}}"], cwd=repository,
            text=True).strip()
        payload[f"source/{resolved}.tar"] = subprocess.check_output(
            ["git", "archive", "--format=tar", resolved], cwd=repository)
    payload["README.txt"] = (
        "Offline pypcc performance archive. Original and sensitivity results remain separate.\n"
        "results/ preserves complete original directory names and byte contents.\n"
        "source/ contains exact measured committed source trees (tar format).\n"
        "checksums.json covers every other member; verify ZIP with the repository's\n"
        "tools/archive_performance_results.py --verify PATH. No benchmarks are executed.\n"
        "Historical executor scripts are provenance, not archive verification commands.\n"
        "Consult docs/performance_archiving.md for accounting and interpretation.\n"
    ).encode()
    payload["checksums.json"] = json.dumps(
        {name: digest(data) for name, data in sorted(payload.items())}, indent=2).encode()
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, data in sorted(payload.items()):
            archive.writestr(name, data)
    return verify_archive(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--input", type=Path, action="append", default=[])
    parser.add_argument("--source-commit", action="append", default=[])
    args = parser.parse_args()
    if args.verify and (args.output or args.input or args.source_commit):
        parser.error("--verify cannot be combined with packaging arguments")
    if args.verify:
        result = verify_archive(args.verify)
    elif args.output and args.input:
        result = build_archive(args.output, args.input, args.source_commit)
    else:
        parser.error("Use --verify PATH or --output PATH --input DIRECTORY")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
