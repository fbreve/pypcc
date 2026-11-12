"""Reproducible *tracked-text only* scan of pinned original MATLAB PCC Git.

Runs on a local Git checkout supplied explicitly by the auditor. Does not fetch
GitHub, read binary/dataset blobs, execute MATLAB or PCC, or claim a complete
scientific-literature exposure audit. No matches means ONLY these exact code
files on this exact Git commit have no literal candidate-name matches.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess


REPO_ROOT = Path(__file__).resolve().parents[1]
RECORD = REPO_ROOT / "docs/current_p1_original_matlab_exposure_audit_2026-11-12.json"


class SourceAuditBlocked(RuntimeError):
    pass


def git_bytes(root, *args):
    proc = subprocess.run(
        ["git", "-C", str(root), *args], stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, check=False,
    )
    if proc.returncode:
        raise SourceAuditBlocked(
            f"Git source unavailable: {' '.join(args)}: " +
            proc.stderr.decode("utf-8", errors="replace")[:400]
        )
    return proc.stdout


def git_text(root, *args):
    return git_bytes(root, *args).decode("ascii").strip()


def git_blob_sha(contents):
    return hashlib.sha1(
        b"blob " + str(len(contents)).encode("ascii") + b"\0" + contents
    ).hexdigest()


def audit(repo_path, record):
    repo_path = Path(repo_path).resolve()
    if not (repo_path / ".git").exists():
        raise SourceAuditBlocked("Requires explicitly supplied local Git checkout")
    if record.get("status") != "PUBLIC_TRACKED_TEXT_SCAN_ONLY":
        raise SourceAuditBlocked("Refuse to upgrade source scan to scientific approval")
    actual = git_text(repo_path, "rev-parse", "HEAD^{commit}")
    if actual != record["master_commit_sha"]:
        raise SourceAuditBlocked("Historical MATLAB source commit changed")
    tree = git_text(repo_path, "rev-parse", "HEAD^{tree}")
    if tree != record["master_git_tree_sha"]:
        raise SourceAuditBlocked("MATLAB Git tree identity mismatch")
    names = set(
        x.decode("utf-8") for x in
        git_bytes(repo_path, "ls-tree", "-r", "-z", "--name-only", "HEAD")
        .split(b"\0") if x
    )
    if len(names) != record["tracked_entries_total"]:
        raise SourceAuditBlocked("Git tracked-file count has changed")
    checked = record["checked_files"]
    if len(checked) != record["tracked_text_files_checked"]:
        raise SourceAuditBlocked("Declared tracked-text scope mismatch")
    terms = [re.compile(x, re.IGNORECASE) for x in record["tokens_case_insensitive"]]
    total_hits = 0
    for item in checked:
        path = item["path"]
        if path not in names:
            raise SourceAuditBlocked("Pinned tracked source path missing: " + path)
        source = git_bytes(repo_path, "show", "HEAD:" + path)
        if git_blob_sha(source) != item["blob_sha"]:
            raise SourceAuditBlocked("Pinned source Git blob mismatch: " + path)
        text = source.decode("utf-8", errors="replace")
        matches = sum(len(list(token.finditer(text))) for token in terms)
        if matches != item["text_search_hits"]:
            raise SourceAuditBlocked(
                f"Candidate-name literal hit count differs in {path}: {matches}"
            )
        total_hits += matches
    if total_hits != record["search_hits_total"]:
        raise SourceAuditBlocked("Total observed textual hits differ")
    if not (
        record["novelty_approved"] is False and
        record["nontracked_or_prior_commits_checked"] is False and
        record["external_papers_or_theses_exhaustively_checked"] is False and
        record["collaborator_or_local_archives_checked"] is False and
        record["P1_selected_dataset_ids"] == [] and
        record["P2_execution_preflight_sha256"] is None and
        record["execution_authorized"] is False and
        record["model_fits_executed"] == 0
    ):
        raise SourceAuditBlocked("Cannot present tracked-text scan as scientific approval")
    return {
        "schema": "pypcc-p1-original-matlab-tracked-text-check/v1",
        "commit_sha": actual, "git_tree_sha": tree,
        "tracked_entries": len(names), "checked_text_files": len(checked),
        "candidate_literal_hits_in_checked_sources": total_hits,
        "previous_commits_or_external_papers_checked": False,
        "novelty_approved": False,
        "execution_authorized": False,
        "P1_selected_dataset_ids": [],
        "model_fits_executed": 0,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matlab-repo", type=Path, required=True,
                        help="Explicit local clone pinned to original MATLAB commit")
    parser.add_argument("--record", type=Path, default=RECORD)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    record = json.loads(args.record.read_text(encoding="utf-8"))
    report = audit(args.matlab_repo, record)
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
