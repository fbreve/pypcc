"""Read-only, source-only PyPCC three-backend research preflight.

NO dataset loads, NO model fits, NO JIT/native build and NO experiment runner.
This audits three *independent* experimental Git commits against the same
modern master base. Its SHA-256 identifies a SOURCE INVENTORY ONLY, never an
authorized C2/G1 execution preflight or a scientific protocol approval.

Usage:
  python -m experiments.preflight_three_backend_sources --output report.json
Requires the pinned commits to be available in the local Git object store.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess


SCHEMA = "pypcc-three-backend-source-inventory/v1"
BASE = "ea69f690208620d5563c8efece6b2e04369f52c0"
CYTHON_PRODUCTION_BLOB = "021c9a2c34f912e0d8382f05d0e21b9713ffd5fc"

PRODUCTION = (
    "pcc.py", "pcc_numpy.py", "pcc_numba.py", "pcc_step.pyx",
    "pcc_graph.py", "pcc_validation.py", "pcc_metrics.py",
    "setup.py", "pyproject.toml", "VERSION", "LICENSE",
)

BACKENDS = {
    "numpy": {
        "pr": 253,
        "branch": "research/current-numpy-live-milestones-2026-11-11",
        "head": "6d908e78878c4922674277e3313dbfe7f7924146",
        "files": (
            ".github/workflows/research-numpy-live-fullfit-cpu.yml",
            "docs/current_master_numpy_live_milestones_protocol_2026-11-11.md",
            "experiments/bench_live_numpy_fullfit.py",
            "experiments/live_numpy_milestones.py",
            "tests/test_live_numpy_milestones.py",
        ),
    },
    "numba": {
        "pr": 255,
        "branch": "research/current-numba-live-milestones-2026-11-12",
        "head": "4c8e6df4e01d7a4dfc4efc31eb73db04d6e212e5",
        "files": (
            ".github/workflows/research-numba-live-fullfit-cpu.yml",
            ".github/workflows/research-numba-live-milestones.yml",
            "docs/current_master_numba_live_milestones_protocol_2026-11-12.md",
            "experiments/bench_live_numba_fullfit.py",
            "experiments/live_numba_milestones.py",
            "tests/test_live_numba_milestones.py",
        ),
    },
    "cython": {
        "pr": 256,
        "branch": "research/current-cython-live-milestones-2026-11-12",
        "head": "af34f50ebabb0a7204d3f30679adaf3f00169742",
        "files": (
            ".github/workflows/research-cython-live-fullfit-cpu.yml",
            ".github/workflows/research-cython-live-milestones.yml",
            "experiments/bench_live_cython_fullfit.py",
            "experiments/build_live_cython.py",
            "experiments/live_cython_milestones.py",
            "tests/test_live_cython_milestones.py",
        ),
    },
}


class PreflightBlocked(RuntimeError):
    """A pinned input is missing, changed or inconsistent."""


def _git(root, *args):
    result = subprocess.run(
        ["git", "-C", str(root), *args],
        text=True, encoding="utf-8", errors="strict",
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False,
    )
    if result.returncode:
        raise PreflightBlocked(
            "Git provenance check failed: " + " ".join(args) + " (" +
            (result.stderr.strip() or f"exit {result.returncode}") + ")"
        )
    return result.stdout.strip()


def _verify_commit(root, sha):
    actual = _git(root, "rev-parse", "--verify", sha + "^{commit}")
    if actual != sha or _git(root, "cat-file", "-t", sha) != "commit":
        raise PreflightBlocked("Commit identity mismatch: " + sha)


def _blob(root, commit, path):
    sha = _git(root, "rev-parse", "--verify", commit + ":" + path)
    if len(sha) != 40 or _git(root, "cat-file", "-t", sha) != "blob":
        raise PreflightBlocked("Unexpected Git file identity: " + path)
    return sha


def _is_ancestor(root, base, head):
    completed = subprocess.run(
        ["git", "-C", str(root), "merge-base", "--is-ancestor", base, head],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False,
    )
    if completed.returncode:
        raise PreflightBlocked(
            "Research head does not descend from pinned production base: " + head
        )


def _verify_remote_head(root, branch, head):
    actual = _git(root, "rev-parse", "--verify", "refs/remotes/origin/" + branch)
    if actual != head:
        raise PreflightBlocked(
            "Remote tracking branch drift: " + branch + " expected " + head +
            " got " + actual
        )


def _sha_json(payload):
    canonical = json.dumps(
        payload, sort_keys=True, ensure_ascii=True, separators=(",", ":"),
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(canonical).hexdigest()


def audit(root, *, check_remote_heads=True):
    """Read only Git objects. Fail closed; no experimental computation."""
    root = Path(root).resolve()
    if _git(root, "rev-parse", "--is-inside-work-tree") != "true":
        raise PreflightBlocked("Not a Git working tree")
    _verify_commit(root, BASE)
    production_blobs = {path: _blob(root, BASE, path) for path in PRODUCTION}
    if production_blobs["pcc_step.pyx"] != CYTHON_PRODUCTION_BLOB:
        raise PreflightBlocked("Cython production blob differs from pinned hotloops")

    # Bind the P0 result to the version of THIS audit implementation, not
    # merely to the three independently pinned observer branches.
    audit_head = _git(root, "rev-parse", "HEAD")
    _verify_commit(root, audit_head)
    audit_files = (
        "experiments/preflight_three_backend_sources.py",
        "experiments/test_three_backend_source_preflight.py",
        "docs/current_three_backend_source_preflight_2026-11-12.md",
        ".github/workflows/research-three-backend-source-preflight.yml",
    )
    audit_blobs = {path: _blob(root, audit_head, path) for path in audit_files}
    local_script = (root / audit_files[0]).read_bytes().replace(b"\\r\\n", b"\\n")
    local_blob = hashlib.sha1(
        b"blob " + str(len(local_script)).encode("ascii") + b"\\0" + local_script
    ).hexdigest()
    if audit_blobs[audit_files[0]] != local_blob:
        raise PreflightBlocked("Running audit script differs from its committed Git blob")

    entries = {}
    for name, config in sorted(BACKENDS.items()):
        head = config["head"]
        _verify_commit(root, head)
        _is_ancestor(root, BASE, head)
        if check_remote_heads:
            _verify_remote_head(root, config["branch"], head)

        # Exact changed-file set; no production patches, no source cherry-picks,
        # and no unnoticed new runtime/packaging files.
        actual_files = tuple(sorted(filter(
            None, _git(root, "diff", "--name-only", BASE, head).splitlines()
        )))
        expected_files = tuple(sorted(config["files"]))
        if actual_files != expected_files:
            raise PreflightBlocked(
                f"{name} changed-file set differs: expected {expected_files}, "
                f"got {actual_files}"
            )
        for path, blob in production_blobs.items():
            if _blob(root, head, path) != blob:
                raise PreflightBlocked(f"{name} changes production file {path}")

        entries[name] = {
            "pr": config["pr"],
            "branch": config["branch"],
            "head_commit": head,
            "base_commit": BASE,
            "research_file_blobs": {
                path: _blob(root, head, path) for path in expected_files
            },
        }

    # This digest binds source refs and Git blobs only. It does NOT bind the
    # native binaries, libraries, OS, datasets, labels, graph or study design.
    inventory = {
        "schema": SCHEMA,
        "base_commit": BASE,
        "production_blobs": production_blobs,
        "audit_implementation": {
            "commit": audit_head, "file_blobs": audit_blobs,
        },
        "research_branches": entries,
    }
    return {
        "source_inventory": inventory,
        "source_inventory_sha256": _sha_json(inventory),
        "kind": "source-only, non-executable research inventory",
        "source_validation": "pass",
        "remote_tracking_verified": bool(check_remote_heads),
        "execution_authorized": False,
        "scientific_protocol_frozen": False,
        "real_dataset_inspected": False,
        "model_fits_executed": 0,
        "execution_preflight_sha256": None,
        "blockers": [
            "No agreed prospective scientific study, dataset IDs or reserved label/seed policy",
            "No verified real fixture hashes or symmetric graph/particle-order manifests",
            "No OS/Python/NumPy/Numba/SciPy/scikit-learn/Cython/compiler provenance",
            "No compiled Cython binary hash or Numba cache/dispatch signatures",
            "No independently audited memory/disk/export budget for proposed real fixtures",
            "No full three-backend actual-data source-exact prefix/continuation preflight",
            "No explicit maintainer approval of an execution-specific preflight SHA",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=None,
        help="Write read-only source inventory JSON (no fits, no datasets)",
    )
    parser.add_argument(
        "--allow-local-only", action="store_true",
        help="Skip remote branch-tip check; for offline Git-object audits only",
    )
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    report = audit(root, check_remote_heads=not args.allow_local_only)
    rendered = json.dumps(
        report, sort_keys=True, indent=2, ensure_ascii=True,
    ) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
