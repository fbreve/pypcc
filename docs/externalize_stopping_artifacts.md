# Proposed external artifact storage (approval pending)

Base: master `26f74788b55785eb38bb6d7d5b0d3ee5d8fa0920`.
The four ZIPs remain tracked until publication and external reproduction are approved
and verified. This migration changes storage only, not scientific results or protocols.

## Release proposal

Create a dedicated GitHub Release with tag `stopping-trajectories-v1`, targeting
the maintenance commit approved for publication. Attach the four original ZIPs,
`artifacts_manifest.json`, `archive_manifest.csv`, and `SHA256SUMS`.
Copy the existing ZIP bytes; do not recompress them. Total ZIP size: 121,491,163 bytes.
The committed artifact manifest remains the trust anchor, with SHA-256
`95a2028effe6a48089bf45489c3208d8eae27e01eaf2edab32e5208c1db0378c`.
It authenticates all 800 members and their original paths, as well as each container.
The check-log manifest additionally authenticates decompressed logs.

No Release has been published by this preparation. The proposed download endpoint
is fixed in `experiments/stopping_artifact_store.py`; it becomes usable only after
approval and publication. Do not replace assets in place after publication.

## References and compatibility

| Reference | Treatment |
|---|---|
| `experiments/patience_archive.py` | Shared reader accepts an explicit directory or environment variable; checks archives and members. |
| `docs/reproduce_stopping_patience.py`, `docs/report_stopping_patience.py` | Public reproduction inherits the shared reader; no pickle or propagation required. |
| `docs/inspect_adaptive_stopping_archives.py` | Published adaptive schema inspection inherits the same reader. |
| `docs/compare_adaptive_stopping.py` on the adaptive experimental branch | Its existing shared-reader call also supports external storage without changing its frozen protocol. This comparison is not yet in the base master. |
| `test_patience_archive.py` | Synthetic fixtures continue to work; external-directory regression added. |
| `docs/package_stopping_patience.py` | Historical packager remains unchanged; not needed for migration. Do not run it over published files. |
| `artifacts_manifest.json`, `archive_manifest.csv`, frozen protocols | Preserve byte-for-byte, including paths, hashes and provenance. |
| Historical reports, consolidation validation and adaptive inspection metadata | Preserve scientific evidence; file names identify archive contents, not a required storage directory. |

## Offline and explicit online use

From the repository root, copy the four ZIPs into ignored `.cache/stopping_artifacts`.
Then verify them without network access:

```powershell
conda run -n pypcc-dev python docs/fetch_stopping_artifacts.py
conda run -n pypcc-dev python docs/fetch_stopping_artifacts.py --stage-release .cache/stopping_release_stage
conda run -n pypcc-dev python docs/reproduce_stopping_patience.py --output .cache/stopping_patience_reproduced
conda run -n pypcc-dev python docs/inspect_adaptive_stopping_archives.py
```

For another directory, set `$env:PYPCC_STOPPING_ARTIFACTS` to its absolute path
and pass the same path as `--artifact-dir` to the fetch/verify command.
An explicitly selected missing or corrupt directory fails; it does not silently
fall back to another copy. Without configuration, a complete default cache is
preferred; the original tracked location is a transitional fallback.

After publication, opt in to downloads with
`python docs/fetch_stopping_artifacts.py --download`.
Existing valid files never trigger network access. Downloads use temporary files,
are SHA-256/length checked before atomic installation, and leave no partial artifact
after failure. A corrupt existing file is rejected rather than overwritten.
The reader itself never downloads anything.

## Validation and removal gate

1. Prepare ignored staging copies and checksum lists; verify archive and all member hashes.
2. Reproduce patience tables, figures and report from the cache, without local checkpoints.
3. Reproduce adaptive inspection and, using the separately registered adaptive branch,
   its retrospective comparison; compare published scientific CSVs without changing them.
4. Run regression tests locally and the Python 3.10/3.12 CI.
5. Present this preparation for maintainer approval **before publishing or removing files**.
6. Once approved, publish the assets, download all four into a fresh directory,
   verify their committed hashes and both manifests, and repeat both public reproductions.
7. Only after successful external validation, remove the four ZIPs from the current
   Git tree in a normal commit. Preserve local copies and manifests. No history rewrite.

Deep particle-state/RNG auditing still requires trusted local checkpoints; it is
optional and distinct from public statistical reproduction. No trajectories are rerun.
Deleting ZIPs from the current tree will not shrink old Git history. Release assets
require ongoing hosting; offline copies and hash-pinned manifests remain usable if
the hosting endpoint is unavailable. PNG bytes depend on plotting versions/fonts.
