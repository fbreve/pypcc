# Early-stopping archive: local preparation v1

**Not published. PR #17's original branch has not been rewritten.**
Proposed research tag: `python-early-stopping-benchmarks-2026-10-15-v1`. Publication and branch replacement require
maintainer approval after review of this local candidate.

Asset: `pypcc-early-stopping-campaign-v1.zip`; **335,484,211 bytes**.
SHA-256: `3929218be0066793441bbc48cdb40b83b65a309b402335088f056a1dcd1d89bf`.
See [archive index](benchmarks/early_stopping_archive/archive-index.json),
[SHA256SUMS](benchmarks/early_stopping_archive/SHA256SUMS.txt) and
[local verification](benchmarks/early_stopping_archive/archive-verification.json).

The archive preserves all original pilot/calibration/full campaign roots,
seed-10/11/final checkpoint snapshots, synthetic recovery-validation roots,
source-cache files, original detailed report artifacts, recorded source and
Windows Cython binary, checksums, sessions, warm-ups, events, logs and summaries.
No original file or measurement was modified, deleted or repeated. Files under
`original/` retain their original relative paths and byte contents.

`baseline/` contains only the verbatim fixed-10,000 measurements and 200 fixture
metadata records needed for independent comparisons. Every subset byte hash was
checked against the existing [PR #16 Release](https://github.com/fbreve/pypcc/releases/tag/python-fixed-benchmarks-2026-10-13)
ZIP (SHA-256 `78d724d1fc8f908f0f1201033a24293edf4223514c203cec079eba90eaaac0bf`).
The full PR #16 Release is referenced by immutable version, source commit, URL,
size and checksum in `baseline/reference.json`; it is not duplicated or changed.

`checksums.json` indexes every other member, including original checksum files.
ZIP verification checks exact inventory, member hashes and safe extraction;
NPZ verification reconstructs the canonical array fingerprints without NumPy.
The portable analyzer uses only Python's standard library. It validates saved
identities, event chains, full task/permutation balance, stopping metrics, fixture
fingerprints and repeated state hashes; recomputes endpoint accuracy/timing and
matched fixed-baseline statistics; and checks the two calibrations separately.
It does not execute PCC, compile, reconstruct graphs, download data or assert
that the current computer matches the measured environment. Recorded invariant
checks are provenance, not a newly executed state/RNG replication.

After approval and publication the proposed retrieval URL will be:
`https://github.com/fbreve/pypcc/releases/download/python-early-stopping-benchmarks-2026-10-15-v1/pypcc-early-stopping-campaign-v1.zip`.
It does not exist as a published asset yet. Download it explicitly, compare its
SHA-256 against the repository's `SHA256SUMS.txt`, then run:

```console
python -I -S tools/verify_early_stopping_archive.py --archive PATH_TO_ZIP --extract NEW_DIRECTORY --output NEW_STATISTICS
python -I -S tools/verify_early_stopping_archive.py --root NEW_DIRECTORY --output ANOTHER_NEW_STATISTICS
```

All extraction/output directories must be new. Historical executor scripts are
preserved evidence and must not be used for verification. Checksums prove byte
integrity, not authenticity or independent scientific replication. Git may
normalize text newlines; byte inventories refer to original local/archive bytes,
and canonical JSON payload checksums remain distinct from file-byte checksums.

The original pre-cleanup Git bundle is **local only**, verified for recovery and
excluded from Release assets. Its inspection found no configured credential
patterns, but historical files contain user paths and email patterns; a pattern
scan cannot prove absence of private information. The archive itself retains
original machine/user paths in scientific provenance. Original compiler flags
are unavailable; the preserved binary is provenance, not an installable wheel.

Only the isolated candidate omits archived files. No PR-branch rewrite, force
push, remote tag, Release publication, original-file deletion or merge occurred.
