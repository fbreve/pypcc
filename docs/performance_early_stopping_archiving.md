# Published early-stopping archive and offline verification

The [research Release](https://github.com/fbreve/pypcc/releases/tag/python-early-stopping-benchmarks-2026-10-15-v1) was published on 2026-10-15
and independently downloaded and verified. Tag `python-early-stopping-benchmarks-2026-10-15-v1`
points exactly to clean candidate commit `35b6da469c7d72d0a791ce800ae3842f32d0a4ad`.
PR #17's branch has been replaced with that clean candidate; the PR remains a
draft and has not been merged. This documentation update appends to the clean
branch without changing the Release tag, assets or scientific measurements.

Asset: `pypcc-early-stopping-campaign-v1.zip`; **335,484,211 bytes**.
SHA-256: `3929218be0066793441bbc48cdb40b83b65a309b402335088f056a1dcd1d89bf`.
See [current repository status index](benchmarks/early_stopping_archive/archive-index.json),
[SHA256SUMS](benchmarks/early_stopping_archive/SHA256SUMS.txt) and
[original local verification](benchmarks/early_stopping_archive/archive-verification.json).
The [independent publication-verification record](benchmarks/early_stopping_archive/publication-verification.json)
records downloaded asset hashes, GitHub digests and exact offline reproduction.

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

Published assets:

- [Campaign ZIP](https://github.com/fbreve/pypcc/releases/download/python-early-stopping-benchmarks-2026-10-15-v1/pypcc-early-stopping-campaign-v1.zip)
- [SHA256SUMS.txt](https://github.com/fbreve/pypcc/releases/download/python-early-stopping-benchmarks-2026-10-15-v1/SHA256SUMS.txt)
- [Frozen published archive-index.json](https://github.com/fbreve/pypcc/releases/download/python-early-stopping-benchmarks-2026-10-15-v1/archive-index.json)

The frozen published index is **1,705 bytes**, SHA-256
`126bff44a6e9128fdf16920d3ce66927cac436608debe74dbfa6a8638323bdfb`.
Its `published: false` and proposed-tag fields describe archive preparation.
They are immutable historical information, not current availability. The
repository's status index now records publication and branch replacement,
preserves the full preparation object under `historical_preparation`, and pins
the frozen asset's URL, byte size and checksum separately. Its own file hash
therefore differs from the published index; neither published asset was changed.
The cleanup manifest similarly preserves its original preparation-time flags.

All three assets were independently downloaded without authentication and
matched both the frozen local publication manifest and GitHub's asset digests.
All **96,437 ZIP members**, file checksums and safe extraction passed. The
downloaded standard-library verifier ran under `python -I -S` and exactly
reproduced the frozen local statistics: **3,600 fits, 1,200 warm-ups, 200 fixtures,
568,303,200 iterations, 3,120 early stops and 480 censored outcomes**. The separate
72-fit pilot and 48-fit calibration were also reproduced; their results remain
separate from the full campaign.

The final whole-metadata guard stopped because each asset's `download_count`
increased by one. Read-only inspection confirmed that only these counters changed;
asset identities, sizes and digests were unchanged. No integrity/statistical
check failed, and no asset replacement, re-upload or automatic retry occurred.

Download the ZIP explicitly, compare its SHA-256 against the unchanged
`SHA256SUMS.txt`, then run:

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

The clean PR branch retains source, tests, documentation, compact summaries,
essential flat measurements, figures and archive manifests. Its omitted generated
artifacts are preserved byte-for-byte in the published archive. Historical stage
reports retain their original scientific results and dated experimental status;
use this guide and the current status metadata for publication/cleanup status.
Original local campaign roots and the existing PR #16 Release remain unchanged.
The local Git backup bundle was not uploaded. No merge or readiness change is
included in this documentation cleanup.
