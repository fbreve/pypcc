# Fixed-performance archive and offline reproduction

The raw original campaign and separate follow-up are preserved locally. An
archive is prepared for review, **not published**. Packaging and the analysis
commands here only read existing measurements; they do not call PCC, download
datasets or invoke historical executor scripts.

## Proposed Release assets

After approval, use a research-specific GitHub Release such as
`python-fixed-benchmarks-2026-10-13`, separate from package-version releases.
Attach `pypcc-fixed-results-windows-20261013.zip`, `SHA256SUMS.txt`, and
`archive-index.json` (sizes, counts, source commits, verification result).
Create the tag/release only after its source revision and publication have been
approved. No Release or tag has been created by this preparation.

The ZIP retains distinct result roots, so no sensitivity data are pooled into
the primary campaign:

| Archive directory | Contents |
| --- | --- |
| `results/windows-fixed-full-c9f512c-01/` | 3,600 raw fits, 1,200 warm-ups, 200 fixture JSON/NPZ pairs, manifests, preparation/completion records, all timing fields/state hashes, source/environment and binary hashes, frozen fingerprint audit, execution/monitor logs, analysis JSON/CSV and historical scripts |
| `results/windows-fixed-review-c9f512c-01/` | Independent raw-derived review, per-dataset/native comparisons, order/repetition/mask diagnostics, diagnostic PDF/PNG and analysis scripts |
| `results/windows-permutation-followup-01/` | Three separate session directories, each with 144 fits, 48 warm-ups, eight fixtures and full manifests/completion records; timestamps, schedule/order seeds, monitoring, checksums, session/order tables and diagnostic PDF/PNG |
| `results/final-review-materials/` | Final reports and committed tables, portable offline verifier/analyzer, freshly recomputed accounting/tables, measured-source line-ending verification and their checksums |
| `source/<commit>.tar` | Exact committed source trees for primary `c9f512c2da576a4095e9bf0192133ee34896f494` and follow-up `a1a2460d80d40c5adcdc59e8622048157a5d8c60` |
| `checksums.json` | SHA-256 for every other ZIP member, including original checksum manifests |
| `README.txt` | Portable layout, integrity-check instructions and separation of experiments |

Original artifact manifests cover 423 primary files, eight detailed-review
files and 94 follow-up files. They remain byte-for-byte intact. Raw JSONL
includes every outlier, initialization/dispatch, propagation, postprocessing and
total-fit timing. Original scripts/logs may contain machine-specific absolute
paths; the new portable offline analyzer uses explicit paths and writes to a
new directory. Historical executor scripts are provenance, not verification steps.
The saved processed fixture arrays support offline inspection without OpenML
downloads; the original cached raw ARFF files and pilot/calibration directories
remain separately preserved locally and are not required to read these results.
The recorded Cython binary is identified by hash, not distributed as an
installable runtime artifact in this archive; original compiler provenance is
unavailable. No claim of binary-identical rebuilding is made.

## Verify and recompute, without executing benchmarks

Run from this repository checkout with Python (the archive tool/analyzer use
only the standard library). On the measured Windows machine:

```powershell
$condaExe = 'C:\Users\fbrev\miniforge3\Scripts\conda.exe'
$archive = '.cache/performance/final-archive-02/pypcc-fixed-results-windows-20261013.zip'
& $condaExe run --no-capture-output -n pypcc-dev python tools/archive_performance_results.py --verify $archive
Get-FileHash -Algorithm SHA256 -LiteralPath $archive

# Existing local raw directories; output must not already exist.
& $condaExe run --no-capture-output -n pypcc-dev python tools/analyze_fixed_results.py `
  --primary .cache/performance/windows-fixed-full-c9f512c-01 `
  --sensitivity .cache/performance/windows-permutation-followup-01 `
  --output .cache/performance/recomputed-for-review
```

After extracting a verified ZIP, pass the two corresponding `results/`
directories to the analyzer instead. Its new output contains separate primary
and session CSV/JSON tables; no original file is overwritten. Quartiles use
linear interpolation at `(n-1)*q`, matching the original NumPy summaries.
It checks exact unique record grids, fixture/frozen-fingerprint metadata,
three-label coverage, actual iteration counts, backend identity, finite times,
within-backend states/accuracy, recorded ordering and follow-up agreement with
the original trajectories. Byte integrity comes from archive/source checksums;
the earlier detailed review independently recomputed saved-array fingerprints.
Archive checksum verification detects changed bytes, not authenticity or
independent scientific replication.

All 32 recorded measured-source hashes were compared with their committed Git
blobs. Where raw hashes differ, applying the Windows checkout's LF-to-CRLF
conversion reproduces the recorded hash exactly; no source discrepancy remains
unexplained. The per-file evidence is in `source-line-ending-verification.json`.
This is source-file byte provenance, separate from the processed-data and
ordered-graph fingerprints, which matched directly without normalization.

To rebuild the package locally, first assemble `final-review-materials` with
its own `artifact_sha256.json`, then use a **new** ZIP path:

```powershell
& $condaExe run --no-capture-output -n pypcc-dev python tools/archive_performance_results.py `
  --output .cache/performance/rebuilt-archive/pypcc-fixed-results-windows-20261013.zip `
  --input .cache/performance/windows-fixed-full-c9f512c-01 `
  --input .cache/performance/windows-fixed-review-c9f512c-01 `
  --input .cache/performance/windows-permutation-followup-01 `
  --input .cache/performance/final-review-materials `
  --source-commit c9f512c2da576a4095e9bf0192133ee34896f494 `
  --source-commit a1a2460d80d40c5adcdc59e8622048157a5d8c60
```

The builder refuses existing output, unknown/unlisted source files, missing
files and checksum differences. It verifies every ZIP member after creation.
ZIP metadata timestamps can change between builds; use checksums of the actual
produced ZIP and its member inventory rather than assuming identical ZIP bytes.

## Historical execution commands (documentation only)

No new benchmark is needed or authorized to finalize the report. For future
approved replication, use the measured source tree/environment, cached sources
and fingerprint audit, not merely a current checkout with arbitrary dependencies.
Different library versions or a rebuilt native binary may change fingerprints
or runtime; stop and investigate discrepancies instead of overriding them.
The [harness guide](performance_benchmarking.md) describes installation,
Windows DLL activation, cache preparation, strict inventory audit, timing
boundaries and explicit download/execution flags.

The original command arguments, stored verbatim in its manifest, were:

```powershell
python -m benchmarks.run --historical --label-seeds 10 11 12 13 14 15 16 17 18 19 --backends numpy numba cython --repeats 6 --iterations 10000 --mode fixed --labels-per-class 3 --k-nn 10 --historical-cache .cache/stopping_patience/sources --allow-expensive --execute --output NEW-PRIMARY-DIRECTORY
```

For the follow-up, the session-specific deterministic dataset orders were
`digits satimage pageblocks wine`, `digits satimage wine pageblocks`, and
`wine digits satimage pageblocks`; `--order-seed` was respectively 1, 2, 3,
with matching `--session-id session-1`, `session-2`, `session-3` and separate
output directories. Each uses seeds 10/11, three backends, six repetitions,
10,000 fixed iterations and `--order-policy permutations`. Exact arguments
and timestamps are in each original manifest/execution record. Sessions were
consecutive without artificial delays, not distinct periods of machine use.
