# Published artifacts and reproduction

The ZIPs are hosted in the [dedicated Release](https://github.com/fbreve/pypcc/releases/tag/stopping-trajectories-v1).
Download and offline-directory instructions are documented in
[the storage guide](../../externalize_stopping_artifacts.md). The manifests remain
versioned; the ZIPs are no longer in the current Git tree.

The four `trajectories-01.zip` through `trajectories-04.zip` archives retain all
800 original committed files from the 200 full trajectories. Alphabetically
sorted datasets are grouped into four groups of five, independently of outcomes.
Each archive preserves the original `trajectories/<dataset>/<seed>/<filename>`
paths. Existing gzip bytes are stored unchanged; JSON/CSV text is deflated.

`artifacts_manifest.json` records every original path, byte length, SHA-256,
archive assignment, archive SHA-256, and source commit. The packager reads Git
blobs, not checkout bytes: Windows newline conversion cannot change published
content. ZIP timestamps, permissions, member ordering and compression settings
are fixed. Rebuilding with the recorded Python/zlib environment reproduces the
ZIP hashes; other compression-library versions may produce different container
bytes while preserving all member hashes. The original `archive_manifest.csv`
continues to authenticate both gzip bytes and uncompressed check-log bytes.

## Public reproduction: no kernels or checkpoints

From the repository root, with NumPy and Matplotlib installed:

```powershell
conda run -n pypcc-dev python docs/fetch_stopping_artifacts.py --download
conda run -n pypcc-dev python docs/reproduce_stopping_patience.py `
  --source docs/benchmarks/stopping_patience `
  --output .cache/stopping_patience_reproduced
```

The first command is required only when the ZIPs are absent locally; it verifies
downloaded SHA-256 hashes and sizes. Omit it for fully offline reproduction.

`python docs/report_stopping_patience.py` is an equivalent entry point.
Output must be separate from the published input. No propagation is performed;
no Numba/Cython backend, dataset download, graph cache or pickle is required.
The reader authenticates archives and every member, rejects duplicate/unsafe
paths, and independently audits all ten million original check records. It
reconstructs individual results, timings, intermediate observations, record
diagnostics, dataset/stratum/overall statistics, all four full-study figures and
the full report. Reference metadata and prior state/RNG evidence are copied as
historical provenance, never presented as a newly executed deep audit.

Numerical tables should match the published values. PNG byte identity is only
expected with the same plotting environment; fonts and Matplotlib versions can
change rendering without changing the underlying data. Original phase-1 pilot
artifacts remain published unchanged and are not rerun by this command.

## Optional deep audit

Trusted local `trajectories/<dataset>/<seed>/recovery.pkl` checkpoints and raw
`checks.csv` remain available on the original Windows machine. They are neither
discarded nor published. To repeat final-state/RNG invariant comparisons without
rerunning propagation:

```powershell
conda run -n pypcc-dev python docs/reproduce_stopping_patience.py --deep-audit
```

This optional legacy procedure uses the original runtime and writes local audit
outputs; run it in a copy if preserving published files byte-for-byte. It requires
Numba and the version-locked trusted checkpoint format. Public ZIPs cannot repeat
the particle-array/RNG comparison because those states were intentionally not
published. `full_verification.csv`, `final_invariants.csv`, and per-case recorded
verification JSON retain the original audit evidence, not independent proof of
the underlying states. No pickle is loaded by public reproduction.

## Packaging and preservation

`python docs/package_stopping_patience.py` can rebuild packages from the manifest's
original Git commit (full history required). It validates all members against
that commit and never deletes files. The four ZIPs and manifest replace individual
files in the current Git tree; local individual files remain ignored. Earlier
commits retain the original blobs: this consolidation reduces the final PR file
count, not historical repository size. The ZIPs were subsequently externalized
after Release download, hash verification and both public study reproductions.
The historical packaging procedure does not publish or remove artifacts.
