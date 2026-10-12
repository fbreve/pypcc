# Adaptive stopping: Release-backed reproduction

The PR #9 branch integrates master storage maintenance from PR #10. No ZIPs
are tracked. The original frozen protocol, registration, parameter grids and
scientific results remain unchanged. Historical validation metadata retains
the original script/source hashes; a new reproduction has different provenance.

Download once, with hash verification:

```powershell
conda run -n pypcc-dev python docs/fetch_stopping_artifacts.py --download
```

Alternatively `docs/compare_adaptive_stopping.py --download` explicitly uses the
same downloader. Without `--download`, the comparison never accesses the network.
Select a different local directory with `--artifact-dir` or
`PYPCC_STOPPING_ARTIFACTS`. Explicit directories do not fall back to other copies.

The scripts write into `docs/benchmarks/adaptive_stopping`, so reproduce in an
isolated checkout/copy to preserve registered outputs. Copy the scripts,
`experiments/` and published metadata (without trajectory ZIPs or checkpoints)
into a working directory under ignored `.cache`. The parent Git repository
provides the frozen protocol commit required by the comparison. From that copy:

```powershell
conda run -n pypcc-dev python docs/compare_adaptive_stopping.py --artifact-dir <absolute-verified-release-directory>
conda run -n pypcc-dev python docs/summarize_adaptive_stopping.py
```

From the original repository, compare the copied outputs against the historical
registered study with the master validation helper:

```powershell
python docs/validate_stopping_reproduction.py --output <copied-output-directory> --reference docs/benchmarks/adaptive_stopping --ref ec9205fc06f98381ecb33150873c059041f82049 --required individual.csv dataset_summary.csv overall_summary.csv primary_secondary_pairs.csv report.md primary_comparison.png
```

This replays public archived signals only. It performs no propagation, uses no
`recovery.pkl`, and does not retune parameters or estimate missing accuracies.
CSV cells and normalized report text must match; PNG byte comparisons require
the original plotting environment. The previous experiment remains exploratory
reuse of previously examined cases, rather than independent validation.
