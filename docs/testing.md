# Automated test layout

Run the complete suite from the repository root:

```console
python -m unittest discover -s tests -v
```

Individual modules use package names, for example `python -m unittest -v tests.test_packaging_version`.

## Inventory before and after

All 14 root-level test modules move into `tests/`; all 91 test identities are preserved. No benchmarks or experiment drivers are moved.

| Before | After | Tests | Dependencies and imports |
| --- | --- | ---: | --- |
| `test_adaptive_stopping.py` | `tests/test_adaptive_stopping.py` | 11 | `experiments.adaptive_stopping`, `unittest` |
| `test_checkpoint_retry.py` | `tests/test_checkpoint_retry.py` | 2 | `copy`, `experiments.checkpoint_retry`, `experiments.stopping_patience`, `numpy`, `pathlib`, `tempfile`, `tests.test_propagation_metrics`, `tests.test_stopping_patience`, `unittest`, `unittest.mock` |
| `test_convergence_measurements.py` | `tests/test_convergence_measurements.py` | 4 | `experiments.convergence_measurements`, `numba`, `numpy`, `tests.test_propagation_metrics`, `tests.test_sequential_safety`, `unittest` |
| `test_numpy_regression.py` | `tests/test_numpy_regression.py` | 9 | `numpy`, `pcc`, `pcc_numba`, `pcc_numpy`, `pcc_step`, `unittest`, `unittest.mock` |
| `test_numpy_synchronous.py` | `tests/test_numpy_synchronous.py` | 20 | `numpy`, `pcc`, `pcc_numpy`, `unittest`, `unittest.mock` |
| `test_packaging_version.py` | `tests/test_packaging_version.py` | 3 | `email.parser`, `json`, `os`, `pathlib`, `subprocess`, `sys`, `tarfile`, `unittest`, `zipfile` |
| `test_patience_archive.py` | `tests/test_patience_archive.py` | 5 | `experiments.patience_archive`, `json`, `pathlib`, `tempfile`, `unittest`, `unittest.mock`, `zipfile` |
| `test_patience_results.py` | `tests/test_patience_results.py` | 2 | `experiments.patience_results`, `unittest` |
| `test_propagation_metrics.py` | `tests/test_propagation_metrics.py` | 2 | `copy`, `functools`, `numba`, `numpy`, `pcc_numba`, `pcc_numpy`, `tests.test_sequential_safety`, `unittest` |
| `test_sequential_safety.py` | `tests/test_sequential_safety.py` | 13 | `copy`, `numba`, `numpy`, `pcc`, `pcc_numba`, `pcc_numpy`, `pcc_step`, `unittest`, `unittest.mock` |
| `test_stopping_artifact_store.py` | `tests/test_stopping_artifact_store.py` | 4 | `experiments.stopping_artifact_store`, `hashlib`, `io`, `pathlib`, `tempfile`, `unittest` |
| `test_stopping_patience.py` | `tests/test_stopping_patience.py` | 10 | `copy`, `experiments.patience_data`, `experiments.patience_summary`, `experiments.stopping_patience`, `io`, `json`, `numpy`, `pathlib`, `pcc_numba`, `scipy.io`, `tempfile`, `tests.test_propagation_metrics`, `tests.test_sequential_safety`, `unittest` |
| `test_stopping_range_prototype.py` | `tests/test_stopping_range_prototype.py` | 3 | `copy`, `experiments.stopping_range`, `numpy`, `pcc_numba`, `tests.test_propagation_metrics`, `tests.test_sequential_safety`, `unittest` |
| `test_stopping_reproduction.py` | `tests/test_stopping_reproduction.py` | 3 | `docs.validate_stopping_reproduction`, `pathlib`, `tempfile`, `unittest` |

`tests/__init__.py` supports package imports and root-level unittest discovery. Shared fixtures remain in their original test modules. `test_packaging_version.py` resolves the repository root one directory higher; `PCC_DIST_DIR` remains relative to the working directory.

The diagnostic scripts `docs/diagnose_digits_stopping.py`, `docs/evaluate_stopping_candidates.py` and `docs/validate_stopping_range.py` import `next_random` from `tests.test_sequential_safety`. They remain optional diagnostics outside the test suite.

CI keeps the Windows/Linux and Python 3.10/3.12 matrix, isolated sdist construction, wheel compilation from the sdist, installation checks outside the checkout, complete regression discovery, editable installation and packaging metadata checks.

The command in `docs/benchmarks/stopping_patience/phase2_validation.json` is a frozen historical validation record. It intentionally retains the module names used at that time; current commands use the `tests.` prefix. Historical results, protocols and manifests are unchanged.

## Local validation

Baseline discovery: 91 tests (one missing-installed-metadata error and two archive skips in the existing development environment). Post-move discovery: the same 91 identities. An AST comparison confirms all modules are preserved except qualified test imports and the repository-root path adjustment.
