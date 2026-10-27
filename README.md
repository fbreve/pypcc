# Particle Competition and Cooperation
Python code for the semi-supervised learning method "Particle Competition and Cooperation". 

This project is a fork of Caio Carneloz’s original implementation, available at
https://github.com/caiocarneloz/pycc

This fork has a few bug fixes and several optimizations for faster execution. 
It also includes the early stopping criterion from the original article.

For Python, there are three implementations available: **cython**, **numba**, and pure **numpy**.  \
**cython** and **numba** are much faster than **numpy**.

In practice, performance depends on the dataset size and structure, so it is
recommended to benchmark both cython and numba on your data.

Use:
```python
ParticleCompetitionAndCooperation(impl="auto")
ParticleCompetitionAndCooperation(impl="cython")
ParticleCompetitionAndCooperation(impl="numba")
ParticleCompetitionAndCooperation(impl="numpy")
```
according to the implementation you would like to use. 
**auto** selects the first available backend in the fixed order Cython, Numba, NumPy; it does not benchmark them.

## Installation and compatibility

Regression CI tests Python **3.10 and 3.12** on Windows/MSVC and Linux/GCC.
Windows development also uses Python 3.12 (`pypcc-dev`). Other Python versions,
macOS and other compiler configurations are not validated by this matrix.
Package metadata requires Python >=3.10; this minimum is not a claim that every
newer version has been tested.

Runtime requirements are **NumPy 2.x, SciPy and scikit-learn**. Graph utilities
import SciPy and scikit-learn even when a graph is supplied manually.
**Numba is optional**, for the Numba backend. Cython and setuptools are build
dependencies, not requirements for running an already compiled extension.
Matplotlib is optional for plots and research reports; it is not needed by the
minimal example. The repository's distribution name remains `pcc_step`; import
the library with `from pcc import ParticleCompetitionAndCooperation`.

Clone the repository and use an isolated environment:

```sh
git clone https://github.com/fbreve/pypcc.git
cd pypcc
python -m venv .venv
```

Activate with `.venv\Scripts\Activate.ps1` in Windows PowerShell, or
`source .venv/bin/activate` on Linux. Install the library and optional Numba backend:

```sh
python -m pip install --upgrade pip
python -m pip install -e ".[numba]"
```

The standard installation builds the Cython extension. Windows needs MSVC Build
Tools with the C++ workload and Windows SDK; Linux needs a C compiler and matching
Python development headers (for example, GCC and python3-dev on Debian/Ubuntu).
Setuptools locates installed MSVC tools automatically. Compiler selection uses
`/O2` for MSVC and `-O3` for GCC/Clang, without fast-math flags.

The Python distribution keeps the name `pcc_step`. Its version is maintained only
in the root `VERSION` file, which `setup.py` reads and includes in source archives.
For an approved version update, edit that file; wheel, source distribution and
installed metadata inherit the value automatically. Read the installed version
with `importlib.metadata.version("pcc_step")`.

`pyproject.toml` declares setuptools, wheel, Cython and NumPy headers for isolated
builds; preinstalling Cython into the runtime environment is unnecessary. Pip
resolves Numba's compatible NumPy requirement when using the optional extra.
Do not independently upgrade NumPy beyond the installed Numba's supported range.

To work from the checkout without a compiler, install just runtime dependencies
and import from the repository root:

```sh
python -m pip install "numpy>=2,<3" scipy scikit-learn
# Optional:
python -m pip install numba
```

This checkout-only route does not install a wheel or build Cython. Explicit
NumPy selection always uses NumPy. Sequential backend selection retains existing
fallbacks: Cython -> Numba -> NumPy; Numba -> Cython -> NumPy. Synchronous mode
requires explicit NumPy and has no fallback.

## Minimal usage

Labels use `-1` for unlabeled nodes. At least two classes must have labeled seeds.
Build the graph before calling the transductive `fit_predict` method:

```python
import numpy as np
from pcc import ParticleCompetitionAndCooperation

X = np.array([[0.0], [0.1], [0.2], [1.0], [1.1], [1.2]])
labels = np.array([0, -1, -1, 1, -1, -1])
model = ParticleCompetitionAndCooperation(impl="numpy")
model.build_graph(X, k_nn=2)
predictions = model.fit_predict(labels, max_iter=100, early_stop=False, seed=42)
print(predictions)
```

This small example limits iterations explicitly; library defaults are unchanged.
`build_graph` receives `k_nn`; propagation parameters belong to `fit_predict`,
while `impl`, `n_jobs` and `update_mode` belong to the constructor.
See [example.py](example.py) for a Wine example with normalization and accuracy.

## Parameters

The main API separates constructor, graph construction and fit options:

| Method | Parameter | Purpose / default |
| --- | --- | --- |
| Constructor | `impl` | Backend: `auto` (default), `cython`, `numba`, `numpy` |
| Constructor | `n_jobs` | Parallelism setting; see backend-specific behavior |
| Constructor | `update_mode` | `sequential` (default); experimental `synchronous` requires NumPy |
| `build_graph` | `k_nn` | Number of nearest neighbors for graph construction |
| `fit_predict` | `p_grd` | Greedy movement probability; default `0.5` |
| `fit_predict` | `delta_v` | Domination-level update rate; default `0.1` |
| `fit_predict` | `deltap` | Particle-strength update rate; default `1.0` |
| `fit_predict` | `dexp` | Exponent in greedy movement weighting; default `2.0` |
| `fit_predict` | `max_iter` | Maximum iterations; default `500000` |
| `fit_predict` | `early_stop` | Enable convergence stopping; default `True` |
| `fit_predict` | `es_chk` | Early-stopping patience control; default `2000` |
| `fit_predict` | `seed` | Optional integer RNG seed; default `None` |

The stopping criterion checks every 10 iterations; reducing `es_chk` may
finish earlier but can change accuracy. See the implementation and
[advanced notes](docs/readme_advanced.md) for additional constraints.

## Reproducibility

Use the optional `seed` keyword in `fit_predict` to reproduce a run with
the **same backend**:

```python
model = ParticleCompetitionAndCooperation(impl="numba")
predictions = model.fit_predict(labels, seed=42)
```

`seed` accepts an integer from 0 through `2**32 - 1` or `None` (the
default). With `seed=None`, each backend retains its previous behavior.
A provided seed is applied before propagation, after input validation.
Select a backend explicitly for repeatable experiments: `impl="auto"` picks
the first installed backend in the order Cython, Numba, NumPy.

- **NumPy:** seeds the legacy global `np.random` generator. This also changes
  the process-global random state seen by other NumPy users.
- **Numba:** seeds the Numba-managed generator from compiled code; ordinary
  Python-level `np.random.seed` does not seed this generator.
- **Cython:** seeds its private xorshift32 generator. With `seed=None`, it
  retains the historical fixed internal seed `123456789`. Since xorshift32
  cannot evolve from zero, `seed=0` maps to that nonzero default.

A fixed seed does **not** imply identical results across backends: RNG
algorithms and draw schedules differ. The software environment, graph, labels,
fit parameters and update mode also matter. The tests
`tests/test_optional_seed.py`, `tests/test_numpy_rng_replay.py`,
`tests/test_numba_rng_replay.py` and `tests/test_cython_rng_replay.py`
check reproducibility within each backend.

If `fit_predict` raises, prior fitted model attributes are restored but
global or backend-specific RNG state is **not** rolled back. A failed run may
consume random draws and affect subsequent unseeded runs.

## Performance

The repository includes both historical MATLAB/Python comparisons and a
20-dataset Python fixed-iteration study (2026-10-13). On that study's
Windows/i7-14700K environment, Numba and Cython substantially outperform pure
NumPy, but neither compiled backend wins on every dataset. Timing boundaries
and experiment protocols differ from the historical MATLAB/MEX comparisons;
results should not be compared as if they used an identical protocol.

See [full benchmark tables and historical measurements](docs/readme_advanced.md),
[fixed-iteration protocol and paired speedups](docs/performance_fixed_results.md),
[order sensitivity](docs/performance_order_sensitivity.md), and
[archived raw results](https://github.com/fbreve/pypcc/releases/tag/python-fixed-benchmarks-2026-10-13).

## MATLAB version

You may also check the original MATLAB version:
https://github.com/fbreve/Particle-Competition-and-Cooperation

For MATLAB, there is a MEX version available, which is comparable in speed to
the Cython implementation on the same machine.

## Advanced behavior

For graph replacement and fit-state invalidation, direct kernel buffer ownership,
and experimental synchronous NumPy updates, see
[advanced behavior and benchmark records](docs/readme_advanced.md).
The synchronous update rule is also described in
[its dedicated document](docs/numpy_synchronous.md).

## Development checks

After installation, run `python -m unittest discover -s tests -v` from the repository root.
See [test inventory and module commands](docs/testing.md).
CI builds and installs the project through pip with build isolation, checks all
three backends outside the source tree, and runs regression tests on both tested
Python versions and operating systems. Optional long experiments under `docs/`
are not part of installation or CI. Archived stopping data are downloaded
separately; see [reproduction instructions](docs/benchmarks/stopping_patience/REPRODUCIBILITY.md).

## Citation
If you use PyPCC in your research, please cite the original PCC paper:

`Breve, Fabricio Aparecido; Zhao, Liang; Quiles, Marcos Gonçalves; Pedrycz, Witold; Liu, Jiming, "Particle Competition and Cooperation in Networks for Semi-Supervised Learning," Knowledge and Data Engineering, IEEE Transactions on , vol.24, no.9, pp.1686,1698, Sept. 2012`

Accepted Manuscript: https://www.fabriciobreve.com/artigos/ieee-tkde-2009.pdf

```bibtex
@article{Breve2012PCC,
  author = {Breve, Fabricio Aparecido and Zhao, Liang and Quiles, Marcos Gon{\c{c}}alves and Pedrycz, Witold and Liu, Jiming},
  title = {Particle Competition and Cooperation in Networks for Semi-Supervised Learning},
  journal = {IEEE Transactions on Knowledge and Data Engineering},
  volume = {24},
  number = {9},
  pages = {1686--1698},
  year = {2012},
  month = sep,
  doi = {10.1109/TKDE.2011.119},
  url = {https://doi.org/10.1109/TKDE.2011.119}
}
```

[DOI: 10.1109/TKDE.2011.119](https://doi.org/10.1109/TKDE.2011.119)
