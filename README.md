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

## Graph replacement and fit state

A successful `build_graph` or `set_graph` call invalidates the previous fit,
even when reinstalling an identical graph or changing only its connectivity.
Fit-derived fields such as `node`, `part`, labels and mappings become `None`;
lazily created fit arrays and scratch buffers are removed. Call `fit_predict`
again to obtain results for the installed graph. Backend configuration and the
new graph remain available. A preparation failure preserves the previous graph
and fit unchanged. Previously returned predictions or other objects retained by
the caller are not modified; they still describe the earlier fit.

## Direct kernel buffer ownership

Direct calls to NumPy, Numba, or Cython propagation functions require
independent, non-overlapping state and scratch buffers. The low-level input
validators do not check memory aliasing; overlapping arrays can silently alter
results. See [direct kernel buffer preconditions](docs/direct_kernel_buffers.md).

## MATLAB version

You may also check the original MATLAB version:
https://github.com/fbreve/Particle-Competition-and-Cooperation

For MATLAB, there is a MEX version available, which is comparable in speed to
the Cython implementation on the same machine.

## Experimental synchronous NumPy updates

Sequential PCC remains the default for every backend. Existing calls retain
their behavior. To opt into the experimental vectorized synchronous variant:

```python
ParticleCompetitionAndCooperation(impl="numpy", update_mode="synchronous")
```

`update_mode="sequential"` selects the existing algorithm explicitly.
Synchronous mode requires `impl="numpy"`; `auto`, Numba and Cython are rejected
with `ValueError`, without backend fallback. All destinations are selected
from the initial iteration state, dominance transfers are aggregated
simultaneously, then every particle evaluates strength and movement against
the same final dominance. This variant may produce different classifications
and is not always faster. It does not use multiple threads.

See [the mathematical rule, tests and Wine/Digits benchmarks](docs/numpy_synchronous.md).

## Python fixed-iteration performance (2026-10-13)

Measured on native Windows 11, Intel i7-14700K, Python 3.12.15; NumPy 2.4.6,
Numba 0.68.0 and Cython 3.2.4. The frozen 20-dataset inventory uses three
labels/class, seeds 10–19, 10,000 fixed iterations and six balanced repetitions
per backend: 3,600 measured fits, with warm-ups recorded separately.

Propagation median (IQR) in **milliseconds**, across 60 records per cell:

| Dataset | NumPy ms | Numba ms | Cython ms |
| --- | ---: | ---: | ---: |
| Wine | 922.591 (8.953) | 5.293 (0.131) | 3.720 (0.073) |
| Digits | 4046.644 (1441.769) | 17.195 (1.384) | 16.869 (3.011) |
| Iris | 1283.383 (207.578) | 5.439 (2.125) | 3.644 (1.603) |
| BreastCancer | 827.364 (170.732) | 4.073 (1.607) | 2.590 (1.065) |
| BalanceScale | 1305.976 (305.553) | 5.353 (2.333) | 3.929 (1.753) |
| GermanCredit | 888.568 (220.313) | 4.124 (0.887) | 2.872 (1.351) |
| Diabetes | 826.162 (186.735) | 4.025 (1.687) | 2.591 (1.402) |
| Sonar | 893.194 (275.428) | 4.072 (1.253) | 2.629 (1.182) |
| Glass | 2607.107 (440.126) | 11.302 (4.627) | 10.973 (4.781) |
| Vehicle | 1717.579 (409.429) | 6.981 (2.758) | 5.633 (2.042) |
| Ionosphere | 880.952 (166.724) | 4.719 (1.861) | 2.874 (1.460) |
| PenDigits | 4527.435 (597.263) | 22.748 (10.719) | 20.741 (7.325) |
| Satimage | 2869.997 (484.835) | 14.539 (7.041) | 15.924 (7.267) |
| Spambase | 851.557 (299.654) | 4.675 (1.461) | 3.445 (2.071) |
| PageBlocks | 2083.968 (408.384) | 9.680 (4.238) | 8.300 (4.461) |
| Banknote | 880.563 (325.210) | 3.839 (1.106) | 2.441 (1.294) |
| BloodTransfusion | 800.653 (175.743) | 3.882 (1.462) | 2.393 (0.821) |
| Yeast | 4121.912 (696.284) | 18.807 (7.214) | 17.325 (6.441) |
| Segment | 2837.218 (488.372) | 12.292 (4.242) | 11.492 (6.130) |
| QSARBiodeg | 792.580 (293.384) | 3.979 (0.334) | 2.609 (0.200) |

These IQRs combine label-mask and timing variation; repetitions are not
independent statistical samples. Preparation and initialization are outside
propagation timing. Numba/Cython differences depend on workload and timing
variability; no universal winner is established. This fixed-iteration experiment
does not measure time to early stopping and has different timing boundaries
from the historical MATLAB/MEX benchmarks below.

See the [complete protocol, paired speedups and limitations](docs/performance_fixed_results.md),
[separate 432-fit order sensitivity analysis](docs/performance_order_sensitivity.md),
and [raw archive/reproduction guide](docs/performance_archiving.md).
The complete raw-results archive and checksum manifests are available in the
[benchmark Release](https://github.com/fbreve/pypcc/releases/tag/python-fixed-benchmarks-2026-10-13).

## Benchmarks:

These are **historical measurements**, not measurements of the latest code.
The MATLAB and Python timings used different preparation and timing protocols.
See [reproducible Python benchmarking](docs/performance_benchmarking.md) for the
new protocol, workload planning, and the historical comparability limitations.

Machine: Intel Core i9 14900K with 128GB of RAM \
Software: MATLAB R2025b, Python 3.12.12, Numpy 2.4.4, Numba 0.64.0, and Cython 3.2.4 \

#### Dataset: Wine (UCI, 178 instances, 13 features, 3 classes)

With early_stop=True (default), max_iter=500000 (default), 100 repetitions:

| Implementation      | Time (s) |
|---------------------|---------:|
| MATLAB MEX          |  0.014   |
| MATLAB pure         |  0.331   |
| Python Cython       |  0.004   |
| Python Numba        |  0.006   |
| Python NumPy (only) |  0.426   |

With early_stop=False and max_iter=500000 (500,000 fixed iterations), 10 repetitions:

| Implementation      | Time (s) |
|---------------------|---------:|
| MATLAB MEX          |   0.972  |
| MATLAB pure         |  30.697  |
| Python Cython       |   0.341  |
| Python Numba        |   0.465  |
| Python NumPy (only) |  33.110  |

#### Dataset: Digits (sklearn, 1797 instances, 64 features, 10 classes)

With early_stop=True (default), max_iter=500000 (default), 100 repetitions:

| Implementation      | Time (s) |
|---------------------|---------:|
| MATLAB MEX          |   0.357  |
| MATLAB pure         |   8.738  |
| Python Cython       |   0.181  |
| Python Numba        |   0.170  |
| Python NumPy (only) |   1.921  |

With early_stop=False and max_iter=500000 (500,000 fixed iterations), 10 repetitions:

| Implementation      | Time (s) |
|---------------------|---------:|
| MATLAB MEX          |  10.866  |
| MATLAB pure         | 284.953  |
| Python Cython       |   5.054  |
| Python Numba        |   5.009  |
| Python NumPy (only) |  63.906  |

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

np.random.seed(0)
X = np.array([[0.0], [0.1], [0.2], [1.0], [1.1], [1.2]])
labels = np.array([0, -1, -1, 1, -1, -1])
model = ParticleCompetitionAndCooperation(impl="numpy")
model.build_graph(X, k_nn=2)
predictions = model.fit_predict(labels, max_iter=100, early_stop=False)
print(predictions)
```

This small example limits iterations explicitly; library defaults are unchanged.
`build_graph` receives `k_nn`; propagation parameters belong to `fit_predict`,
while `impl`, `n_jobs` and `update_mode` belong to the constructor.
See [example.py](example.py) for a Wine example with normalization and accuracy.

## Development checks

After installation, run `python -m unittest discover -s tests -v` from the repository root.
See [test inventory and module commands](docs/testing.md).
CI builds and installs the project through pip with build isolation, checks all
three backends outside the source tree, and runs regression tests on both tested
Python versions and operating systems. Optional long experiments under `docs/`
are not part of installation or CI. Archived stopping data are downloaded
separately; see [reproduction instructions](docs/benchmarks/stopping_patience/REPRODUCIBILITY.md).

## Parameters
The existing constructor, graph and propagation methods receive these parameters:

---
- **k_nn:** value that represents the number of k-nearest neighbours used to build the graph (Euclidean distance).
- **p_grd:** value from 0 to 1 that defines the probability of particles to take the greedy movement. Default: 0.5.
- **delta_v:** value from 0 to 1 to control the rate of change of the domination levels. Default: 0.1.
- **max_iter:** number of iterations until the label propagation stops (if the stop criteria is not met before that).
- **es_chk:** control how many iterations the algorithm performs after reaching some level of stability (stopping criterion). Default: 2000. The implementation checks every 10 iterations and uses `max(1, round((nodes / (particles * k_nn)) * round(es_chk * 0.1)))` checks of patience (isolated graphs use 1). Lower **es_chk** to finish earlier, but it may affect accuracy.
- **impl:** chooses the implementation ("auto", "numpy", "numba", or "cython"). Default is "auto", which uses the fixed Cython -> Numba -> NumPy order. Explicit sequential "numba" uses Numba -> Cython -> NumPy when unavailable.
---

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
