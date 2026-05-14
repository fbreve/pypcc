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
**auto** chooses the fastest implementation available.

For graph construction, you can choose the nearest-neighbor method in `build_graph`:

```python
model.build_graph(X, k_nn=10, nn_method="sklearn")
model.build_graph(X, k_nn=10, nn_method="covariance_qdtree")
# alias accepted:
model.build_graph(X, k_nn=10, nn_method="covariance-qdtree")
```

You may also check the original MATLAB version:
https://github.com/fbreve/Particle-Competition-and-Cooperation

For MATLAB, there is a MEX version available, which is comparable in speed to
the Cython implementation on the same machine.

## Benchmarks:

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

## Getting Started
#### Installation
You need Python 3.7 or later to use **pypcc**. You can find it at [python.org](https://www.python.org/).

The only required package for **pypcc** is **numpy**. \
You also need **numba** and/or **cython** to use Numba and Cython versions, respectively. \
The included example also uses **scikit-learn** to load the example dataset (Wine Dataset), 
to normalize the data, to calculate the accuracy, and to build the confusion matrix.

You can clone this repo to your local machine using:

```bash
git clone https://github.com/fbreve/pypcc.git
```

To compile cython locally use:

```bash
cd pypcc
pip install -e .  # installs pypcc and builds the Cython extension
```

## Usage
The usage of this class is pretty similar to semi-supervised algorithms at scikit-learn. 
An "example" code is available in this repository.

For comprehensive accuracy benchmarks comparing different KNN graph construction methods (`sklearn` vs `covariance_qdtree`) across many datasets (including optional OpenML and local Chapelle-style .mat files), use:

```bash
python benchmark_knn_accuracy.py --suite extended
python benchmark_knn_accuracy.py --suite all --openml-datasets letter,mnist_784
python benchmark_knn_accuracy.py --suite all --chapelle-dir path/to/chapelle_mat
```

Useful options:

```bash
python benchmark_knn_accuracy.py --help
python benchmark_knn_accuracy.py --n-runs 20 --n-workers 8
```

## Parameters
As arguments, **pypcc** receives the values explained below:

---
- **k_nn:** value that represents the number of k-nearest neighbours used to build the graph (Euclidean distance).
- **nn_method:** nearest-neighbor graph construction method. Options: `"sklearn"` (default) or `"covariance_qdtree"` (`"covariance-qdtree"` alias).
- **p_grd:** value from 0 to 1 that defines the probability of particles to take the greedy movement. Default: 0.5.
- **delta_v:** value from 0 to 1 to control the rate of change of the domination levels. Default: 0.1.
- **max_iter:** number of iterations until the label propagation stops (if the stop criteria is not met before that).
- **es_chk:** control how many iterations the algorithm performs after reaching some level of stability (stopping criterion). Default: 2000. The formula is *(total_number_of_nodes / number_of_labeled_nodes) * es_chk*. Lower **es_chk** to finish earlier, but it may affect accuracy.
- **impl:** chooses the implementation ("auto", "numpy", "numba", or "cython"). Default is "auto", which selects the fastest implementation available. "cython" falls back to "numba", and "numba" falls back to "numpy" when a specific implementation is not available.
- **qdtree_max_depth:** optional cap for `covariance_qdtree` tree depth.
- **qdtree_min_points_split:** minimum points required to split a tree node in `covariance_qdtree` (default 2).
---

## Citation
If you use this algorithm, please cite the original publication:

`Breve, Fabricio Aparecido; Zhao, Liang; Quiles, Marcos Gonçalves; Pedrycz, Witold; Liu, Jiming, "Particle Competition and Cooperation in Networks for Semi-Supervised Learning," Knowledge and Data Engineering, IEEE Transactions on , vol.24, no.9, pp.1686,1698, Sept. 2012`

https://doi.org/10.1109/TKDE.2011.119

Accepted Manuscript: https://www.fabriciobreve.com/artigos/ieee-tkde-2009.pdf
