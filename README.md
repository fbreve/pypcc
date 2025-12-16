# Particle Competition and Cooperation
Python code for the semi-supervised learning method "Particle Competition and Cooperation". 

This is a fork of Caio Carneloz' implementation:
https://github.com/caiocarneloz/pycc

This fork has a few bug fixes and lots of optimizations for faster execution. It also includes the early stop criteria from the original article.

If you need more speed, the MATLAB MEX version is still up to ~2-3 times faster than the Python Numba version. It is available at:
https://github.com/fbreve/Particle-Competition-and-Cooperation

If you do not want or cannot use **numba**, a pure Numpy version is available.\   
Import **ParticleCompetitionAndCooperation()** from **pcc_numpy** instead of **pcc**.\
However, keep in mind that the Numba version is up to ~55 times faster than the pure Numpy version.

Benchmarks:

Machine: Intel Core i9 14900K with 128GB of RAM
Softwares: MATLAB R2025b, Python 3.12.12, Numpy 2.3.5, and Numba 0.62.1
Dataset: Wine Dataset
Repetitions: 100

With early_stop=true (default) and max_iter=500000 (default):

MATLAB MEX:         0.040s \
MATLAB pure:        0.903s \
Python Numba:       0.033s \
Python Numpy only:  1.381s

With early_stop=False and max_iter=500000 (500,000 fixed iterations):

MATLAB MEX:           1.000s \
MATLAB pure:         34.478s \
Python Numba:         2.250s \
Python Numpy only:  121.410s


## Getting Started
#### Installation
You need Python 3.7 or later to use **pypcc**. You can find it at [python.org](https://www.python.org/).

The required packages for **pypcc** are **numpy** and **numba**. The included example also uses **scikit-learn** to load the example dataset (Wine Dataset), to normalize the data, to calculate the accuracy, and to build the confusion matrix.
If you do not want or cannot use **numba**, a pure Numpy version is available, though it is much slower.

You can clone this repo to your local machine using:
```
git clone https://github.com/fbreve/pypcc.git
```

## Usage
The usage of this class is pretty similar to semi-supervised algorithms at scikit-learn. An "example" code is available in this repository.

## Parameters
As arguments, **pypcc** receives the values explained below:

---
- **k_nn:** value that represents the number of k-nearest neighbours used to build the graph (Euclidean distance).
- **p_grd:** value from 0 to 1 that defines the probability of particles to take the greedy movement. Default: 0.5.
- **delta_v:** value from 0 to 1 to control the rate of change of the domination levels. Default: 0.1.
- **max_iter:** number of iterations until the label propagation stops (if the stop criteria is not met before that).
- **es_chk:** control how many iterations the algorithm performs after reaching some level of stability (stop criteria). Default: 2000. The formula is *(total_number_of_nodes / number_of_labeled_nodes) * es_chk*. Lower **es_chk** to finish earlier, but it may affect accuracy.
---

## Citation
If you use this algorithm, please cite the original publication:

`Breve, Fabricio Aparecido; Zhao, Liang; Quiles, Marcos Gonçalves; Pedrycz, Witold; Liu, Jiming, "Particle Competition and Cooperation in Networks for Semi-Supervised Learning," Knowledge and Data Engineering, IEEE Transactions on , vol.24, no.9, pp.1686,1698, Sept. 2012`

https://doi.org/10.1109/TKDE.2011.119

Accepted Manuscript: https://www.fabriciobreve.com/artigos/ieee-tkde-2009.pdf
