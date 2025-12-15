# Particle Competition and Cooperation
Python code for the semi-supervised learning method "Particle Competition and Cooperation". 

This is a fork of Caio Carneloz' implementation:
https://github.com/caiocarneloz/pycc

This fork has a few bug fixes and lots of optimizations for faster execution. It also includes the early stop criteria from the original article.

If you need more speed, the MATLAB MEX version is still ~2.5 times faster than the Python Numba version. It is available at:
https://github.com/fbreve/Particle-Competition-and-Cooperation

If you do not want or cannot use **numba**, a pure Numpy version is available, though it is much slower. Import **ParticleCompetitionAndCooperation()** from **pcc_numpy** instead of **pcc**.

Alternatively, the pure MATLAB version is still ~6 times faster than the pure Python Numpy version, though ~35 times slower than the MATLAB MEX version, which is the fastest.

These numbers are based on the execution on the Wine Dataset for 500,000 iteration (without early stop) using MATLAB R2025b, Python 3.12.12, Numpy 2.3.5, and Numba 0.62.1, running on a Intel Core i9 14900K with 128GB of RAM.

MATLAB MEX:        ~ 0.97s

Python Numba:      ~ 2.5s

MATLAB pure:       ~ 33s

Python Numpy only: ~ 190s

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
