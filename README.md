# Particle Competition and Cooperation
Python code for the semi-supervised learning method "Particle Competition and Cooperation". 

This is a fork of Caio Carneloz' implementation:
https://github.com/caiocarneloz/pycc

This fork has a few bug fixes and lots of optimizations for faster execution. It also includes the early stop criteria from the original article.

However, if you need more speed, the pure Matlab version is still ~10 times faster. And the Matlab MEX version is ~100 times faster. They are available at:
https://github.com/fbreve/Particle-Competition-and-Cooperation

## Getting Started
#### Installation
You need Python 3.7 or later to use **pypcc**. You can find it at [python.org](https://www.python.org/).

You can clone this repo to your local machine using:
```
git clone https://github.com/fbreve/pypcc.git
```

## Usage
The usage of this class is pretty similar to [semi-supervised algorithms at scikit-learn](https://scikit-learn.org/stable/modules/label_propagation.html). A "example" code is available in this repository.

## Parameters
As arguments, **pypcc** receives the values explained below:

---
- **k_nn:** value that represents the amount of k-nearest neighbours to connect in the graph build (Euclidean distance).
- **p_grd:** value from 0 to 1 that defines the probability of particles to take the greedy movement. Default: 0.5.
- **delta_v:** value from 0 to 1 to control changing rate of the domination levels. Default: 0.1.
- **max_iter:** number of iterations until the label propagation stops (if the stop criteria is not met before that).
- **es_chk:** control how much iterations the algorithm performs after reaching some level of stability (stop criteria). Default: 2000. The formula is *(total_number_of_nodes / number_of_labeled_nodes) * es_chk*. Lower **es_chk** to finish earlier, but it may affect accuracy.
---

## Citation
If you use this algorithm, please cite the original publication:

`Breve, Fabricio Aparecido; Zhao, Liang; Quiles, Marcos Gonçalves; Pedrycz, Witold; Liu, Jiming, "Particle Competition and Cooperation in Networks for Semi-Supervised Learning," Knowledge and Data Engineering, IEEE Transactions on , vol.24, no.9, pp.1686,1698, Sept. 2012`

https://doi.org/10.1109/TKDE.2011.119

Accepted Manuscript: https://www.fabriciobreve.com/artigos/ieee-tkde-2009.pdf
