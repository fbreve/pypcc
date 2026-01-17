# -*- coding: utf-8 -*-
"""
Created on Mon Dec 15 18:26:11 2025

@author: fbrev

Benchmark PCC: Numpy x Numba x Cython

Some results:
    
Machine: Intel Core i9 14900K with 128GB of RAM
Software: Python 3.12.12, Numpy 2.3.5, Numba 0.63.1, and Cython 3.2.2

EARLY_STOP = True, N_RUNS=100

=== Wine ===
Numpy: mean time = 1.044s (std = 0.304s), mean acc = 0.9322
Numba: mean time = 0.031s (std = 0.007s), mean acc = 0.9330
Cython: mean time = 0.033s (std = 0.009s), mean acc = 0.9316
Speedup (Numpy / Numba):  33.6x
Speedup (Numpy / Cython): 31.3x
Speedup (Numba / Cython): 0.93x

=== Digits ===
Numpy: mean time = 27.626s (std = 7.295s), mean acc = 0.9535
Numba: mean time = 0.672s (std = 0.146s), mean acc = 0.9532
Cython: mean time = 0.356s (std = 0.081s), mean acc = 0.9530
Speedup (Numpy / Numba):  41.1x
Speedup (Numpy / Cython): 77.5x
Speedup (Numba / Cython): 1.89x

EARLY_STOP = False, N_RUNS=10

=== Wine ===
Numpy: mean time = 87.506s (std = 0.562s), mean acc = 0.9161
Numba: mean time = 2.164s (std = 0.011s), mean acc = 0.9180
Cython: mean time = 2.384s (std = 0.114s), mean acc = 0.9124
Speedup (Numpy / Numba):  40.4x
Speedup (Numpy / Cython): 36.7x
Speedup (Numba / Cython): 0.91x

=== Digits ===
Numpy: mean time = 942.190s (std = 4.391s), mean acc = 0.9603
Numba: mean time = 19.568s (std = 0.071s), mean acc = 0.9605
Cython: mean time = 9.618s (std = 0.058s), mean acc = 0.9595
Speedup (Numpy / Numba):  48.1x
Speedup (Numpy / Cython): 98.0x
Speedup (Numba / Cython): 2.03x

"""

import time
import numpy as np
from sklearn.datasets import load_wine, load_digits
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

from pcc import ParticleCompetitionAndCooperation  # classe única com impl='...'

N_RUNS = 100
K_NN = 10
P_GRD = 0.5
DELTA_V = 0.1
MAX_ITER = 500000
EARLY_STOP = True
ES_CHK = 2000
LABELED_FRACTION = 0.1  # fração de exemplos rotulados


def make_ssl_labels(y, labeled_fraction, rng):
    n = len(y)
    idx = np.arange(n)
    rng.shuffle(idx)
    n_labeled = int(n * labeled_fraction)
    labeled_idx = idx[:n_labeled]
    unlabeled_idx = idx[n_labeled:]
    y_ssl = -1 * np.ones_like(y)
    y_ssl[labeled_idx] = y[labeled_idx]
    return y_ssl, labeled_idx, unlabeled_idx


def run_impl(X, y, impl, label):
    times = []
    accs = []

    # warm-up (importante para Numba; inócuo para os outros)
    rng = np.random.RandomState(0)
    y_ssl, labeled_idx, _ = make_ssl_labels(y, LABELED_FRACTION, rng)
    pcc = ParticleCompetitionAndCooperation(impl=impl)
    pcc.build_graph(X, k_nn=K_NN)
    pcc.fit_predict(
        y_ssl, p_grd=P_GRD, delta_v=DELTA_V,
        max_iter=MAX_ITER, early_stop=EARLY_STOP, es_chk=ES_CHK
    )

    for r in range(N_RUNS):
        rng = np.random.RandomState(r)
        y_ssl, labeled_idx, unlabeled_idx = make_ssl_labels(
            y, LABELED_FRACTION, rng
        )

        pcc = ParticleCompetitionAndCooperation(impl=impl)
        pcc.build_graph(X, k_nn=K_NN)

        t0 = time.perf_counter()
        y_pred = pcc.fit_predict(
            y_ssl, p_grd=P_GRD, delta_v=DELTA_V,
            max_iter=MAX_ITER, early_stop=EARLY_STOP, es_chk=ES_CHK
        )
        t1 = time.perf_counter()
        times.append(t1 - t0)

        accs.append(accuracy_score(y[unlabeled_idx], y_pred[unlabeled_idx]))

    times = np.array(times)
    accs = np.array(accs)
    print(f"{label}: mean time = {times.mean():.3f}s "
          f"(std = {times.std():.3f}s), "
          f"mean acc = {accs.mean():.4f}")
    return times.mean()


def run_benchmark_dataset(name, X, y):
    print(f"\n=== {name} ===")
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    t_numpy = run_impl(X, y, impl="numpy",  label="Numpy")
    t_numba = run_impl(X, y, impl="numba",  label="Numba")
    t_cyth  = run_impl(X, y, impl="cython", label="Cython")

    print(f"Speedup (Numpy / Numba):  {t_numpy / t_numba:.1f}x")
    print(f"Speedup (Numpy / Cython): {t_numpy / t_cyth:.1f}x")
    print(f"Speedup (Numba / Cython): {t_numba / t_cyth:.2f}x")


def main():
    wine = load_wine()
    run_benchmark_dataset("Wine", wine.data, wine.target)

    digits = load_digits()
    run_benchmark_dataset("Digits", digits.data, digits.target)


if __name__ == "__main__":
    main()
