# -*- coding: utf-8 -*-
"""
Created on Mon Dec 15 18:26:11 2025

@author: fbrev

Some results:
    
Machine: Intel Core i9 14900K with 128GB of RAM
Software: Python 3.12.12, Numpy 2.3.5, and Numba 0.62.1

EARLY_STOP = True

=== Wine ===
Numba: mean time = 0.029s (std = 0.009s), mean acc = 0.9309
Numpy: mean time = 1.296s (std = 0.372s), mean acc = 0.9304
Speedup (Numpy / Numba): 44.0x

=== Digits ===
Numba: mean time = 0.607s (std = 0.132s), mean acc = 0.9532
Numpy: mean time = 57.465s (std = 34.906s), mean acc = 0.9544
Speedup (Numpy / Numba): 94.7x

EARLY_STOP = False

=== Wine ===
Numba: mean time = 2.018s (std = 0.015s), mean acc = 0.9340
Numpy: mean time = 129.612s (std = 21.768s), mean acc = 0.9326
Speedup (Numpy / Numba): 64.2x

"""

import time
import numpy as np
from sklearn.datasets import load_wine, load_digits
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

from pcc import ParticleCompetitionAndCooperation as PCCNumba
from pcc_numpy import ParticleCompetitionAndCooperation as PCCNumpy

N_RUNS = 100
K_NN = 10
P_GRD = 0.5
DELTA_V = 0.1
MAX_ITER = 500000
EARLY_STOP = False
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


def run_benchmark_dataset(name, X, y):
    print(f"\n=== {name} ===")
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    def run_impl(PCCClass, label):
        times = []
        accs = []

        # primeira chamada opcional para “aquecer” Numba
        rng = np.random.RandomState(0)
        y_ssl, labeled_idx, _ = make_ssl_labels(y, LABELED_FRACTION, rng)
        pcc = PCCClass()
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

            pcc = PCCClass()
            pcc.build_graph(X, k_nn=K_NN)

            t0 = time.perf_counter()
            y_pred = pcc.fit_predict(
                y_ssl, p_grd=P_GRD, delta_v=DELTA_V,
                max_iter=MAX_ITER, early_stop=EARLY_STOP, es_chk=ES_CHK
            )
            t1 = time.perf_counter()
            times.append(t1 - t0)

            # avalia só nos não rotulados (semelhante ao MATLAB)
            accs.append(accuracy_score(y[unlabeled_idx], y_pred[unlabeled_idx]))

        times = np.array(times)
        accs = np.array(accs)
        print(f"{label}: mean time = {times.mean():.3f}s "
              f"(std = {times.std():.3f}s), "
              f"mean acc = {accs.mean():.4f}")
        return times.mean()

    t_numba = run_impl(PCCNumba, "Numba")
    t_numpy = run_impl(PCCNumpy, "Numpy")

    print(f"Speedup (Numpy / Numba): {t_numpy / t_numba:.1f}x")


def main():
    wine = load_wine()
    run_benchmark_dataset("Wine", wine.data, wine.target)

    digits = load_digits()
    run_benchmark_dataset("Digits", digits.data, digits.target)


if __name__ == "__main__":
    main()
