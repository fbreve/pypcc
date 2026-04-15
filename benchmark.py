# -*- coding: utf-8 -*-
"""
Created on Mon Dec 15 18:26:11 2025

@author: fbrev

Benchmark PCC: Numpy x Numba x Cython

Some results:
    
Machine: Intel Core i9 14900K with 128GB of RAM
Software: Python 3.12.12, Numpy 2.4.4, Numba 0.64.0, and Cython 3.2.4

==================================================
CONFIG: EARLY_STOP=True, N_RUNS=100
==================================================

=== Wine (N_RUNS=100, EARLY_STOP=True) ===
Numpy: mean time = 0.409s (std = 0.122s), mean acc = 0.9330
Numba: mean time = 0.006s (std = 0.002s), mean acc = 0.9357
Cython: mean time = 0.004s (std = 0.001s), mean acc = 0.9330
Speedup (Numpy / Numba):  70.0x
Speedup (Numpy / Cython): 103.9x
Speedup (Numba / Cython): 1.48x

=== Digits (N_RUNS=100, EARLY_STOP=True) ===
Numpy: mean time = 2.027s (std = 0.514s), mean acc = 0.9529
Numba: mean time = 0.168s (std = 0.038s), mean acc = 0.9538
Cython: mean time = 0.161s (std = 0.035s), mean acc = 0.9538
Speedup (Numpy / Numba):  12.1x
Speedup (Numpy / Cython): 12.6x
Speedup (Numba / Cython): 1.04x

==================================================
CONFIG: EARLY_STOP=False, N_RUNS=10
==================================================

=== Wine (N_RUNS=10, EARLY_STOP=False) ===
Numpy: mean time = 38.546s (std = 9.475s), mean acc = 0.9168
Numba: mean time = 0.617s (std = 0.056s), mean acc = 0.9161
Cython: mean time = 0.391s (std = 0.038s), mean acc = 0.9143
Speedup (Numpy / Numba):  62.5x
Speedup (Numpy / Cython): 98.6x
Speedup (Numba / Cython): 1.58x

=== Digits (N_RUNS=10, EARLY_STOP=False) ===
Numpy: mean time = 113.889s (std = 19.781s), mean acc = 0.9588
Numba: mean time = 7.420s (std = 0.213s), mean acc = 0.9583
Cython: mean time = 6.725s (std = 0.168s), mean acc = 0.9567
Speedup (Numpy / Numba):  15.3x
Speedup (Numpy / Cython): 16.9x
Speedup (Numba / Cython): 1.10x

"""

import time
import numpy as np
from sklearn.datasets import load_wine, load_digits
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

from pcc import ParticleCompetitionAndCooperation  # classe única com impl='...'

# Configurações padrão (podem ser sobrescritas)
K_NN = 10
P_GRD = 0.5
DELTA_V = 0.1
MAX_ITER = 500000
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


def run_impl(X, y, impl, label, n_runs, early_stop):
    times = []
    accs = []

    # warm-up (importante para Numba; inócuo para os outros)
    rng = np.random.RandomState(0)
    y_ssl, labeled_idx, _ = make_ssl_labels(y, LABELED_FRACTION, rng)
    pcc = ParticleCompetitionAndCooperation(impl=impl)
    pcc.build_graph(X, k_nn=K_NN)
    pcc.fit_predict(
        y_ssl, p_grd=P_GRD, delta_v=DELTA_V,
        max_iter=MAX_ITER, early_stop=early_stop, es_chk=ES_CHK
    )

    for r in range(n_runs):
        rng = np.random.RandomState(r)
        y_ssl, labeled_idx, unlabeled_idx = make_ssl_labels(
            y, LABELED_FRACTION, rng
        )

        pcc = ParticleCompetitionAndCooperation(impl=impl)
        pcc.build_graph(X, k_nn=K_NN)

        t0 = time.perf_counter()
        y_pred = pcc.fit_predict(
            y_ssl, p_grd=P_GRD, delta_v=DELTA_V,
            max_iter=MAX_ITER, early_stop=early_stop, es_chk=ES_CHK
        )
        t1 = time.perf_counter()
        times.append(t1 - t0)

        acc = accuracy_score(y[unlabeled_idx], y_pred[unlabeled_idx])
        accs.append(acc)

        # display progress with running mean
        avg_time = np.mean(times)
        avg_acc = np.mean(accs)
        print(f"  [{impl}] Run {r+1}/{n_runs}: avg_time={avg_time:.3f}s, avg_acc={avg_acc:.4f}      ", end='\r')

    times = np.array(times)
    accs = np.array(accs)
    print(f"\n{label}: mean time = {times.mean():.3f}s "
          f"(std = {times.std():.3f}s), "
          f"mean acc = {accs.mean():.4f}")
    return times.mean()


def run_benchmark_dataset(name, X, y, n_runs, early_stop):
    print(f"\n=== {name} (N_RUNS={n_runs}, EARLY_STOP={early_stop}) ===")
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    t_numpy = run_impl(X, y, impl="numpy",  label="Numpy",  n_runs=n_runs, early_stop=early_stop)
    t_numba = run_impl(X, y, impl="numba",  label="Numba",  n_runs=n_runs, early_stop=early_stop)
    t_cyth  = run_impl(X, y, impl="cython", label="Cython", n_runs=n_runs, early_stop=early_stop)

    print(f"Speedup (Numpy / Numba):  {t_numpy / t_numba:.1f}x")
    print(f"Speedup (Numpy / Cython): {t_numpy / t_cyth:.1f}x")
    print(f"Speedup (Numba / Cython): {t_numba / t_cyth:.2f}x")


def main():
    datasets = [
        ("Wine", load_wine()),
        ("Digits", load_digits())
    ]

    configs = [
        {"early_stop": True,  "n_runs": 100},
        {"early_stop": False, "n_runs": 10}
    ]

    for config in configs:
        print("\n" + "="*50)
        print(f"CONFIG: EARLY_STOP={config['early_stop']}, N_RUNS={config['n_runs']}")
        print("="*50)
        
        for name, data in datasets:
            run_benchmark_dataset(name, data.data, data.target, 
                                 n_runs=config['n_runs'], 
                                 early_stop=config['early_stop'])


if __name__ == "__main__":
    main()
