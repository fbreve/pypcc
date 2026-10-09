"""End-to-end backend comparison; same labeled sets per seed.

Note: Cython/Numba are sequential, while NumPy has both update modes.
Timing excludes graph construction but includes propagation and prediction.
"""
import time
import numpy as np
from sklearn.datasets import load_wine, load_digits
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from pcc import ParticleCompetitionAndCooperation


def evaluate(name, loader, seeds=10, iterations=1000):
    ds = loader()
    x = StandardScaler().fit_transform(ds.data)
    y = ds.target
    variants = (('cython', 'sequential'), ('numba', 'sequential'),
                ('numpy', 'sequential'), ('numpy', 'parallel'))
    results = {variant: {'accuracy': [], 'time': []} for variant in variants}
    # Warm up compiled/JIT backends outside the timed experiment.
    warm_x = x
    warm_y = y
    warm_labeled = np.concatenate([np.flatnonzero(y == cls)[:3] for cls in np.unique(y)])
    warm_observed = np.full(len(y), -1, dtype=np.int64)
    warm_observed[warm_labeled] = warm_y[warm_labeled]
    for impl, mode in variants:
        np.random.seed(123456)
        warm_model = ParticleCompetitionAndCooperation(impl=impl, update_mode=mode)
        warm_model.build_graph(warm_x, k_nn=10)
        warm_model.fit_predict(warm_observed, max_iter=2, early_stop=False)
    # Numba maintains a separate RNG state inside njit; seed it explicitly.
    try:
        from numba import njit
        @njit
        def seed_numba(value):
            np.random.seed(value)
    except ImportError:
        seed_numba = None
    for seed in range(seeds):
        rng = np.random.default_rng(seed)
        labeled = np.concatenate([
            rng.choice(np.flatnonzero(y == cls), 3, replace=False)
            for cls in np.unique(y)
        ])
        observed = np.full(len(y), -1, dtype=np.int64)
        observed[labeled] = y[labeled]
        for impl, mode in variants:
            np.random.seed(seed)
            if impl == 'numba' and seed_numba is not None:
                seed_numba(seed)
            model = ParticleCompetitionAndCooperation(impl=impl, update_mode=mode)
            model.build_graph(x, k_nn=10)
            selected = model._get_backend_fn()
            # Do not silently report fallback results under the requested name.
            module = selected.__module__
            if impl == 'cython' and not module.startswith('pcc_step'):
                raise RuntimeError(f'Cython unavailable; selected {module}')
            if impl == 'numba' and not module.startswith('pcc_numba'):
                raise RuntimeError(f'Numba unavailable; selected {module}')
            start = time.perf_counter()
            pred = model.fit_predict(observed, max_iter=iterations, early_stop=False)
            elapsed = time.perf_counter() - start
            accuracy = accuracy_score(y[observed == -1], pred[observed == -1])
            assert np.array_equal(pred[labeled], y[labeled])
            np.testing.assert_allclose(model.node.dominance.sum(axis=1), 1, atol=1e-8)
            assert np.min(model.node.dominance) >= -1e-10
            results[(impl, mode)]['accuracy'].append(accuracy)
            results[(impl, mode)]['time'].append(elapsed)
            print(f'DETAIL dataset={name} seed={seed} backend={impl} mode={mode} '
                  f'accuracy={accuracy:.4f} elapsed={elapsed:.4f}s', flush=True)
    for (impl, mode), data in results.items():
        print(f'SUMMARY dataset={name} backend={impl} mode={mode} '
              f'accuracy={np.mean(data["accuracy"]):.4f} '
              f'std={np.std(data["accuracy"], ddof=1):.4f} '
              f'time={np.mean(data["time"]):.4f}s', flush=True)


if __name__ == '__main__':
    evaluate('Wine', load_wine)
    evaluate('Digits', load_digits)
