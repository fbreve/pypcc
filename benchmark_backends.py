"""End-to-end backend comparison; same labeled sets per seed.

Note: Cython/Numba are sequential, while NumPy has both update modes.
Timing excludes graph construction but includes propagation and prediction.
"""
import time
import argparse
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
    results = {variant: {'accuracy': [], 'time': [], 'dominance': [], 'prediction': []} for variant in variants}
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
            results[(impl, mode)]['prediction'].append(pred.copy())
            results[(impl, mode)]['accuracy'].append(accuracy)
            results[(impl, mode)]['time'].append(elapsed)
            # Mean maximum dominance measures convergence independent of RNG paths.
            results[(impl, mode)]['dominance'].append(
                float(np.max(model.node.dominance, axis=1).mean()))
            print(f'DETAIL dataset={name} seed={seed} backend={impl} mode={mode} '
                  f'accuracy={accuracy:.4f} elapsed={elapsed:.4f}s', flush=True)
    # Paired differences use the same labeled examples for every backend.
    reference = results[('numpy', 'sequential')]
    for (impl, mode), data in results.items():
        if (impl, mode) != ('numpy', 'sequential'):
            acc_diff = np.asarray(data['accuracy']) - np.asarray(reference['accuracy'])
            dom_diff = np.asarray(data['dominance']) - np.asarray(reference['dominance'])
            prediction_mismatches = [int(np.count_nonzero(a != b)) for a, b in
                                     zip(data['prediction'], reference['prediction'])]
            mismatch_seeds = [i for i, n in enumerate(prediction_mismatches) if n]
            for seed_index in mismatch_seeds:
                print(f'MISMATCH dataset={name} backend={impl} mode={mode} '
                      f'seed={seed_index} count={prediction_mismatches[seed_index]} '
                      f'accuracy_diff={acc_diff[seed_index]:+.6f} '
                      f'dominance_diff={dom_diff[seed_index]:+.6f}', flush=True)
            print(f'PAIRED dataset={name} backend={impl} mode={mode} '
                  f'reference=numpy/sequential '
                  f'accuracy_diff={np.mean(acc_diff):+.4f} '
                  f'accuracy_diff_std={np.std(acc_diff, ddof=1):.4f} '
                  f'dominance_diff={np.mean(dom_diff):+.4f} '
                  f'dominance_diff_std={np.std(dom_diff, ddof=1):.4f} '
                  f'accuracy_diff_max_abs={np.max(np.abs(acc_diff)):.4f} '
                  f'accuracy_wins={np.count_nonzero(acc_diff > 1e-12)} '
                  f'accuracy_ties={np.count_nonzero(np.abs(acc_diff) <= 1e-12)} '
                  f'accuracy_losses={np.count_nonzero(acc_diff < -1e-12)} '
                  f'prediction_mismatch_seeds={mismatch_seeds} '
                  f'prediction_mismatch_total={sum(prediction_mismatches)}', flush=True)
        print(f'SUMMARY dataset={name} backend={impl} mode={mode} '
              f'accuracy={np.mean(data["accuracy"]):.4f} '
              f'std={np.std(data["accuracy"], ddof=1):.4f} '
              f'time={np.mean(data["time"]):.4f}s '
              f'dominance={np.mean(data["dominance"]):.4f} '
              f'dominance_std={np.std(data["dominance"], ddof=1):.4f}', flush=True)


def diagnose_sequential_digits(seed=0, steps=1000):
    """Find the first state divergence between NumPy and Numba, one step at a time."""
    from pcc_numpy import pcc_step_numpy
    from pcc_numba import pcc_step_numba
    from numba import njit

    @njit
    def seed_numba(value):
        np.random.seed(value)

    ds = load_digits()
    x = StandardScaler().fit_transform(ds.data)
    y = ds.target
    rng = np.random.default_rng(seed)
    labeled = np.concatenate([
        rng.choice(np.flatnonzero(y == cls), 3, replace=False)
        for cls in np.unique(y)
    ])
    observed = np.full(len(y), -1, dtype=np.int64)
    observed[labeled] = y[labeled]
    models = []
    for impl in ('numpy', 'numba'):
        model = ParticleCompetitionAndCooperation(impl=impl, update_mode='sequential')
        model.build_graph(x, k_nn=10)
        # Initialize the same state without running propagation.
        model.fit_predict(observed, max_iter=0, early_stop=False)
        models.append(model)
    numpy_model, numba_model = models
    np.random.seed(seed)
    seed_numba(seed)
    for iteration in range(1, steps + 1):
        for model, step_fn in ((numpy_model, pcc_step_numpy), (numba_model, pcc_step_numba)):
            kwargs = {'update_mode': 'sequential'} if model is numpy_model else {}
            step_fn(model.neib_list, model.neib_qt, model.mapped_labels,
                    model.p_grd, model.delta_v, model.c, model.zerovec,
                    model.part.curnode, model.part.label, model.part.strength,
                    model.part.dist_table, model.node.dominance, model.owndeg,
                    model.deltap, model.dexp, model.buf_dom_row, model.buf_reduc,
                    model.buf_dom_list, model.buf_dist_list, model.buf_prob,
                    model.buf_slices, model.dist_weights, **kwargs)
        fields = (
            ('positions', numpy_model.part.curnode, numba_model.part.curnode),
            ('strength', numpy_model.part.strength, numba_model.part.strength),
            ('distance', numpy_model.part.dist_table, numba_model.part.dist_table),
            ('dominance', numpy_model.node.dominance, numba_model.node.dominance),
            ('owndeg', numpy_model.owndeg, numba_model.owndeg),
        )
        differences = []
        for name, left, right in fields:
            count = int(np.count_nonzero(left != right))
            if count:
                max_abs = float(np.max(np.abs(left.astype(np.float64) - right.astype(np.float64))))
                differences.append(f'{name}:{count}:max_abs={max_abs:.3g}')
        if differences:
            print(f'FIRST_DIVERGENCE dataset=Digits seed={seed} iteration={iteration} '
                  + ' '.join(differences), flush=True)
            return iteration
    print(f'NO_DIVERGENCE dataset=Digits seed={seed} iterations={steps}', flush=True)
    return None


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seeds', type=int, default=10)
    parser.add_argument('--iterations', type=int, default=1000)
    parser.add_argument('--diagnose-digits', type=int, default=None,
                        help='Find first NumPy/Numba divergence for this Digits seed')
    args = parser.parse_args()
    if args.seeds < 2 or args.iterations < 1:
        parser.error('--seeds must be >= 2 and --iterations must be >= 1')
    if args.diagnose_digits is not None:
        diagnose_sequential_digits(seed=args.diagnose_digits, steps=args.iterations)
        raise SystemExit(0)
    evaluate('Wine', load_wine, seeds=args.seeds, iterations=args.iterations)
    evaluate('Digits', load_digits, seeds=args.seeds, iterations=args.iterations)
