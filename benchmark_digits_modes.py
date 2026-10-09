"""Paired evaluation of NumPy PCC modes on sklearn Digits."""
import time
import numpy as np
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from pcc import ParticleCompetitionAndCooperation


def main():
    dataset = load_digits()
    x = StandardScaler().fit_transform(dataset.data)
    y = dataset.target
    modes = ('parallel', 'sequential')
    scores = {m: [] for m in modes}
    times = {m: [] for m in modes}
    for seed in range(30):
        rng = np.random.default_rng(seed)
        labeled = np.concatenate([
            rng.choice(np.flatnonzero(y == cls), 3, replace=False)
            for cls in np.unique(y)
        ])
        observed = np.full(len(y), -1, dtype=np.int64)
        observed[labeled] = y[labeled]
        for mode in modes:
            np.random.seed(seed)
            model = ParticleCompetitionAndCooperation(impl='numpy', update_mode=mode)
            model.build_graph(x, k_nn=10)
            start = time.perf_counter()
            prediction = model.fit_predict(observed, max_iter=1000, early_stop=False)
            elapsed = time.perf_counter() - start
            accuracy = accuracy_score(y[observed == -1], prediction[observed == -1])
            assert np.array_equal(prediction[labeled], y[labeled])
            assert np.all(np.isfinite(model.node.dominance))
            np.testing.assert_allclose(model.node.dominance.sum(axis=1), 1, atol=1e-8)
            assert np.min(model.node.dominance) >= -1e-10
            scores[mode].append(accuracy)
            times[mode].append(elapsed)
            print(f"seed={seed} mode={mode} accuracy={accuracy:.4f} elapsed={elapsed:.4f}s")
    for mode in modes:
        print(f"SUMMARY mode={mode} mean_accuracy={np.mean(scores[mode]):.4f} "
              f"std_accuracy={np.std(scores[mode], ddof=1):.4f} "
              f"mean_time={np.mean(times[mode]):.4f}s")
    diff = np.asarray(scores['parallel']) - np.asarray(scores['sequential'])
    print(f"PAIRED difference_mean={diff.mean():.4f} "
          f"difference_std={diff.std(ddof=1):.4f} "
          f"wins={(diff > 0).sum()} ties={(diff == 0).sum()} "
          f"losses={(diff < 0).sum()}")


if __name__ == '__main__':
    main()
