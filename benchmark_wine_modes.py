"""End-to-end Wine comparison; exploratory, not a statistical conclusion."""
import time
import numpy as np
from sklearn.datasets import load_wine
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from pcc import ParticleCompetitionAndCooperation


def main():
    wine = load_wine()
    data = StandardScaler().fit_transform(wine.data)
    y = wine.target
    n_classes = len(np.unique(y))
    scores = {mode: [] for mode in ('parallel', 'sequential')}
    times = {mode: [] for mode in ('parallel', 'sequential')}
    for seed in range(30):
        rng = np.random.default_rng(seed)
        labeled = np.concatenate([rng.choice(np.where(y == cls)[0], 3, replace=False)
                                  for cls in range(n_classes)])
        observed = np.full(y.shape, -1, dtype=np.int64)
        observed[labeled] = y[labeled]
        for mode in ("parallel", "sequential"):
            np.random.seed(seed)
            model = ParticleCompetitionAndCooperation(impl="numpy", update_mode=mode)
            model.build_graph(data, k_nn=10)
            start = time.perf_counter()
            predicted = model.fit_predict(observed, max_iter=1000, early_stop=False)
            elapsed = time.perf_counter() - start
            mask = observed == -1
            accuracy = accuracy_score(y[mask], predicted[mask])
            assert np.isfinite(accuracy)
            assert np.array_equal(predicted[labeled], y[labeled])
            assert np.all(np.isfinite(model.node.dominance))
            np.testing.assert_allclose(model.node.dominance.sum(axis=1), 1, atol=1e-8)
            assert np.min(model.node.dominance) >= -1e-10
            scores[mode].append(accuracy)
            times[mode].append(elapsed)
            print(f"seed={seed} mode={mode} accuracy={accuracy:.4f} "
                  f"elapsed={elapsed:.4f}s")
    for mode in ('parallel', 'sequential'):
        print(f"SUMMARY mode={mode} mean_accuracy={np.mean(scores[mode]):.4f} "
              f"std_accuracy={np.std(scores[mode], ddof=1):.4f} "
              f"mean_time={np.mean(times[mode]):.4f}s")
    diff = np.array(scores['parallel']) - np.array(scores['sequential'])
    print(f"PAIRED difference_mean={np.mean(diff):.4f} "
          f"difference_std={np.std(diff, ddof=1):.4f} "
          f"wins={(diff > 0).sum()} ties={(diff == 0).sum()} "
          f"losses={(diff < 0).sum()}")


if __name__ == "__main__":
    main()
