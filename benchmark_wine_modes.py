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
    for seed in (11, 29, 47):
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
            print(f"seed={seed} mode={mode} accuracy={accuracy:.4f} "
                  f"elapsed={elapsed:.4f}s")


if __name__ == "__main__":
    main()
