"""Run outside the source tree to smoke-test installed modules and backends."""
import argparse
import numpy as np
import pcc
from pcc import ParticleCompetitionAndCooperation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--require-all', action='store_true')
    args = parser.parse_args()
    available = ['numpy']
    if pcc._HAS_NUMBA:
        available.append('numba')
    if pcc._HAS_CYTHON:
        available.append('cython')
    if args.require_all and len(available) != 3:
        raise RuntimeError(f'Missing installed backend: {available}')
    for backend in available:
        np.random.seed(0)
        model = ParticleCompetitionAndCooperation(impl=backend)
        model.build_graph(np.array([[0.], [.1], [.2], [1.], [1.1], [1.2]]), k_nn=2)
        labels = np.array([0, -1, -1, 1, -1, -1])
        predictions = model.fit_predict(labels, max_iter=20, early_stop=False)
        assert predictions.shape == labels.shape
        assert np.array_equal(predictions[labels >= 0], labels[labels >= 0])
        assert np.isfinite(model.node.dominance).all()
        assert np.allclose(model.node.dominance.sum(axis=1), 1.)
        assert (model.node.dominance >= -1e-12).all() and (model.node.dominance <= 1 + 1e-12).all()
        print(f'{backend}: installed smoke test passed')
    print('Imported:', pcc.__file__)


if __name__ == '__main__':
    main()
