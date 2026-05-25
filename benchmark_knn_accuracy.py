# -*- coding: utf-8 -*-
"""Accuracy benchmark for KNN graph builders in PCC.

Compares only the graph construction methods:
1) sklearn
2) covariance_qdtree

This benchmark is accuracy-focused and always runs with early_stop=True.
"""

import argparse
import os
import sys
from dataclasses import dataclass

import numpy as np
from sklearn.datasets import (
    fetch_openml,
    load_breast_cancer,
    load_digits,
    load_iris,
    load_wine,
    make_circles,
    make_moons,
)
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

from pcc import ParticleCompetitionAndCooperation


_print = print

import os
results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(results_dir, exist_ok=True)
results_path = os.path.join(results_dir, "benchmark_accuracy.txt")

def print(*args, **kwargs):
    # Print to the console
    _print(*args, **kwargs)
    
    # Also write to results file dynamically
    import io
    f = io.StringIO()
    _print(*args, **kwargs, file=f)
    val = f.getvalue()
    try:
        with open(results_path, "a", encoding="utf-8") as f_out:
            f_out.write(val)
            f_out.flush()
    except Exception:
        pass




@dataclass
class DatasetItem:
    name: str
    X: np.ndarray
    y: np.ndarray
    source: str


def _encode_labels(y):
    y = np.asarray(y)
    labels, y_idx = np.unique(y, return_inverse=True)
    if labels.size < 2:
        raise ValueError("Dataset must contain at least 2 classes.")
    return y_idx.astype(np.int64)


def _subsample_dataset(X, y, max_samples, rng):
    if max_samples is None or X.shape[0] <= max_samples:
        return X, y
    idx = rng.choice(X.shape[0], size=max_samples, replace=False)
    return X[idx], y[idx]


def make_ssl_labels(y, labeled_fraction, rng):
    """Create SSL labels preserving at least one labeled sample per class."""
    y = np.asarray(y, dtype=np.int64)
    y_ssl = -1 * np.ones_like(y)

    unique = np.unique(y)
    labeled_idx = []
    for cls in unique:
        cls_idx = np.where(y == cls)[0]
        rng.shuffle(cls_idx)
        n_labeled_cls = max(1, int(round(labeled_fraction * len(cls_idx))))
        labeled_idx.append(cls_idx[:n_labeled_cls])

    labeled_idx = np.concatenate(labeled_idx)
    unlabeled_mask = np.ones(y.shape[0], dtype=bool)
    unlabeled_mask[labeled_idx] = False
    unlabeled_idx = np.where(unlabeled_mask)[0]
    y_ssl[labeled_idx] = y[labeled_idx]

    return y_ssl, unlabeled_idx


def run_one_method(X, y, y_ssl, unlabeled_idx, args, nn_method):
    model = ParticleCompetitionAndCooperation(impl=args.impl)
    model.build_graph(
        X,
        k_nn=args.k_nn,
        nn_method=nn_method,
        qdtree_max_depth=args.qdtree_max_depth,
        qdtree_min_points_split=args.qdtree_min_points_split,
    )

    y_pred = model.fit_predict(
        y_ssl,
        p_grd=args.p_grd,
        delta_v=args.delta_v,
        max_iter=args.max_iter,
        early_stop=True,
        es_chk=args.es_chk,
    )

    return float(accuracy_score(y[unlabeled_idx], y_pred[unlabeled_idx]))


def _run_trial(ds, run_seed, args):
    X = StandardScaler().fit_transform(ds.X)
    y = ds.y
    rng = np.random.RandomState(args.seed + run_seed)
    y_ssl, unlabeled_idx = make_ssl_labels(y, args.labeled_fraction, rng)

    a_sk = run_one_method(X, y, y_ssl, unlabeled_idx, args, nn_method="sklearn")
    a_qd = run_one_method(X, y, y_ssl, unlabeled_idx, args, nn_method="covariance_qdtree")

    return (run_seed, a_sk, a_qd)


def _run_trial_unpack(args_tuple):
    return _run_trial(*args_tuple)


def benchmark_dataset(ds, args, pool):
    # Parallelize trials across workers to utilize available CPU cores.
    print(
        f"\n=== {ds.name} [{ds.source}] "
        f"(n={ds.X.shape[0]}, d={ds.X.shape[1]}, classes={np.unique(ds.y).size}, runs={args.n_runs}) ==="
    )

    tasks = [(ds, rs, args) for rs in range(args.n_runs)]
    
    results = []
    for i, r in enumerate(pool.imap_unordered(_run_trial_unpack, tasks), 1):
        results.append(r)
        print(f"  Finished trial {i}/{args.n_runs}...", end="\r", flush=True)
    print(" " * 50, end="\r", flush=True) # Clear line after finishing

    # results: list of tuples (run_seed, a_sk, a_qd)
    results_sorted = sorted(results, key=lambda t: t[0])
    a_sk = np.array([r[1] for r in results_sorted])
    a_qd = np.array([r[2] for r in results_sorted])
    delta = a_qd - a_sk

    wins = int(np.sum(delta > 0))
    losses = int(np.sum(delta < 0))
    draws = int(np.sum(delta == 0))

    print(
        f"sklearn mean={a_sk.mean():.4f} std={a_sk.std():.4f} | "
        f"covariance_qdtree mean={a_qd.mean():.4f} std={a_qd.std():.4f} | "
        f"delta(mean)={delta.mean():+.4f} | wins/draws/losses={wins}/{draws}/{losses}"
    )

    return {
        "dataset": ds.name,
        "source": ds.source,
        "n": int(ds.X.shape[0]),
        "d": int(ds.X.shape[1]),
        "classes": int(np.unique(ds.y).size),
        "sk_mean": float(a_sk.mean()),
        "sk_std": float(a_sk.std()),
        "qd_mean": float(a_qd.mean()),
        "qd_std": float(a_qd.std()),
        "delta_mean": float(delta.mean()),
        "wins": wins,
        "draws": draws,
        "losses": losses,
    }


def load_builtin_datasets(suite):
    datasets = [
        DatasetItem("Wine", load_wine().data.astype(np.float64), _encode_labels(load_wine().target), "sklearn"),
        DatasetItem("Digits", load_digits().data.astype(np.float64), _encode_labels(load_digits().target), "sklearn"),
    ]

    if suite in ("extended", "all"):
        iris = load_iris()
        bc = load_breast_cancer()
        datasets.extend(
            [
                DatasetItem("Iris", iris.data.astype(np.float64), _encode_labels(iris.target), "sklearn"),
                DatasetItem("BreastCancer", bc.data.astype(np.float64), _encode_labels(bc.target), "sklearn"),
            ]
        )

        X_moons, y_moons = make_moons(n_samples=2000, noise=0.08, random_state=0)
        X_circ, y_circ = make_circles(n_samples=2000, noise=0.06, factor=0.5, random_state=0)
        datasets.extend(
            [
                DatasetItem("Moons2k", X_moons.astype(np.float64), _encode_labels(y_moons), "synthetic"),
                DatasetItem("Circles2k", X_circ.astype(np.float64), _encode_labels(y_circ), "synthetic"),
            ]
        )

    return datasets


def load_openml_datasets(dataset_names, max_samples=5000, random_state=0):
    datasets = []
    rng = np.random.RandomState(random_state)

    import time
    for name in dataset_names:
        name = name.strip()
        if not name:
            continue
            
        print(f"Fetching OpenML dataset '{name}' (this might take a while to download)...", flush=True)
        success = False
        for attempt in range(3):
            try:
                ds = fetch_openml(name=name, version="active", as_frame=False)
                X = np.asarray(ds.data, dtype=np.float64)
                y = _encode_labels(ds.target)
                X, y = _subsample_dataset(X, y, max_samples=max_samples, rng=rng)
                datasets.append(DatasetItem(f"OpenML:{name}", X, y, "openml"))
                print(f"Loaded OpenML dataset '{name}' with {X.shape[0]} samples.")
                success = True
                break
            except Exception as exc:
                print(f"Attempt {attempt+1}/3: Failed to load OpenML dataset '{name}': {exc}")
                time.sleep(2)
                
        if not success:
            print(f"Skipping OpenML dataset '{name}' after 3 failed attempts.")

    return datasets


def _extract_xy_from_mat(mat_dict):
    x_keys = ("X", "x", "data", "features", "T", "t")
    y_keys = ("Y", "y", "labels", "target", "targets")

    X = None
    y = None

    for key in x_keys:
        if key in mat_dict:
            import scipy.sparse
            if scipy.sparse.issparse(mat_dict[key]):
                X = np.asarray(mat_dict[key].toarray(), dtype=np.float64)
            else:
                X = np.asarray(mat_dict[key], dtype=np.float64)
            break

    for key in y_keys:
        if key in mat_dict:
            y = np.asarray(mat_dict[key]).squeeze()
            break

    if X is None or y is None:
        raise ValueError("Missing X/Y keys. Expected common feature/label keys.")

    if X.ndim != 2:
        raise ValueError("X must be 2-D.")

    if y.ndim == 2 and y.shape[1] > 1:
        y = np.argmax(y, axis=1)
    elif y.ndim > 1:
        y = y.ravel()

    if X.shape[0] != y.shape[0]:
        if X.shape[1] == y.shape[0]:
            X = X.T
        else:
            raise ValueError("X and y have incompatible sizes.")

    return X, _encode_labels(y)


def load_chapelle_mat_datasets(chapelle_dir, max_samples=5000, random_state=0):
    datasets = []
    if not chapelle_dir:
        return datasets
    if not os.path.isdir(chapelle_dir):
        print(f"Chapelle directory not found: {chapelle_dir}")
        return datasets

    try:
        from scipy.io import loadmat
    except ImportError:
        print("scipy is required to read Chapelle .mat datasets. Skipping Chapelle loader.")
        return datasets

    rng = np.random.RandomState(random_state)
    mat_files = [f for f in os.listdir(chapelle_dir) if f.lower().endswith(".mat") and "splits" not in f.lower()]

    for fname in sorted(mat_files):
        path = os.path.join(chapelle_dir, fname)
        try:
            mat = loadmat(path)
            X, y = _extract_xy_from_mat(mat)
            X, y = _subsample_dataset(X, y, max_samples=max_samples, rng=rng)
            datasets.append(DatasetItem(f"Chapelle:{os.path.splitext(fname)[0]}", X, y, "chapelle"))
            print(f"Loaded Chapelle-like dataset '{fname}' with {X.shape[0]} samples.")
        except Exception as exc:
            print(f"Skipping file '{fname}': {exc}")

    return datasets


def print_summary(rows):
    print("\n" + "=" * 120)
    print("KNN METHOD ACCURACY SUMMARY (early_stop=True)")
    print("=" * 120)
    print(
        "dataset | source | n | d | cls | "
        "sk_mean | qd_mean | delta(qd-sk) | sk_std | qd_std | wins/draws/losses"
    )
    print("-" * 120)
    for r in rows:
        print(
            f"{r['dataset']:<20} | {r['source']:<8} | {r['n']:>5} | {r['d']:>4} | {r['classes']:>3} | "
            f"{r['sk_mean']:.4f} | {r['qd_mean']:.4f} | {r['delta_mean']:+.4f} | "
            f"{r['sk_std']:.4f} | {r['qd_std']:.4f} | {r['wins']}/{r['draws']}/{r['losses']}"
        )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare PCC accuracy between sklearn and covariance_qdtree KNN methods."
    )

    parser.add_argument(
        "--suite",
        default="extended",
        choices=["quick", "extended", "all"],
        help="quick=Wine+Digits, extended=quick+more built-in, all=extended+OpenML/Chapelle if provided.",
    )
    parser.add_argument("--openml-datasets", default="", help="Comma-separated OpenML names (optional).")
    parser.add_argument("--chapelle-dir", default="", help="Path to Chapelle-style .mat files (optional).")
    parser.add_argument("--max-samples", type=int, default=5000, help="Subsample cap per dataset.")

    parser.add_argument("--impl", default="numba", choices=["auto", "numpy", "numba", "cython"],
                        help="PCC propagation backend kept fixed for fair KNN-method comparison.")
    parser.add_argument("--n-runs", type=int, default=30, help="Number of random label masks per dataset.")
    parser.add_argument("--seed", type=int, default=0, help="Base random seed.")
    parser.add_argument("--labeled-fraction", type=float, default=0.1)

    parser.add_argument("--k-nn", type=int, default=10)
    parser.add_argument("--p-grd", type=float, default=0.5)
    parser.add_argument("--delta-v", type=float, default=0.1)
    parser.add_argument("--max-iter", type=int, default=500000)
    parser.add_argument("--es-chk", type=int, default=2000)
    parser.add_argument("--qdtree-max-depth", type=int, default=None)
    parser.add_argument("--qdtree-min-points-split", type=int, default=2)
    parser.add_argument("--n-workers", type=int, default=8, help="Number of parallel worker processes to use.")

    return parser.parse_args()


def main():
    args = parse_args()

    # Clear the results log file at startup (only in main process)
    try:
        with open(results_path, "w", encoding="utf-8") as f:
            f.write("")
    except Exception:
        pass

    datasets = load_builtin_datasets(args.suite)

    openml_names = [x.strip() for x in args.openml_datasets.split(",") if x.strip()]
    if args.suite == "all" or openml_names:
        if not openml_names:
            openml_names = ["letter", "mnist_784", "vehicle", "segment"]
        datasets.extend(load_openml_datasets(openml_names, max_samples=args.max_samples, random_state=args.seed))

    if args.suite == "all" or args.chapelle_dir:
        datasets.extend(load_chapelle_mat_datasets(args.chapelle_dir, max_samples=args.max_samples, random_state=args.seed))

    if not datasets:
        raise RuntimeError("No datasets available for benchmarking.")

    print("\nBenchmark configuration")
    print("=" * 70)
    print("focus=accuracy")
    print("early_stop=True (fixed)")
    print(f"impl={args.impl}")
    print(f"k_nn={args.k_nn}, labeled_fraction={args.labeled_fraction}, n_runs={args.n_runs}")
    print(f"datasets={len(datasets)}")

    from multiprocessing import Pool
    workers = max(1, int(args.n_workers))

    rows = []
    with Pool(processes=workers) as pool:
        for ds in datasets:
            print("\n" + "=" * 70)
            print(
                f"Dataset: {ds.name} [{ds.source}] "
                f"(n={ds.X.shape[0]}, d={ds.X.shape[1]}, classes={np.unique(ds.y).size})"
            )
            print("=" * 70)
            rows.append(benchmark_dataset(ds, args, pool))

    print_summary(rows)

    print("\nBenchmark completed successfully.")


if __name__ == "__main__":
    main()
