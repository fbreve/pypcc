"""Deterministic, shared inputs for optional Python performance experiments."""
import hashlib
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from sklearn import datasets
from sklearn.preprocessing import StandardScaler

from pcc_graph import build_knn_graph


REAL_DATASETS = {
    "iris": (150, 4, 3, datasets.load_iris),
    "wine": (178, 13, 3, datasets.load_wine),
    "breast_cancer": (569, 30, 2, datasets.load_breast_cancer),
    "digits": (1797, 64, 10, datasets.load_digits),
    "mnist_5000": (5000, 784, 10, None),
    "mnist_10000": (10000, 784, 10, None),
}


@dataclass(frozen=True)
class Case:
    name: str
    nodes: int
    features: int
    classes: int
    k_nn: int = 10
    labels_per_class: int | None = 3
    labeled_fraction: float | None = None
    data_seed: int = 0
    label_seed: int = 0
    synthetic: bool = False

    def validate(self):
        for field in ("nodes", "features", "classes", "k_nn", "data_seed", "label_seed"):
            if type(getattr(self, field)) is not int:
                raise ValueError(f"{field} must be an integer")
        if self.labels_per_class is not None and type(self.labels_per_class) is not int:
            raise ValueError("labels_per_class must be an integer")
        if not self.name or not all(c.isalnum() or c in "_-" for c in self.name):
            raise ValueError("case names must contain only letters, digits, '_' or '-'")
        if self.nodes < 2 or self.features < 1 or not 2 <= self.classes <= self.nodes:
            raise ValueError("invalid dataset dimensions or class count")
        if not 1 <= self.k_nn < self.nodes:
            raise ValueError("require 1 <= k_nn < nodes")
        if (self.labels_per_class is None) == (self.labeled_fraction is None):
            raise ValueError("choose labels_per_class or labeled_fraction, exclusively")
        if self.labels_per_class is not None and self.labels_per_class < 1:
            raise ValueError("labels_per_class must be positive")
        if self.labeled_fraction is not None and not 0 < self.labeled_fraction < 1:
            raise ValueError("require 0 < labeled_fraction < 1")
        if self.particles >= self.nodes or self.particles < self.classes:
            raise ValueError("label policy must cover every class and leave unlabeled nodes")
        for seed in (self.data_seed, self.label_seed):
            if not 0 <= seed < 2**32:
                raise ValueError("seeds must be unsigned 32-bit integers")
        if not self.synthetic:
            if self.name not in REAL_DATASETS or (self.nodes, self.features, self.classes) != REAL_DATASETS[self.name][:3]:
                raise ValueError("real dataset name/dimensions must match the documented dataset")

    @property
    def particles(self):
        if self.labels_per_class is not None:
            return self.labels_per_class * self.classes
        return max(self.classes, math.floor(self.nodes * self.labeled_fraction))

    @property
    def case_id(self):
        return f"{self.name}-labels{self.label_seed}"


SYNTHETIC_DEFAULTS = (
    Case("synthetic_small", 150, 4, 3, synthetic=True),
    Case("synthetic_medium", 500, 16, 5, synthetic=True),
    Case("synthetic_particles", 1000, 32, 10, labels_per_class=None,
         labeled_fraction=.1, synthetic=True),
)


def array_hash(array):
    # Canonical little-endian C order makes hashes independent of memory layout.
    canonical = np.ascontiguousarray(array, dtype=array.dtype.newbyteorder("<"))
    digest = hashlib.sha256()
    digest.update(json.dumps([list(canonical.shape), canonical.dtype.str]).encode())
    digest.update(canonical.tobytes())
    return digest.hexdigest()


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def select_labels(truth, case):
    rng = np.random.default_rng(case.label_seed)
    classes = np.unique(truth)
    known = []
    for cls in classes:
        candidates = np.flatnonzero(truth == cls)
        count = case.labels_per_class if case.labels_per_class is not None else 1
        if len(candidates) <= count:
            raise ValueError("each class must retain at least one unlabeled sample")
        known.extend(rng.choice(candidates, count, replace=False).tolist())
    if case.labels_per_class is None:
        remaining = np.setdiff1d(np.arange(len(truth)), known)
        known.extend(rng.choice(remaining, case.particles - len(known), replace=False).tolist())
    labels = np.full(len(truth), -1, dtype=np.int64)
    labels[known] = truth[known]
    return labels


def load_data(case, mnist_cache=None):
    if case.synthetic:
        x, truth = datasets.make_blobs(n_samples=case.nodes, n_features=case.features,
                                      centers=case.classes, random_state=case.data_seed)
        return x, truth, np.arange(case.nodes, dtype=np.int64), {
            "source": "sklearn.datasets.make_blobs", "parameters": asdict(case),
            "cluster_std": 1., "center_box": [-10., 10.], "shuffle": True,
        }
    if case.name not in REAL_DATASETS:
        raise ValueError(f"unknown dataset: {case.name}")
    _, _, _, loader = REAL_DATASETS[case.name]
    if loader is not None:
        data = loader()
        return data.data, data.target, np.arange(case.nodes, dtype=np.int64), {
            "source": f"sklearn.datasets.{loader.__name__}",
        }
    if mnist_cache is None:
        raise FileNotFoundError("MNIST requires --mnist-cache; prepare it with --prepare-mnist NEW_CACHE")
    cache = Path(mnist_cache)
    if not (cache / "source.json").is_file() or not (cache / "mnist.npz").is_file():
        raise FileNotFoundError("MNIST cache incomplete/missing; prepare a new directory with --prepare-mnist NEW_CACHE")
    metadata = json.loads((cache / "source.json").read_text(encoding="utf-8"))
    source_file = cache / "mnist.npz"
    if file_hash(source_file) != metadata["sha256"]:
        raise ValueError("MNIST cache hash mismatch")
    with np.load(source_file, allow_pickle=False) as stored:
        x, truth = stored["x"], stored["truth"]
    if x.shape != (70000, 784) or truth.shape != (70000,):
        raise ValueError("expected the complete 70000 x 784 MNIST source")
    # Both subsets use prefixes of the same permutation and share one cache.
    indices = mnist_indices(case.nodes, case.data_seed)
    return x[indices], truth[indices], indices.astype(np.int64), metadata


def mnist_indices(nodes, seed):
    return np.random.default_rng(seed).permutation(70000)[:nodes]


def prepare_fixture(case, mnist_cache=None):
    case.validate()
    start = time.perf_counter()
    raw, truth, indices, source = load_data(case, mnist_cache)
    raw = np.asarray(raw, dtype=np.float64)
    truth = np.asarray(truth, dtype=np.int64)
    if raw.shape != (case.nodes, case.features) or len(np.unique(truth)) != case.classes:
        raise ValueError("dataset dimensions/class coverage differ from the case")
    if not np.isfinite(raw).all():
        raise ValueError("dataset must contain finite numeric features")
    loaded = time.perf_counter()
    scaler = StandardScaler()
    x = scaler.fit_transform(raw)
    labels = select_labels(truth, case)
    prepared = time.perf_counter()
    neighbors, degrees = build_knn_graph(x, case.k_nn, n_jobs=1)
    end = time.perf_counter()
    arrays = dict(raw=raw, x=x, truth=truth, labels=labels, neighbors=neighbors,
                  degrees=degrees, source_indices=indices,
                  scaler_mean=scaler.mean_, scaler_scale=scaler.scale_)
    hashes = {name: array_hash(value) for name, value in arrays.items()}
    metadata = dict(schema=1, case=asdict(case), source=source,
                    preprocessing="full transductive StandardScaler; float64; no row removal",
                    graph="symmetric Euclidean k-NN; production builder; n_jobs=1; ordered lists",
                    hashes=hashes,
                    fixture_sha256=hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest(),
                    labeled_nodes=int(np.sum(labels != -1)),
                    labeled_class_counts={str(c): int(np.sum(labels == c)) for c in np.unique(truth)},
                    timings=dict(dataset_load_s=loaded-start,
                                 preprocessing_and_labels_s=prepared-loaded,
                                 graph_build_s=end-prepared))
    return arrays, metadata


def save_fixture(directory, arrays, metadata):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    with (directory / "arrays.npz").open("xb") as stream:
        np.savez_compressed(stream, **arrays)
    with (directory / "fixture.json").open("x", encoding="utf-8") as stream:
        json.dump(metadata, stream, indent=2, allow_nan=False)


def prepare_mnist(cache):
    """Explicit network operation; never called by benchmark execution or tests."""
    cache = Path(cache)
    cache.mkdir(parents=True, exist_ok=False)
    data = datasets.fetch_openml(name="mnist_784", version=1, as_frame=False,
                                 data_home=str(cache / "openml"))
    x, truth = np.asarray(data.data, dtype=np.float64), np.asarray(data.target, dtype=np.int64)
    if x.shape != (70000, 784) or truth.shape != (70000,):
        raise ValueError("unexpected MNIST source dimensions")
    path = cache / "mnist.npz"
    with path.open("xb") as stream:
        np.savez_compressed(stream, x=x, truth=truth)
    with (cache / "source.json").open("x", encoding="utf-8") as stream:
        json.dump(dict(source="OpenML mnist_784", version=1, data_id=data.details.get("id"),
                       url=data.url, sha256=file_hash(path), nodes=70000, features=784), stream, indent=2)
