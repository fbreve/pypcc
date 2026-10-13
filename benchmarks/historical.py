"""Frozen PR #8 inventory adapter. Execution reads caches; preparation opts in."""
import hashlib
import json
from pathlib import Path

import numpy as np
from experiments.patience_data import load, retrieve

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = ROOT / "docs/benchmarks/stopping_patience/protocol.json"
INVENTORY_PATH = ROOT / "docs/benchmarks/stopping_patience/inventory.json"
PROTOCOL = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
INVENTORY = json.loads(INVENTORY_PATH.read_text(encoding="utf-8"))
if hashlib.sha256(PROTOCOL_PATH.read_bytes().replace(b"\r\n", b"\n")).hexdigest() != INVENTORY["protocol_sha256"]:
    raise ValueError("Frozen protocol hash mismatch")
ALIASES = {"Wine": "wine", "Digits": "digits", "Iris": "iris", "BreastCancer": "breast_cancer"}
ENTRIES = {ALIASES.get(e["name"], e["name"].lower()): e for e in PROTOCOL["datasets"]}
FROZEN = {ALIASES.get(e["name"], e["name"].lower()): e for e in INVENTORY["datasets"]}
if set(ENTRIES) != set(FROZEN) or len(ENTRIES) != 20:
    raise ValueError("Frozen dataset inventory mismatch")


def fingerprint(*arrays):
    digest = hashlib.sha256()
    for array in arrays:
        digest.update(str((array.shape, array.dtype.str)).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def cached_read(url, path):
    if not path.is_file():
        raise FileNotFoundError(f"Missing {path}; use --prepare-historical CACHE --datasets NAME explicitly")
    return path.read_bytes()


def load_historical(name, cache=None, *, download=False):
    entry, frozen = ENTRIES[name], FROZEN[name]
    if "openml_id" in entry and cache is None:
        raise FileNotFoundError("Historical OpenML data require --historical-cache CACHE; no automatic downloads")
    raw, scaled, truth, provenance = load(entry, Path(cache or "."),
                                          fetch=retrieve if download else cached_read)
    for key in ("nodes", "processed_features", "classes", "class_counts", "class_names",
                "numeric_features", "nominal_features"):
        if provenance[key] != frozen[key]:
            raise ValueError(f"{name}: frozen {key} mismatch")
    if "openml_id" in entry:
        for key in ("openml_id", "version", "source_name", "raw_sha256", "raw_md5"):
            if provenance[key] != frozen[key]:
                raise ValueError(f"{name}: frozen source {key} mismatch")
    if fingerprint(raw, truth) != frozen["raw_data_fingerprint"]:
        raise ValueError(f"{name}: frozen raw data mismatch")
    provenance.update(historical_name=entry["name"],
                      protocol_sha256=INVENTORY["protocol_sha256"],
                      inventory_sha256=hashlib.sha256(INVENTORY_PATH.read_bytes()).hexdigest(),
                      processed_fingerprint_matches=fingerprint(scaled, truth) == frozen["processed_data_fingerprint"],
                      preprocessing=PROTOCOL["preprocessing"])
    return raw, truth, np.arange(len(truth), dtype=np.int64), provenance
