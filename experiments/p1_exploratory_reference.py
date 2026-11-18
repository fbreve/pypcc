"""P1 seed and outcome specification: SYNTHETIC ONLY, not an execution runner.

Never downloads source data, builds graphs, draws label masks, imports PyPCC
or executes propagation. All choices remain unapproved P1 proposals.
"""
import hashlib
import math

IDS = (212, 850, 292)  # candidate order only, NOT dataset selection
BACKENDS = ("numpy_sequential", "numba_sequential_nopython",
            "cython_sequential_cached_or_legacy_as_defined_by_early_stop")
N_REPLICATES = 8


class P1ScientificInvalid(ValueError):
    pass


def require(ok, reason):
    if not ok:
        raise P1ScientificInvalid(reason)


def proposed_mask_seed_words(index, replicate):
    """Proposed draft SeedSequence words; no NumPy RNG or mask drawn."""
    require(type(index) is int and 0 <= index < len(IDS), "Invalid registry index")
    require(type(replicate) is int and 0 <= replicate < N_REPLICATES,
            "Invalid mask replicate")
    return [20261112, index, replicate]


def proposed_propagation_seed(dataset_id, replicate, backend):
    """Review-only native seed derived independently from prospective mask RNG.

    First 4 bytes little endian of SHA-256 for
      pypcc-P1-propagation-v1|uci:<ID>|mask:<0-7>|<BACKEND>
    mapped to [1,2**32-1] to avoid xorshift32 zero-seed special handling.
    Exact backend and source versions remain separate future P2 gates.
    """
    require(type(dataset_id) is int and dataset_id in IDS,
            "Unknown prospective UCI ID")
    require(type(replicate) is int and 0 <= replicate < N_REPLICATES,
            "Invalid replicate")
    require(backend in BACKENDS, "Unknown backend")
    key = f"pypcc-P1-propagation-v1|uci:{dataset_id}|mask:{replicate}|{backend}"
    h = hashlib.sha256(key.encode("ascii")).digest()
    return 1 + (int.from_bytes(h[:4], "little") % (2**32 - 1))


def _argmax_and_tie(row, nclasses):
    require(isinstance(row, (tuple, list)) and len(row) == nclasses,
            "Class axis mismatch")
    require(all(type(x) in (int, float) and math.isfinite(x) and x >= 0
                for x in row), "Nonfinite or invalid dominance")
    maximum = max(row)
    require(maximum > 0, "Zero dominance total")
    ties = [k for k, v in enumerate(row) if v == maximum]
    return ties[0], len(ties) > 1


def synthetic_primary_flips(dominance_5000, dominance_50000, labeled_indices):
    """Calculate outcome only on synthetic fixture arrays, never real PCC data."""
    require(isinstance(dominance_5000, (tuple, list)) and
            isinstance(dominance_50000, (tuple, list)) and
            len(dominance_5000) == len(dominance_50000) >= 3,
            "Expected paired fixed-node checkpoints")
    n = len(dominance_5000)
    require(isinstance(labeled_indices, (tuple, list)) and
            len(labeled_indices) == len(set(labeled_indices)) and
            all(type(x) is int and 0 <= x < n for x in labeled_indices),
            "Invalid originally labeled node IDs")
    remaining = n - len(labeled_indices)
    require(remaining > 0, "No initially unlabeled nodes")
    k = len(dominance_5000[0])
    require(k >= 2, "Invalid class count")
    skip = set(labeled_indices)
    changed = tie_a = tie_b = 0
    for i, (before, after) in enumerate(zip(dominance_5000, dominance_50000)):
        a, ta = _argmax_and_tie(before, k)
        b, tb = _argmax_and_tie(after, k)
        if i not in skip:
            changed += int(a != b)
            tie_a += int(ta)
            tie_b += int(tb)
    return {
        "window": [5000, 50000],
        "unlabeled_n": remaining,
        "excluded_labeled_n": len(skip),
        "flip_n": changed,
        "flip_fraction": changed / remaining,
        "exact_ties_5000_n": tie_a,
        "exact_ties_50000_n": tie_b,
        "tie_rule": "first argmax index",
        "model_fits_executed": 0,
    }


def synthetic_72_case_descriptive_summary(cases):
    """Strictly complete paired analysis; no missing unit is silently dropped.

    Input can ONLY be synthetic for this research-only reference; NO real
    protocol is approved. Mask/graph SHA are mock fixture identities.
    """
    expected = {
        (ds, r, backend)
        for ds in IDS for r in range(N_REPLICATES) for backend in BACKENDS
    }
    require(isinstance(cases, (tuple, list)) and len(cases) == len(expected),
            "Incomplete or excessive 72-unit study")
    seen, graph_sha, masks, scores = set(), {}, {}, {}
    for row in cases:
        require(isinstance(row, dict), "Invalid case row")
        unit = row.get("dataset_id"), row.get("replicate"), row.get("backend")
        require(unit in expected and unit not in seen,
                "Unregistered or duplicate case")
        seen.add(unit)
        score = row.get("flip_fraction")
        require(type(score) in (int, float) and
                math.isfinite(score) and 0 <= score <= 1,
                "Invalid fraction")
        g, m = row.get("graph_sha256"), row.get("mask_sha256")
        require(all(type(x) is str and len(x) == 64 and
                    all(c in "0123456789abcdef" for c in x) for x in (g,m)),
                "Graph and label-mask SHA256 are mandatory")
        ds, rep, be = unit
        require(graph_sha.setdefault(ds, g) == g and
                masks.setdefault((ds,rep), m) == m,
                "Same dataset and replicate require paired graph/mask IDs")
        scores.setdefault((ds, be), []).append(float(score))
    require(seen == expected, "Missing registered case")
    summary = {}
    for be in BACKENDS:
        datasets = []
        for ds in IDS:
            values = sorted(scores[(ds,be)])
            require(len(values) == N_REPLICATES, "Incomplete mask replication")
            datasets.append({
                "dataset_id": ds,
                "n_masks": N_REPLICATES,
                "mean": sum(values)/N_REPLICATES,
                "median": (values[3]+values[4])/2,
                "min": values[0], "max": values[-1],
            })
        summary[be] = {
            "dataset_summaries": datasets,
            "unweighted_dataset_mean":
                sum(x["mean"] for x in datasets)/len(datasets),
        }
    return {
        "schema": "pypcc-P1-synthetic-exploratory-summary/v1",
        "status": "SYNTHETIC_ONLY_NOT_PCC_RESULTS",
        "n_proposed_cases": len(expected),
        "backends": summary,
        "CI_or_equivalence_claim": False,
        "independent_node_replicates_claimed": False,
        "datasets_selected": 0,
        "P1_approved": False,
        "P2_execution_sha256": None,
        "real_PCC_fits_executed": 0,
        "execution_authorized": False,
    }
