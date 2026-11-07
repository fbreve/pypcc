# Frozen phase 2: cross-platform replication of NumPy continuous observer

Created **after the first Linux-only phase completed 36/36 parity and observer overhead <50%**, and **before phase-2 implementation or seeing any new outcome**. The original data and thresholds are not retrospectively edited. The source remains only `pcc_numpy.py` observer opt-in and experimental modules; no Cython, Numba, `pcc.py`, RNG, graph, stopping or production default modifications.

## New independent replication

- Both **Windows and Linux** (native GitHub runners, Python 3.12) × Iris, Wine, Breast Cancer, Digits, with the canonical **production undirected union-symmetrized kNN=10** graph and two labeled nodes/class.
- Fresh, never-seen-before phase-2 seeds: **13001 and 13109**, used for independent label selection via `np.random.default_rng(seed)` and matching NumPy propagation seed. No seed selection after evaluating accuracy or stopping.
- For each OS × dataset × seed, use both fixed policies: `no_stop` 6001 steps with `early_stop=False`, and **aggressive_potential** 6001 maximum, `early_stop=True, es_chk=10` to cover early stopping where it occurs. Standard potential was already checked in phase 1 and is not repeated here.
- Each case has **two fresh fits**, one ordinary NumPy propagation without observer (but requesting the original `return_info` metadata), one true single-run observed propagation at milestones `0,1,11,101,501,2001,6001` plus actual stopping iteration. **4 datasets × 2 OS × 2 seeds × 2 policies × 2 fits = 64 timed fits, 32 strict observed/unobserved pairs**. Each job records every sample, plus frozen full-stop metadata and **six full-state arrays, full NumPy RNG state and next 16 draws equal**; no tolerance-based equivalence.
- Four dataset jobs on each OS (8 total), plus a mandatory complete-cohort summary job checking exact **32 cases / 64 fits**, no exclusions, both OS present, exact raw sample schedule and observer bytes <=48MiB.
- Median observed full-fit overhead for **no_stop** across two fresh seeds on each OS/dataset must stay <=**50%** (performance is exploratory; occasional negative/positive samples retained). Every state, stopping, RNG or schedule mismatch is an unconditional failure.
- The original phase-1 `--dataset` CLI retains its original **3 fixed default seeds 12007/12101/12203**, and the original initial summary continues to expect them. Phase 2 may pass an explicit `--seeds 13001 13109` to the same underlying audit. The new summary must independently assert these seed IDs and both OSes. This CLI change affects experiment scripts only, not `pcc_numpy.py`.

## Interpretation constraints

- Cross-platform parity is **within each backend/OS seeded pair**, not equality of binary NumPy stochastic trajectories *across* Windows/Linux.
- This still compares two variants in a **single Python source tree** with and without the observer argument, not a separately checked-out historical master Python source. A later independent master-vs-candidate source benchmark may be valuable if a production merge is contemplated.
- Record cases where aggressive stopping did not occur; do not pretend it was exercised. Never call net between-milestone differences a count of all possible intermediate flips.
- No permission to merge to master follows automatically from a green run. The historical #18 continuous native tracer remains unmerged and incompatible with the current Cython core.
