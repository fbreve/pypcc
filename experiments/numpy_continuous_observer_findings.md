# NumPy continuous trajectory observer: completed frozen experiments

**Status:** Experimental draft PR #182, not merged into production. This is **true uninterrupted single-run instrumentation** at fixed sampled milestones, not the repeated independent seeded prefix replays in historical draft #181. Old PR #18, with incompatible native NumPy/Numba/Cython instrumentation from before seeded Cython's global-potential cache and finite-entry checks, remains unmerged.

## What changed

Only `pcc_numpy.py` changes the executable model: a new **trailing optional `observer=None`** argument in `pcc_propagate_numpy`; normal `ParticleCompetitionAndCooperation(impl='numpy').fit_predict()` calls use **exactly the previous behavior and public API**. At opt-in execution, the existing particle-update loop, RNG calls and stopping cadence are unchanged. A bounded collector copies state only when a scheduled milestone is actually reached and at the real final iteration; an already scheduled final row is marked terminal without duplication. Every snapshot copies **full MT19937 RNG state** (no draws), positions, strengths, distances, dominance, raw ownership and derived assignments. Collectors are limited to **9 snapshots and 48 MiB**, and explicitly fail on invalid budgets, duplicate rows or inconsistent final states. No modifications to NumPy movement, Numba, Cython, graph construction, potential stopping or any production default.

## Phase 1 — four Linux datasets, three new seeds and three policies

- Frozen [protocol](numpy_continuous_observer_protocol.md); [CI **42324534877**](https://github.com/fbreve/pypcc/actions/runs/42324534877) **4/4 dataset jobs and mandatory summary PASS**. **36/36 strict observed/unobserved pairs**, **72 actual fresh fits**, no excluded observations. Every pair passes bitwise final six-array state, *complete* native stopping information, **full NumPy RNG state and next 16 random outputs**, and exactly matched observed final state + RNG vs actual run. Expected milestone schedule including real early stop verified.
- Standardized canonical undirected production union-10NN: Iris, Wine, Breast Cancer, Digits. Two labeled nodes/class. Seeds `12007,12101,12203`. Fixed 6001-step no-stop policy, real potential (`es_chk=2000`) and aggressive potential (`es_chk=10`). Real potential did not stop within this 6001-step horizon; aggressive potential stopped in Iris **3/3**, Wine **3/3**, Breast Cancer **2/3**, Digits **0/3**, with exact stop count and last sample equality.
- Phase-1 raw full case/snapshot/hash CSV artifacts: Iris **13553984655**, Wine **13554811791**, Breast Cancer **13554262359**, Digits **13554004617**. Complete mandatory report **13554731395**.

| Dataset | No-stop median full-fit observer overhead | Real potential median | Aggressive potential median |
|---|---:|---:|---:|
| Iris | +2.13% | +1.01% | +2.51% |
| Wine | +2.06% | +1.40% | +1.65% |
| Breast Cancer | +1.05% | +1.46% | +1.14% |
| Digits | +1.10% | +0.17% | +0.20% |

The preregistered exploratory overhead threshold (<={50}% median per dataset in the 6001 no-stop policy) **passes**.

## Phase 2 — independent Windows/Linux replication

- Second frozen [protocol](numpy_continuous_observer_cross_os_protocol.md), written before new outcomes. Fresh independent seeds `13001,13109`; both Windows and Linux, all four datasets, 6001-step no-stop and aggressive-potential policies. **32/32 strict pairs, 64 fits, 8/8 OS×dataset jobs and mandatory complete-cohort audit PASS**. [CI **42324925020**](https://github.com/fbreve/pypcc/actions/runs/42324925020), full report artifact **13554588311**.
- All **six arrays, all native stop metadata, entire RNG state and next 16 draws** match the unobserved run bitwise for every OS-specific pair, all sample schedules valid and within memory budget. Cross-platform equality of absolute stochastic trajectories is *not* claimed; comparison is within each OS/seed/backend. The aggressive policy actually stopped in Iris **2/2 on each OS**, Wine **2/2 on each OS**, Breast Cancer **1/2 on each OS**, and Digits **0/2** in both OS.
- Raw OS/dataset artifacts: Linux Iris **13554847494**, Wine **13554493278**, Breast Cancer **13554558066**, Digits **13554383306**; Windows Iris **13555230745**, Wine **13554398274**, Breast Cancer **13554618089**, Digits **13555315514**.

| Dataset | 6001 no-stop median overhead: Linux | Windows |
|---|---:|---:|
| Iris | +0.50% | +0.63% |
| Wine | +0.65% | +2.36% |
| Breast Cancer | +0.54% | +1.04% |
| Digits | −1.06% | +1.41% |

The worst frozen OS/dataset median is **+2.36%**, below the +50% ceiling. Small negative percentages in CI timing should be treated as host noise, not acceleration. All raw measurements, including negative differences, were retained.

## Correctness and scientific scope

- Phase 1+2: **68/68 strict paired observed vs unobserved fits**, 136 actual fits, plus synthetic collector tests. All state/RNG/stop equality requirements passed in the tested conditions, and 48 MiB/9-snapshot budgets held.
- Original regression CI on the phase-1 code [42324534830](https://github.com/fbreve/pypcc/actions/runs/42324534830) passed all four Linux/Windows Python 3.10/3.12 jobs. The later cross-platform commit also triggered the same full regression suite; check its final state separately.
- The collector measures **net changes between selected snapshots**, not every possible flip between them. A quiet interval does not prove convergence; no new stopping rule, accuracy claim, or causal attribution is supported.
- Unlike PR #181, these samples come from **one uninterrupted fit**; unlike old #18, this is currently **NumPy only**. Numba and Cython require separate opt-in ports with native RNG and stop exact-parity audits; the Cython global-potential cache and finite preflight must never be replaced by old code.
- The measured overhead is **observer-enabled vs observer-disabled within the same candidate Python source**, not a separate untouched-master-vs-candidate build. Before production merge, review callback failure behavior, zero-iteration and unseeded paths, synchronous-mode handling, native/OS baseline overhead and the interface's packaging contract.

**Decision:** Keep #182 draft and unmerged. The tool is scientifically useful for low-resolution real continuous trajectories, and both frozen experimental gates passed. It is not yet a production feature nor evidence that potential/P10 stopping is universally safe.
