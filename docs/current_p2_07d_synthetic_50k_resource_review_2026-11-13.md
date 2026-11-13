# PyPCC P2-07D — Nine artificial 50,000-iteration cases, original native observers and ENFORCED resource watchdog

**Research-only synthetic engineering.** This PR is stacked on #272 and the frozen scientific P1 SHA-256 `f0111f153ade3d72191c107ace039e1835e5c27db2cf4b3df8d610b41b2f4166`. Its code and tests run full synthetic PCC fits on fake graphs and fake labels, **not the 72 real UCI fits**. Scientific P2 remains unauthorized.

## Rationale and exact synthetic design

P2-07C #272 proposed a conservative 72-case host/resource envelope. It was deliberately not a measured real-data upper bound or proof of enforcement. P2-07D moves from static budget equations and tiny 1,000-iteration pilots to actual **50,000-iteration SYNTHETIC** fits with a monitored OS child process, while preserving the original production and observer kernels.

| UCI dimension reference ONLY | Artificial nodes | Artificial features allocated | Artificial classes × 3 labels | Dense int64 `neib_list` max width |
| --- | ---: | ---: | ---: | ---: |
| 212 Vertebral | 310 | 6 | 3 × 3 = 9 particles | 22 |
| 850 Raisin | 900 | 7 | 2 × 3 = 6 particles | 20 |
| 292 Wholesale | 440 | 6 | 2 × 3 = 6 particles | 24 |

The graphs are created **only from synthetic node IDs**, using a deterministic symmetric 10-neighbor ring plus one extra-spoke hub chosen to reach the previously measured dense width. Their actual adjacency topology, degree distribution, node feature values, distances, class-label positions and random seed are **not the original UCI data** and are **not claimed to be equivalent to real kNN graphs**. Active neighbors are sorted, symmetric and non-self, and all unused dense entries are literal **`-1`**, not a CSR replacement.

The model graph attributes are installed **directly in a fresh synthetic research model** solely to preserve the tested dense `-1` layout: this is **not** a claim to have exercised P1's real `build_graph` route (already checked separately in P2-04). Class labels consist of exactly three artificial labeled nodes per synthetic class, with all other nodes `-1`. Both full `fit_predict` arms use the SAME generated graph, labels, seed **71** (a synthetic-only seed, not a frozen P1 propagation seed), `early_stop=False`, **50,000** propagation iterations, and the original P1 eight milestones `0,1,10,100,1000,5000,25000,50000`.

## Unmodified backend provenance and within-backend correctness

- Original NumPy/Numba/Cython milestone observer sources are **byte/Git-SHA pinned** to the approved research branches #253/#255/#256; the original NumPy/Numba synthetic fullfit helper files are reused only for their already-validated RNG-state digest utilities, also byte-pinned.
- The *new* native production/isolated Cython modules come exclusively from prospective P2-01F #271 [run #43488066633](https://github.com/fbreve/pypcc/actions/runs/43488066633) and are independently re-SHA-verified before copying to ephemeral CI import paths. They are **not** the historical P2-01C binaries.
- All 13 later candidate package wheel bytes are replayed from archived P2-01D #264 using an offline `--require-hashes --no-index` install; CPython patches Linux **3.12.16** and Windows **3.12.10** remain exact.
- Each of three backends on each synthetic dimension performs **two full 50k synthetic fits**: baseline original kernel and original research live observer. Check the same-backend final class predictions and **full mutable arrays** with SHA, including distance table, particle state, node dominance and ownership. For NumPy compare next Python RNG state, for Numba compare next native RNG draw; the Cython native next-RNG byte identity is not independently asserted here (prior shorter synthetic #263/#269 parity work exists).
- **Nine synthetic cases × two full synthetic fits = 18 synthetic fits per OS, 36 across Linux+Windows.** This does not secretly execute any of the 72 approved-design *real* UCI fits.

## Actual parent resource WATCHDOG, not post-hoc guess

A separate **parent interpreter process** launches nine isolated synthetic child interpreters **serially**, preserving append-only case STARTED/PASS/FAILED transitions. On Linux it polls child `/proc/<pid>/status` VmHWM/VmRSS and `/proc/<pid>/stat` process user+system ticks; on Windows it polls child `GetProcessMemoryInfo.PeakWorkingSetSize` and `GetProcessTimes`. It also samples elapsed wall time and scratch-directory disk use every approximately **100 ms** and uses `process.kill` on threshold violation or an independent **4-minute synthetic-only child safety cap**, which is stricter than the unapproved 30-minute real-fit candidate wall cap.

The child separately reports its own **whole-process** lifetime peak (Linux `ru_maxrss` or Windows PeakWorkingSetSize), per-arm fit CPU/wall seconds, and original observer snapshot array bytes. Parent verifies all are under the drafted limits of 2 GiB process memory, 60 CPU minutes, 30 wall minutes **per fit**, 64 MiB snapshot arrays, 1 GiB scratch, and a 1 MiB output JSON + 2 MiB console log. These thresholds are **proposal-level**, not a claim that 50k *real* fits are safe.

The unit-test suite includes **FOUR actual kill/termination probes** on interpreter-only fake processes that intentionally violate:
1. Wall-time through `sleep(3)` with 0.3s cap;
2. Whole-process peak memory through a 128 MiB fake `bytearray` with 40 MiB cap;
3. CPU through a 3s busy loop with 0.25s CPU cap;
4. Scratch disk through a 2 MiB fake temporary file with 128 KiB cap.

These tests are not PCC runs and cannot contact UCI data. They prove an OS-monitored child is really stopped when *these artificial breaches* are observed, not that every possible fast spike or host OS limit is caught: polling at 100 ms may miss a fast transient; a true production gate still needs durable execution host and supervisor/process-tree controls.

The CI archives **only** safe synthetic timing, resource totals, SHA fingerprints, native source SHA and an append-only attempt ledger; not the fake graph arrays or mutable particle checkpoint content, and absolutely no original UCI records, labels, masks, graph arrays or nodewise output.

## Precisely what passing would establish

- Both OS could carry out all **nine artificial 50k cases** using the original three backends and exact P1 snapshot schedule, with positive within-backend final-state agreement and recorded peak process RAM/time/disk values.
- The candidate OS resource enforcement is exercised on the normal paths **and** four forced-breach paths per OS. For comparison, the cross-OS job verifies only workload dimensions, exact snapshot byte count and provenance, **not** equal CPU timing, resident memory or native RNG trajectories.
- This would make the proposal more defensible than the original 1k pilot but **still not bound the real UCI graphs** or justify selecting final machine limits automatically. Real kNN graph structure, memory caches, observer export costs, OS load, CPU frequency and full 72-case scheduling remain different.

**STOP remains mandatory:** `P2_07_full_resources_gate_pass=false`, `P2_01_final_execution_toolchain_lock_pass=false`, `P2_execution_preflight_sha256=null`, `P2_execution_authorized=false`, **zero real PCC fits**. No merge to production, no new P2-08 SHA or automatic approval. Final host/toolchain and complete resource/release policy require separate maintainer review.

**CI state:** candidate until both Linux/Windows jobs and independent third scope report actually pass. All failure paths must publish failed status without retry or substituted case values.
