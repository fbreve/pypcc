# Preregistered long-duration native timing replication — timing stability only

**Frozen before new long-run outcomes. This is a follow-up diagnostic, not a change to the cached Cython algorithm, model defaults, or the original master.**

## Why needed

The clean-master PR #178 V2 optimization already passes exact correctness gates (6,480 Cython numeric cases, independent binary parity and 4/4 regression), and the default-on activation was validated in stacked #179 before being integrated into PR #178's branch. The most recent post-integration cross-binary workflow [42289838928](https://github.com/fbreve/pypcc/actions/runs/42289838928) passed **240/240 native fits and 80/80 exact master/off/on state + stop parity**, but **failed the prespecified default-OFF speed gate** on Linux Breast Cancer: median -17.41% vs master. The preceding activation workflow [42289015365](https://github.com/fbreve/pypcc/actions/runs/42289015365), on the same executable source and same package versions, instead showed +2.69% on Linux Breast Cancer (explicit OFF); its original-master baseline calls had median runtimes around 5.4 ms versus 3.2 ms on the post-integration run. The short 20k benchmark is susceptible to per-runner scheduling, CPU scaling and process/warmup effects. It would be improper to ignore the failing run or relax its frozen threshold after seeing it.

## Distinct fixed diagnostic

- Compare **two independently compiled native binaries** in separate checkouts and subprocesses: untouched master at SHA `2902009c80b8edd3fc5f4961769444563fcf5b8a` and current clean PR candidate. Same Python 3.12/NumPy/Cython deps in each OS job, same native flags. Record native file SHA-256, loaded path and revision for every worker.
- Four matrix jobs: **Linux/Windows × canonical production undirected union-10-NN Breast Cancer (2 classes) / Digits (10 classes)**, exactly two initially labeled nodes per class. **Fresh seeds 9103, 9209, 9301, 9403**. Same `p_grd=0.5, delta_v=0.1, deltap=1, dexp=2`, independent deterministic graph and seed.
- For each OS×dataset×seed, compare three variants in balanced rotating order: **original master**, candidate Cython with **explicit `experimental_rowmax_cache=False`**, candidate with **no keyword (actual default ON)**. Run **300,000 iterations per fit** with `early_stop=True, es_chk=10000000` and assert the calculated `stop_max` exceeds the maximum possible 30k monitoring checks. The production potential sample schedule (every 10 iterations) is preserved while preventing any early termination.
- Each process handles **four back-to-back fresh-state native propagations** (clock/warmup controls), saving every duration and checking that every repetition has **the exact same six state arrays and complete stop metadata**. Compare all three variants' final NPZ arrays and all per-repetition SHA-256 state digests; no tolerance. Run **two independently launched blocks per seed**, with variant ordering rotated by seed and block.
- Exactly **2 OS × 2 datasets × 4 seeds × 2 blocks × 3 variants × 4 repeated fits = 384 timed native fits**, **32 three-way comparison groups**, each containing 12 matching independent/repeated native fits. All durations retained; do not discard first fit or negative savings.
- For each paired block, compare within-variant median of four durations, then report the **median of two block savings per seed**, and **median of four per-seed medians** for each OS×dataset×candidate variant. Include full raw duration range and absolute base/candidate times (ms). This is a separate 300k workload and cannot overwrite the original 20k result.
- Independent summary job requires all **384 duration rows, 32 paired groups, state hashes and binary identities**. Keep the original failed short-run and passed earlier short-run reports unchanged.

## Decision gates (frozen before any measurement)

1. Reject for any state or stopping-metadata mismatch, native identity/checksum mismatch, directed graph, or early termination. This remains unconditional.
2. The longer fixed-work **candidate explicit-OFF** median saving versus original master must be **>=-8% on all four OS×dataset combinations**. Otherwise preserve the default-off regression as unresolved.
3. The longer fixed-work default-ON cache must save **>=+15%** for Digits on **both OS** and have **no worse than -5%** median regression for Breast Cancer on either OS.
4. A pass suggests the short-run Linux Breast Cancer OFF failure is not stable on the longer workload and motivates dedicated profiling / hardware review. **It does not retroactively turn the failed 20k experiment into a success** or justify merging to master without recognizing unstable short-run estimates. If results disagree, report both and continue isolated profiling rather than tuning cache semantics.

No production source changes, no master merge and no modification to the already frozen early-stop policy.
