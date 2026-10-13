# Python fixed-iteration campaign: reproducibility report

The completed primary experiment comprises 3,600 measured fits on the frozen
20-dataset PR #8 inventory. The separate [432-fit sensitivity analysis](performance_order_sensitivity.md)
does not replace or contribute records to these summaries. Every timing outlier
is retained. No early-stopping experiment was run. This report describes the
observed machine and workloads; it does not select a universal native backend winner.

## Protocol and dataset provenance

Measured source: `c9f512c2da576a4095e9bf0192133ee34896f494` (clean working tree).
The [inventory and preprocessing dimensions](performance_campaign_plan.md#frozen-20-dataset-inventory)
are frozen from PR #8: four scikit-learn classification datasets and 16 OpenML
datasets pinned to version 1. No MNIST, row subsampling, or replacement datasets.
The [frozen protocol](benchmarks/stopping_patience/protocol.json) and
[inventory](benchmarks/stopping_patience/inventory.json) remain unchanged.
Cached raw sources, processed-data fingerprints, ordered graph fingerprints,
and historical label-mask fingerprints passed strict validation. No mismatch
was accepted or explained away by a library-version difference.

Original preprocessing uses numeric median imputation, full one-hot nominal
encoding with an explicit missing category, LabelEncoder targets, then full
transductive StandardScaler. Missing targets are rejected; no rows are removed.
The production symmetric Euclidean k-NN graph uses `k_nn=10`, `n_jobs=1` and
ordered neighbor lists. Prepared data/scaling/graph are reused across consecutive
label seeds; each mask is freshly selected and all saved arrays are hashed.
Reuse does not move work into propagation timers or change fixture contents.

Label seeds 10 through 19 choose exactly three samples per class without
replacement, with deterministic `default_rng(label_seed)` selection. Identical
saved fixtures are used across backends. PCC parameters are `p_grd=0.5`,
`delta_v=0.1`, `deltap=1`, `dexp=2`, `es_chk=2000`. Each measured fit performs
exactly 10,000 fixed iterations (`early_stop=False`, zero stopping checks).
NumPy uses its global RandomState seeded 0; Numba's independent generator is
seeded 0 by the recorded helper; Cython resets xorshift32 to 123456789.
Backend trajectories are deterministic individually, but differ between backends.
Neither graph/label identity nor valid outputs imply identical random paths.

Each fixture/backend has two retained two-iteration warm-ups, followed by six
measured fits. Default execution rotates `(numpy,numba,cython)` by
`(case_index + repeat) % 3`; every position occurs twice. Only three of the six
permutations occur, twice each; native pair directions are not equally balanced.
This limitation motivated the separate fully counterbalanced follow-up.
There are 200 fixtures, 600 fixture/backend groups, 1,200 warm-ups and
3,600 measured fits. Final state hashes, backend identities, fixed iteration
counts and unlabeled accuracy are checked without discarding measurements.

## Hardware, software and execution provenance

Native Windows 11 Pro 10.0.26300; physical ASUS ROG STRIX Z790-E GAMING WIFI
machine with Intel Core i7-14700K (20 cores, 28 logical processors), about
128 GiB installed RAM (137,183,023,104 bytes reported). The host reports a
hypervisor present; physical manufacturer/model and native local process
checks distinguish the execution from a container guest. Conda environment
`pypcc-dev`: Python 3.12.15, NumPy 2.4.6, Numba 0.68.0, Cython 3.2.4,
SciPy 1.18.1, scikit-learn 1.9.1, llvmlite 0.50.0 and setuptools 84.0.0.
Windows Conda DLL paths were activated through `conda run`.

Loaded Cython binary: `pcc_step.cp312-win_amd64.pyd`, SHA-256
`e29083201d16ae30b1470f7abd878ec922c2ba645bff77aa50522a49b39b92a0`.
Its original compiler/build log and effective extension compiler flags are
unavailable. Python's compiler banner is not proof of extension build flags.
The binary hash identifies the measured artifact; source reproduction cannot
guarantee an identical rebuilt binary. Manifests record source/module paths,
source hashes, library/threadpool information and full arguments.

Balanced power plan; frequency, foreground activity, CPU core migrations and
affinity were not controlled. Thread environment overrides were unset; the
preflight observed MKL 20 and OpenMP 28 threads, while graph construction
explicitly uses one job. The original external monitor samples RAM/disk every
15 seconds, not process peak memory or executing CPU frequency. Follow-up
monitoring is described separately. Timing causes are not identifiable from
these records alone.

Native campaign process: **2026-10-13 15:20:27.458339–15:57:15.756259 UTC**
(12:20–12:57 in UTC−03:00); **2,208.297945 seconds**, approximately 36 min 48 s.
This includes startup, warm-ups, preparation, artifact writing and the external
monitor; excludes Conda launcher, preflight, fingerprint audit and analysis.

## Timing definitions and preparation costs

Elapsed timers use `perf_counter`; wall timestamps are separate provenance.

| Field | Included work |
| --- | --- |
| `dataset_load_s` | Original loader, decoding, imputation/encoding and reference standardization |
| `preprocessing_and_labels_s` | Saved scaler transformation and label selection initially; label selection only on reuse |
| `graph_build_s` | Production graph construction, once per dataset |
| `initialization_and_dispatch_s` | Fit entry to backend entry: particle/state allocation and dispatch |
| `propagation_s` | Backend entry to exit, including backend validation and returned information |
| `postprocessing_s` | Backend exit to fit return: normalization/predictions |
| `fit_predict_s` | Sum of initialization/dispatch, propagation and postprocessing |

Model construction, graph copying, seeding, identity/signature/hash/invariant
checks and output serialization are outside measured fit timers. Compilation
and first calls remain in separate warm-up/helper records; signature checks
verify no new compilation occurred in the measured fits. `propagation_s` is
the reported comparison boundary, not total training, graph time, or compiler time.
Dataset preparation sums loader plus preprocessing/labels: **0.675787 s**;
20 graph constructions: **0.610412 s**. Reused loads/graphs are recorded as
zero charges, with source-case references rather than duplicated costs.

## Raw-derived results and diagnostics

The following review recomputed aggregates from JSONL, independently checked
saved arrays/fingerprints, and retained every raw fit. Committed tables:
[timings](performance_results/dataset_timings.csv),
[paired speedups](performance_results/paired_speedups.csv) and
[native comparisons](performance_results/native_comparisons.csv).
The offline [analysis tool](../tools/analyze_fixed_results.py) recomputes
medians/quartiles and exact accounting without running PCC. Detailed original
scripts, tables and diagnostic PDF/PNG are retained in the verified archive.

## All 20 dataset comparisons

Times are propagation median (IQR), milliseconds, across all 60 measured records per dataset/backend. These pooled IQRs include both label-mask and timing variation. Paired speedups are medians of time ratios for matching dataset, label seed and repetition; they are not ratios of marginal medians. This distinction explains cases such as Satimage where the marginal-median ranking differs from the median paired ratio. Detailed per-fixture data and label-mask ratio summaries are in the archived `detailed_analysis.json`.

| Dataset | NumPy ms | Numba ms | Cython ms | NumPy/Numba | NumPy/Cython |
| --- | ---: | ---: | ---: | ---: | ---: |
| wine | 922.591 (8.953) | 5.293 (0.131) | 3.720 (0.073) | 173.7x | 247.5x |
| digits | 4046.644 (1441.769) | 17.195 (1.384) | 16.869 (3.011) | 185.6x | 194.1x |
| iris | 1283.383 (207.578) | 5.439 (2.125) | 3.644 (1.603) | 214.5x | 326.0x |
| breast_cancer | 827.364 (170.732) | 4.073 (1.607) | 2.590 (1.065) | 186.4x | 296.9x |
| balancescale | 1305.976 (305.553) | 5.353 (2.333) | 3.929 (1.753) | 226.8x | 296.8x |
| germancredit | 888.568 (220.313) | 4.124 (0.887) | 2.872 (1.351) | 198.7x | 282.3x |
| diabetes | 826.162 (186.735) | 4.025 (1.687) | 2.591 (1.402) | 194.1x | 293.0x |
| sonar | 893.194 (275.428) | 4.072 (1.253) | 2.629 (1.182) | 195.9x | 303.4x |
| glass | 2607.107 (440.126) | 11.302 (4.627) | 10.973 (4.781) | 212.2x | 250.6x |
| vehicle | 1717.579 (409.429) | 6.981 (2.758) | 5.633 (2.042) | 220.1x | 265.8x |
| ionosphere | 880.952 (166.724) | 4.719 (1.861) | 2.874 (1.460) | 172.4x | 271.4x |
| pendigits | 4527.435 (597.263) | 22.748 (10.719) | 20.741 (7.325) | 207.9x | 206.4x |
| satimage | 2869.997 (484.835) | 14.539 (7.041) | 15.924 (7.267) | 192.8x | 196.3x |
| spambase | 851.557 (299.654) | 4.675 (1.461) | 3.445 (2.071) | 163.3x | 220.9x |
| pageblocks | 2083.968 (408.384) | 9.680 (4.238) | 8.300 (4.461) | 188.4x | 231.5x |
| banknote | 880.563 (325.210) | 3.839 (1.106) | 2.441 (1.294) | 206.8x | 312.4x |
| bloodtransfusion | 800.653 (175.743) | 3.882 (1.462) | 2.393 (0.821) | 185.5x | 291.4x |
| yeast | 4121.912 (696.284) | 18.807 (7.214) | 17.325 (6.441) | 203.0x | 214.9x |
| segment | 2837.218 (488.372) | 12.292 (4.242) | 11.492 (6.130) | 209.1x | 225.1x |
| qsarbiodeg | 792.580 (293.384) | 3.979 (0.334) | 2.609 (0.200) | 192.7x | 284.0x |

## Direct Numba-versus-Cython comparison

Ratio = Numba propagation time / Cython propagation time: >1 means Cython faster. Q1-Q3 is the middle 50% of 60 paired ratios, not a confidence interval. Pair counts and seed counts are descriptive consistency checks, not independent Bernoulli trials or significance tests.

| Dataset | Median Numba/Cython | Q1-Q3 | Cython faster pairs /60 | Seed median >1.10 /10 | Assessment |
| --- | ---: | --- | ---: | ---: | --- |
| wine | 1.429x | 1.407-1.452 | 58 | 10 | clear descriptive Cython advantage |
| digits | 1.047x | 0.957-1.065 | 40 | 0 | unresolved |
| iris | 1.453x | 1.353-1.567 | 56 | 10 | clear descriptive Cython advantage |
| breast_cancer | 1.558x | 1.412-1.607 | 58 | 10 | clear descriptive Cython advantage |
| balancescale | 1.362x | 1.286-1.424 | 55 | 9 | clear descriptive Cython advantage |
| germancredit | 1.462x | 1.332-1.530 | 52 | 10 | clear descriptive Cython advantage |
| diabetes | 1.545x | 1.441-1.603 | 56 | 10 | clear descriptive Cython advantage |
| sonar | 1.555x | 1.394-1.641 | 56 | 10 | clear descriptive Cython advantage |
| glass | 1.167x | 0.986-1.250 | 43 | 7 | unresolved |
| vehicle | 1.260x | 1.150-1.371 | 51 | 9 | clear descriptive Cython advantage |
| ionosphere | 1.526x | 1.333-1.741 | 57 | 10 | clear descriptive Cython advantage |
| pendigits | 1.043x | 0.889-1.249 | 34 | 3 | unresolved |
| satimage | 1.025x | 0.887-1.096 | 36 | 1 | unresolved |
| spambase | 1.358x | 1.106-1.419 | 47 | 10 | clear descriptive Cython advantage |
| pageblocks | 1.157x | 1.078-1.234 | 47 | 10 | modest/variable Cython advantage |
| banknote | 1.604x | 1.380-1.657 | 53 | 10 | clear descriptive Cython advantage |
| bloodtransfusion | 1.601x | 1.505-1.710 | 55 | 10 | clear descriptive Cython advantage |
| yeast | 1.023x | 0.925-1.124 | 35 | 1 | unresolved |
| segment | 1.103x | 0.999-1.168 | 45 | 6 | unresolved |
| qsarbiodeg | 1.529x | 1.502-1.555 | 58 | 10 | clear descriptive Cython advantage |

Descriptive screening rule: Descriptive screening only: Cython paired speedup median >=1.20, Q1 >1.10, >=9/10 seed medians >1.10, and median >1.10 in each NumPy-position stratum and both three-repeat cycles. Thresholds are post-analysis review criteria, not preregistered inference.
Clear descriptive Cython advantage: wine, iris, breast_cancer, balancescale, germancredit, diabetes, sonar, vehicle, ionosphere, spambase, banknote, bloodtransfusion, qsarbiodeg.
Unresolved: digits, glass, pendigits, satimage, yeast, segment. PageBlocks has a modest/variable Cython advantage. No dataset establishes a comparably clear Numba advantage in this run.
The paired middle ranges for strong cases lie above parity even when marginal backend IQRs overlap. Pairing removes some common timing disturbance; it does not make different backend trajectories identical or observations independent.
Workload-weighted total-time NumPy/Numba and NumPy/Cython ratios are 195.79x and 226.08x. Total Numba/Cython propagation ratio is 1.155x. These weight expensive workloads more heavily and must not be presented as universal/equal-dataset speedups.

## Repetition, order, warm-up and drift

Normalize each measured time by its own fixture/backend median before comparing repetition/position; this prevents large graphs or different label masks from dominating the descriptive averages. Every cell below includes 200 fixture ratios per repetition. IQRs remain large despite nearly flat median curves.

| Backend | Rep 1 | Rep 2 | Rep 3 | Rep 4 | Rep 5 | Rep 6 | Late/early cycle median (Q1-Q3) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| numpy | 1.015 | 0.992 | 0.992 | 0.998 | 1.003 | 0.997 | 1.001 (0.917-1.098) |
| numba | 0.996 | 1.000 | 0.996 | 1.002 | 1.003 | 0.999 | 1.003 (0.946-1.092) |
| cython | 1.013 | 0.998 | 0.997 | 1.001 | 0.999 | 1.000 | 1.002 (0.892-1.077) |

| Backend | Position 1 | Position 2 | Position 3 | First/warm two-iteration ratio median | Typical within-fixture IQR/median |
| --- | ---: | ---: | ---: | ---: | ---: |
| numpy | 0.997 | 1.000 | 1.004 | 1.176 | 13.40% |
| numba | 1.000 | 1.001 | 0.999 | 1.211 | 23.21% |
| cython | 1.000 | 0.998 | 1.002 | 1.243 | 24.68% |

No large universal six-repeat warm-up or monotonic drift appears: repeat-one medians are ~1.5% above the fixture median for NumPy, ~1.3% for Cython, and slightly below for Numba; late/early cycle medians differ by <=0.3%. Global position medians differ by <=0.4%. These are observations, not equivalence tests or proof of no environmental drift.
Two-iteration first calls are typically 18-24% slower than their paired warm call; the first Numba propagation call reaches 0.735 s, consistent with included first-call/possible compilation cost. It is not compiler-only timing. Both warm-ups were retained separately. Short warm-up overhead cannot be extrapolated directly to a 10,000-iteration fit.
Important schedule limitation: only three cyclic permutations were used (NumPy-Numba-Cython, Numba-Cython-NumPy, Cython-NumPy-Numba), repeated twice per fixture. Every backend occupied each position twice, but Numba preceded Cython four times and followed it twice. This balances positions, not all pair directions/preceding-backend contexts. Ratios pair same repetition with distinct sequential positions, sometimes separated by a long NumPy fit.
Within-dataset order can matter even when global position medians are flat: Satimage paired Numba/Cython median is 1.061 when Numba precedes Cython and 0.988 in the reverse direction; PageBlocks is 1.173 versus 1.034. Thus a near-parity rank should not be presented as settled.
Matched normalized log-time correlation is 0.338 for Numba/Cython, versus 0.125/0.131 for NumPy/native. This modest shared jitter is compatible with some common environmental influence, but does not identify frequency, CPU core migration, scheduler load, cache effects or a causal mechanism.
Within-fixture median IQR percentages vary through the fixed dataset sequence: Wine is quiet (0.7%/1.6%/1.1% for NumPy/Numba/Cython), while Satimage is 13.8%/29.6%/41.6% and late QSAR is 18.3%/3.5%/10.2%. This is nonstationary-looking jitter, not a simple universal warming curve. Dataset, masks and chronological order are confounded; no identical global anchor or CPU frequency/load trace was repeated throughout the session.
Bins contain complete fixture groups centered at their own medians; near-one bin medians are largely a normalization consequence and cannot diagnose session-wide drift. Use within-fixture cycles and jitter levels; no repeated global anchor or CPU-frequency trace exists.

## Size and backend effects

Particle count is 3 x class count. Raw times necessarily reflect that workload difference. Use time / (10,000 x particles) for a descriptive size check; neither that normalization nor graph mean degree fully captures evolving PCC branch costs.

| Backend | Per-particle cost vs nodes, Spearman rho | vs classes | vs mean degree |
| --- | ---: | ---: | ---: |
| numpy | 0.104 | -0.015 | 0.188 |
| numba | 0.198 | -0.442 | 0.669 |
| cython | 0.618 | 0.639 | 0.316 |

Direct paired Numba/Cython ratio is more strongly associated with class/particle count (rho -0.906) than nodes (-0.559); features (-0.149) and mean degree (0.072) are weaker. Larger/high-class workloads tend toward parity. Spambase is large but has two classes and a larger Cython advantage, illustrating why node count alone is insufficient. Classes and particle count cannot be separated here, and all correlations are descriptive over 20 deliberately selected, chronologically confounded datasets; no p-values or causal claims are supplied.

## What is statistically supported?

Exact accounting, deterministic repeatability, and the observed medians/ratios/order patterns are directly verified. Large NumPy/native gaps and consistent Cython advantages in the listed cells are well-supported descriptions of this one session. They are not population-level significance claims, predictions for other machines or proof of identical algorithm trajectories.
Six repetitions replay the same seeded trajectory within a fixture/backend; they are technical timing repetitions. Ten masks share the same transductive dataset, graph and possibly overlapping labeled nodes. Historical seeds 10-19 were already examined in PR #8. Dataset order, label order and rotations are fixed; scheduler/frequency effects can induce dependence. Treating 60 records as iid for a t-test, binomial test, ordinary bootstrap confidence interval or regression p-value would give unjustified precision. There are no independent repeated-machine sessions in this campaign.
No inferential p-values or population confidence intervals are reported. IQRs describe observations; broad differences with consistency across masks/cycles are distinguished from close, order-sensitive comparisons. Shared-input pairing reduces confounding without creating independence.

## Chronological label-block changes (confounded)

Compare median timing for label seeds 17-19 with 10-12 within each dataset/backend. These are different masks measured at different times; the ratio cannot separate workload from environmental drift. Unlike centered complete-fixture time bins, this check can show larger between-fixture changes, but cannot attribute their cause.

| Dataset | NumPy late/early mask-block ratio | Numba | Cython |
| --- | ---: | ---: | ---: |
| wine | 1.003 | 1.001 | 1.003 |
| digits | 1.468 | 1.089 | 1.115 |
| iris | 1.016 | 1.073 | 0.980 |
| breast_cancer | 0.915 | 0.858 | 0.901 |
| balancescale | 1.148 | 1.233 | 1.368 |
| germancredit | 0.983 | 1.004 | 1.137 |
| diabetes | 0.933 | 0.795 | 0.937 |
| sonar | 0.926 | 0.846 | 0.865 |
| glass | 1.075 | 1.185 | 1.315 |
| vehicle | 0.814 | 1.035 | 0.960 |
| ionosphere | 0.944 | 0.828 | 0.820 |
| pendigits | 1.024 | 1.182 | 1.052 |
| satimage | 1.072 | 1.214 | 1.298 |
| spambase | 1.075 | 0.922 | 0.870 |
| pageblocks | 1.124 | 1.153 | 1.063 |
| banknote | 0.880 | 1.018 | 0.991 |
| bloodtransfusion | 0.862 | 1.068 | 1.067 |
| yeast | 0.897 | 0.899 | 1.125 |
| segment | 1.158 | 1.007 | 1.242 |
| qsarbiodeg | 0.779 | 0.859 | 0.878 |

Digits' NumPy late-mask block is 46.8% slower than its early block; BalanceScale native blocks are 23.3%/36.8% slower; QSAR late blocks are 22.1%/14.1%/12.2% faster for NumPy/Numba/Cython. These larger changes qualify any statement that the whole campaign was temporally stable. A repeated identical anchor and counterbalanced mask/dataset order across sessions would help distinguish environmental effects from trajectories. Current data support neither a unique cause nor a global monotonic drift claim.

## Aggregate recorded fit component times

| Backend | Initialization/dispatch s | Propagation s | Postprocess s | Measured fits s |
| --- | ---: | ---: | ---: | ---: |
| numpy | 0.185006 | 2172.192211 | 0.186586 | 2172.563803 |
| numba | 0.201679 | 11.094768 | 0.218299 | 11.514746 |
| cython | 0.194464 | 9.607926 | 0.229087 | 10.031477 |

Ratio of total propagation times: NumPy/Numba 195.79x; NumPy/Cython 226.08x. These are workload-weighted aggregates, not equal-dataset speedup averages.


## Separate sensitivity analysis and remaining limits

The completed [432-fit permutation follow-up](performance_order_sensitivity.md)
uses the same inputs/parameters and three consecutive processes. It supports
a consistent descriptive Wine advantage for Cython and a modest PageBlocks
median advantage; Digits and Satimage remain close and variable. PageBlocks'
order-direction effect changes across sessions. These results qualify, without
pooling or replacing, the primary campaign. Consecutive sessions do not establish
stability across different periods of machine use.

Sampled minimum free RAM was 53.99 GiB and free disk 713.52 GiB, not peak
process memory. No correctness failure, fallback or resource stop was observed.
All 600 groups retain six repetitions, including 280 groups flagged descriptively
by IQR/median >20% or max/min >2. Typical within-fixture IQR/median is
13.40% NumPy, 23.21% Numba and 24.68% Cython. Passing resource/drift monitors
does not establish a stable CPU frequency or low timing variability.

Early stopping answers a different question: time to the unchanged criterion,
effective iterations, stopping/censor fractions and resulting accuracy. This
fixed experiment cannot answer it. Tenfold scaling of measured propagation
gives a **6.09-hour fixed-throughput equivalent** at 100,000 iterations, not a
prediction or validated upper bound for early stopping; check overhead and
evolving branch/state costs are unknown. No early-stopping or 500,000-iteration
campaign was executed. The completed fixed report needs no further benchmarks.

Historical MATLAB/MEX results remain intact and use different machines,
protocols and timing boundaries; the new Python results do not replace them.
See the [archive and reproduction guide](performance_archiving.md) for exact
raw/source provenance, integrity checks and historical command templates.
