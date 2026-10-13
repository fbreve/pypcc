# Six-permutation consecutive-session follow-up

All 432 measured fits and 144 warm-ups accounted for across three fresh consecutive native Windows processes. Eight input fixtures (24 saved snapshots) match the original campaign. All final state hashes match the original 10,000-iteration trajectories per backend. All six permutations occur once per fixture/session, every position twice, each native pair direction three times. All outliers retained.

Sessions are consecutive by explicit revised authorization, not distinct periods of machine use. This probes short-term session/order repeatability, not temporal stability or independent-session population inference.

The separate session measurements and diagnostics are preserved in the
[published benchmark Release](https://github.com/fbreve/pypcc/releases/tag/python-fixed-benchmarks-2026-10-13),
alongside the unpooled primary campaign. See the [archive guide](performance_archiving.md)
for verified asset checksums and provenance.

## Session timing and environment

| Session | Actual start UTC | Actual end UTC | Wall seconds | Dataset order |
| --- | --- | --- | ---: | --- |
| session-1 | 2026-10-13T16:50:00.938379+00:00 | 2026-10-13T16:52:05.420929+00:00 | 124.483 | digits, satimage, pageblocks, wine |
| session-2 | 2026-10-13T16:52:05.475051+00:00 | 2026-10-13T16:54:13.181799+00:00 | 127.706 | digits, satimage, wine, pageblocks |
| session-3 | 2026-10-13T16:54:13.234648+00:00 | 2026-10-13T16:56:20.505211+00:00 | 127.271 | wine, digits, satimage, pageblocks |
Inter-session 1/2 gap: 0.054 s (launcher housekeeping only; no artificial sleep).
Inter-session 2/3 gap: 0.053 s (launcher housekeeping only; no artificial sleep).

Same native i7-14700K/Windows 11 and pypcc-dev dependency versions/native binary as original campaign; production sources unchanged. Source commit a1a2460 adds only opt-in scheduling/metadata plus docs/tests. Full permutation defaults are not enabled for ordinary campaigns. Power/affinity not changed. External samples record free RAM/disk, aggregate system CPU utilization and affinity; they do not identify executing core or actual frequency. Original binary compiler/build log remains unavailable.

## Propagation median (IQR), milliseconds

| Session | Dataset | NumPy | Numba | Cython | Paired Numba/Cython median (Q1-Q3) |
| --- | --- | ---: | ---: | ---: | --- |
| session-1 | wine | 1083.743 (138.583) | 5.303 (0.257) | 3.720 (0.433) | 1.422 (1.313-1.441) |
| session-1 | digits | 4144.829 (665.123) | 19.781 (2.784) | 17.375 (4.582) | 1.095 (0.938-1.227) |
| session-1 | satimage | 2662.044 (198.304) | 11.622 (3.471) | 12.417 (5.310) | 1.075 (0.795-1.139) |
| session-1 | pageblocks | 2051.559 (298.988) | 8.762 (3.250) | 8.517 (3.655) | 1.163 (1.087-1.179) |
| session-2 | wine | 1265.826 (229.517) | 7.084 (2.504) | 4.890 (1.818) | 1.424 (1.329-1.571) |
| session-2 | digits | 3655.070 (536.346) | 18.388 (3.878) | 16.904 (3.399) | 1.059 (1.040-1.147) |
| session-2 | satimage | 2792.526 (592.467) | 11.469 (6.391) | 11.234 (2.679) | 1.066 (1.012-1.090) |
| session-2 | pageblocks | 2312.699 (211.279) | 9.498 (3.534) | 7.729 (1.121) | 1.171 (1.098-1.446) |
| session-3 | wine | 1187.475 (351.763) | 5.336 (0.702) | 3.711 (0.105) | 1.437 (1.383-1.462) |
| session-3 | digits | 3910.590 (903.342) | 17.941 (4.864) | 16.745 (7.519) | 1.029 (0.883-1.096) |
| session-3 | satimage | 2719.826 (255.407) | 15.728 (6.302) | 13.872 (7.137) | 1.061 (0.989-1.205) |
| session-3 | pageblocks | 2116.671 (351.247) | 8.967 (2.086) | 7.476 (0.451) | 1.164 (1.139-1.228) |

Twelve observations per dataset/backend/session combine two label fixtures and six technical repetitions. Ratio >1 favors Cython. Quartile ranges are descriptive, not confidence intervals. Paired-ratio medians need not equal ratios of marginal time medians. Raw per-seed paired ratios and all timing extrema remain in analysis.json and JSONL.

## Native order diagnostics

| Session | Dataset | Numba precedes Cython | Cython precedes Numba | NumPy position 1 | NumPy position 2 | NumPy position 3 | Cython faster /12 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| session-1 | wine | 1.428 | 1.422 | 1.313 | 1.432 | 1.431 | 12 |
| session-1 | digits | 1.064 | 1.196 | 1.196 | 1.018 | 1.088 | 8 |
| session-1 | satimage | 0.992 | 1.084 | 0.947 | 0.800 | 1.110 | 7 |
| session-1 | pageblocks | 1.170 | 1.121 | 1.170 | 1.125 | 1.174 | 10 |
| session-2 | wine | 1.375 | 1.511 | 1.519 | 1.382 | 1.442 | 12 |
| session-2 | digits | 1.055 | 1.082 | 1.175 | 0.932 | 1.058 | 10 |
| session-2 | satimage | 1.066 | 1.044 | 1.031 | 1.081 | 1.057 | 11 |
| session-2 | pageblocks | 1.094 | 1.470 | 1.132 | 1.259 | 1.362 | 11 |
| session-3 | wine | 1.460 | 1.411 | 1.444 | 1.719 | 1.389 | 12 |
| session-3 | digits | 1.017 | 1.031 | 1.055 | 0.998 | 0.936 | 7 |
| session-3 | satimage | 1.069 | 1.057 | 1.053 | 1.529 | 1.043 | 9 |
| session-3 | pageblocks | 1.164 | 1.139 | 1.164 | 1.155 | 1.171 | 10 |

Each direction has six paired ratios per dataset/session; each NumPy-position stratum four. These tiny strata are descriptive, not independently randomized outcome samples. Schedule permutation order and dataset-block order are recorded deterministic shuffles; case/RNG/graph semantics remain fixed. Compare matching seeds rather than treating the original ten-seed marginals as identical follow-up workload mixtures.

## Preparation and monitoring

| Session | Dataset preparation s | Graph s | Minimum sampled free RAM GiB | CPU busy median (sampled system %) | Affinity masks |
| --- | ---: | ---: | ---: | ---: | --- |
| session-1 | 0.216853 | 0.232301 | 58.85 | 14.2 | ['0xfffffff'] |
| session-2 | 0.293838 | 0.240516 | 58.29 | 16.4 | ['0xfffffff'] |
| session-3 | 0.217248 | 0.286125 | 58.21 | 17.9 | ['0xfffffff'] |

CPU samples aggregate the whole machine, not benchmark CPU attribution. Frequency and core migration were not measured, so timing causes cannot be diagnosed from utilization alone. Four graph preparations/session were reused across label seeds; fit initialization/dispatch and propagation boundaries remain unchanged. Raw full timing breakdowns are in the per-session summaries/manifests and measurement records.

## Statistical and provenance limits

Within-backend trajectories replay exactly. Two masks share one graph/dataset; consecutive sessions share the same machine-use period. No IID claims, inferential p-values, or population confidence intervals are warranted. No result is excluded because it reverses a preferred ranking. Near-parity effects require the per-seed and direction qualifications shown above.

No early-stopping campaign or change to original campaign artifacts occurred. This sensitivity analysis remains separate from the primary campaign.


## Interpretation and recommendation

Wine has a consistent descriptive Cython advantage: paired Numba/Cython medians
1.422, 1.424 and 1.437; Cython is faster in all 36 pairs. All six session/seed
medians exceed 1.40. Absolute native times vary: Numba 5.303/7.084/5.336 ms,
Cython 3.720/4.890/3.711 ms. Pairing reduces some common disturbance but does
not establish its cause.

Digits remains close/unresolved: paired medians 1.095/1.059/1.029. Session 1
seed medians disagree (1.234 and 0.910); paired Q1-Q3 spans parity in sessions
1 and 3. Satimage also remains close: 1.075/1.066/1.061, with wide paired
ranges and direction-sensitive values. A small native ranking is not settled.

PageBlocks has a modest repeatable median Cython advantage (1.163/1.171/1.164),
all six session/seed medians above 1.11, and 31/36 pairs favoring Cython.
Order/noise qualify that result: session 2 gives 1.094 when Numba precedes
Cython and 1.470 in the reverse direction. This reverses the original campaign's
directional pattern. Satimage direction ratios are 0.992/1.084, 1.066/1.044,
and 1.069/1.057 in sessions 1-3. The earlier directional effect is not stable.
Full counterbalancing improves symmetry; it does not eliminate timing noise.

Normalized-by-fixture Numba position medians are [0.986,1.072,1.001],
[0.972,1.125,0.995], [1.029,1.010,0.987] in sessions 1-3. A slower middle
position in sessions 1 and 2 is attenuated in session 3. Cython position medians
are [1.004,0.991,0.998], [1.050,0.982,0.977], [1.044,0.996,0.994]. These
describe only eight fixtures/session; they do not establish a general mechanism.
All repetition, seed, direction and NumPy-position cells remain in analysis.json.

Recommend finalizing the original 20-dataset fixed report for review with its
original ten-seed/rotation data and variability qualifications intact. Add this
fully counterbalanced follow-up as a separate sensitivity appendix. Do not
replace or pool original measurements: the follow-up has two masks, another
schedule and three consecutive processes. Wine's advantage is consistent within
observed conditions; PageBlocks' advantage is modest; Digits/Satimage are close.
There is no reliable universal native winner.

Do not claim independent periods, population statistical significance, or
long-term stability. No distinct machine-use periods were sampled. Power,
affinity, compiler/build provenance limits and unsampled frequency/migration
must remain explicit. Absolute native times move despite identical states.
A broader stability claim would require separately authorized genuine periods.

The README now contains a separate Python fixed-propagation median/IQR table
linked to the primary report and this sensitivity analysis. Historical
MATLAB/MEX/Python tables and timing scopes remain intact. The archive Release
was subsequently published with approval; merging and marking the draft ready
still require maintainer approval.
No early-stopping run or 500,000 comparison occurred.

## Actual local times (America/Sao_Paulo, 2026-10-13)

- session-1: 2026-10-13T13:50:00.938379-03:00 to 2026-10-13T13:52:05.420929-03:00, 124.483 s.
- session-2: 2026-10-13T13:52:05.475051-03:00 to 2026-10-13T13:54:13.181799-03:00, 127.706 s.
- session-3: 2026-10-13T13:54:13.234648-03:00 to 2026-10-13T13:56:20.505211-03:00, 127.271 s.


The [primary report](performance_fixed_results.md) and [archive guide](performance_archiving.md) provide raw-data provenance and offline reproduction commands. The [session table](performance_results/sensitivity_sessions.csv) is also committed. Diagnostic PNG/PDF and the full per-seed/order analysis are retained in the archive.
