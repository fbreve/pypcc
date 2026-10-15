# Full early-stopping campaign: seed-10 stage results

The authorized first stage completed **360 measured fits, 120 warm-ups and 20
fixtures**, with all six balanced backend permutations. Seeds 11–19 were not
executed: the immutable campaign remains globally partial at **360/3,600**, with
the seed-10 stage complete and 3,240 unauthorized tasks pending.

Measured source: `0bda57de73410d1b306c958231478517ae54f60b`, clean at launch; all four
[Windows/Linux × Python 3.10/3.12 implementation CI jobs](https://github.com/fbreve/pypcc/actions/runs/38502813665)
passed before execution. The [prepared plan](performance_full_campaign_plan.md)
and its original estimates are preserved. The exact documented start command was
used in the validated environment with the verified frozen historical cache.

Protocol: seed 10, three labels/class, frozen full-data preprocessing and ordered
symmetric Euclidean `k_nn=10` graphs, `n_jobs=1`; `p_grd=.5`, `delta_v=.1`,
`deltap=1`, `dexp=2`, `es_chk=2000`, original early stopping, cap 500,000.
NumPy and Numba reset independent RNGs to 0; Cython resets xorshift32 to
123456789. Each backend occupies each execution position twice per fixture.

## Completion and correctness

There were exactly **360 started attempts**, one session, zero failed or
ambiguous attempts, zero retries, zero recovery actions, zero overwrites/repeated
completed tasks, and zero excluded measurements or outliers. Native exit code was
zero, with no resource stop or runtime cutoff. The session/global `partial`
status describes the unexecuted nine seeds, not an interrupted seed-10 stage.

Strict offline audit passed the plan pin, complete specification/task identities,
event chain, session/warm-up/result checksums, exact six-permutation coverage and
stage summaries. It rehashed all 20 fixtures and their raw/scaled data, ordered
graphs and historical label masks, matching every array/fixture hash to PR #16.
The original harness/backend runtime invariants, labeled-node preservation,
finite valid dominance, particle state, backend identity and Numba compilation
guards ran for every measurement. All six terminal state hashes, iterations,
stopping reasons and accuracy repeated exactly within each fixture/backend;
cross-backend equality was not imposed. Measured production/harness source and
Cython binary identities were independently verified against retained snapshots.
All **1148** prior fixed/sensitivity/pilot/calibration files
were rehashed unchanged.
All sixty fixture/backend groups also exactly reproduced their earlier seed-10
calibration terminal state hashes, stopping counts/reasons and accuracy.

## Stopping, censoring and accuracy

**324 successful early stops; 36 capped/censored fits
(10.0%)**, across 58,429,704 measured iterations.
`max_iter` is a completed censored observation, never evidence of convergence.
All six repetitions follow the same trajectory per backend; their repetition
counts quantify timing variation, not independent convergence samples.

| Dataset | Backend | Iterations per fit | Reason | Censored / 6 | Unlabeled accuracy (%) |
| --- | --- | ---: | --- | ---: | ---: |
| wine | numpy | 9,701 | early_stop | 0 | 95.8580 |
| wine | numba | 11,221 | early_stop | 0 | 94.0828 |
| wine | cython | 10,611 | early_stop | 0 | 94.6746 |
| digits | numpy | 81,941 | early_stop | 0 | 89.4737 |
| digits | numba | 87,691 | early_stop | 0 | 89.1907 |
| digits | cython | 85,051 | early_stop | 0 | 89.3605 |
| iris | numpy | 10,811 | early_stop | 0 | 95.0355 |
| iris | numba | 8,191 | early_stop | 0 | 95.0355 |
| iris | cython | 10,441 | early_stop | 0 | 94.3262 |
| breast_cancer | numpy | 38,161 | early_stop | 0 | 92.7176 |
| breast_cancer | numba | 35,031 | early_stop | 0 | 91.6519 |
| breast_cancer | cython | 47,141 | early_stop | 0 | 94.1385 |
| balancescale | numpy | 48,041 | early_stop | 0 | 60.2273 |
| balancescale | numba | 34,391 | early_stop | 0 | 59.2532 |
| balancescale | cython | 60,911 | early_stop | 0 | 60.5519 |
| germancredit | numpy | 157,941 | early_stop | 0 | 51.1066 |
| germancredit | numba | 148,881 | early_stop | 0 | 55.3320 |
| germancredit | cython | 106,611 | early_stop | 0 | 52.5151 |
| diabetes | numpy | 82,731 | early_stop | 0 | 64.5669 |
| diabetes | numba | 73,971 | early_stop | 0 | 64.6982 |
| diabetes | cython | 51,271 | early_stop | 0 | 64.0420 |
| sonar | numpy | 16,441 | early_stop | 0 | 70.7921 |
| sonar | numba | 21,111 | early_stop | 0 | 71.2871 |
| sonar | cython | 11,231 | early_stop | 0 | 69.8020 |
| glass | numpy | 8,671 | early_stop | 0 | 38.7755 |
| glass | numba | 5,971 | early_stop | 0 | 35.7143 |
| glass | cython | 7,681 | early_stop | 0 | 40.3061 |
| vehicle | numpy | 39,101 | early_stop | 0 | 49.8801 |
| vehicle | numba | 93,311 | early_stop | 0 | 50.5995 |
| vehicle | cython | 70,961 | early_stop | 0 | 50.7194 |
| ionosphere | numpy | 41,901 | early_stop | 0 | 68.9855 |
| ionosphere | numba | 27,511 | early_stop | 0 | 73.0435 |
| ionosphere | cython | 38,111 | early_stop | 0 | 72.4638 |
| pendigits | numpy | 500,000 | max_iter | 6 | 89.2355 |
| pendigits | numba | 500,000 | max_iter | 6 | 84.8477 |
| pendigits | cython | 500,000 | max_iter | 6 | 89.9015 |
| satimage | numpy | 386,091 | early_stop | 0 | 77.1366 |
| satimage | numba | 439,901 | early_stop | 0 | 77.3706 |
| satimage | cython | 489,151 | early_stop | 0 | 76.3880 |
| spambase | numpy | 500,000 | max_iter | 6 | 79.1295 |
| spambase | numba | 499,171 | early_stop | 0 | 85.7671 |
| spambase | cython | 500,000 | max_iter | 6 | 84.1567 |
| pageblocks | numpy | 418,541 | early_stop | 0 | 31.8981 |
| pageblocks | numba | 494,901 | early_stop | 0 | 27.5009 |
| pageblocks | cython | 500,000 | max_iter | 6 | 24.7527 |
| banknote | numpy | 234,871 | early_stop | 0 | 71.0835 |
| banknote | numba | 301,481 | early_stop | 0 | 79.6486 |
| banknote | cython | 227,291 | early_stop | 0 | 74.6706 |
| bloodtransfusion | numpy | 103,731 | early_stop | 0 | 60.5121 |
| bloodtransfusion | numba | 77,991 | early_stop | 0 | 55.6604 |
| bloodtransfusion | cython | 80,981 | early_stop | 0 | 64.4205 |
| yeast | numpy | 101,801 | early_stop | 0 | 37.1389 |
| yeast | numba | 100,931 | early_stop | 0 | 35.9697 |
| yeast | cython | 107,111 | early_stop | 0 | 37.7579 |
| segment | numpy | 284,941 | early_stop | 0 | 73.9624 |
| segment | numba | 277,401 | early_stop | 0 | 73.7003 |
| segment | cython | 181,601 | early_stop | 0 | 74.0061 |
| qsarbiodeg | numpy | 102,771 | early_stop | 0 | 63.4890 |
| qsarbiodeg | numba | 138,691 | early_stop | 0 | 63.5844 |
| qsarbiodeg | cython | 106,191 | early_stop | 0 | 67.3975 |

Accuracy differences between backends reflect distinct RNG trajectories and
terminal states, with differing stopping counts/censoring. They are not timing
errors or grounds to force state equality. Dataset spreads follow:

| Dataset | Censored / 18 | Propagation total (s) | Total fit (s) | Accuracy spread (percentage points) |
| --- | ---: | ---: | ---: | ---: |
| wine | 0 | 14.759930 | 14.765477 | 1.7751 |
| digits | 0 | 396.521118 | 396.531512 | 0.2830 |
| iris | 0 | 6.527799 | 6.531225 | 0.7092 |
| breast_cancer | 0 | 16.007552 | 16.012340 | 2.4867 |
| balancescale | 0 | 29.773951 | 29.778214 | 1.2987 |
| germancredit | 0 | 66.049851 | 66.054618 | 4.2254 |
| diabetes | 0 | 34.507911 | 34.512435 | 0.6562 |
| sonar | 0 | 6.783875 | 6.787897 | 1.4851 |
| glass | 0 | 10.176083 | 10.180171 | 4.5918 |
| vehicle | 0 | 32.203199 | 32.207983 | 0.8393 |
| ionosphere | 0 | 17.453828 | 17.457688 | 4.0580 |
| pendigits | 18 | 1272.799889 | 1272.823873 | 5.0538 |
| satimage | 0 | 527.881272 | 527.895621 | 0.9825 |
| spambase | 12 | 228.777976 | 228.786129 | 6.6376 |
| pageblocks | 6 | 466.528304 | 466.540197 | 7.1455 |
| banknote | 0 | 104.338682 | 104.344261 | 8.5652 |
| bloodtransfusion | 0 | 46.356258 | 46.361968 | 8.7601 |
| yeast | 0 | 221.070610 | 221.080439 | 1.7882 |
| segment | 0 | 431.897094 | 431.909039 | 0.3058 |
| qsarbiodeg | 0 | 45.500081 | 45.506125 | 3.9085 |

## Timing medians and variability

Each cell gives **median; IQR; [minimum, maximum] in seconds**, computed within
one fixture/backend over six measured repetitions using linear percentiles.
Warm-ups and prepared graphs remain outside measured fit times. All raw timings,
including extremes, remain available; no trimming or outlier selection occurred.

| Dataset | Backend | Propagation: median; IQR; range (s) | Total fit: median; IQR; range (s) |
| --- | --- | --- | --- |
| wine | numpy | 2.671456; 0.048846; [1.263233, 2.711844] | 2.671750; 0.048821; [1.263430, 2.712124] |
| wine | numba | 0.009422; 0.000601; [0.009176, 0.010610] | 0.009767; 0.000638; [0.009493, 0.011057] |
| wine | cython | 0.006962; 0.000307; [0.004472, 0.007823] | 0.007255; 0.000350; [0.004697, 0.008203] |
| digits | numpy | 70.786347; 19.406931; [50.024275, 75.971788] | 70.786897; 19.407001; [50.024647, 75.972472] |
| digits | numba | 0.239070; 0.127028; [0.175127, 0.316939] | 0.239590; 0.127274; [0.175550, 0.317633] |
| digits | cython | 0.284694; 0.132265; [0.214073, 0.360725] | 0.285405; 0.132426; [0.214462, 0.361425] |
| iris | numpy | 1.077640; 0.018032; [1.061506, 1.098938] | 1.077825; 0.018035; [1.061697, 1.099113] |
| iris | numba | 0.004432; 0.000107; [0.004375, 0.004856] | 0.004622; 0.000114; [0.004546, 0.005055] |
| iris | cython | 0.004200; 0.000005; [0.004134, 0.004622] | 0.004389; 0.000055; [0.004315, 0.004854] |
| breast_cancer | numpy | 2.641217; 0.018042; [2.620002, 2.653043] | 2.641406; 0.017639; [2.620183, 2.653245] |
| breast_cancer | numba | 0.015384; 0.000558; [0.014687, 0.017072] | 0.015614; 0.000569; [0.014888, 0.017322] |
| breast_cancer | cython | 0.015111; 0.000659; [0.014315, 0.015350] | 0.015339; 0.000608; [0.014522, 0.015954] |
| balancescale | numpy | 4.909889; 0.033102; [4.843241, 4.955510] | 4.910095; 0.033114; [4.843471, 4.955718] |
| balancescale | numba | 0.020157; 0.003116; [0.019558, 0.024114] | 0.020400; 0.003124; [0.019804, 0.024364] |
| balancescale | cython | 0.032402; 0.001911; [0.031164, 0.035099] | 0.032656; 0.001886; [0.031413, 0.035362] |
| germancredit | numpy | 10.886543; 0.135117; [10.802406, 11.018721] | 10.886782; 0.135127; [10.802650, 11.018955] |
| germancredit | numba | 0.072087; 0.004245; [0.068649, 0.075015] | 0.072382; 0.004279; [0.068926, 0.075294] |
| germancredit | cython | 0.039683; 0.001981; [0.039514, 0.042679] | 0.039957; 0.002004; [0.039763, 0.042960] |
| diabetes | numpy | 5.697104; 0.068697; [5.640249, 5.791152] | 5.697314; 0.068679; [5.640455, 5.791377] |
| diabetes | numba | 0.032967; 0.000283; [0.031787, 0.034244] | 0.033205; 0.000295; [0.032016, 0.034504] |
| diabetes | cython | 0.017750; 0.001377; [0.016663, 0.019113] | 0.018121; 0.001455; [0.016876, 0.019387] |
| sonar | numpy | 1.119819; 0.006843; [1.097469, 1.127499] | 1.120014; 0.006776; [1.098123, 1.127727] |
| sonar | numba | 0.008878; 0.000228; [0.008813, 0.009130] | 0.009102; 0.000190; [0.009000, 0.009376] |
| sonar | cython | 0.003294; 0.000054; [0.003263, 0.005173] | 0.003470; 0.000095; [0.003416, 0.005402] |
| glass | numpy | 1.681445; 0.011972; [1.665236, 1.701907] | 1.681638; 0.011977; [1.665417, 1.702120] |
| glass | numba | 0.006392; 0.000330; [0.006090, 0.006760] | 0.006617; 0.000355; [0.006269, 0.007015] |
| glass | cython | 0.007814; 0.000579; [0.007368, 0.013558] | 0.008083; 0.000536; [0.007579, 0.013908] |
| vehicle | numpy | 5.237952; 0.061040; [5.164474, 5.288808] | 5.238183; 0.061034; [5.164733, 5.289029] |
| vehicle | numba | 0.072538; 0.001615; [0.070580, 0.073759] | 0.072816; 0.001631; [0.070855, 0.074042] |
| vehicle | cython | 0.059657; 0.002148; [0.054819, 0.061387] | 0.059935; 0.002137; [0.055068, 0.061685] |
| ionosphere | numpy | 2.884109; 0.026220; [2.860582, 2.914083] | 2.884285; 0.026189; [2.860777, 2.914293] |
| ionosphere | numba | 0.011932; 0.003399; [0.011717, 0.016756] | 0.012147; 0.003416; [0.011917, 0.016997] |
| ionosphere | cython | 0.011582; 0.000454; [0.011466, 0.012973] | 0.011800; 0.000461; [0.011685, 0.013305] |
| pendigits | numpy | 185.046953; 2.909261; [172.252176, 306.400735] | 185.048114; 2.909080; [172.253331, 306.401960] |
| pendigits | numba | 1.964789; 0.138757; [1.891597, 2.377084] | 1.965964; 0.139022; [1.892724, 2.379565] |
| pendigits | cython | 6.602622; 0.291369; [6.463805, 6.957325] | 6.603870; 0.291595; [6.465184, 6.958561] |
| satimage | numpy | 85.187775; 0.637823; [84.460366, 85.503676] | 85.188493; 0.637561; [84.461027, 85.505303] |
| satimage | numba | 0.895375; 0.033257; [0.866292, 0.914528] | 0.896052; 0.033103; [0.866951, 0.915183] |
| satimage | cython | 2.044454; 0.049927; [1.989272, 2.121207] | 2.045300; 0.050026; [1.990125, 2.121884] |
| spambase | numpy | 37.388451; 0.339352; [36.777722, 37.531343] | 37.388811; 0.339349; [36.778160, 37.531801] |
| spambase | numba | 0.383648; 0.008952; [0.372628, 0.390023] | 0.384056; 0.008973; [0.373158, 0.390547] |
| spambase | cython | 0.456240; 0.026643; [0.428491, 0.492439] | 0.456764; 0.026612; [0.429036, 0.493032] |
| pageblocks | numpy | 75.515932; 0.467551; [75.215323, 76.019301] | 75.516537; 0.467674; [75.215859, 76.019833] |
| pageblocks | numba | 0.751829; 0.034398; [0.727590, 0.849568] | 0.752628; 0.034511; [0.728289, 0.850260] |
| pageblocks | cython | 1.441572; 0.016698; [1.374421, 1.470909] | 1.442237; 0.016665; [1.374987, 1.471499] |
| banknote | numpy | 17.135841; 0.132344; [17.072931, 17.258506] | 17.136083; 0.132407; [17.073186, 17.258813] |
| banknote | numba | 0.148561; 0.016118; [0.139765, 0.163961] | 0.148902; 0.016031; [0.140057, 0.164254] |
| banknote | cython | 0.089265; 0.006480; [0.084308, 0.092923] | 0.089553; 0.006445; [0.084712, 0.093209] |
| bloodtransfusion | numpy | 7.717565; 0.183525; [7.386543, 7.880822] | 7.717778; 0.183533; [7.386780, 7.881049] |
| bloodtransfusion | numba | 0.040748; 0.001302; [0.033605, 0.044149] | 0.041067; 0.001623; [0.033919, 0.044496] |
| bloodtransfusion | cython | 0.030250; 0.003831; [0.026686, 0.033799] | 0.030558; 0.003825; [0.027311, 0.034167] |
| yeast | numpy | 36.303689; 0.380730; [35.747281, 37.045781] | 36.304201; 0.380640; [35.747758, 37.046216] |
| yeast | numba | 0.225240; 0.017560; [0.210580, 0.253035] | 0.226127; 0.017636; [0.210977, 0.253494] |
| yeast | cython | 0.293777; 0.016074; [0.283058, 0.314950] | 0.294375; 0.016034; [0.283628, 0.315513] |
| segment | numpy | 71.146816; 0.224206; [70.835032, 71.436662] | 71.147402; 0.224245; [70.835444, 71.437643] |
| segment | numba | 0.469841; 0.072638; [0.444327, 0.530713] | 0.470539; 0.072178; [0.444746, 0.531227] |
| segment | cython | 0.371367; 0.014227; [0.354789, 0.384217] | 0.372230; 0.013994; [0.355455, 0.384899] |
| qsarbiodeg | numpy | 7.462624; 0.176663; [7.307180, 7.691969] | 7.462870; 0.176708; [7.307380, 7.692243] |
| qsarbiodeg | numba | 0.068428; 0.005755; [0.063487, 0.076102] | 0.068728; 0.005733; [0.063816, 0.076695] |
| qsarbiodeg | cython | 0.045534; 0.002471; [0.042890, 0.050437] | 0.045983; 0.002500; [0.043182, 0.050711] |

Largest observed absolute within-group total-fit ranges follow. These are
descriptive variability observations, not exclusions or diagnosed causes.
Machine load, scheduling and power/thermal behavior were not controlled; the
monitor cannot establish a cause. Terminal states remained repeatable.

| Dataset | Backend | Fit range width (s) | Maximum / minimum | IQR / median |
| --- | --- | ---: | ---: | ---: |
| pendigits | numpy | 134.148629 | 1.7788 | 0.0157 |
| digits | numpy | 25.947825 | 1.5187 | 0.2742 |
| wine | numpy | 1.448694 | 2.1466 | 0.0183 |
| yeast | numpy | 1.298458 | 1.0363 | 0.0105 |
| satimage | numpy | 1.044276 | 1.0124 | 0.0075 |

Backend aggregates are sums over the same twenty heterogeneous datasets and
120 fits each, not pooled within-workload medians. Macro accuracy gives each
dataset equal weight; it is descriptive, not a universal backend ranking.

| Backend | Early stops | Censored / 120 | Censor rate (%) | Iterations | Propagation total (s) | Total fit (s) | Macro accuracy (%) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| numpy | 108 | 12 | 10.0 | 19,009,128 | 3871.022939 | 3871.069965 | 68.0502 |
| numba | 114 | 6 | 5.0 | 20,266,494 | 33.309831 | 33.361974 | 68.1969 |
| cython | 102 | 18 | 15.0 | 19,154,082 | 71.582493 | 71.635272 | 68.8176 |

## Runtime comparison and checkpoint overhead

Native process wall duration: **1 h 6 min 27.7 s**
(3987.742911 s). Measured propagation totals
3975.915263 s; measured complete-fit totals 3976.067211 s.

| Pre-execution scenario | Planned duration | Actual / planned | Difference (s) |
| --- | ---: | ---: | ---: |
| observed_stopping | 58 min 35.0 s | 1.1345 | +472.749813 |
| observed_rates_cap | 2 h 28 min 33.6 s | 0.4474 | -4925.883040 |
| historical_median_cap | 2 h 59 min 13.4 s | 0.3708 | -6765.689307 |
| historical_maximum_cap | 3 h 24 min 28.5 s | 0.3250 | -8280.780657 |
| conservative_cap | 5 h 6 min 42.8 s | 0.2167 | -14415.042441 |

These scenarios were planning estimates, not timeouts or confidence intervals.
The stopping-based projection used the same mask with fewer/differently ordered
repetitions; cap scenarios intentionally assume all measurements run to the cap.

The stage used exactly the projected 58,429,704 iterations and 36 censored fits.
Actual native wall time exceeded the stopping forecast by
7.88 minutes
(13.45%).
Timing-rate differences, rather than changed stopping behavior, dominate this
gap. The largest absolute fixture/backend timing forecast differences follow:

| Dataset | Backend | Forecast six-fit total (s) | Actual six-fit total (s) | Difference (s) |
| --- | --- | ---: | ---: | ---: |
| pendigits | numpy | 985.138208 | 1220.567388 | +235.429180 |
| digits | numpy | 159.448565 | 393.344585 | +233.896020 |
| satimage | numpy | 633.536115 | 510.212512 | -123.323603 |
| pageblocks | numpy | 403.755103 | 453.336783 | +49.581680 |
| yeast | numpy | 192.818489 | 217.921426 | +25.102937 |


| Recorded overhead component | Seconds |
| --- | ---: |
| Warm-up fit timers | 1.438578 |
| Dataset/preprocessing/label/graph preparation | 1.399175 |
| First RNG helper call | 0.310896 |
| Measured helper wall spans minus fit timers | 0.254850 |
| Complete-fit validation/result publication windows, total | 1.817858 |
| Native wall less measured fits, warm-up fits, preparation and RNG helper | 8.527051 |
| Native wall outside controller session timer | 2.067436 |

Publication windows: 0.003167; 0.001676; [0.001002, 0.031030] s.
Overhead rows overlap and must not be added together. The residual includes
untimed validation, graph copies/model construction/RNG seeding, checkpoint and
session/ledger I/O, startup and final audit/summary. Publication windows include
post-fit validation and result writes but exclude subsequent completion-event
fsync. **Pure filesystem/checkpoint I/O is not independently timed** by the
established controller. No instrumentation changed production PCC or its timing
boundaries. Offline report audit took 1.074110 s separately,
outside execution wall time.

Resource monitor minimum free RAM: 52.12 GiB;
minimum free disk: 706.74 GiB.
There were no observed severe resource problems, execution errors, interruptions
or retries. Timing variation is explicitly retained and described above.

## Artifacts and recommendation

The [360-row CSV (archived)](performance_early_stopping_archiving.md),
[flat raw measurements (archived)](performance_early_stopping_archiving.md)
and [complete audit/statistics](benchmarks/early_stopping_full_seed10/results/summary.json)
retain all timing/state/stopping/accuracy data. Canonical plan, pin, event/result/
session/warm-up/fixture/summary records and execution/resource logs are committed
under `results/raw/`; the local byte-checksum inventories cover all original
files. Full NPZs, staging evidence and measured-source snapshots remain locally
retained at `.cache/performance/windows-full-early-stopping-01` and its sibling
`windows-full-early-stopping-01-provenance`. Git text normalization may change
committed copies' line endings; these never replace original checkpoint bytes.
No source/NPZ archive or Release asset was published.

A byte-identical point-in-time copy of all 2,504 checkpoint files is retained at
`.cache/performance/windows-full-early-stopping-01-seed10-snapshot`. Use this copy
to reproduce the seed-10 audit after the live root grows in a separately authorized
stage. The original live root remains the only documented execution target.
The read-only audit tool reproduces all scientific data/statistics (excluding its
own elapsed audit duration); an independent CSV check confirmed 360 unique task
IDs, 58,429,704 iterations and 36 censored observations:

```console
python -m tools.analyze_full_seed10 --root .cache/performance/windows-full-early-stopping-01-seed10-snapshot --baseline .cache/performance/windows-fixed-full-c9f512c-01/fixed-10000 --provenance .cache/performance/windows-full-early-stopping-01-provenance --output NEW_AUDIT
```

The output directory must be new. This command validates and summarizes saved
artifacts; it performs no fits, warm-ups, retries or checkpoint mutations.

Seed 10 establishes six-repetition timing variability and deterministic stopping
for every historical dataset/backend. It does not predict seeds 11–19, and its
censored observations do not supply convergence times beyond the cap. Preserve
the complete plan, environment and source/binary identities for any authorized
later stage; review this stage's costs and censoring before selecting another
seed. **Await further authorization.** PR #17 remains a draft; historical fixed
results, README tables, protocol, version and Release remain unchanged.
