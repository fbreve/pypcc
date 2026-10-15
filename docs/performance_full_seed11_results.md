# Full early-stopping campaign: seed-11 results and two-seed audit

The separately authorized seed-11 stage completed **360 measured fits, 120
warm-ups and twenty fixtures** in the existing campaign root. Seeds 10 and 11
are complete: **720 measured fits, 240 warm-ups, forty fixtures**. Seeds 12–19
remain unstarted: **2,880 measured tasks require further authorization**.

The 3,600-task immutable plan and original runtime source, Cython binary,
dependencies and environment match the original campaign. Seed 10's 360 results
passed strict preflight; seed 11 had zero completed, ambiguous or previous
attempts. Every one of the 2,504 original
checkpoint files remained byte-identical after resume. A data/report commit
changed Git HEAD to `50aff0bbe34caa1687857fb897c54eedf2ae2153`; this is recorded per session,
not mistaken for a runtime source change. Original campaign source commit:
`0bda57de73410d1b306c958231478517ae54f60b`. Production and controller source bytes
were not changed or rebuilt for this stage.

The exact command was:

```console
python -m benchmarks.resumable --action resume --stage-seed 11 --root .cache/performance/windows-full-early-stopping-01 --execute
```

No campaign was recreated, no seed-10 result repeated, and no retry was authorized.
Protocol remains twenty frozen historical datasets, three labels/class, original
preprocessing and symmetric ordered Euclidean `k_nn=10` graphs (`n_jobs=1`),
`p_grd=.5`, `delta_v=.1`, `deltap=1`, `dexp=2`, `es_chk=2000`, early stopping and
500,000 cap. Every fixture retains all six balanced backend permutations; each
backend occupies each position twice. NumPy/Numba independently reset to seed 0;
Cython retains xorshift32 seed 123456789. Twenty validated anchor graphs/data
were reused with independently validated/sealed historical seed-11 labels.

## Audit and completion

Strict audit passed **all 720 completed results**, forty fixtures against PR #16
and frozen raw/scaled/ordered-graph/label fingerprints, plan pin/specification,
task identities, event chain, source/binary snapshots, session/warm-up/result
checksums and summaries. There are exactly 720 unique started attempts: 360
per seed, zero failed/ambiguous attempts, retries, recovery actions or overwrites.
All runtime invariants, labeled-node preservation, valid finite dominance and
particle state, backend identity and Numba compilation guards passed. Every
fixture/backend reproduced all six terminal state hashes, iteration counts,
reasons and accuracy across its six repetitions; cross-seed/backend state
equality is not imposed. The twelve earlier seed-11 pilot groups were compared;
differences: **0**.

All 3,684 previous fixed/sensitivity/calibration/
seed-10 provenance/snapshot files remained unchanged, in addition to the
2,504 existing live checkpoint files.
One new uninterrupted session completed with native exit code zero; no resource
stop, runtime cutoff or discarded observations/outliers. Global/session status
`partial` means eight seeds remain, while stage summaries 10 and 11 are complete.

## Runtime, stopping and accuracy

Seed-11 native runtime: **0 h 55 min 34.4 s**
(3334.422492 s), versus seed 10's
1 h 6 min 27.7 s; ratio
0.8362.
Propagation totals 3321.086036 s; complete-fit totals
3321.229328 s. There were **336 early stops and
24 capped/censored fits (6.67%)**,
totaling 54,359,256 iterations. Across both seeds: 660
early stops, 60 censored fits and
112,788,960 iterations. Cap outcomes are valid completed
censored observations, not convergence times beyond the cap.

| Dataset | Backend | Iterations per fit | Reason | Censored / 6 | Unlabeled accuracy (%) |
| --- | --- | ---: | --- | ---: | ---: |
| wine | numpy | 13,221 | early_stop | 0 | 96.4497 |
| wine | numba | 8,911 | early_stop | 0 | 97.0414 |
| wine | cython | 7,321 | early_stop | 0 | 96.4497 |
| digits | numpy | 92,381 | early_stop | 0 | 90.3792 |
| digits | numba | 134,341 | early_stop | 0 | 90.3226 |
| digits | cython | 93,821 | early_stop | 0 | 89.4737 |
| iris | numpy | 7,891 | early_stop | 0 | 88.6525 |
| iris | numba | 7,021 | early_stop | 0 | 85.8156 |
| iris | cython | 6,041 | early_stop | 0 | 88.6525 |
| breast_cancer | numpy | 40,411 | early_stop | 0 | 93.0728 |
| breast_cancer | numba | 46,841 | early_stop | 0 | 92.8952 |
| breast_cancer | cython | 66,231 | early_stop | 0 | 93.7833 |
| balancescale | numpy | 34,221 | early_stop | 0 | 54.0584 |
| balancescale | numba | 41,851 | early_stop | 0 | 56.4935 |
| balancescale | cython | 40,241 | early_stop | 0 | 56.6558 |
| germancredit | numpy | 152,781 | early_stop | 0 | 52.8169 |
| germancredit | numba | 83,171 | early_stop | 0 | 52.8169 |
| germancredit | cython | 84,481 | early_stop | 0 | 55.0302 |
| diabetes | numpy | 64,171 | early_stop | 0 | 63.7795 |
| diabetes | numba | 80,461 | early_stop | 0 | 62.5984 |
| diabetes | cython | 61,761 | early_stop | 0 | 63.2546 |
| sonar | numpy | 17,411 | early_stop | 0 | 62.8713 |
| sonar | numba | 25,281 | early_stop | 0 | 56.4356 |
| sonar | cython | 19,751 | early_stop | 0 | 55.9406 |
| glass | numpy | 9,361 | early_stop | 0 | 45.4082 |
| glass | numba | 9,971 | early_stop | 0 | 46.4286 |
| glass | cython | 13,301 | early_stop | 0 | 42.8571 |
| vehicle | numpy | 57,731 | early_stop | 0 | 48.4412 |
| vehicle | numba | 39,051 | early_stop | 0 | 47.7218 |
| vehicle | cython | 74,291 | early_stop | 0 | 45.0839 |
| ionosphere | numpy | 18,721 | early_stop | 0 | 69.2754 |
| ionosphere | numba | 32,571 | early_stop | 0 | 68.1159 |
| ionosphere | cython | 26,671 | early_stop | 0 | 67.8261 |
| pendigits | numpy | 500,000 | max_iter | 6 | 84.9754 |
| pendigits | numba | 500,000 | max_iter | 6 | 84.8203 |
| pendigits | cython | 500,000 | max_iter | 6 | 83.6982 |
| satimage | numpy | 499,941 | early_stop | 0 | 81.4566 |
| satimage | numba | 455,411 | early_stop | 0 | 81.7374 |
| satimage | cython | 307,411 | early_stop | 0 | 81.6594 |
| spambase | numpy | 421,591 | early_stop | 0 | 75.2992 |
| spambase | numba | 419,561 | early_stop | 0 | 76.9097 |
| spambase | cython | 452,031 | early_stop | 0 | 77.9543 |
| pageblocks | numpy | 500,000 | max_iter | 6 | 44.1920 |
| pageblocks | numba | 451,471 | early_stop | 0 | 36.6435 |
| pageblocks | cython | 433,091 | early_stop | 0 | 34.8113 |
| banknote | numpy | 261,791 | early_stop | 0 | 84.1874 |
| banknote | numba | 241,361 | early_stop | 0 | 69.6925 |
| banknote | cython | 197,151 | early_stop | 0 | 88.0673 |
| bloodtransfusion | numpy | 83,281 | early_stop | 0 | 40.8356 |
| bloodtransfusion | numba | 76,691 | early_stop | 0 | 42.1833 |
| bloodtransfusion | cython | 90,341 | early_stop | 0 | 49.4609 |
| yeast | numpy | 93,391 | early_stop | 0 | 39.0646 |
| yeast | numba | 98,311 | early_stop | 0 | 41.1279 |
| yeast | cython | 99,671 | early_stop | 0 | 35.9697 |
| segment | numpy | 152,901 | early_stop | 0 | 70.8170 |
| segment | numba | 156,571 | early_stop | 0 | 70.7733 |
| segment | cython | 141,351 | early_stop | 0 | 71.8218 |
| qsarbiodeg | numpy | 95,691 | early_stop | 0 | 51.0010 |
| qsarbiodeg | numba | 139,571 | early_stop | 0 | 57.2927 |
| qsarbiodeg | cython | 179,611 | early_stop | 0 | 53.1935 |

Backend totals aggregate the same twenty heterogeneous datasets, 120 fits each;
macro accuracy weights each dataset equally. Different trajectories and
stopping counts prevent interpreting these totals as pure implementation-speed
or universal accuracy rankings.

| Backend | Early stops | Censored / 120 | Rate (%) | Iterations | Propagation total (s) | Total fit (s) | Macro accuracy (%) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| numpy | 108 | 12 | 10.00 | 18,701,328 | 3229.875202 | 3229.921306 | 66.8517 |
| numba | 114 | 6 | 5.00 | 18,290,514 | 29.946318 | 29.994177 | 65.8933 |
| cython | 114 | 6 | 5.00 | 17,367,414 | 61.264517 | 61.313845 | 66.5822 |

| Dataset | Censored / 18 | Propagation total (s) | Total fit (s) | Accuracy spread (percentage points) |
| --- | ---: | ---: | ---: | ---: |
| wine | 0 | 20.313764 | 20.319789 | 0.5917 |
| digits | 0 | 176.707240 | 176.714342 | 0.9055 |
| iris | 0 | 4.396882 | 4.400315 | 2.8369 |
| breast_cancer | 0 | 15.722808 | 15.726696 | 0.8881 |
| balancescale | 0 | 19.633791 | 19.637953 | 2.5974 |
| germancredit | 0 | 58.952976 | 58.957844 | 2.2133 |
| diabetes | 0 | 25.055280 | 25.059562 | 1.1811 |
| sonar | 0 | 6.621166 | 6.624251 | 6.9307 |
| glass | 0 | 10.327419 | 10.331050 | 3.5714 |
| vehicle | 0 | 43.776001 | 43.780510 | 3.3573 |
| ionosphere | 0 | 7.278212 | 7.281620 | 1.4493 |
| pendigits | 18 | 1053.802852 | 1053.829947 | 1.2771 |
| satimage | 0 | 607.467385 | 607.481149 | 0.2807 |
| spambase | 0 | 178.170153 | 178.179297 | 2.6551 |
| pageblocks | 6 | 509.194885 | 509.206445 | 9.3807 |
| banknote | 0 | 106.626150 | 106.631765 | 18.3748 |
| bloodtransfusion | 0 | 33.790164 | 33.795031 | 8.6253 |
| yeast | 0 | 188.871494 | 188.879569 | 5.1582 |
| segment | 0 | 215.002108 | 215.011129 | 1.0485 |
| qsarbiodeg | 0 | 39.375306 | 39.381064 | 6.2917 |

## Seed-10 versus seed-11 comparison

Each entry is deterministic across its six timing repetitions. A changed label
mask can change stopping, censoring and accuracy; repetitions are not six
independent convergence samples. The following comparison covers every group:

| Dataset | Backend | Seed 10: iterations / reason | Seed 11: iterations / reason | Accuracy 10 → 11 (%) | Accuracy change (pp) | Fit median 11 / 10 |
| --- | --- | --- | --- | --- | ---: | ---: |
| wine | numpy | 9,701 / early_stop | 13,221 / early_stop | 95.8580 → 96.4497 | +0.5917 | 1.3500 |
| wine | numba | 11,221 / early_stop | 8,911 / early_stop | 94.0828 → 97.0414 | +2.9586 | 0.7948 |
| wine | cython | 10,611 / early_stop | 7,321 / early_stop | 94.6746 → 96.4497 | +1.7751 | 0.7254 |
| digits | numpy | 81,941 / early_stop | 92,381 / early_stop | 89.4737 → 90.3792 | +0.9055 | 0.4091 |
| digits | numba | 87,691 / early_stop | 134,341 / early_stop | 89.1907 → 90.3226 | +1.1319 | 1.1466 |
| digits | cython | 85,051 / early_stop | 93,821 / early_stop | 89.3605 → 89.4737 | +0.1132 | 0.8243 |
| iris | numpy | 10,811 / early_stop | 7,891 / early_stop | 95.0355 → 88.6525 | -6.3830 | 0.6741 |
| iris | numba | 8,191 / early_stop | 7,021 / early_stop | 95.0355 → 85.8156 | -9.2199 | 0.8777 |
| iris | cython | 10,441 / early_stop | 6,041 / early_stop | 94.3262 → 88.6525 | -5.6738 | 0.6266 |
| breast_cancer | numpy | 38,161 / early_stop | 40,411 / early_stop | 92.7176 → 93.0728 | +0.3552 | 0.9739 |
| breast_cancer | numba | 35,031 / early_stop | 46,841 / early_stop | 91.6519 → 92.8952 | +1.2433 | 1.2815 |
| breast_cancer | cython | 47,141 / early_stop | 66,231 / early_stop | 94.1385 → 93.7833 | -0.3552 | 1.3602 |
| balancescale | numpy | 48,041 / early_stop | 34,221 / early_stop | 60.2273 → 54.0584 | -6.1688 | 0.6571 |
| balancescale | numba | 34,391 / early_stop | 41,851 / early_stop | 59.2532 → 56.4935 | -2.7597 | 1.1716 |
| balancescale | cython | 60,911 / early_stop | 40,241 / early_stop | 60.5519 → 56.6558 | -3.8961 | 0.6347 |
| germancredit | numpy | 157,941 / early_stop | 152,781 / early_stop | 51.1066 → 52.8169 | +1.7103 | 0.8937 |
| germancredit | numba | 148,881 / early_stop | 83,171 / early_stop | 55.3320 → 52.8169 | -2.5151 | 0.5439 |
| germancredit | cython | 106,611 / early_stop | 84,481 / early_stop | 52.5151 → 55.0302 | +2.5151 | 0.8145 |
| diabetes | numpy | 82,731 / early_stop | 64,171 / early_stop | 64.5669 → 63.7795 | -0.7874 | 0.7227 |
| diabetes | numba | 73,971 / early_stop | 80,461 / early_stop | 64.6982 → 62.5984 | -2.0997 | 1.0936 |
| diabetes | cython | 51,271 / early_stop | 61,761 / early_stop | 64.0420 → 63.2546 | -0.7874 | 1.1188 |
| sonar | numpy | 16,441 / early_stop | 17,411 / early_stop | 70.7921 → 62.8713 | -7.9208 | 0.9695 |
| sonar | numba | 21,111 / early_stop | 25,281 / early_stop | 71.2871 → 56.4356 | -14.8515 | 1.1485 |
| sonar | cython | 11,231 / early_stop | 19,751 / early_stop | 69.8020 → 55.9406 | -13.8614 | 1.6313 |
| glass | numpy | 8,671 / early_stop | 9,361 / early_stop | 38.7755 → 45.4082 | +6.6327 | 1.0087 |
| glass | numba | 5,971 / early_stop | 9,971 / early_stop | 35.7143 → 46.4286 | +10.7143 | 1.5733 |
| glass | cython | 7,681 / early_stop | 13,301 / early_stop | 40.3061 → 42.8571 | +2.5510 | 1.5871 |
| vehicle | numpy | 39,101 / early_stop | 57,731 / early_stop | 49.8801 → 48.4412 | -1.4388 | 1.3750 |
| vehicle | numba | 93,311 / early_stop | 39,051 / early_stop | 50.5995 → 47.7218 | -2.8777 | 0.4034 |
| vehicle | cython | 70,961 / early_stop | 74,291 / early_stop | 50.7194 → 45.0839 | -5.6355 | 0.9151 |
| ionosphere | numpy | 41,901 / early_stop | 18,721 / early_stop | 68.9855 → 69.2754 | +0.2899 | 0.4125 |
| ionosphere | numba | 27,511 / early_stop | 32,571 / early_stop | 73.0435 → 68.1159 | -4.9275 | 1.1694 |
| ionosphere | cython | 38,111 / early_stop | 26,671 / early_stop | 72.4638 → 67.8261 | -4.6377 | 0.7182 |
| pendigits | numpy | 500,000 / max_iter | 500,000 / max_iter | 89.2355 → 84.9754 | -4.2602 | 0.9141 |
| pendigits | numba | 500,000 / max_iter | 500,000 / max_iter | 84.8477 → 84.8203 | -0.0274 | 1.0089 |
| pendigits | cython | 500,000 / max_iter | 500,000 / max_iter | 89.9015 → 83.6982 | -6.2032 | 0.9685 |
| satimage | numpy | 386,091 / early_stop | 499,941 / early_stop | 77.1366 → 81.4566 | +4.3200 | 1.1753 |
| satimage | numba | 439,901 / early_stop | 455,411 / early_stop | 77.3706 → 81.7374 | +4.3668 | 0.9724 |
| satimage | cython | 489,151 / early_stop | 307,411 / early_stop | 76.3880 → 81.6594 | +5.2714 | 0.5532 |
| spambase | numpy | 500,000 / max_iter | 421,591 / early_stop | 79.1295 → 75.2992 | -3.8303 | 0.7702 |
| spambase | numba | 499,171 / early_stop | 419,561 / early_stop | 85.7671 → 76.9097 | -8.8575 | 0.8143 |
| spambase | cython | 500,000 / max_iter | 452,031 / early_stop | 84.1567 → 77.9543 | -6.2024 | 0.8366 |
| pageblocks | numpy | 418,541 / early_stop | 500,000 / max_iter | 31.8981 → 44.1920 | +12.2939 | 1.0984 |
| pageblocks | numba | 494,901 / early_stop | 451,471 / early_stop | 27.5009 → 36.6435 | +9.1425 | 0.9049 |
| pageblocks | cython | 500,000 / max_iter | 433,091 / early_stop | 24.7527 → 34.8113 | +10.0586 | 0.7674 |
| banknote | numpy | 234,871 / early_stop | 261,791 / early_stop | 71.0835 → 84.1874 | +13.1040 | 1.0229 |
| banknote | numba | 301,481 / early_stop | 241,361 / early_stop | 79.6486 → 69.6925 | -9.9561 | 0.8179 |
| banknote | cython | 227,291 / early_stop | 197,151 / early_stop | 74.6706 → 88.0673 | +13.3968 | 0.8709 |
| bloodtransfusion | numpy | 103,731 / early_stop | 83,281 / early_stop | 60.5121 → 40.8356 | -19.6765 | 0.7195 |
| bloodtransfusion | numba | 77,991 / early_stop | 76,691 / early_stop | 55.6604 → 42.1833 | -13.4771 | 0.8365 |
| bloodtransfusion | cython | 80,981 / early_stop | 90,341 / early_stop | 64.4205 → 49.4609 | -14.9596 | 1.0246 |
| yeast | numpy | 101,801 / early_stop | 93,391 / early_stop | 37.1389 → 39.0646 | +1.9257 | 0.8520 |
| yeast | numba | 100,931 / early_stop | 98,311 / early_stop | 35.9697 → 41.1279 | +5.1582 | 0.9198 |
| yeast | cython | 107,111 / early_stop | 99,671 / early_stop | 37.7579 → 35.9697 | -1.7882 | 0.8962 |
| segment | numpy | 284,941 / early_stop | 152,901 / early_stop | 73.9624 → 70.8170 | -3.1455 | 0.4949 |
| segment | numba | 277,401 / early_stop | 156,571 / early_stop | 73.7003 → 70.7733 | -2.9270 | 0.5454 |
| segment | cython | 181,601 / early_stop | 141,351 / early_stop | 74.0061 → 71.8218 | -2.1844 | 0.7871 |
| qsarbiodeg | numpy | 102,771 / early_stop | 95,691 / early_stop | 63.4890 → 51.0010 | -12.4881 | 0.8593 |
| qsarbiodeg | numba | 138,691 / early_stop | 139,571 / early_stop | 63.5844 → 57.2927 | -6.2917 | 1.0126 |
| qsarbiodeg | cython | 106,191 / early_stop | 179,611 / early_stop | 67.3975 → 53.1935 | -14.2040 | 1.5027 |

The four highlighted workloads show the practical importance of mask-specific
stopping and censoring; caps do not establish convergence beyond 500,000:

- pendigits: numpy 500,000 (max_iter) → 500,000 (max_iter), accuracy -4.2602 pp; numba 500,000 (max_iter) → 500,000 (max_iter), accuracy -0.0274 pp; cython 500,000 (max_iter) → 500,000 (max_iter), accuracy -6.2032 pp.
- satimage: numpy 386,091 (early_stop) → 499,941 (early_stop), accuracy +4.3200 pp; numba 439,901 (early_stop) → 455,411 (early_stop), accuracy +4.3668 pp; cython 489,151 (early_stop) → 307,411 (early_stop), accuracy +5.2714 pp.
- pageblocks: numpy 418,541 (early_stop) → 500,000 (max_iter), accuracy +12.2939 pp; numba 494,901 (early_stop) → 451,471 (early_stop), accuracy +9.1425 pp; cython 500,000 (max_iter) → 433,091 (early_stop), accuracy +10.0586 pp.
- spambase: numpy 500,000 (max_iter) → 421,591 (early_stop), accuracy -3.8303 pp; numba 499,171 (early_stop) → 419,561 (early_stop), accuracy -8.8575 pp; cython 500,000 (max_iter) → 452,031 (early_stop), accuracy -6.2024 pp.

## Six-repetition timing variability

Each cell is **median; IQR; [minimum, maximum] in seconds**, calculated within
fixture/backend using all six observations and linear percentiles. All extremes
are retained. No trimming, rerunning or replacement of slow measurements occurred.

| Dataset | Backend | Propagation median; IQR; range (s) | Total fit median; IQR; range (s) |
| --- | --- | --- | --- |
| wine | numpy | 3.606509; 0.067532; [2.102543, 3.693674] | 3.606801; 0.067532; [2.102808, 3.693962] |
| wine | numba | 0.007451; 0.000350; [0.007280, 0.018407] | 0.007763; 0.000373; [0.007589, 0.018808] |
| wine | cython | 0.004932; 0.000280; [0.004869, 0.007187] | 0.005263; 0.000320; [0.005173, 0.007832] |
| digits | numpy | 28.961453; 0.094936; [28.858184, 29.003958] | 28.961809; 0.094925; [28.858548, 29.004311] |
| digits | numba | 0.274309; 0.002938; [0.271808, 0.279700] | 0.274711; 0.002946; [0.272182, 0.280131] |
| digits | cython | 0.234819; 0.002619; [0.233645, 0.238906] | 0.235257; 0.002635; [0.234039, 0.239326] |
| iris | numpy | 0.726435; 0.002951; [0.719430, 0.735402] | 0.726608; 0.002949; [0.719601, 0.735558] |
| iris | numba | 0.003852; 0.000146; [0.003776, 0.003932] | 0.004057; 0.000146; [0.003950, 0.004131] |
| iris | cython | 0.002540; 0.000156; [0.002464, 0.003454] | 0.002750; 0.000202; [0.002637, 0.003729] |
| breast_cancer | numpy | 2.572203; 0.031388; [2.557840, 2.606012] | 2.572400; 0.031308; [2.558036, 2.606217] |
| breast_cancer | numba | 0.019803; 0.000122; [0.019655, 0.019935] | 0.020009; 0.000106; [0.019858, 0.020160] |
| breast_cancer | cython | 0.020630; 0.000967; [0.019952, 0.025804] | 0.020864; 0.001015; [0.020152, 0.026055] |
| balancescale | numpy | 3.226308; 0.024854; [3.199113, 3.268493] | 3.226508; 0.024865; [3.199324, 3.268706] |
| balancescale | numba | 0.023645; 0.000248; [0.023384, 0.028777] | 0.023901; 0.000241; [0.023600, 0.029040] |
| balancescale | cython | 0.020484; 0.000803; [0.020064, 0.022042] | 0.020728; 0.000840; [0.020240, 0.022318] |
| germancredit | numpy | 9.728839; 0.034317; [9.711874, 9.858926] | 9.729046; 0.034214; [9.712085, 9.859141] |
| germancredit | numba | 0.039095; 0.004027; [0.037861, 0.046306] | 0.039367; 0.004039; [0.038090, 0.046574] |
| germancredit | cython | 0.032201; 0.002394; [0.031345, 0.036743] | 0.032544; 0.002617; [0.031580, 0.037033] |
| diabetes | numpy | 4.117454; 0.015422; [4.072256, 4.166784] | 4.117659; 0.015418; [4.072471, 4.166997] |
| diabetes | numba | 0.036042; 0.004429; [0.034272, 0.042424] | 0.036312; 0.004465; [0.034495, 0.042690] |
| diabetes | cython | 0.020043; 0.000849; [0.019831, 0.021299] | 0.020273; 0.000993; [0.020057, 0.021534] |
| sonar | numpy | 1.085723; 0.004976; [1.082908, 1.097797] | 1.085898; 0.004988; [1.083077, 1.097966] |
| sonar | numba | 0.010282; 0.000026; [0.010254, 0.010303] | 0.010454; 0.000026; [0.010428, 0.010491] |
| sonar | cython | 0.005494; 0.000029; [0.005450, 0.005578] | 0.005661; 0.000005; [0.005618, 0.005745] |
| glass | numpy | 1.696073; 0.001149; [1.689425, 1.713264] | 1.696252; 0.001149; [1.689622, 1.713449] |
| glass | numba | 0.010211; 0.000034; [0.010179, 0.010303] | 0.010410; 0.000049; [0.010365, 0.010518] |
| glass | cython | 0.012620; 0.000114; [0.012516, 0.016335] | 0.012829; 0.000154; [0.012727, 0.016580] |
| vehicle | numpy | 7.202481; 0.016108; [7.191032, 7.266443] | 7.202711; 0.016102; [7.191271, 7.266667] |
| vehicle | numba | 0.029125; 0.000884; [0.029036, 0.030681] | 0.029376; 0.000901; [0.029276, 0.030963] |
| vehicle | cython | 0.054591; 0.001521; [0.053836, 0.058174] | 0.054847; 0.001537; [0.054074, 0.058454] |
| ionosphere | numpy | 1.189654; 0.011572; [1.180646, 1.198820] | 1.189839; 0.011577; [1.180838, 1.198995] |
| ionosphere | numba | 0.014016; 0.000062; [0.013937, 0.014256] | 0.014205; 0.000061; [0.014119, 0.014457] |
| ionosphere | cython | 0.008291; 0.000145; [0.008209, 0.013589] | 0.008474; 0.000171; [0.008385, 0.013843] |
| pendigits | numpy | 169.151124; 9.689248; [159.920738, 172.946059] | 169.153208; 9.688897; [159.921821, 172.947580] |
| pendigits | numba | 1.982053; 0.118051; [1.863745, 2.049055] | 1.983507; 0.118109; [1.865131, 2.050796] |
| pendigits | cython | 6.394681; 0.248251; [6.192223, 6.703592] | 6.396133; 0.248288; [6.193633, 6.705518] |
| satimage | numpy | 100.118858; 1.120188; [95.256466, 100.513535] | 100.119553; 1.120680; [95.257040, 100.514116] |
| satimage | numba | 0.870597; 0.036698; [0.832868, 0.914908] | 0.871322; 0.036759; [0.833457, 0.915537] |
| satimage | cython | 1.130380; 0.044672; [1.104252, 1.166166] | 1.131493; 0.044532; [1.104939, 1.167076] |
| spambase | numpy | 28.794923; 0.358862; [28.589286, 30.137920] | 28.795384; 0.358903; [28.589725, 30.139218] |
| spambase | numba | 0.312318; 0.017870; [0.309968, 0.337272] | 0.312735; 0.017893; [0.310563, 0.337794] |
| spambase | cython | 0.381732; 0.020757; [0.366374, 0.426058] | 0.382128; 0.020754; [0.366741, 0.426634] |
| pageblocks | numpy | 82.947159; 0.688142; [82.334525, 83.738075] | 82.947658; 0.688118; [82.335079, 83.738696] |
| pageblocks | numba | 0.680499; 0.007114; [0.655636, 0.688216] | 0.681058; 0.007140; [0.657227, 0.688787] |
| pageblocks | cython | 1.106094; 0.035672; [1.086206, 1.154627] | 1.106761; 0.035623; [1.086917, 1.155154] |
| banknote | numpy | 17.528441; 0.133746; [17.398643, 17.915809] | 17.528707; 0.133641; [17.398869, 17.916135] |
| banknote | numba | 0.121505; 0.008350; [0.115026, 0.128129] | 0.121794; 0.008321; [0.115313, 0.128488] |
| banknote | cython | 0.077663; 0.006233; [0.072742, 0.083043] | 0.077989; 0.006348; [0.073044, 0.083397] |
| bloodtransfusion | numpy | 5.552326; 0.015832; [5.528206, 5.633468] | 5.552567; 0.015885; [5.528527, 5.633674] |
| bloodtransfusion | numba | 0.034059; 0.005524; [0.031693, 0.041276] | 0.034355; 0.005500; [0.031913, 0.041636] |
| bloodtransfusion | cython | 0.031051; 0.004190; [0.027922, 0.042434] | 0.031310; 0.004198; [0.028139, 0.042817] |
| yeast | numpy | 30.929298; 0.699679; [30.502002, 31.768389] | 30.929730; 0.699674; [30.502321, 31.768716] |
| yeast | numba | 0.207492; 0.012556; [0.197823, 0.217980] | 0.207999; 0.012596; [0.198344, 0.218485] |
| yeast | cython | 0.263257; 0.012974; [0.252165, 0.277201] | 0.263823; 0.012972; [0.252540, 0.277590] |
| segment | numpy | 35.209378; 0.485100; [34.847551, 35.995532] | 35.209814; 0.484895; [34.848049, 35.995999] |
| segment | numba | 0.256092; 0.006414; [0.239078, 0.261707] | 0.256644; 0.006396; [0.239485, 0.262255] |
| segment | cython | 0.292479; 0.022241; [0.274046, 0.308392] | 0.292978; 0.022301; [0.274466, 0.308840] |
| qsarbiodeg | numpy | 6.412841; 0.032682; [6.350324, 6.528234] | 6.413124; 0.032486; [6.350709, 6.528550] |
| qsarbiodeg | numba | 0.069287; 0.006600; [0.062373, 0.075193] | 0.069592; 0.006556; [0.062630, 0.075553] |
| qsarbiodeg | cython | 0.068790; 0.002667; [0.067374, 0.081811] | 0.069099; 0.002651; [0.067605, 0.082174] |

Largest absolute total-fit ranges, described without an arbitrary outlier cutoff:

| Dataset | Backend | Range width (s) | Maximum / minimum | IQR / median |
| --- | --- | ---: | ---: | ---: |
| pendigits | numpy | 13.025759 | 1.0815 | 0.0573 |
| satimage | numpy | 5.257076 | 1.0552 | 0.0112 |
| wine | numpy | 1.591154 | 1.7567 | 0.0187 |
| spambase | numpy | 1.549493 | 1.0542 | 0.0125 |
| pageblocks | numpy | 1.403617 | 1.0170 | 0.0083 |

Balanced-position diagnostics normalize each fit by its own fixture/backend
median, retaining all observations: forty per position/backend. These are
descriptive ratios, not causal order effects or independent samples. Time/order,
load and uncontrolled power/thermal behavior are confounded. Every exact
permutation/repetition/position remains in the raw CSV and canonical records.

| Backend | Position (0-based) | Fits | Normalized fit median; IQR; range |
| --- | ---: | ---: | --- |
| numpy | 0 | 40 | 0.999506; 0.004449; [0.945805, 1.022432] |
| numpy | 1 | 40 | 0.999870; 0.013164; [0.945426, 1.046668] |
| numpy | 2 | 40 | 1.000221; 0.007654; [0.583012, 1.027126] |
| numba | 0 | 40 | 0.998376; 0.024444; [0.899961, 1.069361] |
| numba | 1 | 40 | 1.000000; 0.031506; [0.933140, 2.422886] |
| numba | 2 | 40 | 1.001643; 0.022721; [0.928923, 1.211951] |
| cython | 0 | 40 | 1.003542; 0.038422; [0.904355, 1.355742] |
| cython | 1 | 40 | 0.999592; 0.053179; [0.898711, 1.488162] |
| cython | 2 | 40 | 0.997454; 0.041806; [0.936584, 1.633534] |

All six orders are also examined separately. Each order/backend cell contains
twenty fits, one per dataset, normalized to the corresponding fixture/backend
median. These descriptive pooled ratios retain all outliers and cannot establish
a causal order effect:

| Order | Backend | Fits | Normalized fit median; IQR; range |
| --- | --- | ---: | --- |
| numpy → numba → cython | numpy | 20 | 0.998894; 0.003298; [0.945805, 1.022329] |
| numpy → numba → cython | numba | 20 | 1.001679; 0.017880; [0.936842, 2.422886] |
| numpy → numba → cython | cython | 20 | 0.995512; 0.016032; [0.936584, 1.367487] |
| numpy → cython → numba | numpy | 20 | 1.001133; 0.009445; [0.988313, 1.022432] |
| numpy → cython → numba | numba | 20 | 1.001205; 0.016873; [0.928923, 1.183074] |
| numpy → cython → numba | cython | 20 | 1.003472; 0.055311; [0.940409, 1.488162] |
| numba → numpy → cython | numpy | 20 | 0.998059; 0.014048; [0.945426, 1.017998] |
| numba → numpy → cython | numba | 20 | 0.998814; 0.039886; [0.952592, 1.069361] |
| numba → numpy → cython | cython | 20 | 1.005262; 0.073019; [0.936815, 1.633534] |
| numba → cython → numpy | numpy | 20 | 1.000355; 0.007661; [0.989725, 1.027126] |
| numba → cython → numpy | numba | 20 | 0.997379; 0.015938; [0.899961, 1.050744] |
| numba → cython → numpy | cython | 20 | 0.996760; 0.032763; [0.898711, 1.189221] |
| cython → numpy → numba | numpy | 20 | 1.000934; 0.009697; [0.986181, 1.046668] |
| cython → numpy → numba | numba | 20 | 1.002883; 0.029341; [0.965008, 1.211951] |
| cython → numpy → numba | cython | 20 | 1.009016; 0.032044; [0.976498, 1.355742] |
| cython → numba → numpy | numpy | 20 | 0.999995; 0.005887; [0.583012, 1.022103] |
| cython → numba → numpy | numba | 20 | 0.997078; 0.042575; [0.933140, 1.214990] |
| cython → numba → numpy | cython | 20 | 1.000392; 0.024124; [0.904355, 1.292394] |

## Overhead and resources

| Component | Seconds |
| --- | ---: |
| Warm-up fit timers | 1.299823 |
| Reused fixture label/preparation timers | 0.002206 |
| First RNG helper call | 0.795913 |
| Measured helper wall spans minus fit timers | 0.220851 |
| Post-fit validation/result-publication windows, total | 2.046709 |
| Native wall less fits, warm-up fits, preparation and RNG helper | 11.095223 |
| Native wall outside session timer | 3.256680 |

Publication-window median; IQR; range: 0.003712; 0.002491; [0.001547, 0.031240] s.
These rows overlap and must not be added. Residuals include verification,
model/graph copies and RNG seeding, checkpoint/session/ledger I/O, startup and
initial/final audits/summaries. Publication windows include validation/result
writes but exclude subsequent completion-event fsync. **Pure filesystem I/O is
not separately timed**; the original fit timer boundaries remain unchanged.
Offline audit took 2.389419 s, outside execution duration.
Minimum free RAM/disk: 51.78/
714.78 GiB. No severe resource stop
occurred; all timing variation remains in the report.

## Revised remaining-runtime scenarios — no execution authorized

Eight seeds remain: **2,880 measured fits, 960 warm-ups in uninterrupted stages,
and 1,440,000,000 cap iterations**. Estimates use both completed masks and all
timing observations. The stage-wall extrapolation includes outliers; cap rates
use the twelve observations per dataset/backend. Historical cap rates use the
exact frozen fixed-work groups for seeds 12–19. Overhead charge per future stage:
12.434432 s, with further allowance
needed as audit/ledger/fixture inventories grow.

| Remaining-seeds scenario | Estimated duration |
| --- | ---: |
| Observed two-stage mean wall | 8 h 8 min 8.7 s |
| All capped: observed median rates | 21 h 34 min 14.8 s |
| All capped: observed maximum rates | 27 h 16 min 29.5 s |
| All capped: historical median rates | 24 h 17 min 51.3 s |
| All capped: historical maximum rates | 28 h 31 min 49.5 s |
| Conservative cap work, max scenario × 1.5 | 42 h 47 min 44.3 s |

The two observed stage-wall scenarios span 7 h 24 min 35.4 s
to 8 h 51 min 41.9 s for eight repeats. These are
planning scenarios, not confidence intervals, runtime bounds or cutoffs. Only
two label masks are observed per dataset; later masks can change convergence,
censoring, accuracy and per-iteration cost. Interrupted sessions require fresh
pending-group warm-ups and explicitly authorized ambiguous retries.

## Artifacts and next action

The [complete analysis](benchmarks/early_stopping_full_seed11/results/summary.json),
[360-row CSV (archived)](performance_early_stopping_archiving.md),
[raw measurements (archived)](performance_early_stopping_archiving.md)
and canonical raw/provenance records preserve every fit, warm-up, permutation,
state hash, fixture seal, task event and summary generation. Original byte
checksum inventories accompany the records; Git-normalized copies never
replace original checkpoint bytes. Full NPZs, staging evidence and source
snapshots remain local; no source/NPZ archive or Release asset is published.
The read-only audit and snapshot reproduction command is provided with the
optional `tools.analyze_full_seed11` helper. Live root remains
`.cache/performance/windows-full-early-stopping-01`; seed-11 provenance is its
`-seed11-provenance` sibling. A point-in-time `-seed11-snapshot` is retained for
later reproduction without changing the live campaign.

The raw bundle is incremental: it adds seed-11 canonical records and session
provenance while referencing the unchanged prefix in the
[seed-10 bundle (archived)](performance_early_stopping_archiving.md).
The [cumulative raw measurements (archived)](performance_early_stopping_archiving.md)
contain all 720 fits. The retained point-in-time snapshot contains all 5,002
checkpoint files byte-identical to the live root at completion. Reproduce its
read-only audit in a new output directory:

```console
python -m tools.analyze_full_seed11 --root .cache/performance/windows-full-early-stopping-01-seed11-snapshot --baseline .cache/performance/windows-fixed-full-c9f512c-01/fixed-10000 --provenance .cache/performance/windows-full-early-stopping-01-seed11-provenance --output NEW_AUDIT
```

This command performs no fits, warm-ups, retries or checkpoint mutations. The
audit elapsed duration is observational; scientific data and statistics reproduce
exactly. Historical execution is excluded from CI.

**Wait for further authorization.** No seeds 12–19 were executed. Original
protocol and task identities, historical fixed results, README performance
tables, version and Release are unchanged. PR #17 remains an open draft and
has not been merged.
