# Prospective in-loop hybrid stopping: initial matched comparison

Source: GitHub Actions run **41847394991**, artifact `inloop-global-stopping/inloop-global-stopping.csv`, commit `71932c4`. The in-loop and PCC regression workflows passed.

## Protocol and limitations

Experimental NumPy backend only. Two datasets (Iris and Breast Cancer), seeds 11/23/37, 10-NN, two labeled examples per class, cap 50,000 iterations. Compare (a) label-only global stability and (b) global label stability **plus** 95th-percentile dominance-change threshold, with monitoring intervals 1,000 and 5,000 and stable-check patience 1,2,3. Thresholds 0.001 and 0.01. A same-seed 50k run without stopping is the capped reference, **not a convergence guarantee**. For each configuration three stopped runs and three references were compared; timing data are single measurements per dataset/seed in one GitHub Actions run, not replicate timing distributions.

## Iris: interval 1,000 (three seeds, 150,000 total reference iterations)

| Criterion | Threshold | Patience | Early triggers | Iterations saved | Mean runtime | Agreement with 50k |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Label-only | 0 | 1 | 3/3 | 135,000 (90.0%) | 0.416 s | 97.22% |
| Label-only | 0 | 2 | 3/3 | 129,000 (86.0%) | 0.583 s | 97.92% |
| Hybrid | 0.001 | 1 | 2/3 | 28,000 (18.7%) | 3.398 s | 100% |
| Hybrid | 0.001 | 2 | 2/3 | 7,000 (4.7%) | 3.972 s | 100% |
| Hybrid | 0.001 | 3 | 0/3 | 0 | 4.176 s | 100% |
| Hybrid | 0.01 | 1 | 3/3 | 66,000 (44.0%) | 2.338 s | 99.54% |
| **Hybrid** | **0.01** | **2** | **3/3** | **43,000 (28.7%)** | **2.966 s** | **100%** |
| Hybrid | 0.01 | 3 | 1/3 | 24,000 (16.0%) | 3.506 s | 100% |

Mean Iris no-stop runtime: 4.167 s. At hybrid threshold 0.01, interval 1,000, patience 2, individual stop points were seed11 49k, seed23 25k, seed37 33k; all three exactly matched 50k predictions on initially unlabeled nodes. The corresponding mean measured runtime was 28.8% lower than no-stop. More aggressive hybrid patience=1 at the same threshold changed one unlabeled classification in each of seeds 23 and 37, with higher measured accuracy in both cases. An incidental accuracy improvement is not proof of convergence.

For interval **5,000**, neither hybrid threshold stopped on Iris (all patience levels); this may reflect both reduced stopping opportunity and larger changes across each monitoring interval.

## Breast Cancer

For **both hybrid thresholds**, intervals 1,000 and 5,000, and patience 1–3: **zero early stops out of three** in every configuration, reference agreement 100%, and no iteration savings. Mean no-stop runtime 2.895 s; mean hybrid runtime was near or slightly above this. Label-only interval 1,000, patience 1 stopped in 2/3 runs (15,000 iterations saved), but reference agreement averaged 97.88%, with a worst case of 95.58% for seed11. This demonstrates the hybrid's greater conservatism, not an overall performance advantage.

## Assessment

The hybrid rule filters clearly premature label-only stops in this small pilot. The configuration (0.01, interval 1,000, patience 2) is an **interesting candidate** on Iris, but it is not validated: only three seeds, one run per timing, two datasets, a capped reference, and no evidence of benefit on Breast Cancer. Do not merge into master or make it the default. Next: a small, independently triggered pilot on Wine and Digits using identical graph/seeds and reporting both actual runtimes and predictions, before considering a broader benchmark. Also check whether a more suitable dominance metric can recognize stability for hard datasets without allowing premature stops.
