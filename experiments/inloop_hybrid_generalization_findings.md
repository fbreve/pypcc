# Hybrid in-loop stopping generalization: Wine and Digits

Source: GitHub Actions run **41854042109**, artifact `inloop-hybrid-generalization/inloop-hybrid-generalization.csv`, commit `faf267d`. The targeted workflow passed.

## Protocol

Experimental NumPy PCC backend, datasets Wine and Digits, seeds 11/23/37, 10-NN graph and two labeled samples per class. Reference 50,000 iterations without early stopping. In-loop label-only or hybrid label + P95 dominance-change criterion (tolerance 0.01), checks every 1,000 iterations and patience 1 or 2. Three runs per dataset/configuration. Time measures one CI execution per run, so treat as descriptive only. Percent agreements below are computed on initially unlabeled nodes. A capped endpoint is *not* proof of convergence.

| Dataset | Criterion | Patience | Stops / 3 | Iterations saved / 150,000 | Mean elapsed (s) | Mean reference (s) | Agreement with 50k | Mean accuracy delta (percentage points) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Wine | Labels only | 1 | 3 | 105,000 (70.0%) | 1.959 | 6.489 | 97.674% | -2.326 |
| Wine | Labels only | 2 | 3 | 90,000 (60.0%) | 2.616 | 6.489 | 99.031% | -0.969 |
| Wine | Hybrid | 1 | 3 | 90,000 (60.0%) | 2.608 | 6.489 | 99.031% | -0.969 |
| Wine | Hybrid | 2 | 3 | 72,000 (48.0%) | 3.403 | 6.489 | 99.225% | -0.775 |
| Digits | Labels only | 1 | 2 | 41,000 (27.3%) | 16.111 | 22.021 | 97.167% | -0.694 |
| Digits | Labels only | 2 | 0 | 0 | 22.221 | 22.021 | 100% | 0 |
| Digits | Hybrid | 1 | 0 | 0 | 22.203 | 22.021 | 100% | 0 |
| Digits | Hybrid | 2 | 0 | 0 | 22.164 | 22.021 | 100% | 0 |

Wine hybrid (patience 2): seed 11 stopped at 40k and matched 50k exactly; seed 23 stopped at 16k and differed by one initially unlabeled prediction (accuracy -0.5814 pp); seed 37 stopped at 22k and differed by three predictions (accuracy -1.7442 pp). The mean accuracy deficit is 0.7752 pp. Digits label-only/patience 1 stopped at 39k for seed 23 and 20k for seed 37, with respective prediction agreement of 98.424% and 93.078%; hybrid ran to cap for all seeds.

**Assessment:** Hybrid stopping suppresses especially risky early label-only triggers in Digits, but with no savings. In Wine, it still stops before the 50k endpoint for all seeds and introduces accuracy loss on two seeds. The earlier promising Iris 3-seed result (hybrid tolerance 0.01/patience 2/interval 1k) does **not** generalize cleanly. Retain experimental-only status; no master merge.

**Next methodological check:** Since 50k is not guaranteed to represent convergence and late flips are already documented, compare capped 50k and 100k no-stop references against a *prospective* hybrid rule with cap 100k, using additional seeds. This tests whether apparent reference agreement was partly a short-horizon artifact; it cannot establish convergence even at 100k.
