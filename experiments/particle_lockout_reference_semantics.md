# Particle lockout vs original PCC implementations: semantic cross-check

This is a **source-level audit**, not a MATLAB/Python execution-parity test.

Sources:
- MATLAB reference: [`fbreve/Particle-Competition-and-Cooperation/pcc.m`](https://github.com/fbreve/Particle-Competition-and-Cooperation/blob/master/pcc.m) (master, fetched 2026-11-04)
- MEX C reference: [`fbreve/Particle-Competition-and-Cooperation/pccloop.c`](https://github.com/fbreve/Particle-Competition-and-Cooperation/blob/master/pccloop.c) (master, fetched 2026-11-04)
- Python NumPy production kernel [`pcc_numpy.py`](../pcc_numpy.py) and Python Numba kernel [`pcc_numba.py`](../pcc_numba.py)
- Experimental interventions: [`lockout_rescue_intervention_findings.md`](lockout_rescue_intervention_findings.md), GitHub Actions run 41920906992.

## Key rules present in both MATLAB and Python

| Rule | MATLAB-only `pcc.m` | MEX `pccloop.c` | Python NumPy / Numba |
| --- | --- | --- | --- |
| Update destination dominance on an unlabeled target, transferring from competitors to visiting class | Yes | Yes | Yes |
| Update particle strength toward target **after dominance update**, whether the movement is accepted or rejected | Yes (`potpart(j) += (pot(target, class)-potpart(j))*deltap` before collision test) | Yes (same assignment before collision test) | Yes (strength update before acceptance test) |
| Move particle only when own-class dominance ties for maximum at target | Yes (`>= max`) | Yes (`>= maxpot`) | Yes (`== max`); equality with row maximum expresses the same tie test for finite values |
| Start at labeled home node, strength = 1 | Yes | Yes | Yes |

**Result:** The specific feedback loop involved in the Wine seed-83 collapse—failed visits reduce particle strength, while the particle remains at the same node—is **consistent with the original MATLAB and MEX rules**, not a Python-only deviation. This rules out that *particular* cross-language semantic mismatch as an explanation for the observed lockout, but does not prove the whole implementations are equivalent or free of bugs.

## Caveats

- MATLAB-only code draws random arrays per iteration; Python sequential kernels draw per-particle values in a different order and use different generators. Identical numeric seeds are not expected to reproduce bit-identical trajectories across implementations.
- MEX and MATLAB differ from Python in implementation details (e.g. distance table initialization, random selection, accumulated ownership, floating-point details); no full parity audit was conducted here.
- The strength/acceptance rule is faithful to the reference **as written**. That does not prove every instance of class extinction is scientifically desirable or that resetting strength/position would generally improve accuracy.
- Do not integrate a change to strength, movement acceptance, or early-stopping behavior into `master` based on a single selected example.

## Recommendation

Pause tuning early-stop patience and tolerance on the repeatedly examined Wine seed 83. The controlled intervention showed the collapse is reversible in that instance, but the original algorithm can naturally exhibit the same feedback mechanism. If the research objective is reliable early stopping, the next stage should define an out-of-sample, multi-seed evaluation with fixed decision criteria before running further heuristic comparisons.
