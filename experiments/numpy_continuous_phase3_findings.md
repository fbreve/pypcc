# NumPy continuous observer — phase-3 boundary and independent master comparison

**Status:** Experiment-only draft [PR #182](https://github.com/fbreve/pypcc/pull/182), no merge to `master`. Frozen [protocol](numpy_continuous_phase3_protocol.md) was committed before CI results. This phase extends the prior successful 68/68 observed/unobserved strict parity pairs from Linux and Windows. The *only* executable production diff against original master remains `pcc_numpy.py` with an opt-in trailing `observer=None` and read-only callback dispatch. The normal high-level `ParticleCompetitionAndCooperation(impl="numpy").fit_predict` does **not** enable the observer, and neither Numba nor Cython has been touched.

## Boundary compatibility: PASS

- [Workflow **42327705460**](https://github.com/fbreve/pypcc/actions/runs/42327705460): **4/4 OS×dataset jobs + mandatory summary green**. Full summary artifact **13556043359**; raw CSVs and dependency versions: Linux Iris **13555368018**, Linux Digits **13555602444**, Windows Iris **13555412809**, Windows Digits **13556093088**.
- **36/36 matched pairs = 72 real fits**, all pass **exactly six full final state arrays, complete NumPy native stop metadata, full MT19937 generator state and the next 16 draws**, real initial/final snapshot, actual checkpoint schedule, and <=9 snapshots / <=48 MiB. All 36 cases included.
- Cohort: standardized canonical union-symmetrized **undirected 10NN** on Iris and Digits × Windows/Linux × new harness seeds **14003/14107/14209** × (sequential no explicit fit seed; **zero-iteration** with no explicit fit seed; NumPy **synchronous** with no explicit fit seed). For `seed=None` the test explicitly reinstalls the **same global RNG state before each fit**; it does not claim independent uncontrolled unseeded fits are identical.
- **Zero iterations:** valid in the existing propagation API, effective steps 0, exactly one physical snapshot at iteration 0 upgraded to final, **zero RNG draws** and full final-state/stop parity. No fabricated positive iteration.
- **Synchronous:** original synchronized movement and batching preserved; selected initial and final states, RNG and stopping policy agree exactly between observer-off and observer-on paths.
- The fail-first callback safety regression proves a callback raising before step 1 cannot mutate particle buffers. Failures from arbitrary callbacks *during* an execution are not promised transactional.

### Descriptive observer-on full-fit cost in short boundary workloads

| OS | Dataset | Sequential unseeded (111 maximum) | Zero-step (0) | Synchronous (111 maximum) |
|---|---|---:|---:|---:|
| Linux | Iris | +6.43% | +122.93% | +1.35% |
| Linux | Digits | +10.59% | +189.61% | +12.50% |
| Windows | Iris | +6.68% | +115.56% | +3.33% |
| Windows | Digits | +1.58% | +165.39% | +14.81% |

**Interpretation:** Long 6001-step observed fits in phases 1+2 showed typically ~0–3% full-fit observer overhead; **short 111-step observed runs can exhibit considerably larger overhead**, and zero-step percentages are setup/snapshot-dominated. These are expected opt-in costs, not evidence of a slowdown for users who never pass an observer. Do not extrapolate the low overhead at 6001 iterations to very short or densely sampled traces.

## Independent source benchmark against untouched production master: PASS

- [Workflow **42327705451**](https://github.com/fbreve/pypcc/actions/runs/42327705451): **6/6 OS×dataset matrix jobs and mandatory independent summary green**. Aggregate report **13556416429**. Raw paired fit CSV and full NPZ/JSON array/RNG outputs: Linux Iris **13556520239**, Linux Breast Cancer **13556615072**, Linux Digits **13556043181**, Windows Iris **13555823816**, Windows Breast Cancer **13556018435**, Windows Digits **13555499260**.
- Original baseline is a **separate checkout at SHA `8298ece05c4edf4bbdc871d0aa5b13d9b8ed4128`**, candidate a **separate checkout** at `474568ce3beab343fb1467a9852581ec98bce918`. The test enforces that `pcc.py` and `pcc_numpy.py` were imported directly from the **correct independent checkout** in each fresh Python subprocess. Baseline source has *no* observer parameter, whereas the candidate source does; source SHA-256 and Git commit identities recorded and distinct.
- **24/24 exact original-master vs candidate-default-off paired comparisons = 48 fresh `fit_predict` calls**; all saved full-state arrays, NumPy RNG state and next 16 draws equal, with no missing cases. No observer keyword is passed on either side. Same undirected production union-10NN, two labeled nodes/class, `early_stop=False`, 6001 iterations, fresh seeds **15013/15101**, two alternating paired blocks per seed.
- Full `fit_predict` timing excludes subprocess launch, imports, graph setup and output serialization. Each stratum reports the median of two seed-level medians of **paired time savings**, not a ratio of overall means; all outliers are retained.

| OS | Dataset | Median saving with default observer OFF vs untouched master | Range of individual pairs |
|---|---|---:|---:|
| Linux | Iris | **+6.69%** | −0.63% to +14.49% |
| Linux | Breast Cancer | **−0.74%** | −2.78% to +0.25% |
| Linux | Digits | **+0.72%** | −0.12% to +1.44% |
| Windows | Iris | **+0.50%** | +0.29% to +0.73% |
| Windows | Breast Cancer | **+0.14%** | −2.19% to +5.93% |
| Windows | Digits | **+0.35%** | −2.18% to +2.54% |

**Frozen default-off speed gate PASS:** worst OS×dataset median **−0.74%**, versus prior minimum acceptable **−8%**. Differences of this size are subject to hosted runner scheduling and should *not* be claimed as an optimization or an established speed increase. The important finding is the absence of an appreciable default-off regression in this frozen cohort.

## Remaining research and integration constraints

- The **total validated diagnostic record through phase 3** is 68 + 36 + 24 = **128 strictly compared pairs** under their respective conditions, including 68 observer-on vs off medium-run pairs, 36 extra boundary pairs, and 24 isolated master-vs-candidate *default-off* pairs. All three studies have distinct goals, so never combine their timings into one mean.
- Exact equality includes real stop behavior and NumPy RNG continuation **within matched pairs**; there is no implied cross-OS identity of independent stochastic trajectories or equivalence with Numba/Cython RNG implementations.
- Sampled classifications are **net checkpoint differences**, not full counts of individual flips, so this is still not a convergence or stopping safety proof.
- Arbitrary observer callbacks can raise after mutations and are outside atomic execution guarantees. The optional callback interface, packaging surface, memory budgeting at very large node/particle counts, malformed custom observers and high-frequency checkpoint overhead still merit engineering review before any production merge.
- Keep #182 **draft and unmerged** until that review. Do not transplant old PR #18's native kernels or Cython/Numba experiment code into modern production. A future **Numba-only** continuous tracer, if scientifically valuable, must use its own current-master opt-in instrumentation and exact state/RNG/stopping parity study rather than assuming NumPy results transfer.
