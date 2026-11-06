# Observations: controlled-choice production-backend parity (2026-11-05)

**Study protocol frozen before execution:** [backend_choice_controlled_parity_protocol.md](backend_choice_controlled_parity_protocol.md). This is **experimental** work nested under experimental PR #169; no production behavior or `master` changes.

## CI / raw data

- Workflow: [42143063511](https://github.com/fbreve/pypcc/actions/runs/42143063511), **4/4 matrix jobs successful**.
- Four dataset artifacts: `backend-choice-parity-iris` (ID 13469605054), `backend-choice-parity-wine` (ID 13469525137), `backend-choice-parity-breast_cancer` (ID 13468294285), `backend-choice-parity-digits` (ID 13468309073).
- All results are independent no-stop fits at frozen caps/seeds, **not** checkpoint sampling of one continuous trajectory. A single pairwise comparison checks final prediction, current particle node, particle strength, distance table, dominance matrix, and normalized ownership matrix. Floating comparison was prespecified with `rtol=0, atol=1e-12`; discrete comparisons require exact equality.

## Observations

| Diagnostic cohort | Full-state comparisons matching | Largest floating-state difference | Interpretation |
| --- | ---: | ---: | --- |
| Undirected *degree-one* forced destination; NumPy/Numba/Cython, `p_grd=0` | **30/30** | **0** | Actual destination independent of RNG; tested transitions identical |
| Same forced destination, `p_grd=1` | **30/30** | **0** | Tested greedy transitions also identical |
| Canonical undirected 10-NN; NumPy vs Numba; `p_grd=1` | **56/56** | **0** | Exact full-state agreement with weighted greedy-only sampling through 1,000 steps |
| Same canonical graph; NumPy vs Numba; `p_grd=0` | **0/56** | up to **1.0** in dominance | First sampled checkpoint (cap 1) already differs for all 8 dataset/seed cases, consistent with different random-walk index selection |

There are **60** forced-destination comparisons = 2 seeds × 2 `p_grd` values × 5 caps × 3 backend pairs, using actual production backends. Canonical: **56** comparisons per `p_grd` value = 4 datasets × 2 seeds × 7 caps, using actual production NumPy and Numba. Fresh seeds: 4409 and 4513.

Even the float matrices matched **exactly** in the successful cases, stronger than the required 1e-12 tolerance. On the canonical `p_grd=0` cases, mean final-prediction agreement between the *different trajectories* was 97.49% after 1 step and 46.20% after 1,000 steps; these deliberately short, unconverged horizons **must not** be misreported as ordinary PCC classification agreement.

## Interpretation and next decision

- The full-state agreement in the `p_grd=1` canonical tests provides strong *tested-scope* evidence that NumPy and Numba sequential update calculations agree when they make the same greedy destination choices. Together with the degree-one forced-choice control, this supports RNG index-selection differences as an important source of divergence when random movement is enabled.
- This does **not** prove equivalence for all graphs, seed distributions, parameters, stopping criteria, or 100k+ trajectories. The forced-choice graph is disconnected and is not a canonical kNN benchmark. Cython's XORShift generator is not aligned on canonical multi-neighbor graphs.
- The observed all-differing `p_grd=0` case is **not** a correctness defect by itself: NumPy uses `int(np.random.random()*degree)`, Numba uses `np.random.randint(degree)`, and Cython uses XORShift32 modulo. All are allowed random-neighbor policies, but they are not sample-path equivalent.
- To establish practical equivalence under the **default `p_grd=0.5`**, a separately prespecified, multi-seed/100k *distributional* comparison should assess differences in accuracy and stopping behavior without demanding identical trajectories. Avoid choosing a favorable stopping criterion based on these diagnostic traces.

**Decision:** retain current production kernels and stopping defaults; no merge to `master`.
