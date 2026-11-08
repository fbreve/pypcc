# Verified edge-index conditional diagonal cleanup — research findings

**Status: preregistered study PASSED all strict scientific and candidate-promotion gates. Do not merge the experimental PR wholesale.** Protocol [`edge_index_diagonal_protocol.md`](edge_index_diagonal_protocol.md) was committed **before** modifying production code or collecting outcomes. Base is original production `master` **`35dc81caf0388be8ea1729837ea6d54358b53964`**, already containing the independent exact ordered CSR offset optimization #192. No change was made to the original kNN builder or particle propagation/stopping backends.

## Reproducibility and strict evidence

- [Scientific run **42446734260**](https://github.com/fbreve/pypcc/actions/runs/42446734260) — original Windows job SUCCESS, Linux job SUCCESS, mandatory two-host strict summary SUCCESS.
- [Machine-checked consolidated report artifact **13603512595**](https://github.com/fbreve/pypcc/actions/runs/42446734260/artifacts/13603512595) and exact original/candidate raw array hashes, 30-pair time distributions for every case, original native six-state SHA hashes, error cases and versions: [Linux **13603957402**](https://github.com/fbreve/pypcc/actions/runs/42446734260/artifacts/13603957402), [Windows **13603344565**](https://github.com/fbreve/pypcc/actions/runs/42446734260/artifacts/13603344565).
- The original reference was imported from the **unchanged pinned master Git blob**, and the candidate source was guarded to be **identical except one conditional cleanup hunk** in `build_graph_from_edge_index`. Original `build_knn_graph`, `NearestNeighbors(algorithm='auto')`, all NumPy/Numba/Cython propagation, RNG, model, graph defaults, label selection and stopping kernels remain untouched.
- **1,320 complete, alternated original vs candidate paired edge-index graph constructions**: 12 canonical original reciprocal union kNN derived edge listings (Iris/Wine/Breast Cancer/Digits × k1/3/10) + 10 independently frozen synthetic edge-index cases, × 30 whole-builder repetitions per input × 2 OS. Every `neib_list` (full ordered output and -1 padding), `neib_qt`, shape and dtype matched exactly in every pair. The graph and input SHA identities also matched across OS.
- **96 original native PCC state comparison pairs**: 4 datasets × 2 new frozen initial label masks (28319/28447) × all three original NumPy, Numba, Cython sequential backends × original `early_stop=False/True` modes × Linux/Windows. Exact six full final array SHA256 hashes, native actual iteration count/stopping metadata and unlabeled accuracy in every original/candidate pair. The 1000-iteration cap is a semantics check, not proof of convergence. Different backend RNG streams are not compared against each other.
- **16 extra invalid/empty behavior controls**: n=0 special output, zero-node inputs with edges, negative node count, out-of-range endpoints, negative indices, malformed shape, floating input converted to indices, all-self-loop graph; original result/exception type and error text precisely matched for all.
- All package versions, host identity, source hashes and frozen environments were saved with the outputs. **No timing results were filtered**. CI demanded complete source/graph/native equivalence but deliberately **did not require positive speedup**.

## Measured complete `build_graph_from_edge_index` speed gains

Values are **median paired reductions in elapsed wall-clock time**, each using 30 counterbalanced original/candidate pairs on each hosted runner (not ratios of separate median values). All values reflect the *complete* original edge-index graph builder, not microphase only.

| Real source of edge lists | k | Linux | Windows |
|---|---:|---:|---:|
| Iris | 1 | +46.08% | +44.37% |
| Iris | 3 | +46.62% | +44.53% |
| Iris | 10 | **+42.30%** | **+41.76%** |
| Wine | 1 | +46.88% | +47.66% |
| Wine | 3 | +47.31% | +43.47% |
| Wine | 10 | **+44.77%** | **+40.91%** |
| Breast Cancer | 1 | +50.20% | +45.44% |
| Breast Cancer | 3 | +47.69% | +43.54% |
| Breast Cancer | 10 | **+39.19%** | **+35.05%** |
| Digits | 1 | +51.39% | +48.49% |
| Digits | 3 | +45.08% | +41.04% |
| Digits | 10 | **+32.94%** | **+28.70%** |

| Prespecified synthetic edge-index condition | Linux | Windows |
|---|---:|---:|
| empty_1 | +45.20% | +45.99% |
| empty_10000 | +60.47% | +58.57% |
| isolated_30000_1000 | +54.57% | +54.22% |
| sparse_1000_1000 | +51.13% | +48.37% |
| sparse_10000_1000 | +52.25% | +52.77% |
| medium_1000_3000 | +45.26% | +41.07% |
| repeated_reciprocal_1000_3000 | +44.47% | +40.92% |
| dense_1000_30000 | +31.29% | +28.92% |
| dense_10000_100000 | +30.13% | +28.73% |
| **mixed_loops_1000_2000** | **−2.07%** | **−3.01%** |

A no-loop scan has a small but genuine cost for the loop-containing case, which uses the **unchanged old cleanup path** after the scan. That unfavorable observation is explicitly retained and should not be hidden. All 22 conditions met the *no worse than 10%* material regression gate on both hosted OS.

## Frozen promotion decision

The precommitted acceptance rule required ≥10% lower paired median complete builder time in **at least five of seven named sparse loop-free cases on each OS**, with **no >10% paired median regression** anywhere in 22 real/synthetic cases on either OS.

**RESULT: PASS — 7/7 sparse cases ≥10% faster on Linux, 7/7 on Windows, 0/44 total OS/case regressions >10%.** Complete native graph/state equivalence also passed.

**Engineering action:** extract a **separate independent master-based minimal production PR** containing only the conditional cleanup in `pcc_graph.py` and one permanent compatibility test for empty, duplicate, loop-containing, isolated, irregular and invalid graph inputs. Run all original four Windows/Linux × Python3.10/3.12 wheel/sdist regression jobs; do not merge the scientific benchmark infrastructure from this PR into master.

## Procedural provenance, scope and caveats

Initial [CI run **42446637762**](https://github.com/fbreve/pypcc/actions/runs/42446637762) failed *before any scientific work ran* because the source-audit script had accidentally introduced literal backslash-n characters into a Python statement while fixing Git blob newline preservation. That syntax error was corrected at experimental script commit `7d7cca214231ecb50842aa09b9058fb11e7720fc`, **without changing the candidate source, data, precommitted thresholds, timings or backend behavior**; the subsequently complete scientific run passed. Do not present the original failed run as valid measurements.

The large gains here apply to the **edge-index graph constructor**, which is different from the original `build_knn_graph` that computes neighbors via scikit-learn. Claiming that kNN graph construction or entire end-to-end PCC propagation accelerates by 30–60% would be incorrect. Graph reuse amortizes build cost. Windows/Linux runners and numerical libraries/compilers can differ from each other and from a local workstation; technical repeats inside one host process do not represent independent machines or a population estimate. The simple boolean CSR/no-self-loop invariant is necessary to justify skipping cleanup; **do not adapt this conditional removal unexamined to the int8 `build_knn_graph` adjacency**, whose addition and edge semantics differ.

**Final status:** this PR is a frozen research archive, DRAFT/UNMERGED. The smallest proven source change may be considered separately only after its independent regression suite passes.
