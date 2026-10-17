# G1-R1 causal label-selection adapter

Implementation only; real execution requires separate approval of an executable
preflight SHA. PR18 stays draft. No Banknote/Sonar, additional label candidates,
new propagation seeds, altered PCC loops or changed stopping defaults.

The immutable envelope `causal_selection_scientific.json` contains the approved
protocol text, all 34 C/D selections, candidate registry, matching certificates,
mask supports/hashes, hypotheses, outcomes, non-evaluability conditions and old
document hashes. Canonical payload SHA:
`4057a6a1cec696a663707ae2db7160b1f63ba7d3a95f5a256f02cbe06a677be6`.
Catalog payload SHA:
`24f832695a1b93e46b101e0c8d86e0110c79401e7cacac7bb8d4f8182199be59`.
Earlier proposals, G0 and C2-R1 remain historical records. H1/H2/H3 retain their
original classifications. C2 exact/near-tie tolerances, selective thresholds and
margin bins are the frozen values in `c2_r1_scientific.json`, payload SHA
`955dca598c123d059269f18fbd7fe67f4d021b3bd211d2f1da1846557989c825`.

## Execution contract

102 unique tasks = (PageBlocks10 + Spambase7) × two arms × three backends.
Every fit receives exactly three known labels/class. All other nodes receive -1.
No graph, feature, truth, node index, competition rule or public PCC API changes.
The selector accepts only the graph and classes in the historical candidate set.
The evaluator can read truth, but cannot send it to selection or matching.
Preflight verifies graph fingerprints through the unchanged historical fixture
loader, rechecks all G0 fixture/document seals, reconstructs matching and
regenerates the PageBlocks DP choices. Spambase uses the seven exact approved
witness pairs; it is not resampled using the new PageBlocks generator.

NumPy/Numba seed0 once/fit, native Cython123456789 unchanged. Dedicated local
PCG64 chooses execution order using [dataset_code,30000+block]; task identities
remain stable despite order randomization. No global NumPy seed is consumed by
preflight or selection. Home-node sorting remains the public PCC convention.
Different arms/backends need not share trajectories or conditional draw counts.

Each task's preflight record binds dataset/block/arm, original node selections,
slate ID and backend. Snapshot summary `dataset` is an intervention annotation
containing those identities; `label_seed` is the block number, explicitly a
caller-declared annotation, **not** a historical random-label seed. Snapshot
home/class identities and label hashes must match the immutable task exactly.
The plan binds the science/catalog/preflight SHAs. No schema additions are made
to the existing NPY/JSON/CSV formats.

The old summary `accuracy_unlabeled_retrospective` uses V minus the actual labels;
it is not the primary causal score. Primary E=V minus **all candidates** is stored
and hashed in preflight. Use `evaluate` on each snapshot's assignments with this
mask and stored components to obtain global accuracy, BA, support, correct counts,
recall and confusion matrices for all E and components with/without candidates.
Missing-class stratum BA is NA. Full dominance permits all frozen ties/margins
and 10k-cohort decompositions offline without another fit. No causal conclusions
or outcome analysis are part of technical preparation.

## Bounds, failures and compatibility

Ten milestones:0,1k,2k,5k,10k,20k,40k,60k,80k,100k, final deduplicated;
max_iter100000, no early stopping. At most128 observations, independent64MiB
array/memory/export budgets per fit. Estimates exclude total process RSS/JIT and
extra failed attempts. Geometry computes candidate-to-all distances, not N²
APSP; PageBlocks candidate distance matrix is about5.5MiB float64. The geometric
working budget128MiB must be checked against dimensions before allocating.

Store ledger and locks are reused unchanged. Result publication requires a full
snapshot validation; the completion manifest is written last. A valid interrupted
publication is reconciled without fitting. An incomplete attempt requires an
explicit task/reason retry and is preserved. Corruption blocks; it never triggers
automatic refitting. Recovery is whole-fit, not propagation checkpointing.
Wrong source commit/environment/binary/science/catalog/fixture/mask/hash or budget
blocks before a fit. Real stores require the exact separately reviewed preflight
SHA. Default one fit allows integral integration audit before the remaining101.

Historical PR16/17 parameters, graph and RNG conventions are comparable. Labels
and E deliberately differ, so historical prediction/accuracy equality is **not**
expected and must not be asserted. No-intervention parity is tested synthetically
inside each backend; native Cython continuation is covered by the existing
copy-only RNG-probe test. No cross-backend bitwise equality is required.

## Commands (execution still requires authorization)

Prepare only, from a clean committed checkout with frozen local fixtures:

```powershell
python -m experiments.causal_selection preflight --output .cache/causal-selection-preflight --fixture-cache .cache/convergence-stage-b-preflight/fixtures
```

After explicit approval of that payload SHA:

```powershell
python -m experiments.causal_selection init --preflight .cache/causal-selection-preflight --root .cache/causal-selection
python -m experiments.causal_selection run --root .cache/causal-selection --reviewed-preflight APPROVED_SHA --max-runs 1
python -m experiments.causal_selection status --root .cache/causal-selection
```

Only after the first fit is audited, run remaining tasks with --max-runs101 and
the same approved SHA. Status/recovery never authorize another experiment.
