# P1 — Scientific decision packet: review-ready, **not frozen** (2026-11-12)

**No selected datasets, no model fits, no scientific GO and no P2 execution authorization.** This packet gives the maintainer an explicit decision surface rather than continuing exploratory data-quality audits. It complements, **without overriding**, [draft P1 design #258](https://github.com/fbreve/pypcc/pull/258) and source-only [audit PR #260](https://github.com/fbreve/pypcc/pull/260).

Machine-readable [decision packet](current_p1_scientific_decision_packet_2026-11-12.json); stdlib-only structural validator `python -m experiments.review_p1_scientific_decisions`. Passing validates internal consistency and an exact current-master graph source SHA. It **never approves** the science or generates a P1/P2 seal.

## 1. Proposed three-domain shortlist (NOT selected)

| Candidate | UCI source | n / d / classes | Proposed primary target | Source-integrity checks and current issue |
| --- | --- | --- | --- | --- |
| **Vertebral Column, 3-class** | [UCI 212](https://archive.ics.uci.edu/dataset/212/vertebral+column) | 310 / 6 / 3 | ARFF three-class `Class`: Hernia 60, Normal 100, Spondylolisthesis 150 | Original official ZIP/member SHA recorded, finite numeric data and no exact/thresholded-near pairs in prior study-specific audit. Clinical-origin reuse and patient independence remain unresolved |
| **Raisin** | [UCI 850](https://archive.ics.uci.edu/dataset/850/raisin) | 900 / 7 / 2 | `Class`: Besni 450, Kecimen 450 | Same quality check; imaging/batch independence unverified; retain **one** morphology domain rather than selecting Rice as another distinct domain |
| **Wholesale Customers** | [UCI 292](https://archive.ics.uci.edu/dataset/292/wholesale+customers) | 440 / 6 / 2 | **Derived** `Channel`: HoReCa 298, Retail 142 | Exact feature vectors are unique, no label conflicts; original UCI metadata table lists `Region` as target. **Must explicitly approve reinterpretation**; customer/batch independence unavailable |

Raisin is tentatively preferred over Rice for a smaller 50,000-iteration run (900 rather than 3,810 instances), *not because the PCC performed better on either*. A science reviewer may reject this choice if it compromises applicability; choosing the smaller sample for cost is a design trade-off, not proof of superior representativeness. **Website Phishing** was set aside because its nine features are coded categories and the official tabular data contain **48 distinct feature patterns associated with conflicting classes**; it is not interchangeable with continuous numeric-source graph assumptions.

The actual [UCI quality results](https://github.com/fbreve/pypcc/blob/research/current-p1-offline-fixture-and-matlab-audit-2026-11-12/docs/current_p1_official_uci_quality_results_2026-11-12.json), [thresholded near-pair screen](https://github.com/fbreve/pypcc/blob/research/current-p1-offline-fixture-and-matlab-audit-2026-11-12/docs/current_p1_official_uci_near_duplicate_results_2026-11-12.json), [alternate-domain check](https://github.com/fbreve/pypcc/blob/research/current-p1-offline-fixture-and-matlab-audit-2026-11-12/docs/current_p1_alternative_domain_quality_results_2026-11-12.json) and [Wholesale descriptive profile](https://github.com/fbreve/pypcc/blob/research/current-p1-offline-fixture-and-matlab-audit-2026-11-12/docs/current_p1_wholesale_region_spend_source_results_2026-11-12.json) are **source-quality evidence only**; they are not prior PCC outcomes, nor approval of independent patient/client/grain samples. None of the candidate names occurred in the historical 20-set benchmark inventory or the pinned 12 textual files of the public original MATLAB repository. A bounded review of [three public experimental papers (2015, 2017, 2025)](current_p1_public_literature_exposure_2026-11-12.md), expanded with [six additional published experimental works (2009–2016) and one 2026 arXiv preprint](current_p1_additional_public_papers_2026-11-12.md), found no candidate-name mention **in those TEN specified experimental source sections only**. Crucially, the 2015 paper used **g241c and Semeion**, which are **absent from the 20-item PyPCC campaign registry**: that registry demonstrably is **not** an exhaustive historical PCC publication exposure list. Other publications, historical source, theses, local MATLAB results and collaborators remain unchecked. No complete novelty, ethical approval or subject-independence claim is authorized.

### Source file pins — for reviewer verification, NOT new P1 hashes

| ID | Official ZIP SHA-256 | Selected data member SHA-256 |
| --- | --- | --- |
| 212 | `899ed1d137ceb4c36d59fa35f3cbab39ec3b78fd3ca1fdaf2fae155459c1372f` | `cf768854cf4cf42b3a3b792a239a81ebfd4fbfafc25fbe2a19d15b3f73249879` |
| 850 | `5d516c040923fd154ecf85e31b6e1b5a096724fa0e68d588c33fb5a2f0c3e5db` | `9370b5f749a572a0bbeae15c93bf2e15c859b06d2626ae401b844c0a925fa553` |
| 292 | `647e6a61683ed23f48c7b04e1e2be78835eca8758b32267603fc42a1479dbfb8` | `c3d018c643565b85cee733c4a2ac76dd76e080e857cb23f0ccfcc2e15a6c17ef` |

Source SHA values are observed direct-UCI HTTPS file hashes. They are **not independently published signed UCI hashes** and do not authenticate private/unpublished copies. No source data are loaded by this packet or its validator.

## 2. Proposed preprocessing — decisions for review, not approved

**Common rule (proposed):** retain every original record, preserve source row order, reject unrecognized/missing/nonfinite source values, do not impute, clip, balance or remove high-spending customers, and do not use class labels to fit transformations. In the transductive setup, an **unsupervised** feature transform fitted on the entire public feature matrix may be permissible **if that use of all unlabeled X values is explicitly approved before runs**.

- **Vertebral, Raisin:** per-column population z-score `(X - mean(X[:,j])) / std(X[:,j], ddof=0)` across all source rows, with zero-variance rejection.
- **Wholesale:** verify all six spending columns nonnegative, apply `log1p` elementwise, then the same population z-score. This is proposed in response to *source distribution*, not to PCC performance: very unequal raw spending scales and upper tails were documented in #260.
- **No candidate** is permitted to switch preprocessing after any graph/particle result. The exact floating implementation, column order, transformed byte hash and versions must be fixed **before P2**. A reviewer can instead prescribe another preprocessing rule, but then this packet must be revised, independently revalidated and only subsequently sealed.

These rules are **not yet executed on real datasets**, even without PCC. Their reasonableness and compatibility with the paper's intended experiment remain review questions.

## 3. Existing production graph contract — verified by source, not by running it

Exact master `ea69f690208620d5563c8efece6b2e04369f52c0`; `pcc_graph.py` Git **blob SHA-1** `3e8943462555b4feafbd162125b51e41f29d7ab7`.

Proposed (not approved) parameterization uses the production routine **unchanged**:

```python
neib_list, neib_qt = build_knn_graph(
    X, k_nn=10, metric="minkowski", p=2, n_jobs=1
)
```

The source implementation fits `NearestNeighbors(n_neighbors=11, algorithm="auto", metric="minkowski", p=2)`, discards neighbor column zero as the presumed self-neighbor, symmetrizes the temporary sparse adjacency by adding its transpose and returns **dense ordered int64 `neib_list` with inactive padding `-1`**, and `neib_qt`. The temporary **CSR construction structure is not a proposed replacement** for the dense propagation representation. This must remain an *undirected* graph; no directed PCC extension is proposed.

**Crucial reproducibility condition:** `algorithm="auto"`, library versions and nearest-neighbor tie order can affect the graph bytes. P2 must check self exclusion, absence of active self-loops, strict undirected symmetry, positive degrees, all active neighbor values, source and transformed feature hashes, and the **exact ordered dense array hashes** on the actual execution environment. An array or order mismatch blocks execution; it is not fixed by a silent graph rewrite. `ParticleCompetitionAndCooperation.build_graph()` stores the builder's output directly, whereas the alternative `set_graph()` normalizes inactive padding to zero. Therefore **the exact installation path is another protocol/P2 decision**: declare it and hash what each backend actually consumes. Do not confuse unused padding differences with active-edge differences, but also do not silently claim byte-identical graph fixtures if different installation paths are mixed.

No real graph has been generated by this protocol-preparation PR.

## 4. Planned P1 scientific design remains unchanged, unapproved

The existing P1 draft still proposes **three datasets × eight stratified masks × three sequential native backends = 72 fits**, three labeled nodes per class, `max_iter=50,000`, `early_stop=False` and observation milestones `[0,1,10,100,1000,5000,25000,50000]`.

The predeclared primary quantity is the **fraction of initially unlabeled nodes whose predicted class differs at 5,000 and 50,000 iterations**, calculated **within the same backend, fixed graph and label mask**. Backends may use distinct native RNG streams, so bitwise inter-backend trajectory identity is *not* an assumption. The descriptive unit is *dataset × label mask × backend*, not independent individual nodes. We have not fixed confirmatory practical effect thresholds or statistical uncertainty estimates, so the status remains **exploratory**, not confirmatory.

No masks, seeds, graph hashes or P2 compiled-binary hashes have been generated in this PR. These are reviewer and execution preflight prerequisites, not tasks to backfill from earlier C2/G1 runs.

## 5. Finite remaining decision gates

1. **Scientific eligibility:** decide the three actual dataset versions, check former paper/unpublished MATLAB/collaborator exposure and record any historical contamination explicitly; consider smaller Raisin versus larger Rice on scientific grounds, not just resource convenience.
2. **Target and provenance:** approve or reject derived Wholesale Channel versus declared UCI Region; assess clinical data reuse, CC BY attribution, subject/customer/batch independence and inability to establish external identifiers.
3. **Feature transformation:** select an exact preprocessing rule and implementation, whether full transductive X fitting is allowed, missingness handling and no extreme-value exclusions.
4. **Graph generation:** approve exact `k_nn`, Euclidean metric and builder path; specify self, symmetry, tie, order, padding and cross-platform byte-hash audits without changing the deliberate dense representation.
5. **Experimental identity:** freeze P1 mask sampler and 3/class/8-mask plan, backend-native seed table and eight milestones **before** any real propagation.
6. **Inference and budget:** decide exploratory/confirmatory claim and uncertainty procedure; justify the proposed sample size, fixed 50k horizon and CPU/RAM/disk estimates.
7. **Two separate approvals:** only after all previous decisions may a human independently approve a **new immutable scientific P1 SHA**. Then P2 must verify source/compiled binaries, exact graph/labels and resource budgets, issuing a **different P2 execution SHA**. **Neither SHA exists now.** Explicit approval of the exact P2 SHA is required before one real PCC fit.

**Decision as of this packet:** **GO for reviewer discussion only**, **NO GO to freeze P1, build real-data PCC graphs, generate label masks, start a model fit or merge research branches**. Keep [PR #258](https://github.com/fbreve/pypcc/pull/258) DRAFT/UNMERGED. The existing P0 observer PRs [#253](https://github.com/fbreve/pypcc/pull/253), [#255](https://github.com/fbreve/pypcc/pull/255), [#256](https://github.com/fbreve/pypcc/pull/256) remain separate.

