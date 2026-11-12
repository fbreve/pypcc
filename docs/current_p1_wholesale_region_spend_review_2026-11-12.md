# P1 — Wholesale Customers: Region × Channel and spending profile, SOURCE ONLY

**Date: 2026-11-12. Status: descriptive source-data assessment, not scientific preregistration approval or experimental execution.** The candidate remains **UNSELECTED**. No PCC propagation, source graph, label-mask generation, model fitting, changes to production, row deletion or feature normalization occurred.

This is a follow-up to the [draft PR #260](https://github.com/fbreve/pypcc/pull/260) P1 data-origin audit. The candidate is official [UCI 292 Wholesale Customers](https://archive.ics.uci.edu/dataset/292/wholesale+customers) (DOI 10.24432/C5030X). The official ZIP hash remains `647e6a61683ed23f48c7b04e1e2be78835eca8758b32267603fc42a1479dbfb8` and selected member `Wholesale customers data.csv` has SHA-256 `c3d018c643565b85cee733c4a2ac76dd76e080e857cb23f0ccfcc2e15a6c17ef`. These are reproducible observed HTTPS-source identifiers, not an independent UCI digital signature.

The source metadata lists **Region** as its target. For possible P1 use, we have proposed a **different derived task**, binary **Channel** (1 HoReCa vs 2 Retail), based on six yearly numeric spending fields. **Region is excluded from the proposed predictive feature vector** and inspected only in this audit to reveal domain confounding. This target reinterpretation has **not been approved**.

## A. Source-label geographic association (no model fitting)

| Region, original UCI code | Channel 1 HoReCa | Channel 2 Retail | Total | HoReCa share |
| --- | ---: | ---: | ---: | ---: |
| 1 — Lisbon | 59 | 18 | 77 | 76.62% |
| 2 — Oporto | 28 | 19 | 47 | 59.57% |
| 3 — Other | 211 | 105 | 316 | 66.77% |
| **Total** | **298** | **142** | **440** | **67.73%** |

**Descriptive Pearson chi-square = 4.349142**; **Cramér's V = 0.0994204**, calculated as `sqrt(chi_square/440)` for a 3×2 contingency table. This is a **weak association in this sample**, not proof of geographic independence, no statistical significance claim, no causal inference, and not a license to assume unrelated customers or transactions. Region IDs are coarse geography **not customer identifiers or acquisition batches**.

## B. Six unmodified spending attributes

The following quantiles are **nearest-rank** values on original monetary units, with `Q_p = sorted_values[ceil(p*n)-1]`. "High" outliers are a descriptive count above `Q3 + 1.5*(Q3-Q1)`, using nearest-rank Q1/Q3. No winsorization, transformation or outlier exclusion was made.

| Annual spending feature | Median | 95th percentile | Maximum | Rows above Tukey high fence |
| --- | ---: | ---: | ---: | ---: |
| Fresh | 8,475 | 36,817 | 112,151 | 20 |
| Milk | 3,620 | 16,784 | 73,498 | 28 |
| Grocery | 4,754 | 23,998 | 92,780 | 24 |
| Frozen | 1,517 | 9,927 | 60,869 | 43 |
| Detergents_Paper | 813 | 12,034 | 40,827 | 30 |
| Delicassen | 964 | 4,478 | 47,943 | 27 |

The distributions show **unequal scales and heavy upper tails**. They materially affect unnormalized Euclidean distances and graph neighborhoods; nevertheless, no graph or prediction has been computed in this audit. A scientifically reviewed P1 would need to freeze a single transformation and graph rule **before observing any PCC outputs**. Possible methodological choices for reviewer deliberation (NOT approved or compared by PCC here) include raw-feature standardization versus `log1p` of nonnegative expenditures followed by a documented scaling rule. An unsupervised transform fitted using all observations may be compatible with a **transductive** setup if explicitly predeclared; whether that is the intended design must be decided before P1 freeze. Avoid changing treatment of extreme values after seeing particle behavior.

## C. Procedure and reproducibility

The independent script `experiments/p1_wholesale_region_spend_profile.py` obtains only official UCI HTTPS 292 source bytes through the source-quality reader, **verifies the previously recorded exact ZIP/member hashes first**, and computes aggregate statistics using only Python's standard library. No scientific stack (NumPy/sklearn/Numba/Cython), node labels sampling, graph construction or fitted model is imported. Its output contains only region/channel counts and six aggregate summaries, not customer-level rows.

Source profile conventions were implemented and tested on **synthetic-only** fixtures before the first official source profile ran. The immutable source-quality result manifest `docs/current_p1_wholesale_region_spend_source_results_2026-11-12.json` records all aggregate quantiles, source hashes, the descriptive contingency/association and explicit `execution_authorized=false`/zero fits. The separate workflow `.github/workflows/research-p1-alternative-domains-source-only.yml` re-downloads the file, checks its exact previously observed ZIP/member SHA and **fails closed on aggregate statistics drift** with `--expect-diagnostics` against the committed manifest. Synthetic adversarial tests also reject changes to the Region/Channel feature separation and invented protocol approval. This is a **source-quality replay gate only**, not a P1 scientific SHA or a P2 execution-preflight SHA.

## D. Review decision

**GO: retain Wholesale as an additional, distinct-domain candidate for independent P1 scientific review.** It has no exact duplicate or identical-feature/class-conflict patterns in the 440 source rows, and Region is only weakly associated with the derived Channel label by this one descriptive statistic.

**NOT YET GO for inclusion in P1:** source-candidate eligibility and scientific novelty in older papers, unpublished local stores and collaborator work have not been independently established; customer/group-level identities are absent; Channel is an unapproved derived target (the UCI metadata names Region); preprocessing and graph-building choices remain unsealed; source-level geographic association does not prove independence. **Neither Rice nor Raisin is approved,** and using both would still duplicate the grain-morphology domain. P1 remains DRAFT, there is no approved P2 execution preflight SHA, no permission for any real-data PCC fit and no master merge.

