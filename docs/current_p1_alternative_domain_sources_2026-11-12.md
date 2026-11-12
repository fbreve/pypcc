# P1: review-only *distinct-domain* alternatives to Rice/Raisin redundancy

**2026-11-12. Not a P1 amendment, dataset selection or run approval.** This extension to [draft source-quality PR #260](https://github.com/fbreve/pypcc/pull/260) independently screens two alternatives to including both Rice and Raisin as separate application domains in proposed study [draft #258](https://github.com/fbreve/pypcc/pull/258). Both remain **UNSELECTED**. No PCC code was changed; no PCC fitting, graph construction, labeled-mask sampling or prospective result inspection is authorized.

## Historical and scientific rationale

The exact PyPCC production source `docs/performance_campaign_plan.md` lists 20 previously studied datasets and **does not name Wholesale Customers or Website Phishing**. The original MATLAB public tracked-code literal search covered 12 text/code files on pinned commit `f13c4ac5c2cc925bedf9fa5983af742502243a4f`, but **did not** search alternative candidate names in earlier paper PDFs, local MATLAB files, collaborators' work, historical branches or other private sources. Therefore **neither candidate can yet be called previously unseen/confirmatory**. The original P1 three-dataset design is still a proposal; this addition does not change it.

The original UCI files for Vertebral, Rice and Raisin were screened by [source-only PR #260](https://github.com/fbreve/pypcc/pull/260) and showed zero duplicate feature vectors and zero near pairs under the **declared** `1e-6..1e-2` per-source RMS standardized Euclidean thresholds. These limited source-quality results do not establish statistical independence of patients, objects or batches. In particular **Rice and Raisin both originate from morphology extracted from grain images**; presenting both as independent application domains would be scientifically misleading without justification.

### New UCI candidates for data-integrity review only

| UCI official source | Data type and original documentation | **Candidate task** | Important review caveats |
| --- | --- | --- | --- |
| **292 Wholesale customers**: https://archive.ics.uci.edu/dataset/292/wholesale+customers, DOI `10.24432/C5030X` | 440 wholesale customers; six numerical annual spending columns, plus `Channel` and `Region` categorical codes; UCI describes 298 Horeca and 142 Retail customers; CC BY 4.0, no missing values | **Propose Channel as the 2-class target**; numeric X in exact order `Fresh, Milk, Grocery, Frozen, Detergents_Paper, Delicassen`; explicitly **drop Region** as categorical non-target. Do NOT include the Channel target in X | UCI's current metadata table labels **Region** as its declared target (Channel as feature), even though Channel classification is a studied derived problem; this is a **prospectively proposed task reinterpretation**, requiring approval. No customer/customer-group identifiers; Region and Channel may be associated; spending distributions are heavily skewed. Freeze any preprocessing/normalization later, before outcomes |
| **379 Website Phishing**: https://archive.ics.uci.edu/dataset/379/website+phishing, DOI `10.24432/C5B301` | 1,353 websites; nine discrete `{-1,0,+1}`-coded features; 702 phishing, 548 legitimate and 103 suspicious labels reported; CC BY 4.0, no missing values | Three original categories, `Result` target with exact codes `-1/0/+1`. Retain all nine feature codes as **categorical descriptions for SOURCE inspection only** | Integer codes represent categorical status, **not continuous physical measurements**. Euclidean distance on them or reusing the unchanged numeric PCC graph is *not scientifically justified without a separately reviewed encoding/distance*. Distinct website rows may collapse to same nine-dimensional code vector, including conflicting labels. Never treat a feature-vector collision as proof of the same URL |

Source citations are the official UCI pages linked above; dataset descriptions, target coding and class counts are *not* independently confirmed until each official ZIP has been read in the no-fit CI. No individual URL, patient, customer row or feature vector should be printed or uploaded.

### Offline/ephemeral data-quality audit

`experiments/p1_alternative_domain_source_quality.py` is a standard-library-only, isolated audit. It accepts only explicitly listed official UCI HTTPS source ZIP URLs (ID 292 or ID 379), validates safe ZIP paths and CRC, refuses unexpected/ambiguous member names, requires exact historical column spelling and order, verifies finite numeric codes, row counts and fixed candidate-class supports; checks repeated feature vectors, exact full-row duplicates and **conflicting labels for identical feature descriptions**. For Wholesale, confirms `Region` is **excluded** from six numeric features, not mistakenly used as a predictor or target; publishes only aggregate Region counts. For Website Phishing, validates each feature is exactly `-1, 0 or +1` and explicitly flags categorical/Euclidean mismatch. Raw source bytes are kept solely in the transient GitHub runner and are **not uploaded**. The script cannot build a PCC graph or run any algorithm.

`experiments/test_p1_alternative_domain_source_quality.py` tests both tasks on **synthetic-only** fixtures and rejects altered column order, nonfinite/invalid codes, invalid ZIP paths, unexpected source members and cross-class feature collisions while preserving those collisions as quality warnings rather than opportunistically removing samples. The four-platform Linux/Windows × Python3.10/3.12 CI does not install the scientific stack; official UCI source-only acquisition is separate on two Ubuntu jobs.

After verifying both source files, record exact **raw ZIP SHA-256, selected member SHA-256, actual class support, missing/collision/duplicate counts and any exceptions**. Do **not** rename, resample, drop duplicate records, rebalance, change graph/distance representation or revise P1 based on PCC outcomes.

### Review decision gates

- Only after independently documented source integrity, previous-publication/local research novelty, and customer/group/website provenance can either dataset be considered **eligible for a scientific P1 revision**.
- A task with integer-coded categories should not automatically satisfy the proposal's **continuous numeric feature** assumption. Website Phishing needs an explicit, independently reviewed categorical representation/graph mechanism before it can be treated as compatible.
- Wholesale's **Channel** target must be approved as a derived task rather than being misrepresented as the default UCI metadata target. The six numeric annual spending features are a potentially more straightforward graph-compatible alternative, subject to clearly fixed normalization and skew/outlier treatment.
- If an independent reviewer recommends three domains, the prospective choices could include the clinical-origin Vertebral source, **one of** Rice/Raisin, and a distinct-domain candidate like Wholesale. This is a **review option**, **not** a dataset selection or new frozen P1 protocol.
- **No P1 scientific approval, no P2 execution preflight digest, and no real-data PCC model fits.** Keep all research PRs DRAFT/UNMERGED, production unchanged.


## Source-only results on the real official UCI archives

The independent audit [GitHub Actions #43305443733](https://github.com/fbreve/pypcc/actions/runs/43305443733) retrieved both datasets directly by HTTPS from the official UCI archive and verified the actual source names and class counts. An initial strict import intentionally refused UCI 379 because the original ARFF spells one field **`popUpWidnow`** rather than the documentation's `popUpWindow`; the verifier was corrected to accept *only* the observed original spelling. All subsequent source checks passed. Raw ZIPs, user/customer observations and website feature rows were not committed.

The complete source-quality evidence is [`current_p1_alternative_domain_quality_results_2026-11-12.json`](current_p1_alternative_domain_quality_results_2026-11-12.json).

| Exact source metric | Wholesale UCI 292 | Website Phishing UCI 379 |
| --- | ---: | ---: |
| Number of rows | **440** | **1,353** |
| Candidate feature dimensions | **6 continuous expenditure fields** | **9 categorical integer codes** |
| Class support | Channel 1: **298**, Channel 2: **142** | Result -1: **702**, 0: **103**, +1: **548** |
| Distinct source feature patterns | **440** | **676** |
| Extra occurrences of repeated feature patterns | **0** | **677** |
| Extra duplicate full records (features plus target) | **0** | **629** |
| **Feature patterns mapped to more than one class** | **0** | **48** |

Observed official ZIP SHA-256:
- Wholesale: `647e6a61683ed23f48c7b04e1e2be78835eca8758b32267603fc42a1479dbfb8`
- Website Phishing: `fa029c3b079b6bfb4f0833c0ec94a778a20522161b028273e81ac3b93ee8bfee`

Selected official data-member SHA-256:
- Wholesale `Wholesale customers data.csv`: `c3d018c643565b85cee733c4a2ac76dd76e080e857cb23f0ccfcc2e15a6c17ef`
- Website Phishing `PhishingData.arff`: `ec1c4594cb152e99bd0820855b19eff78a367d623ac5e7f419e6ba3c47dd2432`

The **48** Website Phishing collisions are *distinct feature-description patterns containing conflicting class labels*, not 48 mislabeled objects and not an assertion that there are only 48 unavoidable errors. The counts arise from nine coarse categorical codes and do **not** demonstrate identical source URLs. Nevertheless, any deterministic predictor using *only* those identical feature codes must assign the same class to conflicting rows, so these records limit how completely that representation can explain the labels, regardless of the PCC backend. No classification accuracy, uncertainty or propagation was measured.

**Source-quality interpretation:** Wholesale is a more straightforward *prospective third-domain candidate* than Website Phishing, but it is not automatically scientifically fit for P1: the **Channel** task is intentionally different from the **Region** target in the UCI metadata, `Region` was excluded from features and has not been assessed for customer-cluster confounding, repeated customer records cannot be ruled out without customer IDs, and spend features are skewed. Website Phishing should **not** be treated as continuous numeric Euclidean data, and its conflicting class patterns need a separate rationale before inclusion. Neither is yet selected or deemed newly validated.

The source auditor now optionally accepts `--expect-record` and **fails closed** if the official UCI ZIP/member hashes, fixed feature mapping, class distribution, duplicate counts or conflicting-label counts differ from these source-only observations. That check establishes reproducibility of this *research-quality inspection*, not P1 science approval or a P2 execution seal. A later official UCI revision would require a new, explicitly reviewed version, not silent overwriting.

**Next review:** independent prior-publication/MATLAB/local-archive exposure audit for Wholesale; independent decision on Channel task and source grouping, followed by scientific review of a *possible* three-domain design (Vertebral, one grain source, Wholesale). Rice/Raisin overlap remains a concern. No model fit has been run; source-only audits are not C2/G1/P1 scientific outcomes.
