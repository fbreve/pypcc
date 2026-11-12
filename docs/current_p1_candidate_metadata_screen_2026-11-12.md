# P1 candidate-data metadata screening — **review only**, no dataset selected

Date: 2026-11-12. Scope: **public source metadata**, not actual fixture validation, feature processing or model inference. This is an independent research branch based on production commit `ea69f690208620d5563c8efece6b2e04369f52c0`, referring to the unapproved P1 proposal [#258](https://github.com/fbreve/pypcc/pull/258) and Git-only source inventory [#257](https://github.com/fbreve/pypcc/pull/257). The authoritative editable candidate list is [`current_p1_candidate_metadata_registry_2026-11-12.json`](current_p1_candidate_metadata_registry_2026-11-12.json). **No dataset has been chosen for the study.**

## Correction to an incomplete historical screen

The previously discussed four C2/G1 datasets (Banknote, Sonar, PageBlocks, Spambase) are **not** an adequate inventory of exposure. In production `docs/performance_campaign_plan.md`, Git blob `35944c493d38046e71790ca4521cc42ce0aa570d`, a **20-dataset** experiment is explicitly documented:

Wine, Digits, Iris, BreastCancer, BalanceScale, GermanCredit, Diabetes, Sonar, Glass, Vehicle, Ionosphere, PenDigits, Satimage, Spambase, PageBlocks, Banknote, BloodTransfusion, Yeast, Segment and QSARBiodeg. This source also describes distinct historical label-mask seeds and graph preprocessing. Never present these 20 as a previously unseen external validation cohort.

The tool verifies this actual source table, including **all 20 names and aliases**. Its exhaustive coverage claim is deliberately **false**: it cannot search original MATLAB research, earlier publications, the user's own local `.cache` stores, old non-default branches or collaborator experiments. An external reviewer must perform and sign an independent exposure audit **before** any candidate is considered newly unexamined.

## Three UCI *candidates* (no decision on selection)

| UCI ID / exact task | Publicly reported shape | Source and license | Screening caution |
| --- | --- | --- | --- |
| [212 — Vertebral Column, **3-class**](https://archive.ics.uci.edu/dataset/212/vertebral-column) | 310 rows, 6 numeric features, 3 classes, smallest reported class 60 | [UCI dataset and DOI](https://doi.org/10.24432/C5K89B), **CC BY 4.0** | Orthopedic clinical-origin data; only the three-class `column_3C_weka.arff` is proposed. The alternate **two-class** task must not be mixed. Verify privacy/anonymization, patients' independent records and applicable research permissions. Never present PCC classification as medical advice |
| [545 — Rice (Cammeo and Osmancik)](https://archive.ics.uci.edu/dataset/545/rice%2Bcammeo%2Band%2Bosmancik) | 3,810 rows, 7 numeric features, 2 classes | [UCI dataset and DOI](https://doi.org/10.24432/C5MW4Z), **CC BY 4.0** | Proposed `Rice_Cammeo_Osmancik.arff`; class counts and exact bytes need checking. Grain morphology features overlap the **type** of Raisin's features |
| [850 — Raisin](https://archive.ics.uci.edu/dataset/850/raisin) | 900 rows, 7 numeric features, 2 classes, 450 rows/class as described | [UCI dataset and DOI](https://doi.org/10.24432/C5660T), **CC BY 4.0** | Proposed archive `Raisin_Dataset.zip`; verify actual file inside ZIP, repeated feature rows and potential duplicate records. Grain morphology shares a domain with Rice |

Sources: UCI pages above, inspected 2026-11-12. They report no missing values, but the actual contents have **not** been downloaded or parsed. The license listings are **descriptions of source licensing**, not separate vetting of permitted patient data reuse, file authenticity, proper attribution, derivative redistributions or downstream rights. The two plant-grain cases may be too similar for the desired breadth; this is explicitly an unresolved scientific design concern.

**Reading order is deterministic:** ascending UCI numeric ID (212, 545, 850). It is an audit ordering, **not an outcome-based ranking or permission to pick the first three**. No substitutions or backups are authorized. If a candidate fails exposure, privacy, duplication, completeness, graph or resource checks, leave a documented exclusion and request an explicit **new P1 scientific revision before replacing it**. Do not silently nominate another after running PCC.

## Required sequence before even marking candidates selected

1. **Exposure / provenance:** independently cross-check earlier PCC papers (including original MATLAB), dissertations, local archives, all relevant branches, course/research scripts and external publications. The current repository's 20-dataset table is only one source. Record every match/uncertainty; a candidate shown to have influenced the study is **not** a fresh confirmatory dataset. The review must explicitly address prior inspection of these three candidates.
2. **Data integrity without fitting:** verify exact UCI archive bytes and SHA-256, license and attribution, target schema, class counts, actual missing/non-finite entries, duplicates/near duplicates and subject/group correlation, while keeping oracle labels out of propagation and any outcome-sensitive selection. This metadata and fixture inspection is **not** a PCC run.
3. **Design variety / feasibility:** separately review whether two grain-morphology datasets and one clinical-origin dataset meet the intended scientific scope; approve preprocessing and graph parameters **before examining algorithm outcomes**. Maintain the deliberate **dense padded `neib_list`** and symmetrically undirected active graph.
4. **Scientific P1 approval:** freeze the specific dataset+target+transform mapping, sampled labels per class, independent native propagation seeds, data version, measures, exclusions, uncertainty procedure, proposed 50,000 iterations, eight checkpoints and resource rationale. Current P1 proposal #258 still has **zero selected datasets** and no scientific SHA. Scientific approval must be explicit and separate from this metadata-list checksum.
5. **P2 compiled-execution preflight:** bind exactly the final three independent observer Git commits, Python/NumPy/Numba/Cython/native compiler and compiled extension hashes, graph/label/particle ordering, and per-case memory/disk/CPU budgets. Generate a **new execution-specific SHA** and get explicit human authorization **before any real data fit**. Do not reuse the C2/G1 frozen hashes or #257's source SHA.

The automated auditor `python -m experiments.review_p1_candidate_metadata` **does not import NumPy, scikit-learn or PyPCC**, connect to the internet, download datasets, inspect their contents, build graphs, generate label masks or run particle propagation. Its output `candidate_checksum_not_approval` only binds the proposed JSON text. Approval fields must remain `false`/`null`, `selected_dataset_ids=[]`, `datasets_downloaded=0` and `model_fits_executed=0`.

### Decision after metadata screening

**GO only for independent human screening of these public dataset descriptions. NO GO for selecting datasets as novel or freezing P1; NO GO for P2 completion; NO GO for real-data C2, G1 or this prospective P1 experiment.** Leave this PR draft/unmerged alongside #253/#255/#256/#257/#258, and leave `master` untouched.
