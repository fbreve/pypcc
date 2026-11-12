# P1: independent exposure check + offline source-fixture quality gate

**2026-11-12. Research-only. No dataset has been selected or used in a PCC fit.** This stage follows [candidate registry PR #259](https://github.com/fbreve/pypcc/pull/259), scientific proposal [#258](https://github.com/fbreve/pypcc/pull/258) and source-only P0 [#257](https://github.com/fbreve/pypcc/pull/257). Its purpose is to distinguish (a) what has genuinely been checked from (b) missing UCI byte-level checks and (c) the much larger scientific prior-exposure audit.

## A. Confirmed and limited source-exposure evidence

The current PyPCC repository has a published **20-dataset historical benchmark inventory** at production `docs/performance_campaign_plan.md`, Git blob `35944c493d38046e71790ca4521cc42ce0aa570d`. The P1 candidates were **not** among the 20 named entries, but that is **not** evidence that they never appeared in other research.

The separate original [MATLAB Particle Competition and Cooperation repository](https://github.com/fbreve/Particle-Competition-and-Cooperation), commit `f13c4ac5c2cc925bedf9fa5983af742502243a4f`, tree `71b003825d76e6e70649e93d8947c6aed43f2dc1`, contains 19 tracked entries. A case-insensitive literal scan across **all 12 tracked text/code files** (.m, .asv, .c, .md, .txt) for `vertebral`, `rice`, `raisin`, `cammeo`, `osmancik`, `kecimen`, `besni`, and `column[_ ]?3c` found **zero literal occurrences**. The inspected text file paths and exact Git blob SHAs are frozen in [`current_p1_original_matlab_exposure_audit_2026-11-12.json`](current_p1_original_matlab_exposure_audit_2026-11-12.json). In the original repo's `example.m`, Wine is the sample dataset; `benchmark.m` compares Wine and Digits. Neither candidate-name absence nor the 20-dataset list excludes use under a different name, in binary `.mat` fixtures, older Git commits, local artifacts, collaborator research, papers or unpublished studies.

The **read-only** verifier `python -m experiments.p1_original_matlab_exposure --matlab-repo PATH` checks exact original commit, its underlying tree, all tracked-file count, 12 recorded text blob SHA values, literal search results and the mandatory `novelty_approved=false` disposition. It **never fetches** the remote itself and does not open binary dataset blobs. The dedicated CI provides a pinned public Git reference to this local verifier without loading or fitting any candidate data. This is a **tracked-source-only**, not an exhaustive literature review.

No assertion of "no previous use in all publications" is made; previous MATLAB paper PDFs, full bibliographies, older Git refs and the user's local archives still require independently documented screening.

## B. Official UCI candidate file identity (do NOT silently substitute)

| Candidate | Metadata page | Archive/source name expected | Exact task and classes |
| --- | --- | --- | --- |
| **212: Vertebral Column, 3-class** | https://archive.ics.uci.edu/dataset/212/vertebral-column | `column_3C_weka.arff` inside source ZIP or as an explicitly supplied file | 310 rows, 6 numerical features; DH 60, SL 150, NO 100. **Do not substitute `column_2C_weka.arff`** |
| **545: Rice (Cammeo/Osmancik)** | https://archive.ics.uci.edu/dataset/545/rice%2Bcammeo%2Band%2Bosmancik | `Rice_Cammeo_Osmancik.arff` | 3810 rows, 7 numerical features, two declared classes. Actual support must be checked |
| **850: Raisin (Kecimen/Besni)** | https://archive.ics.uci.edu/dataset/850/raisin | UCI lists `Raisin_Dataset.zip`; proposed inner workbook `Raisin_Dataset.xlsx` (exact inner member still requires inspection) | 900 rows, 7 numerical features, 450/class |

UCI pages report no missing values and a CC BY 4.0 license. This is a **source metadata claim only**; it does not substitute for verifying the actual archive bytes, data processing rights, privacy or citation obligations. Rice and Raisin are both grain-morphology data, which may undermine prospective diversity. Vertebral's clinical origin warrants independent ethical/data-lineage consideration and no claims of clinical utility. None is chosen for P1.

## C. Offline no-PCC integrity inspector (ready, without UCI bytes)

Source: `experiments/p1_offline_candidate_integrity.py`.

This **only reads explicitly provided local files**. It does not connect to the internet, import a PyPCC backend, build kNN graphs, sample labeled masks, inspect model predictions or execute PCC. Supports exact **ARFF** for 212 and 545, and **XLSX** (requires optional `openpyxl`) for 850, directly or within a ZIP using `--member` for ambiguous archives. It checks hard archive/decompression bounds, unsafe ZIP paths, target task, numeric feature finiteness, row and attribute counts, declared class support, exact duplicate full rows, repeated raw feature vectors and identical features with conflicting class labels. It records the SHA-256 of the *actual supplied source archive and selected member*.

Examples for future **no-fit local verification only**, after authentic files have been sourced and archived:

```sh
python -m experiments.p1_offline_candidate_integrity --dataset-id 212 --file /local/vertebral_column.zip --member column_3C_weka.arff --output /local/report-212.json
python -m experiments.p1_offline_candidate_integrity --dataset-id 545 --file /local/Rice_Cammeo_Osmancik.arff --output /local/report-545.json
python -m experiments.p1_offline_candidate_integrity --dataset-id 850 --file /local/Raisin_Dataset.zip --member Raisin_Dataset/Raisin_Dataset.xlsx --output /local/report-850.json
```

The third `--member` is a **working example**, not a verified UCI ZIP listing: inspect the archive's real member path first and pass that path verbatim. The tool will block if the format/target differs or if multiple supported data files are found without an explicit member. It neither copies the dataset into the repository nor transmits real source bytes. Do **not commit actual datasets, raw rows, patient records or private archive paths** into Git.

**Important limits:** detected repeated raw feature vectors or mixed-class duplicates are *descriptive quality flags*, not prespecified exclusions. Near-duplicates and within-subject correlation are **not checked** and require a separately reviewed threshold/procedure. A supplied local ZIP's SHA-256 proves *local file identity*, **not official UCI authenticity** unless independently compared to a trusted official value or a reviewed source-pinned acquisition record. XLSX requires optional openpyxl installed and pinned for an eventual P2 execution environment; its reader rejects formula cells and multiple ambiguous worksheets. Invalid file/format/schema or missing optional dependency **fails closed**.

## D. What actually ran

The offline verifier's CI uses **synthetic ARFF/XLSX/ZIP fixtures only**, including malformed, duplicate/conflicting, wrong 2-class target and unsafe/ambiguous cases. CI also replays the source-only original MATLAB file audit against the exact pinned public Git commit. **No UCI candidate archives have been downloaded or parsed in this task**: direct retrieval was unavailable in the current environment. Consequently **there are no real class-count, duplicate, near-duplicate, missingness, archive SHA or authenticity measurements to report yet**, regardless of source metadata.

Review report fields always retain `datasets_selected=0`, `execution_authorized=false`, `P2_execution_preflight_sha256=null` and `model_fits_executed=0`.

### Decision

**GO: reproduce original MATLAB tracked-text exposure check and validate the offline reader on synthetic fixtures.** **NO GO: declare the three datasets fresh, approve candidate selection, freeze the P1 science, claim UCI integrity, or execute any real PCC propagation.** Next evidence required is legitimate source archive acquisition, explicit no-fit data integrity measurement, wider publication/local archive/previous-code exposure review and independent scientific decision on Rice/Raisin domain overlap and clinical-origin suitability. P2 execution-specific preflight and maintainer approval remain separate future gates.
