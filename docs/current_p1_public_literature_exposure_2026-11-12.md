# P1 — scoped published-literature and UCI source-lineage audit (2026-11-12)

**Review status:** `SCOPED_PUBLIC_TEXT_CHECK_ONLY_NOT_PROOF_OF_NOVELTY`. This is a supporting appendix to the unapproved [P1 scientific proposal #258](https://github.com/fbreve/pypcc/pull/258). It documents a **bounded and independently traceable** examination of named datasets in three publicly accessible published experimental sections; it is NOT an exhaustive review of previous PCC studies, the author's thesis, local experiments, aliases, unpublished notebooks or all collaborators.

See the [machine-readable evidence register](current_p1_public_literature_exposure_2026-11-12.json), whose three per-paper entries include full-text URLs, section locations, dataset lists, and explicit negative-claim limits.

## 1. Published experimental sections actually checked

| Paper (published) | Exact section examined | Named experimental datasets |
| --- | --- | --- |
| [Breve, Zhao & Quiles (2015), Neurocomputing](https://www.fabriciobreve.com/artigos/2015/Neurocomputing2015.pdf), *Particle Competition and Cooperation for Semi-Supervised Learning with Label Noise* | Section 4, Figures 1–6 and Tables 1–6 | **Iris**, **Wine**, synthetic four-class Gaussian, **g241c**, **Semeion Handwritten Digit**, **Optical Recognition of Handwritten Digits** |
| [Guerreiro & Breve (2017), ICCSA](https://www.fabriciobreve.com/artigos/2017/iccsa2017lucas.pdf), *Analyzing and Inferring Distance Metrics on the Particle Competition and Cooperation Algorithm* | Section 3 and results tables | **Iris**, **Wine**, **Banknote Authentication** |
| [Leticio et al. (2025), VISAPP](https://www.scitepress.org/Papers/2025/132670/132670.pdf), *Graph Convolutional Networks and Particle Competition and Cooperation for Semi-Supervised Learning* | Section 5.1, Table 1 | **g241c**, **g241n**, **Digit1**, **USPS**, **COIL**, **COIL2**, **BCI** |

**Bounded outcome:** no literal mention of the three proposed P1 candidate dataset names, **Vertebral Column (UCI 212), Raisin (UCI 850) or Wholesale customers (UCI 292)**, was observed in the experimental-dataset descriptions of these **three specific papers**. The publication's complete text and task details should still be independently checked before invoking claims beyond those sections.

**Important correction:** the historical `docs/performance_campaign_plan.md` 20-dataset inventory does **not** include **g241c** or **Semeion**, although the 2015 paper explicitly reports their use. Therefore, the 20-set list is a *version-/campaign-specific inventory*, **NOT an exhaustive registry of all datasets historically studied with PCC**. A source name absent from this list cannot be deemed independently novel.

The SCItePress 2025 experimental table was obtained from published PDF **text extraction**; screenshot retrieval of this publisher PDF was blocked with HTTP 403. The 2015 and 2017 PDFs were also rendered/screenshot-checked at relevant pages. Do not imply visual verification of the inaccessible 2025 screenshot.

## 2. Official UCI data origin and reuse status

| Candidate | Public UCI information verified | What is NOT known |
| --- | --- | --- |
| [Vertebral Column](https://archive.ics.uci.edu/dataset/212/vertebral+column), DOI **10.24432/C5K89B** | Orthopedic **patient** biomechanical features collected by Henrique da Mota during residency at GARO, Lyon, France; three-class version has 100 Normal, 60 Hernia and 150 Spondylolisthesis; UCI donated 2011-08-08; **CC BY 4.0** | The numeric source does not demonstrate original patient/visit identifiers, independence between patients or clinical reuse/privacy clearance |
| [Raisin](https://archive.ics.uci.edu/dataset/850/raisin), DOI **10.24432/C5660T** | 900 raisins (450 Kecimen, 450 Besni), seven image-derived morphology attributes; **2020** introductory publication, UCI donated 2023-08-13; **CC BY 4.0** | Independent image capture/physical grain/acquisition batch or cross-sample independence is unverified from the provided tabular columns |
| [Wholesale Customers](https://archive.ics.uci.edu/dataset/292/wholesale+customers), DOI **10.24432/C5030X** | 440 customers of a wholesale distributor, six annual monetary spending attributes, `Channel` and `Region`; UCI donated 2014-03-30; metadata roles designate **Region as target** and **Channel as feature**; **CC BY 4.0** | Proposed **Channel as target** is explicitly a derived research task, requiring approval; no organization/customer ID or common purchasing group information to verify identity independence |

All three pages license the published dataset under CC BY 4.0 and require proper attribution. A public license is **not evidence of privacy clearance, independent patient/client/grain acquisition or universal reuse authorization**. The exact official archive/member hashes and unmodified source-content quality checks are pinned separately in draft source audit [#260](https://github.com/fbreve/pypcc/pull/260).

## 3. Remaining scientific eligibility decision (NO approval here)

1. Complete **an independently documented review of additional key PCC papers and theses**, including relevant 2009–2012, 2019–2023 and 2026 research, and any aliases/source names; compare with the original MATLAB work where available.
2. Have the project maintainer review whether these datasets or outcomes were previously used in **local/unpublished MATLAB/Python trials** or coauthor studies. Public searching cannot establish this; if used, mark as historically exposed and limit claims to exploratory analysis.
3. Assess clinical provenance and responsible reuse, whether any **patient/customer/physical-grain grouping** can be verified, and what scientifically justifiable assumptions remain when no group identifiers exist.
4. Have a reviewer explicitly approve/reject **Wholesale's derived Channel target** versus the source UCI metadata `Region` target, plus preprocessing, undirected dense graph, dataset-order and exploratory seed/analysis protocol.
5. Only then decide whether to seal a reviewed P1 scientific protocol. A **distinct compiled P2 preflight SHA and explicit execution approval** are required before *any* real-data PCC fit.

**Status as recorded:** publication exposure **partially reviewed**, subject/group independence **not proven**, `dataset_selected_ids=[]`, P1 scientific SHA **null**, P2 execution SHA **null**, `execution_authorized=false`, zero PCC fits. Keep PR #258 DRAFT / UNMERGED and master unchanged.
