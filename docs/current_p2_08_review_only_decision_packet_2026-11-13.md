# PyPCC — P2-08 non-authorizing technical review packet (2026-11-13)

**Scientific scope:** The P1 approved protocol remains EXPLORATORY S1–S6 ONLY, immutable SHA-256 `f0111f153ade3d72191c107ace039e1835e5c27db2cf4b3df8d610b41b2f4166`. This branch is stacked on P2-07D #273, and its pre-existing original source and production regressions have passed. **No production edit or merge to `master`**.

## What this artifact is, and is not

This is a **read-only predecision dossier**, not the P2-08 final execution preflight. It does **not** download the three UCI ZIPs, load real source rows, create original kNN graphs or masks, run synthetic PCC or propagate any real particles. Instead it fetches exactly **11 already-passing, source-free JSON/metadata GitHub Actions artifacts** from #266/#267/#263/#272/#273, each pinned by run ID, artifact ID, name and GitHub-recorded artifact ZIP SHA-256 in `docs/current_p2_08_review_only_evidence_registry_2026-11-13.json`. The `download-artifact` action validates downloaded archive digests; the dossier separately hashes every extracted JSON body and checks exact content, project-science source identities, backend-specific results and known no-execution flags.

**Crucial cryptographic distinction:** the output `review_packet_SHA256_NOT_P2_execution_SHA` is a stable fingerprint of this **review-only evidence packet**, useful for discussing its exact content. It MUST NOT be stored in `P2_execution_preflight_sha256`, treated as a fully approved P2 manifest, or used as consent to begin the 72 scientific fits. The latter field intentionally remains **null**. A separate final technical P2 manifest with a new exact SHA cannot be finalized until the resource and environment gates are actually resolved, and it then needs explicit human approval of that exact final SHA.

## Evidence inspected, without repeating source experiments

| P2 technical gate | Previously passing candidate evidence | Current FINAL execution gate |
| --- | --- | --- |
| **01 — Code/native environment** | Original sources, 13 archived candidate wheels, P2-01F newly compiled Cython modules with compiler hashes, partial linker/SDK evidence | **BLOCKED:** complete durable host/OS/compiler/SDK/runtime byte closure absent |
| **02 — Official sources** | #266 Linux/Windows exact original UCI ZIP/member, 310/900/440 original rows, target class supports, feature/class row order and target SHA | **Candidate verification PASS**, not an execution permission |
| **03 — Transformed features** | #266 exact raw and approved z-score / log1p+zscore little-endian float64 and source order SHA across OS | **Candidate verification PASS** |
| **04 — Undirected dense kNN** | #267 real-source `build_graph`, identical ordered dense `int64 neib_list` with **-1 padding**, exact full graph SHA across OS | **Candidate verification PASS** |
| **05 — Masks / native seeds** | #267 eight original-source PCG64 masks per dataset, 24 mask SHA total, 72 backend/domain-separated original seeds **not used** | **Candidate verification PASS** |
| **06 — Observers** | #263 original 21/21 synthetic within-backend parity tests each OS; #273 original source/pinned new candidate native 50k synthetic parity | **Candidate parity PASS; final native execution lock BLOCKED** |
| **07 — Resource / export budget** | #272 candidate one-fit-at-a-time cap policy; #273 nine 50k artificial original-vs-observer pairs/OS and four forced CPU/RSS/disk/wall kills per OS | **Candidate synthetic enforcement PASS; real-data process resource proof BLOCKED** |
| **08 — Final sealed SHA and human GO** | This report identifies exact PASS candidates and blockers | **NOT SEALED / NO EXECUTION APPROVAL** |

P2-02 through P2-05 intentionally worked with verified original source data **in their earlier read-only/preparation stages**: the graph/mask preflight #267 generated *real* graphs and 24 real label masks in memory, but **no PCC fits**. This P2-08 review **does not re-generate** any of them; it verifies only the permitted SHA aggregates already archived. The P2-06 and P2-07D experiments executed **synthetic** PCC fits; therefore a blanket claim that no PCC propagation has occurred anywhere in P2 would be false. **Zero fits on any of the three real UCI datasets** is the relevant invariant.

The 72-case original P1 order remains original UCI registry `212 → 850 → 292`, replicates `0..7`, backend `NumPy → Numba → Cython`, sequential, `early_stop=False`, `max_iter=50000`, with milestones `[0,1,10,100,1000,5000,25000,50000]`. No scientific resampling or outcome-dependent replacement. Dense padded neighbor list is preserved, never replaced with CSR.

## Five mandatory decisions before producing a FINAL P2-08 technical execution SHA

1. **Durable execution host:** Select and content-address the final OS/toolchain/environment. Linux CPython 3.12.16, with archived P2-01F new native modules and P2-01D candidate wheels, is a *candidate primary* only. Windows CPython 3.12.10 is a crosscheck. Actual GCC/MSVC driver hashes and candidate SDK libraries exist but **complete OS images/runtime dependency byte closure are not sealed**. These GitHub-hosted runner images are mutable and archive retention expires around **2026-12-13**.
2. **Actual-graph resource feasibility:** A defended bound for 50k PCC on real knn graph degree distribution and memory/CPU/disk is still lacking. Synthetic ring+hub runs and 100ms polling are not a proof of actual-graph performance, native specialization stability or a production-grade process-tree supervisor.
3. **Approve/revise numerical operational policy:** The #272 candidate bounds (8 GiB host RAM, 12 GiB free disk, 2 GiB whole process peak cap, 30min fit wall/60min fit CPU, 48h overall, 1 GiB scratch, bounded logs/JSON and total evidence, one fit at a time) require a deliberate human decision and reliable enforcement. If any parameter changes, create a **new reviewed policy SHA**.
4. **Create the distinct final manifest:** Only *after* the above gates pass, create an exact P2-08 technical manifest with required native/source/graph/mask/seed hashes, final host, limits, final monitoring and failure/export contract, and source-free complete preflight logs.
5. **Receive separate exact-SHA execution approval:** The project maintainer must explicitly authorize the exact **new final P2 SHA** before the first real-source PCC propagation. The earlier P1 approval is not sufficient and cannot be inferred from a generic `prossiga`.

**CI behavior:** First two jobs run **six adversarial, stdlib-only synthetic metadata tests** on Linux/Windows, refusing fake approvals, missing artifact identity and false final native parity. The third job downloads the 11 exact prior aggregate reports, cross-checks original P1 source/masks/graph digests, old source/feature content hashes, original synthetic tests and candidate 50k resource evidence, and computes the *review-only* checksum. Only source-free aggregates, source Git hashes, original row/class *counts*, and cryptographic digests are saved. No original source rows, graph adjacency arrays, label mask indices, individual particle state, images, or raw source files are exported.

**Required output no-GO conditions, regardless of CI success:**
`P2_execution_preflight_sha256=null`,
`P2_execution_authorized=false`,
`all_eight_P2_execution_gates_pass=false`,
`final_execution_preflight_manifest_created=false`,
`real_PCC_fits_executed=0`.

This work should stay **DRAFT and stacked**. A passing review-only CI documents current evidence; it does **not** automatically permit a P3 experiment or merge into production.