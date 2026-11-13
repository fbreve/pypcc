# PyPCC — P2-02A official ZIP/member byte identity gate (2026-11-13)

**Scientific scope:** exclusively the three *already selected* official UCI sources in the frozen P1 JSON, approved exploratory-only by scientific SHA-256 `f0111f153ade3d72191c107ace039e1835e5c27db2cf4b3df8d610b41b2f4166`. This is a technical source-integrity check, **not** a request to redo dataset selection or a PCC execution authorization. Source confidence is based on HTTPS and observed SHA-256; there is **no independent UCI signed checksum**.

The P1 source registry uniquely fixes original order and exact members:

| Approved registry index | UCI ID | Dataset | Required file | P1 official ZIP / member SHA |
| --- | --- | --- | --- | --- |
| 0 | 212 | Vertebral Column 3-class | `column_3C_weka.arff` | Already pinned in frozen P1 protocol |
| 1 | 850 | Raisin | `Raisin_Dataset.xlsx` | Already pinned in frozen P1 protocol |
| 2 | 292 | Wholesale Customers | `Wholesale customers data.csv` | Already pinned in frozen P1 protocol |

## P2-02A byte-only validation

- Independently verify canonical SHA of the **entire frozen P1 scientific JSON** matches the approved P1 manifest before downloading any source. Reject changed data set list/order, filename, URL, or approval fields.
- Use strictly official exact `https://archive.ics.uci.edu/static/public/...` URLs from that P1 manifest, refuse unexpected URL/redirect and HTTP response problems. A network failure **blocks the audit**, never triggers substitution or alternate data.
- Limit downloaded ZIP bytes to 16 MiB; reject unsafe ZIP names, traversal, duplicate entries, symbolic links, encryption, too many members and excessive decompressed size. Read selected bytes into ephemeral runner memory only, **never extract/write the source member to disk**.
- Compare **full original ZIP SHA-256** to P1 **before** opening a member. Select exactly the named file by unique basename and compare **its original untransformed byte SHA-256**. The Raisin official ZIP wraps `Raisin_Dataset.zip` containing `Raisin_Dataset.xlsx`; verify outer and inner member constraints, record observed inner-wrapper ZIP SHA separately.
- Validate on Linux and Windows Python 3.12 independently, with six adversarial synthetic ZIP/URL/digest tests. CI artifact contains **JSON hashes, HTTP metadata, selected member metadata and source-check proof only**, never patient, grain or customer feature rows, label columns, workbook, ZIP bytes or record-level exports.

## What this stage intentionally does not claim

This is **P2-02A (file-byte identity) only**. The P1 record already describes Vertebral 310 rows / 3 classes, Raisin 900 rows / 7 features, and Wholesale 440 rows / derived Channel target; the P2-02A script **does not parse ARFF, XLSX or CSV records**, verify worksheet/row ordering, source class supports, data column order, or target exclusion. Those items remain for **P2-02B** and must not be represented as freshly validated. P2-03 scaling/feature-transformation, P2-04 undirected ordered dense-neighbor graphs and P2-05 masks are likewise pending. Previous P1 source-quality audits (#260) are historical evidence, not an automatic substitute for the P2 gated checks.

If any official archive or selected member SHA drifts, P2-02A must **FAIL CLOSED**, preserving the original P1 protocol rather than accepting new hashes or reselecting data. No production source edits, no observer changes, no compilation, no synthetic or real PCC execution, no graph or mask. `P2_02_entire_gate_pass=false`, `P2_execution_preflight_sha256=null`, `P2_execution_authorized=false`, `real_PCC_fits_executed=0`. The PR is stacked research-only **DRAFT**; merging into master is outside scope.

**Evidence status:** script, adversarial tests and two-platform CI are prepared. Actual source hashes are not claimed independently verified until both platform jobs have completed successfully.
