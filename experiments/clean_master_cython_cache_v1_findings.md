# V1 clean-master result: default-disabled Linux regression, optimization still beneficial

**Historical v1 result — preserved, not superseded or relabeled as a V2 success.** Pre-registered protocol: [clean_master_cython_cache_protocol.md](clean_master_cython_cache_protocol.md).

## Evidence and build isolation

[Workflow 42282512047](https://github.com/fbreve/pypcc/actions/runs/42282512047) passed **all four independent native-compiler OS×dataset jobs and mandatory summary**. It compiled untouched `master` Cython independently from the opt-in cache version in separate checkouts, importing each `.so` or `.pyd` through separate processes and retaining full-state NPZ files, binary SHA-256 checksums and Git revisions.

- Untouched baseline commit: `2902009c80b8edd3fc5f4961769444563fcf5b8a`.
- V1 candidate commit: `8e97a99ea1f48e371541a3375141b7eff59ec828`.
- **240/240 timed independent native calls, 80/80 triplets with exact predictions, positions, particle labels, strengths, distances, dominance and raw ownership, and exact complete propagation stopping metadata.**
- Raw CSV/state artifacts: Linux Digits **13535852769**, Linux Breast Cancer **13536780708**, Windows Digits **13536805761**, Windows Breast Cancer **13536805676**. Mandatory complete report **13535693263**.
- First CI attempt [42282193972](https://github.com/fbreve/pypcc/actions/runs/42282193972) failed on Windows because the experiment progress logger used a Unicode arrow unsupported by the runner's active console code page; it did **not** identify a dynamics or compiler failure. Corrected to ASCII and fully rerun as listed above.

## Native-propagation time savings vs separately compiled untouched master

Positive = faster than untouched master; negative = regression; **four-seed median of per-seed medians**.

| OS / dataset | 20k monitoring, cache disabled | 20k monitoring, cache enabled | 100k-cap actual stop, cache enabled |
|---|---:|---:|---:|
| Linux Digits | **-30.22%** | **+30.31%** | +31.22% |
| Windows Digits | +3.43% | +34.01% | +34.14% |
| Linux Breast Cancer | -1.22% | -1.47% | -1.68% |
| Windows Breast Cancer | -0.39% | +3.30% | -8.35% |

The cache-on Digits result is promising, but **the frozen default-off regression limit (-8%) fails** on Linux Digits by a wide margin. The v1 candidate must NOT be merged, because default-disabled behavior would become much slower on one tested platform. The first same-binary experiment on stacked draft #177 could not reveal this relative-to-untouched-master regression.

## Investigation and V2 response

The main code-level difference plausibly responsible is the **new `cache_enabled` runtime test executed inside every particle's hot movement loop**, even when the flag is false, which can inhibit compiler optimizations. This is a hypothesis, not instruction-profile proof. The repair is preregistered separately in [clean_master_cython_cache_v2_protocol.md](clean_master_cython_cache_v2_protocol.md): preserve original master's hot function verbatim and introduce a specialized cached step without the flag inside either hot loop; select once per propagation iteration. V2 must use fresh seeds 8101/8209/8303/8407 and repeat all 80 three-way cross-binary gates.

**No `master` change; V1 remains a failed performance experiment regardless of V2 outcome.**
