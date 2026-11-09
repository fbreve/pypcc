# Original PyPCC repeated-fit non-native phase audit — 2026-11-09

This is a **permanent compact scientific summary proposal**, not raw evidence and not a production optimization. Original frozen production source: [`b4c100bd22eeff1979a1a581bc73682b8c3705b8`](https://github.com/fbreve/pypcc/commit/b4c100bd22eeff1979a1a581bc73682b8c3705b8). Separate scientific [PR #225](https://github.com/fbreve/pypcc/pull/225).

- Original source/CPU measurements: [run 42667018276](https://github.com/fbreve/pypcc/actions/runs/42667018276), native Linux [raw 13698080622](https://github.com/fbreve/pypcc/actions/runs/42667018276/artifacts/13698080622) and Windows [raw 13696628520](https://github.com/fbreve/pypcc/actions/runs/42667018276/artifacts/13696628520). Both native measurement jobs completed successfully; the first aggregator rejected a platform-dependent **decoded source-text SHA** (Git output newline translation). Frozen original Git source byte guards, original native six state hashes and original RNG were independently validated on each host; no production source changed.
- Strict independent **artifact reuse**, without rerunning native fits: [run 42667891453](https://github.com/fbreve/pypcc/actions/runs/42667891453), [report and machine-readable decision artifact 13695929663](https://github.com/fbreve/pypcc/actions/runs/42667891453/artifacts/13695929663), **SUCCESS**. Requires exactly 36 cohorts × 4 original/observed alternating pairs × 8 full fits per arm × 2 hosts = **4,608 timed full fits**. Both OS true state/predictions/native RNG replay, graph order, independent old-state arrays and invalid-seed rollback checked.
- The actual source was not optimized. The traced variant inserted only 15 disjoint **wall-clock** timestamps into a separately loaded full original `pcc.py` module. Timed original unmodified complete-fit **wall and process CPU** batches were the baseline; traces carry measurement overhead. Windows short process CPU samples quantize to zero and must not be treated as no-cost CPU.

## Selected measured timings, native Numba

All values milliseconds **per fit**, from the final independent report; *baseline total* is uninstrumented full-fit wall time, while named phases are observed instrumented wall time, not standalone CPU savings.

| OS | Corpus / iterations | Baseline full fit | Create particles | Create nodes | Native backend | Normalize own-degree |
|---|---|---:|---:|---:|---:|---:|
| Linux | Breast Cancer / 0 | 0.2699 | 0.0225 | 0.0262 | 0.0594 | 0.0160 |
| Linux | Digits / 0 | 0.5034 | 0.0276 | 0.0942 | 0.0655 | 0.0353 |
| Linux | Digits / 500 | 2.0137 | 0.0289 | 0.0903 | 1.5911 | 0.0390 |
| Linux | Symmetric ring 100,000 × 50 / 0 | **29.8115** | 1.9125 | **11.4166** | 0.1639 | **6.2082** |
| Linux | Symmetric ring 100,000 × 50 / 5 | 32.3140 | 1.9249 | 11.4824 | 2.3743 | 6.9845 |
| Windows | Breast Cancer / 0 | 0.2626 | 0.0223 | 0.0262 | 0.0562 | 0.0152 |
| Windows | Digits / 0 | 0.4412 | 0.0279 | 0.0900 | 0.0592 | 0.0321 |
| Windows | Digits / 500 | 1.9057 | 0.0295 | 0.0911 | 1.4785 | 0.0408 |
| Windows | Symmetric ring 100,000 × 50 / 0 | **47.7197** | 2.7521 | **14.8278** | 0.1611 | **15.5559** |
| Windows | Symmetric ring 100,000 × 50 / 5 | 50.1238 | 2.7372 | 14.8362 | 0.5658 | 17.8277 |

The remaining non-native costs include input validation, class discovery/known-label mapping, scratch allocation, unlabeled-node label assignment, inverse output mapping and the small observed instrumentation overhead. These are stored as **fifteen distinct recorded stages** in the raw artifacts, not collapsed into native propagation. The compact table intentionally does not assign these remaining times to a single unmeasured operation. The 100k×50 ring is a controlled stress case, not proof that large-class problems dominate ordinary use.

## Engineering interpretation and roadmap

1. **Theoretical CPU saving is workload dependent.** On brief 100k×50 ring fits, node creation and own-degree normalization are expensive in absolute time. At 500 steps on real Digits, the original native Numba propagation already consumes most of the fit; a large percentage microphase gain may produce little whole-fit gain.
2. Do not repeat **#199**'s exact one-line in-place `np.divide(out=self.owndeg)` change. That already failed preregistered whole-normalizer CPU/memory gate. A new hypothesis must isolate a different cost and preserve exact per-call array ownership/exception rollback; the whole original `owndeg` output is externally inspectable.
3. Do not retry **#202**'s one-hot node initialization or **#222**'s class-discovery candidate just on the strength of the same type of microphase evidence: both received valid whole-fit scientific NO-GO.
4. A promising **next observational target** is the precise cost of `node.label[unlabeled] = np.argmax(node.dominance[unlabeled, :], axis=1)`, including the temporary advanced-indexed dominance matrix on many-class graphs. It is already recorded as the `unlabeled_assignment` phase in the raw #225 JSON but not displayed separately in this compact summary. Read the frozen stage before proposing a different, source-paired candidate.
5. Any new optimization requires precommitted full `fit_predict` CPU GO on Linux and Windows (not just a faster fragment), no material regressions on real long trajectories, exact six states, RNG, original stop conditions, all backed by full 4-platform CI. Keep the intentional dense padded int64 `neib_list` unchanged.

**Disposition: observational SCIENCE COMPLETE; no production candidate accepted.** Closed science PR evidence belongs in this docs-only archive, but the raw GitHub Actions artifacts have finite retention and are **not** backed up merely by these links. They should be exported separately if long-term raw-data preservation is required.
