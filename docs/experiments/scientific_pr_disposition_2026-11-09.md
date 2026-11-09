# Scientific pull request disposition and PyPCC optimization roadmap — 2026-11-09

> Administrative scientific archive; this document changes **no** runtime source, propagation kernel, graph, RNG, public API or benchmark conditions. It records experimental outcomes; an Actions green check means completeness/semantic success, **not** necessarily an optimization GO.

## Provenance and policy

- Repository: [`fbreve/pypcc`](https://github.com/fbreve/pypcc), observed production `master`: [`b4c100bd22eeff1979a1a581bc73682b8c3705b8`](https://github.com/fbreve/pypcc/commit/b4c100bd22eeff1979a1a581bc73682b8c3705b8).
- Pre-audit: **35 open draft PRs**, all inspected against their descriptions and applicable independent Actions results. **25** scientifically concluded/promoted PRs may close *without merge*; **10** need separate disposition and stay open.
- Closing a scientific PR preserves GitHub discussion, commits and links; it is not a production merge. Branch deletion is not part of this cleanup. No reopening, rebasing or rerunning failed precommitted gates simply to seek favorable samples.
- Science GO may be extracted only into a separate minimal candidate reviewed and tested on Linux/Windows Python 3.10/3.12, as already done for #192/#195/#198/#201/#205/#207/#211/#216. Negative experiments are archived as valid results.
- The original **dense padded int64 neighbor representation** is a deliberate CPU-memory tradeoff. Do not propose CSR replacement solely for smaller memory use; preserve exact topology/order, all six native states and full-fit CPU evidence.
- The GitHub Actions artifact links below may expire according to retention settings, **even while PR pages remain accessible**. This index preserves conclusions and immutable source/run identifiers, *not* the raw data bytes. Back up any artifacts needed permanently to a durable research archive before expiration; do not mislabel linked artifacts as permanent.

## 25 concluded scientific PRs — administrative closure without merge

| PR (original head SHA) | Final scientific disposition | Rationale retained | Frozen Actions evidence |
|---|---|---|---|
| [#190](https://github.com/fbreve/pypcc/pull/190) (`76a99a931169`) | **Observation** | Original-master Linux/Windows 216-fit native CPU baseline; graph, initialization and propagation phases measured. | [run 42413774663](https://github.com/fbreve/pypcc/actions/runs/42413774663) |
| [#191](https://github.com/fbreve/pypcc/pull/191) (`b4d47573eecf`) | **Promoted via #192** | Ordered CSR-offset dense-neighbor construction was extracted into minimal production PR #192; original neighbor order and padded int64 layout retained. | [run 42441878460](https://github.com/fbreve/pypcc/actions/runs/42441878460) |
| [#193](https://github.com/fbreve/pypcc/pull/193) (`3011ab3b27d2`) | **NO-GO** | No single sklearn nearest-neighbor algorithm meets frozen universal compatibility and performance gates; retain `algorithm='auto'`. | [run 42443482733](https://github.com/fbreve/pypcc/actions/runs/42443482733) |
| [#194](https://github.com/fbreve/pypcc/pull/194) (`33594bc48910`) | **Promoted via #195** | Conditional diagonal cleanup for loop-free edge-index graph construction; independent production #195 merged. | [run 42446970589](https://github.com/fbreve/pypcc/actions/runs/42446970589) |
| [#196](https://github.com/fbreve/pypcc/pull/196) (`b1485c419266`) | **NO-GO** | Native CSR index widths by themselves failed the frozen whole-builder memory/time gate; superseded by independent V2 #197. | [run 42486726302](https://github.com/fbreve/pypcc/actions/runs/42486726302) |
| [#197](https://github.com/fbreve/pypcc/pull/197) (`584f56d6b8c2`) | **Promoted via #198** | V2 native CSR index and degree reuse met frozen memory/CPU gate; minimal production #198 merged. | [run 42487584743](https://github.com/fbreve/pypcc/actions/runs/42487584743) |
| [#199](https://github.com/fbreve/pypcc/pull/199) (`79e98b494d5f`) | **NO-GO** | In-place `owndeg` normalization matched exact states but failed preregistered CPU/memory promotion gate. | [run 42498623224](https://github.com/fbreve/pypcc/actions/runs/42498623224) |
| [#200](https://github.com/fbreve/pypcc/pull/200) (`79a20293ce44`) | **Promoted via #201** | Vectorized active-neighbor validation in `set_graph` met scientific GO; independently reviewed production #201 merged. | [run 42516277670](https://github.com/fbreve/pypcc/actions/runs/42516277670) |
| [#202](https://github.com/fbreve/pypcc/pull/202) (`12be306dc976`) | **NO-GO** | One-hot labeled dominance initialization failed preregistered complete initializer CPU improvement gate. | [run 42518319011](https://github.com/fbreve/pypcc/actions/runs/42518319011) |
| [#204](https://github.com/fbreve/pypcc/pull/204) (`d38db2921828`) | **Promoted via #205** | Sparse known-label mapping improved; minimal production #205 merged. | [run](https://github.com/fbreve/pypcc/actions/runs/42546330247) |
| [#206](https://github.com/fbreve/pypcc/pull/206) (`04d77f1a7e05`) | **Promoted via #207** | Guarded vectorized inverse class-label mapping improved; minimal production #207 merged. | [run](https://github.com/fbreve/pypcc/actions/runs/42548096743) |
| [#208](https://github.com/fbreve/pypcc/pull/208) (`fe7e45807ebb`) | **Observation** | Original native fit phase attribution: 10k-step Cython propagation contributes roughly 93–98% to full fit; graph rebuilding may dominate other workloads. | [run 42549738065](https://github.com/fbreve/pypcc/actions/runs/42549738065) |
| [#209](https://github.com/fbreve/pypcc/pull/209) (`cb7dd84189d9`) | **Observation** | Source-exact graph-stage and `set_graph`/`build_graph` attribution; NN search dominates original kNN10 Digits build, dense packing already small. | [run 42552117106](https://github.com/fbreve/pypcc/actions/runs/42552117106) |
| [#210](https://github.com/fbreve/pypcc/pull/210) (`b7924f8261bb`) | **Promoted via #211** | Loop-free kNN CSR diagonal-cleanup guard passed frozen gate; minimal production #211 merged. | [run](https://github.com/fbreve/pypcc/actions/runs/42553580578) |
| [#212](https://github.com/fbreve/pypcc/pull/212) (`1eff85fe3c47`) | **Observation** | Native early-stop versus fixed-iteration original-source baseline and exact prefix/stop checks completed. | [run 42557758034](https://github.com/fbreve/pypcc/actions/runs/42557758034) |
| [#213](https://github.com/fbreve/pypcc/pull/213) (`f1c26c15f09f`) | **NO-GO** | Numba per-node row-maximum monitoring cache failed whole-fit CPU gate despite exact state/RNG parity. | [run 42561479069](https://github.com/fbreve/pypcc/actions/runs/42561479069) |
| [#214](https://github.com/fbreve/pypcc/pull/214) (`a1c84b4b5e5b`) | **NO-GO** | Numba direct dense neighbor row indexing did not meet >=5% full-fit CPU gate in both OS. | [run 42562781487](https://github.com/fbreve/pypcc/actions/runs/42562781487) |
| [#215](https://github.com/fbreve/pypcc/pull/215) (`485119261d88`) | **Promoted via #216** | Source-exact persistent Numba JIT disk-cache study passed gate; four-decorator production #216 merged. | [run](https://github.com/fbreve/pypcc/actions/runs/42606049712) |
| [#217](https://github.com/fbreve/pypcc/pull/217) (`5473b25a91a6`) | **Observation** | Original first-PCC-prediction full process profile confirmed useful merged Numba JIT disk cache, not a new code change. | [run 42609705820](https://github.com/fbreve/pypcc/actions/runs/42609705820) |
| [#218](https://github.com/fbreve/pypcc/pull/218) (`90a07f0d2b9e`) | **Observation** | 20 repeated fits on one model and graph confirmed amortized benefit of JIT cache already merged by #216. | [run 42617888885](https://github.com/fbreve/pypcc/actions/runs/42617888885) |
| [#219](https://github.com/fbreve/pypcc/pull/219) (`140c9ad8087e`) | **Observation** | Original native Numba Callgrind dynamic-instruction hotspot study completed; instructions are not direct CPU-time evidence. | [run 42625092167](https://github.com/fbreve/pypcc/actions/runs/42625092167) |
| [#220](https://github.com/fbreve/pypcc/pull/220) (`bf88acdea841`) | **NO-GO** | Private Numba RNG-pointer reuse failed >=5% Windows Digits full-fit CPU gate; also depends on unstable private API. | [run 42629786968](https://github.com/fbreve/pypcc/actions/runs/42629786968) |
| [#221](https://github.com/fbreve/pypcc/pull/221) (`97b76ae8865d`) | **NO-GO** | Replacing greedy uniform expression with random()*total retained native parity, but <5% CPU benefit. | [run 42637441249](https://github.com/fbreve/pypcc/actions/runs/42637441249) |
| [#222](https://github.com/fbreve/pypcc/pull/222) (`ba3e4d7defc5`) | **NO-GO** | Class discovery microphase faster by >90% on large sparse inputs, but original full-fit performance gate failed. | [run 42643993915](https://github.com/fbreve/pypcc/actions/runs/42643993915) |
| [#223](https://github.com/fbreve/pypcc/pull/223) (`af8faa0b757b`) | **Observation** | Same model/graph repeated-fit phase profile complete; 100k×50 ring initialized nodes/particles in ~13–16 ms, Digits 500-step native propagation ~78% of traced fit. No reuse candidate yet. | [run 42660626479](https://github.com/fbreve/pypcc/actions/runs/42660626479) |

## Ten PRs retained open for a separate research/documentation decision

| PR | Why it remains open |
|---|---|
| [#18](https://github.com/fbreve/pypcc/pull/18) | Unfinished mechanistic pilot (C2 not run); requires separate decision on continued scientific work. |
| [#170](https://github.com/fbreve/pypcc/pull/170) | Documentation-only software-paper/license/provenance readiness audit; review for an eventual docs-only merge. |
| [#182](https://github.com/fbreve/pypcc/pull/182) | Base opt-in NumPy trajectory observer for dependent research #183–#188; retain while dependency lineage is consolidated. |
| [#183](https://github.com/fbreve/pypcc/pull/183) | Continuous per-step trajectory and stopping study, stacked on #182. |
| [#184](https://github.com/fbreve/pypcc/pull/184) | Canonical reciprocal k3 versus k10 Digits mobility study, stacked on #183. |
| [#185](https://github.com/fbreve/pypcc/pull/185) | Undirected disconnect/bridge mobility study, stacked on #184. |
| [#186](https://github.com/fbreve/pypcc/pull/186) | Nested bridge-dose mobility study, stacked on #185. |
| [#187](https://github.com/fbreve/pypcc/pull/187) | Wine/Breast Cancer bridge-dose generalization, stacked on #186. |
| [#188](https://github.com/fbreve/pypcc/pull/188) | Canonical labeled-home degree placement study, stacked on #187. |
| [#189](https://github.com/fbreve/pypcc/pull/189) | Documentation-only research audit of #182–#188; review independently, do not merge stacked experiment code. |

## Next optimization roadmap (priorities)

1. **Finish administrative cleanup:** post each concluded PR's immutable reason and this index, close without merging; review #170/#189 separately as documentation, and do not disrupt stacked lineage #182–#188.
2. **Decompose repeated-fit non-native overhead:** build a new prospective observational profile distinct from #222 and #223 to isolate validation, label preparation, distance-table fill/allocation, dominance/ownership fill, scratch arrays, normalization and label output; true uninstrumented CPU plus measured observer overhead, Linux/Windows.
3. **Re-evaluate amortization and safe reuse:** if attribution shows a meaningful cost under real repeated-fit workloads, design a separate source-exact A/B candidate. Never mutate state from a prior returned fit, break exception rollback or reuse a graph with changed topology. Measure process CPU, complete fit, memory, state/stop/RNG and previous-fit object identity.
4. **Native hot path for long fits:** focus on measured Numba/Cython propagation hotspots; earlier attempts #213/#214/#220/#221 have complete negative evidence and should not be rerun without a genuinely new hypothesis. The merged Numba persistent JIT #216 and Cython stopping cache #178 are the baseline.
5. **Promote only demonstrated wins:** precommit GO/NO-GO criteria, independent Linux/Windows complete-fit replication plus four-platform production package tests, then review a minimal master-derived PR. Keep research experiments separate from production merges.
6. **Keep scientific convergence research independent:** finish/curate #18 and #182–#188 if scientifically useful; only investigate trapped/immobile particles with canonical undirected graph evidence, not directed kNN artifacts.

## Administrative execution notes

- This file and PR link serve as the audit decision log; each closed experimental PR should receive a short final closure comment linking this index and its own original run artifacts.
- The archive must distinguish **scientific evidence complete** from **performance acceptance**, and note if the outcome was only observational.
- Do not alter `master` or merge the documentation PR without maintainer review.
