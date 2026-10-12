# Adaptive stopping: feasibility and review gate

Branch: `experiment/adaptive-stopping-study`, based on merged PR #8 master
`479952fc988d48c754202cd4d12eac898157d513`. This is schema inspection and a
protocol proposal, **not a comparative study or parameter selection**. No
propagation, kernel/API change or new default is included.

## Historical signals actually published

The [inspection](benchmarks/adaptive_stopping/inspection.json) authenticates the
four ZIPs and all 800 member hashes using the published reader. It checks schemas
and schedules across all 200 trajectories. Reproduce from repository root with
`conda run -n pypcc-dev python docs/inspect_adaptive_stopping_archives.py`.

| Source | Signals | Resolution and limits |
|---|---|---|
| `checks.csv.gz` | `iteration, metric, record_gain, record, counter, resets` | 50,000 rows/case, after iterations 1, 11, …, 499991: ten million exact original checks. No metric at every iteration. |
| `samples.csv` | `iteration, metric, correct_nodes, accuracy_unlabeled, changed_nodes, is_reference` | 500 regular observations/case at 1000, 2000, …, 500000, plus first events of seven legacy multipliers. Total 101,218 rows, including 1,218 extra event rows; consecutive gaps 1–1000 iterations. |
| `complete.json` | Identity/runtime; full elapsed/checkpoint-I/O time; seven first-stop/censor rows; final RNG hash; recorded verification | Exact legacy event/cap accuracy and correct counts; endpoint dominance norms, endpoint prediction differences, sampled post-event changes/returns. These summaries are anchored to legacy events, not arbitrary candidate stops. |
| `verification.json` | Iteration and recorded state/RNG/original-stop equality flags | Historical audit evidence; no particle arrays or RNG sequence. |
| Root protocol/inventory/manifests | Frozen parameters; 20 datasets; graph/data/all label fingerprints; sizes/classes/labeled counts; SHA-256 | Provenance and denominators, not node-level time series. |

Let D(t) be node-by-class dominance. The metric is
M(t) = (1/n) sum_i max_c D_ic(t), **including labeled nodes**. Its serial Numba
reduction is the original reference; truth is never an input to stopping.
At checks t_j=1+10j, b_-1=0, b_j=max(b_(j-1),M(t_j)),
g_j=max(0,M(t_j)-b_(j-1)). A strict improvement resets c_j to zero; otherwise
c_j=c_(j-1)+1. `record_gain` measures improvement over the previous running
best, not change since the previous metric or improvement in accuracy. The
running best can be reconstructed exactly from the check history.

`changed_nodes` is the Hamming count against the **immediately preceding saved
classification observation**, including intervening legacy-event observations;
the first row compares with the initialized prediction. Labeled predictions
are fixed, so changes arise only on initially unlabeled nodes. Normalize by
U=n−labeled_nodes when proposing a rate. Actual vectors, changed-node identities,
class transitions, per-node confidence/margins, dominance matrices and movements
are absent. No saved prediction vector exists even at the regular 1000-step grid.
Thus irregular counts cannot be collapsed into a net endpoint Hamming count.
An out-and-back change inside an unobserved interval remains invisible.

Accuracy exists at saved observations and the legacy first events only. It is
not recoverable at arbitrary new stops from these aggregate signals. Likewise,
full-run wall time cannot identify prefix cost, actual speedup, observer overhead
or a new candidate's post-stop node identities. Checkpoint pickles are excluded
from this public feasibility study. Earlier PRs have other diagnostics; do not
silently mix them with this fixed 200-case archive.

## Four precisely defined candidate families

The proposed sampled definitions below avoid pretending that predictions were
observed at every iteration. They are experimental alternatives, not statements
about the existing API. W is an iteration window, rho a normalized sampled-change
budget, eta a cumulative dominance-record budget, and t_min a minimum stop age.

For saved classification times s_k, h_k is `changed_nodes`. At s_k, take
a(k)=max{a: s_a <= s_k−W}, requiring an actual saved starting observation; do not
invent a t=0 vector or partial-interval count. Define
V_k(W)=sum_(l=a(k)+1)^k h_l/U. The actual interval may exceed W by up to 999
iterations, and must be reported. V counts sampled transitions with multiplicity;
it is neither the fraction of distinct changed nodes nor full-interval stability.

| Candidate | Mathematical first-stop condition | Required signals; exactness | Parameters | Added online cost / memory | Failure modes; retrospective feasibility |
|---|---|---|---|---|---|
| **A — original PCC** | First t_j with c_j>T; no added warm-up. T=max(1, round[n/(P k_nn) × round(0.1 es_chk)]), isolated-graph safeguard T=1. | Original metric/check phase/counter; exact on original ten-step check grid. | Existing T formula and es_chk, frozen unchanged. | Original O(nC) metric reduction per check, O(1) counter memory; offline replay O(J), O(1). | Any strict record, however tiny, restarts patience; no accuracy guarantee; structural cap censoring. Fully retrospective event and accuracy because legacy multiplier 1 is archived. |
| **B — sampled classification stability** | First saved s_k >= t_min with valid starting observation and V_k(W)<=rho. | Saved consecutive change counts and U. Exact for this saved-observation rule, sampled for actual dynamics. | W, rho, t_min; frozen observation schedule. | Compute argmax O(nC) and compare O(n) per observation; store previous predictions O(n), rolling counts/timestamps O(W/delta_s); scalar sums O(1) amortized. Offline no argmax is needed. | Temporary plateaus, hidden flips/returns, confidently wrong stable predictions; repeated flips counted repeatedly; irregular endpoint convention matters. Event and accuracy fully retrospective on the prescribed saved schedule. Per-iteration or distinct-node stability is not identifiable. |
| **C — joint stability and patience** | First saved s_k >= t_min with V_k(W)<=rho **AND** c_(j(s_k))>T, j(s)=max{j:t_j<=s}. | B signals plus latest original check counter. Counter held between original checks; sample-time evaluation explicitly differs from evaluating a conjunction at every check. | B parameters; original T unchanged. | B plus original metric/counter cost; O(n+W/delta_s) memory. Offline timestamp merge O(J+K). | Tiny records can keep blocking the conjunction; conditions may never overlap. Not max(first A,first B), and do not latch a past A event. Latest counter is up to nine iterations old at saved times. Exact retrospective replay of this defined sampled hybrid and its observed accuracy is feasible. A check-grid variant with held classifications would require a separately declared rule and often unsaved stop accuracy. |
| **D — cumulative record magnitude** | First original check t_j>=t_min with sum_{l: t_j−W<t_l<=t_j} g_l <= eta, after sufficient history for a complete W-window. | Exact check gains/running best; ten-step resolution. | W (multiple of 10), eta, t_min. | Original O(nC) metric/check plus O(1) amortized rolling sum, O(W/10) gain memory; no classification argmax required. Offline O(J). | No records during a dip/oscillation below an earlier maximum looks stable despite changing dominance/classifications; aggregate mean hides uncertain nodes and class/label-fraction effects. Exact retrospective event/censor/iteration savings feasible; accuracy at a non-saved event is missing, not interpolated. |

For D, a per-record cutoff (ignore each g_j<=eta and reset modified patience only
on larger records) is a different candidate: it may ignore many individually
small but cumulatively substantial gains. The proposed D uses cumulative gains
to expose that distinction. This is also different from a metric-range or
absolute-variation criterion: historical-best increments ignore downward motion.
Both quantities could be examined later from check metrics, but are outside this
four-family proposal.

C is deliberately conservative: with unchanged T and an AND conjunction, its
first event cannot precede A's first event on the same continuous trajectory.
It therefore cannot improve iteration savings over A; it tests whether adding
sampled classification evidence makes stopping later or more selective. An OR
combination would be a different policy and is not silently substituted.

Memory/cost figures above are algorithmic estimates, not measured overhead.
Online timestamps and counters can be streamed; storing all historical particle
states is unnecessary. For classification rules, argmax dominates observation
cost; labeled-node exclusion does not remove the need to observe unknown-node
dominance. Repeated checks should not reorder the original metric reduction.

## Proposed frozen protocol — awaiting approval

The companion [protocol JSON](benchmarks/adaptive_stopping/proposed_protocol.json)
is **proposed, not approved or experimentally frozen yet**. Parameter choices
below are a small predeclared sensitivity grid based on observation spacing and
dimensionless signal scale; they were not optimized on archive accuracies or
candidate outcomes. No comparative candidate outcomes have been computed.

1. Freeze this proposal's commit, JSON SHA-256, archive manifest hash and analysis
   definitions before implementing the comparison. Development cohort: all 20
   PR #8 datasets, all seeds 10–19, all full 500000-step references; these outcomes
   have already been examined and are **exploratory**, not independent validation.
2. A keeps original patience. Primary B/C: W=3000, rho=0, t_min=5000. Primary D:
   W=3000, eta=5e-4, t_min=5000. Sensitivity grids: W in {1000,3000,5000}; rho in
   {0,1e-3}; eta in {1e-5,1e-4,5e-4}; t_min fixed at 5000. The 5e-4 value has
   already appeared in earlier experimental work, and is not a novel held-out
   threshold. No selection of the best grid cell or per-dataset tuning.
3. Replay once per case in bounded memory. A/D evaluate only original checks;
   B/C use the published schedule, including legacy first-event augmentation.
   That schedule is causally determined by the pre-existing seven patience
   observers, not by labels/truth or new policies. Validate A against archived
   multiplier-1 events. Test boundaries, strict inequalities, first events,
   irregular windows, cumulative tiny gains, counter resets after eligibility,
   and missing accuracy. Do not execute kernels during this phase.
4. Record first event, effective window endpoints/span, counter/check age, gains
   or sampled changes used, stop/censor status and iterations saved. At cap, mark
   right censoring; do not count a cap accuracy as observed stop accuracy. Use
   exact saved accuracy only when timestamps match; otherwise mark missing.
   Nearest saved accuracies and their distances may be shown as contextual
   brackets, **never bounds, interpolated stop estimates or substitutes**.
   A separate observation-aligned D sensitivity may be defined explicitly as
   a different policy; do not round the exact D stop silently.
5. Report individual cases and equal-dataset means, sample SD, medians, ranges,
   stop/censor rates, paired differences vs full-budget accuracy and vs A,
   and accuracy-availability rates. Never average only observed stops without
   explaining selection/censoring. Compare accuracies on a common observed
   subset and disclose exclusions; do not claim four-way accuracy comparison
   where D timestamps are missing. Processed-node counts depend on dataset size.
   Retrospective iteration savings are not measured speedups. Post-stop legacy
   reversion summaries cannot be reassigned to new B/C/D stops.
6. Independent seed replication is a later, separately approved experiment:
   reserve seeds 100–109 on the same pinned datasets/graphs and labeling protocol
   and keep all primary/grid parameters fixed before viewing those outcomes.
   These would be fresh RNG/label replicates, **not independent dataset validation**;
   audit earlier use of these seeds before the freeze. Generalization to new
   datasets needs a separate inventory fixed before outcomes. No new trajectory
   is authorized or started by this proposal.
7. If later approved, future observations must use a common causal schedule
   matching the legacy augmentation for comparability, plus exact first-event
   observations for new policies. Record sufficient prediction information for
   declared node-level diagnostics, but use streaming comparisons/selective
   saves rather than full snapshots each iteration. Verify state/RNG equality
   against unchanged monolithic propagation. Measure unobserved vs observed
   full trajectories and observed prefix stopping runs on matched seeds with
   warm-up/order alternation, reporting real wall time, observation overhead
   and preparation separately. Do not estimate those timings from archive totals.

## Feasibility conclusion

**A is fully available. B/C are available as explicitly sampled rules with exact
saved-time accuracy. D's event timing is available, but generally not its exact
stop accuracy.** Continuous classification stability, per-node confidence and
new stop-to-final reversion identities require new observation data. Public
archives suffice for exploratory replay; checkpoints cannot repair unobserved
historical signals. Review the window semantics, sampled C schedule and D's
missing-accuracy treatment before approving implementation. No preference among
candidate families or permanent stopping change is proposed.
