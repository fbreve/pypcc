# Prespecified Digits supervision-density lockout comparison

**Recorded before any executions/results of the new study.** The preceding four-dataset, 20-run cohort found operational class-particle lockout on Digits in 4/5 runs (5/50 classes), but none in the 15 runs on Iris, Wine or Breast Cancer. This motivates a narrowly controlled follow-up within **Digits**.

## Fixed design

- **Dataset:** Digits only, StandardScaler, fixed **10-NN** graph.
- **New, non-overlapping seeds:** 601, 701, 809 (three matched replicates).
- **Compared conditions:** 2 vs **5 labeled nodes per class** (20 vs 50 particles). For each dataset-seed pair, independently sample *five* labeled observations per class in deterministic RNG choice order, then use their **first two** for the two-label condition. Thus the 2-label set is a **strict subset** of the 5-label set. Graph, class labels and seeds are otherwise identical. This changes *both* amount of supervision and number of particles, which must not be conflated with isolated effects of particle count.
- **Algorithm:** experimental sequential NumPy PCC with the existing default parameters, no intervention, **no early stopping**, cap 50,000 iterations, windows of 1,000 iterations.
- **Fixed operational lockout definition:** five consecutive windows with zero accepted moves by **all** particles of a class, valid visits across the windows, max class-particle strength <= 0.01 at observation and zero currently winning outgoing adjacency entries.
- **Integrity:** all expected valid visits, finite state, per-node simplex sum within 1e-8, particle strength within [0,1] up to roundoff <=1e-8, and final predictions equal matched same-seed uninstrumented runs for each condition.
- **Primary endpoint:** number of dataset-seed runs with ≥1 class lockout / three per condition. Secondary: affected classes / 30, first lockout iteration, subsequent recovery, zero-acceptance windows, accuracy on initially unlabeled nodes, and number of class-particle accepted moves. We will not tune the definition after seeing results.
- **Interpretation:** descriptive three-seed pilot; no statistical generality or causal identification of strength versus particle count. Different numbers of labeled nodes also yield different sets of unlabeled evaluation nodes, so direct raw accuracy comparisons use different denominators and are **not** a paired test of accuracy on identical test instances.

## Decision rule

This is a mechanistic *diagnostic*, not a candidate modification. If lockout decreases with more labels in the matched seeds, study whether greater particle coverage provides resilience with a larger, independently predeclared sample. If not, investigate graph topology and class competition, also on new seeds. **Do not modify production PCC, stopping criteria, or merge experimental code to `master` based on this pilot.**
