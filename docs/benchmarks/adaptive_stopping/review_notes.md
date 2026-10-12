## Validation and review notes

All **80 local regression tests passed** on Windows pypcc-dev with native Cython.
All 6800 result rows passed iteration-budget identities, censor/missing-accuracy
checks, periodic primary timing, counter-age bounds and C-not-before-A checks.
The 2400 paired schedule rows satisfy secondary-not-after-primary timing.

The original reference cohort has event/cap accuracy differences ranging from
-12.38 to +5.45 percentage points. Primary B ranges from -12.15 to +4.32 pp,
and C from -5.52 to +2.76 pp. Means alone conceal seed-level variation. C's
54 censored cases contribute full-budget reference accuracies, whereas A has
28 and B 31: smaller event/cap mean loss does not independently establish safer
observed stopping. Exact primary D stop accuracy is unavailable in all cases,
so neither a four-way accuracy comparison nor a D accuracy tradeoff is justified.

This is exploratory reuse of previously examined seeds/datasets. The frozen
grid is reported in full; no parameter or candidate is recommended as a winner.
Independent validation, measured stopping overhead and denser observations need
a separately approved future experiment. No production change is proposed.
