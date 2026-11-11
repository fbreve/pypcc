# PyPCC current-code authorship and license decision memo — 2026-11-11

**Status: independent source-authorship check complete; proposed MIT license NOT yet chosen or installed.** This report concerns the **current Python/Cython/Numba implementation**, not copyright in the 2012 scientific Particle Competition and Cooperation (PCC) model or a legal opinion about all possible historical contributions.

## Fixed-source evidence

- Current production version inspected: [`fbreve/pypcc` at `8d5f5aa9061a51ec5c4cd782b8ca10a0365049da`](https://github.com/fbreve/pypcc/commit/8d5f5aa9061a51ec5c4cd782b8ca10a0365049da). Seven current runtime modules were verified **bytewise unmodified** on the isolated audit branch before inspecting their true blame history.
- Independent source-only Actions job [43085403543](https://github.com/fbreve/pypcc/actions/runs/43085403543) **SUCCESS**: Python stdlib script [`experiments/audit_current_source_authors.py`](https://github.com/fbreve/pypcc/blob/audit/pypcc-current-line-authorship-2026-11-11/experiments/audit_current_source_authors.py) ran ordinary `git blame --line-porcelain` on a clone with full history at the **pinned production SHA**, and published a machine-readable per-source-line audit. [Unmodified original report artifact](https://github.com/fbreve/pypcc/actions/runs/43085403543) is subject to finite GitHub Actions retention.
- An earlier aggressive `git blame -w -M -C -C` [audit 43085258605](https://github.com/fbreve/pypcc/actions/runs/43085258605) assigned a handful of trivial shared syntactic lines (such as `else:`) to old commits as *copy-detection heuristics*. The corrected audit **explicitly separates standard Git line attribution from heuristic cross-file matches**, retaining both original runs and their outputs. Neither method constitutes a legal copyright attribution test.

### Present current-source line attribution

The category “code” in this count excludes blank lines and lines consisting entirely of a `#` comment. It includes docstrings and code; it does **not** claim that all lines have equal creative or legal significance.

| Core module | Present nonblank/non-full-line-comment lines | Git-attributed to Fabricio Breve or fbreve | Git-attributed to caiocarneloz |
|---|---:|---:|---:|
| `pcc.py` | 433 | 432 | 1 |
| `pcc_numpy.py` | 226 | 226 | 0 |
| `pcc_numba.py` | 218 | 218 | 0 |
| `pcc_step.pyx` | 432 | 432 | 0 |
| `pcc_graph.py` | 132 | 132 | 0 |
| `pcc_validation.py` | 105 | 105 | 0 |
| `pcc_metrics.py` | 51 | 51 | 0 |
| **Seven-module total** | **1,597** | **1,596** | **1** |

The **one** code line still attributed to Caio in ordinary Git blame is **`import numpy as np`** (`pcc.py:53`), a common import, not a proprietary algorithm implementation. Including blank/comment lines, 4 out of 1,917 current physical lines are attributed to Caio (that import plus three blank lines); 1,913 are attributed to the current maintainer. **All 1,164 code lines in the six newly implemented runtime modules are Git-attributed to the maintainer.**

Git blame assigns a current line to the last Git revision responsible for it, subject to the chosen similarity settings. This is excellent documentation of **source ancestry**, but is neither proof of legal ownership nor proof that no behavior or implementation idea was borrowed or learned from earlier implementations.

### Independent source chronology and similarities

- The PCC scientific method was published in 2012 with multiple research coauthors. The maintainer's [MATLAB/MEX/C implementation](https://github.com/fbreve/Particle-Competition-and-Cooperation) existed publicly by 2019-01-21, with a `pcc.m` comment naming Fabricio Breve as its author. Caio Carneloz later implemented the existing PCC model in Python while supervised by the current maintainer; [his original GitHub Python source](https://github.com/caiocarneloz/pycc) was created 2019-05-22.
- The earlier direct comparison, [documented separately](software_license_provenance_preflight_2026-11-11.md), found only 12 of 133 original nonblank/noncomment lines literally anywhere among the current seven core source modules. Only 5 of 101 original longer source lines survive literally, and no contiguous original 12-token sequence survives. This is a **textual clone indicator**, not a test of software license rights.
- The present API and graph construction were redesigned and the entire Cython/Numba/NumPy kernel architecture separately developed. The present code still implements the **same published mathematical rules**, so semantic similarity is deliberate. The original fork relationship and the historical attribution to Caio should remain visible in the README.
- The [original author's PyPI source archive `pypcc==0.0.6`](https://pypi.org/project/pypcc/0.0.6/) was independently SHA256 verified and its `model.py` shown **byte-for-byte identical** to the original GitHub `pcc.py`; the original `setup.py` and `PKG-INFO` explicitly declare **MIT License**, although **no complete `LICENSE` text or copyright notice is present in that historical archive**. This is credible first-party evidence of permissive MIT *intent* for the original Python code, not a substitute for deciding a license for the current new implementation.

## Third-party code, contributors, generated code: scope of this audit

The seven separately maintained current runtime modules directly **import external libraries** (principally NumPy and, for graph construction, SciPy and scikit-learn; optionally Numba; Cython is a build dependency), rather than copying the source of these packages into the seven Python/Cython modules. Their licenses are separate from any PyPCC project license and their own installation/distribution terms remain in force. The root file `pcc_step.c` is a **Cython-generated artifact** of `pcc_step.pyx`, not a separately human-authored Python-source module; its generated runtime boilerplate and any associated notices need separate packaging review if it is distributed. Benchmark data, diagrams, documentation, and other files are **outside this seven-runtime-module line audit**; do not silently represent their copyright status as fully cleared.

The current seven modules' historical commit-list authors are exclusively the maintainer for new source paths, with older Caio commits on the inherited `pcc.py`; the exact standard-blame analysis above is more specific about which lines are still attributed to which history. It **does not exclude AI assistance, external code contributed under a maintainer-authored commit, third-party test datasets, or untracked out-of-band rights**.

## Recommended license and action gate

**Recommended license for new PyPCC source: MIT**, subject to the maintainer choosing it explicitly. This matches the original Python publisher's stated permissive MIT intent, is OSI-approved, is simple for research software and is compatible with a full open-source dissemination goal. BSD-3-Clause is a reasonable alternative if the maintainer prefers an explicit non-endorsement clause; Apache-2.0 differs materially, including an express patent license, and should not be selected automatically just for journal eligibility.

**Do not add a root `LICENSE` automatically** or assert that the 2019 original author's copyright belongs to the maintainer. Once the responsible current author(s)/rightsholders approve **MIT**, prepare a new minimal independent `LICENSE` proposal using the exact canonical MIT permission text, an accurately reviewed present-code copyright notice (likely author name `Fabricio Aparecido Breve` and relevant modification years, subject to explicit approval), and retain a small provenance/credit line in the README for Caio's historical Python fork. If required by any rights notice, include a separate `NOTICE`/attribution, not a fabricated copyright line for the original author. The package distribution should carry the exact adopted license and its metadata, and its wheel/sdist contents should be checked independently before release.

As of this report the repository still **does not have a root `LICENSE`**. The current [JOSS official review checklist](https://joss.readthedocs.io/en/latest/review_checklist.html) requires an actual plain-text **OSI-approved** license file; JOSS also checks authorship, development history, public code, tests, examples, community participation and an explicit **AI usage disclosure** in the software paper. Mere metadata `License: MIT License` is not a substitute for an actual file. These are *publication-readiness requirements*, not reasons to rewrite PCC algorithms or to automatically involve Caio in every licensing decision.

**Proposed disposition:** keep [#170](https://github.com/fbreve/pypcc/pull/170) open as the separate journal/licensing task; have the maintainer explicitly choose MIT (or another OSI license), then introduce the true `LICENSE` and packaging/attribution metadata in a **separate reviewable PR**. Do not merge this audit branch or make a license decision merely because its factual line-blame report passes CI.
