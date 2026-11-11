# PyPCC current-code authorship and license decision memo — 2026-11-11

**Status: source-authorship audit complete on pinned pre-license source; the maintainer subsequently approved and merged the MIT license in production via [PR #248](https://github.com/fbreve/pypcc/pull/248), commit [`1786afa`](https://github.com/fbreve/pypcc/commit/1786afa5063c9e949741b443103d0bfd14706952).** This report concerns the **current Python/Cython/Numba implementation**, not copyright in the 2012 scientific Particle Competition and Cooperation (PCC) model or a legal opinion about all possible historical contributions.

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

## License decision completed; publication-readiness boundary — updated 2026-11-11

The current maintainer **explicitly approved MIT** for the rewritten PyPCC implementation. The separate production [PR #248](https://github.com/fbreve/pypcc/pull/248) subsequently merged as [`1786afa5063c9e949741b443103d0bfd14706952`](https://github.com/fbreve/pypcc/commit/1786afa5063c9e949741b443103d0bfd14706952). This implemented, and four-platform-tested, the **complete official MIT [`LICENSE`](../LICENSE)** with `Copyright (c) 2026 Fabricio Aparecido Breve`; [`setup.py`](../setup.py) declares `MIT`, [`MANIFEST.in`](../MANIFEST.in) explicitly includes the license, and packaging tests validate the literal license bytes and metadata in both the wheel and sdist. The exact-head Linux/Windows, Python 3.10/3.12 package regression passed **4/4** in [run 43088117328](https://github.com/fbreve/pypcc/actions/runs/43088117328). **No PCC mathematical rule, backend, RNG, graph, stopping condition or numerical kernel was changed.**

The original publisher's `pypcc==0.0.6` artifact indicated MIT in `setup.py` and `PKG-INFO`, and its original `model.py` was byte-identical to upstream GitHub `pcc.py`. This corroborates permissive original-source *intent*, but did not contain the complete MIT permission/copyright notice. The current `LICENSE` notice addresses the maintained implementation; it does **not** purport to reassign Caio Carneloz's copyright, the original PCC method's article authorship, or third-party package rights to Fabricio. The historical Python fork credit and citation remain part of [README](../README.md).

This memo's **1,596/1,597** Git-blame result is frozen at pre-license runtime source `8d5f5aa`. Adding MIT/metadata/docs did not change any of the seven runtime modules, so the result remains relevant to the integrated code; the pinned SHA should **not** be retroactively substituted with the later master commit. The previous suggestion to choose MIT was a **historical proposal now fulfilled**, not pending work.

### Remaining work for a potential JOSS / software-paper submission

- **Publication decision and actual research use:** establish whether the current *software implementation*, rather than just the original 2012 PCC algorithm, has sufficient documented research use and distinct scholarly contribution for the target venue. A large benchmark history alone does not guarantee acceptance.
- **Project/paper metadata:** assess a `CITATION.cff` distinguishing software from the original [2012 PCC method paper](https://doi.org/10.1109/TKDE.2011.119), and a software paper `paper.md` with defensible authorship, statement of need, features, actual research use and comparisons. Do not invent software DOI, archived release, author contribution or acceptance status.
- **Reproducibility and installation:** review examples, exact supported operating systems/Python versions, native compilation, dependency metadata and benchmark protocols; ensure package distribution identity `pcc_step` vs source repository name `pypcc` is explained, not silently renamed.
- **Rights outside the seven runtime modules:** separately review third-party datasets, benchmark graphics, generated `pcc_step.c` build artifacts and any copied ancillary documents before asserting that *every* non-code asset is covered by the newly added MIT notice. Dependencies retain their own licenses. Git authorship is evidence, not a legal clean-room certificate.
- **Disclosure, attribution, release:** follow the target journal's **currently applicable** editorial requirements, including appropriate AI-use disclosure and author approval, and only publish/tag a version with the maintainer's explicit authorization.

**Disposition:** the licensing task of [#170](https://github.com/fbreve/pypcc/pull/170) is now **completed**; #170's remaining open draft work is *software-paper feasibility*, not license selection. This documentation PR introduces no new license instrument. The README introduction is updated to accurately credit the 2012 PCC model, Fabricio's earlier MATLAB/MEX implementation and Caio's original Python version, while retaining the MIT section introduced by #248. No production or benchmark change is introduced here.
