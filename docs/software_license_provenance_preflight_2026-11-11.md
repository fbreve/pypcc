# PyPCC software licensing and upstream provenance — verification preflight (2026-11-11)

**Status: EVIDENCE FOUND, GRANT SCOPE UNCONFIRMED; NO LICENSE CHANGE AUTHORIZED.**

This is a preliminary **source/provenance decision memo**, not legal advice, a legal determination, an actual license grant, an automatic relicensing operation or a JOSS-ready submission. The original research/optimization production code remains unchanged.

## Confirmed direct-source facts

1. The live GitHub repository metadata for [`fbreve/pypcc`](https://github.com/fbreve/pypcc) identifies it as a real GitHub fork (`fork=true`) of [`caiocarneloz/pycc`](https://github.com/caiocarneloz/pycc). The fork's current root README expressly attributes the original implementation to Caio Carneloz and links the same upstream repository.
2. The upstream `master` root presently contains **only three files**: `README.md`, `demo.py` and `pcc.py`. The README and those two Python sources have no visible `LICENSE` file or explicit permission/copyright grant text; live GitHub repository metadata reports `license=null`. This is a statement about the **present GitHub tree**, not proof that no license was ever granted in another distribution or earlier revision.
3. The fork's GitHub repository metadata likewise reports `license=null`, and a current `master` root inventory found no `LICENSE`, `LICENSE.md`, `COPYING` or other recognizably named license document. The earlier preliminary [#170 software-paper audit](https://github.com/fbreve/pypcc/pull/170) accurately reported this **limited** root observation but should not be read as a global historical no-license finding.
4. **Important additional contemporary publisher evidence:** [Caio Carneloz's original `pypcc` project on PyPI](https://pypi.org/project/pypcc/) explicitly lists `Author: Caio Carneloz` and **`License: MIT License`** in its *publisher-supplied, PyPI-unverified project metadata*. Its latest 0.0.6 source release was published **2020-04-11**, predating creation of the fork (2020-08-27). The PyPI release page lists the exact source distribution `pypcc-0.0.6.tar.gz` with **SHA256 `cc807593f9c809158d3fc1473b0087090fe84417a84bf5e0294d9cb8a0802c28`**.
5. **Critical scope limit:** an attempt to obtain that upstream source-distribution tarball from the current runtime did not succeed due network/DNS restrictions. Its exact `LICENSE*`, `PKG-INFO`, `setup.py`, package source and copyright notices have **not** been examined here, and the PyPI metadata line alone has not been shown to apply to every original GitHub commit/source file currently incorporated into this fork.

## Why the distinction matters

GitHub's [licensing guidance](https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/customizing-your-repository/licensing-a-repository) distinguishes public visibility and fork availability from explicit reusable-distribution permissions; repository metadata `license=null` is a license-recognition observation, **not** proof of absence of an out-of-band grant. MIT is an OSI-approved open-source license, but a statement in registry metadata is less complete evidence than the publisher's actual accompanying license text, matching source revision and retained original copyright/notice terms.

[JOSS submission requirements](https://joss.readthedocs.io/en/latest/submitting.html), [JOSS review checklist](https://joss.readthedocs.io/en/latest/review_checklist.html), and [review criteria](https://joss.readthedocs.io/en/latest/review_criteria.html) require source access, an actual OSI-approved license file, appropriate authorship and research use, public open development, runnable tests/documentation and an actual software paper. Neither an absent root `LICENSE` nor the existence of an upstream PyPI MIT classifier alone resolves eligibility. The scientific PCC papers and published algorithm are not themselves a code software-license instrument.

## Safe next steps — do not silently publish a LICENSE yet

1. **Verify the publisher's distribution:** obtain `pypcc-0.0.6.tar.gz` directly from the linked original PyPI project (not another package named `pycc`), verify SHA256 **before inspection**, and inspect every actual `LICENSE*`/`COPYING*`/`NOTICE*`, `PKG-INFO`, `setup.py` and original source file. Preserve a checksum and an inventory, not the archived copyrighted source in a PR unnecessarily.
2. **Bind the grant to source and people:** compare the licensed PyPI code and time/revision with upstream GitHub `master` history and this actual `fork` lineage. Confirm the original publisher/author's identity and license permissions for redistribution, derivative modification and inherited source code. Identify any other contributors with rights to inherited portions.
3. **Retain appropriate attribution:** if the proper grant is established, prepare a small, explicit, **separately reviewed** `LICENSE` and any necessary upstream attribution/`NOTICE` for inherited work and new contributions, accurately reflecting the copyright holders and applicable terms. If the actual sdist leaves scope or wording ambiguous, request written confirmation/permission from Caio before adding or changing license declarations.
4. **Treat journal readiness separately:** only then address a `CITATION.cff`, JOSS `paper.md`, research-use evidence specifically using this implementation, reproducible install/test instructions and author list. The existing original algorithm paper citation is not, by itself, evidence of adoption of the **software fork**.

**Decision:** [PR #170](https://github.com/fbreve/pypcc/pull/170) remains OPEN/DRAFT/UNMERGED as a separate paper/license review. No guessed MIT text, unilateral source relicensing, contributor-rights assertion or JOSS submission is made by this documentation-only preflight.

**Evidence snapshot:** GitHub upstream and fork metadata/root/source inspected on 2026-11-11; original PyPI published project page 2020-04-11; standards-linked source pages consulted in November 2026. A later change in these external pages does not retroactively establish source coverage as of this audit.

## Verified original source distribution — independent follow-up, later 2026-11-11

**This section updates the earlier preliminary observation that the source archive could not yet be inspected. The earlier evidence limitation is now resolved; historical statements above remain as dated observations rather than current blockers.** An isolated read-only [GitHub Actions inspection 43070546416](https://github.com/fbreve/pypcc/actions/runs/43070546416) **PASSED** using only Python stdlib downloads, archive reading and source-hash comparisons. The runnable [prospectively recorded protocol and exact source-only inspector](https://github.com/fbreve/pypcc/tree/audit/upstream-pypi-mit-sdist-2026-11-11) remain on a **separate unmerged audit branch**, with its [original JSON/text artifact 13878905328](https://github.com/fbreve/pypcc/actions/runs/43070546416/artifacts/13878905328). **No original code was executed, installed, imported or extracted to disk.**

### Authenticated package ↔ upstream source connection

| Evidence | Independently verified outcome |
|---|---|
| [Original release](https://pypi.org/project/pypcc/0.0.6/) | `pypcc==0.0.6`, published 2020-04-11 by Caio Carneloz |
| Downloaded archive SHA-256 | **`cc807593f9c809158d3fc1473b0087090fe84417a84bf5e0294d9cb8a0802c28`** — exactly the PyPI-recorded file hash |
| Original `setup.py` | Line 6 links `https://github.com/caiocarneloz/pycc`; line 7: `license='MIT License'`; line 8 identifies Caio Carneloz as author |
| Both `PKG-INFO` copies | `Name: pypcc`, `Version: 0.0.6`, `Author: Caio Carneloz`, **`License: MIT License`** |
| Original source comparison | Published `pypcc/model.py` is **exact byte match** to `caiocarneloz/pycc:pcc.py` at immutable Git SHA `a0a12cae2607c01eb2f5ffab7a52540fa721234b` |
| Matching code SHA-256 | Both source copies **`ad67c77678b6d104eb25a2f98bef4b500dd542cbb1adf6b5d54458917fa2d0d5`**, 5,976 bytes |
| License text/copyright notice in archive | No `LICENSE`, `LICENCE`, `COPYING`, `COPYRIGHT`, `NOTICE`, `License-File` metadata, or standalone text of the MIT permission conditions |
| Code execution or package installation | **None** (archive members inspected in bounded memory, read-only) |

The archive contains `README.md`, `setup.py`, `setup.cfg`, `pypcc/__init__.py`, **`pypcc/model.py`**, two `PKG-INFO` copies and other `pypcc.egg-info` metadata files. The source identity and publisher's explicit setup metadata together constitute **direct, strong historical evidence of an intended MIT license for Caio's exact original PCC code**, not merely a third-party catalog guess or a similarly named package. At the same time, the sdist demonstrably **omits the actual complete MIT grant text and copyright notice**, so it cannot independently answer what exact attribution/notice text should be maintained in derivative redistributions.

### Updated decision (supersedes the unresolved-archive step above)

- **Proven:** exact PyPI file identity, origin/author declaration, MIT publisher metadata, immutable GitHub upstream source code identity.
- **Not yet proven:** a complete MIT license instrument/copyright notice for this source, whether later independently copyrighted fork contributions follow the same intended terms, and any third-party contribution obligations. Do not claim the registry's one-line metadata is a substitute for every required notice.
- **No automatic production licensing:** do not invent a copyright line, assign copyright for Caio's code to the fork maintainer, silently remove upstream attribution, or merge a new `LICENSE` solely on this metadata. Retain [#170](https://github.com/fbreve/pypcc/pull/170) as the focused unresolved licensing paper-readiness review.
- **Narrow next action:** request or locate **Caio's original MIT license text and preferred copyright notice** covering the verified exact `pypcc/model.py` / `pycc:pcc.py` code. With a confirmed grant and copyright holder(s), independently review a minimal `LICENSE` plus attribution/notice file if needed, and separately check authorship and contributions to the fork. This evidence does not change PCC computation, benchmark outcomes, or any earlier scientific PR disposition.

Raw GitHub Actions artifacts are finite-retention. This committed evidence summary, the SHA-pinned unmerged script/protocol branch, the upstream exact Git SHA and original PyPI package SHA allow a future independent inspection without trusting a copied executable.
