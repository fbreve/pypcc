# PyPCC — software-paper readiness audit (working document)

Status: **preliminary; not a submission or an assertion of JOSS eligibility**. This document is deliberately separate from experimental stopping criteria. No production behavior changes are proposed here.

## Confirmed from the `master` branch

- README documents installation, backend selection, Python compatibility and a minimal runnable example.
- Three implementations are described: NumPy, Numba and Cython; the distribution name is currently `pcc_step`.
- The README identifies the upstream project as Caio Carneloz's `pycc` and describes this repository as a fork.
- Packaging uses `setup.py` plus a build-system-only `pyproject.toml`.
- The following **exact root paths** were not found: `LICENSE`, `LICENSE.md`, `CITATION.cff`, `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`, `paper.md`. This is not proof that equivalent material is absent elsewhere or under another filename.

## Blocking questions before a JOSS submission

1. **License and provenance.** Verify the original `pycc` license, the fork's license and rights to redistribute inherited code. Do not choose or add a license until provenance is confirmed. If no valid open-source license covers the repository, resolve this first.
2. **Distinct contribution.** Document what this fork contributes beyond the original implementation, with references to changes, tests, benchmarks and technical rationale. Performance alone is not necessarily sufficient.
3. **Research use and significance.** Collect verifiable examples of research using this implementation (not merely the underlying PCC algorithm), and explain the intended research audience.
4. **Documentation and reproducibility.** Review API documentation, worked examples, dependency constraints, install-from-source instructions, testing instructions, supported platforms and expected results.
5. **Testing and releases.** Verify current CI across supported backends/platforms, regression coverage, release tagging and a stable archival DOI where appropriate.
6. **Authorship and citation.** Agree on contributors, author order, acknowledgments, upstream attribution and the canonical citation. Add `CITATION.cff` only after these are settled.
7. **Editorial requirements.** Check the current JOSS submission guidelines, scope and review checklist immediately before submission; requirements can change.

## Proposed article structure (not yet a manuscript)

- Summary: what PyPCC does and who needs it.
- Statement of need: research problem, existing implementations, concrete gap addressed.
- Design and functionality: graph construction, transductive labeling, three backends, reproducibility and input validation.
- Validation: test strategy, cross-backend comparisons and representative benchmarks with exact environment specifications.
- Related work and references: original PCC papers, original software, other implementations.
- Acknowledgments and contributor roles.

## Research separation

The experimental stopping-criterion study belongs in a **separate methodological paper**. It must not be presented as a validated feature of the software paper or merged into production before evidence and approval.

## Immediate next actions

1. Verify licensing/provenance with upstream history and any existing license files under alternate names.
2. Inventory actual tests and CI workflows, documentation, examples and existing software citations.
3. Build a reproducible baseline benchmark table for the stable implementation, without making unvalidated superiority claims.
4. Draft the paper only after the above checks, then request an explicit review before submission.
