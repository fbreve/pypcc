# Agent instructions for pypcc

## Project purpose
This repository implements Particle Competition and Cooperation (PCC) for semi-supervised classification. Preserve the algorithm's intended dynamics and the existing public API. The repository has NumPy, Numba and Cython implementations. The original PCC paper is the semantic reference; do not assume different floating-point results are bugs.

## Scope and workflow
- Work on a dedicated branch and a focused pull request. Never commit directly to `master`, merge a PR, or mark a draft PR ready without explicit maintainer approval.
- Keep fixes, performance changes, packaging changes, and API redesigns in separate PRs. Avoid unrelated cleanup.
- Before changing algorithm behavior, describe the concrete defect, its reproduction, the expected behavior, and the supporting evidence.
- Prefer small, reviewable commits and a concise PR description that explains the change, tests, and known limitations.
- Do not modify algorithm semantics to force bitwise equality between backends. Tiny floating-point differences may be legitimate; investigate only reproducible behavioral defects.
- Do not silently change defaults, randomness, stopping criteria, graph construction, or labels.

## Verification
- Add a deterministic regression test that fails before a bug fix and passes afterward whenever feasible.
- Test invariants: finite and valid dominance values, preservation of labeled nodes, valid particle positions and distances, and correct sequential state updates.
- Run the relevant tests locally when possible and confirm CI before declaring work complete. Report tests that could not be run.
- Compare NumPy, Numba, and Cython on representative small graphs; distinguish numerical differences from semantic discrepancies.
- Keep temporary instrumentation out of final production code and permanent CI. Long-running diagnostics should be optional.

## Current PR #1: NumPy sequential correctness
- Existing draft PR: https://github.com/fbreve/pypcc/pull/1
- Existing experimental branch: `fix/numpy-propagation-correctness`.
- First inspect the diff against `master` and establish exactly which original NumPy behavior is wrong, ideally with a minimal failing test.
- Reduce the final change to the NumPy correction, required API wiring, and essential regression tests. Keep benchmarks and exploratory diagnostics separate where possible.
- Preserve the experimental branch/history as a reference; do not discard work or force-push without explicit approval.
- Do not pursue perfect prediction or bitwise equality across NumPy and Numba as an acceptance criterion.
- Do not merge or alter `master`; present a concise summary and request maintainer review.
