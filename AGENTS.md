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

## Architecture
- `pcc.py`: public class, graph/particle initialization and backend selection.
- `pcc_graph.py`: graph construction with NumPy, SciPy and scikit-learn.
- `pcc_numpy.py`: sequential reference and optional experimental synchronous variant.
- `pcc_numba.py`: sequential Numba kernels; `pcc_step.pyx`: sequential Cython kernels.
- `pcc_validation.py` and `pcc_metrics.py`: shared input validation and optional metrics.
- `experiments/` and `docs/`: optional research and published reproduction, separate from runtime modules.

## Installation and testing
- Tested CI matrix: Windows/MSVC and Linux/GCC, Python 3.10 and 3.12.
- `python -m pip install -e ".[numba]"` builds Cython using isolated build dependencies.
- `python -m unittest discover -v` runs local regression tests; require all three backends before claiming cross-backend validation.
- `python -m pip wheel --no-deps . --wheel-dir .cache/wheels` verifies isolated wheel compilation.
- Run `tools/check_installation.py --require-all` from outside the source tree against the installed wheel to detect missing runtime modules and fallback.
- Inspect CI for the exact final commit before declaring completion. Do not infer support for untested Python versions or platforms.

## Conventions and preservation
- Keep the public imports and method signatures stable. Distribution name is `pcc_step`; do not change the version or publish releases without approval.
- Maintain compiler-specific optimization flags; never introduce fast-math or other floating-point semantic changes as packaging cleanup.
- Preserve sequential updates, tie acceptance, uint8 distance saturation, label handling and RNG consumption. NumPy synchronous is a distinct experimental algorithm.
- Prefer deterministic small-graph invariant tests. Final predictions need not be bitwise identical across backends.
- Preserve frozen protocols, parameter grids, scientific results and manifests. Historical validation is provenance, not a newly repeated state/RNG audit.
- ZIP trajectories are external Release assets; never reintroduce them into Git. Public reproduction uses verified local artifacts and optional explicit downloads, without checkpoints.
- Never execute long experiments, retune stopping criteria or publish external artifacts merely to validate maintenance work.
