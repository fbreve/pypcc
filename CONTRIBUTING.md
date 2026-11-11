# Contributing to PyPCC

Thank you for helping improve the Particle Competition and Cooperation (PCC)
Python implementation. The published PCC scientific method predates this
repository; see the [original paper](https://doi.org/10.1109/TKDE.2011.119).
The current project uses separate Cython, Numba, and NumPy backends.

## Where to ask for help or report a problem

Use [GitHub Issues](https://github.com/fbreve/pypcc/issues) for bug reports,
reproducible installation failures, usage questions, feature requests and
support requests. Search existing issues and [the README](README.md) first.
Include the purpose of the report in the title; please do not assume a response
time or support service-level agreement.

For a problem report, please provide as many of the following as apply:

- Operating system, Python version, installed PyPCC distribution version
  (`importlib.metadata.version("pcc_step")`), build/compiler information,
  NumPy, SciPy, scikit-learn and optional Numba versions.
- Backend (`cython`, `numba`, or `numpy`), `update_mode`, `k_nn`,
  whether the graph was constructed or explicitly supplied, and whether it
  is reciprocal/undirected.
- Data shape, label encoding, parameters, `seed` (or `None`),
  `max_iter`, `early_stop`, `es_chk` and a short reproducible example.
- Actual versus expected behavior, relevant traceback, and exact commands
  used. Include the source Git commit if working from a clone.
- A **small synthetic dataset** wherever possible. Remove credentials,
  personal data and proprietary or unpublished datasets from public reports.

For a bug that may expose security-sensitive details or private data, do not
post the exploit or secrets publicly in an issue.

## Local development

Use an isolated Python environment; the current documented CI matrix is
Python **3.10 and 3.12** on **Linux and Windows**, not a claim of validation
on all Python versions/platforms. From a checkout, install all three backends
to exercise native behavior:

```sh
python -m pip install --upgrade pip
python -m pip install -e ".[numba]"
python -m unittest discover -s tests -v
```

The editable build compiles the Cython extension; Windows requires a
compatible MSVC toolchain, Linux requires a C compiler and development headers.
See the [installation guidance](README.md#installation-and-compatibility),
[testing guide](docs/testing.md) and actual
[regression workflow](.github/workflows/pcc-regression.yml).
For an isolated installed-distribution check, run
`python tools/check_installation.py --require-all` **outside** the checkout
after building/installing the wheel; that prevents local-source imports from
masking a packaging issue.

A wheel/source-archive build can be checked with `python -m build` after
installing the `build` package. Note that packaging tests use `PCC_DIST_DIR`
to locate the freshly built archives. Every pull request triggers the existing
four-combination CI regression; contributors should not present local results
as Windows/Linux validation unless they actually ran those platforms.

## Propose changes in focused pull requests

1. Open an issue first for behavioral or API changes, and explain the
   reproducible problem, intended behavior, impact and alternatives.
2. Use a dedicated branch and a focused pull request. Explain the source
   commit, what changed, exact tests/hosts run, expected limitations and any
   scientific evidence. Documentation, packaging, correctness fixes and CPU
   performance experiments should be separate changes.
3. Keep existing numerical semantics stable by default. **Never** silently
   change label handling, reciprocal graph semantics, graph construction,
   movement decisions, RNG draw schedules, `early_stop` or public defaults.
   Fixed seeds do not imply bitwise equality *across different backends*.
4. For optimization claims, compare independent, pinned original and candidate
   versions with genuine **complete `fit_predict` CPU** measurements on
   relevant real and synthetic datasets, with source/semantics/stopping/RNG
   equivalence and both Windows and Linux whenever the claim is cross-platform.
   A microbenchmark speedup alone does not justify a production merge.
5. The intentionally **dense padded `neib_list`** layout trades memory for
   fast particle-loop CPU access. Do not propose replacing it with CSR/sparse
   merely to reduce memory; show full propagation/fit equivalence and
   independently measured whole-fit CPU benefit.
6. Add targeted tests for changed behavior, keep experimental instrumentation
   out of the default production path, and preserve frozen historical
   research protocols and result references without retroactively revising
   old measurements.

Existing research experiments in `docs/` and `experiments/` do not
automatically authorize an API or production algorithm change.

## Code attribution, licensing and assisted contributions

The current implementation is published under the [MIT License](LICENSE).
Submit code, documentation, figures and datasets only when you are entitled
to contribute them under the applicable terms, and preserve other authors'
copyright notices and necessary third-party attribution. This repository
credits Caio Carneloz's historical Python implementation and the prior PCC
scientific work; contributing to PyPCC is not a transfer of authorship of the
mathematical method.

If you used generative AI or externally generated code in a contribution,
describe the **scope and tools** in the PR and personally review, test and
verify the result, including licenses and originality. An author of a
future software paper must separately follow that venue's AI disclosure rules.

The maintainer reviews proposed changes; opening a PR does not guarantee
acceptance or imply permission to publish a package version or release.
