# Immutable two-host archive-only Cython rowmax recheck authorization

**No new CPU measurements, no baseline changes, no candidate changes.**

The full preregistered original source-vs-candidate native experiment at [GitHub Actions run 43023683759](https://github.com/fbreve/pypcc/actions/runs/43023683759), exact frozen experimental runner head `41d128e015453a68dde07043dff1ad92a9d21f84`, completed **both independent Windows and Linux source-native jobs SUCCESS**. Each host archived 80 uniquely specified original/candidate subprocess JSON records (20 frozen cohorts, alternating O–C–C–O, 3 CPU batches per subprocess, batches 16/64/128/256 complete fits according to prior frozen 40ed37a addendum), plus original native source manifest and full original scientific state/stop/cache verification.

- Linux original-source/candidate raw immutable artifact [13855499960](https://github.com/fbreve/pypcc/actions/runs/43023683759/artifacts/13855499960).
- Windows original-source/candidate raw immutable artifact [13856048639](https://github.com/fbreve/pypcc/actions/runs/43023683759/artifacts/13856048639).

The original run's automatic auditor used the wrong ZIP-root-relative `measurements/` location. Artifact uploader includes `cython_native_summary.log` at the workspace root, so ZIP actually contains `.cache/cython_native_rowmax/measurements/`. This independent read-only auditor bug was documented **before** changing the source-native benchmark runner or any data and corrected in commit `a71b17a93dba8d265001219e2e97262fc4b48c01`. The [archive-only workflow](../.github/workflows/experiment-cython-canonical-rowmax-archive-only.yml) downloads the EXACT existing run 43023683759 archives, strictly searches the uniquely named measurements folder, validates both source-native host corpora, ten SHA256 terminal states, Cython stopping/native cache parity, all CPU/memory batches and the immutable preregistered GO/NO-GO criteria. No modified fit is run; no data can be adjusted.

**Never merge scientific branch.** No source production edits. CPU_GO would require a distinct minimal production candidate; CPU_NO_GO means close without merge. Raw artifacts expire per Actions retention.
