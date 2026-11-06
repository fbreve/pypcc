# Final narrowed corrective scope after single-step performance rejection

This decision follows two **separately preregistered and failed** direct-step default-validation speed experiments in draft PR #180, which remain archived without exclusion:
- V1 NumPy-based `np.isfinite` full active-dominance preflight performed before each public `pcc_step`: Digits/Linux 5k-call slowdowns +30.77%, +40.36%, +43.46%, exceeding the original +20% median limit.
- V2 native Cython `isfinite` full active-dominance preflight, with fresh seeds **10103/10211/10301**: Digits/Linux **+28.19% median**, Windows **+40.54% median**; separate OS×dataset full-state parity passed, but the same +20% ceiling failed. The slower result is intrinsic to rescanning ~n_nodes×n_classes on every single-step call, not a reason to loosen the gate.

## Scope decision

**Do not integrate either failed direct-step change.** The public single-step `pcc_step` body must be **byte-identical to current master** (at branch base). It retains historical input-finiteness responsibilities, API signature and runtime cost. Do not claim direct single-step NaN/Inf is atomically rejected.

Retain only the **already independently validated once-per-propagation active-finiteness preflight** in `pcc_step.pcc_propagate`:
- Validate active `part_strength[:n_particles]`, `dominance[:n_nodes,:c]`, and `dist_weights[:256]` through existing `pcc_validation.validate_finite` before cache initialization and `nogil` particle updates.
- Preserve unused oversized buffer padding, all scratch-buffer behavior, cached global potential default, RNG, step movement and stop semantics.
- The original frozen cross-binary protocol was **96 production native fits, 48 exact state+stop pairs** on Linux/Windows across Breast Cancer/Digits, 200k iterations, both monitoring/no-stop, seeds 9503/9601/9709; it passed all correctness gates and the prespecified <=8% median overhead bound (worst 5.58% slower). CI [42303213572](https://github.com/fbreve/pypcc/actions/runs/42303213572), full report artifact **13543869547**.
- Replace regression expectations: test mandatory NaN/Inf atomic rejection **only** for `pcc_propagate`, test the no-stop path, F/C layouts and legal padding, and require the original direct `pcc_step` to remain an unchanged, validated entry point for **finite** inputs. No tests should rely on ignoring nonfinite input and obtaining meaningful model results.
- Re-run the standard project regression matrix and updated independent compiled native parity. A direct-step benchmark with the restored *identical* legacy source is no longer an optimization gate because there is no direct-step code change; the two failed raw benchmark cohorts stay visible as rationale.

## Scientific/engineering conclusion

The safe and performant path is **boundary-specific validation once per native propagation**, not strict revalidation before every repeated public step. Atomically checking the entire global dominance state on each direct step has real computational cost and cannot meet our unchanged integration threshold in the tested 10-class workload. A separate future API design could provide optional strict direct-step validation, but no untested extra flag or behavior is introduced here.

No change to master is authorized until updated CI and code-review checks pass. The historical PR #36 is superseded only with respect to the **main propagation path**; its proposed single-step validation is explicitly *not* adopted.
