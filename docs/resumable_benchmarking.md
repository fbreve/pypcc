# Whole-fit resumable benchmark controller

Original detailed artifacts and legacy analysis inputs are preserved in the
[early-stopping archive](performance_early_stopping_archiving.md). This dated
report retains its original scientific results; local paths describe the original run.

`python -m benchmarks.resumable` is an optional repository tool. Its default is
a read-only frozen plan; mutations require `--execute`. The existing
`benchmarks.run` and all production PCC modules are unchanged. The historical
action defaults to the approved [second calibration](performance_second_calibration_plan.md):
16 datasets, seed 10, Numba → Cython → NumPy, one fit each, original parameters
and 500,000 cap. It cannot silently continue into a full campaign. Explicit
`--full-campaign` freezes a separate 3,600-task matrix and requires `--stage-seed`
for execution; see the [prepared full-campaign plan](performance_full_campaign_plan.md).
The [seed-10 historical stage is complete](performance_full_seed10_results.md);
the [seed-11 stage and combined audit are complete](performance_full_seed11_results.md).
The [complete campaign audit](performance_full_campaign_results.md) covers all
3,600 measured fits and 1,200 warm-ups, with no pending or ambiguous attempts.
Legacy identities/results retain their original protocol.

## Commands

Inspect the plan without loading datasets or compiling kernels:

```console
python -m benchmarks.resumable
```

After the required recovery tests and synthetic validation pass, start the
approved calibration in a new output directory:

```console
python -m benchmarks.resumable --action start --root NEW_CAMPAIGN --historical-cache .cache/stopping_patience/sources --execute
```

Read-only validation/status, followed by safe resume when there are no ambiguous
attempts or source/environment mismatches:

```console
python -m benchmarks.resumable --action status --root CAMPAIGN
python -m benchmarks.resumable --action resume --root CAMPAIGN --execute
```

`--max-tasks N` stops cleanly after N pending measured tasks, without changing
their fit cap. A completed campaign's resume only publishes another derived
summary generation; it runs no fits or warm-ups.

An interrupted task **never retries automatically**. After inspecting its logs
and retained evidence, explicit authorization can be expressed as:

```console
python -m benchmarks.resumable --action resume --root CAMPAIGN --retry-task t0000 --reason "Approved retry after inspecting interrupted attempt" --execute
```

The retry creates a new attempt ID and retains the old started/failed events,
temporary files, warm-ups and session records. A completed/capped task cannot be
retried. If multiple tasks are ambiguous, each requires authorization; execution
stays blocked while any remain unresolved.

An unsealed partial fixture also blocks preparation until an explicit
`--retry-fixture CASE_ID --reason ...` on resume. Its entire file inventory is
preserved and checksummed in the authorization event before preparing a new
fixture. A fully sealed, valid temporary fixture is recovered directly without
rebuilding its graph. A corrupted published fixture is never silently rebuilt.

Stale writer locks require the exact saved token and a reason:

```console
python -m benchmarks.resumable --action unlock --root CAMPAIGN --lock-token TOKEN --reason "Reviewed exited owner" --execute
```

Unlock refuses a live owner, unverifiable owner or different host, preserves the
stale lock record and appends a recovery event. No timeout steals a lock. If a PID
has been reused or a lock is malformed, recovery refuses; inspect it rather than
bypassing the check. Normal release verifies that the lock still belongs to the
current writer before unlinking it. This lock file is the only routinely removed
campaign file; historical owner records remain immutable.

Derived summaries can be regenerated without any fits:

```console
python -m benchmarks.resumable --action summarize --root CAMPAIGN --execute
```

Corrupt derived generations require the same action with an explicit `--reason`.
The recovery event inventories and preserves damaged files; a new generation is
published. This option never relaxes validation of plans, events, fixtures,
warm-ups, environment identity or measured results.

## Storage and validation

Every canonical JSON record has a SHA-256 envelope over canonical JSON, with
duplicate keys and nonfinite values rejected. `plan.json` is independently pinned
by `plan_identity.json`. Record publication writes and fsyncs a uniquely named
staging file, then hard-links it to a previously absent canonical destination.
An existing destination fails; replacement semantics are never used. Staging
files remain preserved, including files from interrupted writes/publication.

Fixtures are saved by the original harness, fsynced, sealed with file checksums,
and revalidated before exclusive-lock-protected directory publication. On resume,
arrays and metadata are rehashed, dimensions/labels checked, and historical raw,
processed, ordered-graph and label fingerprints recomputed against frozen
references. The original fixture/source checks remain in place at preparation.

Event records form an ordered SHA-256 chain. Each measured fit has a durable
started event and two saved warm-up references before entering `measure`. A
validated canonical result is published before its completion event. If a crash
occurs between them, resume appends reconciliation and skips the existing fit.
If no canonical result exists, the attempt remains ambiguous even if a valid
temporary blob exists: inspection and explicit retry are required.

Each session records its full environment/source/binary provenance, seed-helper
latency, warm-ups and elapsed/status record when it exits normally. A fresh
resumed process warms only pending fixture/backend tasks. Result validation
checks exact backend/RNG identity, all six state hashes, original stopping
schedule/patience, cap/censor semantics, finite timings, timing-boundary sums,
accuracy, task order and fixture identity. `measure` retains all original runtime
dominance/labeled-node/particle/distance/signature invariant checks.

Only whole fixtures and completed fits are checkpointed. No particle/RNG state
is resumed inside a loop. `max_iter` is a completed censored observation; an
`early_stop` exactly at the cap is separately retained. There is one timing
observation per task, so summaries retain raw values without fabricated IQRs.

Summary generations are new, immutable directories with their own checksum
inventories and publication events. Partial generations from crashes remain
preserved. Completion requires the exact task matrix and no ambiguous attempts.

Source-file hashes, binary hash, dependency versions, executable, platform,
hardware and thread settings must match the frozen campaign on resume. Commit
IDs and dirty status are additionally recorded in each session; a documentation
commit alone does not change execution identity. Machine frequency/load cannot
be completely frozen and are not implied equal between sessions.
After a controller-source revision, use its measured source snapshot for any
live status/resume operation on an older campaign. The offline historical audit
does not execute fits; it validates archived source hashes, recorded identities,
the identified binary and authoritative data rather than claiming that current
controller code is the original execution environment.

Files are fsynced and POSIX directory metadata is fsynced. Windows lacks portable
directory fsync; file flush/publication failures surface, and restart validation
is mandatory. The store detects corruption but is not a guarantee against every
power or storage failure. Canonical-result corruption stops execution; it cannot
be repaired by rerunning or overwriting a measurement automatically.

## Verification

Recovery tests use fake fits and 30-node synthetic fixtures; no historical fits,
downloads or long benchmarks run in CI. Tests include actual subprocess exits,
missing completion events, ambiguous retries, corrupted plans/results/fixtures/
warm-ups/events, missing records, stale/live locks, changed execution identity,
duplicate results, summary interruptions and explicit derived-summary repair.

```console
python -m unittest discover -s tests -p test_resumable_benchmarks.py -v
```

The separate small real-backend validation used a 30-node, two-class fixture,
historical stopping parameters, synthetic cap 5,000, and three measured fits.
It stopped after one fit, resumed two, and resumed the completed campaign again.
All original invariant checks passed; completed result bytes were preserved;
only two measurement sessions and six warm-ups existed. Its local artifacts are
`.cache/performance/resumable-synthetic-01/`, with a validation/checksum record.
Final event-validation hardening passed 22 recovery tests and a second small
real-backend validation at `.cache/performance/resumable-synthetic-02/`; the
historical measurements were not repeated.
All four implementation CI jobs passed before the approved historical execution.
The [completed second calibration](performance_second_calibration_results.md)
contains the 48 measured outcomes and strict offline audit. No historical
calibration runs are part of CI.
