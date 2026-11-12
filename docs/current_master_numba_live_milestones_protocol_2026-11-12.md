# Compiled Numba live milestone observation — synthetic feasibility pilot

Source base: current `master` at `ea69f690208620d5563c8efece6b2e04369f52c0`. This independent research branch does not depend on NumPy-only PR #253 or frozen historical PR #18.

## Method
The observer calls the **current production** `_pcc_step_numba_sequential` on every iteration from a separate `@njit` compiled outer loop. The mean-maximum plateau counter, cadence `it % 10 == 0`, strict `>` increase, and break threshold match the current compiled Numba propagator. There are no per-step Python callbacks, graph conversions, CSR structures, changes to particle ordering, or modifications to production source. Immutable-by-convention snapshots of positions, strengths, `uint8` distances, dominance, and ownership are captured after full iteration / stopping check. Zero is the initial state. Memory is preallocated from a strict schedule and 64 MiB cap before seeding the compiled RNG. Output records only completed checkpoints.

## Synthetic validation gates
- Independent ordinary Numba runs up to **every observed prefix**; same seed, same per-backend state (not cross-backend numerical identity).
- Exact final `return_info`, all mutable kernel arrays, and **next Numba RNG draw** after observed vs unobserved runs.
- Zero iterations, stuck/no-move early stop, duplicate / unordered / out-of-bounds milestones, and over-budget rejection without changing state or compiled RNG.
- Build/test on Linux and Windows, Python 3.10 and 3.12; explicitly assess compiled nopython signatures before declaring native coverage.
- Follow with a separate **paired whole-fit CPU benchmark**, including warm-up and native compilation exclusion, CPU measurements and snapshots/export costs before GO assessment.

## Limits and disposition
This is not a complete C2 study. No real datasets, archived exact-prefix claims, historical frozen SHA reuse, default stopping changes, or production merge are authorized. Even if all synthetic tests pass, leave PR draft/unmerged and separately measure performance overhead. The current Numba RNG state is internal; the next compiled random draw checks continuity but is not a byte-level export of the generator's internal state. The snapshot outputs are read-only NumPy views backed by preallocated storage; callers should treat them as immutable research data, not as a security or isolation boundary.
