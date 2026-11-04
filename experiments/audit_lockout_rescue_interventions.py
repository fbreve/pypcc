"""Compare targeted particle interventions to a matched no-stop control.

Wine seed 83 only. At iteration 48k, experimentally reset class-1
strength, relocate class-1 particles to their labeled home nodes, or both.
These are *causal counterfactuals*, not proposed production behavior.
All trajectories are run with the same seed/graph and no early stopping.
"""
import argparse
import csv

import numpy as np

from experiments.benchmark_inloop_global_stopping import run
from experiments.compare_dominance_stopping import prepare


VARIANTS = ("control", "restore_strength", "relocate_home", "both")
TRACE_FIELDS = (
    "dataset", "seed", "variant", "iteration", "class1_accepted_since_last",
    "class1_mass", "class1_prediction_count", "class1_mean_strength",
    "class1_max_strength", "class1_unique_positions", "accuracy",
)
SUMMARY_FIELDS = (
    "dataset", "seed", "variant", "intervention_iteration", "cap",
    "intervention_calls", "class1_moves_after_intervention",
    "class1_moves_first_1k", "last_class1_accepted_iteration",
    "class1_mass_final", "class1_prediction_count_final",
    "accuracy_final", "accuracy_control",
    "agreement_with_control", "class1_mass_control_final",
)


def run_variant(graph, degrees, labels, truth, seed, cap, interval,
                intervention_iteration, variant):
    unknown = labels == -1
    homes = np.flatnonzero(labels == 1)
    if homes.size != 2:
        raise AssertionError("Expected two labeled Wine class-1 nodes")
    snapshots = {}
    accepted = {}
    last_accepted = None
    interventions = 0

    def observe(it, dominance, positions, particle_classes, strengths):
        indices = np.flatnonzero(particle_classes == 1)
        snapshots[it] = dict(
            mass=float(dominance[unknown, 1].sum()),
            count=int(np.count_nonzero(
                np.argmax(dominance[unknown], axis=1) == 1)),
            mean_strength=float(strengths[indices].mean()),
            max_strength=float(strengths[indices].max()),
            positions=tuple(int(v) for v in positions[indices]),
            accuracy=float(np.mean(
                np.argmax(dominance[unknown], axis=1) == truth[unknown])),
        )

    def observe_event(it, cls, is_unlabeled, greedy, is_accepted, donor_loss):
        nonlocal last_accepted
        if cls != 1:
            return
        window = ((it - 1) // interval + 1) * interval
        accepted[window] = accepted.get(window, 0) + int(is_accepted)
        if is_accepted:
            last_accepted = it

    def intervene(it, positions, particle_classes, strengths):
        nonlocal interventions
        interventions += 1
        indices = np.flatnonzero(particle_classes == 1)
        if it != intervention_iteration or indices.size != homes.size:
            raise AssertionError("Unexpected intervention state")
        if variant in ("restore_strength", "both"):
            strengths[indices] = 1.0
        if variant in ("relocate_home", "both"):
            positions[indices] = homes

    prediction, elapsed, info = run(
        graph, degrees, labels, seed, cap, 0, interval,
        state_checkpoint_callback=observe, checkpoint_interval=interval,
        particle_event_callback=observe_event,
        event_start_iteration=intervention_iteration,
        intervention_callback=(None if variant == "control" else intervene),
        intervention_iteration=(None if variant == "control"
                                else intervention_iteration))
    if info["effective_iterations"] != cap:
        raise AssertionError("Unexpected early stop")
    if interventions != (0 if variant == "control" else 1):
        raise AssertionError("Intervention did not occur exactly once")
    if len(snapshots) != cap // interval:
        raise AssertionError("Missing observation checkpoints")
    if not np.isclose(snapshots[cap]["accuracy"],
                      np.mean(prediction[unknown] == truth[unknown])):
        raise AssertionError("Snapshot accuracy mismatch")
    if sum(accepted.values()) > 2 * (cap - intervention_iteration):
        raise AssertionError("Invalid number of accepted movements")
    return dict(prediction=prediction, elapsed=elapsed, snapshots=snapshots,
                accepted=accepted, last_accepted=last_accepted,
                intervention_calls=interventions)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace-output", required=True)
    parser.add_argument("--summary-output", required=True)
    parser.add_argument("--seed", type=int, default=83)
    parser.add_argument("--cap", type=int, default=60000)
    parser.add_argument("--interval", type=int, default=1000)
    parser.add_argument("--intervention-iteration", type=int, default=48000)
    args = parser.parse_args(argv)
    if (args.cap < 1 or args.interval < 1
            or not 0 < args.intervention_iteration < args.cap
            or args.cap % args.interval
            or args.intervention_iteration % args.interval):
        parser.error("Invalid checkpoint or intervention configuration")

    graph, degrees, labels, truth = prepare("wine", args.seed)
    unknown = labels == -1
    experiments = {
        variant: run_variant(graph, degrees, labels, truth, args.seed,
                             args.cap, args.interval,
                             args.intervention_iteration, variant)
        for variant in VARIANTS
    }
    reference, _, _ = run(graph, degrees, labels, args.seed,
                           args.cap, 0, args.interval)
    control = experiments["control"]
    if not np.array_equal(control["prediction"], reference):
        raise AssertionError("Instrumented control changed predicted labels")

    for variant in VARIANTS[1:]:
        outcome = experiments[variant]
        # Unmodified prefixes must remain the same, regardless of
        # intervention after the selected iteration has finished.
        for checkpoint in range(args.interval,
                                args.intervention_iteration, args.interval):
            if outcome["snapshots"][checkpoint] != control["snapshots"][checkpoint]:
                raise AssertionError(
                    f"Intervention changed earlier history: {variant}, {checkpoint}")

    control_final = control["snapshots"][args.cap]
    control_accuracy = float(np.mean(
        reference[unknown] == truth[unknown]))
    with (open(args.trace_output, "w", newline="", encoding="utf-8") as tfile,
          open(args.summary_output, "w", newline="", encoding="utf-8") as sfile):
        writer = csv.DictWriter(tfile, fieldnames=TRACE_FIELDS)
        summary = csv.DictWriter(sfile, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        summary.writeheader()
        for variant, result in experiments.items():
            for it, snap in sorted(result["snapshots"].items()):
                if it < args.intervention_iteration:
                    continue
                writer.writerow(dict(
                    dataset="wine", seed=args.seed, variant=variant,
                    iteration=it,
                    class1_accepted_since_last=int(
                        result["accepted"].get(it, 0)),
                    class1_mass=snap["mass"],
                    class1_prediction_count=snap["count"],
                    class1_mean_strength=snap["mean_strength"],
                    class1_max_strength=snap["max_strength"],
                    class1_unique_positions=len(set(snap["positions"])),
                    accuracy=snap["accuracy"],
                ))
            final = result["snapshots"][args.cap]
            summary.writerow(dict(
                dataset="wine", seed=args.seed, variant=variant,
                intervention_iteration=args.intervention_iteration,
                cap=args.cap,
                intervention_calls=result["intervention_calls"],
                class1_moves_after_intervention=sum(
                    result["accepted"].values()),
                class1_moves_first_1k=result["accepted"].get(
                    args.intervention_iteration + args.interval, 0),
                last_class1_accepted_iteration=result["last_accepted"],
                class1_mass_final=final["mass"],
                class1_prediction_count_final=final["count"],
                accuracy_final=final["accuracy"],
                accuracy_control=control_accuracy,
                agreement_with_control=float(np.mean(
                    result["prediction"][unknown] == reference[unknown])),
                class1_mass_control_final=control_final["mass"],
            ))


if __name__ == "__main__":
    main()
