"""Descriptive summaries only; never choose a multiplier using accuracy."""
from collections import defaultdict
import statistics

FIELDS = ('observation_iteration', 'iterations_saved', 'accuracy_unlabeled',
          'accuracy_difference', 'correct_nodes_difference', 'final_prediction_changes',
          'sampled_returned_nodes', 'matrix_frobenius')


def describe(values):
    return dict(mean=statistics.mean(values), median=statistics.median(values),
                sd=statistics.stdev(values) if len(values) > 1 else 0.,
                minimum=min(values), maximum=max(values))


def summaries(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[(row['dataset'], row['multiplier'])].append(row)
    dataset_rows = []
    for (name, multiplier), values in groups.items():
        first = values[0]
        stopped = sum(row['status'] == 'stopped' for row in values)
        result = dict(dataset=name, multiplier=multiplier, trajectories=len(values),
                      stop_rate=stopped / len(values), censored_rate=1 - stopped / len(values),
                      binary=int(first['classes'] == 2), nodes=first['nodes'],
                      labeled_fraction=first['labeled_fraction'])
        for field in FIELDS:
            result.update({field + '_' + key: value for key, value in describe([row[field] for row in values]).items()})
        dataset_rows.append(result)
    overall_rows = []
    for multiplier in sorted(set(row['multiplier'] for row in rows)):
        selected = [row for row in dataset_rows if row['multiplier'] == multiplier]
        result = dict(multiplier=multiplier, datasets=len(selected),
                      stop_rate=statistics.mean(row['stop_rate'] for row in selected),
                      censored_rate=statistics.mean(row['censored_rate'] for row in selected))
        # Describe dataset means: equal dataset weights, regardless of available seeds.
        # These medians/SD/ranges describe dataset means, not pooled trajectory values.
        for field in FIELDS:
            result.update({field + '_dataset_mean_' + key: value
                for key, value in describe([row[field + '_mean'] for row in selected]).items()})
        overall_rows.append(result)
    return dataset_rows, overall_rows
