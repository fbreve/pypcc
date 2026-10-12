"""Independent streaming audits of the frozen retrospective patience results."""
import math


def audit_checks(rows, threshold, multipliers, cap):
    best = 0.
    counter = resets = longest = checks = late_records = tiny_late = 0
    gain_sum = 0.
    gain_min = float('inf')
    gain_max = 0.
    events = {}
    last_reset = None
    for row in rows:
        iteration, metric = int(row['iteration']), float(row['metric'])
        if iteration != 1 + 10 * checks or not math.isfinite(metric):
            raise ValueError('Invalid check phase or metric')
        checks += 1
        gain = max(0., metric - best)
        record = metric > best
        if record:
            best, counter = metric, 0
            resets += 1
            last_reset = iteration
            if iteration > 30000:
                late_records += 1
                tiny_late += gain <= 1e-12
                gain_sum += gain
                gain_min = min(gain_min, gain)
                gain_max = max(gain_max, gain)
        else:
            counter += 1
            longest = max(longest, counter)
        if (int(row['record']) != record or int(row['counter']) != counter or
                int(row['resets']) != resets or float(row['record_gain']) != gain):
            raise ValueError('Stored record/counter disagrees with independent replay')
        for multiplier in multipliers:
            adjusted = threshold if multiplier == 1. else max(1, round(threshold*multiplier))
            if multiplier not in events and counter > adjusted:
                events[multiplier] = dict(iteration=iteration, resets=resets, checks=checks)
    if checks != (cap + 9)//10:
        raise ValueError('Missing or duplicated checks')
    return dict(events=events, checks=checks, resets=resets, longest_stall=longest,
                last_reset_iteration=last_reset, late_records=late_records,
                late_gains_le_1e_minus12=tiny_late,
                late_record_gain_mean=gain_sum/late_records if late_records else 0.,
                late_record_gain_min=gain_min if late_records else 0., late_record_gain_max=gain_max)


def validate_rows(rows, audit, unknown_nodes, cap):
    for row in rows:
        event = audit['events'].get(row['multiplier'])
        if (row['status'] == 'stopped') != bool(event):
            raise ValueError('Stop/censor status mismatch')
        if row['stop_iteration'] != (event['iteration'] if event else None):
            raise ValueError('First event mismatch')
        if row['resets'] != (event['resets'] if event else audit['resets']):
            raise ValueError('Reset count mismatch')
        for key in ('', 'final_'):
            if not math.isclose(row[key+'accuracy_unlabeled'], row[key+'correct_nodes']/unknown_nodes, abs_tol=1e-15):
                raise ValueError('Accuracy/count mismatch')
        if row['correct_nodes_difference'] != row['correct_nodes'] - row['final_correct_nodes']:
            raise ValueError('Correct-node difference mismatch')
        if not math.isclose(row['accuracy_difference'], row['correct_nodes_difference']/unknown_nodes, abs_tol=1e-15):
            raise ValueError('Accuracy difference mismatch')
        if row['iterations_saved'] != cap - (event['iteration'] if event else cap):
            raise ValueError('Iteration saving mismatch')
