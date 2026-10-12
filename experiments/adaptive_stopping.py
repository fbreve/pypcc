"""Pure retrospective observers. No propagation, truth, accuracy or RNG inputs."""
from collections import deque
import math
from bisect import bisect_right


def record_stagnation(checks, window, budget, minimum_iteration):
    """First original check with a complete window and cumulative record gains <= budget.

    Only a historical prefix is used; metrics below previous maxima are deliberately
    not interpreted as general dominance stability.
    """
    if window <= 0 or window % 10 or budget < 0 or minimum_iteration < 0:
        raise ValueError('Invalid record-stagnation parameters')
    gains = deque()
    total = 0.0
    first = None
    previous = None
    for row in checks:
        t, gain = int(row['iteration']), float(row['record_gain'])
        if previous is not None and t != previous + 10:
            raise ValueError('Invalid original-check schedule')
        if not math.isfinite(gain) or gain < 0:
            raise ValueError('Invalid record gain')
        previous = t
        if first is None:
            first = t
        gains.append((t, gain))
        total += gain
        while gains and gains[0][0] <= t-window:
            total -= gains.popleft()[1]
        # Recompute the bounded sum at the decision, avoiding cancellation drift.
        if t >= minimum_iteration and t-first >= window:
            value = math.fsum(g for _, g in gains)
            if value <= budget:
                return dict(iteration=t, window_start=t-window, window_end=t,
                            record_gain_sum=value)
    return None


def original_event(checks, threshold):
    """Original strict patience, evaluated at check rows; never latch eligibility."""
    for row in checks:
        if int(row['counter']) > threshold:
            return dict(iteration=int(row['iteration']), counter=int(row['counter']))
    return None


def classification_event(samples, checks, window, budget, minimum_iteration,
                         unknown_nodes, threshold, *, joint=False, regular=True):
    """Hybrid periodic decisions or secondary all-observation decisions.

    All transitions inside each past window are counted once, including legacy
    events. No prediction vector, accuracy, future counter or truth is accessed.
    """
    if window <= 0 or budget < 0 or unknown_nodes <= 0:
        raise ValueError('Invalid classification parameters')
    times = [int(r['iteration']) for r in samples]
    check_times = [int(r['iteration']) for r in checks]
    if times != sorted(set(times)) or check_times != sorted(set(check_times)):
        raise ValueError('Observations must be strictly time-ordered')
    prefix = [0]
    for row in samples:
        value = int(row['changed_nodes'])
        if not 0 <= value <= unknown_nodes:
            raise ValueError('Invalid transition count')
        prefix.append(prefix[-1]+value)
    for k,t in enumerate(times):
        if t < minimum_iteration or (regular and t % 1000):
            continue
        start = bisect_right(times, t-window, 0, k)-1
        if start < 0:
            continue
        changed = prefix[k+1]-prefix[start+1]
        if changed/unknown_nodes > budget:
            continue
        j = bisect_right(check_times,t)-1
        if j < 0:
            continue
        counter = int(checks[j]['counter'])
        if joint and counter <= threshold:
            continue
        return dict(iteration=t,window_start=times[start],window_end=t,
                    transition_count=changed,transition_rate=changed/unknown_nodes,
                    counter=counter,counter_iteration=check_times[j],
                    counter_age=t-check_times[j],
                    intermediate_observations=sum(times[start]<s<=t and s%1000!=0
                                                  for s in times[:k+1]))
    return None
