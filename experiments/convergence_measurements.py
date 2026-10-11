"""Pure diagnostic calculations, outside production PCC and its stopping rule."""
import numpy as np

LOW_GAP = .05  # Descriptive only; never used to stop or propagate.


def maxima_gaps(dominance):
    top = np.partition(dominance, -2, axis=1)[:, -2:]
    return top[:, 1], top[:, 1]-top[:, 0]


def norms(current, previous):
    delta = current-previous
    absolute = np.abs(delta)
    return dict(matrix_mean_abs=float(absolute.mean()),
                matrix_frobenius=float(np.sqrt(np.sum(delta*delta))),
                matrix_max_abs=float(absolute.max()),
                max_node_l1=float(absolute.sum(axis=1).max()))


def distribution(dominance, unknown):
    maxima, gaps = maxima_gaps(dominance)
    q = np.quantile(maxima, [0,.1,.25,.5,.75,.9,1])
    result = {key:float(value) for key,value in zip(
        ('maximum_min','maximum_p10','maximum_p25','maximum_p50','maximum_p75','maximum_p90','maximum_max'),q)}
    result.update(low_gap_fraction=float(np.mean(gaps < LOW_GAP)),
                  low_gap_fraction_unlabeled=float(np.mean(gaps[unknown] < LOW_GAP)),
                  gap_mean=float(gaps.mean()),gap_p10=float(np.quantile(gaps,.1)),gap_p50=float(np.median(gaps)))
    hist,_ = np.histogram(maxima,[0,.25,.5,.75,.9,.95,.99,1.000000001])
    for key,value in zip(('max_0_025','max_025_05','max_05_075','max_075_09','max_09_095','max_095_099','max_099_1'),hist):
        result[key] = int(value)
    return result


def hidden_changes(current, reference, ever_changed):
    return int(np.count_nonzero((current == reference) & ever_changed))


def is_checkpoint(iteration, original_event, cap=30000):
    return (iteration == cap or (iteration-1)%500 == 0 or
            (iteration >= 25001 and (iteration-1)%10 == 0) or
            (abs(iteration-original_event) <= 500 and (iteration-1)%10 == 0))
