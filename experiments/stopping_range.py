"""Experimental observer around unchanged sequential Numba kernels.

No configuration is added to PCC. Windows follow the PR #5 sampling phase:
1..501, 501..1001, ... (51 original checks including shared endpoints).
"""
import pcc_numba

THRESHOLD = 5e-4
WINDOW = 500
PATIENCE = 3
FIRST_ALLOWED = 5001


class RangeObserver:
    def __init__(self):
        self.start = None
        self.low = self.high = 0.
        self.streak = self.windows = 0
        self.last_range = None

    def observe(self, iteration, metric):
        if self.start is None:
            self.start = iteration
            self.low = self.high = metric
            return False
        self.low = min(self.low, metric)
        self.high = max(self.high, metric)
        if iteration-self.start != WINDOW:
            return False
        self.last_range = self.high-self.low
        self.windows += 1
        self.streak = self.streak+1 if self.last_range <= THRESHOLD else 0
        self.start = iteration
        self.low = self.high = metric
        return iteration >= FIRST_ALLOWED and self.streak >= PATIENCE


def propagate_range(*args):
    """Same input arrays as low-level NumPy/Numba propagation; experimental only.

    Supports the experiment's early_stop=True protocol only. RNG is never seeded
    or inspected here. Original stopping has priority if both criteria fire.
    """
    if len(args) != 26 or not args[23]:
        raise ValueError('experimental observer requires the existing 26 arguments and early_stop=True')
    degree = pcc_numba.validate_step_inputs(*args[:3], *args[5:13])
    pcc_numba.validate_buffers(args[5], degree, *args[15:22])
    cap, stop_max = args[22], args[25]
    n, c = args[0].shape[0], args[5]
    observer = RangeObserver()
    iteration = checks = stalled = 0
    best = 0.
    reason = 'max_iter'
    while iteration < cap:
        end = min(cap, 1 if iteration == 0 else iteration+10)
        pcc_numba._pcc_propagate_numba_sequential(
            *args[:22], end-iteration, False, args[24], stop_max, False)
        iteration = end
        if (iteration-1) % 10 != 0:
            continue
        metric = float(pcc_numba._pcc_calc_mmpot_sequential(args[11], n, c))
        checks += 1
        if metric > best:
            best, stalled = metric, 0
        else:
            stalled += 1
        alternative = observer.observe(iteration, metric)
        if stalled > stop_max:
            reason = 'early_stop'
            break
        if alternative:
            reason = 'alternative_range'
            break
    return dict(effective_iterations=iteration, checks=checks, stop_reason=reason,
                reached_max_iter=iteration==cap, stop_max=stop_max, stalled_checks=stalled,
                range_windows=observer.windows, qualifying_windows=observer.streak,
                last_window_range=observer.last_range)
