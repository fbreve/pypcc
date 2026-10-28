"""Read-only propagation metrics, computed after the existing stopping loop."""


def propagation_info(iterations, early_stop, max_iter, stop_max, stop_cnt):
    stopped = bool(early_stop and stop_cnt > stop_max)
    return {
        "effective_iterations": int(iterations),
        "checks": int((iterations + 9) // 10) if early_stop else 0,
        "stop_reason": "early_stop" if stopped else "max_iter",
        "reached_max_iter": bool(iterations == max_iter),
        "stop_max": int(stop_max),
        "stalled_checks": int(stop_cnt),
    }


class DominanceChangeStopTracker:
    """Opt-in experimental convergence observer; does not alter propagation.

    Call observe() at the existing check cadence (after iterations 1, 11, ...).
    The first observation establishes a baseline and cannot request stopping.
    Subsequent observations count consecutive changes <= tolerance; stopping
    is requested after more than stop_max such checks, matching legacy patience.
    """

    def __init__(self, tolerance=0.0, stop_max=0):
        import math
        import numbers
        if (isinstance(tolerance, bool) or not isinstance(tolerance, numbers.Real)
                or not math.isfinite(tolerance) or tolerance < 0):
            raise ValueError("tolerance must be a finite nonnegative real number")
        if (isinstance(stop_max, bool) or not isinstance(stop_max, numbers.Integral)
                or stop_max < 0):
            raise ValueError("stop_max must be a nonnegative integer")
        self.tolerance = float(tolerance)
        self.stop_max = int(stop_max)
        self.stalled_checks = 0
        self.checks = 0
        self.last_change = None
        self._previous = None

    def observe(self, dominance):
        import numpy as np
        current = np.asarray(dominance)
        if current.ndim != 2 or current.size == 0 or not np.all(np.isfinite(current)):
            raise ValueError("dominance must be a nonempty finite matrix")
        if self._previous is not None and current.shape != self._previous.shape:
            raise ValueError("dominance shape must remain constant between checks")
        self.checks += 1
        if self._previous is None:
            self._previous = current.copy()
            return False
        self.last_change = float(np.max(np.abs(current - self._previous)))
        self._previous = current.copy()
        if self.last_change <= self.tolerance:
            self.stalled_checks += 1
        else:
            self.stalled_checks = 0
        return self.stalled_checks > self.stop_max
