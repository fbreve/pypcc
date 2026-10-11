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
