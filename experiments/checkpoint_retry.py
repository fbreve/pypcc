"""Bounded retries for experimental checkpoint I/O, outside PCC propagation."""
import time


def retry_writer(writer, *, attempts=40, pause=.05, sleeper=time.sleep, notify=None):
    if attempts < 1:
        raise ValueError('attempts must be positive')

    def guarded(path, state):
        for attempt in range(attempts):
            try:
                return writer(path, state)
            except PermissionError as error:
                if getattr(error, 'winerror', None) not in (5, 32) or attempt + 1 == attempts:
                    raise
                if notify is not None:
                    notify(path, attempt + 1, error.winerror)
                sleeper(pause)
    return guarded
