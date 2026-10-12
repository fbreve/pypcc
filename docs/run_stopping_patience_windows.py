"""Run the existing --full workflow with bounded Windows checkpoint-write retries.

This optional I/O wrapper preserves the frozen driver/observer source hashes and
completed-case identities. Retry delays are included in observed trajectory time.
"""
import datetime
import hashlib
import json
import runpy
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import experiments.stopping_patience as experiment
from experiments.checkpoint_retry import retry_writer

ROOT = Path('docs/benchmarks/stopping_patience')


def main():
    if sys.argv[1:] != ['--full']:
        raise ValueError('This launcher accepts only --full')
    retries = []

    def notify(path, attempt, winerror):
        retries.append(dict(path=str(path), attempt=attempt, winerror=winerror,
                            utc=datetime.datetime.now(datetime.timezone.utc).isoformat()))
        print(f'Checkpoint retry {attempt}: {path} (WinError {winerror})', flush=True)

    experiment.atomic_pickle = retry_writer(experiment.atomic_pickle, notify=notify)
    started = datetime.datetime.now(datetime.timezone.utc)
    begin = time.perf_counter()
    success = False
    try:
        sys.argv = ['docs/study_stopping_patience.py', '--full']
        runpy.run_path(sys.argv[0], run_name='__main__')
        success = True
    finally:
        metadata = dict(started_utc=started.isoformat(),
                        finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                        wall_s=time.perf_counter()-begin, success=success, retries=retries,
                        wrapper_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                        retry_helper_sha256=hashlib.sha256(Path('experiments/checkpoint_retry.py').read_bytes()).hexdigest())
        (ROOT/'phase2_retry_execution.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
