"""Optional shadow stopping and exact recovery; production kernels stay unchanged."""
import copy
import csv
import hashlib
import json
import pickle
import time
import platform
from pathlib import Path

import numpy as np
from numba import _helperlib

import pcc_numba
import numba
from experiments.convergence_measurements import norms

MULTIPLIERS = (.25, .5, .75, 1., 1.5, 2., 4.)
STATE_INDICES = (7, 8, 9, 10, 11, 12)


def original_threshold(nodes, particles, k_nn, es_chk=2000, *, isolated=False):
    if isolated or k_nn == 0:
        return 1
    return max(1, round((nodes / (particles * k_nn)) * round(es_chk * .1)))


def adjusted_threshold(original, multiplier):
    return original if multiplier == 1. else max(1, round(original * multiplier))


class Observer:
    def __init__(self, threshold):
        self.original = threshold
        self.best = 0.
        self.counter = self.checks = self.resets = 0
        self.events = {}

    def check(self, iteration, metric):
        if (iteration - 1) % 10:
            raise ValueError('Not an original stopping checkpoint')
        self.checks += 1
        gain = max(0., metric - self.best)
        record = metric > self.best
        if record:
            self.best, self.counter = metric, 0
            self.resets += 1
        else:
            self.counter += 1
        fired = []
        for multiplier in MULTIPLIERS:
            if multiplier not in self.events and self.counter > adjusted_threshold(self.original, multiplier):
                self.events[multiplier] = dict(iteration=iteration, resets=self.resets,
                                              checks=self.checks, metric=metric)
                fired.append(multiplier)
        return fired, gain, record


def rng_state():
    return _helperlib.rnd_get_state(_helperlib.rnd_get_np_state_ptr())


def restore_rng(value):
    # Numba private MT API: version recorded, round-trip verified in tests.
    _helperlib.rnd_set_state(_helperlib.rnd_get_np_state_ptr(), value)


def predicted(a):
    labels = a[2].copy()
    unknown = labels == -1
    labels[unknown] = np.argmax(a[11][unknown], axis=1)
    return labels


def accuracy(labels, case):
    unknown = case['labels'] == -1
    correct = int(np.count_nonzero(labels[unknown] == case['truth'][unknown]))
    return correct, correct / int(unknown.sum())


class CSVStream:
    def __init__(self, path, fields, offset=None):
        self.file = path.open('r+' if offset is not None else 'w', encoding='utf-8', newline='')
        if offset is not None:
            self.file.seek(offset)
            self.file.truncate()
        self.writer = csv.DictWriter(self.file, fieldnames=fields)
        if offset is None:
            self.writer.writeheader()

    def write(self, row):
        self.writer.writerow(row)

    def offset(self):
        self.file.flush()
        return self.file.tell()

    def close(self):
        self.file.close()


def atomic_pickle(path, value):
    temporary = path.with_suffix('.tmp')
    with temporary.open('wb') as stream:
        pickle.dump(value, stream, protocol=5)
        stream.flush()
        # Flush user-space data. Power-loss durability is not guaranteed.
    temporary.replace(path)


def identity(protocol, case, fingerprint):
    return hashlib.sha256(json.dumps(dict(protocol=protocol, dataset=case['dataset'],
                    seed=case['seed'], data=fingerprint, runtime=runtime_identity()), sort_keys=True).encode()).hexdigest()


def runtime_identity():
    return dict(python=platform.python_version(), numpy=np.__version__, numba=numba.__version__,
        source_sha256={str(Path(path).name): hashlib.sha256(Path(path).read_bytes().replace(b'\r\n', b'\n')).hexdigest()
            for path in (__file__, pcc_numba.__file__)})


def run(initial, case, cap, directory, run_id, seed_function, *, resume=True,
        spacing=1000, recovery_spacing=10000, references=(30000, 100000, 200000, 500000),
        verify_points=(), interrupt_at=None):
    """One logical continuous trajectory; all seven predicates are shadow observers.

    interrupt_at is a deterministic fault-injection hook for recovery tests.
    Completed cases are reused only when identity matches. Local pickle files are
    trusted recovery artifacts, not a portable archival format.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    complete, recovery = directory / 'complete.json', directory / 'recovery.pkl'
    if resume and complete.exists():
        result = json.loads(complete.read_text(encoding='utf-8'))
        if result['identity'] != run_id or result.get('runtime') != runtime_identity() or result['iterations'] != cap:
            raise ValueError('Completed case identity mismatch')
        return result, None, []
    if resume and recovery.exists():
        with recovery.open('rb') as stream:
            state = pickle.load(stream)
        if state['identity'] != run_id or state['cap'] != cap or state.get('runtime') != runtime_identity():
            raise ValueError('Recovery identity/budget mismatch')
        restore_rng(state['rng'])
        offsets = state['offsets']
    else:
        a = copy.deepcopy(initial)
        n = len(case['labels'])
        state = dict(identity=run_id, runtime=runtime_identity(), cap=cap, args=a, observer=Observer(a[25]),
                     iteration=0, events={}, previous_labels=predicted(a),
                     elapsed_s=0., checkpoint_io_s=0., samples=0)
        seed_function(case['seed'])
        offsets = {}
    a = state['args']
    max_degree = pcc_numba.validate_step_inputs(*a[:3], *a[5:13])
    pcc_numba.validate_buffers(a[5], max_degree, *a[15:22])
    streams = {
        'checks': CSVStream(directory / 'checks.csv',
            ['iteration', 'metric', 'record_gain', 'record', 'counter', 'resets'], offsets.get('checks')),
        'samples': CSVStream(directory / 'samples.csv',
            ['iteration', 'metric', 'correct_nodes', 'accuracy_unlabeled', 'changed_nodes', 'is_reference'], offsets.get('samples'))}
    saved = state.setdefault('verification_snapshots', {})
    tick = time.perf_counter()
    try:
        while state['iteration'] < cap:
            current = state['iteration']
            next_check = 1 if current == 0 else 1 + 10 * ((current - 1) // 10 + 1)
            next_sample = spacing * (current // spacing + 1)
            next_recovery = recovery_spacing * (current // recovery_spacing + 1)
            targets = [cap, next_check, next_sample, next_recovery]
            targets.extend(v for v in references if v > current)
            targets.extend(v for v in verify_points if v > current)
            if interrupt_at is not None and interrupt_at > current:
                targets.append(interrupt_at)
            target = min(targets)
            pcc_numba._pcc_propagate_numba_sequential(*a[:22], target-current, False, a[24], a[25], False)
            state['iteration'] = target
            is_check = (target - 1) % 10 == 0
            metric = float(pcc_numba._pcc_calc_mmpot_sequential(a[11], len(a[2]), a[5]))
            fired = []
            if is_check:
                fired, gain, record = state['observer'].check(target, metric)
                streams['checks'].write(dict(iteration=target, metric=metric, record_gain=gain,
                    record=int(record), counter=state['observer'].counter, resets=state['observer'].resets))
            if fired or target % spacing == 0 or target in references or target == cap:
                labels = predicted(a)
                correct, acc = accuracy(labels, case)
                changed = labels != state['previous_labels']
                for event in state['events'].values():
                    different = labels != event['labels']
                    event['ever_different'] |= different
                    event['returned'] |= event['ever_different'] & ~different
                    event['sampled_flips'] += changed
                for multiplier in fired:
                    state['events'][multiplier] = dict(**state['observer'].events[multiplier],
                        correct_nodes=correct, accuracy_unlabeled=acc, labels=labels.copy(),
                        dominance=a[11].copy(), ever_different=np.zeros(len(labels), bool),
                        returned=np.zeros(len(labels), bool), sampled_flips=np.zeros(len(labels), np.int64))
                streams['samples'].write(dict(iteration=target, metric=metric, correct_nodes=correct,
                    accuracy_unlabeled=acc, changed_nodes=int(changed.sum()), is_reference=int(target in references)))
                state['previous_labels'] = labels
                state['samples'] += 1
            if target in verify_points or 1. in fired:
                saved[target] = ([a[i].copy() for i in STATE_INDICES], rng_state())
            checkpoint = target % recovery_spacing == 0 or target == interrupt_at or target == cap
            if checkpoint:
                state['elapsed_s'] += time.perf_counter() - tick
                state['rng'] = rng_state()
                state['offsets'] = {key: stream.offset() for key, stream in streams.items()}
                begin_io = time.perf_counter()
                atomic_pickle(recovery, state)
                duration = time.perf_counter() - begin_io
                state['checkpoint_io_s'] += duration
                state['elapsed_s'] += duration
                tick = time.perf_counter()
            if target == interrupt_at:
                raise InterruptedError('Injected interruption after durable checkpoint')
    finally:
        for stream in streams.values():
            stream.close()
    final_rng = rng_state()
    verification = []
    for point, (expected, expected_rng) in saved.items():
        baseline = copy.deepcopy(initial)
        seed_function(case['seed'])
        pcc_numba._pcc_propagate_numba_sequential(*baseline[:22], point, False, baseline[24], baseline[25], False)
        equal = all(np.array_equal(baseline[i], value) for i, value in zip(STATE_INDICES, expected))
        rng_equal = rng_state() == expected_rng
        if not (equal and rng_equal):
            raise AssertionError('Instrumented state/RNG differs from monolithic trajectory')
        original_matches = None
        if state['observer'].events.get(1., {}).get('iteration') == point:
            original = copy.deepcopy(initial)
            seed_function(case['seed'])
            effective, _ = pcc_numba._pcc_propagate_numba_sequential(*original[:22], cap, True, original[24], original[25], True)
            original_matches = (effective == point and rng_state() == expected_rng and
                all(np.array_equal(original[i], value) for i, value in zip(STATE_INDICES, expected)))
            if not original_matches:
                raise AssertionError('Multiplier 1 event differs from original early stopping')
        verification.append(dict(iteration=point, state_equal=equal, rng_equal=rng_equal,
                                 original_stop_matches=original_matches))
    restore_rng(final_rng)
    labels = predicted(a)
    metric = float(pcc_numba._pcc_calc_mmpot_sequential(a[11], len(a[2]), a[5]))
    final_correct, final_accuracy = accuracy(labels, case)
    rows = []
    for multiplier in MULTIPLIERS:
        event = state['events'].get(multiplier)
        # For censoring, values are the cap observation, not a stopping event.
        row = dict(dataset=case['dataset'], seed=case['seed'], multiplier=multiplier,
            original_threshold=a[25], adjusted_threshold=adjusted_threshold(a[25], multiplier),
            status='stopped' if event else 'right_censored', stop_iteration=event['iteration'] if event else None,
            observation_iteration=event['iteration'] if event else cap,
            mean_max_dominance=event['metric'] if event else metric,
            final_mean_max_dominance=metric,
            resets=event['resets'] if event else state['observer'].resets,
            checks=event['checks'] if event else state['observer'].checks,
            iterations_saved=cap-event['iteration'] if event else 0,
            accuracy_unlabeled=event['accuracy_unlabeled'] if event else final_accuracy,
            final_accuracy_unlabeled=final_accuracy, correct_nodes=event['correct_nodes'] if event else final_correct,
            final_correct_nodes=final_correct,
            accuracy_difference=(event['accuracy_unlabeled']-final_accuracy) if event else 0.,
            correct_nodes_difference=(event['correct_nodes']-final_correct) if event else 0,
            final_prediction_changes=int(np.count_nonzero(labels != event['labels'])) if event else 0,
            sampled_ever_changed_nodes=int(event['ever_different'].sum()) if event else 0,
            sampled_returned_nodes=int(event['returned'].sum()) if event else 0,
            sampled_class_changes=int(event['sampled_flips'].sum()) if event else 0)
        row.update(norms(a[11], event['dominance'] if event else a[11]))
        rows.append(row)
    result = dict(identity=run_id, runtime=state['runtime'], dataset=case['dataset'], seed=case['seed'], iterations=cap,
                  checks=state['observer'].checks, samples=state['samples'], elapsed_s=state['elapsed_s'],
                  checkpoint_io_s=state['checkpoint_io_s'], rows=rows,
                  verification=verification,
                  rng_sha256=hashlib.sha256(repr(final_rng).encode()).hexdigest())
    temporary = complete.with_suffix('.tmp')
    temporary.write_text(json.dumps(result, indent=2), encoding='utf-8')
    temporary.replace(complete)
    return result, a, verification
