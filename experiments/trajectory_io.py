"""Validated v2 trajectory bundles. Checksums detect damage, not authenticity."""
import copy
import csv
import hashlib
import importlib.metadata
import io
import json
import math
import platform
import re
import subprocess
import time
from pathlib import Path

import numpy as np

PROTOCOL = 'convergence-trajectories-stage-a-v2'
UNSET = object()
HASH = re.compile(r'^[0-9a-f]{64}$')
STATE_KEYS = {'assignments', 'dominance', 'positions', 'strengths', 'distances', 'owndeg'}
SOURCE_FILES = ['pcc.py', 'pcc_numpy.py', 'pcc_numba.py', 'pcc_metrics.py',
                'experiments/convergence_trajectories.py', 'experiments/trajectory_io.py',
                'experiments/numba_trajectories.py']
LEGACY_SOURCE_FILES = set(SOURCE_FILES)
SOURCE_FILES += ['pcc_step.pyx', 'experiments/native_trajectories.py']
ROW_KEYS = {'iteration', 'assignments', 'class_changes', 'dominance_mean_abs_change',
            'dominance_max_abs_change', 'margin_mean', 'margin_min',
            'accuracy_unlabeled_retrospective', 'state_hashes', 'elapsed_instrumented_s',
            'final', 'stop_reason'}


def source_metadata(profile=None):
    began = time.perf_counter()
    root = Path(__file__).resolve().parents[1]
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip()
    commit_end = time.perf_counter()
    dirty = bool(subprocess.check_output(['git', 'status', '--porcelain'], cwd=root, text=True).strip())
    git_end = time.perf_counter()
    def digest(path):
        return hashlib.sha256((root / path).read_bytes().replace(b'\r\n', b'\n')).hexdigest()
    versions = {}
    for name in ('numpy', 'numba', 'scipy', 'scikit-learn'):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = 'not-installed'
    versions_end = time.perf_counter()
    protocol_hash = digest('docs/convergence_trajectories_protocol.md')
    source_hashes = {name: digest(name) for name in SOURCE_FILES}
    hashes_end = time.perf_counter()
    platform_name = platform.platform()
    platform_end = time.perf_counter()
    if profile is not None:
        profile.update(git_commit_s=commit_end - began, git_status_s=git_end - commit_end,
            dependency_versions_s=versions_end - git_end, source_hashes_s=hashes_end - versions_end,
            platform_s=platform_end - hashes_end)
    return dict(protocol=PROTOCOL, source_commit=commit, source_dirty=dirty,
                protocol_sha256=protocol_hash, source_hashes=source_hashes,
                python=platform.python_version(), platform=platform_name,
                dependencies=versions)


def metadata(*, dataset, label_seed, propagation_seed=UNSET, arrays=None, parameters=None,
             collector=None, backend=None, k_nn=None):
    """Prefer collector= for captured execution facts; legacy claims are checked at export.

    Label seed is a caller-declared fixture annotation, not a verified RNG action.
    No collector means arrays and parameters must be supplied, never omitted.
    """
    from .convergence_trajectories import array_hash, effective_parameters
    if not isinstance(dataset, str) or not dataset.strip():
        raise ValueError('dataset must be nonempty')
    if type(label_seed) is not int or not 0 <= label_seed < 2**32:
        raise ValueError('label_seed must be an unsigned 32-bit integer')
    if collector is not None and collector.execution is None:
        raise ValueError('collector has no captured execution; use observe_fit')
    facts = copy.deepcopy(collector.execution) if collector is not None else source_metadata()
    if collector is None and (arrays is None or parameters is None):
        raise ValueError('unbound metadata requires arrays and parameters')
    if arrays is not None:
        required = {'neighbors', 'degrees', 'labels'}
        if not required <= set(arrays) or set(arrays) - required - {'truth'}:
            raise ValueError('require neighbors, degrees, labels and optional truth')
        n = np.asarray(arrays['labels']).size
        if (np.asarray(arrays['labels']).shape != (n,) or
                np.asarray(arrays['degrees']).shape != (n,) or
                np.asarray(arrays['neighbors']).ndim != 2 or np.asarray(arrays['neighbors']).shape[0] != n or
                ('truth' in arrays and np.asarray(arrays['truth']).shape != (n,))):
            raise ValueError('inconsistent fixture shapes')
        # PCC executes with int64 graph and label arrays, not arbitrary input dtypes.
        hashes = {key: array_hash(np.asarray(value, dtype=np.int64)) for key, value in arrays.items()}
        if collector is not None and hashes != facts['fixture_hashes']:
            raise ValueError('input arrays do not match observed execution')
        facts['fixture_hashes'] = hashes
    if parameters is not None:
        effective = effective_parameters(parameters)
        if collector is not None and effective != facts['parameters']:
            raise ValueError('parameters do not match observed execution')
        facts['parameters'] = effective
    if backend is not None:
        if backend not in ('numpy', 'numba', 'cython') or (collector is not None and backend != facts['backend']):
            raise ValueError('backend mismatch')
        facts['backend'] = backend
    elif collector is None:
        facts['backend'] = 'numpy'  # legacy interface
    if k_nn is not None:
        if collector is not None and int(k_nn) != facts['k_nn']:
            raise ValueError('k_nn mismatch')
        facts['k_nn'] = int(k_nn)
    if propagation_seed is not UNSET:
        if propagation_seed is not None and (type(propagation_seed) is not int or not 0 <= propagation_seed < 2**32):
            raise ValueError('invalid propagation seed')
        if collector is not None:
            check_seed(facts, propagation_seed)
        facts['propagation_seed'] = propagation_seed
    elif collector is None:
        facts['propagation_seed'] = None
    facts.update(dataset=dataset, label_seed=label_seed, label_seed_verification='caller-declared',
                 evaluation='retrospective transductive diagnostic only',
                 timing='instrumented; not an ordinary backend benchmark')
    return facts


def check_seed(facts, claimed):
    from .convergence_trajectories import numpy_rng_hash
    actual = facts['propagation_seed']
    if claimed == actual:
        return
    # Legacy NumPy callers often seed immediately before observe_fit. Verify the
    # complete initial state; any intervening draw makes that claim inconsistent.
    if actual is None and claimed is not None and facts['backend'] == 'numpy':
        if numpy_rng_hash(np.random.RandomState(claimed).get_state()) == facts['initial_numpy_rng']:
            return
    raise ValueError('seed claim cannot be verified against observed execution')


def bound_metadata(collector, provenance):
    if collector.execution is None:
        raise ValueError('export requires captured execution; use observe_fit')
    facts = copy.deepcopy(collector.execution)
    allowed = set(facts) | {'dataset', 'label_seed', 'label_seed_verification', 'evaluation',
                           'timing', 'metrics', 'milestones', 'include_checks'}
    if not isinstance(provenance, dict) or set(provenance) - allowed:
        raise ValueError('unknown provenance fields')
    for key in facts:
        if key == 'propagation_seed':
            continue
        if key in provenance and provenance[key] != facts[key]:
            raise ValueError('metadata mismatch: ' + key)
    for key in ('dataset', 'label_seed', 'fixture_hashes', 'parameters', 'propagation_seed'):
        if key not in provenance:
            raise ValueError('missing provenance: ' + key)
    for key, expected in dict(metrics=collector.metrics, milestones=collector.milestones,
                              include_checks=collector.include_checks).items():
        if key in provenance and provenance[key] != expected:
            raise ValueError('metadata mismatch: ' + key)
    check_seed(facts, provenance['propagation_seed'])
    seed = provenance['propagation_seed']
    if seed != facts['propagation_seed']:
        facts['seed_policy'] = 'numpy-caller-seed-verified'
        facts['propagation_seed'] = seed
    # Never let arbitrary caller fields replace execution facts.
    for key in ('dataset', 'label_seed', 'label_seed_verification', 'evaluation', 'timing'):
        facts[key] = provenance[key]
    facts.update(milestones=collector.milestones, include_checks=collector.include_checks,
                 metrics=collector.metrics)
    return facts


def integer(value, minimum=0):
    return type(value) is int and value >= minimum


def finite(value, minimum=0., maximum=None):
    return type(value) in (int, float) and math.isfinite(value) and value >= minimum and (
        maximum is None or value <= maximum)


def validate_payload(payload):
    """Fail closed on shape, ordering, schedule, stop metrics and provenance."""
    from .convergence_trajectories import Trajectory, array_hash, effective_parameters, schedule_for
    from pcc_metrics import propagation_info
    if not isinstance(payload, dict) or set(payload) != {'metadata', 'observations'}:
        raise ValueError('invalid trajectory payload')
    m, rows = payload['metadata'], payload['observations']
    required = {'protocol', 'protocol_sha256', 'source_commit', 'source_dirty', 'source_hashes',
        'python', 'platform', 'dependencies', 'dataset', 'label_seed', 'label_seed_verification',
        'backend', 'parameters', 'fixture_hashes', 'nodes', 'classes', 'particles', 'graph_has_edges', 'class_index_to_label',
        'k_nn', 'stop_max', 'truth_present', 'propagation_seed', 'seed_policy', 'initial_numpy_rng',
        'rng_fingerprint', 'budget', 'milestones', 'include_checks', 'metrics', 'evaluation', 'timing'}
    if not isinstance(m, dict) or not required <= set(m) or m['protocol'] != PROTOCOL:
        raise ValueError('missing/unsupported trajectory metadata')
    if set(m) != required | ({'timings'} if m['backend'] in ('numba', 'cython') else set()):
        raise ValueError('unexpected metadata fields')
    if (m['backend'] not in ('numpy', 'numba', 'cython') or not isinstance(m['dataset'], str) or not m['dataset'].strip()
            or not integer(m['label_seed']) or m['label_seed'] >= 2**32
            or m['label_seed_verification'] != 'caller-declared'
            or type(m['source_dirty']) is not bool or not isinstance(m['source_commit'], str)
            or re.fullmatch('[0-9a-f]{40}', m['source_commit']) is None):
        raise ValueError('invalid provenance')
    for field in ('source_hashes', 'fixture_hashes'):
        if not isinstance(m[field], dict) or not m[field] or any(
                not isinstance(v, str) or HASH.fullmatch(v) is None for v in m[field].values()):
            raise ValueError('invalid hashes: ' + field)
    if not isinstance(m['protocol_sha256'], str) or HASH.fullmatch(m['protocol_sha256']) is None:
        raise ValueError('invalid protocol hash')
    if set(m['source_hashes']) != set(SOURCE_FILES) and not (
            m['backend'] != 'cython' and set(m['source_hashes']) == LEGACY_SOURCE_FILES):
        raise ValueError('incomplete source fingerprints')
    if any(not isinstance(m[k], str) or not m[k] for k in ('python', 'platform')):
        raise ValueError('missing environment metadata')
    if m['evaluation'] != 'retrospective transductive diagnostic only' or m['timing'] != 'instrumented; not an ordinary backend benchmark':
        raise ValueError('invalid diagnostic policy')
    if not isinstance(m['dependencies'], dict) or not {'numpy', 'numba', 'scipy', 'scikit-learn'} <= set(m['dependencies']):
        raise ValueError('missing dependencies')
    if any(not isinstance(v, str) or not v for v in m['dependencies'].values()):
        raise ValueError('invalid dependency version')
    p = m['parameters']
    if not isinstance(p, dict) or p != effective_parameters(p):
        raise ValueError('incomplete effective parameters')
    if set(p) != set(effective_parameters({})) or type(p['early_stop']) is not bool or any(
            not integer(p[k]) for k in ('max_iter', 'es_chk')):
        raise ValueError('invalid effective parameters')
    if not integer(m['nodes'], 1) or not integer(m['classes'], 2) or not integer(m['k_nn']) or not integer(m['stop_max'], 1):
        raise ValueError('invalid execution dimensions')
    if not integer(m['particles'], m['classes']) or m['particles'] > m['nodes'] or type(m['graph_has_edges']) is not bool:
        raise ValueError('invalid particle/graph metadata')
    if m['graph_has_edges'] and m['k_nn'] == 0:
        raise ValueError('invalid effective k_nn')
    stop_max = max(1, round(m['nodes'] / (m['particles'] * m['k_nn']) * round(p['es_chk'] * .1))) if m['graph_has_edges'] else 1
    if m['stop_max'] != stop_max:
        raise ValueError('inconsistent effective stop threshold')
    mapping = m['class_index_to_label']
    if not isinstance(mapping, list) or len(mapping) != m['classes'] or any(type(i) is not int or i == -1 for i in mapping) or mapping != sorted(set(mapping)):
        raise ValueError('invalid class mapping')
    if type(m['truth_present']) is not bool or set(m['fixture_hashes']) != (
            {'neighbors', 'degrees', 'labels', 'truth'} if m['truth_present'] else {'neighbors', 'degrees', 'labels'}):
        raise ValueError('fixture/truth metadata mismatch')
    seed = m['propagation_seed']
    if seed is not None and (not integer(seed) or seed >= 2**32):
        raise ValueError('invalid seed')
    policy = m['seed_policy']
    allowed = {m['backend'] + '-caller-state'} if seed is None else {m['backend'] + '-seed-once'}
    if m['backend'] == 'numpy' and seed is not None:
        allowed.add('numpy-caller-seed-verified')
    if m['backend'] == 'cython':
        allowed = {'cython-fixed-seed'} if seed == 123456789 else set()
    if policy not in allowed:
        raise ValueError('seed policy mismatch')
    if m['backend'] == 'numpy':
        if m['rng_fingerprint'] != 'numpy-state' or not isinstance(m['initial_numpy_rng'], str) or HASH.fullmatch(m['initial_numpy_rng']) is None:
            raise ValueError('invalid NumPy RNG metadata')
        if seed is not None:
            from .convergence_trajectories import numpy_rng_hash
            if m['initial_numpy_rng'] != numpy_rng_hash(np.random.RandomState(seed).get_state()):
                raise ValueError('initial RNG/seed mismatch')
    elif m['backend'] == 'cython':
        if m['rng_fingerprint'] != 'xorshift32-state' or m['initial_numpy_rng'] is not None:
            raise ValueError('invalid Cython RNG metadata')
    elif m['rng_fingerprint'] != 'not-captured' or m['initial_numpy_rng'] is not None:
        raise ValueError('invalid Numba RNG metadata')
    b = m['budget']
    if not isinstance(b, dict) or set(b) != {'max_observations', 'max_bytes', 'scheduled_observations', 'native_buffer_bytes', 'estimated_peak_bytes'} or any(not integer(v) for v in b.values()):
        raise ValueError('invalid observation budget')
    probe = Trajectory([], milestones=m['milestones'], include_checks=m['include_checks'],
                       max_observations=b['max_observations'], max_bytes=b['max_bytes'])
    schedule = schedule_for(p['max_iter'], probe.milestones, probe.include_checks,
                            min(probe.max_observations, b['max_bytes'] // 16384 + 1))
    native_bytes = len(schedule) * (16 * m['nodes'] * m['classes'] + 16 * m['particles'] + m['nodes'] * m['particles'] + 16 + (4 if m['backend'] == 'cython' else 0)) + 16 if m['backend'] in ('numba', 'cython') else 0
    if b['native_buffer_bytes'] != native_bytes:
        raise ValueError('inconsistent native buffer size')
    expected_bytes = native_bytes + len(schedule) * (128 * m['nodes'] + 16384) + 128 * m['nodes'] * m['classes']
    if b['scheduled_observations'] != len(schedule) or b['estimated_peak_bytes'] != expected_bytes or b['estimated_peak_bytes'] > b['max_bytes']:
        raise ValueError('inconsistent storage budget')
    info = m['metrics']
    if not isinstance(info, dict) or any(not integer(info.get(k)) for k in (
            'effective_iterations', 'stalled_checks', 'checks', 'stop_max')) or type(info.get('reached_max_iter')) is not bool:
        raise ValueError('invalid stopping metrics')
    final_it = info['effective_iterations']
    expected = propagation_info(final_it, p['early_stop'], p['max_iter'], m['stop_max'], info['stalled_checks'])
    if (info != expected or final_it > p['max_iter'] or info['stalled_checks'] > info['checks'] or
            (info['stop_reason'] == 'max_iter' and final_it != p['max_iter']) or
            (info['stop_reason'] == 'early_stop' and (final_it == 0 or (final_it - 1) % 10 != 0))):
        raise ValueError('inconsistent stopping metrics')
    expected_iterations = [i for i in schedule if i < final_it] + [final_it]
    if not isinstance(rows, list) or len(rows) != len(expected_iterations):
        raise ValueError('incomplete observation sequence')
    previous = None
    previous_time = 0.
    for index, (row, iteration) in enumerate(zip(rows, expected_iterations)):
        if not isinstance(row, dict) or set(row) != ROW_KEYS:
            raise ValueError('invalid observation schema')
        if not integer(row['iteration']) or row['iteration'] != iteration:
            raise ValueError('invalid iteration ordering/schedule')
        final = index == len(rows) - 1
        if type(row['final']) is not bool or row['final'] != final or row['stop_reason'] != (info['stop_reason'] if final else None):
            raise ValueError('invalid terminal observation')
        assignments = row['assignments']
        if not isinstance(assignments, list) or len(assignments) != m['nodes'] or any(
                not integer(v) or v >= m['classes'] for v in assignments):
            raise ValueError('invalid assignments')
        hashes = row['state_hashes']
        keys = STATE_KEYS | ({'numpy_rng'} if m['backend'] == 'numpy' else {'cython_rng'} if m['backend'] == 'cython' else set())
        if not isinstance(hashes, dict) or set(hashes) != keys or any(
                not isinstance(v, str) or HASH.fullmatch(v) is None for v in hashes.values()):
            raise ValueError('invalid observation hashes')
        if hashes['assignments'] != array_hash(np.array(assignments, dtype=np.int64)):
            raise ValueError('assignment fingerprint mismatch')
        if iteration == 0 and m['backend'] == 'cython' and hashes['cython_rng'] != array_hash(np.array([123456789], dtype=np.uint32)):
            raise ValueError('initial Cython RNG fingerprint mismatch')
        if iteration == 0 and m['backend'] == 'numpy' and hashes['numpy_rng'] != m['initial_numpy_rng']:
            raise ValueError('initial NumPy RNG fingerprint mismatch')
        changes = None if previous is None else sum(a != b for a, b in zip(assignments, previous))
        if row['class_changes'] != changes or (changes is not None and not integer(row['class_changes'])):
            raise ValueError('class change count mismatch')
        for key in ('dominance_mean_abs_change', 'dominance_max_abs_change'):
            if (previous is None and row[key] is not None) or (previous is not None and not finite(row[key])):
                raise ValueError('invalid stability summary')
        if not finite(row['margin_min']) or not finite(row['margin_mean']) or row['margin_min'] > row['margin_mean']:
            raise ValueError('invalid margins')
        accuracy = row['accuracy_unlabeled_retrospective']
        accuracy_available = m['truth_present'] and m['particles'] < m['nodes']
        if (accuracy_available and not finite(accuracy, maximum=1.)) or (not accuracy_available and accuracy is not None):
            raise ValueError('invalid retrospective accuracy')
        elapsed = row['elapsed_instrumented_s']
        if m['backend'] in ('numba', 'cython') and not final:
            if elapsed is not None:
                raise ValueError('native intermediate timing is unavailable')
        elif not finite(elapsed) or elapsed < previous_time:
            raise ValueError('invalid elapsed time')
        else:
            previous_time = elapsed
        previous = assignments
    if m['backend'] in ('numba', 'cython'):
        timings = m.get('timings')
        if not isinstance(timings, dict) or set(timings) != {'native_propagation_s', 'postprocessing_s', 'compilation_occurred'} or any(
                not finite(timings[k]) for k in ('native_propagation_s', 'postprocessing_s')) or type(timings['compilation_occurred']) is not bool:
            raise ValueError('invalid native timings')
        if rows[-1]['elapsed_instrumented_s'] != timings['native_propagation_s']:
            raise ValueError('native timing mismatch')
    return payload


def csv_text(payload):
    stream = io.StringIO(newline='')
    context = {k: payload['metadata'][k] for k in ('dataset', 'label_seed', 'backend')}
    rows = payload['observations']
    writer = csv.DictWriter(stream, fieldnames=[*context, *rows[0]])
    writer.writeheader()
    for row in rows:
        writer.writerow({key: json.dumps(value, allow_nan=False) if isinstance(value, (list, dict))
                         else value for key, value in dict(context, **row).items()})
    return stream.getvalue()


def write_output(directory, collector, provenance):
    if not collector.finished:
        raise ValueError('cannot export incomplete trajectory')
    payload = dict(metadata=bound_metadata(collector, provenance), observations=collector.rows)
    validate_payload(payload)  # validate before creating even a partial bundle
    json_data = json.dumps(payload, indent=2, allow_nan=False) + '\n'
    csv_data = csv_text(payload)
    if len(json_data.encode('utf-8')) + len(csv_data.encode('utf-8')) > collector.max_bytes:
        raise ValueError('serialized output exceeds storage budget')
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    (directory / 'trajectory.json').write_text(json_data, encoding='utf-8')
    with (directory / 'trajectory.csv').open('w', newline='', encoding='utf-8') as stream:
        stream.write(csv_data)
    manifest = {name: hashlib.sha256((directory / name).read_bytes()).hexdigest()
                for name in ('trajectory.json', 'trajectory.csv')}
    (directory / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')


def read_output(directory):
    directory = Path(directory)
    manifest = json.loads((directory / 'manifest.json').read_text(encoding='utf-8'))
    if not isinstance(manifest, dict) or set(manifest) != {'trajectory.json', 'trajectory.csv'}:
        raise ValueError('invalid trajectory manifest')
    for name, expected in manifest.items():
        if hashlib.sha256((directory / name).read_bytes()).hexdigest() != expected:
            raise ValueError('trajectory checksum mismatch: ' + name)
    payload = validate_payload(json.loads((directory / 'trajectory.json').read_text(encoding='utf-8')))
    with (directory / 'trajectory.csv').open(newline='', encoding='utf-8') as stream:
        if stream.read() != csv_text(payload):
            raise ValueError('CSV and JSON observations/metadata disagree')
    return payload
