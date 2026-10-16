"""Stage B preflight and whole-fit resumption. Real execution requires reviewed report SHA.

No downloads, historical warmup fits, loop checkpoints, or benchmark rewrites.
python -m experiments.convergence_pilot --help
"""
import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import socket
from dataclasses import asdict
from pathlib import Path
import time
import uuid

import numpy as np

from benchmarks.checkpoints import (CheckpointError, Store, canonical, digest, durable_file,
                                    file_inventory, read_record, sync_directory)
from benchmarks.fixtures import (Case, FixturePreparer, array_hash, file_hash, save_fixture,
                                 select_labels)
from benchmarks.historical import FROZEN, INVENTORY, PROTOCOL, fingerprint
from benchmarks.runner import expected_backend
from pcc import ParticleCompetitionAndCooperation as PCC
from .convergence_trajectories import (Trajectory, metadata, observe_fit, read_output,
                                      schedule_for, write_output)
from .trajectory_io import source_metadata

ROOT = Path(__file__).resolve().parents[1]
BACKENDS = ('numpy', 'numba', 'cython')
MILESTONES = (0, 1000, 2000, 5000, 10000, 20000, 40000, 60000, 80000, 100000)
MAX_BYTES = 64 * 1024 * 1024
MAX_OBSERVATIONS = 128


def specification(synthetic=False, iterations=25):
    if synthetic:
        if type(iterations) is not int or not 3 <= iterations <= 1000:
            raise ValueError('synthetic iterations must be 3..1000')
        cases = [Case(name, n, features, classes, synthetic=True, k_nn=3, label_seed=seed)
                 for name, n, features, classes in (('pilot_a', 24, 2, 2), ('pilot_b', 30, 3, 3))
                 for seed in (10, 11)]
        milestones = sorted({0, 1, 3, iterations})
    else:
        cases = [Case(name, FROZEN[name]['nodes'], FROZEN[name]['processed_features'],
                      FROZEN[name]['classes'], label_seed=seed)
                 for name in ('pageblocks', 'spambase') for seed in (10, 11)]
        iterations, milestones = 100000, list(MILESTONES)
    tasks = [dict(task_id=f't{len(BACKENDS)*i+j:04d}', case_id=c.case_id, backend=b)
             for i, c in enumerate(cases) for j, b in enumerate(BACKENDS)]
    return dict(protocol='synthetic-stage-b-v1' if synthetic else 'historical-stage-b-v1',
        synthetic=synthetic, cases=[asdict(c) for c in cases], tasks=tasks,
        parameters=dict({k: PROTOCOL[k] for k in ('p_grd', 'delta_v', 'deltap', 'dexp', 'es_chk')},
                        max_iter=iterations, early_stop=False), milestones=milestones,
        include_checks=False, max_observations=MAX_OBSERVATIONS, max_bytes=MAX_BYTES,
        propagation_seeds=dict(numpy=0, numba=0, cython=123456789),
        frozen_protocol_sha256=INVENTORY['protocol_sha256'])


def validate_spec(spec):
    expected = specification(spec['synthetic'], spec['parameters']['max_iter'])
    if spec != expected:
        raise CheckpointError('pilot specification changed')


def output_path(path):
    path = Path(path).resolve()
    if path.is_relative_to(ROOT) and not path.is_relative_to(ROOT / '.cache'):
        raise ValueError('pilot outputs inside the repository must be under .cache')
    return path


def execution_identity():
    source = source_metadata()
    for backend in BACKENDS:
        expected_backend(backend)  # reject fallback before any fit
    import pcc_step
    if not hasattr(pcc_step, '_TrajectoryBuffer'):
        raise CheckpointError('rebuild Cython Stage A instrumentation first')
    files = ['experiments/convergence_pilot.py', 'benchmarks/fixtures.py', 'benchmarks/historical.py',
             'benchmarks/runner.py', 'experiments/patience_data.py', 'pcc_graph.py', 'pcc_validation.py',
             'docs/benchmarks/stopping_patience/protocol.json', 'docs/benchmarks/stopping_patience/inventory.json']
    return dict(source_commit=source['source_commit'], source_hashes=source['source_hashes'],
        protocol_sha256=source['protocol_sha256'], python=source['python'],
        platform=source['platform'], dependencies=source['dependencies'],
        runner_hashes={f: hashlib.sha256((ROOT/f).read_bytes().replace(b'\r\n', b'\n')).hexdigest() for f in files},
        cython_extension_sha256=file_hash(pcc_step.__file__),
        host=socket.gethostname(), machine=platform.machine(), processor=platform.processor(),
        llvmlite=importlib.metadata.version('llvmlite'),
        thread_environment={k: os.environ.get(k) for k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS',
            'OPENBLAS_NUM_THREADS', 'NUMBA_NUM_THREADS', 'NUMBA_CPU_NAME', 'NUMBA_CPU_FEATURES')})


def validate_fixture(case, arrays, info):
    case.validate()
    names = {'raw', 'x', 'truth', 'labels', 'neighbors', 'degrees', 'source_indices', 'scaler_mean', 'scaler_scale'}
    if set(arrays) != names or info['case'] != asdict(case):
        raise CheckpointError('fixture case/inventory mismatch')
    hashes = {name: array_hash(value) for name, value in arrays.items()}
    if hashes != info['hashes'] or digest_fixture(hashes) != info['fixture_sha256']:
        raise CheckpointError('fixture array checksum mismatch')
    if (arrays['x'].shape != (case.nodes, case.features) or arrays['raw'].shape != arrays['x'].shape
            or arrays['truth'].shape != (case.nodes,) or arrays['labels'].shape != (case.nodes,)
            or arrays['neighbors'].ndim != 2 or arrays['neighbors'].shape[0] != case.nodes
            or arrays['degrees'].shape != (case.nodes,) or arrays['neighbors'].dtype != np.int64
            or arrays['degrees'].dtype != np.int64 or arrays['truth'].dtype != np.int64
            or arrays['labels'].dtype != np.int64 or arrays['raw'].dtype != np.float64 or arrays['x'].dtype != np.float64
            or not np.array_equal(arrays['source_indices'], np.arange(case.nodes))
            or arrays['scaler_mean'].shape != (case.features,) or arrays['scaler_scale'].shape != (case.features,)
            or not np.isfinite(arrays['scaler_mean']).all() or not np.isfinite(arrays['scaler_scale']).all()
            or np.any(arrays['scaler_scale'] <= 0)
            or len(np.unique(arrays['truth'])) != case.classes
            or not np.isfinite(arrays['raw']).all() or not np.isfinite(arrays['x']).all()
            or not np.array_equal(select_labels(arrays['truth'], case), arrays['labels'])
            or np.count_nonzero(arrays['labels'] != -1) != case.particles):
        raise CheckpointError('fixture dimensions, labels or data mismatch')
    for neighbors, degree in zip(arrays['neighbors'], arrays['degrees']):
        if not case.k_nn <= degree <= len(neighbors) or np.any(neighbors[:degree] < 0) or np.any(neighbors[:degree] >= case.nodes):
            raise CheckpointError('invalid ordered graph degrees/neighbors')
    if not case.synthetic:
        frozen = FROZEN[case.name]
        checks = dict(raw_data_fingerprint=fingerprint(arrays['raw'], arrays['truth']),
            processed_data_fingerprint=fingerprint(arrays['x'], arrays['truth']),
            graph_fingerprint=fingerprint(arrays['neighbors'], arrays['degrees']))
        if any(checks[k] != frozen[k] for k in checks) or fingerprint(arrays['labels']) != frozen['label_fingerprints'][str(case.label_seed)]:
            raise CheckpointError('frozen raw/processed/ordered-graph/label fingerprint mismatch')
    return hashes


def digest_fixture(hashes):
    # Existing fixture convention intentionally differs from checkpoint canonical().
    return hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()


def load_fixture(directory, case, seal=None):
    directory = Path(directory)
    if seal is not None and file_inventory(directory) != seal:
        raise CheckpointError('prepared fixture file checksum mismatch')
    if any((directory/name).is_symlink() for name in ('fixture.json', 'arrays.npz')):
        raise CheckpointError('fixture symlinks refused')
    info = json.loads((directory/'fixture.json').read_text(encoding='utf-8'))
    with np.load(directory/'arrays.npz', allow_pickle=False) as stored:
        arrays = {name: stored[name] for name in stored.files}
    validate_fixture(case, arrays, info)
    return arrays, info


def resources(case, backend, spec, arrays):
    count = len(schedule_for(spec['parameters']['max_iter'], spec['milestones'], False, MAX_OBSERVATIONS))
    n, c, p = case.nodes, case.classes, case.particles
    native = count * (16*n*c + 16*p + n*p + 16 + (4 if backend == 'cython' else 0)) + 16 if backend != 'numpy' else 0
    estimate = Trajectory(arrays['labels']).estimated_bytes(count, c, native)
    return dict(nodes=n, features=case.features, classes=c, particles=p,
        graph_shape=list(arrays['neighbors'].shape), degree_min=int(arrays['degrees'].min()),
        degree_max=int(arrays['degrees'].max()), directed_edges=int(arrays['degrees'].sum()),
        expected_observations=count, native_snapshot_bytes=native, estimated_peak_bytes=estimate,
        fixture_array_bytes=sum(a.nbytes for a in arrays.values()), fits=estimate <= MAX_BYTES and count <= MAX_OBSERVATIONS)


def timing_projection(summary, spec, fixtures):
    if summary is None:
        return dict(available=False, reason='synthetic validation; no historical timing inputs')
    summary = Path(summary)
    rows = json.loads(summary.read_text(encoding='utf-8'))
    estimates = []
    for task in spec['tasks']:
        selected = [r for r in rows if r['case_id'] == task['case_id'] and r['backend'] == task['backend']]
        if len(selected) != 1:
            raise CheckpointError('missing/duplicate historical timing group')
        row = selected[0]
        if (row['fixture_sha256'] != fixtures[task['case_id']]['fixture_sha256'] or row['mode'] != 'fixed'
                or row['max_iter'] != 10000 or row['repetitions'] != 6
                or row['effective_iterations']['minimum'] != 10000):
            raise CheckpointError('timing reference fixture/protocol mismatch')
        median = row['propagation_s_per_iteration']['median'] * spec['parameters']['max_iter']
        maximum = row['propagation_s_per_iteration']['maximum'] * spec['parameters']['max_iter']
        if not all(math.isfinite(v) and v > 0 for v in (median, maximum)) or maximum < median:
            raise CheckpointError('invalid historical timing rate')
        estimates.append(dict(task_id=task['task_id'], historical_rate_projection_s=median,
            historical_max_rate_projection_s=maximum, planning_s=3*maximum + 10))
    return dict(available=True, reference=str(summary.resolve()), reference_sha256=file_hash(summary),
        rows=estimates, historical_rate_total_s=sum(r['historical_rate_projection_s'] for r in estimates),
        planning_total_s=60 + sum(r['planning_s'] for r in estimates), startup_allowance_s=60,
        method='3x historical maximum 10000-step rate at 100000 steps + 10s/run + 60s/startup',
        limitations='Planning scenario, not a measured pilot or guaranteed bound. Load/state/JIT/export/audit, machine and version changes may exceed allowances. Each additional process may repeat JIT/startup cost.')


def preflight(output, *, synthetic=False, iterations=25, fixture_cache=None, historical_cache=None, timing_summary=None):
    output = output_path(output)
    spec = specification(synthetic, iterations)
    if not synthetic and timing_summary is None:
        raise ValueError('real dry-run requires --timing-summary; no new historical calibration fits')
    identity = execution_identity()
    output.mkdir(parents=True, exist_ok=False)
    (output/'fixtures').mkdir()
    preparer, fixtures, rows = FixturePreparer(), {}, []
    started = time.perf_counter()
    for description in spec['cases']:
        case = Case(**description)
        if fixture_cache is not None:
            arrays, info = load_fixture(Path(fixture_cache)/case.case_id, case)
        else:
            arrays, info = preparer.prepare(case, historical_cache=historical_cache)
            validate_fixture(case, arrays, info)
        save_fixture(output/'fixtures'/case.case_id, arrays, info)
        fixtures[case.case_id] = dict(fixture_sha256=info['fixture_sha256'],
            files=file_inventory(output/'fixtures'/case.case_id),
            frozen_checks='synthetic' if synthetic else 'raw, processed, ordered graph and label seed verified',
            array_hashes=info['hashes'], class_mapping=np.unique(arrays['truth']).tolist())
        for task in (t for t in spec['tasks'] if t['case_id'] == case.case_id):
            rows.append(dict(task, **resources(case, task['backend'], spec, arrays)))
    report = dict(schema=1, specification=spec, identity=identity, fixtures=fixtures, resources=rows,
        preparation_s=time.perf_counter()-started, run_count=len(rows),
        total_iteration_budget=len(rows)*spec['parameters']['max_iter'],
        total_particle_updates=sum(r['particles']*spec['parameters']['max_iter'] for r in rows),
        observation_budget_fits=all(r['fits'] for r in rows),
        smaller_schedule_suggestion=None if all(r['fits'] for r in rows) else [0, 10000, 100000],
        wall_clock=timing_projection(timing_summary, spec, fixtures),
        execution_started=False, preparation_mode='verified fixture cache' if fixture_cache else 'cache-only frozen preparation',
        storage_note='64 MiB per-run estimated diagnostic storage, not total process RSS; libraries/fixtures/JIT/audit are additional; completed bundles accumulate on disk (up to 768 MiB for 12 runs at the export limit).')
    durable_file(output/'report.json', canonical(dict(schema=1, payload=report, sha256=digest(report))) + b'\n')
    return report


def verify_report(directory):
    directory = Path(directory)
    report = read_record(directory/'report.json')
    validate_spec(report['specification'])
    if report['identity'] != execution_identity():
        raise CheckpointError('source/environment/binary differs from preflight; regenerate report')
    expected_rows = []
    for description in report['specification']['cases']:
        case = Case(**description)
        arrays, info = load_fixture(directory/'fixtures'/case.case_id, case, report['fixtures'][case.case_id]['files'])
        entry = report['fixtures'][case.case_id]
        if (info['fixture_sha256'] != entry['fixture_sha256'] or info['hashes'] != entry['array_hashes']
                or entry['class_mapping'] != np.unique(arrays['truth']).tolist()):
            raise CheckpointError('preflight fixture identity mismatch')
        for task in (t for t in report['specification']['tasks'] if t['case_id'] == case.case_id):
            expected_rows.append(dict(task, **resources(case, task['backend'], report['specification'], arrays)))
    if report['resources'] != expected_rows or not all(r['fits'] for r in expected_rows) or not report['observation_budget_fits']:
        raise CheckpointError('preflight observation budget mismatch')
    spec = report['specification']
    if (report['run_count'] != 12 or report['total_iteration_budget'] != 12*spec['parameters']['max_iter']
            or report['total_particle_updates'] != sum(r['particles']*spec['parameters']['max_iter'] for r in expected_rows)
            or report['execution_started'] is not False or (not spec['synthetic'] and not report['wall_clock']['available'])):
        raise CheckpointError('preflight totals/gate mismatch')
    return report


class Pilot:
    def __init__(self, root):
        self.store = Store(output_path(root))
        # Identify the store READ-ONLY before acquiring/writing any lock. Never
        # touch an older benchmark campaign accidentally passed as --root.
        if self.store.plan().get('pilot_kind') != 'convergence_trajectory_pilot':
            raise CheckpointError('not a Stage B pilot store')

    @classmethod
    def create(cls, root, preflight_directory):
        report = verify_report(preflight_directory)
        plan = dict(pilot_kind='convergence_trajectory_pilot', campaign_id=uuid.uuid4().hex, tasks=report['specification']['tasks'],
                    report=report, report_sha256=digest(report), preflight=str(Path(preflight_directory).resolve()))
        Store.create(output_path(root), plan)
        return cls(root)

    def context(self):
        plan = self.store.plan()
        report = verify_report(plan['preflight'])
        if report != plan['report'] or plan['report_sha256'] != digest(report) or plan['tasks'] != report['specification']['tasks']:
            raise CheckpointError('pilot/preflight binding changed')
        return plan, report

    def validate_bundle(self, path, task, report):
        if set(file_inventory(path)) != {'trajectory.json', 'trajectory.csv', 'manifest.json'}:
            raise CheckpointError('invalid trajectory file inventory')
        payload = read_output(path)
        m, spec = payload['metadata'], report['specification']
        case = next(Case(**d) for d in spec['cases'] if Case(**d).case_id == task['case_id'])
        fixture = report['fixtures'][case.case_id]
        hashes = {k: fixture['array_hashes'][k] for k in ('neighbors', 'degrees', 'labels', 'truth')}
        if (m['dataset'] != case.name or m['label_seed'] != case.label_seed or m['backend'] != task['backend']
                or m['parameters'] != spec['parameters'] or m['milestones'] != spec['milestones'] or m['include_checks']
                or m['fixture_hashes'] != hashes or m['propagation_seed'] != spec['propagation_seeds'][task['backend']]
                or m['budget']['max_observations'] != MAX_OBSERVATIONS or m['budget']['max_bytes'] != MAX_BYTES
                or m['nodes'] != case.nodes or m['classes'] != case.classes or m['particles'] != case.particles
                or m['k_nn'] != case.k_nn or m['class_index_to_label'] != fixture['class_mapping']
                or m['source_commit'] != report['identity']['source_commit']
                or m['source_hashes'] != report['identity']['source_hashes']
                or any(m[k] != report['identity'][k] for k in ('python', 'platform', 'dependencies', 'protocol_sha256'))):
            raise CheckpointError('trajectory differs from immutable pilot task')
        return payload

    def audit(self):
        plan, report = self.context()
        tasks = {t['task_id']: t for t in plan['tasks']}
        starts, last, retries, completed = {}, {}, set(), set()
        for event in self.store.events(force=True):
            kind = event['kind']
            if kind == 'lock_recovered':
                continue
            tid, aid = event.get('task_id'), event.get('attempt_id')
            if tid not in tasks or not isinstance(aid, str) or len(aid) != 32 or any(c not in '0123456789abcdef' for c in aid):
                raise CheckpointError('unknown task/attempt in pilot ledger')
            if kind == 'pilot_started':
                if aid in starts or tid in completed or (tid in last and tid not in retries):
                    raise CheckpointError('duplicate/unapproved pilot attempt')
                starts[aid], last[tid] = tid, aid
                retries.discard(tid)
            elif kind == 'pilot_retry':
                if last.get(tid) != aid or tid in completed or tid in retries or not event.get('reason', '').strip():
                    raise CheckpointError('invalid retry authorization')
                if file_inventory(self.store.root/'sessions'/aid) != event['preserved']:
                    raise CheckpointError('preserved interrupted attempt changed')
                retries.add(tid)
            elif kind == 'pilot_completed':
                if last.get(tid) != aid or tid in completed:
                    raise CheckpointError('invalid duplicate completion')
                completed.add(tid)
            else:
                raise CheckpointError('unknown pilot event')
        for path in (self.store.root/'sessions').iterdir():
            if path.name not in starts or path.is_symlink() or not path.is_dir():
                raise CheckpointError('unregistered attempt directory')
            if any(p.name != 'trajectory' or p.is_symlink() or not p.is_dir() for p in path.iterdir()):
                raise CheckpointError('unknown attempt artifact')
        result_files = {p.name for p in (self.store.root/'results').iterdir()}
        if result_files - {tid+'.json' for tid in tasks}:
            raise CheckpointError('unknown result file')
        done, recoverable, ambiguous = [], [], []
        for tid, task in tasks.items():
            result_path = self.store.root/'results'/f'{tid}.json'
            if tid not in last:
                if result_path.exists():
                    raise CheckpointError('result without start')
                continue
            aid = last[tid]
            bundle = self.store.root/'sessions'/aid/'trajectory'
            if result_path.exists():
                result = read_record(result_path)
                if result != dict(task_id=tid, attempt_id=aid, plan_sha256=digest(plan), files=file_inventory(bundle)):
                    raise CheckpointError('committed result checksum/identity mismatch')
                self.validate_bundle(bundle, task, report)
                if tid in retries:
                    raise CheckpointError('committed task cannot be retried')
                done.append(tid)
            elif (bundle/'manifest.json').exists():
                self.validate_bundle(bundle, task, report)  # corruption must never cause an automatic rerun
                if tid in retries:
                    raise CheckpointError('valid completed bundle has retry authorization')
                recoverable.append(tid)
            elif tid not in retries:
                ambiguous.append(tid)
            if tid in completed and tid not in done:
                raise CheckpointError('completion missing its committed result')
        return dict(plan=plan, report=report, done=done, recoverable=recoverable,
                    ambiguous=ambiguous, retries=sorted(retries), last=last, completed=sorted(completed))

    def commit_bundle(self, state, tid, fault=lambda point: None):
        aid = state['last'][tid]
        bundle = self.store.root/'sessions'/aid/'trajectory'
        path = self.store.root/'results'/f'{tid}.json'
        if not path.exists():
            self.store.write(path, dict(task_id=tid, attempt_id=aid, plan_sha256=digest(state['plan']), files=file_inventory(bundle)))
            fault('result_published')
        if tid not in state['completed']:
            self.store.event('pilot_completed', task_id=tid, attempt_id=aid)

    def execute(self, *, reviewed_preflight=None, max_runs=12, retry_task=None, reason=None, fault=lambda point: None):
        if type(max_runs) is not int or not 0 <= max_runs <= 12:
            raise ValueError('max_runs must be 0..12')
        with self.store.lock():
            state = self.audit()
            if not state['report']['specification']['synthetic'] and reviewed_preflight != state['plan']['report_sha256']:
                raise CheckpointError('real execution requires the exact reviewed preflight SHA; review before running')
            for tid in state['done'] + state['recoverable']:
                self.commit_bundle(state, tid)  # finish publication without ever refitting
            if retry_task is not None:
                if retry_task not in state['ambiguous'] or not reason or not reason.strip():
                    raise CheckpointError('retry requires an incomplete attempt and explicit reason')
                self.store.event('pilot_retry', task_id=retry_task, attempt_id=state['last'][retry_task], reason=reason,
                    preserved=file_inventory(self.store.root/'sessions'/state['last'][retry_task]))
            state = self.audit()
            if state['ambiguous']:
                raise CheckpointError('interrupted/partial fit requires explicit --retry-task and --reason: ' + ', '.join(state['ambiguous']))
            launched = 0
            spec = state['report']['specification']
            for task in spec['tasks']:
                tid = task['task_id']
                if tid in state['done'] or launched >= max_runs:
                    continue
                case = next(Case(**d) for d in spec['cases'] if Case(**d).case_id == task['case_id'])
                arrays, _ = load_fixture(Path(state['plan']['preflight'])/'fixtures'/case.case_id, case,
                                        state['report']['fixtures'][case.case_id]['files'])
                aid = uuid.uuid4().hex
                self.store.event('pilot_started', task_id=tid, attempt_id=aid)
                fault('started')
                directory = self.store.root/'sessions'/aid
                directory.mkdir()
                model = PCC(impl=task['backend'])
                model.set_graph(arrays['neighbors'], arrays['degrees'], k_nn=case.k_nn)
                predictions, trace, _ = observe_fit(model, arrays['labels'], truth=arrays['truth'],
                    propagation_seed=spec['propagation_seeds'][task['backend']],
                    milestones=spec['milestones'], include_checks=False,
                    max_observations=MAX_OBSERVATIONS, max_bytes=MAX_BYTES, **spec['parameters'])
                validate_final_state(model, predictions, arrays, trace)
                fault('fit_finished')
                write_output(directory/'trajectory', trace, metadata(dataset=case.name, label_seed=case.label_seed, collector=trace))
                self.validate_bundle(directory/'trajectory', task, state['report'])
                for path in (directory/'trajectory').iterdir():
                    with path.open('r+b') as stream:
                        os.fsync(stream.fileno())
                sync_directory(directory/'trajectory')
                fault('bundle_finished')
                current = self.audit()
                self.commit_bundle(current, tid, fault)
                fault('committed')
                launched += 1
                state = self.audit()
            return self.status()

    def status(self):
        state = self.audit()
        return dict(report_sha256=state['plan']['report_sha256'], complete=len(state['done']),
            total=12, recoverable=state['recoverable'], ambiguous=state['ambiguous'],
            pending=[t['task_id'] for t in state['plan']['tasks'] if t['task_id'] not in state['done']])


def validate_final_state(model, predictions, arrays, trace):
    dom, part = model.node.dominance, model.part
    if (not np.isfinite(dom).all() or np.any(dom < -1e-12) or np.any(dom > 1+1e-12)
            or not np.isfinite(part.strength).all() or np.any(part.strength < -1e-12)
            or np.any(part.strength > 1+1e-12) or not np.isfinite(model.owndeg).all()
            or np.any(part.curnode < 0) or np.any(part.curnode >= len(predictions))
            or part.dist_table.dtype != np.uint8):
        raise CheckpointError('invalid final dominance/particle state')
    np.testing.assert_allclose(dom.sum(axis=1), 1., rtol=0., atol=1e-10)
    known = arrays['labels'] != -1
    np.testing.assert_array_equal(predictions[known], arrays['labels'][known])
    np.testing.assert_array_equal(dom[known], np.eye(model.c)[model.mapped_labels[known]])
    np.testing.assert_array_equal(part.dist_table[part.homenode, np.arange(part.amount)], 0)
    np.testing.assert_array_equal(predictions, model.unique_labels[np.array(trace.rows[-1]['assignments'])])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    dry = commands.add_parser('preflight')
    dry.add_argument('--output', type=Path, required=True)
    dry.add_argument('--fixture-cache', type=Path)
    dry.add_argument('--historical-cache', type=Path)
    dry.add_argument('--timing-summary', type=Path)
    dry.add_argument('--synthetic', action='store_true')
    dry.add_argument('--iterations', type=int, help='synthetic only; real cap is fixed at 100000')
    init = commands.add_parser('init')
    init.add_argument('--preflight', type=Path, required=True)
    init.add_argument('--root', type=Path, required=True)
    for name in ('run', 'status', 'recover-lock'):
        sub = commands.add_parser(name)
        sub.add_argument('--root', type=Path, required=True)
        if name == 'run':
            sub.add_argument('--reviewed-preflight')
            sub.add_argument('--max-runs', type=int, default=12)
            sub.add_argument('--retry-task')
            sub.add_argument('--reason')
        elif name == 'recover-lock':
            sub.add_argument('--token', required=True)
            sub.add_argument('--reason', required=True)
    args = parser.parse_args(argv)
    if args.command == 'preflight':
        if args.iterations is not None and not args.synthetic:
            parser.error('--iterations is only available for synthetic validation')
        report = preflight(args.output, synthetic=args.synthetic, iterations=args.iterations if args.iterations is not None else 25,
                           fixture_cache=args.fixture_cache, historical_cache=args.historical_cache,
                           timing_summary=args.timing_summary)
        result = dict(report=report, report_sha256=digest(report))
    elif args.command == 'init':
        result = Pilot.create(args.root, args.preflight).status()
    else:
        pilot = Pilot(output_path(args.root))
        if args.command == 'run':
            result = pilot.execute(reviewed_preflight=args.reviewed_preflight, max_runs=args.max_runs,
                                   retry_task=args.retry_task, reason=args.reason)
        elif args.command == 'recover-lock':
            pilot.store.recover_lock(args.token, args.reason)
            result = pilot.status()
        else:
            result = pilot.status()
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
