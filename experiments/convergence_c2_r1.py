"""C2-R1: approved science, opt-in rich export, whole-fit transactions.

Preflight never fits or downloads. Real execution requires a separately reviewed
execution-preflight SHA. The frozen scientific record is not execution approval.
"""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time
import uuid

from benchmarks.checkpoints import CheckpointError, Store, canonical, digest, durable_file, file_inventory, read_record
from benchmarks.fixtures import Case, FixturePreparer, file_hash, save_fixture
from . import convergence_pilot as b
from . import convergence_stage_c as c
from . import mechanistic_pilot as m
from .mechanistic_snapshots import SnapshotConfig

SCIENTIFIC_SHA = '955dca598c123d059269f18fbd7fe67f4d021b3bd211d2f1da1846557989c825'
SCIENTIFIC_RECORD = b.ROOT / 'docs/c2_r1_scientific.json'
KIND = 'convergence_c2_r1'
PREFIX_FIELDS = ('predictions', 'dominance', 'positions', 'strengths')


def science(path=SCIENTIFIC_RECORD):
    record = read_record(path)
    if digest(record) != SCIENTIFIC_SHA:
        raise CheckpointError('approved scientific protocol SHA differs')
    if record['scope'] != c.specification('c2'):
        raise CheckpointError('original C2 scope differs from approved science')
    return record


def specification(synthetic=False, iterations=7):
    approved = science()
    spec = c.specification('c2', synthetic, iterations)
    spec.update(protocol='synthetic-c2-r1-v1' if synthetic else 'historical-c2-r1-v1',
                snapshot_config=approved['extension']['snapshot_config'])
    return spec


def identity():
    result = m.identity()
    result['source_dirty'] = b.source_metadata()['source_dirty']
    result['c2_r1_sources'] = {
        name: hashlib.sha256((b.ROOT/name).read_bytes().replace(b'\r\n', b'\n')).hexdigest()
        for name in ('experiments/convergence_c2_r1.py', 'docs/convergence_c2_r1.md',
                     'docs/c2_r1_scientific.json', 'experiments/convergence_stage_c.py',
                     'benchmarks/checkpoints.py')}
    return result


def sealed_science(directory):
    """Validate the original local synthesis without changing or re-sealing it."""
    directory = Path(directory).resolve()
    manifest = json.loads((directory/'manifest.json').read_text(encoding='utf-8'))
    actual = file_inventory(directory)
    if not all(actual.get(k) == v for k, v in manifest['files'].items()):
        raise CheckpointError('scientific synthesis inventory differs')
    if science(directory/'protocol.json') != science():
        raise CheckpointError('local/committed scientific records differ')
    return {str(directory): {**manifest['files'], 'manifest.json': actual['manifest.json']}}


class FrozenMechanistic(m.MechanisticPilot):
    """Read-only old-campaign audit; old commit is provenance, never executable."""
    def context(self):
        plan = self.store.plan()
        report = read_record(Path(plan['preflight'])/'report.json')
        spec = report['specification']
        if (plan['report'] != report or plan['report_sha256'] != digest(report)
                or plan['tasks'] != spec['tasks'] or spec != m.specification()
                or spec['synthetic']):
            raise CheckpointError('frozen mechanistic binding differs')
        for desc in spec['cases']:
            case = Case(**desc)
            entry = report['fixtures'][case.case_id]
            _, info = b.load_fixture(Path(plan['preflight'])/'fixtures'/case.case_id, case, entry['files'])
            if info['fixture_sha256'] != entry['fixture_sha256'] or info['hashes'] != entry['array_hashes']:
                raise CheckpointError('frozen mechanistic fixture differs')
        return plan, report

    def execute(self, **kwargs):
        raise CheckpointError('frozen mechanistic audit cannot execute')


def dependencies(scientific_directory, stage_b, stage_c1, mechanistic):
    sealed = sealed_science(scientific_directory)
    original_audit = json.loads((Path(scientific_directory)/'audit.json').read_text(encoding='utf-8'))
    documents = {str((b.ROOT/name).resolve()): value for name, value in original_audit['report_files'].items()}
    if any(file_hash(name) != value for name, value in documents.items()):
        raise CheckpointError('scientific source report changed')
    totals = []
    for pilot, count, observations in ((c.FrozenStageB(stage_b), 12, 120),
                                      (m.FrozenC1(stage_c1), 48, 480),
                                      (FrozenMechanistic(mechanistic), 12, 192)):
        state = pilot.audit()
        if (len(state['done']) != count or state['recoverable'] or state['ambiguous']):
            raise CheckpointError('previous results incomplete or ambiguous')
        for root in (pilot.store.root, Path(state['plan']['preflight'])):
            sealed[str(root)] = file_inventory(root)
        totals.append(dict(root=str(pilot.store.root), complete=count, observations=observations,
                           report_sha256=state['plan']['report_sha256']))
    return dict(files=sealed, documents=documents, audits=totals)


def verify_dependencies(refs):
    for root, files in refs['files'].items():
        actual = file_inventory(Path(root))
        # Scientific manifests intentionally exclude __pycache__; validate all sealed files.
        comparable = {k: v for k, v in actual.items() if '__pycache__' not in Path(k).parts}
        expected = {k: v for k, v in files.items() if '__pycache__' not in Path(k).parts}
        if comparable != expected:
            raise CheckpointError('sealed scientific or experimental dependency changed')
    if any(file_hash(name) != value for name, value in refs['documents'].items()):
        raise CheckpointError('scientific source report changed')


def fixture_rows(directory, report):
    rows = []
    spec = report['specification']
    for desc in spec['cases']:
        case = Case(**desc)
        entry = report['fixtures'][case.case_id]
        arrays, info = b.load_fixture(Path(directory)/'fixtures'/case.case_id, case, entry['files'])
        if (entry['fixture_sha256'] != info['fixture_sha256'] or entry['array_hashes'] != info['hashes']
                or entry['class_mapping'] != sorted(set(arrays['truth'].tolist()))):
            raise CheckpointError('C2-R1 fixture provenance differs')
        for task in spec['tasks']:
            if task['case_id'] == case.case_id:
                rows.append(dict(task, **m.resource(case, task['backend'], spec),
                    graph_shape=list(arrays['neighbors'].shape), degree_max=int(arrays['degrees'].max()),
                    directed_edges=int(arrays['degrees'].sum())))
    return rows


def historical_report(roots, spec, fixtures):
    history = c.historical_check(*roots, spec, fixtures)
    fixed = [json.loads(line) for line in (Path(roots[0])/'measurements.jsonl').read_text(encoding='utf-8').splitlines() if line.strip()]
    history['prefixes'] = {t['task_id']: [dict(state_hashes=r['state_hashes'],
        accuracy_unlabeled=r['accuracy_unlabeled']) for r in fixed
        if (r['case_id'], r['backend']) == (t['case_id'], t['backend'])] for t in spec['tasks']}
    history['prefix_comparison_fields'] = list(PREFIX_FIELDS)
    history['prefix_limitation'] = 'Four common state fingerprints and retrospective accuracy; historical final owndeg is normalized after propagation, unlike milestone state. No new historical RNG audit.'
    return history


def validate_prefix(payload, task, history):
    row = next((r for r in payload['observations'] if r['iteration'] == 10000), None)
    if row is None:
        raise CheckpointError('missing historical 10000 prefix')
    predictions = b.np.asarray(payload['metadata']['class_index_to_label'], dtype=b.np.int64)[row['assignments']]
    hashes = dict(row['state_hashes'], predictions=b.array_hash(predictions))
    for old in history['prefixes'][task['task_id']]:
        if (any(hashes.get(k) != old['state_hashes'].get(k) or k not in hashes or k not in old['state_hashes'] for k in PREFIX_FIELDS)
                or row['accuracy_unlabeled_retrospective'] != old['accuracy_unlabeled']):
            raise CheckpointError('C2-R1 10000 prefix differs from PR16; preserve and investigate')


def preflight(output, *, synthetic=False, iterations=7, original_c2=None,
              scientific_directory=None, stage_b=None, stage_c1=None, mechanistic=None):
    spec = specification(synthetic, iterations)
    sources = [x for x in (original_c2, scientific_directory, stage_b, stage_c1, mechanistic) if x is not None]
    old = refs = history = None
    if not synthetic:
        if len(sources) != 5:
            raise ValueError('real preflight requires original C2, sealed science, B/C1 and mechanistic stores')
        old = read_record(Path(original_c2)/'report.json')
        if (digest(old) != science()['original_c2_preflight_sha256']
                or old['specification'] != c.specification('c2')):
            raise CheckpointError('original C2 preflight SHA/scope differs')
        c.check_fixtures(original_c2, old)
        refs = dependencies(scientific_directory, stage_b, stage_c1, mechanistic)
        refs['files'][str(Path(original_c2).resolve())] = file_inventory(Path(original_c2))
        sources += list(refs['files']) + old['historical']['roots']
        sources += [Path(name).parent for name in refs['documents']]
    output = c.safe_destination(output, sources)
    execution = identity()
    if not synthetic and execution['source_dirty']:
        raise CheckpointError('real preflight requires a clean committed checkout')
    output.mkdir(parents=True, exist_ok=False)
    (output/'fixtures').mkdir()
    started = time.perf_counter()
    fixtures = {}
    preparer = FixturePreparer()
    for desc in spec['cases']:
        case = Case(**desc)
        arrays, info = (preparer.prepare(case) if synthetic else
                       b.load_fixture(Path(original_c2)/'fixtures'/case.case_id, case, old['fixtures'][case.case_id]['files']))
        b.validate_fixture(case, arrays, info)
        save_fixture(output/'fixtures'/case.case_id, arrays, info)
        fixtures[case.case_id] = dict(fixture_sha256=info['fixture_sha256'],
            files=file_inventory(output/'fixtures'/case.case_id), array_hashes=info['hashes'],
            class_mapping=sorted(set(arrays['truth'].tolist())))
    if not synthetic:
        history = historical_report(old['historical']['roots'], spec, fixtures)
    report = dict(schema=1, specification=spec, identity=execution, fixtures=fixtures,
        scientific_sha256=SCIENTIFIC_SHA, scientific_protocol=science(), dependencies=refs,
        historical=history, references=None, execution_started=False, run_count=12,
        wall_clock=b.timing_projection(None if synthetic else old['wall_clock']['reference'], spec, fixtures),
        preparation_s=time.perf_counter()-started,
        limitations='Prospective limited replication, not blind external validation; two label seeds/dataset, dependent backends. No extra fits/milestones or stopping rules. Timing excludes measured rich-export overhead; diagnostic memory excludes process RSS/JIT. Whole-fit recovery only.')
    report['resources'] = rows = fixture_rows(output, report)
    report.update(total_iteration_budget=12*spec['parameters']['max_iter'],
        total_particle_updates=sum(r['particles']*spec['parameters']['max_iter'] for r in rows),
        total_snapshot_array_bytes=sum(r['snapshot_array_bytes'] for r in rows),
        estimated_export_bytes=sum(r['estimated_export_bytes'] for r in rows),
        maximum_estimated_memory_bytes=max(r['estimated_memory_bytes'] for r in rows),
        aggregate_export_limit_bytes=12*spec['snapshot_config']['max_export_bytes'])
    durable_file(output/'report.json', canonical(dict(schema=1, payload=report, sha256=digest(report)))+b'\n')
    return report


def verify_report(directory):
    report = read_record(Path(directory)/'report.json')
    spec = report['specification']
    if (spec != specification(spec['synthetic'], spec['parameters']['max_iter'])
            or report['scientific_sha256'] != SCIENTIFIC_SHA or report['scientific_protocol'] != science()
            or report['identity'] != identity() or report['execution_started'] is not False
            or report['references'] is not None):
        raise CheckpointError('C2-R1 scientific/execution identity or scope changed')
    rows = fixture_rows(directory, report)
    expected = dict(resources=rows, run_count=12,
        total_iteration_budget=12*spec['parameters']['max_iter'],
        total_particle_updates=sum(r['particles']*spec['parameters']['max_iter'] for r in rows),
        total_snapshot_array_bytes=sum(r['snapshot_array_bytes'] for r in rows),
        estimated_export_bytes=sum(r['estimated_export_bytes'] for r in rows),
        maximum_estimated_memory_bytes=max(r['estimated_memory_bytes'] for r in rows),
        aggregate_export_limit_bytes=12*spec['snapshot_config']['max_export_bytes'])
    if any(report[k] != v for k, v in expected.items()):
        raise CheckpointError('C2-R1 resource totals differ')
    if not spec['synthetic']:
        verify_dependencies(report['dependencies'])
        if not report['historical']['compatible'] or len(report['historical']['groups']) != 12:
            raise CheckpointError('historical compatibility missing')
        for name, value in report['historical']['files'].items():
            if file_hash(name) != value:
                raise CheckpointError('historical reference changed')
        if historical_report(report['historical']['roots'], spec, report['fixtures']) != report['historical']:
            raise CheckpointError('historical binding or prefix inventory differs')
    if b.timing_projection(report['wall_clock'].get('reference'), spec, report['fixtures']) != report['wall_clock']:
        raise CheckpointError('timing projection differs')
    return report


class C2R1(m.MechanisticPilot):
    """Reuse native observation, binary validation and durable transactions unchanged."""
    def __init__(self, root):
        self.store = Store(b.output_path(root))
        if self.store.plan().get('pilot_kind') != KIND:
            raise CheckpointError('not a C2-R1 store; original stores cannot be resumed here')

    @classmethod
    def create(cls, root, preflight_directory):
        report = verify_report(preflight_directory)
        sources = [preflight_directory]
        if report['dependencies']:
            sources += list(report['dependencies']['files']) + report['historical']['roots']
            sources += [Path(name).parent for name in report['dependencies']['documents']]
        c.safe_destination(root, sources)
        plan = dict(pilot_kind=KIND, campaign_id=uuid.uuid4().hex, tasks=report['specification']['tasks'],
            report=report, report_sha256=digest(report), preflight=str(Path(preflight_directory).resolve()))
        Store.create(b.output_path(root), plan)
        return cls(root)

    def context(self):
        plan = self.store.plan()
        report = verify_report(plan['preflight'])
        if (plan['report'] != report or plan['report_sha256'] != digest(report)
                or plan['tasks'] != report['specification']['tasks']):
            raise CheckpointError('C2-R1 plan/preflight binding differs')
        return plan, report

    def validate_bundle(self, path, task, report):
        payload = super().validate_bundle(path, task, report)
        if not report['specification']['synthetic']:
            validate_prefix(payload, task, report['historical'])
        return payload


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    dry = commands.add_parser('preflight')
    dry.add_argument('--output', type=Path, required=True)
    dry.add_argument('--synthetic', action='store_true')
    dry.add_argument('--iterations', type=int)
    for name in ('original-c2', 'scientific-directory', 'stage-b', 'stage-c1', 'mechanistic'):
        dry.add_argument('--'+name, type=Path)
    init = commands.add_parser('init')
    init.add_argument('--preflight', type=Path, required=True)
    init.add_argument('--root', type=Path, required=True)
    for name in ('run', 'status', 'recover-lock'):
        sub = commands.add_parser(name)
        sub.add_argument('--root', type=Path, required=True)
        if name == 'run':
            sub.add_argument('--reviewed-preflight')
            sub.add_argument('--max-runs', type=int, default=1)
            sub.add_argument('--retry-task')
            sub.add_argument('--reason')
        elif name == 'recover-lock':
            sub.add_argument('--token', required=True)
            sub.add_argument('--reason', required=True)
    args = vars(parser.parse_args(argv))
    command = args.pop('command')
    if command == 'preflight':
        if args['iterations'] is not None and not args['synthetic']:
            parser.error('--iterations is synthetic only')
        args['iterations'] = 7 if args['iterations'] is None else args['iterations']
        report = preflight(**args)
        result = dict(report_sha256=digest(report), report=report)
    elif command == 'init':
        result = C2R1.create(args['root'], args['preflight']).status()
    else:
        pilot = C2R1(args.pop('root'))
        if command == 'run':
            result = pilot.execute(**args)
        elif command == 'recover-lock':
            pilot.store.recover_lock(args['token'], args['reason'])
            result = pilot.status()
        else:
            result = pilot.status()
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
