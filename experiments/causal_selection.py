"""G1-R1 experimental adapter. Never changes PCC or selects using hidden truth.

Real preflight does not fit. Run requires a separately approved payload SHA.
All observations use the existing continuous native propagation instrumentation.
"""
import argparse
from collections import Counter
from dataclasses import asdict
import hashlib
from itertools import combinations
import json
from pathlib import Path
import uuid

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components, shortest_path

from benchmarks.checkpoints import CheckpointError, Store, canonical, digest, durable_file, file_inventory, read_record
from benchmarks.fixtures import Case, array_hash
from . import convergence_pilot as b
from . import convergence_stage_c as c
from . import mechanistic_pilot as m
from .convergence_trajectories import metadata, observe_fit
from .mechanistic_snapshots import SnapshotConfig, read_snapshots, write_snapshots, estimates, check_budget

SCIENCE_SHA = '4057a6a1cec696a663707ae2db7160b1f63ba7d3a95f5a256f02cbe06a677be6'
SCIENCE_FILE = b.ROOT/'docs/causal_selection_scientific.json'
KIND = 'causal-selection-G1-R1'


def science():
    result = read_record(SCIENCE_FILE)
    if digest(result) != SCIENCE_SHA or digest(result['catalog']) != result['catalog_sha256']:
        raise CheckpointError('frozen causal science/catalog differs')
    return result


def seal(path, value):
    durable_file(Path(path), canonical(dict(schema=1, payload=value, sha256=digest(value)))+b'\n')


def verify_inventory(root, expected):
    for relative, expected_hash in expected.items():
        path = Path(root)/relative
        if path.is_symlink() or hashlib.sha256(path.read_bytes()).hexdigest() != expected_hash:
            raise CheckpointError('historical scientific source changed: '+str(path))


def verify_sources():
    """All old sealed documents/certificates and frozen fixture bytes, read-only."""
    result = {}
    for name, entry in science()['source_seals'].items():
        root = b.ROOT/name
        expected = dict(entry['files'], **{'manifest.json': entry['manifest_sha256']})
        verify_inventory(root, expected)
        result[str(root)] = expected
    old = json.loads((b.ROOT/'.cache/convergence-g0/input_audit.json').read_text(encoding='utf-8'))
    for name, expected in old['source_fixtures'].items():
        root = b.ROOT/name
        if file_inventory(root) != expected:
            raise CheckpointError('frozen historical fixture changed')
        result[str(root)] = expected
    return result


def candidate_registry(dataset):
    return {int(r['node_id']): int(r['known_class']) for r in science()['candidates'] if r['dataset'] == dataset}


def intervention(nodes, known, selected):
    """Only eligible known labels are accessible to the selector; no truth argument."""
    selected = tuple(map(int, selected))
    if len(set(selected)) != len(selected) or any(v not in known or not 0 <= v < nodes for v in selected):
        raise CheckpointError('invalid/unknown candidate selection')
    if Counter(known[v] for v in selected) != {k: 3 for k in set(known.values())}:
        raise CheckpointError('three labels per known class required')
    labels = np.full(nodes, -1, dtype=np.int64)
    for v in selected:
        labels[v] = known[v]
    evaluation = np.ones(nodes, dtype=bool)
    evaluation[list(known)] = False
    return labels, evaluation


def geometry(neighbors, degrees, known):
    """Inputs deliberately exclude truth, predictions and all PCC state."""
    n = len(degrees); ids = sorted(known)
    # Three dense candidate-to-node work arrays plus sparse graph workspace.
    # Deliberately conservative, independent of the per-fit snapshot budget.
    if 3*len(ids)*n*8+int(np.sum(degrees))*32 > 128*1024**2:
        raise CheckpointError('geometric working estimate exceeds 128MiB')
    rows = np.repeat(np.arange(n), degrees)
    cols = np.concatenate([r[:d] for r, d in zip(neighbors, degrees)])
    graph = csr_matrix((np.ones(len(cols)), (rows, cols)), shape=(n, n))
    if (graph != graph.T).nnz:
        raise CheckpointError('expected frozen undirected graph')
    _, components = connected_components(graph, directed=False)
    distances = shortest_path(graph, directed=False, unweighted=True, indices=ids)
    inv = np.divide(1., distances, out=np.zeros_like(distances), where=(distances > 0) & np.isfinite(distances))
    harmonic = inv.sum(axis=1)/(n-1)
    bins = np.searchsorted(np.unique(np.quantile(degrees, [.2, .4, .6, .8])), degrees, side='right')
    return dict(ids=ids, index={v: i for i, v in enumerate(ids)}, distances=distances,
                harmonic=harmonic, bins=bins, components=components)


def triple(t, g):
    t = tuple(sorted(map(int, t)))
    return dict(nodes=t, signature=(tuple(sorted(g['components'][list(t)].tolist())),
        tuple(sorted(g['bins'][list(t)].tolist()))), harmonic=float(np.mean([g['harmonic'][g['index'][v]] for v in t])),
        numerator=sum(int(min(g['distances'][g['index'][a], z], 64)) for a, z in combinations(t, 2)))


def regenerate_pageblocks(known, g):
    """Exact approved DP sampler, not an optimization of PCC outcomes."""
    classes = sorted(set(known.values())); used = set(); result = []
    for block in range(10):
        pool = np.random.default_rng(np.random.SeedSequence([1, 1000+block]))
        options = []
        for k in classes:
            ids = np.array(sorted(v for v in known if known[v] == k))
            q = np.sort(pool.choice(ids, 6, replace=False))
            ts = [triple(t, g) for t in combinations(q, 3)]
            sd = np.std([g['harmonic'][g['index'][int(v)]] for v in ids])
            options.append([(a, z, z['numerator']-a['numerator']) for a in ts for z in ts
                if a['signature'] == z['signature'] and abs(a['harmonic']-z['harmonic']) <= .25*sd])
        suffix = [None]*(len(options)+1); suffix[-1] = {0: 1}
        for i in reversed(range(len(options))):
            count = Counter()
            for _, _, delta in options[i]:
                for total, weight in suffix[i+1].items():
                    count[total+delta] += weight
            suffix[i] = dict(count)
        rng = np.random.default_rng(np.random.SeedSequence([1, 41000+block]))
        def draw(limit):
            if limit <= 0:
                raise CheckpointError('no eligible structured completion')
            bits = limit.bit_length()
            while True:
                value = int.from_bytes(rng.bytes((bits+7)//8), 'little') & ((1 << bits)-1)
                if value < limit:
                    return value
        for attempt in range(4096):
            left, right, accumulated = [], [], 0
            for i, choices in enumerate(options):
                weights = [sum(w for s, w in suffix[i+1].items() if accumulated+d+s >= 3*len(classes)) for _, _, d in choices]
                ticket = draw(sum(weights))
                for (a, z, delta), weight in zip(choices, weights):
                    if ticket < weight:
                        break
                    ticket -= weight
                left.extend(a['nodes']); right.extend(z['nodes']); accumulated += delta
            pair = (tuple(sorted(left)), tuple(sorted(right)))
            if not any(x in used for x in pair):
                used.update(pair); result.append(pair); break
        else:
            raise CheckpointError('structured attempt cap exceeded')
    return result


def audit_catalog(dataset, neighbors, degrees):
    known = candidate_registry(dataset); g = geometry(neighbors, degrees, known)
    entries = [r for r in science()['catalog'] if r['dataset'] == dataset]
    pairs = []
    for block in sorted({int(r['block']) for r in entries}):
        pair = [next(r for r in entries if int(r['block']) == block and r['arm'] == arm) for arm in ('compact', 'dispersed')]
        ids = [tuple(map(int, r['selected_known_node_ids'].split(';'))) for r in pair]
        nums = []
        for selected in ids:
            intervention(len(degrees), known, selected)
        for k in sorted(set(known.values())):
            a, z = [triple([v for v in selected if known[v] == k], g) for selected in ids]
            sd = np.std([g['harmonic'][g['index'][v]] for v in known if known[v] == k])
            if a['signature'] != z['signature'] or abs(a['harmonic']-z['harmonic']) > .25*sd:
                raise CheckpointError('causal matching fails')
            nums.append(z['numerator']-a['numerator'])
        if sum(nums) < 3*len(set(known.values())):
            raise CheckpointError('causal contrast below one hop')
        pairs.append(tuple(ids))
    if dataset == 'pageblocks' and regenerate_pageblocks(known, g) != pairs:
        raise CheckpointError('structured regeneration differs from approved catalog')
    return g


def identity():
    result = m.identity()
    result['adapter_sha256'] = hashlib.sha256(Path(__file__).read_bytes().replace(b'\r\n', b'\n')).hexdigest()
    return result


def specification(synthetic=False):
    s = science(); spec = b.specification(synthetic, 3)
    spec.update(protocol=KIND, cases=[], snapshot_config=asdict(SnapshotConfig()), synthetic=synthetic)
    spec['tasks'] = [dict(task_id=f't{3*i+j:04d}', dataset=r['dataset'], block=int(r['block']), arm=r['arm'],
        slate_id=r['slate_id'], selected=list(map(int, r['selected_known_node_ids'].split(';'))), backend=backend,
        dataset_annotation=f"{r['dataset']}:block{r['block']}:{r['arm']}:{r['slate_id']}")
        for i, r in enumerate(s['catalog']) for j, backend in enumerate(b.BACKENDS)]
    ordered = []
    for ds in ('pageblocks', 'spambase'):
        for block in sorted({t['block'] for t in spec['tasks'] if t['dataset'] == ds}):
            group = [t for t in spec['tasks'] if (t['dataset'],t['block']) == (ds,block)]
            rng = np.random.default_rng(np.random.SeedSequence([1 if ds == 'pageblocks' else 2,30000+block]))
            ordered.extend(group[int(i)] for i in rng.permutation(len(group)))
    spec['tasks'] = ordered
    return spec


def preflight(output, fixture_cache=None):
    refs = verify_sources(); spec = specification(); destination = c.safe_destination(output, list(refs)+[fixture_cache])
    if b.source_metadata()['source_dirty']:
        raise CheckpointError('real preflight requires clean committed sources')
    destination.mkdir(parents=True, exist_ok=False); (destination/'fixtures').mkdir()
    fixtures = {}; resources = []; masks = {}
    for ds in ('pageblocks', 'spambase'):
        case = Case(ds, b.FROZEN[ds]['nodes'], b.FROZEN[ds]['processed_features'], b.FROZEN[ds]['classes'], label_seed=10)
        source = Path(fixture_cache)/case.case_id
        if str(source.resolve()) not in refs:
            raise CheckpointError('fixture not bound by G0 audit')
        arrays, _ = b.load_fixture(source, case, refs[str(source.resolve())])
        g = audit_catalog(ds, arrays['neighbors'], arrays['degrees'])
        known = candidate_registry(ds)
        _, mask = intervention(case.nodes, known, spec['tasks'][next(i for i,t in enumerate(spec['tasks']) if t['dataset']==ds)]['selected'])
        expected = next(r for r in science()['masks'] if r['dataset'] == ds)
        if array_hash(np.flatnonzero(mask)) != expected['evaluation_mask_node_ids_sha256']:
            raise CheckpointError('common evaluation mask differs')
        # Truth is evaluator-only. Neither audit_catalog nor intervention receives it.
        supports = {str(k): int(np.sum(arrays['truth'][mask] == k)) for k in sorted(set(known.values()))}
        if supports != json.loads(expected['support_by_class']) or min(supports.values()) < 10:
            raise CheckpointError('evaluation support differs or primary BA not evaluable')
        data = {k: arrays[k] for k in ('neighbors', 'degrees', 'truth', 'source_indices')}
        data.update(evaluation=mask, components=g['components'])
        path = destination/'fixtures'/f'{ds}.npz'
        with path.open('xb') as f:
            np.savez_compressed(f, **data)
        fixtures[ds] = dict(file=path.name, sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            hashes={k: array_hash(v) for k,v in data.items()}, nodes=case.nodes, classes=case.classes, k_nn=10,
            source_fixture=str(source.resolve()), source_files=refs[str(source.resolve())])
        masks[ds] = dict(support=supports, evaluation_nodes=int(mask.sum()), hash=array_hash(np.flatnonzero(mask)))
        for task in spec['tasks']:
            if task['dataset'] == ds:
                estimate = estimates(case.nodes, case.classes, case.particles, 10, task['backend'])
                check_budget(SnapshotConfig(**spec['snapshot_config']), estimate)
                resources.append(dict(task_id=task['task_id'], **estimate))
    report = dict(scientific_sha256=SCIENCE_SHA, catalog_sha256=science()['catalog_sha256'], specification=spec,
        identity=identity(), references=refs, fixtures=fixtures, masks=masks, resources=resources, execution_started=False,
        run_count=102, total_iteration_budget=10200000, observations=1020,
        estimated_export_bytes=sum(x['estimated_export_bytes'] for x in resources),
        peak_diagnostic_memory_bytes=max(x['estimated_memory_bytes'] for x in resources),
        aggregate_export_limit_bytes=102*64*1024**2, planning_seconds=1505.0955686499365, planning_3x_seconds=4515.286705949809,
        limitations='Finite catalog; no IID block/backend inference. Whole-fit recovery, no intra-fit checkpoints. RAM excludes process/JIT. Timing mixed historical operations, not a bound. Label seed annotation is block index, not historical label generation.')
    seal(destination/'report.json', report)
    return report


def load_arrays(directory, report, dataset):
    entry = report['fixtures'][dataset]; path = Path(directory)/'fixtures'/entry['file']
    if path.is_symlink() or hashlib.sha256(path.read_bytes()).hexdigest() != entry['sha256']:
        raise CheckpointError('causal fixture bytes differ')
    with np.load(path, allow_pickle=False) as z:
        arrays = {k:z[k] for k in z.files}
    if {k:array_hash(v) for k,v in arrays.items()} != entry['hashes']:
        raise CheckpointError('causal fixture arrays differ')
    return arrays


def verify_report(directory):
    report = read_record(Path(directory)/'report.json')
    if (report['scientific_sha256'] != SCIENCE_SHA or report['catalog_sha256'] != science()['catalog_sha256']
            or report['specification'] != specification() or report['identity'] != identity()
            or report['references'] != verify_sources() or report['execution_started'] is not False):
        raise CheckpointError('causal preflight science/source/environment differs')
    spec = report['specification']; count = len(spec['tasks'])
    if report['run_count'] != count or report['total_iteration_budget'] != count*spec['parameters']['max_iter'] or report['observations'] != count*len(spec['milestones']):
        raise CheckpointError('causal resource totals differ')
    resources = []
    for ds in report['fixtures']:
        arrays = load_arrays(directory, report, ds)
        mask = np.ones(len(arrays['degrees']), dtype=bool); mask[list(candidate_registry(ds))] = False
        if not np.array_equal(mask, arrays['evaluation']) or array_hash(np.flatnonzero(mask)) != report['masks'][ds]['hash']:
            raise CheckpointError('common causal mask differs')
        entry = report['fixtures'][ds]
        if not np.array_equal(arrays['source_indices'], np.arange(entry['nodes'])):
            raise CheckpointError('node/source identity differs')
        for task in spec['tasks']:
            if task['dataset'] == ds:
                value = estimates(entry['nodes'], entry['classes'], 3*entry['classes'], len(spec['milestones']), task['backend'])
                check_budget(SnapshotConfig(**spec['snapshot_config']), value)
                resources.append(dict(task_id=task['task_id'], **value))
    if (report['resources'] != resources or report['estimated_export_bytes'] != sum(x['estimated_export_bytes'] for x in resources)
            or report['peak_diagnostic_memory_bytes'] != max(x['estimated_memory_bytes'] for x in resources)
            or report['aggregate_export_limit_bytes'] != count*spec['snapshot_config']['max_export_bytes']):
        raise CheckpointError('causal resource estimates/budgets differ')
    return report


def evaluate(assignments, truth, evaluation, components, known, classes):
    """Retrospective evaluator; class-absent strata explicitly have NA macro score."""
    covered = np.isin(components, np.unique(components[list(known)])); result = {}
    for name, mask in [('all', evaluation), ('with_candidates', evaluation & covered), ('without_candidates', evaluation & ~covered)]:
        confusion = np.zeros((classes, classes), dtype=np.int64)
        np.add.at(confusion, (truth[mask], np.asarray(assignments)[mask]), 1)
        support = confusion.sum(axis=1); correct = np.diag(confusion)
        recall = [float(a/n) if n else None for a,n in zip(correct,support)]
        result[name] = dict(nodes=int(mask.sum()), confusion=confusion.tolist(), support=support.tolist(),
            correct=correct.tolist(), recall=recall, accuracy=float(correct.sum()/mask.sum()) if mask.any() else None,
            balanced_accuracy=float(np.mean(recall)) if np.all(support > 0) else None)
    return result


class CausalPilot(b.Pilot):
    def __init__(self, root):
        self.store = Store(b.output_path(root))
        if self.store.plan().get('pilot_kind') != KIND:
            raise CheckpointError('not a causal G1-R1 store')

    @classmethod
    def create(cls, root, preflight_directory):
        report = verify_report(preflight_directory)
        c.safe_destination(root, [preflight_directory]+list(report['references']))
        Store.create(b.output_path(root), dict(pilot_kind=KIND, campaign_id=uuid.uuid4().hex,
            tasks=report['specification']['tasks'], report=report, report_sha256=digest(report), preflight=str(Path(preflight_directory).resolve())))
        return cls(root)

    def context(self):
        plan = self.store.plan(); report = verify_report(plan['preflight'])
        if plan['report'] != report or plan['report_sha256'] != digest(report) or plan['tasks'] != report['specification']['tasks']:
            raise CheckpointError('causal plan differs from preflight')
        return plan, report

    def validate_bundle(self, path, task, report):
        with read_snapshots(path) as bundle:
            payload = bundle['summary']; meta = payload['metadata']; spec = report['specification']
            arrays = load_arrays(self.store.plan()['preflight'], report, task['dataset'])
            labels, _ = intervention(len(arrays['degrees']), candidate_registry(task['dataset']), task['selected'])
            expected = {k:array_hash(v) for k,v in dict(neighbors=arrays['neighbors'], degrees=arrays['degrees'], labels=labels, truth=arrays['truth']).items()}
            if (meta['fixture_hashes'] != expected or meta['dataset'] != task['dataset_annotation'] or meta['label_seed'] != task['block']
                    or meta['backend'] != task['backend'] or meta['parameters'] != spec['parameters']
                    or meta['milestones'] != spec['milestones'] or meta['include_checks']
                    or meta['propagation_seed'] != spec['propagation_seeds'][task['backend']]
                    or meta['k_nn'] != 10 or meta['nodes'] != len(labels)
                    or meta['classes'] != len(set(candidate_registry(task['dataset']).values()))
                    or meta['particles'] != int(np.count_nonzero(labels != -1))
                    or meta['class_index_to_label'] != sorted(set(candidate_registry(task['dataset']).values()))
                    or meta['budget']['max_observations'] != spec['max_observations']
                    or meta['budget']['max_bytes'] != spec['max_bytes']
                    or meta['source_commit'] != report['identity']['source_commit']
                    or meta['source_hashes'] != report['identity']['source_hashes']
                    or any(meta[k] != report['identity'][k] for k in ('python','platform','dependencies','protocol_sha256'))
                    or bundle['metadata']['source_hashes']['experiments/mechanistic_snapshots.py'] != report['identity']['mechanistic_sources']['experiments/mechanistic_snapshots.py']
                    or payload['observations'][-1]['iteration'] != spec['parameters']['max_iter']
                    or bundle['metadata']['limits'] != spec['snapshot_config']):
                raise CheckpointError('causal bundle differs from intervention task')
        return payload

    def execute(self, *, reviewed_preflight=None, max_runs=1, retry_task=None, reason=None, fault=lambda point: None):
        if type(max_runs) is not int or not 0 <= max_runs <= 102:
            raise ValueError('max_runs must be 0..102')
        with self.store.lock():
            state = self.audit()
            if reviewed_preflight != state['plan']['report_sha256']:
                raise CheckpointError('exact separately approved causal preflight SHA required')
            for tid in state['done']+state['recoverable']:
                self.commit_bundle(state, tid)
            if retry_task is not None:
                if retry_task not in state['ambiguous'] or not reason or not reason.strip():
                    raise CheckpointError('explicit retry of incomplete task with reason required')
                self.store.event('pilot_retry', task_id=retry_task, attempt_id=state['last'][retry_task], reason=reason,
                    preserved=file_inventory(self.store.root/'sessions'/state['last'][retry_task]))
            state = self.audit()
            if state['ambiguous']:
                raise CheckpointError('incomplete task requires explicit retry; no automatic refit')
            launched = 0; spec = state['report']['specification']
            for task in spec['tasks']:
                if task['task_id'] in state['done'] or launched >= max_runs:
                    continue
                arrays = load_arrays(state['plan']['preflight'], state['report'], task['dataset'])
                labels, _ = intervention(len(arrays['degrees']), candidate_registry(task['dataset']), task['selected'])
                aid = uuid.uuid4().hex; tid = task['task_id']
                self.store.event('pilot_started', task_id=tid, attempt_id=aid); fault('started')
                directory = self.store.root/'sessions'/aid; directory.mkdir()
                model = m.PCC(impl=task['backend']); model.set_graph(arrays['neighbors'], arrays['degrees'], k_nn=10)
                _, trace, _ = observe_fit(model, labels, truth=arrays['truth'], milestones=spec['milestones'],
                    propagation_seed=spec['propagation_seeds'][task['backend']], max_observations=spec['max_observations'],
                    max_bytes=spec['max_bytes'], snapshot_config=SnapshotConfig(**spec['snapshot_config']), **spec['parameters'])
                fault('fit_finished')
                write_snapshots(directory/'trajectory', trace, metadata(dataset=task['dataset_annotation'], label_seed=task['block'], collector=trace))
                self.validate_bundle(directory/'trajectory', task, state['report']); fault('bundle_finished')
                self.commit_bundle(self.audit(), tid, fault); launched += 1; state = self.audit()
            return self.status()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__); subs = parser.add_subparsers(dest='command', required=True)
    dry = subs.add_parser('preflight'); dry.add_argument('--output', type=Path, required=True); dry.add_argument('--fixture-cache', type=Path, required=True)
    init = subs.add_parser('init'); init.add_argument('--root', type=Path, required=True); init.add_argument('--preflight', type=Path, required=True)
    for command in ('status','run','recover-lock'):
        sub = subs.add_parser(command); sub.add_argument('--root', type=Path, required=True)
        if command == 'run':
            sub.add_argument('--reviewed-preflight', required=True); sub.add_argument('--max-runs', type=int, default=1)
            sub.add_argument('--retry-task'); sub.add_argument('--reason')
        elif command == 'recover-lock':
            sub.add_argument('--token', required=True); sub.add_argument('--reason', required=True)
    args = vars(parser.parse_args(argv)); command = args.pop('command')
    if command == 'preflight':
        r = preflight(**args); result = dict(report_sha256=digest(r), run_count=r['run_count'])
    elif command == 'init':
        result = CausalPilot.create(args['root'], args['preflight']).status()
    else:
        pilot = CausalPilot(args.pop('root'))
        if command == 'run':
            result = pilot.execute(**args)
        elif command == 'recover-lock':
            pilot.store.recover_lock(args['token'], args['reason']); result = pilot.status()
        else:
            result = pilot.status()
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
