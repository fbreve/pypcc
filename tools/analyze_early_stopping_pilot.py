"""Offline pilot audit and cost projections. Does not execute propagation."""
import argparse
from collections import defaultdict
import csv
import hashlib
import itertools
import json
from pathlib import Path
import statistics

from benchmarks.fixtures import array_hash
from benchmarks.runner import summarize
import numpy as np

DATASETS = ('wine', 'digits', 'germancredit', 'satimage')
BACKENDS = ('numpy', 'numba', 'cython')

def read(path):
    return json.loads(path.read_text(encoding='utf-8'))

def lines(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]

def analyze(root, baseline):
    if not __debug__:
        raise RuntimeError('Run audit without Python -O; assertions must be enabled')
    run = root / 'pilot'
    rows, warms = lines(run/'measurements.jsonl'), lines(run/'warmups.jsonl')
    cases = {f'{d}-labels{s}' for d in DATASETS for s in (10, 11)}
    assert read(run/'completion.json')['status'] == 'complete'
    expected = set(itertools.product(cases, BACKENDS, range(3)))
    assert len(rows) == 72 and {(r['case_id'], r['backend'], r['repeat']) for r in rows} == expected
    assert len(warms) == 48 and {(r['case_id'], r['backend'], r['call']) for r in warms} == set(itertools.product(cases, BACKENDS, ('first','warm')))
    manifest = read(run/'manifest.json')
    previous_manifest = read(baseline/'manifest.json')
    for name, digest in manifest['repository']['source_hashes'].items():
        if name.startswith('pcc') or name in ('benchmarks/fixtures.py', 'benchmarks/runner.py', 'benchmarks/historical.py'):
            assert digest == previous_manifest['repository']['source_hashes'][name]
    assert manifest['cython_extension']['sha256'] == previous_manifest['cython_extension']['sha256']
    assert not manifest['repository']['dirty']
    assert manifest['configuration']['parameters'] == dict(p_grd=.5,delta_v=.1,deltap=1.,dexp=2.,es_chk=2000)
    fixtures = {}
    for case in sorted(cases):
        meta = read(run/'fixtures'/case/'fixture.json')
        ref = read(baseline/'fixtures'/case/'fixture.json')
        assert meta['fixture_sha256'] == ref['fixture_sha256'] and meta['hashes'] == ref['hashes']
        assert meta['fixture_sha256'] == hashlib.sha256(json.dumps(meta['hashes'],sort_keys=True).encode()).hexdigest()
        assert meta['source']['label_fingerprint_matches']
        assert all(v['matches'] and v['actual'] == v['expected'] for v in meta['source']['frozen_fingerprints'].values())
        with np.load(run/'fixtures'/case/'arrays.npz', allow_pickle=False) as arrays:
            assert {name: array_hash(arrays[name]) for name in arrays.files} == meta['hashes']
        fixtures[case] = meta['fixture_sha256']
    groups = defaultdict(list)
    for row in rows + warms:
        assert row['fixture_sha256'] == fixtures[row['case_id']]
        assert row['backend_identity'].startswith(dict(numpy='pcc_numpy.',numba='pcc_numba.',cython='pcc_step.')[row['backend']])
        assert row['mode'] == 'early_stopping'
        n = row['effective_iterations']
        assert 0 < n <= row['max_iter'] and row['checks'] == (n+9)//10
        assert row['reached_max_iter'] == (n == row['max_iter'])
        assert row['stopped_before_cap'] == (n < row['max_iter'])
        if row['stop_reason'] == 'early_stop':
            assert row['stalled_checks'] > row['stop_max'] and (n-1)%10 == 0
        else:
            assert row['stop_reason'] == 'max_iter' and n == row['max_iter']
            assert row['stalled_checks'] <= row['stop_max']
        assert set(row['state_hashes']) == {'predictions','dominance','positions','strengths','distances','owndeg'}
        assert all(len(h) == 64 and int(h,16) >= 0 for h in row['state_hashes'].values())
        assert row['propagation_seed_effective'] == (123456789 if row['backend']=='cython' else 0)
        assert np.isfinite(row['accuracy_unlabeled']) and 0 <= row['accuracy_unlabeled'] <= 1
        for field in ('propagation_s','fit_predict_s','initialization_and_dispatch_s','postprocessing_s'):
            assert np.isfinite(row[field]) and row[field] > 0
        assert abs(row['fit_predict_s'] - sum(row[f] for f in ('propagation_s','initialization_and_dispatch_s','postprocessing_s'))) < 1e-8
        if 'repeat' in row:
            assert row['max_iter'] == 500000
            assert row['execution_order'] == manifest['configuration']['execution_schedule'][row['case_id']][row['repeat']]
            assert row['execution_order'][row['order_index']] == row['backend']
            groups[row['case_id'],row['backend']].append(row)
        else:
            assert row['max_iter'] == 2
    assert summarize(rows) == read(run/'summary.json')
    for case, backend in itertools.product(cases, BACKENDS):
        selected = [r for r in warms if (r['case_id'],r['backend']) == (case,backend)]
        assert selected[0]['state_hashes'] == selected[1]['state_hashes']
    table = []
    for dataset in DATASETS:
        for seed in (10, 11):
            case = f'{dataset}-labels{seed}'
            for backend in BACKENDS:
                selected = groups[case,backend]
                identities = [(r['state_hashes'],r['effective_iterations'],r['stop_reason'],r['accuracy_unlabeled']) for r in selected]
                assert all(i == identities[0] for i in identities)
                item = dict(dataset=dataset,seed=seed,backend=backend,iterations=selected[0]['effective_iterations'],
                    reason=selected[0]['stop_reason'],accuracy=selected[0]['accuracy_unlabeled'],
                    cap_hits=sum(r['reached_max_iter'] for r in selected),
                    censored=sum(r['stop_reason']=='max_iter' for r in selected))
                for field in ('propagation_s','fit_predict_s'):
                    values = [r[field] for r in selected]
                    q1,q3=np.percentile(values,[25,75],method='linear')
                    item.update({f'{field}_median':statistics.median(values),f'{field}_iqr':float(q3-q1),
                                 f'{field}_min':min(values),f'{field}_max':max(values)})
                table.append(item)
    preservation = read(root/'prior-artifact-preservation.json')
    assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest() == h for p,h in preservation.items())
    prior = read(baseline/'summary.json')
    assert len(prior) == 600
    full = {}
    overhead = sum(6*(r['initialization_and_dispatch_s']['median']+r['postprocessing_s']['median']) for r in prior)
    warm_s=sum(r['fit_predict_s'] for r in lines(baseline/'warmups.jsonl'))
    prep_s=sum(read(baseline/'preparation.json')['totals'][f] for f in ('dataset_load_s','preprocessing_and_labels_s','graph_build_s'))
    charges=overhead+warm_s+prep_s+read(baseline/'seed_helper.json')['first_call_s']
    for field in ('q1','median','q3','maximum'):
        full[f'historical_cap_{field}_s']=sum(r['propagation_s'][field]*50*6 for r in prior)+charges
    pilot_cap_by_dataset = {d:sum(max(r['propagation_s_per_iteration'] for r in groups[f'{d}-labels{s}', b])*500000*3
                                  for s in (10,11) for b in BACKENDS) for d in DATASETS}
    measured_four_projection = 10*sum(statistics.median(r['propagation_s'] for r in selected) for selected in groups.values())
    other_cap = sum(r['propagation_s']['median']*50*6 for r in prior if r['case_id'].rsplit('-labels',1)[0] not in DATASETS)
    full.update(four_dataset_pilot_behavior_projection_s=measured_four_projection,
                other_sixteen_historical_median_cap_s=other_cap,
                hybrid_pilot_behavior_plus_other_caps_s=measured_four_projection+other_cap+charges,
                conservative_scenario_s=1.5*max(full['historical_cap_maximum_s'],other_cap+10*sum(pilot_cap_by_dataset.values())+charges),
                overhead_charges_s=charges,
                note='Hybrid extrapolates two masks to ten only for four tested datasets; remaining 16 assume cap. Conservative scenario applies 50% planning headroom to the larger historical-max or pilot-rate cap projection, not a runtime bound. No stopping savings assumed for untested workloads.')
    accuracy_differences = []
    for case in sorted(cases):
        values = {b:groups[case,b][0]['accuracy_unlabeled'] for b in BACKENDS}
        accuracy_differences.append(dict(case_id=case,accuracy=values,
            spread_percentage_points=100*(max(values.values())-min(values.values())),
            numba_minus_numpy_pp=100*(values['numba']-values['numpy']),
            cython_minus_numpy_pp=100*(values['cython']-values['numpy'])))
    return dict(audit='passed',measurements=72,warmups=48,fixtures=8,groups=table,
                accuracy_differences=accuracy_differences,
                cap_hits=sum(r['reached_max_iter'] for r in rows),censored=sum(r['stop_reason']=='max_iter' for r in rows),
                actual_measured_iterations=sum(r['effective_iterations'] for r in rows),
                measured_propagation_s=sum(r['propagation_s'] for r in rows),
                measured_fit_s=sum(r['fit_predict_s'] for r in rows),
                execution=read(root/'execution.json'),full_campaign=full,
                prior_files_verified=len(preservation),outliers_excluded=0)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--baseline',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    result=analyze(args.root,args.baseline)
    args.output.mkdir(parents=True,exist_ok=False)
    (args.output/'analysis.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    with (args.output/'results.csv').open('w',encoding='utf-8',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(result['groups'][0]))
        writer.writeheader()
        writer.writerows(result['groups'])
    print(json.dumps({k:v for k,v in result.items() if k != 'groups'},indent=2))

if __name__ == '__main__':
    main()
