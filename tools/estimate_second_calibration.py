"""Offline estimate of the remaining 16 datasets; never executes PCC."""
import argparse
import hashlib
import json
from pathlib import Path

EXCLUDED = {'wine', 'digits', 'germancredit', 'satimage'}
BACKENDS = ('numpy', 'numba', 'cython')

def read(path):
    return json.loads(path.read_text(encoding='utf-8'))

def estimate(baseline, inventory_path):
    inventory = read(inventory_path)
    aliases = {'Iris': 'iris', 'BreastCancer': 'breast_cancer'}
    datasets = {aliases.get(row['name'], row['name'].lower()): row for row in inventory['datasets']
                if row['name'].lower() not in EXCLUDED}
    if len(datasets) != 16:
        raise ValueError('Require exactly the 16 remaining historical datasets')
    cases = {f'{name}-labels10' for name in datasets}
    selected = [r for r in read(baseline/'summary.json') if r['case_id'] in cases]
    expected = {(case, backend) for case in cases for backend in BACKENDS}
    if len(selected) != 48 or {(r['case_id'], r['backend']) for r in selected} != expected:
        raise ValueError('Require all 48 matching historical groups, without duplicates')
    if any(r['mode'] != 'fixed' or r['max_iter'] != 10000 or r['repetitions'] != 6 for r in selected):
        raise ValueError('Require original six-repeat 10,000-fixed-iteration baseline')
    groups = [dict(case_id=r['case_id'], backend=r['backend'],
        propagation_cap_s={f:50*r['propagation_s'][f] for f in ('q1','median','q3','maximum')},
        fit_overhead_s=sum(r[f]['median'] for f in ('initialization_and_dispatch_s','postprocessing_s')))
        for r in selected]
    warmups = [json.loads(line) for line in (baseline/'warmups.jsonl').read_text().splitlines()]
    warmups = [r for r in warmups if r['case_id'] in cases]
    if len(warmups) != 96 or {(r['case_id'],r['backend'],r['call']) for r in warmups} != {
            (c,b,w) for c in cases for b in BACKENDS for w in ('first','warm')}:
        raise ValueError('Require all 96 matching warm-up records')
    prep = [r for r in read(baseline/'preparation.json')['fixtures'] if r['case_id'] in cases]
    if len(prep) != 16 or any(r['reused'] for r in prep):
        raise ValueError('Require all 16 separately charged preparations')
    charges = dict(fit_overhead_s=sum(r['fit_overhead_s'] for r in groups),
        historical_warmup_fit_s=sum(r['fit_predict_s'] for r in warmups),
        historical_preparation_s=sum(r[f] for r in prep for f in
            ('dataset_load_s','preprocessing_and_labels_s','graph_build_s')),
        historical_seed_helper_s=read(baseline/'seed_helper.json')['first_call_s'])
    totals = {f:sum(r['propagation_cap_s'][f] for r in groups)+sum(charges.values())
              for f in ('q1','median','q3','maximum')}
    workloads = []
    for name in datasets:
        matching = [r for r in groups if r['case_id'] == f'{name}-labels10']
        workloads.append(dict(dataset=name, nodes=datasets[name]['nodes'],
            classes=datasets[name]['classes'],
            median_cap_s_by_backend={r['backend']:r['propagation_cap_s']['median'] for r in matching},
            median_propagation_cap_s=sum(r['propagation_cap_s']['median'] for r in matching),
            maximum_rate_propagation_cap_s=sum(r['propagation_cap_s']['maximum'] for r in matching)))
    workloads.sort(key=lambda r:r['median_propagation_cap_s'], reverse=True)
    # Risk-first anchors, then decreasing measured cap cost. No propagation RNG use.
    order = ['pendigits','pageblocks'] + [r['dataset'] for r in workloads
                                        if r['dataset'] not in ('pendigits','pageblocks')]
    tasks = [dict(task_index=3*i+j,dataset=name,label_seed=10,backend=backend,
                  measurement_index=0,backend_order_index=j)
             for i,name in enumerate(order) for j,backend in enumerate(('numba','cython','numpy'))]
    inputs = {p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in
              (baseline/'summary.json',baseline/'warmups.jsonl',baseline/'preparation.json',
               baseline/'seed_helper.json',inventory_path)}
    return dict(status='plan_only_not_executed',baseline=str(baseline),input_sha256=inputs,
        datasets=workloads,groups=groups,charges=charges,projected_accounted_s=totals,
        conservative_cap_s=1.5*totals['maximum'],conservative_headroom_fraction=.5,
        measured_fits=48,warmup_fits=96,graph_preparations=16,
        measured_iteration_cap=24000000,
        measured_particle_update_cap=1500000*3*sum(r['classes'] for r in workloads),
        warmup_iterations=192,warmup_particle_updates=12*3*sum(r['classes'] for r in workloads),
        execution_order=order,tasks=tasks,
        parameters=dict(p_grd=.5,delta_v=.1,deltap=1.,dexp=2.,es_chk=2000,max_iter=500000,
                        early_stop=True,k_nn=10,labels_per_class=3,propagation_seed=0),
        limitations='No stopping savings assumed. Linear fixed-throughput cap scenario, not a bound or confidence interval. Unmeasured early-stop scanning, startup, copies, hashing, fsync and interruptions may add cost. Persistent-process baseline charges do not include extra process-start warm-ups on resume. Headroom is a planning scenario, not an execution cutoff.')

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline',type=Path,required=True)
    parser.add_argument('--inventory',type=Path,default=Path('docs/benchmarks/stopping_patience/inventory.json'))
    args=parser.parse_args()
    print(json.dumps(estimate(args.baseline,args.inventory),indent=2,allow_nan=False))

if __name__ == '__main__':
    main()
