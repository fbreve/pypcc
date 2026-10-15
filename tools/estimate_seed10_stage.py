"""Offline stage-10 estimates from preserved calibrations and fixture storage."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def estimate(baseline,first,second):
    a=read(first/'analysis.json')
    b=read(second/'analysis.json')
    first_groups=[r for r in a['groups'] if r['seed']==10]
    second_groups=b['rows']
    measured={}
    for r in first_groups:
        measured[r['dataset'],r['backend']]=dict(iterations=r['iterations'],
            propagation_s=r['propagation_s_median'],fit_s=r['fit_predict_s_median'],
            rate_max=r['propagation_s_max']/r['iterations'],source_repeats=3)
    for r in second_groups:
        key=r['dataset'],r['backend']
        if key in measured or r['seed']!=10:
            raise ValueError('duplicate/non-seed10 calibration group')
        measured[key]=dict(iterations=r['iterations'],propagation_s=r['propagation_s'],fit_s=r['fit_s'],
            rate_max=r['propagation_s']/r['iterations'],source_repeats=1)
    inventory=read(Path('docs/benchmarks/stopping_patience/inventory.json'))['datasets']
    names=[{'Wine':'wine','Digits':'digits','Iris':'iris','BreastCancer':'breast_cancer'}.get(r['name'],r['name'].lower()) for r in inventory]
    if set(measured)!={(name,backend) for name in names for backend in ('numpy','numba','cython')}:
        raise ValueError('require all 60 fixture/backend groups')
    old=[r for r in read(baseline/'summary.json') if r['case_id'].endswith('-labels10')]
    if len(old)!=60:
        raise ValueError('require all historical seed-10 baseline groups')
    prep=[r for r in read(baseline/'preparation.json')['fixtures'] if r['case_id'].endswith('-labels10')]
    warm=[json.loads(s) for s in (baseline/'warmups.jsonl').read_text().splitlines()]
    warm=[r for r in warm if r['case_id'].endswith('-labels10')]
    if len(prep)!=20 or len(warm)!=120:
        raise ValueError('preparation/warm-up inventory differs')
    charges=sum(r[f] for r in prep for f in ('dataset_load_s','preprocessing_and_labels_s','graph_build_s'))
    charges+=sum(r['fit_predict_s'] for r in warm)+read(baseline/'seed_helper.json')['first_call_s']
    fit=sum(6*r['fit_s'] for r in measured.values())
    cap=sum(6*r['propagation_s']/r['iterations']*500000 for r in measured.values())
    maximum_cap=sum(6*r['rate_max']*500000 for r in measured.values())
    overhead=sum(6*(r['fit_s']-r['propagation_s']) for r in measured.values())
    historical={field:sum(300*r['propagation_s'][field] for r in old)+overhead+charges
                for field in ('q1','median','q3','maximum')}
    storage=[]
    for name in names:
        path=baseline/'fixtures'/f'{name}-labels10'/'arrays.npz'
        with np.load(path,allow_pickle=False) as data:
            byte_count=sum(data[k].nbytes for k in data.files)
            n=len(data['truth']); c=len(np.unique(data['truth']))
            particles=3*c
        storage.append(dict(dataset=name,npz_bytes=path.stat().st_size,npz_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),uncompressed_fixture_bytes=byte_count,
            dominance_bytes=n*c*8,distance_table_bytes=n*particles))
    groups=[dict(dataset=k[0],backend=k[1],**v,
        stage_fit_s=6*v['fit_s'],stage_measured_iterations=6*v['iterations']) for k,v in measured.items()]
    return dict(status='stage_preparation_only_no_historical_execution',stage_seed=10,
        measured_fits=360,warmup_fits=120,graph_preparations=20,full_plan_fits=3600,full_plan_fixtures=200,
        measured_iteration_cap=180000000,particle_update_cap=2295000000,
        warmup_iterations=240,warmup_particle_updates=3060,
        observed_stopping_accounted_s=fit+charges,observed_stopping_fit_s=fit,
        observed_iterations=sum(6*r['iterations'] for r in measured.values()),
        observed_censored_fit_scenario=6*sum(r['iterations']==500000 for r in measured.values()),
        observed_rate_cap_s=cap+overhead+charges,observed_max_rate_cap_s=maximum_cap+overhead+charges,
        historical_cap_s=historical,
        conservative_cap_s=1.5*max(historical['maximum'],maximum_cap+overhead+charges),
        historical_preparation_warmup_helper_s=charges,groups=groups,storage=storage,
        resource_projection=dict(stage_npz_bytes=sum(r['npz_bytes'] for r in storage),
            full_npz_bytes_proxy=10*sum(r['npz_bytes'] for r in storage),
            stage_cached_array_bytes=sum(r['uncompressed_fixture_bytes'] for r in storage),
            full_cached_array_bytes=10*sum(r['uncompressed_fixture_bytes'] for r in storage),
            max_dominance_bytes=max(r['dominance_bytes'] for r in storage),
            max_distance_table_bytes=max(r['distance_table_bytes'] for r in storage)),
        input_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in
            (first/'analysis.json',second/'analysis.json',baseline/'summary.json',baseline/'preparation.json',baseline/'warmups.jsonl',baseline/'seed_helper.json',Path('docs/benchmarks/stopping_patience/inventory.json'))},
        limitations='Calibrations use the exact seed but fewer/differently ordered repetitions. Six-permutation order, machine load and state costs may change timing. Single-fit rates for 16 datasets provide no IQR. Historical charges exclude new checkpoint/audit I/O, startup and restart overhead. Conservative headroom is a scenario, not a cutoff or guarantee. NPZ full storage is a proxy; memory figures exclude Python/library/allocator/transient workspaces.')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline',type=Path,required=True)
    parser.add_argument('--first',type=Path,default=Path('docs/benchmarks/early_stopping_pilot/results'))
    parser.add_argument('--second',type=Path,default=Path('docs/benchmarks/early_stopping_second_calibration/results'))
    args=parser.parse_args()
    print(json.dumps(estimate(args.baseline,args.first,args.second),indent=2,allow_nan=False))
