"""Fixed, held-out experimental stopping benchmark (Windows pypcc-dev)."""
import argparse
import copy
import hashlib
import json
import platform
import subprocess
import time
from pathlib import Path

import numpy as np
from sklearn.datasets import load_wine, load_digits, load_iris, load_breast_cancer
from sklearn.preprocessing import StandardScaler
import benchmark_backends as fixed
from diagnose_digits_stopping import initial_arguments, predictions
from experiments.stopping_range import propagate_range, THRESHOLD, WINDOW, PATIENCE, FIRST_ALLOWED
from test_sequential_safety import next_random

PROTOCOL_PATH = Path('docs/benchmarks/stopping_validation/protocol.json')


def check_state(a, labels):
    dominance = a[11]
    assert np.isfinite(dominance).all()
    assert np.all((dominance >= -1e-12) & (dominance <= 1+1e-12))
    np.testing.assert_allclose(dominance.sum(axis=1), 1., atol=1e-12)
    known = labels != -1
    np.testing.assert_array_equal(predictions(a)[known], labels[known])
    assert np.all((a[7] >= 0) & (a[7] < len(labels)))
    assert a[10].dtype == np.uint8


def compare(a, b, rng_a, rng_b):
    for i in (7, 9, 10, 11, 12):
        np.testing.assert_array_equal(a[i], b[i])
    assert rng_a == rng_b, 'RNG mismatch'


def benchmark(output):
    if platform.system() != 'Windows' or not fixed.pcc_step.__file__.endswith('.pyd'):
        raise RuntimeError('requires Windows and native Cython')
    protocol = json.loads(PROTOCOL_PATH.read_text())
    assert protocol['threshold']==THRESHOLD and protocol['window_iterations']==WINDOW
    assert protocol['patience_windows']==PATIENCE and protocol['first_allowed_alternative_iteration']==FIRST_ALLOWED
    output.mkdir(parents=True, exist_ok=True)
    metadata = dict(protocol_sha256=hashlib.sha256(PROTOCOL_PATH.read_bytes()).hexdigest(),
                    protocol_commit='65fb06c',
                    source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
                    environment=dict(python=platform.python_version(), platform=platform.platform(),
                                     processor=fixed.processor_name(),
                                     versions={m.__name__:m.__version__ for m in
                                               (np,fixed.numba,fixed.Cython,fixed.scipy,fixed.sklearn)},
                                     cython_path=fixed.pcc_step.__file__,
                                     cython_binary_sha256=hashlib.sha256(Path(fixed.pcc_step.__file__).read_bytes()).hexdigest()),
                    datasets={}, warmups=[])
    rows, pairs = [], []
    fixed.seed_numba(0)
    next_random()
    loaders = dict(Wine=load_wine, Digits=load_digits, Iris=load_iris, BreastCancer=load_breast_cancer)
    for dataset_index, name in enumerate(protocol['datasets']):
        begin = time.perf_counter()
        data = loaders[name]()
        x = StandardScaler().fit_transform(data.data)
        truth = np.asarray(data.target,dtype=np.int64)
        graph = fixed.PCC(impl='numpy')
        graph.build_graph(x,k_nn=protocol['k_nn'],n_jobs=1)
        setup = time.perf_counter()-begin
        hashes = dict(data_sha256=fixed.fingerprint(x,truth), graph_sha256=fixed.fingerprint(graph.neib_list,graph.neib_qt))
        metadata['datasets'][name]=dict(nodes=len(truth),classes=len(np.unique(truth)),shared_setup_s=setup,**hashes,splits=[])
        for seed in protocol['seeds']:
            rng = np.random.default_rng(seed)
            known = np.concatenate([rng.choice(np.flatnonzero(truth==cls),protocol['labels_per_class'],replace=False)
                                    for cls in np.unique(truth)])
            labels = np.full(len(truth),-1,dtype=np.int64)
            labels[known]=truth[known]
            case = dict(dataset=name,seed=seed,truth=truth,labels=labels,edges=graph.neib_list,degrees=graph.neib_qt)
            prep_begin=time.perf_counter()
            initial=initial_arguments(case,protocol,protocol['max_iter'])
            preparation_s=time.perf_counter()-prep_begin
            label_hash=fixed.fingerprint(labels)
            metadata['datasets'][name]['splits'].append(dict(seed=seed,known_indices=known.tolist(),labels_sha256=label_hash))
            if seed==protocol['seeds'][0]:
                for mode in ('original','prototype'):
                    for call in ('first','warm'):
                        a=copy.deepcopy(initial)
                        a[22]=2
                        fixed.seed_numba(seed)
                        start=time.perf_counter()
                        if mode=='original':fixed.pcc_numba.pcc_propagate_numba(*a,return_info=True)
                        else:propagate_range(*a)
                        metadata['warmups'].append(dict(dataset=name,mode=mode,call=call,propagation_s=time.perf_counter()-start))
            signatures=tuple(fixed.pcc_numba._pcc_propagate_numba_sequential.signatures)
            for repeat in range(protocol['repetitions_per_seed']):
                order=['original','prototype']
                if (dataset_index+seed+repeat)%2:order.reverse()
                outputs={}
                for index,mode in enumerate(order):
                    a=copy.deepcopy(initial)
                    fixed.seed_numba(seed)
                    start=time.perf_counter()
                    info=(fixed.pcc_numba.pcc_propagate_numba(*a,return_info=True)
                          if mode=='original' else propagate_range(*a))
                    elapsed=time.perf_counter()-start
                    rng_after=float(next_random())
                    assert signatures==tuple(fixed.pcc_numba._pcc_propagate_numba_sequential.signatures),'JIT during measurement'
                    check_state(a,labels)
                    predicted=predictions(a)
                    unknown=labels==-1
                    row=dict(dataset=name,seed=seed,repeat=repeat,mode=mode,order_index=index,
                             propagation_s=elapsed,propagation_us_per_iteration=1e6*elapsed/info['effective_iterations'],
                             accuracy_unlabeled=float(np.mean(predicted[unknown]==truth[unknown])),
                             prediction_sha256=fixed.fingerprint(predicted),rng_after=rng_after,
                             effective_iterations=info['effective_iterations'],checks=info['checks'],stop_reason=info['stop_reason'],
                             stop_max=info['stop_max'],stalled_checks=info['stalled_checks'],
                             last_window_range=info.get('last_window_range'),qualifying_windows=info.get('qualifying_windows'),
                             max_iter=protocol['max_iter'],initialization_capture_s=preparation_s,
                             shared_dataset_setup_s=setup,labels_sha256=label_hash,**hashes)
                    outputs[mode]=(a,rng_after,row)
                original,prototype=outputs['original'],outputs['prototype']
                o,p=original[2],prototype[2]
                no_alternative=p['stop_reason']!='alternative_range'
                if no_alternative:
                    compare(original[0],prototype[0],original[1],prototype[1])
                    assert o['effective_iterations']==p['effective_iterations'] and o['stop_reason']==p['stop_reason']
                    verification='original_state_rng_exact'
                else:
                    # Verify the actual stopped prototype against the unchanged
                    # original dynamics at the same prefix, outside timings.
                    prefix=copy.deepcopy(initial)
                    prefix[22],prefix[23]=p['effective_iterations'],False
                    fixed.seed_numba(seed)
                    fixed.pcc_numba.pcc_propagate_numba(*prefix)
                    compare(prefix,prototype[0],float(next_random()),prototype[1])
                    verification='same_prefix_state_rng_exact'
                    assert p['effective_iterations']>=FIRST_ALLOWED
                    assert p['last_window_range']<=THRESHOLD and p['qualifying_windows']>=PATIENCE
                pair=dict(dataset=name,seed=seed,repeat=repeat,alternative_triggered=not no_alternative,
                          original_iterations=o['effective_iterations'],prototype_iterations=p['effective_iterations'],
                          saved_iterations=o['effective_iterations']-p['effective_iterations'],
                          saved_fraction=1-p['effective_iterations']/o['effective_iterations'],
                          original_s=o['propagation_s'],prototype_s=p['propagation_s'],
                          speedup=o['propagation_s']/p['propagation_s'],
                          original_accuracy=o['accuracy_unlabeled'],prototype_accuracy=p['accuracy_unlabeled'],
                          accuracy_difference_pp=100*(p['accuracy_unlabeled']-o['accuracy_unlabeled']),
                          prediction_mismatches=int(np.count_nonzero(predictions(original[0])!=predictions(prototype[0]))),
                          state_rng_verification=verification)
                pairs.append(pair)
                for _,_,row in outputs.values():rows.append(row)
            # Measurements must replay exactly across timing repetitions.
            for mode in ('original','prototype'):
                selected=[r for r in rows if (r['dataset'],r['seed'],r['mode'])==(name,seed,mode)]
                assert len({(r['effective_iterations'],r['prediction_sha256'],r['rng_after']) for r in selected})==1
            fixed.write_csv(output/'runs.csv',rows)
            fixed.write_csv(output/'paired_runs.csv',pairs)
            print(json.dumps(dict(dataset=name,seed=seed,original_iterations=o['effective_iterations'],
                                  prototype_iterations=p['effective_iterations'],reason=p['stop_reason'],
                                  accuracy_change_pp=pair['accuracy_difference_pp'],verification=verification)),flush=True)
    seed_rows=[]
    for name in protocol['datasets']:
        for seed in protocol['seeds']:
            pp=[r for r in pairs if (r['dataset'],r['seed'])==(name,seed)]
            item=dict(dataset=name,seed=seed,alternative_triggered=pp[0]['alternative_triggered'])
            for key in ('original_iterations','prototype_iterations','saved_fraction','original_s','prototype_s',
                        'original_accuracy','prototype_accuracy','accuracy_difference_pp','prediction_mismatches'):
                item[key]=float(np.mean([r[key] for r in pp]))
            item['speedup']=item['original_s']/item['prototype_s']
            seed_rows.append(item)
    summary=[]
    for name in protocol['datasets']:
        selected=[r for r in seed_rows if r['dataset']==name]
        item=dict(dataset=name,seeds=len(selected),alternative_triggered_seeds=sum(r['alternative_triggered'] for r in selected))
        for key in ('original_iterations','prototype_iterations','saved_fraction','original_s','prototype_s',
                    'original_accuracy','prototype_accuracy','accuracy_difference_pp','speedup'):
            values=[r[key] for r in selected]
            item[key+'_mean'],item[key+'_std']=float(np.mean(values)),float(np.std(values,ddof=1))
        item['speedup_ratio_of_means']=item['original_s_mean']/item['prototype_s_mean']
        item['worst_accuracy_difference_pp']=min(r['accuracy_difference_pp'] for r in selected)
        summary.append(item)
    fixed.write_csv(output/'seed_summary.csv',seed_rows)
    fixed.write_csv(output/'summary.csv',summary)
    (output/'metadata.json').write_text(json.dumps(metadata,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,default=Path('docs/benchmarks/stopping_validation'))
    benchmark(parser.parse_args().output_dir)
