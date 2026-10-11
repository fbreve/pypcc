"""Streaming diagnostic research on unchanged sequential Numba PCC trajectories."""
import argparse
import copy
import csv
import hashlib
import json
import platform
import os
import subprocess
import time
from pathlib import Path

import numpy as np
from numba import _helperlib
from sklearn.datasets import load_wine,load_digits,load_iris,load_breast_cancer
from sklearn.preprocessing import StandardScaler
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import benchmark_backends as fixed
from diagnose_digits_stopping import initial_arguments, predictions
from experiments.convergence_measurements import LOW_GAP,maxima_gaps,norms,distribution,hidden_changes,is_checkpoint

PROTOCOL_PATH=Path('docs/benchmarks/stopping_validation/protocol.json')


def rng_state():
    # Read the current thread's Numba NumPy MT state, without drawing or reseeding.
    return _helperlib.rnd_get_state(_helperlib.rnd_get_np_state_ptr())


def digest(value):
    return hashlib.sha256(json.dumps(value).encode()).hexdigest()


class Stream:
    def __init__(self,path):
        self.file=path.open('w',encoding='utf-8',newline='')
        self.writer=None
    def write(self,row):
        if self.writer is None:
            self.writer=csv.DictWriter(self.file,fieldnames=list(row))
            self.writer.writeheader()
        self.writer.writerow(row)
    def close(self):self.file.close()


def run_case(initial,case,event,expected_original_hash,streams):
    a=copy.deepcopy(initial)
    degree=fixed.pcc_numba.validate_step_inputs(*a[:3],*a[5:13])
    fixed.pcc_numba.validate_buffers(a[5],degree,*a[15:22])
    n,c=a[11].shape
    unknown=case['labels']==-1
    truth=case['truth']
    before=predictions(a)
    previous_dom=a[11].copy()
    previous_gap=maxima_gaps(a[11])[1]
    rich_labels=before.copy()
    rich_dom=previous_dom.copy()
    rich_metric=float(fixed.pcc_numba._pcc_calc_mmpot_sequential(a[11],n,c))
    rich_iteration=0
    sparse_labels=before.copy()
    sparse_start=0
    ever=np.zeros(n,dtype=bool)
    flips=np.zeros(n,dtype=np.int64)
    late_flips=np.zeros(n,dtype=np.int64)
    node_motion=np.zeros(n)
    late_motion=np.zeros(n)
    low_observations=np.zeros(n,dtype=np.int64)
    late_low_observations=np.zeros(n,dtype=np.int64)
    references={0:before.copy()}
    saved={}
    selected={5001,15001,30000,event}
    best=0.
    previous_metric=rich_metric
    counter=checks=resets=longest=0
    interval_resets=interval_checks=0
    interval_gains=0.
    interval_gain_max=0.
    original_first=None
    stable_streak=longest_stable=episodes=0
    sparse_zero_hidden=hidden_total=sparse_windows=0
    late_resets=late_tiny=late_zero_flip_resets=late_flip_total=late_uncertain_flips=0
    late_gain_sum=0.
    late_gain_min=float('inf')
    late_gain_max=0.
    iteration=0
    fixed.seed_numba(case['seed'])
    start=time.perf_counter()
    while iteration<30000:
        end=min(30000,1 if iteration==0 else iteration+10)
        fixed.pcc_numba._pcc_propagate_numba_sequential(*a[:22],end-iteration,False,a[24],a[25],False)
        iteration=end
        dominance=a[11]
        labels=predictions(a)
        maxima,gaps=maxima_gaps(dominance)
        changed=labels!=before
        count=int(changed.sum())
        node_delta=np.abs(dominance-previous_dom).sum(axis=1)
        node_motion+=node_delta
        flips+=changed
        ever|=changed
        low_observations+=(gaps<LOW_GAP)
        metric=float(fixed.pcc_numba._pcc_calc_mmpot_sequential(dominance,n,c))
        actual_check=(iteration-1)%10==0
        gain=max(0.,metric-best) if actual_check else 0.
        improved=actual_check and metric>best
        if actual_check:
            checks+=1
            interval_checks+=1
            if improved:
                best,counter=metric,0
                resets+=1
                interval_resets+=1
                interval_gains+=gain
                interval_gain_max=max(interval_gain_max,gain)
            else:
                counter+=1
                longest=max(longest,counter)
            if counter>a[25] and original_first is None:original_first=iteration
        if count:
            if stable_streak:episodes+=1
            stable_streak=0
        else:
            stable_streak+=1
            longest_stable=max(longest_stable,stable_streak)
        if iteration>20001:
            late_flips+=changed
            late_motion+=node_delta
            late_low_observations+=(gaps<LOW_GAP)
            late_flip_total+=count
            late_uncertain_flips+=int(np.count_nonzero(changed & (previous_gap<LOW_GAP)))
            if improved:
                late_resets+=1
                late_gain_sum+=gain
                late_gain_min=min(late_gain_min,gain)
                late_gain_max=max(late_gain_max,gain)
                late_tiny+=gain<=1e-12
                late_zero_flip_resets+=count==0
        streams['checks'].write(dict(dataset=case['dataset'],seed=case['seed'],iteration=iteration,
            original_check=int(actual_check),metric=metric,delta_metric=metric-previous_metric,
            record_gain=gain,record_reset=int(improved),running_best=best,stalled_checks=counter,
            original_eligible=int(counter>a[25]),changed_nodes=count,changed_fraction=count/n,
            low_gap_nodes=int(np.count_nonzero(gaps<LOW_GAP)),
            changed_previously_low_gap=int(np.count_nonzero(changed & (previous_gap<LOW_GAP))),
            unchanged_sampled_streak=stable_streak))
        sparse=(iteration-1)%500==0 and iteration>1 or iteration==30000
        sparse_change=hidden=None
        if sparse:
            assert iteration-sparse_start==(499 if iteration==30000 else 500)
            sparse_windows+=1
            sparse_change=int(np.count_nonzero(labels!=sparse_labels))
            hidden=hidden_changes(labels,sparse_labels,ever)
            hidden_total+=hidden
            sparse_zero_hidden+=sparse_change==0 and hidden>0
        if iteration in (5001,10001,20001):references[iteration]=labels.copy()
        if is_checkpoint(iteration,event):
            norm=norms(dominance,rich_dom)
            row=dict(dataset=case['dataset'],seed=case['seed'],iteration=iteration,
                interval_iterations=iteration-rich_iteration,original_check=int(actual_check),
                metric=metric,delta_since_checkpoint=metric-rich_metric,running_best=best,
                record_resets_since_checkpoint=interval_resets,record_gain_sum=interval_gains,
                record_gain_max=interval_gain_max,original_checks_since_checkpoint=interval_checks,
                stalled_checks=counter,stop_max=a[25],original_eligible=int(counter>a[25]),
                first_original_stop=original_first,changed_since_checkpoint=int(np.count_nonzero(labels!=rich_labels)),
                changed_pct_since_checkpoint=100*np.mean(labels!=rich_labels),
                changed_since_initial=int(np.count_nonzero(labels!=references[0])),
                changed_since_5001=int(np.count_nonzero(labels!=references[5001])) if 5001 in references else None,
                changed_since_10001=int(np.count_nonzero(labels!=references[10001])) if 10001 in references else None,
                changed_since_20001=int(np.count_nonzero(labels!=references[20001])) if 20001 in references else None,
                accuracy_unlabeled=float(np.mean(labels[unknown]==truth[unknown])),
                sparse_window_iterations=iteration-sparse_start if sparse else None,
                sparse_endpoint_changes=sparse_change,sparse_hidden_reverting_nodes=hidden,
                **norm,**distribution(dominance,unknown))
            for reference in (0,5001,10001,20001):
                row['changed_pct_since_'+str(reference)]=(100*float(np.mean(labels!=references[reference]))
                                                         if reference in references else None)
            streams['checkpoints'].write(row)
            rich_dom=dominance.copy()
            rich_labels=labels.copy()
            rich_metric,rich_iteration=metric,iteration
            interval_resets=interval_checks=0
            interval_gains=interval_gain_max=0.
        if sparse or iteration==1:
            sparse_labels=labels.copy()
            sparse_start=iteration
            ever[:]=False
        if iteration in selected:
            saved[iteration]=(copy.deepcopy(a),rng_state())
        before=labels
        previous_dom=dominance.copy()
        previous_gap=gaps
        previous_metric=metric
    diagnostic_s=time.perf_counter()-start
    end_state=rng_state()
    # Verification runs occur AFTER the continuous run and cannot affect it.
    for checkpoint,(expected,expected_rng) in saved.items():
        mono=copy.deepcopy(initial)
        mono[22],mono[23]=checkpoint,False
        fixed.seed_numba(case['seed'])
        fixed.pcc_numba.pcc_propagate_numba(*mono)
        for i in (7,9,10,11,12):np.testing.assert_array_equal(mono[i],expected[i])
        assert rng_state()==expected_rng,'checkpoint RNG state mismatch'
        streams['verification'].write(dict(dataset=case['dataset'],seed=case['seed'],iteration=checkpoint,
            state_rng_exact=True,rng_sha256=digest(expected_rng),
            state_sha256=fixed.fingerprint(*[expected[i] for i in (7,9,10,11,12)])))
    assert (original_first or 30000)==event,'original stopping event differs from PR #6'
    assert fixed.fingerprint(predictions(saved[event][0]))==expected_original_hash,'original prediction differs from PR #6'
    known=case['labels']!=-1
    np.testing.assert_array_equal(labels[known],truth[known])
    assert np.isfinite(a[11]).all()
    np.testing.assert_allclose(a[11].sum(axis=1),1.,atol=1e-12)
    assert np.all((a[11]>=-1e-12)&(a[11]<=1+1e-12))
    top_count=max(1,int(np.ceil(n*.05)))
    top_flips=np.sort(late_flips)[-top_count:].sum()
    top_motion=np.sort(late_motion)[-top_count:].sum()
    for node in range(n):
        streams['nodes'].write(dict(dataset=case['dataset'],seed=case['seed'],node=node,labeled=int(known[node]),
            degree=int(case['degrees'][node]),class_changes=int(flips[node]),late_class_changes=int(late_flips[node]),
            cumulative_l1=float(node_motion[node]),late_cumulative_l1=float(late_motion[node]),
            low_gap_observations=int(low_observations[node]),late_low_gap_observations=int(late_low_observations[node]),
            final_maximum=float(maxima[node]),final_gap=float(gaps[node]),
            prediction_5001=int(references[5001][node]),prediction_10001=int(references[10001][node]),
            prediction_20001=int(references[20001][node]),final_prediction=int(labels[node]),true_class=int(truth[node])))
    row=dict(dataset=case['dataset'],seed=case['seed'],iterations=30000,original_checks=checks,
        first_original_stop=original_first,original_stops_within_budget=original_first is not None,
        stop_max=a[25],final_counter=counter,longest_stall=longest,record_resets=resets,
        late_record_resets=late_resets,late_tiny_gains_le_1e_minus12=late_tiny,
        late_record_gain_mean=late_gain_sum/late_resets if late_resets else 0.,
        late_record_gain_min=late_gain_min if late_resets else 0.,late_record_gain_max=late_gain_max,
        late_resets_without_sampled_class_change=late_zero_flip_resets,
        late_class_change_events=late_flip_total,late_nodes_ever_change=int(np.count_nonzero(late_flips)),
        late_top5pct_class_change_share=float(top_flips/late_flip_total) if late_flip_total else 0.,
        late_top5pct_dominance_motion_share=float(top_motion/late_motion.sum()) if late_motion.sum() else 0.,
        late_change_previously_low_gap_share=late_uncertain_flips/late_flip_total if late_flip_total else 0.,
        sparse_windows=sparse_windows,sparse_zero_change_windows_with_hidden_flips=sparse_zero_hidden,
        sparse_hidden_reverting_node_events=hidden_total,temporary_stability_then_change_episodes=episodes,
        longest_unchanged_sampled_streak=longest_stable,
        final_metric=metric,final_low_gap_fraction=float(np.mean(gaps<LOW_GAP)),
        final_accuracy_unlabeled=float(np.mean(labels[unknown]==truth[unknown])),
        diagnostic_s=diagnostic_s,selected_checkpoints_verified=len(saved),final_rng_sha256=digest(end_state))
    streams['summary'].write(row)
    print(json.dumps(row),flush=True)
    return row


def figures(output):
    with (output/'checkpoints.csv').open() as stream:rows=list(csv.DictReader(stream))
    names=('Wine','Digits','Iris','BreastCancer')
    fig,axes=plt.subplots(4,3,figsize=(15,13),sharex=True)
    for di,name in enumerate(names):
        for seed in range(10,20):
            ss=[r for r in rows if r['dataset']==name and int(r['seed'])==seed]
            x=[int(r['iteration']) for r in ss]
            for ax,key in zip(axes[di],('metric','accuracy_unlabeled','low_gap_fraction_unlabeled')):
                ax.plot(x,[float(r[key]) for r in ss],alpha=.65,linewidth=.8,label=f'seed {seed}')
        axes[di,0].set_ylabel(name)
    for ax,title in zip(axes[0],('Mean maximum dominance','Unlabeled accuracy','Unlabeled gap < 0.05')):ax.set_title(title)
    for ax in axes.flat:ax.grid(alpha=.2)
    for ax in axes[-1]:ax.set_xlabel('Continuous iteration')
    fig.legend(*axes[0,0].get_legend_handles_labels(),loc='upper center',ncol=10,fontsize=7)
    fig.tight_layout(rect=[0,0,1,.97]);fig.savefig(output/'comparative_trajectories.png',dpi=150);plt.close(fig)
    fig,axes=plt.subplots(4,3,figsize=(15,13),sharex=True)
    for di,name in enumerate(names):
        for seed in range(10,20):
            ss=[r for r in rows if r['dataset']==name and int(r['seed'])==seed and int(r['iteration'])>=25011]
            x=[int(r['iteration']) for r in ss]
            for ax,key in zip(axes[di],('metric','changed_since_checkpoint','matrix_frobenius')):
                ax.plot(x,[float(r[key]) for r in ss],alpha=.6,linewidth=.7,label=f'seed {seed}')
        axes[di,0].set_ylabel(name)
    for ax,title in zip(axes[0],('Late aggregate dominance','Changes / 10-step sampled interval','Full-matrix Frobenius change')):ax.set_title(title)
    for ax in axes.flat:ax.grid(alpha=.2)
    for ax in axes[-1]:ax.set_xlabel('Continuous iteration; final interval is 9 steps')
    fig.legend(*axes[0,0].get_legend_handles_labels(),loc='upper center',ncol=10,fontsize=7)
    fig.tight_layout(rect=[0,0,1,.97]);fig.savefig(output/'late_dynamics.png',dpi=150);plt.close(fig)


def main(output):
    if platform.system()!='Windows' or not fixed.pcc_step.__file__.endswith('.pyd'):
        raise RuntimeError('requires Windows pypcc-dev and native Cython')
    protocol=json.loads(PROTOCOL_PATH.read_text())
    assert protocol['seeds']==list(range(10,20))
    output.mkdir(parents=True,exist_ok=True)
    baseline=json.loads(Path('docs/benchmarks/stopping_validation/metadata.json').read_text())
    with Path('docs/benchmarks/stopping_validation/seed_summary.csv').open() as stream:
        events={(r['dataset'],int(r['seed'])):int(float(r['original_iterations'])) for r in csv.DictReader(stream)}
    with Path('docs/benchmarks/stopping_validation/runs.csv').open() as stream:
        original_hashes={(r['dataset'],int(r['seed'])):r['prediction_sha256'] for r in csv.DictReader(stream)
                         if r['mode']=='original' and r['repeat']=='0'}
    metadata=dict(source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        source_protocol=protocol,
        diagnostic_protocol=dict(trajectories_per_seed=1,max_iter=30000,propagation_early_stop=False,
            original_stop_shadow=True,alternative_criterion_used=False,
            parameters={key:protocol[key] for key in ('datasets','seeds','k_nn','labels_per_class','p_grd','delta_v','deltap','dexp','es_chk')}),
        seed_status='previously examined in PR #6, not independent validation',
        descriptive_low_gap=LOW_GAP,late_window_start=20001,rich_sampling='500 steps; 10 steps within +/-500 of PR6 stop and from 25001 onward',
        script_sha256_at_measurement=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        environment=dict(python=platform.python_version(),platform=platform.platform(),processor=fixed.processor_name(),
            conda_prefix=os.environ.get('CONDA_PREFIX'),
            versions={m.__name__:m.__version__ for m in (np,fixed.numba,fixed.Cython,fixed.scipy,fixed.sklearn,matplotlib)},
            cython_path=fixed.pcc_step.__file__,cython_binary_sha256=hashlib.sha256(Path(fixed.pcc_step.__file__).read_bytes()).hexdigest(),
            rng_observation='Numba private _helperlib.rnd_get_state; read-only, current thread'),datasets={})
    streams={name:Stream(output/(name+'.csv')) for name in ('checks','checkpoints','nodes','verification','summary')}
    summaries=[]
    loaders=dict(Wine=load_wine,Digits=load_digits,Iris=load_iris,BreastCancer=load_breast_cancer)
    try:
        for name in protocol['datasets']:
            data=loaders[name]()
            truth=np.asarray(data.target,dtype=np.int64)
            x=StandardScaler().fit_transform(data.data)
            graph=fixed.PCC(impl='numpy');graph.build_graph(x,k_nn=10,n_jobs=1)
            saved=baseline['datasets'][name]
            assert fixed.fingerprint(x,truth)==saved['data_sha256']
            assert fixed.fingerprint(graph.neib_list,graph.neib_qt)==saved['graph_sha256']
            rr,cc=[],[]
            for node,degree in enumerate(graph.neib_qt):
                rr.extend([node]*int(degree));cc.extend(graph.neib_list[node,:degree].tolist())
            adjacency=csr_matrix((np.ones(len(rr)),(rr,cc)),shape=(len(truth),len(truth)))
            components=connected_components(adjacency,directed=False,return_labels=False)
            metadata['datasets'][name]=dict(nodes=len(truth),classes=len(np.unique(truth)),
                graph_components=components,degree_min=int(graph.neib_qt.min()),degree_mean=float(graph.neib_qt.mean()),
                degree_max=int(graph.neib_qt.max()),data_sha256=saved['data_sha256'],graph_sha256=saved['graph_sha256'],splits=saved['splits'])
            for split in saved['splits']:
                seed=split['seed']
                labels=np.full(len(truth),-1,dtype=np.int64)
                known=np.asarray(split['known_indices']);labels[known]=truth[known]
                assert fixed.fingerprint(labels)==split['labels_sha256']
                case=dict(dataset=name,seed=seed,truth=truth,labels=labels,edges=graph.neib_list,degrees=graph.neib_qt)
                initial=initial_arguments(case,protocol,30000)
                warm=copy.deepcopy(initial)
                fixed.pcc_numba.pcc_propagate_numba(*warm[:22],2,True,*warm[24:])
                fixed.pcc_numba._pcc_calc_mmpot_sequential(warm[11],len(truth),warm[5])
                signatures=tuple(fixed.pcc_numba._pcc_propagate_numba_sequential.signatures)
                summaries.append(run_case(initial,case,events[name,seed],original_hashes[name,seed],streams))
                assert signatures==tuple(fixed.pcc_numba._pcc_propagate_numba_sequential.signatures)
    finally:
        for stream in streams.values():stream.close()
    aggregates=[]
    for name in protocol['datasets']:
        rows=[r for r in summaries if r['dataset']==name]
        item=dict(dataset=name,seeds=len(rows),original_stopped_seeds=sum(r['original_stops_within_budget'] for r in rows))
        for key in ('late_record_resets','late_tiny_gains_le_1e_minus12','late_record_gain_mean','late_resets_without_sampled_class_change',
            'late_class_change_events','late_nodes_ever_change','late_top5pct_class_change_share','late_top5pct_dominance_motion_share',
            'late_change_previously_low_gap_share','sparse_zero_change_windows_with_hidden_flips','longest_stall',
            'final_metric','final_low_gap_fraction','final_accuracy_unlabeled'):
            values=[r[key] for r in rows]
            item[key+'_mean'],item[key+'_std']=float(np.mean(values)),float(np.std(values,ddof=1))
        aggregates.append(item)
    fixed.write_csv(output/'aggregate.csv',aggregates)
    (output/'metadata.json').write_text(json.dumps(metadata,indent=2)+'\n',encoding='utf-8')
    figures(output)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,default=Path('docs/benchmarks/convergence_dynamics'))
    main(parser.parse_args().output_dir)
