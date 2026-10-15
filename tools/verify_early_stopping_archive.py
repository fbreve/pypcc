"""Portable saved-results verification and statistics. Standard library only; no PCC.

Archive byte checks establish integrity, not authenticity or independent replication.
Historical execution scripts are never imported. Outputs must be new directories.
"""
import argparse
import ast
from collections import Counter, defaultdict
import hashlib
import itertools
import json
import math
from pathlib import Path, PurePosixPath
import statistics
import zipfile


BACKENDS = ('numpy', 'numba', 'cython')
CAMPAIGN = Path('original/.cache/performance/windows-full-early-stopping-01')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def pairs(items):
    out = {}
    for key, value in items:
        require(key not in out, 'duplicate JSON key')
        out[key] = value
    return out


def loads(text):
    return json.loads(text, object_pairs_hook=pairs,
                      parse_constant=lambda x: require(False, 'nonfinite JSON'))


def read(path):
    return loads(Path(path).read_text(encoding='utf-8'))


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def record(path):
    d = read(path)
    require(set(d) == {'schema', 'payload', 'sha256'} and d['schema'] == 1,
            f'invalid envelope: {path}')
    require(digest(d['payload']) == d['sha256'], f'payload checksum: {path}')
    return d['payload']


def safe_name(name):
    p = PurePosixPath(name)
    require(bool(name) and not p.is_absolute() and not any(x in ('', '.', '..') for x in name.split('/'))
            and '\\' not in name and ':' not in name, f'unsafe member: {name}')
    return name


def verify_zip(path, extract=None):
    with zipfile.ZipFile(path) as z:
        names = z.namelist()
        require(len(names) == len(set(names)), 'duplicate ZIP member')
        for info in z.infolist():
            safe_name(info.filename)
            require((info.external_attr >> 16) & 0o170000 != 0o120000, 'ZIP symlink')
        index = loads(z.read('checksums.json').decode('utf-8'))
        require(set(names) == set(index) | {'checksums.json'}, 'member inventory mismatch')
        if extract is not None:
            extract = Path(extract)
            require(not extract.exists(), 'extraction target exists')
            extract.mkdir(parents=True)
        for name in names:
            h = hashlib.sha256()
            target = None
            if extract is not None:
                target = extract / name
                target.parent.mkdir(parents=True, exist_ok=True)
            with z.open(name) as src:
                if target is None:
                    for block in iter(lambda: src.read(1024 * 1024), b''):
                        h.update(block)
                else:
                    with target.open('xb') as dst:
                        for block in iter(lambda: src.read(1024 * 1024), b''):
                            h.update(block)
                            dst.write(block)
            if name != 'checksums.json':
                require(h.hexdigest() == index[name], f'ZIP checksum: {name}')
                if target is not None:
                    require(sha(target) == index[name], f'extraction checksum: {name}')
        return dict(audit='passed',members=len(names),bytes=Path(path).stat().st_size,sha256=sha(path))


def stats(xs):
    ys = sorted(xs)
    def q(f):
        i = (len(ys)-1)*f
        return ys[int(i)] + (ys[min(int(i)+1,len(ys)-1)]-ys[int(i)])*(i-int(i))
    return dict(median=q(.5),q1=q(.25),q3=q(.75),iqr=q(.75)-q(.25),
                minimum=min(ys),maximum=max(ys),total=sum(xs))


def npz_hashes(path):
    """Recompute benchmark array hashes for the archived little-endian C arrays."""
    result = {}
    with zipfile.ZipFile(path) as z:
        require(len(z.namelist()) == len(set(z.namelist())), 'duplicate NPZ member')
        for name in z.namelist():
            safe_name(name)
            with z.open(name) as f:
                require(f.read(6) == b'\x93NUMPY', 'NPY magic')
                version = f.read(2)
                require(version in (b'\x01\x00', b'\x02\x00'), 'NPY version')
                width = 2 if version[0] == 1 else 4
                length = int.from_bytes(f.read(width), 'little')
                require(length < 1024*1024, 'NPY header too large')
                header = ast.literal_eval(f.read(length).decode('latin1'))
                dtype, shape = header['descr'], header['shape']
                require(isinstance(dtype,str) and dtype[:1] in ('<','|') and dtype[1:2] in ('i','u','f','b'), 'unsupported NPY dtype')
                require(not header['fortran_order'], 'non-C archived array')
                expected = math.prod(shape)*int(dtype[2:])
                h = hashlib.sha256(json.dumps([list(shape),dtype]).encode())
                size = 0
                for block in iter(lambda:f.read(1024*1024),b''):
                    size += len(block); h.update(block)
                require(size == expected, 'NPY byte count')
                result[name.removesuffix('.npy')] = h.hexdigest()
    return result


def rows(path):
    return [loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines()]


def analyze(extracted):
    extracted = Path(extracted)
    root = extracted / CAMPAIGN
    plan = record(root/'plan.json')
    require(digest(plan) == '08ee758ffdfc8f11b605c99374955b9d18c087b71ad488cd98aff59a148e7433', 'frozen plan identity')
    original = extracted/'original/docs/benchmarks/early_stopping_full/results'
    expected = read(original/'analysis.json')
    source_root=extracted/'original/.cache/performance/windows-full-early-stopping-01-remaining-provenance/source_files'
    require(all(sha(source_root/name)==h for name,h in plan['identity']['source_hashes'].items()), 'recorded source identity')
    extension=plan['identity']['cython_extension']
    binary_name=extension['path'].replace('\\','/').split('/')[-1]
    require(sha(extracted/'source/recorded-binary'/binary_name)==extension['sha256'], 'recorded binary identity')
    rr = rows(original/'measurements.jsonl')
    require(len(rr) == 3600 and {r['task_id'] for r in rr} == {f't{i:04d}' for i in range(3600)}, 'task grid')
    tasks = {t['task_id']:t for t in plan['tasks']}
    cases = {c['name']+'-labels'+str(c['label_seed']):c for c in plan['cases']}
    events = []; previous = None
    for i,path in enumerate(sorted((root/'events').glob('*.json'))):
        e = record(path)
        require(e['sequence']==i and e['previous_sha256']==previous, 'event chain')
        previous = read(path)['sha256']; events.append(e)
    starts = {e['task_id']:e for e in events if e['kind']=='started'}
    require(Counter(e['task_id'] for e in events if e['kind']=='started') == {k:1 for k in tasks}, 'started attempts')
    require(Counter(e['task_id'] for e in events if e['kind']=='completed') == {k:1 for k in tasks}, 'completed attempts')
    require(not any(e['kind'] in ('failed','retry_authorized','fixture_retry','lock_recovered','reconciled') for e in events), 'unexpected recovery/failure')
    baseline = { (r['case_id'],r['backend'],r['repeat']):r for r in rows(extracted/'baseline/fixed-10000/measurements.jsonl') }
    require(len(baseline)==3600, 'fixed baseline grid')
    fixture_count = 0
    for cid,case in cases.items():
        directory = root/'fixtures'/cid
        seal = record(directory/'seal.json')
        require(seal['plan_sha256']==digest(plan), 'fixture plan')
        require(set(seal['files'])=={'fixture.json','arrays.npz'}, 'fixture seal files')
        require(all(sha(directory/n)==h for n,h in seal['files'].items()), 'fixture sealed bytes')
        meta = read(directory/'fixture.json')
        require(npz_hashes(directory/'arrays.npz')==meta['hashes'], 'fixture array fingerprints')
        require(hashlib.sha256(json.dumps(meta['hashes'],sort_keys=True).encode()).hexdigest()==meta['fixture_sha256'], 'fixture fingerprint')
        fixed = read(extracted/'baseline/fixed-10000/fixtures'/cid/'fixture.json')
        require((meta['hashes'],meta['fixture_sha256'])==(fixed['hashes'],fixed['fixture_sha256']), 'matched fixed fixture')
        require(meta['labeled_nodes']==3*case['classes'] and all(n==3 for n in meta['labeled_class_counts'].values()), 'label class counts')
        fixture_count += 1
    grouped = defaultdict(list)
    for r in rr:
        t = tasks[r['task_id']]; case = cases[r['case_id']]
        require(all(r[k]==t[k] for k in ('case_id','backend','execution_order','order_index')), 'task identity/order')
        require(r['repeat']==t['measurement_index'], 'task repetition')
        saved = record(root/'results'/(r['task_id']+'.json'))
        require(saved['row']==r and saved['plan_sha256']==digest(plan) and saved['attempt_id']==starts[r['task_id']]['attempt_id'], 'result/attempt identity')
        n = r['effective_iterations']; stopped = r['stalled_checks']>r['stop_max']
        require(type(n) is int and 0<n<=500000 and r['max_iter']==500000 and r['mode']=='early_stopping', 'iteration cap')
        patience = max(1,round(case['nodes']/(3*case['classes']*case['k_nn'])*round(plan['parameters']['es_chk']*.1)))
        require(r['stop_max']==patience and r['checks']==(n+9)//10 and 0<=r['stalled_checks']<=r['checks'], 'stopping counters')
        require(r['stop_reason']==('early_stop' if stopped else 'max_iter') and r['reached_max_iter']==(n==500000) and r['stopped_before_cap']==(n<500000), 'stopping semantics')
        require((not stopped and n==500000) or (stopped and (n-1)%10==0), 'stopping boundary')
        require(r['backend_identity']=={'numpy':'pcc_numpy.pcc_propagate_numpy','numba':'pcc_numba.pcc_propagate_numba','cython':'pcc_step.pcc_propagate'}[r['backend']], 'backend identity')
        require(r['propagation_seed_requested']==0 and r['propagation_seed_effective']==(123456789 if r['backend']=='cython' else 0), 'RNG policy')
        require(0<=r['accuracy_unlabeled']<=1 and all(math.isfinite(r[k]) and r[k]>0 for k in ('propagation_s','fit_predict_s')), 'accuracy/timing validity')
        require(r['fixture_sha256']==baseline[r['case_id'],r['backend'],r['repeat']]['fixture_sha256'], 'paired fixture')
        grouped[r['case_id'],r['backend']].append(r)
    groups = []
    for (cid,b),rs in grouped.items():
        require(len(rs)==6 and {r['repeat'] for r in rs}==set(range(6)) and Counter(r['order_index'] for r in rs)=={0:2,1:2,2:2}, 'six repetitions/balance')
        require({tuple(r['execution_order']) for r in rs}==set(itertools.permutations(BACKENDS)), 'six permutations')
        require(all(all(r[k]==rs[0][k] for k in ('state_hashes','effective_iterations','stop_reason','accuracy_unlabeled')) for r in rs), 'repeat state/accuracy/stopping')
        groups.append(dict(dataset=cases[cid]['name'],seed=cases[cid]['label_seed'],backend=b,iterations=rs[0]['effective_iterations'],stop_reason=rs[0]['stop_reason'],accuracy_percent=100*rs[0]['accuracy_unlabeled'],propagation_s=stats([r['propagation_s'] for r in rs]),fit_predict_s=stats([r['fit_predict_s'] for r in rs])))
    require(len(groups)==600, 'fixture/backend grid')
    for g in groups:
        ref = next(x for x in expected['groups'] if (x['dataset'],x['seed'],x['backend'])==(g['dataset'],g['seed'],g['backend']))
        for field in ('propagation_s','fit_predict_s'):
            require(all(math.isclose(v,ref[field][k],rel_tol=1e-10,abs_tol=1e-10) for k,v in g[field].items()), 'timing summary reproducibility')
        require(all(g[k]==ref[k] for k in ('iterations','stop_reason','accuracy_percent')), 'scientific summary reproducibility')
    dataset_summary=[]; matched=[]
    for d in dict.fromkeys(c['name'] for c in plan['cases']):
        for b in BACKENDS:
            gs=[g for g in groups if g['dataset']==d and g['backend']==b]
            acc=[g['accuracy_percent'] for g in gs]
            ref=next(x for x in expected['dataset_backend_summary'] if x['dataset']==d and x['backend']==b)
            require(math.isclose(statistics.mean(acc),ref['accuracy_percent']['mean'],abs_tol=1e-10) and math.isclose(statistics.stdev(acc),ref['accuracy_percent']['sample_sd'],abs_tol=1e-10), 'accuracy summary reproducibility')
            dataset_summary.append(dict(dataset=d,backend=b,accuracy_mean_percent=statistics.mean(acc),accuracy_sd_percent=statistics.stdev(acc),accuracy_range_percent=[min(acc),max(acc)],censored_masks=sum(g['stop_reason']=='max_iter' for g in gs)))
            delta=[g['accuracy_percent']-100*baseline[d+'-labels'+str(g['seed']),b,0]['accuracy_unlabeled'] for g in gs]
            matched.append(dict(dataset=d,backend=b,mean_delta_pp=statistics.mean(delta),minimum_delta_pp=min(delta),maximum_delta_pp=max(delta),improved=sum(x>0 for x in delta),deteriorated=sum(x<0 for x in delta),unchanged=sum(x==0 for x in delta)))
    sessions = list((root/'sessions').iterdir()); warmups=[]
    require(len(sessions)==10, 'session count')
    for directory in sessions:
        session=record(directory/'session.json'); end=record(directory/'end.json')
        require(session['identity']==plan['identity'] and end['measured_tasks_completed']==360, 'session identity/completion')
        ws=[record(p) for p in (directory/'warmups').glob('*.json')];require(len(ws)==120, 'stage warmups');warmups+=ws
    require(len(warmups)==1200, 'warmup count')
    for field in ('propagation_s','fit_predict_s'):
        require(math.isclose(sum(r[field] for r in rr),expected['propagation_total_s' if field=='propagation_s' else 'fit_total_s'],rel_tol=1e-12), 'timing totals')
    require(sum(r['effective_iterations'] for r in rr)==expected['iterations'] and sum(r['stop_reason']=='max_iter' for r in rr)==expected['censored'], 'convergence totals')
    # Both calibration analyses are recomputed separately; never pooled with the full campaign.
    calibrations=[]
    for name,count in (('early_stopping_pilot',72),('early_stopping_second_calibration',48)):
        d=extracted/'original/docs/benchmarks'/name/'results'
        rs=rows(d/('raw/measurements.jsonl' if count==72 else 'measurements.jsonl'))
        saved=read(d/'analysis.json'); require(len(rs)==count, 'calibration count')
        prop=sum(r['propagation_s'] for r in rs);fit=sum(r['fit_predict_s'] for r in rs)
        require(math.isclose(prop,saved['measured_propagation_s' if count==72 else 'propagation_s'],rel_tol=1e-12) and math.isclose(fit,saved['measured_fit_s' if count==72 else 'fit_s'],rel_tol=1e-12), 'calibration totals')
        cg=defaultdict(list)
        for r in rs:cg[r['case_id'],r['backend']].append(r)
        calibration_groups=[]
        for (cid,b),members in cg.items():
            require(len(members)==(3 if count==72 else 1), 'calibration repetitions')
            require(all(all(r[k]==members[0][k] for k in ('state_hashes','effective_iterations','stop_reason','accuracy_unlabeled')) for r in members), 'calibration repeated states')
            dataset,seed=cid.rsplit('-labels',1)
            ref=next(x for x in saved['groups' if count==72 else 'rows'] if x['dataset']==dataset and x['seed']==int(seed) and x['backend']==b)
            require(ref['iterations']==members[0]['effective_iterations'] and ref['reason']==members[0]['stop_reason'], 'calibration stopping reproduction')
            ps=stats([r['propagation_s'] for r in members]);fs=stats([r['fit_predict_s'] for r in members])
            if count==72:
                for prefix,s in [('propagation_s',ps),('fit_predict_s',fs)]:
                    for key,suffix in [('median','median'),('iqr','iqr'),('minimum','min'),('maximum','max')]:
                        require(math.isclose(s[key],ref[prefix+'_'+suffix],rel_tol=1e-10,abs_tol=1e-10),'pilot timing reproduction')
            else:
                require(math.isclose(ps['median'],ref['propagation_s'],abs_tol=1e-10) and math.isclose(fs['median'],ref['fit_s'],abs_tol=1e-10),'calibration timing reproduction')
            calibration_groups.append(dict(case_id=cid,backend=b,iterations=members[0]['effective_iterations'],stop_reason=members[0]['stop_reason'],accuracy_percent=100*members[0]['accuracy_unlabeled'],propagation_s=ps,fit_predict_s=fs))
        calibrations.append(dict(name=name,measurements=count,propagation_s=prop,fit_predict_s=fit,censored=sum(r['stop_reason']=='max_iter' for r in rs),groups=calibration_groups))
    return dict(audit='passed',measurements=3600,warmups=1200,fixtures=fixture_count,iterations=sum(r['effective_iterations'] for r in rr),early_stops=sum(r['stop_reason']=='early_stop' for r in rr),censored=sum(r['stop_reason']=='max_iter' for r in rr),propagation_s=sum(r['propagation_s'] for r in rr),fit_predict_s=sum(r['fit_predict_s'] for r in rr),groups=groups,dataset_backend_summary=dataset_summary,matched_fixed=matched,calibrations=calibrations,note='Saved-byte, fixture-array and statistical verification; no PCC execution, graph construction or current-environment equivalence claim.')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--archive',type=Path)
    p.add_argument('--extract',type=Path)
    p.add_argument('--root',type=Path)
    p.add_argument('--output',type=Path)
    a=p.parse_args()
    require(bool(a.archive) != bool(a.root), 'choose --archive or --root')
    if a.archive:
        result=verify_zip(a.archive,a.extract)
        if a.extract and a.output:
            analysis=analyze(a.extract)
        else:
            require(a.output is None, '--output requires --extract');analysis=None
    else:
        require(a.extract is None and a.output is not None, '--root requires --output')
        analysis=analyze(a.root);result={k:analysis[k] for k in ('audit','measurements','warmups','fixtures','censored')}
    if analysis is not None:
        a.output.mkdir(parents=True,exist_ok=False)
        (a.output/'analysis.json').write_text(json.dumps(analysis,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps(result))


if __name__=='__main__':
    main()
