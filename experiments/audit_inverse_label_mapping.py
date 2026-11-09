"""Exact original Git inverse-label block timing and whole-PCC native state parity."""
import gc
import hashlib
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time
import tracemalloc
import types

import numpy as np
from sklearn.datasets import load_iris, load_wine, load_breast_cancer, load_digits
from sklearn.preprocessing import StandardScaler
from pcc import ParticleCompetitionAndCooperation as Candidate
from pcc_graph import build_knn_graph

BASE="15695279884a82d0c71b4cb395187a04a1c8a698"
ALL_BACKENDS=("numpy","numba","cython")
DATASETS=(("Iris",load_iris),("Wine",load_wine),
          ("Breast Cancer",load_breast_cancer),("Digits",load_digits))
LABEL_SEEDS=(38027,38053)
CASES=((150,3,False),(569,2,False),(1023,10,False),
       (1024,10,False),(1797,10,False),(30000,2,False),
       (30000,7,False),(30000,8,False),(30000,10,False),
       (30000,50,False),(100000,2,False),(100000,10,False),
       (100000,50,False),(100000,50,True))
OLD="""            # Unmap labels before returning
            final_labels = np.full(self.node.label.shape, -1, dtype=np.int64)
            for idx, lbl in self.idx_to_label.items():
                final_labels[self.node.label == idx] = lbl
            return final_labels"""
NEW="""            # Unmap labels before returning
            final_labels = np.full(self.node.label.shape, -1, dtype=np.int64)
            if self.c < 8 or self.node.label.size < 1024:
                for idx, lbl in self.idx_to_label.items():
                    final_labels[self.node.label == idx] = lbl
            else:
                valid = (self.node.label >= 0) & (self.node.label < self.c)
                final_labels[valid] = self.unique_labels[self.node.label[valid]]
            return final_labels"""


def array_sha(value):
    a=np.asarray(value)
    return hashlib.sha256(json.dumps([a.shape,a.dtype.str]).encode()
                          +np.ascontiguousarray(a).tobytes()).hexdigest()


def reference_class_and_blocks():
    original_bytes=subprocess.check_output(["git","show",BASE+":pcc.py"])
    original=original_bytes.decode("utf-8").replace("\r\n","\n")
    current=Path("pcc.py").read_text(encoding="utf-8").replace("\r\n","\n")
    if original.count(OLD)!=1 or original.replace(OLD,NEW)!=current:
        raise AssertionError("changed source outside one exact inverse-label hunk")
    module=types.ModuleType("original_pcc_inverse_labels")
    sys.modules[module.__name__]=module
    exec(compile(original,BASE+":pcc.py","exec"),module.__dict__)
    funcs=[]
    for code,name in ((OLD,"original"),(NEW,"candidate")):
        namespace={"np":np}
        exec(compile("def inverse_block(self):\n"+code+"\n",name,"exec"),namespace)
        funcs.append(namespace["inverse_block"])
    return module.ParticleCompetitionAndCooperation,funcs,hashlib.sha256(original_bytes).hexdigest()


def fixture(n,c,with_invalid):
    rng=np.random.default_rng(38011+n+c+int(with_invalid))
    original_labels=np.arange(c,dtype=np.int64)*271 - 1000000
    values=rng.integers(0,c,size=n,dtype=np.int64)
    if with_invalid:
        bad=np.array([-1,-2,c,c+1,np.iinfo(np.int64).min,
                      np.iinfo(np.int64).max],dtype=np.int64)
        rows=rng.choice(n,size=n//20,replace=False)
        values[rows]=bad[np.arange(len(rows))%len(bad)]
        # Make the internal node.label a non-contiguous view in this fixture.
        backing=np.empty(n*2,dtype=np.int64)
        backing[::2]=values
        values=backing[::2]
    class_map={idx:lbl for idx,lbl in enumerate(original_labels)}
    return values,original_labels,class_map


def new_box(values,original_labels,class_map):
    return types.SimpleNamespace(
        c=len(original_labels),unique_labels=original_labels,
        idx_to_label=class_map,node=types.SimpleNamespace(label=values))


def assert_output(result,values,original_labels,oracle_hash):
    if result.dtype!=np.int64 or result.shape!=values.shape:
        raise AssertionError("output int64 shape/dtype mismatch")
    if not result.flags.c_contiguous or not result.flags.writeable:
        raise AssertionError("output ownership/layout changed")
    if np.shares_memory(result,values):
        raise AssertionError("output aliases internal node labels")
    if array_sha(result)!=oracle_hash:
        raise AssertionError("exact decoded class labels differ")
    invalid=(values<0)|(values>=len(original_labels))
    if np.any(result[invalid]!=-1):
        raise AssertionError("out-of-range index no longer returns unknown sentinel")


def timer(func,box):
    started=time.perf_counter_ns()
    result=func(box)
    elapsed=(time.perf_counter_ns()-started)/1e6
    return elapsed,result


def measured_peak(func,box):
    gc.collect()
    tracemalloc.start()
    try:
        result=func(box)
        _,peak=tracemalloc.get_traced_memory()
        return peak,result
    finally:
        tracemalloc.stop()


def micro(blocks):
    rows=[]
    for n,c,invalid in CASES:
        values,original_labels,class_map=fixture(n,c,invalid)
        boxes=[new_box(values,original_labels,class_map) for _ in blocks]
        original=blocks[0](boxes[0])
        expected=np.full(n,-1,dtype=np.int64)
        for idx,label in class_map.items():
            expected[values==idx]=label
        if not np.array_equal(original,expected):
            raise AssertionError("original block is not historical output")
        oracle_hash=array_sha(expected)
        assert_output(original,values,original_labels,oracle_hash)
        for index in (0,1):
            for _ in range(3):
                _,r=timer(blocks[index],boxes[index])
                assert_output(r,values,original_labels,oracle_hash)
        times=[]
        for i in range(25):
            order=(0,1) if i%2 else (1,0)
            pair=[None,None]
            for which in order:
                value,output=timer(blocks[which],boxes[which])
                assert_output(output,values,original_labels,oracle_hash)
                pair[which]=value
            times.append(pair)
        peaks=[]
        for i in range(5):
            order=(0,1) if i%2 else (1,0)
            pair=[None,None]
            for which in order:
                value,output=measured_peak(blocks[which],boxes[which])
                assert_output(output,values,original_labels,oracle_hash)
                pair[which]=value
            peaks.append(pair)
        percent=100*(1-statistics.median([b/a for a,b in times]))
        print(f"{n}x{c} invalid={invalid}: inverse CPU {percent:+.2f}%; "
              f"tracked peaks {statistics.median(v[0] for v in peaks):.0f}/"
              f"{statistics.median(v[1] for v in peaks):.0f} B",flush=True)
        rows.append(dict(nodes=n,classes=c,invalid=invalid,
                         internal_sha=array_sha(values),
                         source_labels_sha=array_sha(original_labels),
                         decoded_sha=oracle_hash,output_c_contiguous=True,
                         output_writable=True,
                         timings_ms=times,traced_peak_bytes=peaks))
    return rows


def fit_with_metrics(cls, neighbors, degrees, labels, backend, early_stop, seed):
    m = cls(impl=backend)
    m.set_graph(neighbors, degrees, k_nn=10)
    underlying = m._get_backend_fn()
    expected = {"numpy": "pcc_numpy", "numba": "pcc_numba", "cython": "pcc_step"}[backend]
    if underlying.__module__ != expected:
        raise RuntimeError("native fallback " + backend)
    phase = {}
    def timed_backend(*args, **kwargs):
        kwargs["return_info"] = True
        entered = time.perf_counter_ns()
        info = underlying(*args, **kwargs)
        exited = time.perf_counter_ns()
        phase.update(info)
        phase["entered_ns"] = entered
        phase["exited_ns"] = exited
        return None
    # Crucial: old benchmark runner.measure once failed to preserve this name (#203 fixed).
    # __genNodes tests this exact function name to choose C layout for Numba >=8 classes.
    timed_backend.__name__ = underlying.__name__
    m._get_backend_fn = lambda: timed_backend
    start = time.perf_counter_ns()
    predictions = m.fit_predict(labels, max_iter=1000, early_stop=early_stop,
                                es_chk=10, seed=seed)
    end = time.perf_counter_ns()
    if "entered_ns" not in phase:
        raise AssertionError("timed backend not invoked")
    expected_order = ("C" if backend == "numba" and len(np.unique(labels[labels != -1])) >= 8
                      else "F")
    flag = (m.node.dominance.flags.c_contiguous if expected_order == "C"
            else m.node.dominance.flags.f_contiguous)
    if not flag:
        raise AssertionError("incorrect native dominance C/F layout")
    arrays = {
        "predictions": predictions,
        "dominance": m.node.dominance,
        "positions": m.part.curnode,
        "strengths": m.part.strength,
        "distances": m.part.dist_table,
        "owndeg": m.owndeg,
    }
    hashes = {k: array_sha(v) for k,v in arrays.items()}
    rng_sha = None
    if backend == "numpy":
        state = np.random.get_state()
        rng_sha = hashlib.sha256(state[1].tobytes()
                                 + repr((state[0],state[2],state[3],state[4])).encode()).hexdigest()
    timings = dict(fit_predict_s=(end-start)/1e9,
                   initialization_and_dispatch_s=(phase["entered_ns"]-start)/1e9,
                   propagation_s=(phase["exited_ns"]-phase["entered_ns"])/1e9,
                   postprocessing_s=(end-phase["exited_ns"])/1e9)
    return dict(state_hashes=hashes, info={k:v for k,v in phase.items()
                                         if k not in ("entered_ns", "exited_ns")},
                backend=backend, backend_identity=f"{underlying.__module__}.{underlying.__name__}",
                layout=expected_order, rng_sha=rng_sha, timings=timings)


def real_native(reference):
    rows=[]
    for dataset, loader in DATASETS:
        raw, truth=loader(return_X_y=True)
        x=StandardScaler().fit_transform(raw)
        neighbors, degrees=build_knn_graph(x, k_nn=10, n_jobs=1)
        graph_sha=(array_sha(neighbors), array_sha(degrees))
        for label_seed in LABEL_SEEDS:
            rng=np.random.default_rng(label_seed)
            labels=np.full(len(truth), -1, dtype=np.int64)
            for cls in np.unique(truth):
                homes=rng.choice(np.flatnonzero(truth==cls), size=3, replace=False)
                labels[homes]=int(cls)
            label_sha=array_sha(labels)
            for backend in ALL_BACKENDS:
                # Precompile identical native argument shapes before timed comparisons.
                if backend=="numba":
                    for cls in (reference, Candidate):
                        fit_with_metrics(cls, neighbors, degrees, labels, backend, False, 38059)
                for early_stop in (False, True):
                    order=(0,1) if label_seed==LABEL_SEEDS[0] else (1,0)
                    pair=[None,None]
                    for idx in order:
                        pair[idx]=fit_with_metrics((reference,Candidate)[idx],
                                                   neighbors,degrees,labels,
                                                   backend,early_stop,38059)
                    a,b=pair
                    if any(a[field]!=b[field] for field in
                           ("state_hashes", "info", "backend_identity", "layout", "rng_sha")):
                        raise AssertionError("full native parity FAIL "
                                             +repr((dataset,label_seed,backend,early_stop)))
                    rows.append(dict(dataset=dataset,label_seed=label_seed,
                                     label_sha=label_sha,graph_sha=graph_sha,
                                     backend=backend,early_stop=early_stop,
                                     state_hashes=a["state_hashes"],
                                     info=a["info"],backend_identity=a["backend_identity"],
                                     layout=a["layout"],rng_sha=a["rng_sha"],
                                     reference_timings=a["timings"],
                                     candidate_timings=b["timings"]))
        print(f"{dataset}: completed {2*3*2} exact native state-pairs",flush=True)
    return rows



def main():
    original,blocks,sha=reference_class_and_blocks()
    rows=dict(schema=1,base_commit=BASE,original_pcc_sha256=sha,
              head_commit=subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),
              os=sys.platform,host=platform.platform(),cpu=platform.processor(),
              python=sys.version,numpy=np.__version__,
              micro=micro(blocks),real=real_native(original))
    output=Path(os.environ.get("PCC_INVERSE_RESULT","inverse_label_result.json"))
    output.write_text(json.dumps(rows,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    print("Finished exact inverse mapping original-vs-candidate:",
          len(rows["micro"]),"micro groups and",len(rows["real"]),
          "native full-PCC pairs",flush=True)


if __name__=="__main__":
    main()
