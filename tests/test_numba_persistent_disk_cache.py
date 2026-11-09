"""Production Numba JIT disk cache: actual new-interpreter replay on both OS.

No change to Numba particle, graph, RNG, stopping or public source contracts.
"""
import ast
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import pcc_numba


CACHE_FUNCTIONS={
    "_seed_numba_rng",
    "_pcc_step_numba_sequential",
    "_pcc_calc_mmpot_sequential",
    "_pcc_propagate_numba_sequential",
}

# Run as completely independent fresh Python programs: merely repeating a
# call within one interpreter never proves disk cache support.
CHILD_PROGRAM=r'''
import hashlib
import json
import numpy as np
from numba import njit
from pcc import ParticleCompetitionAndCooperation as PCC

@njit
def peek_numba_rng():
    return np.random.random()

n=12
degrees=np.full(n,2,dtype=np.int64)
neighbors=np.empty((n,2),dtype=np.int64)
for i in range(n):
    neighbors[i,:]=((i-1)%n,(i+1)%n)
labels=np.full(n,-1,dtype=np.int64)
labels[0]=0
labels[n//2]=1
model=PCC(impl="numba")
model.set_graph(neighbors,degrees,k_nn=2)
pred=model.fit_predict(labels,seed=0,max_iter=60,early_stop=True,es_chk=2000)
raw={
    "predictions":pred,
    "dominance":model.node.dominance,
    "positions":model.part.curnode,
    "strengths":model.part.strength,
    "distances":model.part.dist_table,
    "owndeg":model.owndeg,
}
hashes={
    name:hashlib.sha256(
        json.dumps([list(arr.shape),arr.dtype.str]).encode()+
        np.ascontiguousarray(arr).tobytes()
    ).hexdigest()
    for name,arr in raw.items()
}
print("PCC_NATIVE_CACHE_REPLAY="+json.dumps({
    "states":hashes,
    "rng_next":float(peek_numba_rng()),
    "dominance_layout":"C" if model.node.dominance.flags.c_contiguous else "F",
    "iteration_cap":60
},sort_keys=True))
'''


class NumbaPersistentDiskCacheTests(unittest.TestCase):
    def test_only_original_native_numba_dispatchers_are_cached(self):
        source=Path(pcc_numba.__file__).read_text(encoding="utf-8")
        module=ast.parse(source)
        decorated={}
        for obj in module.body:
            if isinstance(obj,(ast.FunctionDef,ast.AsyncFunctionDef)):
                for decorator in obj.decorator_list:
                    if (isinstance(decorator,ast.Call)
                            and isinstance(decorator.func,ast.Name)
                            and decorator.func.id=="njit"):
                        decorated[obj.name]=decorator
        self.assertEqual(set(decorated),CACHE_FUNCTIONS)
        for name,decorator in decorated.items():
            self.assertEqual(len(decorator.args),0,name)
            self.assertEqual(len(decorator.keywords),1,name)
            self.assertEqual(decorator.keywords[0].arg,"cache",name)
            self.assertIsInstance(decorator.keywords[0].value,ast.Constant)
            self.assertIs(decorator.keywords[0].value.value,True,name)

    def test_real_disk_cache_loads_in_a_second_python_process(self):
        with tempfile.TemporaryDirectory(prefix="pypcc_numba_cache_test_") as temporary:
            environment=os.environ.copy()
            environment["NUMBA_CACHE_DIR"]=str(Path(temporary).resolve())
            environment["NUMBA_DEBUG_CACHE"]="1"
            for name in ("OPENBLAS_NUM_THREADS","OMP_NUM_THREADS",
                         "MKL_NUM_THREADS","NUMBA_NUM_THREADS"):
                environment[name]="1"

            def fresh_python():
                proc=subprocess.run(
                    [sys.executable,"-c",CHILD_PROGRAM],
                    env=environment,capture_output=True,text=True,
                    timeout=100)
                self.assertEqual(proc.returncode,0,
                                 proc.stdout[-3000:]+"\n"+proc.stderr[-5000:])
                logs=proc.stdout+"\n"+proc.stderr
                matches=[
                    line for line in logs.splitlines()
                    if line.startswith("PCC_NATIVE_CACHE_REPLAY=")
                ]
                self.assertEqual(len(matches),1,logs[-3000:])
                return json.loads(matches[0].split("=",1)[1]),logs

            first,first_logs=fresh_python()
            second,second_logs=fresh_python()
            self.assertEqual(first,second)
            self.assertEqual(first["iteration_cap"],60)
            self.assertEqual(first["dominance_layout"],"F")
            self.assertEqual(len(first["states"]),6)
            original_source_cache_logs=[
                line for line in first_logs.splitlines()
                if "[cache]" in line and "pcc_numba" in line
            ]
            fresh_cache_logs=[
                line for line in second_logs.splitlines()
                if "[cache]" in line and "pcc_numba" in line
            ]
            self.assertTrue(any("data saved" in line
                                for line in original_source_cache_logs),
                            "first Numba use did not compile/store cached machine code:\n"
                            +first_logs[-3500:])
            self.assertTrue(any("data loaded" in line
                                for line in fresh_cache_logs),
                            "second independent Python process did not load Numba disk code:\n"
                            +second_logs[-3500:])
            saved=[
                p for p in Path(temporary).rglob("*")
                if p.is_file() and p.suffix in (".nbi",".nbc")
            ]
            self.assertGreater(len(saved),0)
            self.assertLess(sum(p.stat().st_size for p in saved),32*1024*1024)


if __name__=="__main__":
    unittest.main()
