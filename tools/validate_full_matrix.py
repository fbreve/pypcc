"""Explicit scheduling/storage validation with mock fixtures/fits, never PCC.

This optional 3,600-record I/O exercise is not an expensive historical benchmark
and is not automatically run in CI. Output must be a new local directory.
"""
import argparse
from contextlib import redirect_stdout
import json
from pathlib import Path
import time
from unittest.mock import patch

from benchmarks.checkpoints import file_inventory
from benchmarks.resumable import full_specification
from tests.test_full_campaign import MockMatrixController
from tests.test_resumable_benchmarks import fake_manifest, fake_measure


def validate(output):
    output.mkdir(parents=True,exist_ok=False)
    calls=[]
    def measured(*args,**kwargs):
        if args[3]>2:
            calls.append((args[1].case_id,args[2]))
        return fake_measure(*args,**kwargs)
    controller=MockMatrixController.create(output/'campaign',full_specification(),
        manifest_fn=fake_manifest,measure_fn=measured)
    plan_bytes=(output/'campaign/plan.json').read_bytes()
    timings=[]
    start=time.perf_counter()
    with patch('benchmarks.resumable.expected_backend'),patch('benchmarks.resumable.seed_backend'),\
            (output/'mock.log').open('x',encoding='utf-8') as log,redirect_stdout(log):
        # Stop inside a permutation, then finish only seed 10.
        controller.execute(stage_seed=10,max_tasks=5)
        old=file_inventory(output/'campaign/results')
        before=time.perf_counter()
        first=controller.execute(stage_seed=10)
        timings.append(dict(stage_seed=10,elapsed_s=time.perf_counter()-before))
        assert len(calls)==360 and first['stages']['11']['status']=='not_started'
        after=file_inventory(output/'campaign/results')
        assert all(after[k]==v for k,v in old.items())
        controller.execute(stage_seed=10)
        assert len(calls)==360
        for seed in range(11,20):
            before=time.perf_counter()
            summary=controller.execute(stage_seed=seed)
            timings.append(dict(stage_seed=seed,elapsed_s=time.perf_counter()-before))
            assert len(calls)==360*(seed-9)
            assert (output/'campaign/plan.json').read_bytes()==plan_bytes
        before=len(calls)
        controller.execute(stage_seed=19)
        assert len(calls)==before
    state=controller.audit()
    assert len(state['results'])==3600 and not state['pending'] and not state['ambiguous']
    assert all(s['status']=='complete' for s in summary['stages'].values())
    assert len(summary['groups'])==600 and all(g['repetitions']==6 for g in summary['groups'])
    record=dict(status='passed',type='mock_scheduler_and_storage_only_no_historical_fits',
        mock_results=3600,fixture_backend_groups=600,stage10_results=360,stable_plan_bytes=True,
        completed_results_preserved=True,terminal_stage_no_repeat=True,
        elapsed_s=time.perf_counter()-start,stage_io_timings=timings,
        warmups=len(list((output/'campaign/sessions').glob('*/warmups/*.json'))),
        note='Mock fit times/iterations/accuracy are placeholders, not measurements or throughput estimates.')
    (output/'validation.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
    (output/'checksums.json').write_text(json.dumps(file_inventory(output),indent=2)+'\n')
    print(json.dumps(record,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    validate(parser.parse_args().output)
