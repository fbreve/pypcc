"""Offline planning validation; no dataset loading, compilation or fits."""
import json
from pathlib import Path
import tempfile
import unittest

from tools.estimate_second_calibration import BACKENDS, EXCLUDED, estimate

class SecondCalibrationPlanTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.inventory = Path(__file__).resolve().parents[1]/'docs/benchmarks/stopping_patience/inventory.json'
        names = [r['name'].lower() for r in json.loads(self.inventory.read_text())['datasets']
                 if r['name'].lower() not in EXCLUDED]
        names = ['breast_cancer' if n=='breastcancer' else n for n in names]
        self.rows = [dict(case_id=f'{n}-labels10',backend=b,mode='fixed',max_iter=10000,repetitions=6,
            propagation_s=dict(q1=1.,median=2.,q3=3.,maximum=4.),
            initialization_and_dispatch_s=dict(median=.01),postprocessing_s=dict(median=.01))
            for n in names for b in BACKENDS]
        self.save('summary.json',self.rows)
        warms = [dict(case_id=f'{n}-labels10',backend=b,call=c,fit_predict_s=.001)
                 for n in names for b in BACKENDS for c in ('first','warm')]
        (self.root/'warmups.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in warms))
        self.save('preparation.json',dict(fixtures=[dict(case_id=f'{n}-labels10',reused=False,
            dataset_load_s=.1,preprocessing_and_labels_s=.1,graph_build_s=.1) for n in names]))
        self.save('seed_helper.json',dict(first_call_s=.1))

    def save(self,name,data):
        (self.root/name).write_text(json.dumps(data))

    def test_cost_matrix_and_order(self):
        result=estimate(self.root,self.inventory)
        self.assertEqual(result['measured_fits'],48)
        self.assertEqual(result['warmup_fits'],96)
        self.assertEqual(result['measured_iteration_cap'],24000000)
        self.assertEqual(result['measured_particle_update_cap'],288000000)
        self.assertEqual(result['warmup_particle_updates'],2304)
        self.assertAlmostEqual(result['projected_accounted_s']['median'],4800+48*.02+96*.001+16*.3+.1)
        self.assertAlmostEqual(result['conservative_cap_s'],1.5*result['projected_accounted_s']['maximum'])
        self.assertEqual(result['execution_order'][:2],['pendigits','pageblocks'])
        self.assertEqual(len({(t['dataset'],t['backend']) for t in result['tasks']}),48)
        self.assertTrue(all(t['label_seed']==10 and t['measurement_index']==0 for t in result['tasks']))
        self.assertFalse(set(result['execution_order']) & EXCLUDED)
        self.assertEqual(result,estimate(self.root,self.inventory))

    def test_missing_duplicate_or_wrong_baseline_rejected(self):
        for rows in (self.rows[:-1],self.rows+[self.rows[0]],
                     [dict(self.rows[0],max_iter=500000)]+self.rows[1:]):
            with self.subTest(rows=len(rows)):
                self.save('summary.json',rows)
                with self.assertRaises(ValueError):
                    estimate(self.root,self.inventory)

    def test_missing_warmup_or_reused_preparation_rejected(self):
        path=self.root/'warmups.jsonl'
        original=path.read_text()
        path.write_text('\n'.join(original.splitlines()[:-1])+'\n')
        with self.assertRaises(ValueError):
            estimate(self.root,self.inventory)
        path.write_text(original)
        prep=json.loads((self.root/'preparation.json').read_text())
        prep['fixtures'][0]['reused']=True
        self.save('preparation.json',prep)
        with self.assertRaises(ValueError):
            estimate(self.root,self.inventory)

if __name__ == '__main__':
    unittest.main()
