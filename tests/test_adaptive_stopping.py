"""Synthetic boundaries for optional retrospective observers; no PCC execution."""
import unittest
from experiments.adaptive_stopping import original_event, record_stagnation, classification_event


class AdaptiveTests(unittest.TestCase):
    def checks(self, gains):
        return [dict(iteration=1+10*j,record_gain=g,counter=j) for j,g in enumerate(gains)]

    def test_original_patience_strict_boundary_and_censor(self):
        rows=self.checks([0]*4)
        self.assertEqual(original_event(rows,1)['iteration'],21)
        self.assertIsNone(original_event(rows,3))

    def test_record_window_left_open_budget_inclusive(self):
        rows=self.checks([1,.1,.2,0])
        self.assertEqual(record_stagnation(rows,20,.31,0)['iteration'],21)
        self.assertEqual(record_stagnation(rows,20,.2,0)['iteration'],31)
        self.assertIsNone(record_stagnation(rows,20,.19,0))

    def test_minimum_iteration_and_complete_history(self):
        rows=self.checks([0]*5)
        self.assertEqual(record_stagnation(rows,20,0,25)['iteration'],31)
        self.assertIsNone(record_stagnation(rows[:2],20,0,0))

    def test_future_values_cannot_change_first_event(self):
        prefix=self.checks([1,0,0])
        result=record_stagnation(prefix,20,0,0)
        future=self.checks([1,0,0,1e3,1e3])
        self.assertEqual(record_stagnation(future,20,0,0),result)

    def test_small_gains_are_accumulated(self):
        rows=self.checks([1,.0003,.0003])
        self.assertIsNone(record_stagnation(rows,20,.0005,0))

    def samples(self, pairs):
        return [dict(iteration=t,changed_nodes=h) for t,h in pairs]

    def test_intermediate_change_and_reversal_are_both_counted(self):
        rows=self.samples([(1000,0),(1501,1),(1801,1),(2000,0),(3000,0)])
        checks=[dict(iteration=t,counter=10) for t in (991,1991,2991)]
        self.assertEqual(classification_event(rows,checks,1000,0,2000,10,1)['iteration'],3000)
        event=classification_event(rows,checks,1000,.2,2000,10,1)
        self.assertEqual(event['transition_count'],2)
        self.assertEqual(event['intermediate_observations'],2)
        self.assertEqual(event['iteration'],2000)
        self.assertEqual(event['counter_age'],9)

    def test_left_endpoint_excluded_without_duplicate_counts(self):
        rows=self.samples([(1000,10),(1501,1),(2000,1),(3000,0)])
        checks=[dict(iteration=t,counter=2) for t in (991,1991,2991)]
        event=classification_event(rows,checks,1000,.2,2000,10,1)
        self.assertEqual(event['transition_count'],2)
        self.assertEqual(event['window_start'],1000)

    def test_primary_secondary_schedules_and_inclusive_budget(self):
        rows=self.samples([(1000,0),(2000,0),(2501,0),(3000,0)])
        checks=[dict(iteration=t,counter=2) for t in (991,1991,2501,2991)]
        self.assertEqual(classification_event(rows,checks,1000,0,2500,10,1)['iteration'],3000)
        self.assertEqual(classification_event(rows,checks,1000,0,2500,10,1,regular=False)['iteration'],2501)

    def test_joint_counter_reset_not_latched_and_no_future_counter(self):
        rows=self.samples([(1000,0),(2000,0),(3000,0)])
        checks=[dict(iteration=991,counter=4),dict(iteration=1991,counter=0),
                dict(iteration=2001,counter=4),dict(iteration=2991,counter=1)]
        self.assertIsNone(classification_event(rows,checks,1000,0,2000,10,1,joint=True))
        checks[-1]['counter']=2
        self.assertEqual(classification_event(rows,checks,1000,0,2000,10,1,joint=True)['iteration'],3000)

    def test_future_classification_accuracy_and_gains_do_not_leak(self):
        rows=self.samples([(1000,0),(1501,1),(2000,0)])
        checks=[dict(iteration=991,counter=0),dict(iteration=1991,counter=2)]
        event=classification_event(rows,checks,1000,.1,2000,10,1,joint=True)
        rows += self.samples([(2501,10),(3000,10)])
        for r in rows:r['accuracy_unlabeled']=999
        checks += [dict(iteration=2001,counter=0)]
        self.assertEqual(classification_event(rows,checks,1000,.1,2000,10,1,joint=True),event)

    def test_complete_classification_window_and_censor(self):
        rows=self.samples([(1000,0),(2000,0)])
        checks=[dict(iteration=1991,counter=0)]
        self.assertIsNone(classification_event(rows,checks,2000,0,2000,10,1))
        self.assertIsNone(classification_event(rows,checks,1000,0,2000,10,1,joint=True))


if __name__=='__main__':
    unittest.main()
