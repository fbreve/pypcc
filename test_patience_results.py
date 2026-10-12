import unittest
from experiments.patience_results import audit_checks, validate_rows


class ResultAuditTests(unittest.TestCase):
    def checks(self):
        return [dict(iteration=1,metric=.5,record=1,counter=0,resets=1,record_gain=.5),
                dict(iteration=11,metric=.5,record=0,counter=1,resets=1,record_gain=0.),
                dict(iteration=21,metric=.4,record=0,counter=2,resets=1,record_gain=0.)]

    def test_independent_first_event_censoring_and_count_difference(self):
        audit = audit_checks(self.checks(), 1, [1.,4.], 30)
        self.assertEqual(audit['events'][1.]['iteration'],21)
        self.assertNotIn(4.,audit['events'])
        row=dict(multiplier=1.,status='stopped',stop_iteration=21,resets=1,
                 correct_nodes=3,final_correct_nodes=4,accuracy_unlabeled=.6,
                 final_accuracy_unlabeled=.8,correct_nodes_difference=-1,
                 accuracy_difference=-.2,iterations_saved=9)
        validate_rows([row],audit,5,30)
        row['accuracy_difference']=0.
        with self.assertRaises(ValueError):
            validate_rows([row],audit,5,30)

    def test_invalid_phase_and_counter_are_rejected(self):
        rows=self.checks();rows[1]['iteration']=12
        with self.assertRaises(ValueError):audit_checks(rows,1,[1.],30)
        rows=self.checks();rows[1]['counter']=2
        with self.assertRaises(ValueError):audit_checks(rows,1,[1.],30)
        with self.assertRaises(ValueError):audit_checks(self.checks()[:2],1,[1.],30)


if __name__=='__main__':unittest.main()
