"""Fast, outcome-free guards for the frozen master performance pilot."""
import unittest
from itertools import permutations

from benchmarks.fixtures import Case
from benchmarks.run import execution_order,plan
from experiments.run_post207_phase_baseline import (
    BACKENDS,DATASETS,SEEDS,FROZEN_MASTER,
    PRODUCTION_FILES,frozen_command,validate_plan,
)


class Post207PhaseBaselineProtocolTests(unittest.TestCase):
    def test_cost_inventory_and_all_six_distinct_backend_orders(self):
        sizes={"wine":(178,13,3),"breast_cancer":(569,30,2),
               "digits":(1797,64,10)}
        fixtures=[
            Case(ds,*sizes[ds],k_nn=10,labels_per_class=3,
                 data_seed=0,label_seed=seed)
            for ds in DATASETS for seed in SEEDS]
        p=plan(fixtures,list(BACKENDS),6,10000)
        p.update(mode="fixed",order_policy="permutations",
                 order_seed=1901)
        validate_plan(p)
        self.assertEqual(p["measured_executions"],108)
        self.assertEqual(p["warmup_executions"],36)
        self.assertEqual(p["particle_update_cap"],16200000)
        for n,case in enumerate(fixtures):
            orders=[
                tuple(execution_order(list(BACKENDS),n,repeat,
                                      "permutations",1901))
                for repeat in range(6)]
            self.assertEqual(len(set(orders)),6)
            self.assertEqual(set(orders),set(permutations(BACKENDS)))

    def test_unchanged_source_guard_and_exact_cli_inputs(self):
        cmd=frozen_command()
        self.assertEqual(FROZEN_MASTER,
                         "c1073ae53ea785031de9e474b678e94f44a2124d")
        self.assertIn("pcc_step.pyx",PRODUCTION_FILES)
        self.assertIn("benchmarks",PRODUCTION_FILES)
        self.assertNotIn("--allow-expensive",cmd)
        for flag,value in (("--mode","fixed"),
                           ("--iterations","10000"),
                           ("--repeats","6"),
                           ("--order-policy","permutations"),
                           ("--order-seed","1901"),
                           ("--propagation-seed","0"),
                           ("--k-nn","10")):
            self.assertEqual(cmd[cmd.index(flag)+1],value)
        self.assertEqual(
            cmd[cmd.index("--datasets")+1:cmd.index("--label-seeds")],
            list(DATASETS))
        self.assertEqual(
            cmd[cmd.index("--label-seeds")+1:cmd.index("--backends")],
            [str(x) for x in SEEDS])


if __name__=="__main__":
    unittest.main()
