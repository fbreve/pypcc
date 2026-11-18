"""Strictly source-only P2 technical planning checks, no real data/models."""
import copy
import json
import unittest

from experiments import review_p2_preflight_plan as p2


class P2PlanningTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.plan=json.loads(p2.P2_PLAN.read_text(encoding="utf-8"))
        cls.manifest=json.loads(p2.P1_MANIFEST.read_text(encoding="utf-8"))

    def test_p2_is_only_a_plan_with_exact_approved_p1_parent_sha(self):
        report=p2.review(self.plan,self.manifest)
        self.assertEqual(report["P2_plan_integrity"],"PASS_PLANNING_ONLY")
        self.assertEqual(report["planned_gates"],8)
        self.assertEqual(report["gates_executed"],0)
        self.assertEqual(report["P1_scientific_sha256"],p2.EXPECTED_SHA)
        self.assertEqual(report["real_PCC_fits_executed"],0)
        self.assertIsNone(report["P2_preflight_sha256"])
        self.assertFalse(report["P2_execution_authorized"])

    def test_any_forged_p2_approval_rejected(self):
        for key,value in (
            ("preflight_started",True),
            ("native_binaries_built_for_P2",True),
            ("source_data_loaded_under_P2",True),
            ("label_masks_generated_under_P2",True),
            ("real_knn_graph_generated_under_P2",True),
            ("all_8_gates_passed",True),
            ("user_execution_approval_for_P2_SHA",True),
            ("preflight_SHA256","0"*64),
            ("real_PCC_fit_count",1),
        ):
            obj=copy.deepcopy(self.plan)
            obj["current_checks"][key]=value
            with self.subTest(key=key):
                with self.assertRaises(p2.P2NotReady):
                    p2.review(obj,self.manifest)

    def test_changed_science_sha_gates_and_sealed_state_rejected(self):
        obj=copy.deepcopy(self.plan)
        obj["parent_science"]["P1_scientific_sha256"]="0"*64
        with self.assertRaises(p2.P2NotReady):
            p2.review(obj,self.manifest)
        obj=copy.deepcopy(self.plan)
        obj["technical_steps"][0]["status"]="PASS"
        with self.assertRaises(p2.P2NotReady):
            p2.review(obj,self.manifest)
        obj=copy.deepcopy(self.plan)
        obj["technical_steps"].pop()
        with self.assertRaises(p2.P2NotReady):
            p2.review(obj,self.manifest)
        obj=copy.deepcopy(self.manifest)
        obj["P2_execution_authorized"]=True
        with self.assertRaises(p2.P2NotReady):
            p2.review(self.plan,obj)


if __name__=="__main__":
    unittest.main()
