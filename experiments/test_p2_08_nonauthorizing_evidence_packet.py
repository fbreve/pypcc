"""P2-08 review-only protocol tests, entirely synthetic metadata, no PCC."""
import copy
import re
import unittest

from experiments import p2_08_nonauthorizing_evidence_packet as gate
from experiments import p2_07c_consolidated_execution_envelope_no_fit as prior


def synthetic_parity(platform="Linux"):
    return {
        "schema":"pypcc-P2-06-synthetic-pinned-backend-parity/v1",
        "P1_scientific_sha256":gate.P1_SHA,
        "original_synthetic_suite_test_count":21,
        "test_errors":0,"test_failures":0,
        "P2_06_synthetic_parity_candidate_pass":True,
        "P2_06_full_execution_gate_pass":False,
        "Numba_native_IR_and_ABI_frozen_for_final_P2":False,
        "real_source_datasets_loaded":0,
        "real_PCC_fits_executed":0,
        "P2_execution_preflight_sha256":None,
        "P2_execution_authorized":False,
        "synthetic_PCC_executed_in_tests":True,
        "source_observer_commit_heads":{
            "numpy":"6d908e78878c4922674277e3313dbfe7f7924146",
            "numba":"4c8e6df4e01d7a4dfc4efc31eb73db04d6e212e5",
            "cython":"af34f50ebabb0a7204d3f30679adaf3f00169742",
        },
        "exact_native_binary_used":{"native_sha":"f"*64},
        "platform":platform,
    }


class P208EvidencePacketProtocolTests(unittest.TestCase):
    def setUp(self):
        self.registry=gate.read_json(gate.REGISTRY)[0]
        self.p1=prior.read_approved_P1()

    def test_exact_eleven_archived_safe_inputs_and_non_authorizing_dossier(self):
        items=gate.validate_registry(self.registry)
        self.assertEqual(len(items),11)
        self.assertEqual(len(gate.EXPECTED_ARTIFACTS),11)
        self.assertEqual(len(set(x["artifact_id"] for x in items)),11)
        self.assertTrue(all(re.fullmatch(r"[0-9a-f]{64}",x["archive_zip_sha256"])
                            for x in items))
        self.assertEqual(self.registry["P1_scientific_sha256"],gate.P1_SHA)
        self.assertIsNone(self.registry["final_approved_P2_execution_preflight_sha256"])
        self.assertFalse(self.registry["P2_execution_authorized"])
        self.assertEqual(self.registry["PCC_real_source_fits_executed"],0)

    def test_reject_forged_approval_and_missing_artifact_identity(self):
        mutation_cases=[
            (("P2_execution_authorized",),True),
            (("final_P2_08_gate_pass",),True),
            (("explicit_maintainer_approval_of_distinct_exact_P2_sha",),True),
            (("final_approved_P2_execution_preflight_sha256",),"f"*64),
            (("PCC_real_source_fits_executed",),1),
            (("P1_scientific_sha256",),"1"*64),
            (("evidence",0,"artifact_id"),99),
            (("evidence",0,"run_id"),1),
            (("evidence",0,"extracted_destination"),"../x"),
        ]
        for keys,value in mutation_cases:
            with self.subTest(mutated_path=keys):
                cloned=copy.deepcopy(self.registry)
                destination=cloned
                for key in keys[:-1]:
                    destination=destination[key]
                destination[keys[-1]]=value
                with self.assertRaises(gate.ReviewPacketBlocked):
                    gate.validate_registry(cloned)

    def test_original_8_gate_order_and_exact_p1_science_dont_grant_P2(self):
        gate.validate_original_p2_plan(gate.read_json(gate.PLAN)[0])
        self.assertEqual(gate.sha256(gate.canonical(self.p1)),gate.P1_SHA)
        self.assertEqual(len(gate.STEPS),8)
        self.assertEqual(self.p1["fit"]["total_proposed_fits"],72)
        self.assertEqual(self.p1["fit"]["milestones"],
                         [0,1,10,100,1000,5000,25000,50000])
        self.assertFalse(self.p1["approvals"]["P2_execution_approved"])
        self.assertTrue(gate.BLOCKERS)
        self.assertTrue(all(isinstance(x,str) and len(x)>30
                            for x in gate.BLOCKERS))

    def test_synthetic_original_parity_only_not_full_native_gate(self):
        left=synthetic_parity()
        right=synthetic_parity("Windows")
        self.assertEqual(len(gate.parity_gate(left,right)),2)
        for key, value in [
            ("P2_06_synthetic_parity_candidate_pass",False),
            ("P2_06_full_execution_gate_pass",True),
            ("Numba_native_IR_and_ABI_frozen_for_final_P2",True),
            ("real_source_datasets_loaded",1),
            ("real_PCC_fits_executed",1),
            ("P2_execution_authorized",True),
            ("original_synthetic_suite_test_count",20),
        ]:
            with self.subTest(key=key):
                broken=copy.deepcopy(left)
                broken[key]=value
                with self.assertRaises(gate.ReviewPacketBlocked):
                    gate.parity_gate(broken,right)
        wrong=synthetic_parity("Windows")
        wrong["source_observer_commit_heads"]["cython"]="x"
        with self.assertRaises(gate.ReviewPacketBlocked):
            gate.parity_gate(left,wrong)

    def test_false_GO_rejected_at_simple_evidence_guard(self):
        data={"P2_execution_authorized":False,"real_PCC_fits_executed":0}
        gate.no_approved_execution(data,label="synthetic")
        for update in ({"P2_execution_authorized":True},
                       {"real_PCC_fits_executed":72}):
            with self.assertRaises(gate.ReviewPacketBlocked):
                gate.no_approved_execution({**data,**update},label="forged")

    def test_candidate_resource_policy_is_not_approval(self):
        policy=prior.load(prior.POLICY)
        limits=prior.validate_policy(policy,self.p1)
        self.assertEqual(limits["max_one_fit_peak_process_resident_bytes"],
                         2*1024**3)
        self.assertEqual(policy["fit_schedule"]["max_parallel_fits"],1)
        self.assertFalse(policy["approved_execution_host_and_resource_limits"])
        self.assertIsNone(policy["P2_08_technical_manifest_sha256"])
        self.assertFalse(policy["P2_execution_authorized"])
        self.assertEqual(policy["real_PCC_fits_executed"],0)


if __name__=="__main__":
    unittest.main()
