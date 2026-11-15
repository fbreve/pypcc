"""P2-07M strict source-only contract tests; no Docker, UCI, PCC or NumPy."""
import copy
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from experiments import p2_07m_exact_oci_original_graph_preparation_no_fit as m


def fake_original_graph(uci_id):
    n,features,width=m.DIMENSIONS[uci_id]
    return {
        "schema":"pypcc-P2-07M-exact-OCI-worker-graph-only/v1",
        "uci_id":uci_id,
        "P1_scientific_sha256":m.P1_SHA,
        "loaded_exact_existing_OCI_config_id":m.IMAGE_ID,
        "original_transformed_X_sha256":m.X_HASHES[uci_id],
        "original_graph_sha256":m.GRAPH_HASHES[uci_id],
        "original_graph_width":width,
        "original_graph_array_bytes":8*n*(width+1),
        "original_graph_dense_minus_one_padding":True,
        "original_graph_undirected":True,
        "graph_build_wall_seconds":0.04,
        "graph_build_cpu_seconds":0.03,
        "process_peak_resident_bytes":300*1024**2,
        "process_RSS_after_graph_bytes":280*1024**2,
        "real_original_PCC_fits_executed":0,
        "synthetic_PCC_fits_executed":0,
        "PCC_propagations_executed":0,
        "original_masks_or_PCC_seeds_generated":0,
        "real_UCI_source_rows_or_graph_edges_exported":False,
        "P2_execution_authorized":False,
    }


class OriginalGraphExactExistingOCIGate(unittest.TestCase):
    def test_original_source_production_blobs_unchanged(self):
        self.assertTrue(m.verify_source_blobs())
        self.assertEqual(tuple(m.IDS),(212,850,292))
        self.assertEqual(set(m.PRODUCTION_BLOBS),
                         {"pcc.py","pcc_graph.py","pcc_numpy.py","pcc_numba.py",
                          "pcc_validation.py"})

    def test_three_exact_frozen_graph_and_feature_hashes(self):
        self.assertEqual(len(m.GRAPH_HASHES),3)
        self.assertEqual(len(m.X_HASHES),3)
        for uci in m.IDS:
            self.assertTrue(m.validate_worker_result(fake_original_graph(uci)))

    def test_false_pcc_fits_propagations_masks_and_approval_blocked(self):
        for key,value in (
            ("real_original_PCC_fits_executed",1),
            ("synthetic_PCC_fits_executed",1),
            ("PCC_propagations_executed",1),
            ("original_masks_or_PCC_seeds_generated",24),
            ("P2_execution_authorized",True),
            ("real_UCI_source_rows_or_graph_edges_exported",True),
        ):
            with self.subTest(key=key):
                r=fake_original_graph(850)
                r[key]=value
                with self.assertRaises(m.RuntimeGraphBlocked):
                    m.validate_worker_result(r)

    def test_wrong_exact_OCI_and_source_identity_denied(self):
        for key,value in (
            ("loaded_exact_existing_OCI_config_id","sha256:"+"0"*64),
            ("P1_scientific_sha256","0"*64),
            ("original_graph_sha256","0"*64),
            ("original_transformed_X_sha256","0"*64),
            ("uci_id",444),
        ):
            with self.subTest(key=key):
                r=fake_original_graph(212)
                r[key]=value
                with self.assertRaises(m.RuntimeGraphBlocked):
                    m.validate_worker_result(r)

    def test_changed_dense_width_padding_and_graph_byte_count_denied(self):
        for key,value in (
            ("original_graph_width",10),
            ("original_graph_array_bytes",0),
            ("original_graph_dense_minus_one_padding",False),
            ("original_graph_undirected",False),
        ):
            with self.subTest(key=key):
                r=fake_original_graph(292)
                r[key]=value
                with self.assertRaises(m.RuntimeGraphBlocked):
                    m.validate_worker_result(r)

    def test_invalid_CPU_wall_peak_and_RSS_claims_denied(self):
        for key,value in (
            ("graph_build_wall_seconds",-1),
            ("graph_build_cpu_seconds",121),
            ("process_peak_resident_bytes",0),
            ("process_RSS_after_graph_bytes",400*1024**2),
        ):
            with self.subTest(key=key):
                r=fake_original_graph(212)
                r[key]=value
                with self.assertRaises(m.RuntimeGraphBlocked):
                    m.validate_worker_result(r)

    def test_original_OCI_worker_command_no_network_no_writable_root(self):
        with tempfile.TemporaryDirectory() as td:
            cmd=m.validated_docker_command(212,"pypcc-p207m-123456abcdef",Path(td))
        for value in ("none","--read-only","65534:65534",
                      "--cap-drop","no-new-privileges","128","2g","1",
                      "PYTHONPATH=/workspace",
                      "NUMBA_CACHE_DIR=/tmp/numba-cache"):
            self.assertTrue(any(value in item for item in cmd),value)
        self.assertIn("pypcc-p2-01h-candidate:immutable",cmd)
        self.assertIn("--worker",cmd)
        self.assertNotIn("fit_predict",cmd)
        self.assertNotIn("docker build"," ".join(cmd))
        self.assertNotIn("docker pull"," ".join(cmd))

    def test_reject_unknown_worker_or_unsafe_container_identity(self):
        with tempfile.TemporaryDirectory() as td:
            for bad_id,name in ((999,"pypcc-p207m-123456abcdef"),
                                (212,"unsafe"),
                                (212,"pypcc-p207m-1234")):
                with self.subTest(bad_id=bad_id,name=name):
                    with self.assertRaises(m.RuntimeGraphBlocked):
                        m.validated_docker_command(bad_id,name,Path(td))

    def test_strict_three_original_order_and_full_review_no_go(self):
        report=m.assemble([fake_original_graph(i) for i in m.IDS])
        self.assertTrue(m.validate_report(report))
        self.assertFalse(report["P2_execution_authorized"])
        self.assertIsNone(report["P2_execution_preflight_sha256"])
        self.assertEqual(report["real_PCC_fits_executed"],0)
        self.assertFalse(report["actual_full_original_50k_fit_CPU_wall_RSS_disk_ceiling_proven"])

    def test_changed_order_duplication_and_missing_original_input_denied(self):
        rows=[fake_original_graph(i) for i in m.IDS]
        for bad in (rows[:2],rows+[rows[0]],[rows[1],rows[0],rows[2]]):
            with self.assertRaises(m.RuntimeGraphBlocked):
                m.assemble(bad)

    def test_false_full_50k_resource_or_runtime_Go_report_denied(self):
        report=m.assemble([fake_original_graph(i) for i in m.IDS])
        for key,val in (
            ("complete_host_compiler_SDK_system_byte_closure_proven",True),
            ("actual_full_original_50k_fit_CPU_wall_RSS_disk_ceiling_proven",True),
            ("P2_01_final_execution_toolchain_lock_pass",True),
            ("P2_07_full_resources_gate_pass",True),
            ("P2_execution_preflight_sha256","a"*64),
            ("P2_execution_authorized",True),
            ("real_PCC_fits_executed",72),
            ("source_dataset_matrices_or_graph_neighbors_persisted_in_artifacts",True),
        ):
            with self.subTest(key=key):
                with self.assertRaises(m.RuntimeGraphBlocked):
                    m.validate_report({**report,key:val})

    def test_high_graph_only_peak_is_observed_failure_of_draft_not_science(self):
        rows=[fake_original_graph(i) for i in m.IDS]
        rows[0]["process_peak_resident_bytes"]=3*1024**3
        report=m.assemble(rows)
        self.assertFalse(report["observed_peak_below_unapproved_candidate_2GiB_for_graph_only"])
        self.assertFalse(report["P2_07_full_resources_gate_pass"])
        self.assertTrue(m.validate_report(report))

    def test_worker_nonzero_or_invalid_json_fail_closed_and_cleanup(self):
        with tempfile.TemporaryDirectory() as td:
            for fake in (
                subprocess.CompletedProcess(["docker"],5,"","error"),
                subprocess.CompletedProcess(["docker"],0,"invalid",""),
                subprocess.CompletedProcess(["docker"],0,"x"*9000,""),
            ):
                with self.subTest(status=fake.returncode,outlen=len(fake.stdout)):
                    with patch.object(m.subprocess,"run",
                        return_value=subprocess.CompletedProcess(["docker"],1,"","")):
                        with self.assertRaises(m.RuntimeGraphBlocked):
                            m.run_original_graph(212,Path(td),run=lambda *a,**kw:fake)

    def test_report_content_deterministic_and_no_secret_input_bytes(self):
        report=m.assemble([fake_original_graph(i) for i in m.IDS])
        self.assertEqual(m.sha_bytes(m.canonical(report)),
                         m.sha_bytes(m.canonical(json.loads(json.dumps(report)))))
        self.assertNotIn("ordered_predictor_names",report)
        self.assertNotIn("source_records",report)


if __name__=="__main__":
    unittest.main()
