"""P2-07L graph-only no-fit adversarial tests, no UCI download or PCC import."""
import copy
import json
import subprocess
import unittest

from experiments import p2_07l_original_graph_preparation_resource_no_fit as l


def mock_child(uci):
    n,f,width=l.DIMENSIONS[uci]
    return {
        "schema":"pypcc-P2-07L-real-graph-prep-child/v1",
        "uci_id":uci,
        "original_source_rows":n,
        "original_feature_count":f,
        "actual_dense_neighbor_width":width,
        "exact_original_graph_SHA256":l.GRAPH_SHA[uci],
        "graph_arrays_bytes":n*(width+1)*8,
        "graph_preparation_wall_seconds":0.25,
        "graph_preparation_process_cpu_seconds":0.21,
        "total_source_download_parse_transform_graph_wall_seconds":0.75,
        "original_process_vm_hwm_bytes":310*1024**2,
        "original_process_vm_rss_after_graph_bytes":300*1024**2,
        "model_graph_was_undirected":True,
        "model_padded_dense_minus_one":True,
        "peak_rss_is_NOT_a_full_50k_PCC_fit_bound":True,
        "real_masks_or_seeds_generated":0,
        "PCC_propagations_executed":0,
        "real_PCC_fits_executed":0,
        "rows_labels_graph_edges_or_masks_exported":False,
        "P2_execution_authorized":False,
    }


class OriginalGraphOnlyNoFitContract(unittest.TestCase):
    def test_source_identity_and_widths(self):
        self.assertEqual(l.SOURCES,(212,850,292))
        self.assertEqual(set(l.DIMENSIONS),set(l.SOURCES))
        self.assertTrue(all(l.inspect_child(mock_child(i)) for i in l.SOURCES))

    def test_original_dense_minus_one_and_hash_tamper_block(self):
        for key,value in (
            ("actual_dense_neighbor_width",99),
            ("exact_original_graph_SHA256","0"*64),
            ("model_padded_dense_minus_one",False),
            ("model_graph_was_undirected",False),
            ("graph_arrays_bytes",1),
        ):
            with self.subTest(key=key):
                row=mock_child(212)
                row[key]=value
                with self.assertRaises(l.PrepBlocked):
                    l.inspect_child(row)

    def test_original_PCC_fit_or_mask_forgery_blocked(self):
        for key,value in (
            ("PCC_propagations_executed",1),
            ("real_PCC_fits_executed",1),
            ("real_masks_or_seeds_generated",24),
            ("rows_labels_graph_edges_or_masks_exported",True),
            ("P2_execution_authorized",True),
            ("peak_rss_is_NOT_a_full_50k_PCC_fit_bound",False),
        ):
            with self.subTest(key=key):
                row=mock_child(850)
                row[key]=value
                with self.assertRaises(l.PrepBlocked):
                    l.inspect_child(row)

    def test_elapsed_and_process_highwater_validation(self):
        for key,value in (
            ("graph_preparation_wall_seconds",-1.0),
            ("graph_preparation_process_cpu_seconds",-1),
            ("total_source_download_parse_transform_graph_wall_seconds",200),
            ("original_process_vm_hwm_bytes",0),
            ("original_process_vm_rss_after_graph_bytes",500*1024**2),
        ):
            with self.subTest(key=key):
                row=mock_child(292)
                row[key]=value
                with self.assertRaises(l.PrepBlocked):
                    l.inspect_child(row)

    def test_fake_subprocess_replay_cannot_change_identity(self):
        def runner(cmd, **kwargs):
            self.assertIn("--child",cmd)
            self.assertEqual(kwargs["env"]["OPENBLAS_NUM_THREADS"],"1")
            return subprocess.CompletedProcess(cmd,0,stdout=json.dumps(mock_child(850)),stderr="")
        with self.assertRaises(l.PrepBlocked):
            l.run_child(212,runner=runner)

    def test_fake_process_nonzero_exit_fails_closed(self):
        def runner(cmd, **kw):
            return subprocess.CompletedProcess(cmd,23,stdout="",stderr="crashed")
        with self.assertRaises(l.PrepBlocked):
            l.run_child(212,runner=runner)

    def test_unbounded_or_corrupted_child_output_rejected(self):
        for value in ("x"*9000, "not json", json.dumps({"fake":True})):
            with self.subTest(value_size=len(value)):
                def runner(cmd,**kw):
                    return subprocess.CompletedProcess(cmd,0,stdout=value,stderr="")
                with self.assertRaises(l.PrepBlocked):
                    l.run_child(212,runner=runner)

    def test_frozen_case_sequence_and_no_go_report(self):
        rows=[mock_child(i) for i in l.SOURCES]
        report=l.final_report(rows)
        self.assertTrue(l.validate_report(report))
        self.assertIsNone(report["P2_execution_preflight_sha256"])
        self.assertFalse(report["P2_execution_authorized"])
        self.assertFalse(report["actual_original_full_50k_PCC_fit_resource_upper_bound_proven"])
        self.assertEqual(report["original_72_scientific_PCC_fit_count"],0)

    def test_false_RSS_viability_and_scientific_approval_rejected(self):
        base=l.final_report([mock_child(i) for i in l.SOURCES])
        for key,value in (
            ("actual_original_full_50k_PCC_fit_resource_upper_bound_proven",True),
            ("P2_07_full_resources_gate_pass",True),
            ("P2_01_final_execution_toolchain_lock_pass",True),
            ("P2_execution_preflight_sha256","1"*64),
            ("P2_execution_authorized",True),
            ("real_PCC_fits_executed",72),
            ("candidate_2GiB_limit_is_NOT_approved_or_full_fit_feasible",False),
            ("observed_graph_only_peak_RSS_below_candidate_2GiB",False),
        ):
            with self.subTest(key=key):
                with self.assertRaises(l.PrepBlocked):
                    l.validate_report({**base,key:value})

    def test_missing_extra_or_permuted_original_graph_child_blocked(self):
        base=[mock_child(i) for i in l.SOURCES]
        for rows in (base[:2],base+base[:1],[base[1],base[0],base[2]]):
            with self.subTest(length=len(rows)):
                with self.assertRaises(l.PrepBlocked):
                    l.final_report(rows)

    def test_original_graph_memory_above_candidate_is_observation_not_approval(self):
        rows=[mock_child(i) for i in l.SOURCES]
        rows[0]["original_process_vm_hwm_bytes"]=3*1024**3
        report=l.final_report(rows)
        self.assertFalse(report["observed_graph_only_peak_RSS_below_candidate_2GiB"])
        self.assertFalse(report["P2_07_full_resources_gate_pass"])

    def test_report_canonical_hash_round_trip(self):
        report=l.final_report([mock_child(i) for i in l.SOURCES])
        self.assertEqual(l.digest(report),l.digest(json.loads(json.dumps(report))))
        self.assertTrue(l.validate_report(report))


if __name__=="__main__":
    unittest.main()
