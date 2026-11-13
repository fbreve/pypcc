"""P2-07D tests use only synthetic fixtures/OS probe processes, NO PCC fits."""
import copy
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from experiments import p2_07d_synthetic_50k_resource_watchdog as gate
from experiments import p2_07c_consolidated_execution_envelope_no_fit as prior


class Synthetic50kGuardTests(unittest.TestCase):
    def setUp(self):
        self.p1=prior.read_approved_P1()
        self.bounds=prior.validate_policy(prior.load(prior.POLICY),self.p1)

    def test_exact_frozen_three_node_feature_class_width_geometries(self):
        self.assertEqual(gate.CASES,((212,310,6,3,22),
                                     (850,900,7,2,20),
                                     (292,440,6,2,24)))
        self.assertEqual(gate.MILESTONES,
                         (0,1,10,100,1000,5000,25000,50000))
        self.assertEqual(gate.ITERATIONS,50000)
        self.assertEqual(gate.SEED,71)
        self.assertEqual(gate.BACKENDS,("numpy","numba","cython"))
        self.assertEqual(len(gate.ORIGINAL_OBSERVERS),3)
        self.assertEqual(len(gate.ORIGINAL_HELPER_SOURCE),2)

    def test_undirected_original_dense_padding_and_class_cardinalities(self):
        import numpy as np
        for uci,n,d,c,w in gate.CASES:
            with self.subTest(uci=uci):
                neighbors,counts=gate.synthetic_graph(n,w)
                X,labels=gate.synthetic_features_and_labels(n,d,c)
                self.assertEqual(neighbors.shape,(n,w))
                self.assertEqual(neighbors.dtype,np.dtype("int64"))
                self.assertEqual(X.shape,(n,d))
                self.assertEqual(X.dtype,np.dtype("float64"))
                self.assertEqual(counts.shape,(n,))
                self.assertEqual((int(counts.min()),int(counts.max())),(10,w))
                self.assertEqual(int(np.count_nonzero(labels>=0)),3*c)
                self.assertTrue(np.all(neighbors[0,:w]>=0))
                self.assertTrue(np.all(neighbors[:, :10]>=0))
                self.assertTrue(np.all(
                    neighbors[np.arange(w)[None,:] >= counts[:,None]] == -1))
                for row in range(n):
                    active=neighbors[row,:counts[row]]
                    self.assertEqual(len(set(active.tolist())),len(active))
                    self.assertTrue(np.all(active!=row))
                    self.assertTrue(np.all(np.diff(active)>0))
                    self.assertTrue(all(row in neighbors[other,:counts[other]]
                                        for other in active))

    def test_reject_graph_drift_and_invalid_source_shape(self):
        with self.assertRaises(gate.SyntheticGuardBlocked):
            gate.synthetic_graph(310,10)
        with self.assertRaises(gate.SyntheticGuardBlocked):
            gate.synthetic_graph(100,60)
        with self.assertRaises(gate.SyntheticGuardBlocked):
            gate.synthetic_features_and_labels(310,10,3)
        with self.assertRaises(gate.SyntheticGuardBlocked):
            gate.synthetic_features_and_labels(310,6,4)

    def test_append_only_ledger_keeps_all_status_transitions(self):
        with tempfile.TemporaryDirectory() as folder:
            ledger=Path(folder)/"attempts.ndjson"
            for status in ("STARTED_SYNTHETIC_ONLY","FAILED_NO_RETRY"):
                gate.append_evidence(ledger,{"uci_id":212,
                    "backend":"numba","status":status})
            lines=ledger.read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(lines),2)
            self.assertIn('"STARTED_SYNTHETIC_ONLY"',lines[0])
            self.assertIn('"FAILED_NO_RETRY"',lines[1])

    def test_policy_false_GO_persistent_even_after_synthetic_success(self):
        policy=prior.load(prior.POLICY)
        self.assertFalse(policy["P2_execution_authorized"])
        self.assertFalse(policy["P2_07_full_resources_gate_pass"])
        self.assertEqual(policy["candidate_guardrails"]["max_one_fit_peak_process_resident_bytes"],
                         2*1024**3)
        self.assertEqual(policy["fit_schedule"]["max_parallel_fits"],1)
        self.assertFalse(policy["proposed_execution_host"]
                               ["physical_or_immutable_execution_host_selected"])

    def test_parent_OS_watchdog_ACTUALLY_kills_four_forced_synthetic_resource_faults(self):
        # Four harmless interpreter-only fault probes: NO PyPCC/NumPy/model import.
        # Proves kill occurs on a forced breach, rather than only comparing a
        # final report to thresholds. Keep total probe runtime under ~10 s.
        scripts={
            "watchdog_wall_time":
                "import time; time.sleep(3)",
            "watchdog_peak_process_memory":
                "import time; sample=bytearray(128*1024*1024); time.sleep(3)",
            "watchdog_process_CPU":
                "import time; s=time.process_time();\n"
                "while time.process_time()-s<3: pass",
            "watchdog_scratch_disk":
                "import sys,time,pathlib; "
                "(pathlib.Path(sys.argv[1])/'probe.bin').write_bytes(b'x'*(2*1024*1024)); "
                "time.sleep(3)",
        }
        for fault,script in scripts.items():
            with self.subTest(forced_fault=fault):
                with tempfile.TemporaryDirectory() as folder:
                    bounds=copy.deepcopy(self.bounds)
                    bounds["max_one_fit_peak_process_resident_bytes"]=4*1024**3
                    bounds["max_one_fit_process_cpu_seconds"]=3600
                    bounds["max_one_fit_scratch_bytes"]=1024**3
                    cap_seconds=5
                    if fault=="watchdog_peak_process_memory":
                        bounds["max_one_fit_peak_process_resident_bytes"]=40*1024**2
                    elif fault=="watchdog_process_CPU":
                        bounds["max_one_fit_process_cpu_seconds"]=.25
                    elif fault=="watchdog_scratch_disk":
                        bounds["max_one_fit_scratch_bytes"]=128*1024
                    else:
                        cap_seconds=.3
                    proc=subprocess.Popen(
                        [sys.executable,"-c",script,folder],
                        stdout=subprocess.PIPE,stderr=subprocess.PIPE,
                        text=True,errors="replace")
                    evidence=gate.poll_subprocess_and_enforce(
                        proc,folder,bounds,cap_seconds)
                    self.assertEqual(evidence["failure"],fault,
                                     "Watchdog did not exercise "+fault+
                                     ": "+evidence["stderr"][-500:])
                    self.assertNotEqual(evidence["returncode"],0)
                    self.assertGreater(evidence["samples"],0)

if __name__=="__main__":
    unittest.main()
