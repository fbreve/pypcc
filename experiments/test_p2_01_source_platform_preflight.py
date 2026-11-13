"""No-fit P2-01A source provenance and runtime metadata adversarial tests."""
import copy
import json
from pathlib import Path
import unittest

from experiments import p2_01_source_platform_preflight as pre


class P201SourceAndPlatformTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.m=json.loads(pre.MANIFEST.read_text(encoding="utf-8"))
        cls.p1=json.loads(pre.P1_MANIFEST.read_text(encoding="utf-8"))
        cls.payload=json.loads(pre.P1_PAYLOAD.read_text(encoding="utf-8"))

    def check_invalid(self,change):
        fake=copy.deepcopy(self.m)
        change(fake)
        with self.assertRaises(pre.P201Blocked):
            pre.verify_manifest_only(fake,self.p1,self.payload)

    def test_static_pins_match_approved_science(self):
        self.assertTrue(pre.verify_manifest_only(
            self.m,self.p1,self.payload))
        self.assertEqual(len(self.m["production_file_git_blobs"]),10)
        self.assertEqual(
            [x["pr"] for x in self.m["research_sources"]],
            [253,255,256,257],
        )
        self.assertEqual(
            self.m["declared_build_and_install_constraints"]["research_isolated_Cython_module"],
            "pcc_step_live_research",
        )
        self.assertIsNone(
            self.m["scope_and_limitations"]["P2_execution_preflight_SHA256"]
        )

    def test_altered_production_or_observer_pins_fail_closed(self):
        self.check_invalid(lambda m: m["production_file_git_blobs"].update(
            {"pcc_step.pyx":"a"*40}))
        self.check_invalid(lambda m: m.update(production_master="0"*40))
        self.check_invalid(lambda m: m["research_sources"][0].update(
            head="0"*40))
        self.check_invalid(lambda m: m["research_sources"][1]["files"].update(
            {"experiments/live_numba_milestones.py":"a"*40}))
        self.check_invalid(lambda m: m["research_sources"][2].update(
            native_artifact=""))
        self.check_invalid(lambda m: m["declared_build_and_install_constraints"].update(
            research_isolated_Cython_module="pcc_step"))
        self.check_invalid(lambda m: m["declared_build_and_install_constraints"].update(
            actual_runtime_resolved_package_versions="TRUST_CI_DEFAULT"))

    def test_forged_P2_execution_or_native_build_rejected(self):
        for key, value in [
            ("P2_01_status","PASS_ALL"),
            ("P2_06_status","PASS"),
            ("binary_identity_and_ABI","VERIFIED"),
            ("runtime_versions","LOCKED"),
            ("P2_execution_preflight_SHA256","b"*64),
            ("P2_execution_authorized",True),
            ("PCC_fits_executed",1),
        ]:
            self.check_invalid(lambda m,k=key,v=value:
                               m["scope_and_limitations"].update({k:v}))

    def test_runtime_metadata_does_not_import_native_packages(self):
        versions=pre.read_installed_metadata()
        self.assertEqual(
            set(versions),
            {"numpy","scipy","scikit-learn","numba","Cython",
             "setuptools","wheel","llvmlite"},
        )
        source=Path(pre.__file__).read_text(encoding="utf-8")
        for prohibited in (
            "import numpy","import scipy","import sklearn","import numba",
            "from pcc","import pcc","fit_predict(","build_knn_graph(",
            "from Cython", "cythonize(", "subprocess.check_call(",
            "urlopen(","requests.get(",
        ):
            self.assertNotIn(prohibited,source)

    def test_git_object_store_pins_match_actual_observer_heads(self):
        checked=pre.verify_git_tree(pre.ROOT,self.m,verify_remote=False)
        self.assertEqual(set(checked),
                         {"master","numpy","numba","cython","source_p0"})
        for name in ("numpy","numba","cython","source_p0"):
            self.assertEqual(checked[name]["production_modifications"],0)
            self.assertEqual(checked[name]["production_files"],10)
        self.assertEqual(
            checked["master"]["head"],
            "ea69f690208620d5563c8efece6b2e04369f52c0",
        )


if __name__=="__main__":
    unittest.main()
