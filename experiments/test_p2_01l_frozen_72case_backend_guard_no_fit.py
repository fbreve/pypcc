"""P2-01L original P1 case identity + no-fit strict runtime backend guard tests."""
import ast
import json
from pathlib import Path
import unittest

from experiments import p2_01l_frozen_72case_backend_guard_no_fit as g
from experiments import p2_07c_consolidated_execution_envelope_no_fit as c


class ModelStub:
    def __init__(self,impl,n_jobs=1,update_mode="sequential"):
        self.impl=impl
        self.n_jobs=n_jobs
        self.update_mode=update_mode

def original_72_P1():
    return c.read_approved_P1()

def example_rows():
    return [
        {"mode":"baseline_must_reject",
         "original_case":{"uci_id":212,"replicate":0,
                          "backend":c.BACKENDS[2],"status":"NOT_RUN_NO_APPROVAL"},
         "blocked_by":"BACKEND_IDENTITY_MISMATCH_OR_SILENT_FALLBACK",
         "case_dispatch_invocations":0,"PCC_fits_executed":0},
        {"mode":"corrected_72_case","cases_frozen":72,
         "backend_cases":{name:24 for name in c.BACKENDS},
         "fallback_accepted":False,
         "original_Cython_native_sha256":
            g.k.lock.NEW_NATIVE["production_pcc_step"],
         "original_Cython_observer_sha256":
            g.k.lock.NEW_NATIVE["isolated_pcc_step_live_research"],
         "model_graphs_built":0,"Numba_JIT_compilations":0,
         "PCC_propagation_function_calls":0,
         "PCC_fits_executed":0},
    ]

class P201LBackendGuardTests(unittest.TestCase):
    def test_original_P1_sha_72_ids_source_only(self):
        p=original_72_P1()
        self.assertEqual(c.sha256(c.canonical(p)),g.P1_SHA)
        self.assertEqual(len(c.case_order(p)),72)
        self.assertTrue(g.source_bytes_locked())

    def test_every_original_72_case_ID_has_expected_backend_and_NO_FIT(self):
        p=original_72_P1()
        seen=set()
        for candidate in c.case_order(p):
            case=g.case_lookup(p,candidate["uci_id"],
                               candidate["replicate"],candidate["backend"])
            self.assertEqual(case["status"],"NOT_RUN_NO_APPROVAL")
            seen.add((case["uci_id"],case["replicate"],case["backend"]))
        self.assertEqual(len(seen),72)

    def test_invalid_case_or_unapproved_backend_never_admitted(self):
        p=original_72_P1()
        for uci,rep,name in ((111,0,c.BACKENDS[0]),
                            (212,-1,c.BACKENDS[1]),
                            (212,8,c.BACKENDS[2]),
                            (212,0,"auto"),(212,0,"cython"),
                            (212,0,c.BACKENDS[0].upper())):
            with self.subTest(uci=uci,rep=rep,name=name):
                with self.assertRaises(g.BackendRejected):
                    g.case_lookup(p,uci,rep,name)

    def test_wrong_science_or_false_P2_approval_refused(self):
        for mutation in (
            lambda p: p["approvals_and_execution"].update(P2_execution_approved=True),
            lambda p: p["fit"].update(max_iter=1),
            lambda p: p["fit"].update(early_stop=True),
            lambda p: p["fit"].update(total_proposed_fits=71),
            lambda p: p["fit"].update(backend_order=list(reversed(c.BACKENDS))),
        ):
            p=original_72_P1()
            mutation(p)
            with self.assertRaises(g.BackendRejected):
                g.case_lookup(p,212,0,c.BACKENDS[0])

    def test_requested_impl_with_exact_function_identity_passes(self):
        for name in c.BACKENDS:
            impl=g.BACKENDS[name]
            fn=lambda:None
            self.assertTrue(g.assert_selected_exact(name,ModelStub(impl),
                                                    fn,fn))

    def test_silent_Cython_fallback_and_aliases_refused(self):
        requested=c.BACKENDS[2]
        cython_fn=lambda:None
        numba_fn=lambda:None
        with self.assertRaisesRegex(g.BackendRejected,"SILENT_FALLBACK"):
            g.assert_selected_exact(requested,ModelStub("cython"),
                                    numba_fn,cython_fn)
        for name in ("numba","auto","cython ","Cython"):
            with self.subTest(name=name):
                with self.assertRaises(g.BackendRejected):
                    g.assert_selected_exact(requested,ModelStub(name),
                                            cython_fn,cython_fn)

    def test_mutated_n_jobs_update_mode_or_non_callable_function_refused(self):
        name=c.BACKENDS[0]
        fn=lambda:None
        for model in (ModelStub("numpy",n_jobs=2),
                      ModelStub("numpy",update_mode="synchronous")):
            with self.assertRaises(g.BackendRejected):
                g.assert_selected_exact(name,model,fn,fn)
        with self.assertRaises(g.BackendRejected):
            g.assert_selected_exact(name,ModelStub("numpy"),None,None)

    def test_canonical_original_two_modes_and_review_only_report(self):
        rows=example_rows()
        result=g.assemble(rows)
        self.assertTrue(g.validate_report(result))
        self.assertEqual(g.sha(result),g.sha(json.loads(json.dumps(result))))
        self.assertFalse(result["P2_execution_authorized"])
        self.assertEqual(result["real_PCC_fits_executed"],0)

    def test_falsified_fallback_or_compilation_and_fit_rejected(self):
        rows=example_rows()
        for idx,key,value in (
            (0,"blocked_by","UNKNOWN"),
            (0,"case_dispatch_invocations",1),
            (0,"PCC_fits_executed",1),
            (1,"fallback_accepted",True),
            (1,"cases_frozen",71),
            (1,"original_Cython_native_sha256","0"*64),
            (1,"Numba_JIT_compilations",1),
            (1,"PCC_propagation_function_calls",1),
            (1,"PCC_fits_executed",1),
        ):
            with self.subTest(idx=idx,key=key):
                r=example_rows()
                r[idx][key]=value
                with self.assertRaises(g.BackendRejected):
                    g.assemble(r)

    def test_false_P2_GO_or_unproven_fit_bounds_rejected(self):
        result=g.assemble(example_rows())
        for key,value in (
            ("P2_01_final_execution_toolchain_lock_pass",True),
            ("P2_07_full_resources_gate_pass",True),
            ("P2_execution_preflight_sha256","a"*64),
            ("P2_execution_authorized",True),
            ("real_PCC_fits_executed",1),
            ("original_50k_full_fit_resource_limits_proven",True),
            ("frozen_cases_visited_without_fits",71),
        ):
            with self.subTest(key=key):
                with self.assertRaises(g.BackendRejected):
                    g.validate_report({**result,key:value})

    def test_Docker_policy_same_original_OCI_and_path_closed(self):
        fixed=g.container_args("corrected_72_case","pypcc-p201l-012345abcdef")
        old=g.container_args("baseline_must_reject","pypcc-p201l-012345abcdef")
        self.assertEqual(fixed[fixed.index("--name")+1],
                         "pypcc-p201l-012345abcdef")
        for argument in ("--read-only","--network","none","--memory",
                         "--user","65534:65534","--cap-drop","ALL",
                         "PYTHONPATH="+g.k.CORRECT_PATH):
            self.assertIn(argument,fixed)
        self.assertIn("PYTHONPATH="+g.k.FALLBACK_PATH,old)
        self.assertIn("experiments.p2_01l_frozen_72case_backend_guard_no_fit",
                      fixed)
        self.assertNotIn("fit_predict"," ".join(fixed))
        for name in ("foo","pypcc-p201k-012345abcdef","pypcc-p201l-abc"):
            with self.assertRaises(g.BackendRejected):
                g.container_args("corrected_72_case",name)

if __name__=="__main__":
    unittest.main()
