"""P2 plan gate: statically check a NO-FIT preflight proposal, not execution.

No imports of PyPCC, NumPy, network sources or dataset rows. This gate only
proves that the planned technical checklist is tied to the exact approved P1
scientific SHA and cannot silently claim a completed P2/real-data fitting.
"""
import argparse
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
P1_MANIFEST=ROOT/"docs/current_p1_scientific_sha_manifest_2026-11-12.json"
P2_PLAN=ROOT/"docs/current_p2_preflight_plan_2026-11-12.json"
EXPECTED_SHA="f0111f153ade3d72191c107ace039e1835e5c27db2cf4b3df8d610b41b2f4166"
GATES=[
    "P2_01_CODE_AND_VERSIONS",
    "P2_02_OFFICIAL_SOURCES",
    "P2_03_TRANSFORMED_FEATURES",
    "P2_04_KNN_GRAPH",
    "P2_05_MASKS_AND_SEEDS",
    "P2_06_NATIVE_OBSERVERS",
    "P2_07_RESOURCE_AND_EXPORT_LIMITS",
    "P2_08_SEALED_EXECUTION_GATE",
]


class P2NotReady(ValueError):
    pass


def require(ok,reason):
    if not ok:
        raise P2NotReady(reason)


def review(plan,manifest):
    require(plan.get("schema")=="pypcc-P2-execution-preflight-plan/v1" and
            plan.get("status")=="PLANNED_NOT_EXECUTED_NO_FIT_NO_EXECUTION_APPROVAL",
            "P2 plan not run-ready; cannot silently mark it ready")
    p1=plan["parent_science"]
    require(p1["P1_scientific_sha256"]==EXPECTED_SHA and
            manifest["approved_P1_scientific_protocol_sha256"]==EXPECTED_SHA and
            p1["P1_approved_payload"]==
            "docs/current_p1_frozen_scientific_protocol_2026-11-12.json" and
            p1["P1_approved_manifest"]==
            "docs/current_p1_scientific_sha_manifest_2026-11-12.json" and
            p1["no_inference_or_source_changes"] is True,
            "P2 must build on exactly approved P1 SHA and same science")
    steps=plan["technical_steps"]
    require(type(steps) is list and
            [x["id"] for x in steps]==GATES and
            all(x["status"]=="NOT_EXECUTED" and
                len(x["required"])>=45 and len(x["blocking_rule"])>=20
                for x in steps),
            "P2 all eight technical gates remain unexecuted")
    checks=plan["current_checks"]
    require(all(checks[k] is False for k in (
        "preflight_started","native_binaries_built_for_P2",
        "source_data_loaded_under_P2","label_masks_generated_under_P2",
        "real_knn_graph_generated_under_P2",
        "actual_72_case_resource_budget_verified","all_8_gates_passed",
        "user_execution_approval_for_P2_SHA",
        "any_real_PCC_fit_authorized",
    )) and checks["preflight_SHA256"] is None and
            checks["real_PCC_fit_count"]==0,
            "No technical preflight execution or PCC fits are authorized yet")
    require(manifest["P2_preflight_sha256"] is None and
            manifest["P2_execution_authorized"] is False and
            manifest["any_real_PCC_fit_authorized"] is False,
            "Scientific P1 approval is not a P2 permission")
    require("request exact-SHA execution consent" in
            plan["next_action"] and
            "P1 science approved only" in plan["approval_boundary"],
            "Future P2 must require a new exact-SHA user approval")
    identity=json.dumps(plan,sort_keys=True,ensure_ascii=True,
                        allow_nan=False,separators=(",",":"))
    return {
        "schema":"pypcc-P2-review-only-plan-validation/v1",
        "P2_plan_integrity":"PASS_PLANNING_ONLY",
        "P1_scientific_sha256":EXPECTED_SHA,
        "planned_gates":8,
        "gates_executed":0,
        "P2_plan_sha256_NOT_preflight_approval":
            hashlib.sha256(identity.encode("ascii")).hexdigest(),
        "real_dataset_graphs_or_masks_created":0,
        "real_PCC_fits_executed":0,
        "P2_preflight_sha256":None,
        "P2_execution_authorized":False,
    }


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out",type=Path)
    args=p.parse_args()
    result=review(
        json.loads(P2_PLAN.read_text(encoding="utf-8")),
        json.loads(P1_MANIFEST.read_text(encoding="utf-8")),
    )
    value=json.dumps(result,sort_keys=True,indent=2)+"\n"
    if args.out:
        args.out.write_text(value,encoding="utf-8")
    print(value)


if __name__=="__main__":
    main()
