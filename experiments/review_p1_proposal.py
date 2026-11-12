"""Fail-closed, standard-library-only review of an UNAPPROVED P1 design draft.

This program can validate structural consistency and list scientific gaps;
it CANNOT approve or freeze a scientific study or authorize a real-data fit.
No dataset, NumPy, sklearn, PCC backend, Cython build, or RNG is imported.
"""
import argparse
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROPOSAL = ROOT / "docs" / "current_p1_prospective_design_draft_2026-11-12.json"
SCHEMA = "pypcc-p1-prospective-scientific-design/v0.1"
EXPECTED_P0 = "edd5ce4c114fe86697cc2e86ce2be58ea1ec9c89fc5d17e07bd564b98df9ecd2"
EXPECTED_BACKENDS = [
    "numpy_sequential", "numba_sequential_nopython",
    "cython_sequential_cached_or_legacy_as_defined_by_early_stop",
]


class DesignInvalid(ValueError):
    pass


def _require(condition, message):
    if not condition:
        raise DesignInvalid(message)


def _canonical_digest(proposal):
    return hashlib.sha256(json.dumps(
        proposal, sort_keys=True, separators=(",", ":"),
        ensure_ascii=True, allow_nan=False,
    ).encode("ascii")).hexdigest()


def review(proposal):
    """Validate the draft as a *draft*, not a frozen scientific record."""
    _require(type(proposal) is dict, "proposal must be a JSON object")
    _require(proposal.get("schema") == SCHEMA, "unexpected design schema")
    _require(proposal.get("status") == "DRAFT_NOT_APPROVED",
             "P1 review refuses apparently approved or sealed status")
    _require(proposal["p0_reference"]["source_inventory_sha256"] == EXPECTED_P0,
             "P0 source-only inventory reference changed")
    _require(proposal["p0_reference"]["source_inventory_is_not_authorization"] is True,
             "P0 must never be treated as approval")

    experimental = proposal["experimental_design"]
    _require(experimental["backend_conditions"] == EXPECTED_BACKENDS,
             "missing, altered or repeated native backend")
    _require(experimental["early_stop"] is False,
             "proposed no-stop study must not silently change stopping")
    _require(type(experimental["max_iterations_proposed"]) is int,
             "iteration limit must be integer")
    schedule = experimental["milestones_proposed"]
    _require(type(schedule) is list and schedule and
             len(schedule) <= experimental["limits"]["max_observations"] <= 32,
             "invalid milestone count")
    _require(all(type(x) is int and x >= 0 for x in schedule),
             "milestones must be nonnegative exact integers")
    _require(schedule == sorted(set(schedule)) and schedule[0] == 0,
             "milestones must be sorted, unique, and start at zero")
    _require(schedule[-1] == experimental["max_iterations_proposed"],
             "terminal milestone must be fixed run limit")
    primary = experimental["primary_comparison"]
    _require(
        type(primary["from_iteration"]) is int and
        type(primary["to_iteration"]) is int and
        primary["from_iteration"] in schedule and
        primary["to_iteration"] in schedule and
        0 < primary["from_iteration"] < primary["to_iteration"],
        "primary trajectory comparison must use two frozen milestones",
    )
    _require(type(experimental["label_budget_per_class_proposed"]) is int and
             experimental["label_budget_per_class_proposed"] > 0,
             "invalid label budget")
    _require(type(experimental["label_mask_replicates_proposed"]) is int and
             experimental["label_mask_replicates_proposed"] > 1,
             "multiple independent label masks are required")
    _require(type(experimental["namespace_integer_proposed"]) is int and
             experimental["namespace_integer_proposed"] >= 0,
             "invalid prospective independent label RNG namespace")
    _require(experimental["no_adaptive_resampling"] is True,
             "adaptive resampling is prohibited")
    _require("dense" in experimental["graph_policy"] and
             "CSR" not in experimental["graph_policy"],
             "graph policy must preserve dense neighbors without substitutes")

    dataset = proposal["dataset_registry"]
    _require(dataset["selected"] == [],
             "dataset IDs remain unreviewed; cannot declare a draft with selected datasets")
    _require(type(dataset["target_count_proposed"]) is int and
             dataset["target_count_proposed"] >= 2,
             "proposed dataset target must be >=2")
    eligibility = dataset["eligibility_proposed"]
    _require(eligibility["min_instances_per_class"] >
             experimental["label_budget_per_class_proposed"],
             "each class must have unlabeled evaluation nodes")
    _require(experimental["number_of_fits_if_approved_and_feasible"] ==
             dataset["target_count_proposed"] *
             experimental["label_mask_replicates_proposed"] *
             len(EXPECTED_BACKENDS),
             "declared proposed fit count is inconsistent")

    estimands = proposal["estimands"]
    _require(estimands["primary"]["unit"] == "dataset x frozen label mask x backend",
             "the experimental unit cannot be an individual graph node")
    _require("initially unlabeled" in
             estimands["primary"]["denominator"],
             "primary denominator must exclude originally labeled nodes")
    _require(len(estimands["secondary"]) >= 2,
             "all secondary analyses must be declared before outcomes")
    historical = proposal["historical_overlap"]
    _require({"banknote", "sonar", "pageblocks", "spambase"}.issubset(
        set(historical["datasets_examined_in_prior_C2_G1"])),
        "historical datasets must be disclosed to prevent post-selection",
    )

    approval = proposal["approval"]
    for flag in ("scientific_protocol_frozen", "execution_authorized"):
        _require(approval.get(flag) is False,
                 f"{flag} must be false in an unapproved P1 proposal")
    for key in ("scientific_protocol_sha256", "approved_by", "approved_at",
                "execution_preflight_sha256"):
        _require(approval.get(key) is None,
                 f"{key} must remain null")
    _require(approval["real_data_fits_authorized"] == 0,
             "P1 cannot authorize fits")
    _require(proposal["side_effects"] == {
        "imports_no_pcc_backend": True,
        "no_dataset_load": True, "no_graph_build": True,
        "no_fits": True, "no_native_compilation": True,
    }, "the P1 design must be inert")

    return {
        "schema": "pypcc-p1-design-review-report/v1",
        "design_integrity": "PASS",
        "phase": "P1_PROPOSAL_ONLY",
        "draft_content_sha256_not_an_approval": _canonical_digest(proposal),
        "prospective_fits_if_later_approved_and_feasible":
            experimental["number_of_fits_if_approved_and_feasible"],
        "datasets_selected": 0,
        "scientific_protocol_frozen": False,
        "execution_authorized": False,
        "execution_preflight_sha256": None,
        "model_fits_executed": 0,
        "P1_ready_to_freeze": False,
        "P2_ready": False,
        "P3_ready": False,
        "blockers": list(approval["required_before_P1_freeze"]) +
                    list(approval["required_before_any_real_fit"]),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--proposal", type=Path, default=PROPOSAL)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    proposal = json.loads(args.proposal.read_text(encoding="utf-8"))
    report = review(proposal)
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
