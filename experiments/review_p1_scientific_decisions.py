"""Strict stdlib-only review of the UNAPPROVED P1 scientific decision packet.

This does not download datasets, build graphs, run a particle, assign labels,
sample an RNG, compile a backend, approve science, or create P2 credentials.
It checks that the reviewer-facing candidate/preprocessing/graph proposals
are consistent with the existing UNAPPROVED P1 design and current master code.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]
PACKET = ROOT / "docs/current_p1_scientific_decision_packet_2026-11-12.json"
DRAFT = ROOT / "docs/current_p1_prospective_design_draft_2026-11-12.json"
GRAPH_SOURCE = ROOT / "pcc_graph.py"
GRAPH_SHA = "3e8943462555b4feafbd162125b51e41f29d7ab7"
EXPECTED_IDS = (212, 850, 292)
EXPECTED_SUPPORT = {
    212: {"Hernia": 60, "Normal": 100, "Spondylolisthesis": 150},
    850: {"Besni": 450, "Kecimen": 450},
    292: {"1": 298, "2": 142},
}
EXPECTED_SOURCES = {
    212: (
        "899ed1d137ceb4c36d59fa35f3cbab39ec3b78fd3ca1fdaf2fae155459c1372f",
        "cf768854cf4cf42b3a3b792a239a81ebfd4fbfafc25fbe2a19d15b3f73249879",
        310, 6,
    ),
    850: (
        "5d516c040923fd154ecf85e31b6e1b5a096724fa0e68d588c33fb5a2f0c3e5db",
        "9370b5f749a572a0bbeae15c93bf2e15c859b06d2626ae401b844c0a925fa553",
        900, 7,
    ),
    292: (
        "647e6a61683ed23f48c7b04e1e2be78835eca8758b32267603fc42a1479dbfb8",
        "c3d018c643565b85cee733c4a2ac76dd76e080e857cb23f0ccfcc2e15a6c17ef",
        440, 6,
    ),
}
EXPECTED_MILESTONES = [0, 1, 10, 100, 1000, 5000, 25000, 50000]


class ScientificDecisionInvalid(ValueError):
    pass


def require(ok, reason):
    if not ok:
        raise ScientificDecisionInvalid(reason)


def git_blob_sha1(data):
    return hashlib.sha1(
        b"blob " + str(len(data)).encode("ascii") + b"\x00" + data
    ).hexdigest()


def canonical_sha256(value):
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, ensure_ascii=True, allow_nan=False,
        separators=(",", ":"),
    ).encode("ascii")).hexdigest()


def check_graph_source(content):
    require(git_blob_sha1(content) == GRAPH_SHA,
            "Production graph code is not the pinned master source")
    module = ast.parse(content)
    functions = [x for x in module.body if isinstance(
        x, (ast.FunctionDef, ast.AsyncFunctionDef)
    ) and x.name == "build_knn_graph"]
    require(len(functions) == 1, "Expected original graph API exactly once")
    fn = functions[0]
    require([x.arg for x in fn.args.args] ==
            ["data", "k_nn", "metric", "p", "n_jobs"],
            "Graph argument order changed")
    require([ast.literal_eval(v) for v in fn.args.defaults] ==
            [10, "minkowski", 2, None],
            "Current production graph defaults have changed")
    require(b"algorithm=\"auto\"" in content and
            b"adj = adj + adj.T" in content and
            b"neib_list = np.full((n_nodes, max_deg), -1" in content,
            "Expected auto/symmetric/dense-padded graph construction changed")
    # These text checks are not a substitute for P2 actual graph computation.
    return True


def review(packet, scientific_draft, graph_content):
    require(type(packet) is dict, "Decision packet must be an object")
    require(packet.get("schema") == "pypcc-p1-protocol-decision-packet/v0.1",
            "Unrecognized proposal packet schema")
    require(packet.get("status") ==
            "PROPOSAL_FOR_INDEPENDENT_SCIENTIFIC_REVIEW_NOT_APPROVED",
            "Scientific decision packet cannot claim approval")
    require(packet["linked_current_master_commit"] ==
            "ea69f690208620d5563c8efece6b2e04369f52c0",
            "Original master source base changed")
    require(packet["evidence_head_commit"] ==
            "096a6c127eeeaa6b37171c99824a7a80a040aa72",
            "Unreviewed source-quality evidence base changed")
    require(packet["linked_existing_draft"] ==
            "docs/current_p1_prospective_design_draft_2026-11-12.json",
            "Unexpected linked P1 draft")
    for field in ("scientific_protocol_frozen", "execution_authorized"):
        require(packet.get(field) is False,
                "Packet may not authorize P1 or execution: " + field)
    for field in ("scientific_protocol_sha256",
                  "P2_execution_preflight_sha256"):
        require(packet.get(field) is None,
                "Packet cannot contain a scientific/P2 hash")
    require(packet["dataset_registry_selected"] == [] and
            packet["selected_dataset_count"] == 0 and
            packet["real_PCC_fits_executed"] == 0,
            "No dataset is selected and no fit is authorized")
    for flag in ("no_dataset_download_in_this_document",
                 "no_graph_construction_in_this_document",
                 "no_model_fit_in_this_document"):
        require(packet[flag] is True, "Source-only proposal must remain inert")

    options = packet["candidate_shortlist_proposed_not_selected"]
    require([x["dataset_id"] for x in options] == list(EXPECTED_IDS),
            "Unexpected or reordered shortlist: independent review required")
    require(len(set(x["domain"] for x in options)) == 3,
            "Three distinct application domains required for shortlist")
    for entry in options:
        dataset_id = entry["dataset_id"]
        expected = EXPECTED_SOURCES[dataset_id]
        require((entry["source_zip_sha256"],
                 entry["selected_file_sha256"],
                 entry["n_rows"], entry["n_features"]) == expected,
                f"Unreviewed source identity or shape for {dataset_id}")
        require(entry["class_support"] == EXPECTED_SUPPORT[dataset_id] and
                sum(entry["class_support"].values()) == entry["n_rows"],
                f"Class supports do not match source for {dataset_id}")
        require(entry["source_url"].startswith(
            "https://archive.ics.uci.edu/static/public/" +
            str(dataset_id) + "/"), "Candidate must point to UCI source")
        require(len(entry["scientific_blockers"]) >= 2,
                "Independent source/ethics review is still missing")
        require(
            entry["proposed_preprocessing"] in
            packet["preprocessing_candidate_rule_not_approved"],
            "Unknown proposed preprocessing algorithm",
        )
    wholesale = options[2]
    require(wholesale["proposed_target"].startswith("Channel") and
            wholesale["source_metadata_target"].startswith("Region") and
            wholesale["excluded_X"] == ["Region", "Channel"] and
            wholesale["proposed_X"] ==
            ["Fresh", "Milk", "Grocery", "Frozen",
             "Detergents_Paper", "Delicassen"],
            "Source metadata Region and derived target Channel must remain distinct")
    require(options[0]["selected_file_name"] == "column_3C_weka.arff",
            "Vertebral study must not silently switch to two-class task")
    require(options[1]["selected_file_name"] == "Raisin_Dataset.xlsx",
            "Raisin study sheet identity changed")

    not_shortlisted = packet["not_shortlisted"]
    require([x["dataset_id"] for x in not_shortlisted] == [545, 379] and
            "grain" in not_shortlisted[0]["reason"].lower() and
            "categorical" in not_shortlisted[1]["reason"].lower(),
            "Document domain redundancy and categorical-code conflicts")
    prep = packet["preprocessing_candidate_rule_not_approved"]
    require(prep["reviewed_and_approved"] is False and
            "ddof=0" in prep["population_zscore_original_numeric_features"] and
            "log1p" in prep["log1p_nonnegative_then_population_zscore"],
            "Preprocessing must be explicit and unapproved")
    graph = packet["graph_candidate_rule_not_approved"]
    require(graph["source_file"] == "pcc_graph.py" and
            graph["source_git_blob_sha1"] == GRAPH_SHA and
            graph["call"] ==
            "build_knn_graph(X, k_nn=10, metric='minkowski', p=2, n_jobs=1)",
            "Graph construction differs from proposed original master call")
    require(graph["undirected_symmetric"] is True and
            graph["dense_padded_minus_one"] is True and
            graph["cross_platform_graph_identity_not_yet_verified"] is True and
            graph["reviewed_and_approved"] is False,
            "Do not assume graph identity or change graph representation")
    require(len(graph["before_P2_required"]) >= 5,
            "Graph equivalence and budget checklist is incomplete")
    check_graph_source(graph_content)

    require(scientific_draft["status"] == "DRAFT_NOT_APPROVED" and
            scientific_draft["dataset_registry"]["selected"] == [] and
            scientific_draft["approval"]["scientific_protocol_frozen"] is False and
            scientific_draft["approval"]["scientific_protocol_sha256"] is None and
            scientific_draft["approval"]["execution_authorized"] is False,
            "Linked original P1 design must remain unapproved")
    d = scientific_draft["experimental_design"]
    proposed = packet["scientific_design_preserved_from_P1_draft"]
    require(proposed["backend_count"] == len(d["backend_conditions"]) == 3 and
            proposed["mask_repeats_per_dataset"] ==
            d["label_mask_replicates_proposed"] == 8 and
            proposed["labeled_per_class"] ==
            d["label_budget_per_class_proposed"] == 3 and
            proposed["horizon_iterations"] ==
            d["max_iterations_proposed"] == 50000 and
            proposed["early_stop"] is d["early_stop"] is False and
            proposed["milestones"] == d["milestones_proposed"] ==
            EXPECTED_MILESTONES and
            proposed["primary_flip_window"] ==
            [d["primary_comparison"]["from_iteration"],
             d["primary_comparison"]["to_iteration"]] == [5000, 50000] and
            proposed["proposed_fit_count_if_all_approved"] ==
            d["number_of_fits_if_approved_and_feasible"] == 72,
            "P1 trial design changed without independent scientific review")
    require(len(packet["reviewer_decisions_remaining"]) >= 7,
            "Review checklist must be comprehensive")

    return {
        "schema": "pypcc-p1-scientific-decision-review/v1",
        "proposal_checks": "PASS",
        "proposed_candidate_ids_not_selected": list(EXPECTED_IDS),
        "source_provenance_pinned_for_review": True,
        "original_graph_source_blob_checked": True,
        "review_pending_count": len(packet["reviewer_decisions_remaining"]),
        "packet_sha256_NOT_scientific_protocol_approval":
            canonical_sha256(packet),
        "P1_scientifically_frozen": False,
        "P1_ready_to_freeze": False,
        "P2_execution_preflight_sha256": None,
        "PCC_fits_executed": 0,
        "execution_authorized": False,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--packet", type=Path, default=PACKET)
    p.add_argument("--draft", type=Path, default=DRAFT)
    p.add_argument("--graph-source", type=Path, default=GRAPH_SOURCE)
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()
    report = review(
        json.loads(args.packet.read_text(encoding="utf-8")),
        json.loads(args.draft.read_text(encoding="utf-8")),
        args.graph_source.read_bytes(),
    )
    rendered = json.dumps(report, sort_keys=True, indent=2) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(rendered, encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
