"""Validate limits and recorded facts of seven additional PCC experiment sources.

Stdlib-only. Does not authenticate paper PDFs, exhaustively check literature,
build a graph, draw a label mask, import PCC or authorize real fits.
"""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REGISTER = ROOT / "docs/current_p1_additional_public_papers_2026-11-12.json"
EXPECTED = {
    "BREVE_COMPLEX_2009": (2009, "complex2009.pdf", {"Iris", "Wine",
                                                     "synthetic banana-shaped distributions"}),
    "BREVE_IJCNN_2010": (2010, "ijcnn2010.pdf",
                         {"synthetic four-class modular networks"}),
    "BREVE_TKDE_2012": (2012, "ieee-tkde-2009.pdf", {
        "synthetic banana","synthetic Highleyman","synthetic Lithuanian",
        "synthetic Gaussian","g241c","g241d","Digit1","USPS","COIL",
        "BCI","Text","USPS Handwritten Digit full",
        "COIL 2000 Insurance Company","Letter Recognition"}),
    "BREVE_IJCNN_2012_CONCEPT_DRIFT": (2012, "ijcnn2012.pdf",
                                       {"synthetic dynamic four-class Gaussian streams"}),
    "BREVE_IJCNN_2013_ACTIVE": (2013, "ijcnn2013.pdf", {
        "Iris","Wine","Digit1","USPS","COIL2","g241c","COIL"}),
    "BREVE_MLSP_2016": (2016, "PaperMSLP2016.pdf", {
        "Iris","Wine","Digit1","COIL","USPS","g241c"}),
    "BREVE_PCC_GCN_ARXIV_2026": (2026, "2609.22053", {
        "Cora","CiteSeer","PubMed","Amazon-Computers","Amazon-Photos",
        "DBLP","BlogCatalog","Flickr","Amazon-Ratings","Roman-Empire"}),
}


class ExposureScopeInvalid(ValueError):
    pass


def require(ok, msg):
    if not ok:
        raise ExposureScopeInvalid(msg)


def review(register):
    require(isinstance(register, dict) and
            register.get("schema") ==
            "pypcc-P1-seven-additional-public-experimental-papers/v1" and
            register.get("status") ==
            "SEVEN_SCOPED_PUBLIC_SOURCES_ONLY_NOT_EXHAUSTIVE_NOT_NOVELTY_APPROVAL",
            "Unexpected source register or approval claim")
    sources = register.get("reviewed_experimental_sources")
    require(type(sources) is list and len(sources) == 7 and
            {r["id"] for r in sources} == set(EXPECTED),
            "Exactly seven distinct public sources required")
    for item in sources:
        year, uri_end, datasets = EXPECTED[item["id"]]
        require(
            item["year"] == year and item["url"].startswith("https://") and
            item["url"].endswith(uri_end) and
            len(item["experimental_datasets"]) == len(datasets) and
            set(item["experimental_datasets"]) == datasets and
            item["candidate_names_present"] == [] and
            len(item["evidence_location"]) >= 20 and len(item["note"]) >= 20,
            "Historical experiment names/source changed without review",
        )
        if item["id"] == "BREVE_PCC_GCN_ARXIV_2026":
            require(item["type"] == "arxiv_preprint_not_peer_review_claim",
                    "Arxiv paper cannot be promoted to peer-reviewed source")
    require(
        register["checked_extra_source_count"] == 7 and
        register["checked_extra_published_work_count"] == 6 and
        register["checked_extra_public_preprint_count"] == 1 and
        register["previous_three_paper_register"] ==
        "docs/current_p1_public_literature_exposure_2026-11-12.json" and
        register["provisional_candidate_ids_not_selected"] == [212,850,292] and
        register["provisional_candidate_names"] ==
        ["Vertebral Column","Raisin","Wholesale Customers"] and
        register["negative_claim"] ==
        "NONE_OF_THREE_CANDIDATES_NAMED_IN_REVIEWED_EXPERIMENTAL_DATASET_LISTS_OF_THESE_SEVEN_SOURCES_ONLY",
        "Negative-name claim must remain exactly limited to these sources",
    )
    require(
        len(register["limits"]) >= 5 and
        register["historical_20_dataset_campaign_exhaustive"] is False and
        register["all_prior_research_checked"] is False and
        register["all_candidate_aliases_checked"] is False and
        register["clinical_or_subject_group_independence_proved"] is False and
        register["scientific_P1_approved"] is False and
        register["P1_scientific_sha256"] is None and
        register["P2_execution_sha256"] is None and
        register["dataset_selected_ids"] == [] and
        register["PCC_real_data_fits_executed"] == 0 and
        register["execution_authorized"] is False,
        "A bounded historical review cannot authorize a real PCC study",
    )
    canonical = json.dumps(register, sort_keys=True, ensure_ascii=True,
                           allow_nan=False, separators=(",", ":"))
    return {
        "scope_integrity": "PASS",
        "additional_published_sources": 6,
        "additional_arxiv_preprints": 1,
        "total_scoped_sources_with_prior_three": 10,
        "ten_scoped_experimental_sections_without_candidate_names": True,
        "all_prior_research_or_aliases_checked": False,
        "candidate_dataset_novelty_proved": False,
        "patient_customer_grain_independence_proved": False,
        "source_register_sha256_NOT_scientific_P1_approval":
            hashlib.sha256(canonical.encode("ascii")).hexdigest(),
        "scientific_P1_approved": False,
        "P2_execution_sha256": None,
        "PCC_fits_executed": 0,
        "execution_authorized": False,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--register", type=Path, default=REGISTER)
    p.add_argument("--out", type=Path)
    args = p.parse_args()
    result = review(json.loads(args.register.read_text(encoding="utf-8")))
    rendered = json.dumps(result, sort_keys=True, indent=2) + "\n"
    if args.out is not None:
        args.out.write_text(rendered, encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
