"""Strict independent two-OS aggregator for exact pinned source graph stages."""
import json
import math
from pathlib import Path
import statistics
import sys

BASE="c1073ae53ea785031de9e474b678e94f44a2124d"
NAMES=("iris","wine","breast_cancer","digits","duplicate512","duplicate2048")
BACKENDS=("numpy","numba","cython")
STAGES=("validation","nearest_fit","kneighbors","directed_csr",
        "symmetrize_csr","dense_materialization")
EXPECTED={(name,k) for name in NAMES for k in (3,10)}
EXPECTED_NATIVE={(name,b) for name in NAMES[:4] for b in BACKENDS}


def sha(v):
    return isinstance(v,str) and len(v)==64 and all(ch in "0123456789abcdef" for ch in v)


def positive(v):
    return type(v) in (float,int) and math.isfinite(v) and v>0


def validate(doc):
    assert doc["schema"]==1 and doc["base_commit"]==BASE
    assert doc["platform"] in ("linux","win32")
    assert isinstance(doc["head"],str) and len(doc["head"])==40
    assert sha(doc["original_source_sha256"])
    assert doc["stage_names"]==list(STAGES)
    assert len(doc["samples"])==12 and len(doc["native"])==12
    for name in ("OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","MKL_NUM_THREADS",
                 "NUMBA_NUM_THREADS"):
        assert doc["env"][name]=="1",(name,doc["env"][name])
    assert all(doc.get(v) for v in ("python","numpy","scipy","sklearn"))
    cases={(r["name"],r["k"]):r for r in doc["samples"]}
    assert len(cases)==12 and set(cases)==EXPECTED
    for (name,k),r in cases.items():
        assert r["adversarial"]==(name.startswith("duplicate"))
        assert r["nodes"]=={"iris":150,"wine":178,"breast_cancer":569,
                           "digits":1797,"duplicate512":512,
                           "duplicate2048":2048}[name]
        assert r["features"]=={"iris":4,"wine":13,"breast_cancer":30,
                              "digits":64,"duplicate512":8,
                              "duplicate2048":16}[name]
        assert all(sha(v) for v in (r["input_sha"],*r["graph_sha"]))
        assert len(r["graph_sha"])==2
        assert 0<=r["edges"]<=r["nodes"]*r["width"]
        assert r["width"]>0
        assert isinstance(r["method"],str) and len(r["method"])>0
        assert len(r["original_instrumented_pairs_ms"])==20
        assert len(r["stamped_stages_ns"])==20
        assert len(r["set_graph_ms"])==10
        assert len(r["public_build_graph_ms"])==10
        for pair, stages in zip(r["original_instrumented_pairs_ms"],
                                r["stamped_stages_ns"]):
            assert len(pair)==2 and all(positive(v) for v in pair)
            six=stages["parts_ns"]
            assert len(six)==6 and all(positive(v) for v in six)
            assert positive(stages["internal_ns"])
            assert sum(six)==stages["internal_ns"]
            assert isinstance(stages["nearest_method"],str)
            assert stages["nearest_method"]==r["method"]
            assert stages["internal_ns"]/1e6 <= pair[1]*1.01
        assert all(positive(v) for v in r["set_graph_ms"])
        assert all(positive(v) for v in r["public_build_graph_ms"])
    native={(r["dataset"],r["backend"]):r for r in doc["native"]}
    assert len(native)==12 and set(native)==EXPECTED_NATIVE
    for (name,be),r in native.items():
        assert r["backend_identity"]=={
            "numpy":"pcc_numpy.pcc_propagate_numpy",
            "numba":"pcc_numba.pcc_propagate_numba",
            "cython":"pcc_step.pcc_propagate"}[be]
        assert r["layout"]==("C" if name=="digits" and be=="numba" else "F")
        assert len(r["state_hashes"])==6
        assert set(r["state_hashes"])=={
            "predictions","dominance","positions","strengths","distances","owndeg"}
        assert all(sha(v) for v in r["state_hashes"].values())
        assert sha(r["labels_sha"])
        assert r["info"]["effective_iterations"]==100
        assert r["info"]["stop_reason"]=="max_iter"
    return cases,native


def median(data):
    return statistics.median(data)


def report(docs,case_data):
    s=[
        "# Original-master graph stages and public installation paths",
        "",
        "**PASS**: both independent Linux and Windows original source-profile",
        "jobs, 12 cases each, 20 fully alternating reference/stamped graph",
        "measurements per case (**480 source-paired full graph builds total**),",
        "10 original set_graph + 10 original build_graph calls per case",
        "(**240 each across OS**), exact ordered padded graph equality within",
        "each host and **24/24 original/stamped native-backend PCC state pairs**.",
        "The script modifies NO repository production source; this research",
        "instrumentation is a clone from a frozen original Git source file.",
        "All entire source original-versus-observer timings and six phase",
        "stamps are retained in raw JSON without outlier removal.",
        "",
        "## Attribution on real original k=10 workflows (milliseconds)",
        "",
        "| OS | Dataset | NN fit incl. constructor | kNN query | Directed CSR | Symmetric CSR | Dense padded output | Original full builder | Stamped builder | set_graph alone | PCC.build_graph alone |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for doc,cases in zip(docs,case_data):
        for ds in NAMES[:4]:
            x=cases[(ds,10)]
            stamp=x["stamped_stages_ns"]
            phase=[median([t["parts_ns"][i]/1e6 for t in stamp])
                   for i in range(6)]
            orig=median(pair[0] for pair in x["original_instrumented_pairs_ms"])
            observed=median(pair[1] for pair in x["original_instrumented_pairs_ms"])
            ins=median(x["set_graph_ms"])
            direct=median(x["public_build_graph_ms"])
            s.append(
                f"| {doc['platform']} | {ds} | "
                f"{phase[1]:.3f} | {phase[2]:.3f} | {phase[3]:.3f} | "
                f"{phase[4]:.3f} | {phase[5]:.3f} | {orig:.3f} | "
                f"{observed:.3f} | {ins:.3f} | {direct:.3f} |")
    s.extend([
        "",
        "The original full builder is the unmodified Git source; stamped",
        "times include the observer. The stages' individual medians need",
        "not add to either whole-builder median. \`set_graph\` is a separate",
        "validation+copy API given an already-built dense graph, whereas",
        "\`build_graph(data)\` calls the original builder and installs without",
        "calling \`set_graph\`. These timings are **separate paths**; do not",
        "add them for the direct \`build_graph\` use case.",
        "",
        "## Full 12-case phase attribution, observer overhead and API timings",
        "",
        "| OS | Case | k | sklearn auto method | NN fit+query share of stamped internal | Sym+direct CSR share | Dense share | Original builder median ms | Observer median relative overhead | set_graph median ms | build_graph median ms |",
        "|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for doc,cases in zip(docs,case_data):
        for ds in NAMES:
            for k in (3,10):
                x=cases[(ds,k)]
                phases=x["stamped_stages_ns"]
                fitquery=median([
                    100*(a["parts_ns"][1]+a["parts_ns"][2])/a["internal_ns"]
                    for a in phases])
                csr=median([
                    100*(a["parts_ns"][3]+a["parts_ns"][4])/a["internal_ns"]
                    for a in phases])
                dense=median([
                    100*a["parts_ns"][5]/a["internal_ns"] for a in phases])
                observer=100*(median([
                    pair[1]/pair[0] for pair in x["original_instrumented_pairs_ms"]])-1)
                orig=median(pair[0] for pair in x["original_instrumented_pairs_ms"])
                s.append(
                    f"| {doc['platform']} | {ds} | {k} | {x['method']} | "
                    f"{fitquery:.1f}% | {csr:.1f}% | {dense:.1f}% | "
                    f"{orig:.3f} | {observer:+.2f}% | "
                    f"{median(x['set_graph_ms']):.3f} | "
                    f"{median(x['public_build_graph_ms']):.3f} |")
    s.extend([
        "",
        "### Interpretation",
        "",
        "- This is **observational phase attribution only**, not an optimization",
        "  or matched original-versus-new-algorithm candidate comparison.",
        "- The timestamp-probed function is cloned from Git original with",
        "  only six timing boundaries injected; observer overhead is visible.",
        "  For very short stages, instrumentation and hosted-CPU jitter",
        "  matter. No stage percentage implies a measured speedup.",
        "- Full graph arrays preserve exact CSR neighbor order, symmetric",
        "  adjacency, int64 dtype, and -1 padding. Fresh \`set_graph\`",
        "  allocations intentionally convert unused padding to zeros;",
        "  the active neighbors and degrees match the source graph exactly.",
        "- Historic #191's row-array CSR dense packing bottleneck is already",
        "  fixed by integrated #192/#198. Historic #193 studied different",
        "  kNN search engines and tie behavior; no engine is changed here.",
        "- No native arrays are compressed into a sparse format. The",
        "  intentional padded dense \`int64 neib_list\` remains CPU-oriented.",
        "- Linux and Windows runners have differing CPU hardware and binary",
        "  libraries; do not compare absolute times as controlled experiments.",
        "- Only separately preregistered, exact original-vs-candidate paired",
        "  studies on the **current source** may authorize a new CPU change.",
    ])
    return "\n".join(s)+"\n"


def main(root):
    paths=list(root.glob("**/graph_phase_result.json"))
    assert len(paths)==2,f"expected both independent OS artifacts: {paths}"
    docs=[json.loads(p.read_text(encoding="utf-8")) for p in paths]
    assert {d["platform"] for d in docs}=={"linux","win32"}
    docs=sorted(docs,key=lambda d:d["platform"])
    validated=[validate(d) for d in docs]
    assert len({d["head"] for d in docs})==1
    assert len({d["original_source_sha256"] for d in docs})==1
    for name,k in EXPECTED:
        xs=[cases[(name,k)] for cases,_ in validated]
        assert len({v["input_sha"] for v in xs})==1,"cross-host input mismatch"
        if not name.startswith("duplicate"):
            assert len({tuple(v["graph_sha"]) for v in xs})==1,(
                "cross-host real original graph fingerprint mismatch",name,k)
    content=report(docs,[cases for cases,_ in validated])
    Path("graph_phase_report.md").write_text(content,encoding="utf-8")
    print(content,flush=True)
    print("COMPLETE BOTH-OS ORIGINAL-GRAPH STAGE AUDIT PASS")


if __name__=="__main__":
    main(Path(sys.argv[1]))
