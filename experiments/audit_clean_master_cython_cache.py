"""Compare separately compiled untouched-master and candidate Cython binaries.

No experimental PCC high-level API is required; the script invokes exactly
the same direct Cython propagation signature in both independent checkouts.
Protocol: clean_master_cython_cache_protocol.md.
"""
import argparse
import csv
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

SEEDS = (7703, 7807, 7901, 8009)
DATASETS = ("digits", "breast_cancer")
OSES = ("ubuntu-latest", "windows-latest")
VARIANTS = ("master", "candidate_off", "candidate_on")
POLICIES = (
    ("forced_monitor_20k", 20000, True, 2000000, 3),
    ("original_potential_100k", 100000, True, 2000, 1),
    ("no_stop_5k", 5000, False, 2000, 1),
)
STATE = (
    "predictions", "positions", "particle_labels",
    "strengths", "distances", "dominance", "ownership",
)
FIELDS = (
    "runner_os", "dataset", "seed", "policy", "repeat", "variant",
    "order_index", "iterations", "stop_reason", "checks", "stop_max",
    "stalled_checks", "elapsed_seconds", "source_commit",
    "extension_sha256", "extension_file", "master_seconds",
    "candidate_off_seconds", "candidate_on_seconds",
    "candidate_off_saving_percent", "candidate_on_saving_percent",
    "all_three_states_equal", "all_three_stop_metadata_equal",
    "real_cython_native_extension", "reciprocal_graph",
)


def expected_stop_max(n, p, k_nn, es_chk, degrees):
    if not np.any(degrees):
        return 1
    return max(1, round((n / (p * max(k_nn, 1))) * round(es_chk * .1)))


def canonical_direct_kernel_input(dataset, seed):
    from sklearn.datasets import load_digits, load_breast_cancer
    from sklearn.preprocessing import StandardScaler
    from pcc_graph import build_knn_graph

    raw = load_digits() if dataset == "digits" else load_breast_cancer()
    features = StandardScaler().fit_transform(
        raw.data.astype(np.float64))
    truth = np.asarray(raw.target, dtype=np.int64)
    rng = np.random.default_rng(seed)
    labels = np.full(len(truth), -1, dtype=np.int64)
    for cls in np.unique(truth):
        choices = np.flatnonzero(truth == cls)
        labels[rng.choice(choices, size=2, replace=False)] = cls
    graph, degrees = build_knn_graph(features, k_nn=10)
    n = len(labels)
    if (graph.dtype != np.int64 or degrees.dtype != np.int64 or
            graph.shape[0] != n):
        raise AssertionError("Unexpected canonical production graph layout")
    edge_sets = []
    for i in range(n):
        neighbors = list(map(int, graph[i, :int(degrees[i])]))
        if len(set(neighbors)) != len(neighbors) or i in neighbors:
            raise AssertionError("Duplicate/self graph edges")
        edge_sets.append(set(neighbors))
    if any(i not in edge_sets[j] for i,edges in enumerate(edge_sets)
           for j in edges):
        raise AssertionError("Graph contains a directed edge")
    c = len(np.unique(truth))
    homes = np.flatnonzero(labels >= 0).astype(np.int64)
    p = len(homes)
    if p != 2*c:
        raise AssertionError("Expected two labeled homes per class")
    classes = labels[homes].copy()
    strengths = np.ones(p, dtype=np.float64)
    distances = np.full(
        (n,p), min(255,n-1), dtype=np.uint8, order="F")
    distances[homes, np.arange(p)] = 0
    dominance = np.full((n,c),1.0/c,dtype=np.float64,order="F")
    dominance[homes,:] = 0.0
    dominance[homes,classes] = 1.0
    ownership = np.zeros((n,c),dtype=np.float64,order="F")
    width = max(int(degrees.max()),1)
    buffers = [
        np.empty(c,dtype=np.float64), np.empty(c,dtype=np.float64),
        np.empty(width,dtype=np.float64), np.empty(width,dtype=np.float64),
        np.empty(width,dtype=np.float64), np.empty(width,dtype=np.float64),
        1.0/(np.arange(257,dtype=np.float64)+1.0)**2.0,
    ]
    kernel_args = (
        graph, degrees, labels, .5, .1, c,
        np.zeros(c,dtype=np.float64), homes.copy(), classes,
        strengths, distances, dominance, ownership,
        1.0, 2.0, *buffers,
    )
    state = dict(
        positions=kernel_args[7], particle_labels=classes,
        strengths=strengths, distances=distances,
        dominance=dominance, ownership=ownership,
    )
    return kernel_args,state,labels,truth,degrees


def execute_worker(args):
    # The runner changes CWD to the checkout being measured. Put that
    # checkout ahead of this script's directory in sys.path.
    checkout=Path.cwd().resolve()
    sys.path.insert(0,str(checkout))
    import pcc_step
    extension=Path(pcc_step.__file__).resolve()
    if extension.parent != checkout:
        raise AssertionError(
            f"Imported extension from wrong checkout: {extension} != {checkout}")
    if extension.suffix.lower() not in (".so",".pyd"):
        raise AssertionError("Expected an actual compiled extension")
    # Do not infer the Cython signature from __doc__: different generated
    # extensions may omit it. Explicit opt-in invocation below verifies the
    # candidate accepts the flag, while native binary hashes and separate
    # source revisions prove which compiled module was loaded.

    policy = next(x for x in POLICIES if x[0]==args.policy)
    name,cap,monitoring,es_chk,_ = policy
    kernel_args,state,labels,truth,degrees = canonical_direct_kernel_input(
        args.dataset,args.seed)
    n=len(labels)
    stop_max=expected_stop_max(
        n,int(np.count_nonzero(labels>=0)),10,es_chk,degrees)
    if monitoring and name=="forced_monitor_20k" and (
            stop_max <= cap//10+1):
        raise AssertionError("The fixed-work monitor could stop")
    kwargs=dict(return_info=True,seed=args.seed)
    if args.variant=="candidate_on":
        kwargs["experimental_rowmax_cache"]=True
    start=time.perf_counter()
    info=pcc_step.pcc_propagate(
        *kernel_args,cap,int(monitoring),es_chk,stop_max,**kwargs)
    elapsed=time.perf_counter()-start
    if not isinstance(info,dict):
        raise AssertionError("Direct compiled Cython did not return metadata")
    steps=int(info["effective_iterations"])
    if not (0 < steps <= cap):
        raise AssertionError("Invalid effective iteration count")
    if ((name in ("forced_monitor_20k","no_stop_5k")) and steps!=cap):
        raise AssertionError("Unexpected fixed-work termination")
    if not np.all(np.isfinite(state["dominance"])):
        raise AssertionError("Non-finite dominance")
    predicted=np.argmax(state["dominance"],axis=1)
    predicted[labels>=0]=labels[labels>=0]
    state["predictions"]=predicted
    output=Path(args.output).resolve()
    output.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(
        str(output.with_suffix(".npz")),
        **{k:np.asarray(v) for k,v in state.items()})
    commit=subprocess.check_output(
        ["git","rev-parse","HEAD"],text=True,cwd=checkout).strip()
    payload=dict(
        variant=args.variant,policy=name,dataset=args.dataset,
        seed=args.seed,elapsed_seconds=elapsed,
        info=info,n_nodes=n,n_classes=len(np.unique(truth)),
        n_particles=int(np.count_nonzero(labels>=0)),
        source_commit=commit,
        extension_file=str(extension),
        extension_sha256=hashlib.sha256(extension.read_bytes()).hexdigest(),
        real_cython_native_extension=True,reciprocal_graph=True,
    )
    output.with_suffix(".json").write_text(
        json.dumps(payload,sort_keys=True,indent=2),encoding="utf-8")
    print(f"{args.variant}: {args.dataset}/{args.seed}/{name}: "
          f"{elapsed:.6f}s, {steps} iterations, binary {extension.name}",
          flush=True)


def compare_three(entries):
    infos=[e[1]["info"] for e in entries]
    if not (infos[0]==infos[1]==infos[2]):
        raise AssertionError(f"Cross-binary stop metadata differs: {infos}")
    with np.load(entries[0][0],allow_pickle=False) as baseline:
        for path,metadata in entries[1:]:
            with np.load(path,allow_pickle=False) as alternative:
                for key in STATE:
                    if key not in baseline or key not in alternative or not (
                            np.array_equal(baseline[key],alternative[key])):
                        raise AssertionError(
                            f"Cross-binary state mismatch: {key}, "
                            f"case={metadata['dataset']}/{metadata['seed']}/"
                            f"{metadata['policy']}, variant={metadata['variant']}")
    base_commit=entries[0][1]["source_commit"]
    if entries[1][1]["source_commit"]!=entries[2][1]["source_commit"]:
        raise AssertionError("Candidate binary checkouts have different SHAs")
    if entries[1][1]["extension_sha256"]!=entries[2][1]["extension_sha256"]:
        raise AssertionError("Candidate variants imported different binaries")
    if base_commit==entries[1][1]["source_commit"]:
        raise AssertionError("Baseline and candidate commits are identical")
    if entries[0][1]["extension_sha256"] == entries[1][1]["extension_sha256"]:
        raise AssertionError("Cross-revision native binaries are identical")
    return base_commit,entries[1][1]["source_commit"]


def orchestrate(args):
    baseline=Path(args.baseline_dir).resolve()
    candidate=Path(args.candidate_dir).resolve()
    output=Path(args.output_dir).resolve()
    output.mkdir(parents=True,exist_ok=True)
    script=(candidate/"experiments"/
            "audit_clean_master_cython_cache.py").resolve()
    if not script.is_file() or baseline==candidate:
        raise AssertionError("Missing experiment script or split checkouts")

    with open(args.csv,"w",newline="",encoding="utf-8") as file:
        writer=csv.DictWriter(file,fieldnames=FIELDS)
        writer.writeheader()
        file.flush()
        for index,seed in enumerate(SEEDS):
            for policy,cap,monitored,es_chk,repeats in POLICIES:
                for rep in range(repeats):
                    order=VARIANTS[(index+rep)%3:]+VARIANTS[:(index+rep)%3]
                    data={}
                    for order_idx,variant in enumerate(order):
                        checkout=baseline if variant=="master" else candidate
                        prefix=(output/f"{args.runner_os}_{args.dataset}_"
                                f"{seed}_{policy}_{rep}_{variant}")
                        command=[
                            sys.executable,str(script),"worker",
                            "--variant",variant,
                            "--dataset",args.dataset,
                            "--policy",policy,
                            "--seed",str(seed),
                            "--output",str(prefix),
                        ]
                        process=subprocess.run(
                            command,cwd=checkout,check=False,
                            capture_output=True,text=True)
                        if process.returncode!=0:
                            raise RuntimeError(
                                f"Worker crashed: {command} "
                                f"stdout={process.stdout[-4000:]} "
                                f"stderr={process.stderr[-4000:]}")
                        data[variant]=(
                            prefix.with_suffix(".npz"),
                            json.loads(prefix.with_suffix(".json").read_text(
                                encoding="utf-8")),
                            order_idx,
                        )
                    entries=[data[v][:2] for v in VARIANTS]
                    master_sha,candidate_sha=compare_three(entries)
                    elapsed={v:data[v][1]["elapsed_seconds"] for v in VARIANTS}
                    base=elapsed["master"]
                    offsaving=100*(1-elapsed["candidate_off"]/base)
                    onsaving=100*(1-elapsed["candidate_on"]/base)
                    for variant in VARIANTS:
                        path,metadata,order_idx=data[variant]
                        info=metadata["info"]
                        writer.writerow(dict(
                            runner_os=args.runner_os,dataset=args.dataset,
                            seed=seed,policy=policy,repeat=rep,
                            variant=variant,order_index=order_idx,
                            iterations=info["effective_iterations"],
                            stop_reason=info["stop_reason"],
                            checks=info["checks"],
                            stop_max=info["stop_max"],
                            stalled_checks=info["stalled_checks"],
                            elapsed_seconds=elapsed[variant],
                            source_commit=metadata["source_commit"],
                            extension_sha256=metadata["extension_sha256"],
                            extension_file=metadata["extension_file"],
                            master_seconds=base,
                            candidate_off_seconds=elapsed["candidate_off"],
                            candidate_on_seconds=elapsed["candidate_on"],
                            candidate_off_saving_percent=offsaving,
                            candidate_on_saving_percent=onsaving,
                            all_three_states_equal=True,
                            all_three_stop_metadata_equal=True,
                            real_cython_native_extension=True,
                            reciprocal_graph=True,
                        ))
                    file.flush()
                    print(f"{args.runner_os}/{args.dataset} {seed} {policy} "
                          f"rep={rep}: master→cache={onsaving:+.2f}% "
                          f"master→default-off={offsaving:+.2f}% "
                          "three-way exact state/stop PASS",flush=True)
    print(f"{args.runner_os}/{args.dataset}: full cross-revision "
          "three-way cohort passed",flush=True)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest="command",required=True)
    w=sub.add_parser("worker")
    w.add_argument("--variant",choices=VARIANTS,required=True)
    w.add_argument("--dataset",choices=DATASETS,required=True)
    w.add_argument("--policy",choices=[x[0] for x in POLICIES],required=True)
    w.add_argument("--seed",type=int,choices=SEEDS,required=True)
    w.add_argument("--output",required=True)
    r=sub.add_parser("orchestrate")
    r.add_argument("--runner-os",choices=OSES,required=True)
    r.add_argument("--dataset",choices=DATASETS,required=True)
    r.add_argument("--baseline-dir",required=True)
    r.add_argument("--candidate-dir",required=True)
    r.add_argument("--output-dir",required=True)
    r.add_argument("--csv",required=True)
    args=p.parse_args(argv)
    execute_worker(args) if args.command=="worker" else orchestrate(args)


if __name__=="__main__":
    main()
