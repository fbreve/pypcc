"""P2-01A: source-only Git identity and runner metadata, NEVER execution GO.

Uses stdlib and read-only git commands; it neither imports PyPCC/NumPy/Numba,
builds native extensions, downloads data, draws masks, constructs a graph, nor
runs a fit. A PASS proves source Git objects only, never installed versions,
native code ABI/parity, memory budget or P2 execution authorization.
"""
import argparse
import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path
import shutil
import subprocess
import sys
import sysconfig

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "docs/current_p2_01_source_identity_2026-11-12.json"
P1_MANIFEST = ROOT / "docs/current_p1_scientific_sha_manifest_2026-11-12.json"
P1_PAYLOAD = ROOT / "docs/current_p1_frozen_scientific_protocol_2026-11-12.json"
P1_SHA = "f0111f153ade3d72191c107ace039e1835e5c27db2cf4b3df8d610b41b2f4166"
MASTER = "ea69f690208620d5563c8efece6b2e04369f52c0"
HEADS = {
    "numpy": ("6d908e78878c4922674277e3313dbfe7f7924146", 253),
    "numba": ("4c8e6df4e01d7a4dfc4efc31eb73db04d6e212e5", 255),
    "cython": ("af34f50ebabb0a7204d3f30679adaf3f00169742", 256),
    "source_p0": ("fa59ef20c235cf03075bdcc09d1d182bd5ca06fd", 257),
}
EXPECTED_PRODUCTION = {
    "VERSION": "f0bb29e76388856b273698ae6064b0380ce5e5d2",
    "pcc.py": "01606aaee860c3b9bc84710e7362a2ef81046d45",
    "pcc_graph.py": "3e8943462555b4feafbd162125b51e41f29d7ab7",
    "pcc_metrics.py": "967353fc0f6ee88c8f26bbde6266645e6de9aa3b",
    "pcc_numba.py": "86d21d57ef535bbdb8566f10ace38cb54f93f741",
    "pcc_numpy.py": "edf09a4ac5487327d8bd2b63415b5dd6564209f5",
    "pcc_step.pyx": "021c9a2c34f912e0d8382f05d0e21b9713ffd5fc",
    "pcc_validation.py": "b98510431c2281463988321fb7f1b6b159a5f77d",
    "pyproject.toml": "623c4ee8c6558e9e95af661054abb0ef067d3fa9",
    "setup.py": "622feff882bf9e9b3a5bdd890f4cc5b0a6bf91c6",
}
EXPECTED_OBSERVERS = {
    "numpy": {
        "experiments/live_numpy_milestones.py":
            "d642bd0f329eceb4f3dc3111b5f8c91f0e085cac",
        "tests/test_live_numpy_milestones.py":
            "2dd7b87e54ee5acd458cf56b20adf4516c4a8e60",
    },
    "numba": {
        "experiments/live_numba_milestones.py":
            "e10942e96be623d47d5537d45bd365f366bba548",
        "tests/test_live_numba_milestones.py":
            "ef56733245bdc7226edd6c3567de27343cdd3734",
    },
    "cython": {
        "experiments/build_live_cython.py":
            "d30e9b85419b8f31cd5f677f848357fb121d316f",
        "experiments/live_cython_milestones.py":
            "281daf0e66a567e587d0bd3e21dc34329a234b95",
        "tests/test_live_cython_milestones.py":
            "19cce9e695a0351fe135233ba6c069b6bfba95db",
    },
    "source_p0": {
        "experiments/preflight_three_backend_sources.py":
            "31f7ab76ac68d1a688a60e7f64c0d6045b38866c",
    },
}
EXPECTED_BRANCHES = {
    "numpy": "research/current-numpy-live-milestones-2026-11-11",
    "numba": "research/current-numba-live-milestones-2026-11-12",
    "cython": "research/current-cython-live-milestones-2026-11-12",
    "source_p0": "research/current-three-backend-source-preflight-2026-11-12",
}


class P201Blocked(ValueError):
    pass


def require(condition, reason):
    if not condition:
        raise P201Blocked(reason)


def canonical_sha(document):
    encoded = json.dumps(document, ensure_ascii=True, sort_keys=True,
                         allow_nan=False, separators=(",", ":")).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def verify_manifest_only(m, p1, payload):
    """Cross-check literal immutable repository/source pins, no Git or I/O."""
    require(type(m) is dict and
            m.get("schema") ==
            "pypcc-P2-01-source-identity-and-platform-inventory/v1" and
            m.get("status") ==
            "SOURCE_GIT_BLOB_PINS_VERIFIED_VIA_PUBLIC_REPO_METADATA_RUNTIME_NATIVE_ABI_UNVERIFIED",
            "Invalid P2-01 evidence scope")
    sci = m["parent_science"]
    require(sci["P1_sha256"] == p1["approved_P1_scientific_protocol_sha256"]
            == P1_SHA and
            canonical_sha(payload) == P1_SHA and
            sci["P2_execution_approved"] is False,
            "P1 frozen science SHA mismatch or false P2 authorization")
    require(m["repository"] == "fbreve/pypcc" and
            m["production_master"] == MASTER and
            m["production_version"] == "1.3.0" and
            m["production_file_git_blobs"] == EXPECTED_PRODUCTION,
            "Production source tree differs from pinned P1 base")
    sources = m["research_sources"]
    require(type(sources) is list and len(sources) == len(HEADS) and
            [x["name"] for x in sources] == list(HEADS),
            "Expected three live observers plus source-only #257 audit")
    for src in sources:
        name = src["name"]
        head, pr = HEADS[name]
        require(src["head"] == head and src["pr"] == pr and
                src["branch"] == EXPECTED_BRANCHES[name] and
                src["files"] == EXPECTED_OBSERVERS[name] and
                len(src["native_artifact"]) > 10,
                "Unapproved observer/PR/file source drift for " + name)
    deps = m["declared_build_and_install_constraints"]
    require(
        deps["pyproject_build_system_requires"] ==
        ["setuptools>=61","wheel","Cython>=3.2,<4","numpy>=2,<3"] and
        deps["pyproject_build_backend"] == "setuptools.build_meta" and
        deps["setup_python_requires"] == ">=3.10" and
        deps["setup_install_requires"] ==
        ["numpy>=2,<3","scipy","scikit-learn"] and
        deps["setup_numba_extra"] == ["numba"] and
        deps["original_Cython_build_compiler_flags"] ==
        {"msvc":["/O2"],"unix":["-O3"],"mingw32":["-O3"]} and
        deps["original_native_module"] == "pcc_step" and
        deps["research_isolated_Cython_module"] ==
        "pcc_step_live_research" and
        deps["cython_research_generator_requires_production_blob"] ==
        EXPECTED_PRODUCTION["pcc_step.pyx"] and
        deps["build_provenance"] ==
        "NO_NATIVE_P2_BUILD_OR_BINARY_SHA_HAS_BEEN_VERIFIED" and
        deps["actual_runtime_resolved_package_versions"] ==
        "NOT_YET_FROZEN" and
        deps["numba_dispatcher_native_signatures"] ==
        "NOT_YET_OBSERVED",
        "Misstated dependency constraints or native build provenance")
    lim = m["scope_and_limitations"]
    require(
        lim["P2_01_status"] == "PARTIAL_ONLY_SOURCE_STAGE" and
        lim["P2_06_status"] == "NOT_EXECUTED" and
        lim["runtime_versions"] == "NOT_RESOLVED_OR_LOCKED" and
        lim["binary_identity_and_ABI"] == "NOT_VERIFIED" and
        lim["P2_execution_preflight_SHA256"] is None and
        lim["P2_execution_authorized"] is False and
        lim["PCC_fits_executed"] == 0,
        "Source audit must remain partial and non-executable")
    return True


def git(repo, *arguments):
    proc = subprocess.run(
        ["git", "-C", str(repo), *arguments], check=False,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        encoding="utf-8", errors="replace",
    )
    if proc.returncode:
        raise P201Blocked("Git verification failed: " + " ".join(arguments) +
                          " (" + proc.stderr.strip()[:280] + ")")
    return proc.stdout.strip()


def verify_git_tree(repo, manifest, *, verify_remote=True):
    """All checks operate on Git *object* blobs, not CRLF checkout bytes."""
    require(git(repo, "rev-parse", "--is-inside-work-tree") == "true",
            "Required Git object store is unavailable")
    checked = {}
    commits = {"master": MASTER, **{
        x["name"]: x["head"] for x in manifest["research_sources"]
    }}
    for name, head in commits.items():
        require(git(repo,"rev-parse",head+"^{commit}") == head,
                "Missing exact pinned commit " + name)
    for name, commit in commits.items():
        if name != "master":
            git(repo,"merge-base","--is-ancestor",MASTER,commit)
        pins = EXPECTED_PRODUCTION
        for path, expected in pins.items():
            actual=git(repo,"rev-parse",commit+":"+path)
            require(actual==expected,
                    f"{name} unexpected production Git blob for {path}")
        checked[name] = {"head":commit,"production_files":len(pins)}
    for src in manifest["research_sources"]:
        name, commit=src["name"],src["head"]
        for path, blob in src["files"].items():
            actual=git(repo,"rev-parse",commit+":"+path)
            require(actual==blob,
                    f"{name} observer source Git blob drift: {path}")
        changed=set(git(repo,"diff","--name-only",MASTER,commit).splitlines())
        require(changed and all(
            path.startswith(("experiments/","tests/","docs/",".github/workflows/"))
            for path in changed),
            "Experimental research PR modified a production file: "+name)
        require(set(src["files"]).issubset(changed),
                "Tracked observer source missing from research diff: "+name)
        checked[name]["observer_file_blobs"]=src["files"]
        checked[name]["production_modifications"]=0
        if verify_remote:
            remote=git(repo,"ls-remote","origin",
                       "refs/heads/"+src["branch"])
            require(remote.split("\t")[0]==commit and
                    remote.endswith("refs/heads/"+src["branch"]),
                    "Remote branch changed after P2 source pin: "+name)
            checked[name]["remote_head_unchanged"]=True
    return checked



def read_installed_metadata():
    wanted=("numpy","scipy","scikit-learn","numba","Cython",
            "setuptools","wheel","llvmlite")
    versions={}
    for name in wanted:
        try:
            versions[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name]=None
    return versions


def report(manifest, p1, payload, repo, *, verify_remote=True):
    verify_manifest_only(manifest,p1,payload)
    sources=verify_git_tree(repo,manifest,verify_remote=verify_remote)
    # These are observed CI-runner versions, NOT a locked P2 build.
    installed=read_installed_metadata()
    return {
        "schema":"pypcc-P2-01-source-and-platform-observation/v1",
        "result":"SOURCE_GIT_PINS_PASS_RUNTIME_NATIVE_P2_NOT_READY",
        "P1_scientific_sha256":P1_SHA,
        "source_manifest_sha256_NOT_P2_approval":canonical_sha(manifest),
        "git_sources_checked":sources,
        "runner_observed_not_P2_execution_locked":{
            "python":platform.python_version(),
            "python_implementation":platform.python_implementation(),
            "platform":platform.system(),
            "platform_release":platform.release(),
            "architecture":platform.machine(),
            "python_SOABI":sysconfig.get_config_var("SOABI"),
            "python_extension_suffixes":sysconfig.get_config_var("EXT_SUFFIX"),
            "configured_compiler":sysconfig.get_config_var("CC"),
            "compiler_executables_found":{
                k:bool(shutil.which(k))
                for k in ("gcc","clang","cl","cc")
            },
            "installed_package_metadata_not_imported":installed,
            "package_versions_frozen_for_P2":False
        },
        "source_identification_checked":True,
        "P2_01_entire_gate_passed":False,
        "Cython_research_binary_SHA256":None,
        "Numba_compiled_signature_pins":None,
        "P2_preflight_sha256":None,
        "P2_execution_authorized":False,
        "real_datasets_loaded":0,
        "graphs_or_masks_generated":0,
        "real_PCC_fits_executed":0,
    }


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo",type=Path,default=ROOT)
    ap.add_argument("--no-remote",action="store_true",
                    help="Local-only Git blob check; report cannot claim remote pin")
    ap.add_argument("--output",type=Path)
    args=ap.parse_args()
    obj=json.loads(MANIFEST.read_text(encoding="utf-8"))
    p1=json.loads(P1_MANIFEST.read_text(encoding="utf-8"))
    payload=json.loads(P1_PAYLOAD.read_text(encoding="utf-8"))
    result=report(obj,p1,payload,args.repo,verify_remote=not args.no_remote)
    data=json.dumps(result,sort_keys=True,indent=2)+"\n"
    if args.output:
        args.output.write_text(data,encoding="utf-8")
    print(data)


if __name__=="__main__":
    main()
