"""P2-01B: source-exact native BUILD/IMPORT smoke, WITHOUT PCC propagation.

Invoked ONLY after the CI runner has independently checked original Git blobs
and copied three independent observer source files from their pinned PR heads.
No UCI download, no mask/graph generation, no model fit, no P2 execution GO.
Build artifacts are ephemeral; this logs actual observed ABI and binary SHA.
"""
import argparse
import hashlib
import importlib
import importlib.metadata
import json
from pathlib import Path
import platform
import sys
import sysconfig

from experiments import p2_01_source_platform_preflight as pins

ROOT = Path(__file__).resolve().parents[1]
EXPECTED = {
    "numpy": "d642bd0f329eceb4f3dc3111b5f8c91f0e085cac",
    "numba": "e10942e96be623d47d5537d45bd365f366bba548",
    "cython": "281daf0e66a567e587d0bd3e21dc34329a234b95",
    "cython_generator": "d30e9b85419b8f31cd5f677f848357fb121d316f",
}
FILES = {
    "numpy": ROOT / "experiments/live_numpy_milestones.py",
    "numba": ROOT / "experiments/live_numba_milestones.py",
    "cython": ROOT / "experiments/live_cython_milestones.py",
    "cython_generator": ROOT / "experiments/build_live_cython.py",
}


class NativeProvenanceBlocked(ValueError):
    pass


def require(ok, why):
    if not ok:
        raise NativeProvenanceBlocked(why)


def git_blob_sha(data):
    return hashlib.sha1(b"blob "+str(len(data)).encode("ascii")+
                        b"\x00"+data).hexdigest()


def sha256_file(path):
    dig = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            dig.update(chunk)
    return dig.hexdigest()


def verify_copied_observers():
    proof={}
    for name,path in FILES.items():
        require(path.is_file(), "Pinned observer source missing: "+name)
        contents=path.read_bytes()
        if git_blob_sha(contents) != EXPECTED[name]:
            # If working tree materializes CRLF, Git canonicalizes to LF.
            contents=contents.replace(b"\r\n",b"\n")
        require(git_blob_sha(contents) == EXPECTED[name],
                "Observer source Git blob mismatch: "+name)
        proof[name]={"source_file":str(path.relative_to(ROOT)).replace("\\","/"),
                     "git_blob_sha1":EXPECTED[name]}
    return proof


def metadata_versions():
    names=("numpy","scipy","scikit-learn","numba","llvmlite",
           "Cython","setuptools","wheel")
    versions={}
    for name in names:
        try:
            versions[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name]=None
    return versions


def compiled_numba_smoke():
    """Compile a scalar *test* function; never invoke any PCC kernel."""
    from numba import njit
    import numba
    import llvmlite

    @njit(cache=False)
    def _synthetic_increment(x):
        return x + 1

    assert _synthetic_increment(41) == 42
    require(len(_synthetic_increment.signatures) == 1,
            "Expected a single integer-only compiled smoke signature")
    signature=_synthetic_increment.signatures[0]
    llvm=_synthetic_increment.inspect_llvm(signature)
    require(len(llvm)>0, "Numba compiler returned empty LLVM IR")
    return {
        "numba_version": numba.__version__,
        "llvmlite_version": llvmlite.__version__,
        "synthetic_scalar_only_signature":str(signature),
        "synthetic_scalar_only_LLVM_IR_sha256":
            hashlib.sha256(llvm.encode("utf-8")).hexdigest(),
        "PCC_kernel_invoked":False,
        "PCC_kernel_native_signatures_already_validated":False,
    }


def record():
    protocol=json.loads(pins.P1_PAYLOAD.read_text(encoding="utf-8"))
    approved=json.loads(pins.P1_MANIFEST.read_text(encoding="utf-8"))
    manifest=json.loads(pins.MANIFEST.read_text(encoding="utf-8"))
    pins.verify_manifest_only(manifest,approved,protocol)
    src=verify_copied_observers()
    # Imports load extension binaries but DO NOT call propagate/fit.
    pure=importlib.import_module("experiments.live_numpy_milestones")
    jit=importlib.import_module("experiments.live_numba_milestones")
    native=importlib.import_module("pcc_step_live_research")
    stock=importlib.import_module("pcc_step")
    cy_observer=importlib.import_module("experiments.live_cython_milestones")
    require(callable(getattr(pure,"observe_numpy_sequential",None)) and
            callable(getattr(jit,"observe_numba_sequential",None)) and
            callable(getattr(cy_observer,"observe_cython_sequential",None)) and
            callable(getattr(native,"pcc_propagate",None)) and
            callable(getattr(stock,"pcc_propagate",None)),
            "Observer modules or native extensions failed to import")
    compiled={}
    for name,mod in [("production_pcc_step",stock),
                     ("isolated_pcc_step_live_research",native)]:
        location=Path(mod.__file__).resolve()
        require(location.is_file() and location.suffix.lower() in
                (".so",".pyd",".dll"),
                "Expected imported compiled Cython binary: "+name)
        compiled[name]={"binary_filename":location.name,
                        "sha256":sha256_file(location),
                        "bytes":location.stat().st_size}
    generated=ROOT/".cache/research_live_cython/pcc_step_live_research.pyx"
    require(generated.is_file(),
            "Isolated Cython research source did not materialize")
    source_generated_sha=sha256_file(generated)
    versions=metadata_versions()
    require(all(versions[name] is not None for name in (
        "numpy","scipy","scikit-learn","numba","llvmlite",
        "Cython","setuptools","wheel")),
        "Full native/build stack must be present in P2 smoke")
    return {
        "schema":"pypcc-P2-01B-native-synthetic-build-provenance/v1",
        "result":"NATIVE_ARTIFACTS_BUILT_AND_IMPORTED_NO_PCC_FIT",
        "P1_approved_scientific_sha256":pins.P1_SHA,
        "exact_research_source_blobs":src,
        "observed_runtime":{
            "python_version":platform.python_version(),
            "python_impl":platform.python_implementation(),
            "os":platform.system(),
            "os_release":platform.release(),
            "architecture":platform.machine(),
            "python_SOABI":sysconfig.get_config_var("SOABI"),
            "compiler_configured":sysconfig.get_config_var("CC"),
            "compiler_compile_flags":sysconfig.get_config_var("CFLAGS"),
            "shared_link_command":sysconfig.get_config_var("LDSHARED"),
            "package_versions_observed_not_approved_lock":versions,
        },
        "native_Cython_modules":compiled,
        "generated_isolated_Cython_pyx_sha256":source_generated_sha,
        "numba_synthetic_scalar_JIT":compiled_numba_smoke(),
        "P2_01_complete_native_version_lock":False,
        "cross_platform_exact_binary_identity_expected":False,
        "Cython_and_NumPy_observer_parity_under_P2_06_verified":False,
        "real_datasets_loaded":0,
        "real_graphs_generated":0,
        "label_masks_drawn":0,
        "PCC_fits_executed":0,
        "P2_preflight_sha256":None,
        "P2_execution_authorized":False,
    }


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path)
    args=parser.parse_args()
    payload=record()
    data=json.dumps(payload,indent=2,sort_keys=True)+"\n"
    if args.output:
        args.output.write_text(data,encoding="utf-8")
    print(data)


if __name__=="__main__":
    main()
