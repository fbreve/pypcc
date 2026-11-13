"""P2-01H: image-contained runtime SELF-CHECK. No PCC package or UCI sources.

This file is COPY'd into a pinned-base OCI image alongside an exact
CPython 3.12.16 environment (including 13 later P2-01D archived wheels)
and two NEW P2-01F compiled Cython files. It checks byte identity and
IMPORTS ONLY. Never invoke PCC kernels, fit_predict, a Numba PCC JIT, graph,
mask, or actual UCI source access. Read-only image + no-network container.
"""
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import platform
import re
import sys
import sysconfig

RUNTIME = Path("/opt/hostedtoolcache/Python/3.12.16/x64")
MANIFEST = Path("/opt/p2-runtime-manifest.json")
NATIVE = Path("/opt/p2-native")
P1_SHA = "f0111f153ade3d72191c107ace039e1835e5c27db2cf4b3df8d610b41b2f4166"


class OCIImportBlocked(ValueError):
    pass


def require(ok, message):
    if not ok:
        raise OCIImportBlocked(message)


def sha256(raw):
    return hashlib.sha256(raw).hexdigest()


def file_sha(path):
    hasher = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for data in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(data)
    return hasher.hexdigest()


def runtime_tree_digest(root):
    """Content hash over sorted file bytes + symlink targets, no mtime/inodes.

    Does NOT claim that python's binary inputs/toolchain/OS are reproducible;
    it proves only that THIS copied runtime tree's bytes matched the input tree.
    """
    root = Path(root)
    require(root.is_dir() and not root.is_symlink(), "Missing runtime tree")
    entries = []
    total_bytes = 0
    count = 0
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root).as_posix()
        require("\n" not in relative and "\0" not in relative,
                "Unrepresentable runtime path")
        if path.is_symlink():
            target = os.readlink(path)
            require(not Path(target).is_absolute() and
                    ".." not in Path(target).parts,
                    "Runtime tree contains a link outside the copied directory")
            entries.append((relative, "symlink", target))
        elif path.is_file():
            size = path.stat().st_size
            require(size <= 1024**3, "Unexpectedly large individual runtime file")
            entries.append((relative, "file", size, file_sha(path)))
            total_bytes += size
        elif path.is_dir():
            entries.append((relative, "dir"))
        else:
            raise OCIImportBlocked("Unsupported filesystem node inside runtime tree")
        count += 1
        require(count <= 100000 and total_bytes <= 4*1024**3,
                "Candidate copied runtime is unbounded")
    digest = sha256(json.dumps(entries,sort_keys=False,ensure_ascii=True,
                                separators=(",",":")).encode("ascii"))
    return {"sha256":digest,"filesystem_entries":count,
            "regular_file_payload_bytes":total_bytes}


def loaded_file_backed_ELFs():
    maps = Path("/proc/self/maps")
    require(maps.is_file(), "Linux /proc/self/maps required")
    paths=set()
    for row in maps.read_text(encoding="utf-8").splitlines():
        parts=row.split(None,5)
        if len(parts)<6:
            continue
        candidate=parts[5].strip()
        if candidate.startswith("/"):
            require(not candidate.endswith(" (deleted)"),
                    "Deleted ELF mapping is not byte-identifiable")
            paths.add(candidate)
    evidence=[]
    for name in sorted(paths):
        file=Path(name)
        require(file.is_file(), "Mapped native file missing from image: "+name)
        evidence.append({"path":str(file.resolve()),"bytes":file.stat().st_size,
                         "sha256":file_sha(file)})
    require(any("libc.so" in x["path"] for x in evidence),
            "Container mapped libc not found")
    return evidence


def import_exact_Cython(name,source):
    file=Path(source)
    require(file.is_file() and
            file.name.startswith(name+".cpython-312-") and
            file.name.endswith(".so"),
            "Exact CPython 3.12 native filename/ABI unexpected")
    spec=importlib.util.spec_from_file_location(name,file)
    require(spec is not None and spec.loader is not None,
            "Cannot import exact original Cython module")
    module=importlib.util.module_from_spec(spec)
    sys.modules[name]=module
    spec.loader.exec_module(module) # Only import. NEVER call a propagation method.
    require(Path(module.__file__).resolve()==file.resolve(),
            "Loaded a different Cython module")
    return file_sha(file)


def run():
    require(platform.system()=="Linux" and
            platform.machine().lower() in ("x86_64","amd64") and
            platform.python_version()=="3.12.16" and
            sys.prefix==str(RUNTIME),
            "Wrong candidate Linux Python runtime or prefix")
    doc=json.loads(MANIFEST.read_text(encoding="utf-8"))
    require(doc.get("schema")==
            "pypcc-P2-01H-content-addressed-OCI-runtime-substrate/v1" and
            doc.get("P1_scientific_sha256")==P1_SHA and
            doc.get("P2_01D_later_candidate_wheels_not_original_P2_01C") is True and
            doc.get("native_modules_are_NEW_P2_01F_not_historical") is True and
            doc.get("P2_01_final_execution_toolchain_lock_pass") is False and
            doc.get("P2_07_full_resources_gate_pass") is False and
            doc.get("P2_execution_preflight_sha256") is None and
            doc.get("P2_execution_authorized") is False,
            "OCI runtime manifest attempts forbidden scientific execution approval")
    base=doc.get("OCI_base",{})
    require(re.fullmatch(r"ubuntu@sha256:[0-9a-f]{64}",
                         base.get("reference","")) is not None and
            re.fullmatch("[0-9a-f]{64}",base.get("config_image_sha256","")) is not None and
            base.get("platform")=="linux/amd64" and
            base.get("observed_ubuntu_version_id")=="26.04",
            "OCI base is not content addressed for Ubuntu26 amd64")
    expected=doc.get("exact_later_candidate_13_distributions",{})
    require(len(expected)==13 and
            all(re.fullmatch("[0-9a-f]{64}",x.get("wheel_sha256",""))
                for x in expected.values()),
            "Archived candidate wheel identity inventory incomplete")
    installed={key:importlib.metadata.version(key) for key in expected}
    require(installed=={key:x["version"] for key,x in expected.items()},
            "Container copied runtime package versions differ from archived wheels")
    # Verify the original exact extra library needed by this minimal base.
    extra = doc.get("copied_host_native_runtime_libraries", {})
    require(set(extra) == {"libffi.so.8"}, "Required sealed libffi unavailable")
    lib = Path("/opt/p2-hostlibs/libffi.so.8")
    info = extra["libffi.so.8"]
    require(lib.is_file() and lib.stat().st_size == info["bytes"] and
            file_sha(lib) == info["sha256"], "Sealed libffi changed")
    import numpy, scipy, sklearn, numba, llvmlite
    del numpy,scipy,sklearn,numba,llvmlite
    exact=doc.get("P2_01F_exact_new_native",{})
    require(set(exact)=={"production_pcc_step",
                        "isolated_pcc_step_live_research"},
            "Candidate must include exactly both new Cython binaries")
    require({x.name for x in NATIVE.iterdir()}==
            {entry["filename"] for entry in exact.values()},
            "Candidate native image has extra/missing compiled files")
    imports={}
    for label,entry in exact.items():
        target=NATIVE/entry["filename"]
        require(target.is_file() and target.stat().st_size==entry["bytes"] and
                file_sha(target)==entry["sha256"],
                "NEW P2-01F native binary byte mismatch inside OCI candidate")
        module="pcc_step" if label=="production_pcc_step" else "pcc_step_live_research"
        loaded_sha=import_exact_Cython(module,target)
        require(loaded_sha==entry["sha256"],"Native import changed bytes")
        imports[label]={"sha256":loaded_sha,"filename":target.name,
                        "import_only":True}
    runtime=runtime_tree_digest(RUNTIME)
    require(runtime==doc["copied_CP312_runtime_tree"],
            "Python runtime tree differs after COPY or image import")
    mapped=loaded_file_backed_ELFs()
    require(all(any(Path(m["path"]).name==item["filename"] and
                    m["sha256"]==item["sha256"] for m in mapped)
                for item in imports.values()),
            "Imported Cython ELF absent from actual container mapped image")
    etc=Path("/etc/os-release").read_text(encoding="utf-8")
    match=re.search(r'^VERSION_ID="?([^"\n]+)"?$',etc,re.MULTILINE)
    require(match is not None and match.group(1)=="26.04",
            "Base container /etc/os-release differs from pinned Ubuntu26 version")
    report={
        "schema":"pypcc-P2-01H-pinned-container-import-only-receipt/v1",
        "status":"PINNED_UBUNTU26_BASE_AND_EXACT_PYTHON_WHEELS_CYTHON_IMPORTED_OFFLINE_NO_PCC",
        "P1_scientific_sha256":P1_SHA,
        "OCI_base_pinned_index_digest":base["reference"],
        "OCI_base_pinned_local_config_SHA256":base["config_image_sha256"],
        "OCI_image_selfcheck_network_disabled_and_read_only":True,
        "python_version":platform.python_version(),
        "python_SOABI":sysconfig.get_config_var("SOABI"),
        "copied_python_runtime_tree_sha256":runtime["sha256"],
        "copied_python_runtime_file_count":runtime["filesystem_entries"],
        "candidate_package_versions":installed,
        "P2_01F_exact_new_native_imports":imports,
        "actually_loaded_mapped_native_file_sha256":mapped,
        "number_of_loaded_mapped_native_files":len(mapped),
        "not_an_entire_compiler_SDK_header_OS_source_rebuild":True,
        "not_a_final_scientific_execution_image":True,
        "real_UCI_source_rows_loaded":0,
        "graph_or_mask_built":0,
        "synthetic_PCC_fits_executed":0,
        "real_PCC_fits_executed":0,
        "P2_01_final_execution_toolchain_lock_pass":False,
        "P2_07_full_resources_gate_pass":False,
        "P2_execution_preflight_sha256":None,
        "P2_execution_authorized":False,
    }
    if os.getenv("P2_01H_OUTPUT"):
        output=Path(os.environ["P2_01H_OUTPUT"])
        require(str(output).startswith("/out/") and output.parent==Path("/out"),
                "Selfcheck must write only to mounted output directory")
        output.write_text(json.dumps(report,sort_keys=True,indent=2)+"\n",
                          encoding="utf-8")
    print(json.dumps({
        "status":report["status"],
        "pinned_base":base["reference"],
        "python_tree_sha256":runtime["sha256"],
        "mapped_native_file_count":len(mapped),
        "new_Cython_modules_imported":len(imports),
        "installed_exact_distributions":len(installed),
        "real_PCC_fits_executed":0,
        "P2_execution_authorized":False,
    },sort_keys=True))
    return report


if __name__=="__main__":
    run()
