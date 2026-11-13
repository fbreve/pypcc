"""P2-01D: forensic audit of archived P2-01C binaries and NEW wheel candidates.

No build, PCC propagation, or real data. ELF/PE static linking info comes from
exact pre-existing artifact bytes. Wheels are freshly DOWNLOADED candidates,
not evidence that those same wheel bytes were used in P2-01C. Historic Cython
compiler commands come separately from original run #43434870932 logs.
"""
import argparse
from email.parser import Parser
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import re
import struct
import sys
import sysconfig
import zipfile

from packaging.tags import sys_tags
from packaging.utils import canonicalize_name, parse_wheel_filename

from experiments import p2_06_synthetic_parity_exact_artifacts as prior

ROOT = Path(__file__).resolve().parents[1]
HISTORY = ROOT / "docs/current_p2_01d_original_build_log_facts_2026-11-13.json"
PINS = ROOT / "experiments/p2_01c_candidate_py312_requirements.txt"
ALLOWED_HISTORY = {"Linux": 131917249775, "Windows": 131917249362}


class AuditBlocked(ValueError):
    pass


def require(ok, reason):
    if not ok:
        raise AuditBlocked(reason)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for piece in iter(lambda: f.read(1024 * 1024), b""):
            h.update(piece)
    return h.hexdigest()


def u16(b, at):
    require(0 <= at <= len(b)-2, "Truncated binary uint16")
    return struct.unpack_from("<H", b, at)[0]


def u32(b, at):
    require(0 <= at <= len(b)-4, "Truncated binary uint32")
    return struct.unpack_from("<I", b, at)[0]


def cstr(buf, pos, limit=2048):
    require(0 <= pos < len(buf), "Invalid native string offset")
    end = buf.find(b"\0", pos, min(len(buf), pos + limit))
    require(end > pos, "Unterminated/empty native dependency string")
    return buf[pos:end].decode("ascii")


def inspect_elf(data):
    """Read ELF64 x86-64 PT_DYNAMIC directly, without ldd or executing code."""
    require(len(data) >= 64 and data[:4] == bytes((127,)) + b"ELF",
            "Not an ELF binary")
    require(data[4] == 2 and data[5] == 1, "ELF not 64-bit little-endian")
    require(u16(data, 18) == 62, "ELF not x86-64")
    phoff = struct.unpack_from("<Q", data, 32)[0]
    phentsize, phnum = u16(data, 54), u16(data, 56)
    require(phentsize >= 56 and 0 < phnum <= 128 and
            phoff + phentsize * phnum <= len(data), "Invalid ELF program headers")
    loads, dynamic = [], None
    notes = []
    for i in range(phnum):
        pos = phoff + i * phentsize
        typ, flags, off, vaddr, paddr, filesz, memsz, align = struct.unpack_from(
            "<IIQQQQQQ", data, pos)
        require(off + filesz <= len(data), "ELF segment beyond file")
        if typ == 1:
            loads.append((vaddr, off, filesz))
        elif typ == 2:
            dynamic = (off, filesz)
        elif typ == 4:
            notes.append((off, filesz))
    require(loads and dynamic is not None, "ELF missing load/dynamic headers")

    def virtual_to_file(vaddr):
        for base, offset, size in loads:
            if base <= vaddr < base + size:
                return offset + vaddr - base
        raise AuditBlocked("ELF dynamic string table address not file backed")

    start, size = dynamic
    require(size % 16 == 0 and size <= 4*1024*1024,
            "Invalid ELF dynamic table extent")
    tags = {}
    for pos in range(start, start + size, 16):
        tag, value = struct.unpack_from("<qQ", data, pos)
        if tag == 0:
            break
        tags.setdefault(tag, []).append(value)
    require(5 in tags and 10 in tags, "ELF has no dynamic string table")
    string_offset = virtual_to_file(tags[5][0])
    strings_size = tags[10][0]
    require(0 < strings_size <= len(data) - string_offset,
            "ELF dynamic string range invalid")

    def read_string(offset):
        require(offset < strings_size, "Invalid ELF dynamic string index")
        return cstr(data, string_offset + offset, strings_size - offset)

    needed = sorted({read_string(x) for x in tags.get(1, [])})
    rpath = [read_string(x) for x in tags.get(15, [])]
    runpath = [read_string(x) for x in tags.get(29, [])]
    build_ids = []
    for offset, count in notes:
        end = offset + count
        cursor = offset
        while cursor + 12 <= end:
            namesz, descsz, kind = struct.unpack_from("<III", data, cursor)
            cursor += 12
            if namesz > end - cursor:
                break
            name = data[cursor:cursor+namesz].rstrip(b"\0")
            cursor += (namesz + 3) & ~3
            if descsz > end - cursor:
                break
            desc = data[cursor:cursor+descsz]
            cursor += (descsz + 3) & ~3
            if name == b"GNU" and kind == 3:
                build_ids.append(desc.hex())
    return {"format": "ELF64", "machine": "x86-64",
            "DT_NEEDED": needed, "RPATH": rpath, "RUNPATH": runpath,
            "GNU_build_ids": sorted(set(build_ids)),
            "analysis": "static dependency names only; resolved library bytes and symbol ABI NOT locked"}


def inspect_pe(data):
    """Read PE32+ x64 import table with RVA to raw-file mapping."""
    require(len(data) >= 0x100 and data[:2] == b"MZ", "Not a PE file")
    pe = u32(data, 0x3c)
    require(pe + 24 <= len(data) and data[pe:pe+4] == b"PE" + bytes((0, 0)),
            "No PE header")
    require(u16(data, pe+4) == 0x8664, "PE not AMD64")
    sections = u16(data, pe+6)
    optional_bytes = u16(data, pe+20)
    opt = pe + 24
    require(0 < sections <= 128 and optional_bytes >= 136 and
            opt + optional_bytes + 40 * sections <= len(data),
            "Invalid PE sections/optional header")
    require(u16(data, opt) == 0x20b, "PE not PE32+")
    # PE32+ DataDirectory starts at optional header +112. Import dir index 1.
    dir_rva = u32(data, opt + 112 + 8)
    dir_size = u32(data, opt + 112 + 12)
    raw_sections = []
    section_base = opt + optional_bytes
    for i in range(sections):
        pos = section_base + 40*i
        virtual_size = u32(data, pos+8)
        rva = u32(data, pos+12)
        raw_size = u32(data, pos+16)
        raw_pointer = u32(data, pos+20)
        require(raw_pointer + raw_size <= len(data), "PE section exceeds file")
        raw_sections.append((rva, max(raw_size, virtual_size),
                             raw_size, raw_pointer))

    def resolve(rva):
        for base, extent, raw_size, raw_pointer in raw_sections:
            if base <= rva < base + extent:
                delta = rva - base
                require(delta < raw_size, "PE RVA points to non-file-backed memory")
                return raw_pointer + delta
        raise AuditBlocked("PE import RVA not backed by any section")

    imports = []
    if dir_rva:
        require(20 <= dir_size <= 65536, "Invalid PE import directory length")
        descriptor_offset = resolve(dir_rva)
        for index in range(min(dir_size // 20, 1024)):
            cursor = descriptor_offset + index*20
            require(cursor+20 <= len(data), "Truncated PE import descriptor")
            values = struct.unpack_from("<IIIII", data, cursor)
            if values == (0, 0, 0, 0, 0):
                break
            require(values[3] != 0, "PE imported DLL has no name RVA")
            imports.append(cstr(data, resolve(values[3])))
        else:
            raise AuditBlocked("PE import directory lacks terminating descriptor")
    return {"format": "PE32+", "machine": "AMD64",
            "imported_DLL_names": sorted(set(imports), key=str.lower),
            "analysis": "static PE import names only; actual DLL bytes/API-set resolution NOT locked"}


def inspect_binary(path):
    data = Path(path).read_bytes()
    if data[:4] == bytes((127,)) + b"ELF":
        return inspect_elf(data)
    if data[:2] == b"MZ":
        return inspect_pe(data)
    raise AuditBlocked("Unrecognized native file format")


def candidate_wheel_inventory(folder, pins):
    """Inventory newly downloaded wheel candidates, NOT historic installed wheels."""
    root = Path(folder)
    files = sorted(root.iterdir())
    require(files and all(x.is_file() and x.suffix == ".whl" for x in files),
            "Downloaded candidates must contain only wheel files")
    compatible_tags = set(sys_tags())
    inventory = {}
    for path in files:
        try:
            name, version, build, tags = parse_wheel_filename(path.name)
        except Exception as exc:
            raise AuditBlocked("Invalid wheel filename: " + path.name) from exc
        name = canonicalize_name(name)
        require(name in pins and name not in inventory,
                "Unexpected or duplicate wheel: " + name)
        require(str(version) == pins[name], "Wheel version does not match candidate pin: " + name)
        require(bool(tags & compatible_tags),
                "Wheel incompatible with local Python/OS/ABI: " + path.name)
        with zipfile.ZipFile(path) as zipf:
            infos = zipf.infolist()
            require(all(not x.filename.startswith(("/", "\\")) and
                        ".." not in Path(x.filename).parts for x in infos),
                    "Unsafe wheel member path")
            metadata_names = [x.filename for x in infos
                              if x.filename.endswith(".dist-info/METADATA")
                              and x.filename.count("/") == 1]
            require(len(metadata_names) == 1, "Missing or ambiguous wheel METADATA")
            msg = Parser().parsestr(zipf.read(metadata_names[0]).decode("utf-8"))
            require(canonicalize_name(msg["Name"]) == name and
                    msg["Version"] == pins[name],
                    "Wheel METADATA name/version differ from filename/pin")
        inventory[name] = {"filename": path.name, "bytes": path.stat().st_size,
                           "sha256": sha256(path), "version": pins[name],
                           "compatible_tags": sorted(str(x) for x in tags & compatible_tags)[:10]}
    require(set(inventory) == set(pins),
            "Incomplete downloaded wheel set for all candidate pins")
    return dict(sorted(inventory.items()))


def audit(seal_dir, wheel_dir):
    require(platform.system() in ALLOWED_HISTORY, "Unsupported platform for historic audit")
    require(sys.version_info[:2] == (3, 12), "Candidate audit requires CPython 3.12")
    historic = json.loads(HISTORY.read_text(encoding="utf-8"))
    require(historic["original_P2_01C_run_id"] == 43434870932 and
            historic["actual_original_pr_merge_checkout"] ==
            prior.P2_01C_CI_MERGE_COMMIT and
            historic["original_job_ids"] == ALLOWED_HISTORY and
            historic["original_wheel_SHA256_verified"] is False and
            historic["compiler_commands_not_derived_from_current_runner"] is True,
            "Historical build log observations are not correctly scoped")
    archive, binaries = prior.validate_prior_archive(seal_dir)
    # Candidate requirements match bytes preserved with the original bin artifact.
    require(sha256(PINS) == archive["candidate_lock_sha256"] and
            sha256(seal_dir/"candidate-requirements.txt") ==
            archive["candidate_lock_sha256"],
            "Archived and current candidate package pins diverged")
    pins = archive["candidate_locked_versions"]
    wheels = candidate_wheel_inventory(wheel_dir, pins)
    actual = {name: importlib.metadata.version(name) for name in pins}
    require(actual == archive["verified_installed_versions"],
            "Installed package version drift from original P2-01C metadata")
    libraries = {}
    for label, info in binaries.items():
        binpath = seal_dir / "native" / info["binary_filename"]
        libraries[label] = {
            "prior_sha256": info["sha256"],
            "prior_bytes": info["bytes"],
            "binary_static_linkage": inspect_binary(binpath),
        }
    return {
        "schema": "pypcc-P2-01D-original-native-forensics-new-wheel-candidates/v1",
        "status": "ARCHIVED_NATIVE_LINKAGE_OBSERVED_NEW_PLATFORM_WHEELS_HASHED_NOT_FINAL_P2_LOCK",
        "historical_compiler_log_facts": historic,
        "original_P2_01C_artifact_id": prior.ARTIFACT_IDS[platform.system()],
        "P2_01C_seal_sha256": sha256(seal_dir/"p2-01c-candidate-seal.json"),
        "native_binaries_exactly_reused_not_recompiled": libraries,
        "wheel_candidate_evidence": wheels,
        "candidate_wheels_downloaded_now_NOT_original_P2_01C_install_proof": True,
        "runtime": {
            "os": platform.system(),
            "machine": platform.machine(),
            "python": platform.python_version(),
            "SOABI": sysconfig.get_config_var("SOABI"),
            "EXT_SUFFIX": sysconfig.get_config_var("EXT_SUFFIX"),
            "candidate_installed_versions": actual,
        },
        "compiler_command_version_from_original_logs_not_final_sdk_hash_lock": True,
        "original_compiler_toolchain_byte_hashes_sealed": False,
        "original_dependency_wheel_bytes_sealed": False,
        "system_DLL_or_SO_dependency_bytes_sealed": False,
        "P2_01_entire_gate_pass": False,
        "P2_06_full_execution_gate_pass": False,
        "P2_execution_preflight_sha256": None,
        "P2_execution_authorized": False,
        "real_datasets_loaded": 0,
        "synthetic_PCC_invoked_in_this_workflow": False,
        "real_PCC_fits_executed": 0,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sealed-artifact-dir", type=Path, required=True)
    p.add_argument("--candidate-wheel-dir", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    obj = audit(a.sealed_artifact_dir.resolve(), a.candidate_wheel_dir.resolve())
    a.output.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n",
                        encoding="utf-8")
    print(json.dumps({"status": obj["status"],
                      "platform": obj["runtime"]["os"],
                      "wheel_count": len(obj["wheel_candidate_evidence"]),
                      "verified_original_binary_count":
                      len(obj["native_binaries_exactly_reused_not_recompiled"]),
                      "P2_execution_authorized": False,
                      "real_PCC_fits_executed": 0}, sort_keys=True))


if __name__ == "__main__":
    main()
