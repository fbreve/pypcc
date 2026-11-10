"""Capture live Numba machine-code bytes on both sides of the ORIGINAL #237 Callgrind fit.

No production/kernel monkeypatch: only wraps the #237 *research* profiler switch.
The raw executable bytes and exact LLVM entry are captured in the SAME PROCESS
which supplies Callgrind instruction addresses. This is not array attribution.
"""
import hashlib
import json
import os
from pathlib import Path
import platform
import sys

import numba
import numpy as np
import pcc_numba
from experiments import profile_numba_instruction_case as original_probe


def native_snapshot():
    dispatcher = pcc_numba._pcc_step_numba_sequential
    overloads = list(dispatcher.overloads.items())
    if len(overloads) != 1:
        raise AssertionError("expected exactly one original compiled native Numba step signature")
    signature, compiled = overloads[0]
    address = compiled.library.get_pointer_to_function(compiled.fndesc.llvm_func_name)
    if not isinstance(address, int) or address <= 0:
        raise AssertionError("original LLVM function entry pointer missing")
    if platform.machine().lower() not in ("x86_64", "amd64"):
        raise AssertionError("only x86-64 instruction decoding preregistered")

    containing = []
    for line in Path("/proc/self/maps").read_text(encoding="utf-8").splitlines():
        fields = line.split(maxsplit=5)
        if len(fields) < 2:
            continue
        lo_text, hi_text = fields[0].split("-", 1)
        lo, hi = int(lo_text, 16), int(hi_text, 16)
        if lo <= address < hi and "r" in fields[1] and "x" in fields[1]:
            containing.append((lo, hi, fields[1]))
    if len(containing) != 1:
        raise AssertionError("JIT pointer not in one readable executable native mapping")
    lo, hi, permissions = containing[0]
    size = min(16384, hi - address)
    if size < 2048:
        raise AssertionError("native code mapping too short for original hot-offset audit")
    with open("/proc/self/mem", "rb", buffering=0) as handle:
        data = os.pread(handle.fileno(), size, address)
    if len(data) != size:
        raise AssertionError("incomplete live original native machine-code snapshot")
    record = {
        "schema": 1,
        "llvm_native_entry": hex(address),
        "llvm_symbol": compiled.fndesc.llvm_func_name,
        "numba_signature": str(signature),
        "numba_version": numba.__version__,
        "numpy_version": np.__version__,
        "python_version": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "mapping_permissions": permissions,
        "captured_length": size,
        "code_sha256": hashlib.sha256(data).hexdigest(),
        "source_master": original_probe.BASE,
        "same_process_provenance": True,
        "does_not_identify_runtime_data_arrays": True,
    }
    return record, data


def main():
    if sys.platform != "linux":
        raise AssertionError("actual new machine-code capture must run inside Linux Valgrind")
    if sys.argv.count("--out") != 1:
        raise AssertionError("exactly one original --out path required")
    raw_out = Path(sys.argv[sys.argv.index("--out") + 1])
    folder = raw_out.parent
    folder.mkdir(parents=True, exist_ok=True)
    saved = {}
    original_switch = original_probe.invoke_command

    def capture_around_original_callgrind(on):
        if on:
            if saved:
                raise AssertionError("unexpected second profiler activation")
            before, code = native_snapshot()
            (folder / "original_step_code.bin").write_bytes(code)
            saved["before"] = before
            saved["code"] = code
            # Instrumentation begins only AFTER this same-process code snapshot.
            original_switch(True)
        else:
            if "before" not in saved or "after" in saved:
                raise AssertionError("profiler stopped without one preceding activation")
            # Keep the normal original #237 switch order exactly intact.
            original_switch(False)
            after, code = native_snapshot()
            if before := saved["before"]:
                if (after["llvm_native_entry"] != before["llvm_native_entry"]
                    or after["llvm_symbol"] != before["llvm_symbol"]
                    or after["numba_signature"] != before["numba_signature"]
                    or code != saved["code"]):
                    raise AssertionError("compiled LLVM entry/code changed during full original fit")
            after["verified_same_bytes_before_after_original_fit"] = True
            after["capture_phase"] = "immediately before/after genuine profiler activation"
            (folder / "numba_live_assembly_snapshot.json").write_text(
                json.dumps(after, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8")
            saved["after"] = True

    original_probe.invoke_command = capture_around_original_callgrind
    try:
        original_probe.main()
    finally:
        original_probe.invoke_command = original_switch
    if saved.get("after") is not True:
        raise AssertionError("same-process before/after native snapshot not completed")
    print("ORIGINAL_LIVE_NUMBA_CODE_BYTES_SAME_PROCESS_PASS", flush=True)


if __name__ == "__main__":
    main()
