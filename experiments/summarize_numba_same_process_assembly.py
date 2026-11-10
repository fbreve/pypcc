"""Strict post-capture proof for the ONE new same-process Numba JIT opcode probe.

Disassembly of bytes saved during *this* real profiled fit, never reconstruction
from #237 old position offsets. All Callgrind cache misses are SIMULATED.
"""
import hashlib
import json
from pathlib import Path
import sys

from capstone import CS_ARCH_X86, CS_MODE_64, Cs
from capstone.x86_const import X86_OP_MEM

from experiments.summarize_numba_instruction_probe import (
    read_profile, native_step_pointer, source_valid, must
)

SOURCE_FIELDS = (
    "base", "source_hashes", "graph_sha", "labels_sha", "fixture_sha",
    "state_hashes", "rng_next", "original_stop", "native_layout",
    "owndeg_f_layout", "distances_uint8"
)
TARGET = ("digits", 89021, 0.5)


def unique(folder, pattern):
    files = sorted(Path(folder).rglob(pattern))
    must(len(files) == 1, "expected one immutable file for " + pattern +
         " in " + str(folder) + ", found " + str(len(files)))
    return files[0]


def frozen_case(directory, expected_count):
    files = sorted(Path(directory).rglob("numba_*probe_*.json"))
    if not files:  # Original #236 deliberately has another JSON prefix.
        files = sorted(Path(directory).rglob("numba_cache_locality_*.json"))
    must(len(files) == expected_count,
         "wrong number of ORIGINAL archived source cases " + str(directory) +
         " " + str(len(files)))
    candidates = []
    for file in files:
        record = json.loads(file.read_text(encoding="utf-8"))
        if (record["dataset"], record["seed"], record["policy"]) == TARGET:
            source_valid(record)
            candidates.append(record)
    must(len(candidates) == 1, "original source case not unique " + str(directory))
    return candidates[0]


def memory_operands(insn):
    descriptions = []
    for operand in insn.operands:
        if operand.type != X86_OP_MEM:
            continue
        mem = operand.mem
        descriptions.append({
            "segment": insn.reg_name(mem.segment) if mem.segment else None,
            "base_register": insn.reg_name(mem.base) if mem.base else None,
            "index_register": insn.reg_name(mem.index) if mem.index else None,
            "index_scale": mem.scale,
            "displacement": mem.disp,
            "operand_size": operand.size,
            "operand_access_mask": operand.access,
        })
    return descriptions


def main(directory):
    base = Path(directory)
    new_dir = base / "new"
    original237 = base / "original237"
    original236linux = base / "original236linux"
    original236windows = base / "original236windows"
    record = frozen_case(new_dir, 1)
    must((record["dataset"], record["seed"], record["policy"]) == TARGET,
         "the genuine new case is not the single original Digits p_grd=.5 case")
    must(record["linux_actual_valgrind"] is True and
         record["windows_uninstrumented_reference"] is False,
         "new live code sample must be genuine Linux Valgrind")
    for label, path, expected in [
        ("#237 original three Linux", original237, 3),
        ("#236 original twelve Linux", original236linux, 12),
        ("#236 original twelve Windows", original236windows, 12),
    ]:
        old = frozen_case(path, expected)
        for field in SOURCE_FIELDS:
            must(record[field] == old[field],
                 label + " original source/terminal state/RNG/stop differs: " + field)

    pointer = native_step_pointer(record)
    raw = unique(new_dir, "callgrind.out.*")
    result = read_profile(raw, pointer)
    must(result["complete_llvm_entry_relative_addresses"] is True,
         "new exact same-process Callgrind instruction addresses unresolved")
    must(result["all_step_instr_events_reconcile"] is True,
         "new Callgrind per-instruction nine-event sum mismatch")
    must(len(result["top10_simulated_D1_reads"]) == 10,
         "missing exact ten hottest original new instruction addresses")

    snapshot_file = unique(new_dir, "numba_live_assembly_snapshot.json")
    snapshot = json.loads(snapshot_file.read_text(encoding="utf-8"))
    code = unique(new_dir, "original_step_code.bin").read_bytes()
    must(snapshot["schema"] == 1 and
         snapshot["same_process_provenance"] is True and
         snapshot["verified_same_bytes_before_after_original_fit"] is True and
         snapshot["does_not_identify_runtime_data_arrays"] is True,
         "unproven native code provenance")
    must(int(snapshot["llvm_native_entry"], 16) == pointer,
         "Callgrind and in-process snapshot JIT pointers differ")
    must(snapshot["captured_length"] == len(code) and
         hashlib.sha256(code).hexdigest() == snapshot["code_sha256"],
         "original live executable bytes do not match archive checksum")
    must(snapshot["source_master"] == record["base"],
         "live capture was not built from the frozen original production master")
    signatures = record["original_compiled_step_signatures"]
    must(len(signatures) == 1 and
         signatures[0] == snapshot["numba_signature"],
         "native source JIT signature changed between profiler and byte capture")

    decoder = Cs(CS_ARCH_X86, CS_MODE_64)
    decoder.detail = True
    # Explicit start-boundary audit: decode sequentially from the actual native
    # function entry, then require each Callgrind hot address to be an instruction
    # START in this exact captured byte stream. Never decode arbitrary mid-bytes.
    instructions = {inst.address: inst for inst in decoder.disasm(code, pointer)}
    decoded = []
    misses = []
    for item in result["top10_simulated_D1_reads"]:
        absolute = int(item["instruction"], 16)
        offset = int(item["offset"], 16)
        must(absolute == pointer + offset, "raw JIT native offset mapping drift")
        inst = instructions.get(absolute)
        if inst is None:
            misses.append(item["offset"])
            continue
        actual = code[offset:offset + inst.size]
        must(bytes(inst.bytes) == actual,
             "compiled machine instruction bytes not captured at hot offset")
        decoded.append({
            "offset": item["offset"],
            "instruction": item["instruction"],
            "raw_machine_bytes": actual.hex(),
            "mnemonic": inst.mnemonic,
            "operands": inst.op_str,
            "memory_operands": memory_operands(inst),
            "explicit_memory_operand": bool(memory_operands(inst)),
            "Dr_dynamic_references": item["Dr"],
            "D1mr_simulated_read_misses": item["D1mr"],
            "DLmr_simulated_last_level_read_misses": item["DLmr"],
        })
    verdict = ("SAME_PROCESS_HOT_INSTRUCTIONS_DECODED"
               if len(decoded) == 10 and not misses
               else "INSTRUCTION_CODE_UNRESOLVED")
    current_offsets = {row["offset"] for row in decoded}
    historic = ["0x3a0", "0x3a6"]
    output = {
        "schema": 1,
        "verdict": verdict,
        "source_master": record["base"],
        "new_actual_native_case": "digits-k10-labels89021-p0.5",
        "actual_compiled_machine": snapshot["machine"],
        "numba": snapshot["numba_version"],
        "new_native_pointer": snapshot["llvm_native_entry"],
        "live_machine_code_sha256": snapshot["code_sha256"],
        "code_bytes_before_after_exact": True,
        "nine_exclusive_callgrind_event_totals_exact": True,
        "original_full_state_mt_rng_stop_parity_237_236_linux_windows": True,
        "new_current_top10_simulated_D1mr_share_percent":
            result["top10_D1mr_share_of_native_step_pct"],
        "new_current_hot_instructions_decoded": decoded,
        "new_current_unresolved_hot_offsets": misses,
        "historic_237_offsets": {
            "0x3a0": {"historic_D1mr": 206330, "present_in_NEW_top10": "0x3a0" in current_offsets},
            "0x3a6": {"historic_D1mr": 132999, "present_in_NEW_top10": "0x3a6" in current_offsets},
        },
        "nonportable_old_code_warning": "No machine bytes were archived in #237; do not claim the new opcodes are identical to the historical #237 instructions.",
        "no_allocation_identity": True,
        "simulated_cache_not_hardware_not_cpu_time": True,
        "no_production_candidate": True,
    }
    Path("numba_same_process_assembly_decision.json").write_text(
        json.dumps(output, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8")
    lines = [
        "# Same-process genuine Numba JIT machine-opcode audit",
        "",
        "**" + verdict + "** — original production kernel, ONE original Digits full fit.",
        "",
        "Original graph, source SHA, eight full terminal state SHA, real seeded Numba MT RNG and stop "
        "exactly match #237 Linux and #236 Linux/Windows archives.",
        "The exact compiled JIT entry pointer and executable bytes were captured "
        "inside the profiled process immediately before and after the same original fit.",
        "All nine raw exclusive Valgrind/Callgrind costs reconcile, and hot instruction addresses "
        "refer to these same compiled bytes.",
        "",
        "| NEW JIT offset | Exact bytes | Mnemonic | Operands | Dynamic Dr | Simulated D1mr | Explicit memory operands |",
        "|---|---|---|---|---:|---:|---:|",
    ]
    for row in decoded:
        lines.append(
            "| `" + row["offset"] + "` | `" + row["raw_machine_bytes"] + "` | `" +
            row["mnemonic"] + "` | `" + row["operands"].replace("|", "\\|") +
            "` | " + str(row["Dr_dynamic_references"]) + " | " +
            str(row["D1mr_simulated_read_misses"]) + " | " +
            str(len(row["memory_operands"])) + " |")
    lines.extend([
        "",
        "**IMPORTANT:** These are machine instructions from this NEW run only. "
        "The previous #237 experiment did NOT archive code bytes; its offsets "
        "0x3a0 and 0x3a6 are historical comparisons, not guaranteed to be the same opcodes.",
        "The new run's top 10 account for " +
        f"{result['top10_D1mr_share_of_native_step_pct']:.2f}%" +
        " of simulated Numba-step L1D read misses.",
        "Historical #237 0x3a0 present among NEW top-10: " +
        str("0x3a0" in current_offsets) +
        "; 0x3a6 present: " + str("0x3a6" in current_offsets) + ".",
        "These instrumented opcode events do NOT identify which particular "
        "runtime allocation or matrix is accessed; base/index registers "
        "need dynamic address-to-object correlation before any array attribution.",
        "Valgrind misses are a hypothetical cache model, NOT actual CPU PMU misses, "
        "CPU time, bandwidth or available full-fit speedup. "
        "No production code change or CSR proposal is justified.",
    ])
    if misses:
        lines.append("Unresolved Callgrind starts (do NOT invent decode): " +
                     ", ".join(misses))
    Path("numba_same_process_assembly_report.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
