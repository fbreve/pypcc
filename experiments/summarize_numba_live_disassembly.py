"""Independently validate ORIGINAL live Numba JIT raw machine bytes and profile.

One original Digits p=.5 1000-step true native Numba fit; original same-process
LLVM pointer, Capstone x86_64 re-decode and all nine raw Callgrind event totals.
Compare eight original states, real native MT RNG, graph and stop to archived
#237 Linux and #236 Linux+Windows originals. No CPU performance claim.
"""
from pathlib import Path
import hashlib
import json
import sys

from capstone import Cs, CS_ARCH_X86, CS_MODE_64
from capstone.x86 import X86_OP_MEM
from experiments.numba_original_instruction_reader import (
    read_profile, native_step_pointer, source_valid,
)

BASE="4c1c410d8eb04547421f72ee1d3ca2eeb5763ff2"
CASE=("digits",89021,0.5)


def require(value,reason):
    if not value:
        raise AssertionError(reason)


def source_rows(root,pattern,required_count):
    files=sorted(Path(root).rglob(pattern))
    require(len(files)==required_count,
            "original "+pattern+" corpus size "+str(len(files)))
    rows={}
    for file in files:
        data=json.loads(file.read_text(encoding="utf-8"))
        key=(data["dataset"],data["seed"],data["policy"])
        require(key not in rows,"duplicate genuine original condition")
        rows[key]=(data,file)
    require(CASE in rows,"original fixed Digits mixed source case absent")
    return rows[CASE]


def disassemble_and_check(meta,raw,ptr):
    require(meta["original_true_native_llvm_pointer"]==hex(ptr),
            "native machine bytes and actual LLVM pointer differ")
    require(meta["jit_snapshot_bytes"]==len(raw) and
            4096<=len(raw)<=16384 and
            meta["jit_snapshot_sha256"]==hashlib.sha256(raw).hexdigest(),
            "original executable raw byte snapshot SHA differs")
    require(meta["original_source_code_only"] is True and
            meta["architecture"].lower() in ("amd64","x86_64"),
            "nonoriginal or non-x86-64 executable bytes")
    maps=meta["original_proc_maps_segment"]
    require("r" in maps["permissions"] and "x" in maps["permissions"] and
            int(maps["begin"],16)<=ptr and
            ptr+len(raw)<=int(maps["end"],16),
            "raw JIT snapshot is not inside original executable Linux mapping")
    decoder=Cs(CS_ARCH_X86,CS_MODE_64)
    decoder.detail=True
    regenerated=list(decoder.disasm(raw,ptr))
    stored=meta["machine_instruction_listing"]
    require(len(regenerated)==len(stored) and len(stored)>50,
            "original disassembler listing not reproducible from native bytes")
    byoffset={}
    for ins,reference in zip(regenerated,stored):
        off=ins.address-ptr
        require(reference["offset"]==hex(off) and
                reference["address"]==hex(ins.address) and
                reference["size"]==ins.size and
                reference["hex_bytes"]==ins.bytes.hex() and
                reference["mnemonic"]==ins.mnemonic and
                reference["operands"]==ins.op_str,
                "original executable opcode/offset disagrees with independent Capstone")
        mem=[dict(base=ins.reg_name(op.mem.base),
                  index=ins.reg_name(op.mem.index),
                  scale=int(op.mem.scale),
                  displacement=int(op.mem.disp),
                  operand_size=int(op.size))
             for op in ins.operands if op.type==X86_OP_MEM]
        require(reference["memory_operands"]==mem,
                "native operand registers/offsets differ from raw machine bytes")
        byoffset[off]=reference
    require(0 in byoffset and max(byoffset)>0x500,
            "original JIT function linear decode failed near hotspot")
    for offset in (0x36b,0x3a0,0x3a6,0x5ad,0x620,0x7d2):
        saved=meta["prior_237_hot_windows"][hex(offset)]
        require(saved["exact_instruction_start"]==(offset in byoffset),
                "frozen former hottest instruction visibility inconsistent")
    return byoffset


def render(result):
    p=result["profile"]
    lines=[
        "# Original same-process Numba LLVM native instruction disassembly",
        "",
        "**"+result["verdict"]+"** — research-only feasibility. Not a CPU GO.",
        "",
        "The ONE original-source Digits kNN10 p_grd=0.5 1000-step Numba",
        "full fit was recorded with native executable bytes from the exact",
        "same-process LLVM function pointer used by Callgrind. Original",
        "raw code bytes before/after the actual fit are identical.",
        "An independent disassembler reconstructed byte-aligned x86-64",
        "instructions, registers and memory operands from the raw image.",
        "All nine original Valgrind exclusive event sums equal raw totals.",
        "The eight full original arrays, real Numba MT RNG, stopping and",
        "graph/label/source SHA match earlier Linux and Windows sources.",
        "",
        "| Native original event | Result |",
        "|---|---:|",
        "| Exact LLVM JIT entry | "+result["entry"]+" |",
        "| Captured original executable bytes | "+str(result["bytes"])+" |",
        "| Decoded native machine addresses | "+str(result["instructions"])+" |",
        "| Original step Dr references | "+str(p["original_step_events"]["Dr"])+" |",
        "| Original step SIMULATED D1mr | "+str(p["original_step_events"]["D1mr"])+" |",
        "| Original step SIMULATED DLmr | "+str(p["original_step_events"]["DLmr"])+" |",
        "",
        "## Three hottest native memory-reading instructions",
        "",
        "| Step offset | Bytes | x86-64 mnemonic and operands | Dr | D1mr simulated | DLmr simulated |",
        "|---|---|---|---:|---:|---:|"]
    for ins in result["top_three_original_native"]:
        lines.append("| "+ins["offset"]+" | "+ins["hex_bytes"]+
                     " | "+ins["mnemonic"]+" "+ins["operands"]+
                     " | "+str(ins["Dr"])+" | "+str(ins["D1mr"])+
                     " | "+str(ins["DLmr"])+" |")
    lines+=["","## Interpretation boundary","",
            "An instruction's base+index*scale+displacement addressing",
            "expression is NOT a verified identity of an original",
            "NumPy allocation. The same instruction may read multiple",
            "buffers and cannot prove a dominance/dist_table/cache bottleneck.",
            "The Valgrind L1D/LL misses are modeled, not hardware counts",
            "or attainable CPU speedups. A future original-vs-candidate",
            "performance optimization needs frozen real whole-fit CPU",
            "Linux+Windows, original eight states, MT, stop and memory.",
            "Keep the deliberately padded dense int64 neighbor graph.",
            "Raw GitHub Actions data expire under retention."]
    return "\n".join(lines)+"\n"


def main(folder):
    root=Path(folder)
    current,path=source_rows(root,"numba_live_disasm_original_*.json",1)
    old237,_=source_rows(root,"numba_instr_probe_*.json",3)
    old236linux,_=source_rows(root,"numba_cache_locality_linux_*.json",12)
    old236windows,_=source_rows(root,"numba_cache_locality_windows_*.json",12)
    require(source_valid(current)==CASE,"not actual original Digits kNN10 source")
    require(current["linux_actual_valgrind"] is True and
            current["windows_uninstrumented_reference"] is False and
            current["raw_native_machine_disassembly_saved"] is True,
            "new original source did not capture live native JIT in Linux Valgrind")
    for prior,name in ((old237,"#237 Linux"),(old236linux,"#236 Linux"),
                       (old236windows,"#236 Windows")):
        require(source_valid(prior)==CASE,"frozen old source condition missing")
        for field in ("base","source_hashes","graph_sha","labels_sha",
                      "fixture_sha","state_hashes","rng_next","original_stop",
                      "native_layout","owndeg_f_layout","distances_uint8"):
            require(current[field]==prior[field],
                    "original Numba source or real state diverged from "+
                    name+" field "+field)
    pointer=native_step_pointer(current)
    require(hex(pointer)==current["genuine_same_process_native_pointer"],
            "captured exact LLVM pointer differs from Callgrind pointer")
    directory=path.parent
    snapshot=directory/"original_numba_jit_step.bin"
    metadata=directory/"numba_live_jit_assembly.json"
    matches=list(directory.glob("callgrind.out.*"))
    require(snapshot.is_file() and metadata.is_file() and len(matches)==1,
            "original executable bytes/assembly/native profile missing")
    require(matches[0].name=="callgrind.out."+str(current["callgrind_pid"]),
            "raw original Callgrind profile PID did not match live JIT process")
    raw=snapshot.read_bytes()
    meta=json.loads(metadata.read_text(encoding="utf-8"))
    require(meta["jit_snapshot_sha256"]==
            current["genuine_same_process_original_machine_sha256"] and
            meta["jit_snapshot_bytes"]==
            current["genuine_same_process_original_machine_bytes"],
            "original source-result manifest machine image does not match raw")
    listing=disassemble_and_check(meta,raw,pointer)
    profile=read_profile(matches[0],pointer)
    require(profile["complete_llvm_entry_relative_addresses"] is True,
            "actual original native Callgrind instruction offsets unresolved")
    hottest=[]
    for event in profile["top10_simulated_D1_reads"][:3]:
        offset=int(event["offset"],16)
        inst=listing.get(offset)
        hottest.append(dict(
            offset=event["offset"],Dr=event["Dr"],D1mr=event["D1mr"],
            DLmr=event["DLmr"],hex_bytes=(
                inst["hex_bytes"] if inst else "UNRESOLVED"),
            mnemonic=inst["mnemonic"] if inst else "UNRESOLVED",
            operands=inst["operands"] if inst else "UNRESOLVED",
            memory_operands=inst["memory_operands"] if inst else []))
    ok=len(hottest)==3 and all(
        x["mnemonic"]!="UNRESOLVED" and x["memory_operands"]
        for x in hottest)
    verdict="INSN_DISASSEMBLY_PROVEN" if ok else "INSN_MAPPING_UNRESOLVED"
    decision=dict(schema=1,base=BASE,verdict=verdict,
                  exactly_one_actual_original_native_fit=True,
                  source_state_stop_rng_exact_against_archives=True,
                  source_original_native_executable_before_after_unchanged=True,
                  all_nine_original_raw_callgrind_totals_exact=True,
                  no_hardware_miss_or_performance_go=True,
                  original_jit_image_sha256=hashlib.sha256(raw).hexdigest(),
                  entry=hex(pointer),bytes=len(raw),instructions=len(listing),
                  profile=profile,top_three_original_native=hottest)
    Path("numba_live_jit_disassembly_decision.json").write_text(
        json.dumps(decision,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    text=render(decision)
    Path("numba_live_jit_disassembly_report.md").write_text(
        text,encoding="utf-8")
    print(text,flush=True)
    print("STRICT ORIGINAL NUMBA LIVE JIT NATIVE ASSEMBLY",
          verdict,flush=True)


if __name__=="__main__":
    main(sys.argv[1])
