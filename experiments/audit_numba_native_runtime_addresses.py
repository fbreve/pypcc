"""Independent, fail-closed scientific auditor for ONE genuine original GDB run.

Compares source/graph/8 states/MT RNG/stop against immutable #239 + #236
and validates every native register effective address against original arrays.
No speedup or cache/hardware conclusion is possible from these GDB samples.
"""
import json
from pathlib import Path
import sys

from experiments.summarize_numba_instruction_probe import must

TARGET=("digits",89021,0.5)
OFFSETS={
    "0x36b":("neib_list",8,("r14","rsi",1),"4d8b0436"),
    "0x3a0":("dominance",8,("rcx","r9",8),"c4a17b1004c9"),
    "0x3a6":("dist_table",1,("rdx","r8",1),"460fb60402"),
}
FIELDS=("base","source_hashes","graph_sha","labels_sha","fixture_sha",
        "state_hashes","rng_next","original_stop","native_layout",
        "owndeg_f_layout","distances_uint8","original_dense_padded_int64")
EXPECTED_STATES=("predictions","dominance","positions","strengths",
                 "distances","owndeg","node_labels","mapped_labels")


def single_file(root,pattern):
    files=sorted(Path(root).rglob(pattern))
    must(len(files)==1,"not one original file "+pattern+" in "+str(root)+
         ": "+str(len(files)))
    return files[0]


def old_profile(root,pattern,required):
    files=sorted(Path(root).rglob(pattern))
    must(len(files)==required,
         "wrong original frozen raw source case count: "+str(root)+" "+str(len(files)))
    candidates=[]
    for file in files:
        data=json.loads(file.read_text(encoding="utf-8"))
        if (data["dataset"],data["seed"],data["policy"])==TARGET:
            candidates.append(data)
    must(len(candidates)==1,"missing or duplicate frozen original Digits case")
    return candidates[0]


def original_source_exact(source):
    must(source["schema"]==1 and
         source["base"]=="4c1c410d8eb04547421f72ee1d3ca2eeb5763ff2" and
         source["dataset"]=="digits" and source["seed"]==89021 and
         source["policy"]==0.5 and source["steps"]==1000,
         "wrong original real native dataset or source baseline")
    must(set(source["state_hashes"])==set(EXPECTED_STATES),
         "eight full original states missing")
    must(source["full_fit_under_native_gdb"] is True and
         source["state_and_rng_replayed_without_breakpoints"] is True,
         "genuine source full fit or independent replay absent")
    must(source["original_stop"]["effective_iterations"]==1000 and
         source["original_stop"]["stop_reason"]=="max_iter" and
         source["original_stop"]["checks"]==0,
         "native original stop changed")


def check(directory):
    root=Path(directory)
    new=root/"new"
    original=original_source_exact
    source=json.loads(single_file(new,"numba_native_gdb_full_fit_source.json").read_text())
    original(source)
    a239=old_profile(root/"original239","numba_instr_probe_*.json",1)
    a236linux=old_profile(root/"original236linux","numba_cache_locality_linux_*.json",12)
    a236windows=old_profile(root/"original236windows","numba_cache_locality_windows_*.json",12)
    for label,baseline in (
        ("#239 original Linux",a239),
        ("#236 original Linux",a236linux),
        ("#236 original Windows",a236windows)):
        for field in FIELDS:
            must(source[field]==baseline[field],
                 label+" exact original baseline changed: "+field)

    meta=json.loads(single_file(new,"native_live_array_ranges.json").read_text())
    sample=json.loads(single_file(new,"numba_native_gdb_address_samples.json").read_text())
    must(sample["schema"]==meta["schema"]==1 and
         sample["original_master"]==meta["original_master"]==source["base"] and
         sample["original_case"]==meta["original_case"]==
             "digits-k10-labels89021-p0.5" and
         sample["llvm_entry"]==meta["llvm_entry"] and
         sample["numba_step_signature"]==meta["numba_step_signature"] and
         sample["live_hot_machine_bytes"]==meta["hot_machine_bytes"] and
         sample["allocation_ranges"]==meta["allocations"],
         "native debugger and original source pointer/allocation drift")
    must(sample["read_only_samples"] and
         sample["not_a_performance_measurement"] and
         sample["any_gdb_breakpoint_failures"] is False and
         sample["expected_hits_per_instruction"]==16,
         "incomplete native GDB register sample evidence")
    ptr=int(sample["llvm_entry"],16)
    allocations=sample["allocation_ranges"]
    must(set(sample["samples"])==set(OFFSETS),"exact original hot offsets changed")
    for off,(name,width,reg,opcode) in OFFSETS.items():
        must(meta["hot_expectations"][off]=={
            "bytes":opcode,"allocation":name,"bytes_read":width
        } and sample["live_hot_machine_bytes"][off]==opcode,
             "exact original hot native bytes different from #238/#239")
        arr=allocations[name]
        must(arr["nbytes"]>0 and arr["itemsize"]==width,
             "runtime NumPy allocation dtype/width mismatch")
        lo=int(arr["data_ptr"],16);hi=int(arr["end_ptr"],16)
        must(hi-lo==arr["nbytes"],"invalid original NumPy allocation bounds")
        rows=sample["samples"][off]
        must(len(rows)==16,"not 16 strict original native samples: "+off)
        seen=set()
        for ix,row in enumerate(rows,1):
            addr=int(row["effective_address"],16)
            base=int(row["register_base"],16)
            idx=int(row["register_index"],16)
            must(row["sample"]==ix and
                 row["native_instruction"]==hex(ptr+int(off,16)) and
                 row["read_width"]==width and row["allocation"]==name and
                 addr==(base+idx*reg[2])&((1<<64)-1),
                 "effective address not verified from actual original live registers")
            must(lo<=addr and addr+width<=hi,
                 "sampled address outside expected original NumPy allocation")
            for other,otherobj in allocations.items():
                otherlo=int(otherobj["data_ptr"],16)
                otherhi=int(otherobj["end_ptr"],16)
                if other!=name:
                    must(not (otherlo<=addr and addr+width<=otherhi),
                         "sample ambiguous with another original NumPy buffer")
            rel=addr-lo
            dims=arr["shape"]; strides=arr["strides"]
            if arr["C"]:
                i,j=rel//strides[0],(rel%strides[0])//strides[1]
            else:
                j,i=rel//strides[1],(rel%strides[1])//strides[0]
            must([i,j]==row["element_coordinates"] and
                 rel==row["relative_byte_offset"] and
                 0<=i<dims[0] and 0<=j<dims[1] and
                 i*strides[0]+j*strides[1]==rel,
                 "runtime original array-index derivation inconsistent")
            seen.add((i,j))
        must(len(seen)>1,"all 16 samples point to only one matrix element")
    verdict="DYNAMIC_SAMPLED_ARRAY_IDENTITY_PROVEN"
    output={
        "schema":1,"verdict":verdict,"sample_count":48,
        "source_master":source["base"],
        "source_graph_eight_states_seeded_numba_mt_rng_stop_exact_239_236_linux_windows":True,
        "per_instruction_samples":{k:len(v) for k,v in sample["samples"].items()},
        "operand_array_identity":{
            "0x36b":"neib_list", "0x3a0":"dominance", "0x3a6":"dist_table"},
        "limitations":[
            "Proof of 16 observed effective memory addresses each, NOT every dynamic reference.",
            "Native GDB execution is not original Callgrind and cannot measure CPU speedups.",
            "Old Valgrind cache misses remain simulated under hypothetical L1D/LL parameters.",
            "Production remains unchanged and prior CPU-negative layout tests still apply."
        ],
        "no_production_candidate":True,
    }
    Path("numba_native_gdb_allocation_decision.json").write_text(
         json.dumps(output,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    lines=[
      "# Original Numba JIT native runtime effective-address proof",
      "",
      "**"+verdict+"**; ONE original Digits 1000-step native fit; "
      "48 exact register-sampled memory operand evaluations.",
      "",
      "Original source, graph, all eight full states, genuine Numba seeded MT19937 next value "
      "and original stop agree with original #239 Linux and #236 Linux/Windows.",
      "",
      "| Verified original JIT offset | Exact opcode | Expected real NumPy allocation | Live correct samples |",
      "|---|---|---|---:|",
    ]
    for off,(name,width,registers,opcode) in OFFSETS.items():
        lines.append("| `"+off+"` | `"+opcode+"` | `"+name+"` | 16/16 |")
    lines+=["","**Hard boundaries:** Only those sampled *native* effective addresses "
        "are proven. We do not assign all historical Callgrind simulated misses "
        "to those allocations, infer hardware cache hits/misses, estimate CPU "
        "time or propose layout/CSR. The original dense padded int64 neighbors "
        "and all earlier CPU NO-GOs remain binding."]
    Path("numba_native_gdb_allocation_report.md").write_text(
        "\n".join(lines)+"\n",encoding="utf-8")
    print("\n".join(lines),flush=True)


if __name__=="__main__":
    check(sys.argv[1])
