"""Native Linux GDB parent: 3 bounded read-only live effective-address breakpoints.

Loaded with GDB Python 'source' after target runner is configured in --args.
Do not execute this module with normal Python.
"""
import json
import os
from pathlib import Path

import gdb

OUT=Path(os.environ["PCC_NATIVE_GDB_PROBE"])
READY=OUT/"native_live_array_ranges.json"
RESULT=OUT/"numba_native_gdb_address_samples.json"
EXPECTED={
    "0x36b":("neib_list",8,("r14","rsi",1)),
    "0x3a0":("dominance",8,("rcx","r9",8)),
    "0x3a6":("dist_table",1,("rdx","r8",1)),
}
CAP=16


def fail(msg):
    raise gdb.GdbError("SCIENTIFIC_GDB_PROBE_FAIL "+msg)


def element_coord(relative, info):
    dims=info["shape"]; strides=info["strides"]; width=info["itemsize"]
    if len(dims)!=2 or relative % width:
        fail("sample not on aligned 2D ndarray item")
    if info["C"]:
        i=relative//strides[0]
        j=(relative-i*strides[0])//strides[1]
    elif info["F"]:
        j=relative//strides[1]
        i=(relative-j*strides[1])//strides[0]
    else:
        fail("unknown ndarray memory order")
    if not (0<=i<dims[0] and 0<=j<dims[1]) or i*strides[0]+j*strides[1]!=relative:
        fail("sample does not represent an actual matrix element")
    return [i,j]


class ReadOnlyAddressBreakpoint(gdb.Breakpoint):
    def __init__(self, where, offset, expected, width, registers, allocations):
        super().__init__("*"+hex(where),type=gdb.BP_BREAKPOINT,internal=False)
        self.silent=True
        self.offset=offset
        self.expected=expected
        self.width=width
        self.registers=registers
        self.allocations=allocations
        self.samples=[]
        self.failed=False

    def stop(self):
        try:
            if len(self.samples)>=CAP:
                self.enabled=False
                return False
            pc=int(gdb.parse_and_eval("$pc"))
            if pc!=int(self.location[1:],16):
                fail("wrong instruction pointer at breakpoint "+self.offset)
            rbase,rindex,scale=self.registers
            first=int(gdb.parse_and_eval("$"+rbase))
            second=int(gdb.parse_and_eval("$"+rindex))
            address=(first+second*scale)&((1<<64)-1)
            candidates=[]
            for name, obj in self.allocations.items():
                lo=int(obj["data_ptr"],16);hi=int(obj["end_ptr"],16)
                if lo<=address and address+self.width<=hi:
                    candidates.append(name)
            if candidates!=[self.expected]:
                fail("observed address "+hex(address)+" at "+self.offset+
                     " maps to "+repr(candidates)+" not "+self.expected)
            allocation=self.allocations[self.expected]
            relative=address-int(allocation["data_ptr"],16)
            sample={
                "sample":len(self.samples)+1,
                "native_instruction":hex(pc),
                "effective_address":hex(address),
                "allocation":self.expected,
                "relative_byte_offset":relative,
                "element_coordinates":element_coord(relative,allocation),
                "read_width":self.width,
                "register_base":hex(first),"register_index":hex(second),
            }
            self.samples.append(sample)
            if len(self.samples)==CAP:
                self.enabled=False
        except Exception as exc:
            self.failed=True
            print("SCIENTIFIC_GDB_BREAKPOINT_ERROR", self.offset,repr(exc),flush=True)
            # Stop under GDB instead of silently continuing on a bad allocation.
            return True
        return False


gdb.execute("set pagination off")
gdb.execute("set confirm off")
gdb.execute("set print thread-events off")
gdb.execute("handle SIGTRAP stop noprint nopass")
gdb.execute("run")
if not READY.is_file():
    fail("target did not stop at original wrapper SIGTRAP with live NumPy ranges")
metadata=json.loads(READY.read_text(encoding="utf-8"))
if metadata["schema"]!=1 or not metadata["native_gdb_only"]:
    fail("unexpected live allocation schema")
pointer=int(metadata["llvm_entry"],16)
if metadata["hot_expectations"]!={k:{"bytes":v,"allocation":EXPECTED[k][0],"bytes_read":EXPECTED[k][1]}
                                  for k,v in metadata["hot_machine_bytes"].items()}:
    fail("original hot opcode proof metadata inconsistency")
breaks={}
for offset,(name,width,registers) in EXPECTED.items():
    if metadata["hot_machine_bytes"][offset]!=metadata["hot_expectations"][offset]["bytes"]:
        fail("live native machine opcode byte mismatch")
    bp=ReadOnlyAddressBreakpoint(pointer+int(offset,16),offset,name,width,registers,
                                 metadata["allocations"])
    breaks[offset]=bp
print("GDB_THREE_ACTUAL_ORIGINAL_JIT_BREAKPOINTS_SET",hex(pointer),flush=True)
gdb.execute("continue")
results={
    "schema":1,"original_master":metadata["original_master"],
    "original_case":metadata["original_case"],
    "llvm_entry":metadata["llvm_entry"],
    "numba_step_signature":metadata["numba_step_signature"],
    "live_hot_machine_bytes":metadata["hot_machine_bytes"],
    "expected_hits_per_instruction":CAP,
    "allocation_ranges":metadata["allocations"],
    "samples":{k:bp.samples for k,bp in breaks.items()},
    "any_gdb_breakpoint_failures":any(bp.failed for bp in breaks.values()),
    "read_only_samples":True,
    "not_a_performance_measurement":True,
}
temp=RESULT.with_suffix(".json.tmp")
temp.write_text(json.dumps(results,indent=2,allow_nan=False)+"\n",encoding="utf-8")
temp.replace(RESULT)
for off,bp in breaks.items():
    print("GDB_ORIGINAL_ARRAY_SAMPLE",off,bp.expected,len(bp.samples),flush=True)
if results["any_gdb_breakpoint_failures"] or any(len(bp.samples)!=CAP for bp in breaks.values()):
    fail("not 16 original native effective addresses at each exact instruction")
if not (OUT/"numba_native_gdb_full_fit_source.json").is_file():
    fail("genuine original full fit did not finish and write exact source state")
print("GDB_NATIVE_EFFECTIVE_ADDRESSES_OBSERVED_48_PASS",flush=True)
