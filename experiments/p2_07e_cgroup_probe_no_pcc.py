"""Synthetic, standalone cgroup-v2 probes inside exact archived P2-01I OCI image.

NO imports of PCC or UCI, no model/graph/mask/snapshot creation, NO Numba JIT.
Used ONLY as child of the fail-closed host Docker supervisor. Python standard
library only; all source and log output describes synthetic resource conditions.
"""
import errno
import json
import os
from pathlib import Path
import sys
import time

CG=Path("/sys/fs/cgroup")


def cgroup_snapshot():
    names=("memory.max","memory.swap.max","memory.peak",
           "memory.events","cpu.max","cpu.stat","pids.max")
    missing=[name for name in names if not (CG/name).is_file()]
    if missing:
        raise RuntimeError("cgroup v2 required: missing "+",".join(missing))
    return {name:(CG/name).read_text(encoding="utf-8").strip()
            for name in names}


def output(**kwargs):
    print(json.dumps(kwargs,sort_keys=True),flush=True)


def main():
    if len(sys.argv)!=2 or sys.argv[1] not in (
            "baseline","wall_violation","cpu_violation",
            "memory_violation","scratch_violation"):
        raise RuntimeError("Only five synthetic no-PCC modes allowed")
    mode=sys.argv[1]
    snapshot=cgroup_snapshot()
    if mode=="baseline":
        assert snapshot["memory.max"]==str(2*1024**3)
        assert snapshot["memory.swap.max"]=="0"
        quota,period=(int(x) for x in snapshot["cpu.max"].split())
        assert quota>0 and period>0 and quota/period <= 1.0+1e-9
        assert snapshot["pids.max"]=="128"
        st=os.statvfs("/tmp")
        available=st.f_bavail*st.f_frsize
        assert available>0 and available<=64*1024**2
        assert (CG/"cgroup.controllers").is_file()
        output(mode=mode,status="CGROUP_V2_BASELINE_PASS",
               memory_max_bytes=int(snapshot["memory.max"]),
               swap_max_bytes=int(snapshot["memory.swap.max"]),
               quota_microseconds=quota,period_microseconds=period,
               pids_limit=int(snapshot["pids.max"]),
               scratch_available_bytes=available,
               cgroup_memory_peak_bytes=int(snapshot["memory.peak"]),
               cgroup_cpu_stat_counters=snapshot["cpu.stat"],
               real_UCI_rows=0,real_PCC_fits_executed=0)
        return 0
    if mode=="wall_violation":
        time.sleep(60)  # parent MUST enforce tiny wall deadline
        output(mode=mode,unexpected="UNSTOPPED_WALL_PROBE")
        return 10
    if mode=="cpu_violation":
        end=time.process_time()+60.0
        accumulator=0
        while time.process_time()<end:
            accumulator=(accumulator*7+1)%1000000007
        output(mode=mode,unexpected="UNSTOPPED_CPU_PROBE",value=accumulator)
        return 11
    if mode=="memory_violation":
        # cgroup memory.max=128MiB, swap disabled; touch the 256MiB page
        # allocation deliberately. Kernel OOM kill MUST be observed by parent.
        buf=bytearray(256*1024**2)
        for i in range(0,len(buf),4096):
            buf[i]=1
        output(mode=mode,unexpected="UNSTOPPED_MEMORY_PROBE",bytes=len(buf))
        return 12
    if mode=="scratch_violation":
        path=Path("/tmp")/"synthetic-mount-limit.bin"
        try:
            with path.open("wb",buffering=0) as out:
                for i in range(8):
                    out.write(b"\x7a"*(1024**2))
        except OSError as e:
            if e.errno==errno.ENOSPC:
                output(mode=mode,status="KERNEL_TMPFS_ENOSPC_CONFIRMED",
                       errno=e.errno,scratch_limit_candidate_bytes=1024**2,
                       source_rows_loaded=0,PCC_propagations_executed=0)
                return 42
            raise
        output(mode=mode,unexpected="UNLIMITED_TMPFS_WRITE")
        return 13
    raise RuntimeError("unreachable")


if __name__=="__main__":
    sys.exit(main())
