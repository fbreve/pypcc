"""Fixed synthetic three-process worker tree for integrated P2-07H sandbox tests.

Only Python stdlib; no PCC imports, no UCI, no graph/mask, no scientific
dataset or actual particle propagation. Each test spawns root/child/grandchild
in one container cgroup; the host observes all three BEFORE a /tmp/go marker.
"""
import errno
import json
import os
from pathlib import Path
import subprocess
import sys
import time

MODES=("normal","wall","cpu","memory","scratch","crash")
PYTHON="/opt/hostedtoolcache/Python/3.12.16/x64/bin/python3.12"
TMP=Path("/tmp")
GO=TMP/"p207h-go"
CHILD_READY=TMP/"p207h-child"
GRAND_READY=TMP/"p207h-grandchild"


def announce(event,mode,**extra):
    print(json.dumps({"P2_07H_fixture_event":event,"mode":mode,
                      "PCC_fits":0,"UCI_rows":0,**extra},
                     sort_keys=True),flush=True)


def until_go():
    end=time.monotonic()+12
    while time.monotonic()<end:
        if GO.exists():
            return
        time.sleep(0.04)
    raise RuntimeError("Host failed to release synthetic test barrier")


def grandchild(mode):
    GRAND_READY.write_text(str(os.getpid()),encoding="ascii")
    until_go()
    if mode=="normal":
        return 0
    if mode=="cpu":
        limit=time.process_time()+40.0
        value=1
        while time.process_time()<limit:
            value=(value*1103515245+12345)%(2**31)
    elif mode=="memory":
        payload=bytearray(256*1024*1024)
        for index in range(0,len(payload),4096):
            payload[index]=1
    elif mode=="scratch":
        try:
            with (TMP/"p207h-8MiB.bin").open("wb",buffering=0) as f:
                for _ in range(8):
                    f.write(b"x"*(1024*1024))
        except OSError as exc:
            if exc.errno==errno.ENOSPC:
                announce("EXPECTED_ENOSPC",mode,errno=exc.errno)
                while True:
                    time.sleep(0.2)
            raise
    while True:
        time.sleep(0.2)


def child(mode):
    p=subprocess.Popen(
        [PYTHON,__file__,"grandchild",mode],
        stdin=subprocess.DEVNULL,
        # stdout inherited so EXPECTED_ENOSPC is visible to host Docker logs.
        start_new_session=(mode in ("cpu","wall")),
    )
    CHILD_READY.write_text(str(os.getpid()),encoding="ascii")
    until_go()
    if mode=="normal":
        return p.wait(timeout=8)
    while True:
        time.sleep(0.2)


def root(mode):
    p=subprocess.Popen([PYTHON,__file__,"child",mode],
                       stdin=subprocess.DEVNULL,
                       start_new_session=(mode=="wall"))
    deadline=time.monotonic()+10
    while time.monotonic()<deadline:
        if CHILD_READY.exists() and GRAND_READY.exists():
            break
        if p.poll() is not None:
            raise RuntimeError("Fixture descendant prematurely stopped")
        time.sleep(0.06)
    else:
        raise RuntimeError("Fixture descendants failed readiness")
    announce("THREE_PROCESS_TREE_READY",mode,
             root_ns_pid=os.getpid(),
             child_ns_pid=int(CHILD_READY.read_text()),
             grandchild_ns_pid=int(GRAND_READY.read_text()))
    until_go()
    if mode=="normal":
        code=p.wait(timeout=8)
        announce("NORMAL_EXIT_ZERO",mode,child_exit_code=code)
        return 0 if code==0 else 7
    if mode=="crash":
        # A different failure mechanism than the host-initiated time/CPU kill:
        # PID 1 stops while both children are still alive.
        os._exit(23)
    while True:
        time.sleep(0.2)


def main():
    if len(sys.argv)!=3 or sys.argv[1] not in ("root","child","grandchild") \
            or sys.argv[2] not in MODES:
        raise RuntimeError("Only six fixed artificial tree cases are allowed")
    role,mode=sys.argv[1:]
    if role=="root":
        return root(mode)
    if role=="child":
        return child(mode)
    return grandchild(mode)


if __name__=="__main__":
    sys.exit(main())
