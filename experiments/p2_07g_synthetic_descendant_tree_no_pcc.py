"""P2-07G: STAND-ALONE synthetic process TREE used only inside locked OCI.

No PCC imports, no UCI source access, no scientific runner, no graph/mask,
no synthetic or real PCC fit. Three Python standard-library processes:
root -> child -> grandchild. The 'detached' descendant starts a NEW Linux
session, and the 'stubborn' parent and descendants ignore SIGTERM. The
'synthetic_crash' root deliberately exits nonzero on a host-issued marker.
Never fork more than two descendants. All temporary files live on a bounded
container tmpfs, not the host / real research working directory.
"""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

MODES=("normal_tree","detached_tree","stubborn_tree","synthetic_crash")
ROOT=Path("/tmp")
READY_CHILD=ROOT/"p207g-child.pid"
READY_GRANDCHILD=ROOT/"p207g-grandchild.pid"
TRIGGER=ROOT/"p207g-trigger-crash"
PYTHON="/opt/hostedtoolcache/Python/3.12.16/x64/bin/python3.12"


def block_sigterm(mode):
    if mode=="stubborn_tree":
        signal.signal(signal.SIGTERM,signal.SIG_IGN)


def write_pid(path):
    path.write_text(str(os.getpid())+"\n",encoding="ascii")


def sleep_forever():
    while True:
        time.sleep(0.15)


def main():
    if len(sys.argv)==2 and sys.argv[1] in MODES:
        mode=sys.argv[1]
        block_sigterm(mode)
        # child process is intentionally unprivileged, same cgroup, in a
        # distinct session for detached mode; NOT a daemon escaping Docker.
        p=subprocess.Popen(
            [PYTHON,__file__,"--child",mode],
            stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=(mode=="detached_tree"))
        limit=time.monotonic()+8
        while time.monotonic()<limit:
            if READY_CHILD.is_file() and READY_GRANDCHILD.is_file():
                break
            if p.poll() is not None:
                raise RuntimeError("Synthetic child exited unexpectedly")
            time.sleep(0.05)
        else:
            raise RuntimeError("Synthetic child/grandchild readiness timed out")
        print(json.dumps({
            "kind":"P2_07G_THREE_PROCESS_TREE_READY",
            "mode":mode,
            "root_ns_pid":os.getpid(),
            "child_ns_pid":int(READY_CHILD.read_text()),
            "grandchild_ns_pid":int(READY_GRANDCHILD.read_text()),
            "grandchild_has_new_session":mode=="detached_tree",
            "all_three_in_one_container_cgroup":True,
            "PCC_kernel_fits":0,
            "official_UCI_source_rows":0,
        },sort_keys=True),flush=True)
        if mode=="synthetic_crash":
            until=time.monotonic()+12
            while time.monotonic()<until:
                if TRIGGER.is_file():
                    # This is a *deliberate* crash, and the child / grandchild
                    # must not survive PID 1 termination of their container.
                    os._exit(23)
                time.sleep(0.05)
            raise RuntimeError("Host never issued synthetic crash marker")
        sleep_forever()
    if len(sys.argv)==3 and sys.argv[1]=="--child" and sys.argv[2] in MODES:
        mode=sys.argv[2]
        block_sigterm(mode)
        gp=subprocess.Popen(
            [PYTHON,__file__,"--grandchild",mode],
            stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=(mode=="detached_tree"))
        write_pid(READY_CHILD)
        # Leave process running until container cgroup dies.
        sleep_forever()
    if len(sys.argv)==3 and sys.argv[1]=="--grandchild" and sys.argv[2] in MODES:
        block_sigterm(sys.argv[2])
        write_pid(READY_GRANDCHILD)
        sleep_forever()
    raise RuntimeError("P2-07G forbids all other modes and arguments")


if __name__=="__main__":
    main()
