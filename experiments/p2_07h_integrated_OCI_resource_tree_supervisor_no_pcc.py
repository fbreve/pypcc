"""P2-07H integrated synthetic 3-process supervisor in EXACT prior archived OCI.

Real Linux cgroup-v2 memory/swap/pids/CPU quota/tmpfs; host watchdogs for
CPU seconds and wall time; original host PID/starttime cleanup from P2-07G.
NO PCC (including synthetic PCC), official UCI rows, graphs, masks, seeds,
production code, image rebuild or scientific execution authorization.

This tests six deliberately tiny *artificial* process-tree workloads. These
are NOT evidence that real UCI 50k fits stay below proposed resource caps.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import time
import uuid

from experiments import p2_07g_original_OCI_descendant_cgroup_supervisor as g
from experiments import p2_07f_sealed_72case_no_fit_administrative_dryrun as f
from experiments import p2_07e_locked_oci_cgroup_resource_supervisor as e

ROOT=Path(__file__).resolve().parents[1]
FIXTURE=ROOT/"experiments/p2_07h_unified_synthetic_three_process_fixture_no_pcc.py"
MODES=("normal","wall","cpu","memory","scratch","crash")
P1_SHA=g.P1_SHA
IMAGE_SHA=g.IMAGE_SHA
IMAGE_ID=g.IMAGE_ID
F_REPORT_SHA=g.F_REPORT_SHA
EVIDENCE_G_RUN=43540323728
GENESIS="0"*64
CPU_THRESHOLD_SECONDS=0.75
WALL_THRESHOLD_SECONDS=1.20
MAX_WALL_PER_SYNTHETIC_CASE_SECONDS=18

class UnifiedBlocked(ValueError):
    pass

def require(ok,reason):
    if not ok:
        raise UnifiedBlocked(reason)

def canonical(obj):
    return json.dumps(obj,sort_keys=True,ensure_ascii=True,allow_nan=False,
                      separators=(",",":")).encode("ascii")

def sha(obj):
    return hashlib.sha256(canonical(obj)).hexdigest()

def docker_args(mode,name):
    require(mode in MODES and re.fullmatch(
        r"pypcc-p207h-(normal|wall|cpu|memory|scratch|crash)-[0-9a-f]{10}",name),
        "Only six fixed source-free no-PCC jobs may run")
    require(FIXTURE.is_file(),"Missing original fixed three-process probe")
    memory="128m" if mode=="memory" else "2g"
    scratch="size=1m" if mode=="scratch" else "size=64m"
    return [
        "docker","run","-d","--name",name,
        "--platform","linux/amd64","--network","none",
        "--read-only","--user","65534:65534","--cap-drop","ALL",
        "--security-opt","no-new-privileges","--pids-limit","128",
        "--memory",memory,"--memory-swap",memory,"--cpus","1.0",
        "--tmpfs",f"/tmp:rw,nosuid,nodev,noexec,mode=1777,{scratch}",
        "--entrypoint",g.PYTHON,
        "--mount",f"type=bind,src={FIXTURE.resolve()},dst=/opt/p207h-fixture.py,readonly",
        e.IMAGE_TAG,"/opt/p207h-fixture.py","root",mode
    ]

def proc_state(cid):
    data=json.loads(g.call(["docker","inspect",cid]).stdout)
    require(len(data)==1,"Lost exact Docker container identity")
    return data[0]["State"]

def fixture_event(cid,mode,event):
    text=g.call(["docker","logs",cid],timeout=7).stdout
    require(len(text.encode("utf-8"))<=4096,"Unexpected or unbounded source-free logs")
    for line in text.splitlines():
        try:
            value=json.loads(line)
        except json.JSONDecodeError:
            continue
        if value.get("P2_07H_fixture_event")==event:
            require(value.get("mode")==mode and value.get("PCC_fits")==0
                    and value.get("UCI_rows")==0,
                    "Fixture reported forbidden scientific activity")
            return value
    return None

def await_tree_ready(cid,mode):
    deadline=time.monotonic()+12
    while time.monotonic()<deadline:
        require(proc_state(cid)["Running"] is True,
                "Fixture root exited before three descendant identities collected")
        row=fixture_event(cid,mode,"THREE_PROCESS_TREE_READY")
        if row is not None:
            require(len({row["root_ns_pid"],row["child_ns_pid"],
                         row["grandchild_ns_pid"]})==3,
                    "Fixture reports repeated namespace PIDs")
            return row
        time.sleep(0.08)
    raise UnifiedBlocked("Three-process ready barrier was not observed")

def cgroup_probe_source():
    return (
        "import json,os;"
        "from pathlib import Path;"
        "c=Path('/sys/fs/cgroup');"
        "cpu=dict(x.split() for x in (c/'cpu.stat').read_text().splitlines());"
        "mem=dict(x.split() for x in (c/'memory.events').read_text().splitlines());"
        "s=os.statvfs('/tmp');"
        "print(json.dumps({"
          "'cpu_usage_usec':int(cpu['usage_usec']),"
          "'oom_kill':int(mem['oom_kill']),"
          "'memory_peak':int((c/'memory.peak').read_text()),"
          "'memory_max':(c/'memory.max').read_text().strip(),"
          "'memory_swap_max':(c/'memory.swap.max').read_text().strip(),"
          "'pids_max':(c/'pids.max').read_text().strip(),"
          "'cpu_max':(c/'cpu.max').read_text().strip(),"
          "'scratch_available_bytes':s.f_bavail*s.f_frsize},sort_keys=True))"
    )

def cgroup_counters(cid):
    script=cgroup_probe_source()
    p=g.call(["docker","exec","--user","65534:65534",cid,g.PYTHON,
              "-c",script],timeout=9,success=False)
    if p.returncode!=0:
        # Preserve source-free bounded diagnostics. During a REAL container
        # exit the host loop can still inspect Docker State.OOMKilled, but
        # before the release barrier a failed counter read must fail loudly.
        if p.stderr.strip():
            raise UnifiedBlocked("Cgroup-v2 probe failed: "+p.stderr.strip()[-500:])
        return None  # kernel can have stopped the entire cgroup
    try:
        out=json.loads(p.stdout)
    except (json.JSONDecodeError,ValueError) as exc:
        raise UnifiedBlocked("Kernel cgroup counters unavailable/malformed") from exc
    require(all(type(out[k]) is int and out[k]>=0 for k in
                ("cpu_usage_usec","oom_kill","memory_peak",
                 "scratch_available_bytes")),
            "Kernel cgroup counters not nonnegative numbers")
    return out

def check_caps(counters,mode):
    require(counters is not None,
            "No cgroup v2 counters before synthetic release barrier")
    memory="134217728" if mode=="memory" else "2147483648"
    scratch=1024**2 if mode=="scratch" else 64*1024**2
    quota=counters["cpu_max"].split()
    require(counters["memory_max"]==memory and
            counters["memory_swap_max"]=="0" and
            counters["pids_max"]=="128" and
            len(quota)==2 and all(x.isdigit() for x in quota) and
            0<int(quota[0])<=int(quota[1]) and
            0<counters["scratch_available_bytes"]<=scratch,
            "Actual OS cgroup-v2/swap/CPU/pids/tmpfs restrictions differ")
    return {"hard_memory_max_bytes":int(memory),
            "hard_swap_max_bytes":0,
            "hard_pids_max":128,"cpu_quota_cores_at_most":1,
            "hard_tmpfs_quota_bytes":scratch}

def journal_append(path,prev,mode,event,**details):
    require(mode in MODES and event in (
        "STARTED","TREE_AND_HARD_CAPS_VERIFIED","RELEASED_SYNTHETIC_BARRIER",
        "TERMINATION_DECISION_RECORDED","ALL_PIDS_AND_CONTAINER_GONE",
        "FAIL_CLOSED"),
        "Only source-free synthetic events may be journaled")
    require(re.fullmatch("[0-9a-f]{64}",prev) is not None,
            "Prior journal entry SHA missing")
    row={"mode":mode,"event":event,"previous_sha256":prev,
         "P1_72_scientific_fit_states":"ALL_NOT_RUN",
         "synthetic_PCC_fits":0,"real_PCC_fits":0}
    row.update(details)
    row["sha256"]=sha(row)
    encoded=canonical(row)
    require(len(encoded)<=2048,"Journal row source-free size limit exceeded")
    with Path(path).open("ab") as out:
        out.write(encoded+b"\n")
        out.flush()
        os.fsync(out.fileno())
    return row["sha256"]

def validate_journal(path,complete=False):
    rows=[json.loads(x) for x in Path(path).read_bytes().splitlines()]
    prev=GENESIS
    for row in rows:
        require(row.get("mode") in MODES and
                row.get("previous_sha256")==prev and
                row.get("sha256")==sha({k:v for k,v in row.items()
                                       if k!="sha256"}) and
                row.get("P1_72_scientific_fit_states")=="ALL_NOT_RUN" and
                row.get("synthetic_PCC_fits")==row.get("real_PCC_fits")==0,
                "Journal chain tamper, scientific claim or missing record")
        prev=row["sha256"]
    if complete:
        seq=("STARTED","TREE_AND_HARD_CAPS_VERIFIED",
             "RELEASED_SYNTHETIC_BARRIER","TERMINATION_DECISION_RECORDED",
             "ALL_PIDS_AND_CONTAINER_GONE")
        require(len(rows)==30,"Six cases x five complete journal events required")
        for i,mode in enumerate(MODES):
            block=rows[i*5:i*5+5]
            require([x.get("mode") for x in block]==[mode]*5 and
                    tuple(x.get("event") for x in block)==seq,
                    "Duplicate/extra/retried case or after-failure continuation")
    return {"records":len(rows),"tail_sha256":prev}

def confirm_closed(cid,host_identities):
    # PID/starttime prevents accepting a recycled PID as an original survivor.
    g.ensure_all_gone(host_identities)
    p=g.call(["docker","inspect",cid],success=False)
    require(p.returncode!=0,"Even after cleanup container is inspectable")
    return True

def one(mode,ledger,prev):
    name=f"pypcc-p207h-{mode}-{uuid.uuid4().hex[:10]}"
    cid=None
    observed=[]
    started=time.monotonic()
    prev=journal_append(ledger,prev,mode,"STARTED")
    try:
        cid=g.call(docker_args(mode,name),timeout=25).stdout.strip()
        require(re.fullmatch("[0-9a-f]{64}",cid) is not None,
                "Daemon returned invalid Docker container ID")
        await_tree_ready(cid,mode)
        observed=g.observe_three(cid)  # original host PID/starttime+cgroup proof
        before=cgroup_counters(cid)
        caps=check_caps(before,mode)
        prev=journal_append(ledger,prev,mode,"TREE_AND_HARD_CAPS_VERIFIED",
                            exact_three_original_processes=3)
        # Start the synthetic work ONLY after all three host PIDs and kernel
        # cgroup limits are verified (memory OOM may kill one descendant).
        g.call(["docker","exec","--user","65534:65534",cid,g.PYTHON,
                "-c","from pathlib import Path;Path('/tmp/p207h-go').touch()"],
               timeout=10)
        prev=journal_append(ledger,prev,mode,"RELEASED_SYNTHETIC_BARRIER")
        trigger_time=time.monotonic()
        reason=None
        last_cpu_s=0.0
        last_memory_peak=before["memory_peak"]
        kernel_oom_event_delta=0
        while True:
            elapsed=time.monotonic()-trigger_time
            require(elapsed<MAX_WALL_PER_SYNTHETIC_CASE_SECONDS,
                    "Absolute source-free case watchdog failed closed")
            state=proc_state(cid)
            if not state["Running"]:
                if mode=="normal" and state["ExitCode"]==0:
                    reason="NORMAL_ALL_THREE_PROCESS_EXIT_0"
                elif mode=="crash" and state["ExitCode"]==23:
                    reason="ROOT_EXIT_23_AND_CONTAINER_KILLED_DESCENDANTS"
                elif mode=="memory" and state["OOMKilled"]:
                    reason="KERNEL_CGROUP_OOM_KILLED_CONTAINER"
                    kernel_oom_event_delta=1
                else:
                    raise UnifiedBlocked(
                        "Unexpected early synthetic root termination or exit code")
                break
            # A single host loop now owns BOTH resource accounting and
            # descendants; cgroup CPU usage includes docker-exec overhead.
            counters=cgroup_counters(cid)
            if counters is not None:
                last_cpu_s=max(last_cpu_s,
                               (counters["cpu_usage_usec"]-
                                before["cpu_usage_usec"])/1_000_000)
                last_memory_peak=max(last_memory_peak,counters["memory_peak"])
                kernel_oom_event_delta=max(
                    kernel_oom_event_delta,
                    counters["oom_kill"]-before["oom_kill"])
            if mode=="wall" and elapsed>=WALL_THRESHOLD_SECONDS:
                reason="HOST_SAMPLED_WALL_THRESHOLD_STOP"
            elif mode=="cpu" and last_cpu_s>=CPU_THRESHOLD_SECONDS:
                reason="HOST_SAMPLED_CGROUP_CPU_THRESHOLD_STOP"
            elif mode=="memory" and kernel_oom_event_delta>0:
                reason="KERNEL_OOM_EVENT_DESCENDANT_FORCED_FULL_TREE_STOP"
            elif mode=="scratch" and fixture_event(
                    cid,mode,"EXPECTED_ENOSPC") is not None:
                reason="KERNEL_TMPFS_ENOSPC_FORCED_FULL_TREE_STOP"
            if reason is not None:
                g.call(["docker","kill",cid],timeout=15)
                break
            time.sleep(0.12)
        deadline=time.monotonic()+10
        while time.monotonic()<deadline and proc_state(cid)["Running"]:
            time.sleep(0.10)
        final=proc_state(cid)
        require(final["Running"] is False,"Synthetic worker container survived")
        require(reason is not None,"Missing hard-OS or host watchdog decision")
        if mode=="normal":
            require(final["ExitCode"]==0,"Normal tree did not complete zero")
        elif mode=="crash":
            require(final["ExitCode"]==23,"Root crash exit code not preserved")
        else:
            require(final["ExitCode"]!=0,"Failed synthetic case escaped cap")
        if mode=="memory":
            require(kernel_oom_event_delta>0 or final["OOMKilled"],
                    "Must observe actual kernel cgroup OOM event")
        if mode=="cpu":
            require(last_cpu_s>=CPU_THRESHOLD_SECONDS,
                    "No actual cgroup cumulative CPU limit observed")
        if mode=="wall":
            require(time.monotonic()-trigger_time>=WALL_THRESHOLD_SECONDS,
                    "Wall limit not reached")
        if mode=="scratch":
            require(reason=="KERNEL_TMPFS_ENOSPC_FORCED_FULL_TREE_STOP",
                    "Missing actual ENOSPC signal from grandchild")
        prev=journal_append(
            ledger,prev,mode,"TERMINATION_DECISION_RECORDED",
            synthetic_reason=reason,root_exit_code=final["ExitCode"],
            observed_cgroup_OOM_event=kernel_oom_event_delta>0)
        # Destruction/descendant proof required for *every* mode, including
        # kernel OOM of ONLY the grandchild and sudden PID1 crash.
        g.call(["docker","rm","-f",cid],timeout=15)
        confirm_closed(cid,observed)
        prev=journal_append(ledger,prev,mode,"ALL_PIDS_AND_CONTAINER_GONE",
                            exact_original_PID_starttime_identities_gone=3)
        return prev,{
            "mode":mode,"termination_reason":reason,
            "root_exit_code":final["ExitCode"],
            "real_three_initial_host_PID_starttime_identities_gone":True,
            "actual_kernel_cgroup_oom_kill_event_observed":
                kernel_oom_event_delta>0,
            "actual_tmpfs_ENOSPC_observed":
                mode=="scratch" and
                reason=="KERNEL_TMPFS_ENOSPC_FORCED_FULL_TREE_STOP",
            "cgroup_cpu_usage_seconds_at_last_poll":
                round(max(0,last_cpu_s),6),
            "cgroup_memory_peak_bytes_at_last_poll":last_memory_peak,
            "hard_kernel_quotas":caps,
            "synthetic_PCC_fits_executed":0,"real_PCC_fits_executed":0,
        }
    except BaseException as exc:
        journal_append(ledger,prev,mode,"FAIL_CLOSED",
                       exception_class=type(exc).__name__)
        raise
    finally:
        # Unconditional orphan cleanup, regardless of failure stage.
        g.call(["docker","rm","-f",cid or name],timeout=12,success=False)

def validate_report(report):
    require(report.get("schema")==
            "pypcc-P2-07H-unified-source-free-three-process-resource-supervisor/v1"
            and report.get("status")==
            "SIX_INTEGRATED_KERNEL_AND_PARENT_PROBES_PASS_ZERO_PCC" and
            report.get("P1_scientific_sha256")==P1_SHA and
            report.get("exact_OCI_binary_gzip_sha256")==IMAGE_SHA and
            report.get("exact_OCI_config_id")==IMAGE_ID and
            report.get("source_72case_P2_07F_canonical_sha256")==F_REPORT_SHA and
            report.get("prior_P2_07G_actual_descendants_verified") is True,
            "Source provenance or original native runtime changed")
    cases=report.get("synthetic_three_process_integrated_cases",[])
    require(len(cases)==6 and [x.get("mode") for x in cases]==list(MODES)
            and all(x.get("real_three_initial_host_PID_starttime_identities_gone")
                    is True and x.get("synthetic_PCC_fits_executed")==0
                    and x.get("real_PCC_fits_executed")==0 and
                    x.get("hard_kernel_quotas",{}).get("hard_swap_max_bytes")==0
                    and x.get("hard_kernel_quotas",{}).get("hard_pids_max")==128
                    for x in cases),
            "A process survived or kernel quotas were not verified")
    require(cases[0].get("root_exit_code")==0 and
            cases[1].get("termination_reason")==
            "HOST_SAMPLED_WALL_THRESHOLD_STOP" and
            cases[2].get("termination_reason")==
            "HOST_SAMPLED_CGROUP_CPU_THRESHOLD_STOP" and
            cases[3].get("actual_kernel_cgroup_oom_kill_event_observed") is True
            and cases[3].get("hard_kernel_quotas",{}).
            get("hard_memory_max_bytes")==128*1024**2 and
            cases[4].get("actual_tmpfs_ENOSPC_observed") is True
            and cases[4].get("hard_kernel_quotas",{}).
            get("hard_tmpfs_quota_bytes")==1024**2 and
            cases[5].get("root_exit_code")==23,
            "One of the six actual integrated budget/descendant outcomes absent")
    require(report.get("bounded_original_PID_identities_confirmed_gone")==18
            and report.get("journal_records")==30 and
            re.fullmatch("[0-9a-f]{64}",report.get("journal_final_sha256","")) is not None,
            "No complete six-case fail-closed journal")
    for key,value in (
        ("frozen_P1_all_72_scientific_cases_NOT_RUN",True),
        ("P2_07H_is_not_a_scientific_worker_execution_supervisor",True),
        ("real_graph_50k_resource_upper_bounds_known",False),
        ("full_host_OS_compiler_and_native_closure_complete",False),
        ("host_cpu_watchdog_sampling_is_a_hard_limit",False),
        ("per_fit_real_PCC_runner_included",False),
        ("real_UCI_source_rows_loaded",0),
        ("graphs_or_masks_built",0),
        ("synthetic_PCC_fits_executed",0),
        ("real_PCC_fits_executed",0),
        ("P2_01_final_execution_toolchain_lock_pass",False),
        ("P2_07_full_resources_gate_pass",False),
        ("P2_execution_preflight_sha256",None),
        ("P2_execution_authorized",False),
    ):
        require(report.get(key)==value,
                "Synthetic supervisor must not grant scientific P2 GO: "+key)
    return True

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--existing-oci-artifact",type=Path,required=True)
    p.add_argument("--old-72case-report",type=Path,required=True)
    p.add_argument("--old-tree-report",type=Path,required=True)
    p.add_argument("--journal",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    require(platform.system()=="Linux" and
            platform.machine().lower() in ("x86_64","amd64"),
            "Kernel cgroup and /proc PID proof requires Linux amd64 host")
    require(not a.output.exists() and not a.journal.exists(),
            "Do not overwrite or retry existing integrated attempt evidence")
    oldf=json.loads(a.old_72case_report.read_text(encoding="utf-8"))
    oldg=json.loads(a.old_tree_report.read_text(encoding="utf-8"))
    require(f.validate_report(oldf) and f.sha(oldf)==F_REPORT_SHA and
            oldf["P2_execution_authorized"] is False and
            oldf["real_PCC_fits_executed"]==0,
            "Original P2-07F all 72 NOT_RUN receipt required")
    require(g.validate_result(oldg) and
            oldg["real_PCC_fits_executed"]==0,
            "Original P2-07G 4/4 true original descendant evidence required")
    host=e.verify_original_image(a.existing_oci_artifact.resolve())
    require(host["loaded_exact_original_image_ID"]==IMAGE_ID,
            "Cannot use rebuilt or differently identified OCI image")
    cases=[]
    prev=GENESIS
    for mode in MODES:
        prev,row=one(mode,a.journal,prev)
        cases.append(row)
    audit=validate_journal(a.journal,complete=True)
    require(audit["tail_sha256"]==prev,
            "Journal integrity mismatch after six independent source-free cases")
    report={
        "schema":"pypcc-P2-07H-unified-source-free-three-process-resource-supervisor/v1",
        "status":"SIX_INTEGRATED_KERNEL_AND_PARENT_PROBES_PASS_ZERO_PCC",
        "P1_scientific_sha256":P1_SHA,
        "exact_OCI_binary_gzip_sha256":IMAGE_SHA,
        "exact_OCI_config_id":IMAGE_ID,
        "source_72case_P2_07F_canonical_sha256":F_REPORT_SHA,
        "prior_P2_07G_actual_descendants_verified":True,
        "prior_P2_07G_artifact_run":EVIDENCE_G_RUN,
        "synthetic_three_process_integrated_cases":cases,
        "bounded_original_PID_identities_confirmed_gone":18,
        "journal_records":audit["records"],
        "journal_final_sha256":audit["tail_sha256"],
        "frozen_P1_all_72_scientific_cases_NOT_RUN":True,
        "P2_07H_is_not_a_scientific_worker_execution_supervisor":True,
        "real_graph_50k_resource_upper_bounds_known":False,
        "full_host_OS_compiler_and_native_closure_complete":False,
        "host_cpu_watchdog_sampling_is_a_hard_limit":False,
        "per_fit_real_PCC_runner_included":False,
        "real_UCI_source_rows_loaded":0,
        "graphs_or_masks_built":0,
        "synthetic_PCC_fits_executed":0,
        "real_PCC_fits_executed":0,
        "P2_01_final_execution_toolchain_lock_pass":False,
        "P2_07_full_resources_gate_pass":False,
        "P2_execution_preflight_sha256":None,
        "P2_execution_authorized":False,
    }
    validate_report(report)
    a.output.write_text(json.dumps(report,sort_keys=True,indent=2)+"\n",
                        encoding="utf-8")
    print(json.dumps({"status":report["status"],
                      "original_host_PID_identifiers_disappeared":18,
                      "source_free_journal_records":30,
                      "real_PCC_fits_executed":0,
                      "P2_execution_authorized":False},sort_keys=True))

if __name__=="__main__":
    main()
