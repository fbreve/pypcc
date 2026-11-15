"""P2-07R: ACTIVE original-OCI resource supervisor for the 3 P2-07Q FAKE 50k pairs.

Previously, P2-07Q enforced cgroup 2GiB memory and a host subprocess.wait
timeout, but only OBSERVED fit CPU. This research-only integration starts the
historically byte-locked P2-07Q fake-only worker behind a tmpfs release barrier,
checks actual original Docker/cgroup-v2 limits and host PID/starttime before
releasing it, and samples cumulative cgroup CPU/wall while the 50k native
kernel/8-checkpoint observer runs. CPU/wall threshold breach kills the exact
container, verifies that the original root PID disappeared, and STOPs.

3 normal cases execute 6 existing P2-07Q FAKE 50k fits. Two *no-fit*
fault injection modes prove actual sampled wall and CPU termination.
NEVER loads real UCI data, creates P1 masks/seeds, alters scientific source,
approves an execution envelope, or runs one of the 72 original P1 cases.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
import time
import uuid

from experiments import p2_07q_exact_oci_synthetic_live_observer_50k as q
from experiments import p2_07h_integrated_OCI_resource_tree_supervisor_no_pcc as h
from experiments import p2_07g_original_OCI_descendant_cgroup_supervisor as g
from experiments import p2_07e_locked_oci_cgroup_resource_supervisor as e

SCHEMA="pypcc-P2-07R-original-OCI-active-cgroup-CPU-wall-supervisor-FAKE-50k/v1"
STATUS="THREE_FAKE_PCC_OBSERVER_PAIRS_AND_TWO_LIVE_RESOURCE_FAULT_STOPS_PROVED"
MODES=("numpy","numba","cython","cpu_fault","wall_fault")
WORKER_MODE="normal"
BARRIER=Path("/tmp/p207r-release-fake-only")
P1_SHA=q.P1_SHA
OCI_SHA=q.OCI_SHA
OCI_ID=q.OCI_ID
CPU_LIMIT_NORMAL_S=120.0
WALL_LIMIT_NORMAL_S=240.0
CPU_LIMIT_FAULT_S=0.7
WALL_LIMIT_FAULT_S=1.0
MAX_WALL_PROCESS_S=285.0
MAX_SAMPLED_OUTPUT=19000
GENESIS="0"*64
NO_GO=q.NO_GO.copy()

class ResourceBlocked(ValueError):
    pass

def require(ok,why):
    if not ok:
        raise ResourceBlocked(why)

def canonical(obj):
    return json.dumps(obj,sort_keys=True,separators=(",",":"),
                      ensure_ascii=True,allow_nan=False).encode("ascii")

def digest(obj):
    return hashlib.sha256(canonical(obj)).hexdigest()

def source_locked():
    require(q.source_locked(),
            "Original P1, production and historical Cython/Numba/NumPy sources drifted")
    return True

def original_worker(backend,mode):
    """A gate/barrier is not P1 science permission; permits fake only."""
    require(platform.system()=="Linux" and
            platform.python_version()=="3.12.16" and
            platform.machine().lower() in ("x86_64","amd64") and
            os.environ.get("P207Q_FAKE_ONLY")=="1" and
            os.environ.get("P207R_FAKE_ONLY")=="1" and
            os.environ.get("PYTHONPATH")==q.p.k.CORRECT_PATH and
            ((mode=="normal" and backend in q.BACKENDS) or
             (mode in ("cpu_fault","wall_fault") and backend=="numpy")),
            "Only original immutable OCI fake runner with no scientific inputs")
    source_locked()
    print(canonical({
        "kind":"P207R_FAKE_ONLY_READY","mode":mode,"backend":backend,
        "P1_72_real_fits_executed":0,
        "P2_execution_authorized":False
    }).decode("ascii"),flush=True)
    deadline=time.monotonic()+40
    while not BARRIER.is_file():
        if time.monotonic()>deadline:
            raise ResourceBlocked("Fake-only host failed closed before release barrier")
        time.sleep(0.05)
    require(BARRIER.read_text(encoding="ascii")=="SYNTHETIC_ONLY\n",
            "Host barrier missing exact FAKE-only content")
    if mode=="cpu_fault":
        start=time.process_time()
        while time.process_time()-start<12:
            pass  # deliberatively bounded no-fit CPU breach
        raise ResourceBlocked("Synthetic CPU fault was NOT killed by host watchdog")
    if mode=="wall_fault":
        time.sleep(12)
        raise ResourceBlocked("Synthetic wall fault was NOT killed by host watchdog")
    # The previous exact Q.worker implements all full-native 50k checks.
    # Never use source-specific UCI/masks or different fit parameters.
    row=q.worker(backend)
    q.validate_worker(row,backend)
    print(q.canonical(row).decode("ascii"),flush=True)

def docker_command(case,name):
    require(case in MODES and
            re.fullmatch(r"pypcc-p207r-[0-9a-f]{12}",name) is not None,
            "Only one original OCI synthetic observer case or two no-fit faults")
    backend=case if case in q.BACKENDS else "numpy"
    cmd=q.docker_args(backend,name.replace("p207r","p207q",1))
    cmd.remove("--rm")  # Keep exact Docker state until inspect/cleanup verified.
    cmd.insert(2,"-d")
    cmd[cmd.index("--name")+1]=name
    pos=cmd.index("--entrypoint")
    cmd[pos:pos]=["--env","P207R_FAKE_ONLY=1"]
    cmd[cmd.index("experiments.p2_07q_exact_oci_synthetic_live_observer_50k")]=(
        "experiments.p2_07r_supervise_original_oci_fake_checkpoint_fits")
    cmd.extend(["--mode",WORKER_MODE if case in q.BACKENDS else case])
    return cmd

def actual_hard_caps(config,counters):
    """Require actual daemon + kernel settings, not merely Docker CLI args."""
    require(type(config) is dict and type(counters) is dict and
            config.get("Memory")==2*1024**3 and
            config.get("MemorySwap")==2*1024**3 and
            config.get("NanoCpus")==10**9 and
            config.get("PidsLimit")==128 and
            config.get("ReadonlyRootfs") is True and
            config.get("NetworkMode")=="none" and
            "ALL" in [str(x).upper() for x in (config.get("CapDrop") or [])],
            "Original Docker HostConfig hard memory/swap/cpu/pids/rootfs/network restrictions absent")
    require(counters.get("memory_max")=="2147483648" and
            counters.get("memory_swap_max")=="0" and
            counters.get("pids_max")=="128" and
            type(counters.get("memory_peak")) is int and
            0<counters["memory_peak"]<=2*1024**3 and
            type(counters.get("oom_kill")) is int and
            counters["oom_kill"]>=0 and
            type(counters.get("scratch_available_bytes")) is int and
            0<counters["scratch_available_bytes"]<=q.p.TMPFS_MAX,
            "Actual original OCI cgroup v2/readonly 128MiB tmpfs not enforced")
    quota=str(counters.get("cpu_max","")).split()
    require(len(quota)==2 and all(x.isdigit() for x in quota) and
            0<int(quota[0])<=int(quota[1]),
            "Original cgroup CPU quota does not limit runtime to <=1 core")
    return True

def decision(case,elapsed,cpu_since_release,oom):
    require(case in MODES and
            type(elapsed) in (int,float) and elapsed>=0 and
            type(cpu_since_release) in (int,float) and cpu_since_release>=0 and
            type(oom) is int and oom>=0,
            "Invalid sampled runtime watch inputs")
    if oom:
        return "REAL_CGROUP_OOM_FAIL_CLOSED"
    wall_budget=WALL_LIMIT_NORMAL_S if case in q.BACKENDS else (
        WALL_LIMIT_FAULT_S if case=="wall_fault" else MAX_WALL_PROCESS_S)
    cpu_budget=CPU_LIMIT_NORMAL_S if case in q.BACKENDS else (
        CPU_LIMIT_FAULT_S if case=="cpu_fault" else CPU_LIMIT_NORMAL_S)
    if elapsed>=wall_budget:
        return "SAMPLED_WALL_BUDGET_KILL"
    if cpu_since_release>=cpu_budget:
        return "SAMPLED_CGROUP_CPU_BUDGET_KILL"
    return None

def inspect(cid):
    obj=json.loads(g.call(["docker","inspect",cid],timeout=12).stdout)
    require(type(obj) is list and len(obj)==1 and
            obj[0].get("Id")==cid,
            "Historical exact Docker container ID changed")
    return obj[0]

def ready(cid,case):
    """Worker MUST be alive behind fake-only barrier before any PCC execution."""
    deadline=time.monotonic()+20
    while time.monotonic()<deadline:
        item=inspect(cid)
        require(item["State"]["Running"] is True,
                "Original fake-only worker died before host resource preflight")
        log=g.call(["docker","logs",cid],timeout=7).stdout
        require(len(log.encode())<6000,"Original fake-only pre-release log exceeded bound")
        for line in log.splitlines():
            try:
                row=json.loads(line)
            except ValueError:
                continue
            if row.get("kind")=="P207R_FAKE_ONLY_READY":
                expected_backend=case if case in q.BACKENDS else "numpy"
                require(row=={
                    "kind":"P207R_FAKE_ONLY_READY",
                    "mode":WORKER_MODE if case in q.BACKENDS else case,
                    "backend":expected_backend,
                    "P1_72_real_fits_executed":0,
                    "P2_execution_authorized":False,
                },"Original OCI worker prelaunch signaled wrong identity")
                return item
        time.sleep(0.07)
    raise ResourceBlocked("Original worker fake-only READY barrier never observed")

def outcome(case,reason,exit_code,cpu_sample,root_gone,worker_result=None):
    require(case in MODES and type(exit_code) is int and
            type(cpu_sample) in (int,float) and cpu_sample>=0 and
            root_gone is True,
            "Missing original source-free result or process termination evidence")
    expected_fault={
        "cpu_fault":"SAMPLED_CGROUP_CPU_BUDGET_KILL",
        "wall_fault":"SAMPLED_WALL_BUDGET_KILL",
    }
    if case in q.BACKENDS:
        require(reason=="NORMAL_NATIVE_8CHECKPOINTS_50K_FAKE_EXIT_0" and
                exit_code==0 and worker_result is not None and
                q.validate_worker(worker_result,case),
                "Original successful fake 50k checkpoint worker did not exit cleanly")
        orig,observed=worker_result["original_native_fullfit_and_historical_observer"]
        return {
            "case":case,"termination_reason":reason,
            "real_original_P1_fits_executed":0,
            "synthetic_full_50k_fits_executed":2,
            "eight_milestones_verified":True,
            "final_mutable_state_prediction_parity":True,
            "observed_snapshot_bytes":observed["snapshot_array_bytes"],
            "normal_fit_cpu_seconds":orig["fit_cpu_seconds"],
            "observed_fit_cpu_seconds":observed["fit_cpu_seconds"],
            "process_peak_RSS_bytes":worker_result["synthetic_pair_process_peak_RSS_bytes"],
            "cgroup_memory_peak_bytes":worker_result["synthetic_pair_cgroup_v2_peak_bytes"],
            "tmpfs_payload_bytes":worker_result["synthetic_pair_tmpfs_file_payload_bytes"],
            "cgroup_CPU_seconds_at_last_host_sample_includes_exec_overhead":round(cpu_sample,6),
            "original_root_PID_starttime_identity_gone":True,
        }
    require(reason==expected_fault.get(case) and exit_code!=0 and
            worker_result is None,
            "Synthetic forced CPU/wall breach escaped or was misclassified")
    return {
        "case":case,"termination_reason":reason,
        "real_original_P1_fits_executed":0,
        "synthetic_full_50k_fits_executed":0,
        "eight_milestones_verified":False,
        "original_root_PID_starttime_identity_gone":True,
        "cgroup_CPU_seconds_at_last_host_sample_includes_exec_overhead":
            round(cpu_sample,6),
        "host_killed_original_container":True,
    }

def one(case):
    name="pypcc-p207r-"+uuid.uuid4().hex[:12]
    cid=None
    root=None
    try:
        cid=g.call(docker_command(case,name),timeout=25).stdout.strip()
        require(re.fullmatch(r"[0-9a-f]{64}",cid) is not None,
                "No exact 64-hex immutable Docker worker identity")
        item=ready(cid,case)
        config=item["HostConfig"]
        initial=h.cgroup_counters(cid)
        actual_hard_caps(config,initial)
        pid=item["State"]["Pid"]
        root=g.proc_identity(pid)
        require(root is not None and root["pid"]==pid,
                "Original host PID/starttime identity not readable")
        g.call(["docker","exec","--user","65534:65534",cid,
                q.p.PYTHON_IN_OCI,"-c",
                "from pathlib import Path;"
                "Path('/tmp/p207r-release-fake-only').write_text('SYNTHETIC_ONLY\\n',encoding='ascii')"],
                timeout=10)
        released=time.monotonic()
        baseline_cpu=initial["cpu_usage_usec"]
        last_cpu=0.0
        reason=None
        killed_by_host=False
        last_mem=initial["memory_peak"]
        while True:
            elapsed=time.monotonic()-released
            require(elapsed<MAX_WALL_PROCESS_S,
                    "Absolute original OCI FAKE case watchdog timed out")
            state=inspect(cid)["State"]
            if state["Running"] is False:
                require(state["OOMKilled"] is False,
                        "Original OCI actual cgroup OOM kill not acceptable")
                if case in q.BACKENDS and state["ExitCode"]==0:
                    reason="NORMAL_NATIVE_8CHECKPOINTS_50K_FAKE_EXIT_0"
                else:
                    raise ResourceBlocked(
                        "Unexpected original OCI fake root exit before resource decision")
                break
            counters=h.cgroup_counters(cid)
            if counters is not None:
                last_cpu=max(last_cpu,
                             (counters["cpu_usage_usec"]-baseline_cpu)/1e6)
                last_mem=max(last_mem,counters["memory_peak"])
                oom=max(0,counters["oom_kill"]-initial["oom_kill"])
                reason=decision(case,elapsed,last_cpu,oom)
                if reason:
                    g.call(["docker","kill",cid],timeout=12)
                    killed_by_host=True
                    break
            time.sleep(0.09)
        deadline=time.monotonic()+12
        while time.monotonic()<deadline and inspect(cid)["State"]["Running"]:
            time.sleep(.08)
        final=inspect(cid)["State"]
        require(final["Running"] is False,"Original fake worker survived kill/exit")
        if killed_by_host:
            require(final["ExitCode"]!=0,
                    "Original forced kill failed to terminate worker")
        elif case in q.BACKENDS:
            require(final["ExitCode"]==0,
                    "Original fake complete worker exit code corrupted")

        report=None
        if case in q.BACKENDS:
            logs=g.call(["docker","logs",cid],timeout=12).stdout
            require(len(logs.encode())<MAX_SAMPLED_OUTPUT,
                    "Original OCI fake worker emitted unbounded logs")
            rows=[]
            for line in logs.splitlines():
                try:
                    row=json.loads(line)
                except ValueError:
                    continue
                if row.get("schema")==q.WORKER:
                    rows.append(row)
            require(len(rows)==1 and q.validate_worker(rows[0],case),
                    "Original 50k observed scientific source-free receipt missing")
            report=rows[0]
        g.call(["docker","rm","-f",cid],timeout=15)
        require(not g.process_still_present(root),
                "Exact original worker host PID/starttime survived forced deletion")
        require(g.call(["docker","inspect",cid],success=False).returncode!=0,
                "Exact original fake worker container was not deleted")
        result=outcome(case,reason,final["ExitCode"],last_cpu,True,report)
        result["original_docker_cgroup_OS_limits_verified_before_release"]=True
        result["sampled_cgroup_peak_bytes_at_last_poll"]=last_mem
        return result
    finally:
        g.call(["docker","rm","-f",cid or name],timeout=12,success=False)

def validate_report(report):
    require(type(report) is dict and
            report.get("schema")==SCHEMA and report.get("status")==STATUS and
            report.get("P1_scientific_sha256")==P1_SHA and
            report.get("original_OCI_archive_SHA256")==OCI_SHA and
            report.get("original_OCI_config_ID")==OCI_ID and
            report.get("original_native_with_eight_observer_fake_fullfits")==6 and
            report.get("original_real_P1_fits_executed")==0 and
            report.get("actual_synthetic_CPU_and_wall_stop_probes")==2 and
            report.get("active_supervisor_has_not_validated_real_P1_resource_envelope") is True and
            report.get("CPU_quota_is_one_core_NOT_total_CPU_time_limit") is True and
            report.get("host_CPU_wall_decisions_are_sampled_NOT_kernel_hard_thresholds") is True and
            report.get("actual_final_P2_host_selected") is False and
            report.get("actual_full_native_OS_compiler_SDK_source_closure_completed") is False and
            all(report.get(k)==v for k,v in NO_GO.items()),
            "Original active supervisor report claimed unauthorized scientific resource GO")
    rows=report.get("source_free_actual_five_case_supervisor_results")
    require(type(rows) is list and len(rows)==5 and
            [x.get("case") for x in rows]==list(MODES) and
            all(x.get("real_original_P1_fits_executed")==0 and
                x.get("original_root_PID_starttime_identity_gone") is True and
                x.get("original_docker_cgroup_OS_limits_verified_before_release") is True and
                type(x.get("sampled_cgroup_peak_bytes_at_last_poll")) is int and
                0<x["sampled_cgroup_peak_bytes_at_last_poll"]<=q.p.MAX_MEMORY_BYTES
                for x in rows),
            "One original OCI fake source-free case lacked real OS supervisor proof")
    for row in rows[:3]:
        require(row.get("termination_reason")==
                "NORMAL_NATIVE_8CHECKPOINTS_50K_FAKE_EXIT_0" and
                row.get("synthetic_full_50k_fits_executed")==2 and
                row.get("eight_milestones_verified") is True and
                row.get("final_mutable_state_prediction_parity") is True and
                type(row.get("observed_snapshot_bytes")) is int and
                0<row["observed_snapshot_bytes"]<=q.MAX_SNAPSHOT_BYTES and
                type(row.get("process_peak_RSS_bytes")) is int and
                0<row["process_peak_RSS_bytes"]<=q.p.MAX_MEMORY_BYTES,
                "Actual full FAKE original PCC+checkpoint path missing or too expensive")
    for row,why in zip(rows[3:],
                       ("SAMPLED_CGROUP_CPU_BUDGET_KILL",
                        "SAMPLED_WALL_BUDGET_KILL")):
        require(row.get("termination_reason")==why and
                row.get("synthetic_full_50k_fits_executed")==0 and
                row.get("host_killed_original_container") is True,
                "Actual FAKE resource watchdog negative probe did not fail closed")
    return report

def assemble(results):
    require(type(results) is list and len(results)==len(MODES),
            "Exactly three genuine fake PCC+checkpoint and two forced stop probes")
    report={
        "schema":SCHEMA,"status":STATUS,
        "P1_scientific_sha256":P1_SHA,
        "original_OCI_archive_SHA256":OCI_SHA,
        "original_OCI_config_ID":OCI_ID,
        "original_historically_unchanged_observers_SHA":q.OBSERVER_BLOBS,
        "source_free_actual_five_case_supervisor_results":results,
        "original_native_with_eight_observer_fake_fullfits":6,
        "original_real_P1_fits_executed":0,
        "actual_synthetic_CPU_and_wall_stop_probes":2,
        "active_supervisor_has_not_validated_real_P1_resource_envelope":True,
        "CPU_quota_is_one_core_NOT_total_CPU_time_limit":True,
        "host_CPU_wall_decisions_are_sampled_NOT_kernel_hard_thresholds":True,
        "actual_final_P2_host_selected":False,
        "actual_full_native_OS_compiler_SDK_source_closure_completed":False,
        **NO_GO,
    }
    return validate_report(report)

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--worker",action="store_true")
    ap.add_argument("--backend",choices=q.BACKENDS)
    ap.add_argument("--mode",choices=("normal","cpu_fault","wall_fault"))
    ap.add_argument("--existing-exact-original-OCI",type=Path)
    ap.add_argument("--output",type=Path)
    args=ap.parse_args()
    if args.worker:
        require(args.backend is not None and args.mode is not None and
                args.existing_exact_original_OCI is None and args.output is None,
                "Original fake-only worker accepts no real case inputs")
        original_worker(args.backend,args.mode)
        return
    require(args.backend is None and args.mode is None and
            args.existing_exact_original_OCI is not None and
            args.output is not None and not args.output.exists() and
            platform.system()=="Linux",
            "Host must supply original archived OCI and new source-free report only")
    source_locked()
    loaded=e.verify_original_image(args.existing_exact_original_OCI.resolve())
    require(loaded["loaded_exact_original_image_ID"]==OCI_ID,
            "Original OCI verified identity changed before actual supervisor run")
    rows=[]
    for case in MODES:
        rows.append(one(case))
    report=assemble(rows)
    with args.output.open("xb") as f:
        f.write(json.dumps(report,sort_keys=True,indent=2).encode("ascii")+b"\n")
        f.flush()
        os.fsync(f.fileno())
    require(validate_report(json.loads(args.output.read_text(encoding="utf-8"))),
            "Original synthetic cgroup monitored receipts corrupted")
    print(json.dumps({
        "status":STATUS,
        "review_only_SHA256_NOT_EXECUTION":digest(report),
        "source_free_normal_fake_observer_fullfits":6,
        "tested_FAKE_only_watchdog_CPU_wall_stops":2,
        "P2_execution_preflight_sha256":None,
        "P2_execution_authorized":False,
        "real_PCC_fits_executed":0,
    },sort_keys=True))

if __name__=="__main__":
    try:
        main()
    except BaseException as ex:
        reason=(str(ex)[:130] if isinstance(ex,ResourceBlocked)
                else type(ex).__name__)
        print("P207R_FAIL_CLOSED|"+reason,file=sys.stderr)
        sys.exit(1)
