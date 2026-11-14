"""P2-07E: Docker cgroup-v2 resource kill probes on EXACT locked P2-01I image.

Only Python stdlib probe modes from p2_07e_cgroup_probe_no_pcc.py.
NEVER run a PCC kernel (synthetic or real), pull/build an image, use UCI data,
construct scientific graphs/masks, or obtain P2/P3 execution authorization.
This tests hard container quotas and sampled CPU/wall termination on Linux.
It DOES NOT validate the 72 real original-source graph resource budgets.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import time
import uuid

from experiments import p2_01i_verify_exact_exported_runtime_image as locked
from experiments import p2_07c_consolidated_execution_envelope_no_fit as p207c

ROOT=Path(__file__).resolve().parents[1]
IMAGE_TAG="pypcc-p2-01h-candidate:immutable"
IMAGE_ID=locked.EXPECTED_IMAGE_ID
IMAGE_SHA=locked.EXPECTED_IMAGE_SHA
P1_SHA=locked.P1_SHA
POLICY_SHA="b2b7a969cf6349ec4c315cb2a91b72bf9a82868d4304a0575af246e9d85afc04"
PROBE=ROOT/"experiments/p2_07e_cgroup_probe_no_pcc.py"
PYTHON_BIN="/opt/hostedtoolcache/Python/3.12.16/x64/bin/python3.12"
MODES=("baseline","wall_violation","cpu_violation",
       "memory_violation","scratch_violation")


class ResourceProbeBlocked(ValueError):
    pass


def require(ok,reason):
    if not ok:
        raise ResourceProbeBlocked(reason)


def run_checked(argv, *, seconds=35, check=True):
    p=subprocess.run(argv,text=True,stdout=subprocess.PIPE,
                     stderr=subprocess.PIPE,timeout=seconds)
    if check:
        require(p.returncode==0,
                "Command failed: "+argv[0]+" "+argv[1]+
                "; stderr="+p.stderr[-900:])
    require(len(p.stdout.encode("utf-8"))<=32768 and
            len(p.stderr.encode("utf-8"))<=32768,
            "Docker command output exceeded limited source-free safe budget")
    return p


def sha_file(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as src:
        for data in iter(lambda:src.read(1024*1024),b""):
            h.update(data)
    return h.hexdigest()


def docker_args(mode,name):
    require(mode in MODES and re.fullmatch(r"pypcc-p207e-[a-z_-]{3,32}-[0-9a-f]{10}",name),
            "Only named synthetic cgroup child workload permitted")
    memory="128m" if mode=="memory_violation" else "2g"
    scratch="size=1m" if mode=="scratch_violation" else "size=64m"
    return [
        "docker","run","-d","--name",name,
        "--platform","linux/amd64",
        "--network","none","--read-only",
        "--user","65534:65534",
        "--cap-drop","ALL",
        "--security-opt","no-new-privileges",
        "--pids-limit","128",
        "--memory",memory,"--memory-swap",memory,
        "--cpus","1.0",
        "--tmpfs",f"/tmp:rw,nosuid,nodev,noexec,mode=1777,{scratch}",
        "--env","PYTHONDONTWRITEBYTECODE=1",
        "--env","PYTHONNOUSERSITE=1",
        "--entrypoint",PYTHON_BIN,
        "--mount",f"type=bind,src={PROBE.resolve()},dst=/opt/p2-07e-probe.py,readonly",
        IMAGE_TAG,"/opt/p2-07e-probe.py",mode,
    ]


def verify_original_image(artifact_root):
    require(platform.system()=="Linux" and
            platform.machine().lower() in ("x86_64","amd64"),
            "Linux amd64 cgroup v2 supervisor required")
    lock=locked.lock_checked(json.loads(locked.LOCK_PATH.read_text(encoding="utf-8")))
    require(lock["docker_save_gzip_archive"]["sha256"]==IMAGE_SHA and
            lock["docker_save_gzip_archive"]["original_image_config_id"]==IMAGE_ID,
            "Archive seal must be exact previously approved candidate")
    p1=p207c.read_approved_P1()
    policy=p207c.load(p207c.POLICY)
    p207c.validate_policy(policy,p1)
    require(p207c.sha256(p207c.canonical(policy))==POLICY_SHA and
            policy["approved_execution_host_and_resource_limits"] is False and
            policy["P2_execution_authorized"] is False,
            "Do not silently alter or approve draft resource policy")
    image=locked.only_file(artifact_root,"p2-01h-exact-runtime-image.tar.gz")
    require(sha_file(image)==IMAGE_SHA,"Original tar.gz sha256 mismatch")
    run_checked(["docker","load","--input",str(image)],seconds=90)
    obj=json.loads(run_checked(["docker","image","inspect",IMAGE_TAG]).stdout)
    require(len(obj)==1 and obj[0]["Id"]==IMAGE_ID and
            obj[0]["Os"]=="linux" and obj[0]["Architecture"]=="amd64",
            "Exact preexisting binary image must be loaded offline unchanged")
    require(PROBE.is_file(),"Synthetic stdlib-only probe code missing")
    ram=os.sysconf("SC_PHYS_PAGES")*os.sysconf("SC_PAGE_SIZE")
    disk=os.statvfs(ROOT)
    free=disk.f_bavail*disk.f_frsize
    return {
        "host_physical_ram_bytes_at_probe":ram,
        "host_workspace_free_bytes_at_probe":free,
        "candidate_draft_8GiB_RAM_threshold_met":
            ram>=policy["candidate_guardrails"]["min_host_installed_ram_bytes"],
        "candidate_draft_12GiB_disk_threshold_met":
            free>=policy["candidate_guardrails"]["min_free_disk_before_start_bytes"],
        "loaded_exact_original_image_ID":obj[0]["Id"],
    }


def container_state(cid):
    result=json.loads(run_checked(["docker","inspect",cid],seconds=8).stdout)
    require(len(result)==1,"Exactly one owned synthetic container")
    return result[0]["State"]


def cpu_cgroup_usage_seconds(cid):
    # docker exec itself contributes usage to this cgroup. This is
    # conservatively >= workload CPU, but polling is not a hard CPU cap.
    code=("from pathlib import Path;"
          "p=Path('/sys/fs/cgroup/cpu.stat');"
          "v=dict(x.split() for x in p.read_text().splitlines());"
          "print(v['usage_usec'])")
    p=run_checked(["docker","exec","--user","65534:65534",cid,
                   PYTHON_BIN,"-c",code],seconds=7,check=False)
    if p.returncode!=0:
        return None  # a racing container exit; inspector handles it
    try:
        return int(p.stdout.strip())/1_000_000
    except ValueError:
        raise ResourceProbeBlocked("Malformed actual cgroup v2 cpu.stat")


def forced_termination(cid):
    p=run_checked(["docker","kill",cid],seconds=10,check=False)
    # Expected race: container can exit just before kill.
    require(p.returncode in (0,1),"Could not request Docker child termination")


def no_container_left(cid):
    p=run_checked(["docker","inspect",cid],seconds=8,check=False)
    require(p.returncode!=0,"Probe container leaked after rm -f")


def one_synthetic_case(mode,ledger):
    require(mode in MODES,"Original five fixed source-free probes only")
    name=f"pypcc-p207e-{mode}-{uuid.uuid4().hex[:10]}"
    command=docker_args(mode,name)
    ledger_write(ledger,{"case":mode,"event":"STARTED"})
    created=False
    reason=None
    started=time.monotonic()
    cpu_seen=0.0
    raw_output=""
    final_state=None
    try:
        p=run_checked(command,seconds=25)
        cid=p.stdout.strip()
        require(re.fullmatch("[0-9a-f]{64}",cid) is not None,
                "Docker daemon returned invalid container identity")
        created=True
        while True:
            elapsed=time.monotonic()-started
            require(elapsed<=25,"Uncontrolled synthetic test reached absolute safeguard")
            state=container_state(cid)
            if not state["Running"]:
                break
            if mode=="wall_violation" and elapsed>=1.1:
                reason="SAMPLED_WALL_WATCHDOG_KILL"
                forced_termination(cid)
                continue
            if mode=="cpu_violation":
                cpu=cpu_cgroup_usage_seconds(cid)
                if cpu is not None:
                    cpu_seen=max(cpu_seen,cpu)
                if cpu_seen>=0.65:
                    reason="CGROUP_CPU_STAT_WATCHDOG_KILL"
                    forced_termination(cid)
                    continue
            time.sleep(0.12)
        final_state=container_state(cid)
        logs=run_checked(["docker","logs",cid],seconds=8)
        raw_output=logs.stdout.strip()
        require(len(raw_output.encode("utf-8"))<=8192,
                "Unsafe output size for synthetic-only log")
        row={
            "case":mode,"event":"EXPECTED_RESULT",
            "exit_code":final_state["ExitCode"],
            "kernel_reported_OOMKilled":final_state["OOMKilled"],
            "wall_elapsed_s":round(time.monotonic()-started,3),
            "last_sampled_cgroup_cpu_usage_seconds":round(cpu_seen,6),
            "termination_reason":reason,
        }
        if mode=="baseline":
            data=json.loads(raw_output)
            require(final_state["ExitCode"]==0 and
                    final_state["OOMKilled"] is False and
                    data.get("status")=="CGROUP_V2_BASELINE_PASS" and
                    data.get("real_PCC_fits_executed")==0 and
                    data.get("real_UCI_rows")==0 and
                    data.get("memory_max_bytes")==2*1024**3 and
                    data.get("swap_max_bytes")==0 and
                    data.get("pids_limit")==128 and
                    data.get("quota_microseconds") <=
                    data.get("period_microseconds") and
                    data.get("scratch_available_bytes",0)<=64*1024**2,
                    "Kernel cgroup v2 enforcement baseline missing")
            row["cgroup_v2_linux_hard_limits"]={
                "memory_bytes":data["memory_max_bytes"],
                "swap_bytes":data["swap_max_bytes"],
                "cpu_quota_over_period":data["quota_microseconds"]/
                    data["period_microseconds"],
                "pids":data["pids_limit"],
                "tmpfs_scratch_available_bytes":data["scratch_available_bytes"],
            }
        elif mode=="memory_violation":
            require(final_state["OOMKilled"] is True and
                    final_state["ExitCode"] in (137,9),
                    "Kernel must actually OOM kill the 128MiB synthetic case")
        elif mode=="wall_violation":
            require(reason=="SAMPLED_WALL_WATCHDOG_KILL" and
                    final_state["ExitCode"]!=0 and
                    final_state["OOMKilled"] is False,
                    "Parent failed to terminate sleeping child by elapsed time")
        elif mode=="cpu_violation":
            require(reason=="CGROUP_CPU_STAT_WATCHDOG_KILL" and
                    final_state["ExitCode"]!=0 and
                    final_state["OOMKilled"] is False and
                    cpu_seen>=0.65,
                    "Parent failed to terminate busy child by Linux cgroup CPU usage")
        elif mode=="scratch_violation":
            data=json.loads(raw_output)
            require(final_state["ExitCode"]==42 and
                    data.get("status")=="KERNEL_TMPFS_ENOSPC_CONFIRMED" and
                    data.get("PCC_propagations_executed")==0 and
                    final_state["OOMKilled"] is False,
                    "Kernel tmpfs quota did not halt synthetic scratch overflow")
        ledger_write(ledger,row)
        return row
    except BaseException as e:
        ledger_write(ledger,{
            "case":mode,"event":"FAILED","error_type":type(e).__name__,
            "termination_reason":reason,
        })
        raise
    finally:
        if created:
            run_checked(["docker","rm","-f",name],seconds=10,check=False)
            no_container_left(name)


def ledger_write(path,row):
    require(row.get("case") in MODES and
            row.get("event") in ("STARTED","EXPECTED_RESULT","FAILED"),
            "Unexpected source or mutable fit ID in synthetic-only append ledger")
    data=json.dumps(row,sort_keys=True,separators=(",",":"))
    require(len(data)<4096,"No large logs or source samples in ledger")
    with Path(path).open("a",encoding="utf-8") as f:
        f.write(data+"\n")
        f.flush()
        os.fsync(f.fileno())


def validate_report(report):
    require(report.get("schema")==
            "pypcc-P2-07E-locked-OCI-Linux-cgroup-enforcement-synthetic-probes/v1"
            and report.get("status")==
            "FIVE_STANDALONE_CGROUP_TESTS_PASS_IMPORT_ONLY_OCI_NO_PCC",
            "Missing exact source-free cgroup probe result")
    require(report.get("P1_scientific_sha256")==P1_SHA and
            report.get("original_locked_OCI_gzip_sha256")==IMAGE_SHA and
            report.get("verified_docker_image_config_SHA256")==IMAGE_ID and
            report.get("draft_resource_policy_SHA256")==POLICY_SHA and
            report.get("Linux_cgroup_v2_only") is True and
            report.get("actual_PCC_kernel_fits_executed")==0 and
            report.get("UCI_datasets_loaded")==0 and
            report.get("real_graphs_masks_reconstructed")==0 and
            report.get("scientific_runner_present_in_OCI_image") is False,
            "Original OCI/protocol provenance or scientific boundary changed")
    cases=report.get("cgroup_synthetic_tests",[])
    require(len(cases)==5 and
            [x.get("case") for x in cases]==list(MODES) and
            all(x.get("event")=="EXPECTED_RESULT" for x in cases) and
            cases[0].get("exit_code")==0 and
            cases[0].get("cgroup_v2_linux_hard_limits",{}).get("memory_bytes")==
            2*1024**3 and
            cases[3].get("kernel_reported_OOMKilled") is True and
            cases[4].get("exit_code")==42 and
            cases[1].get("termination_reason")=="SAMPLED_WALL_WATCHDOG_KILL" and
            cases[2].get("termination_reason")=="CGROUP_CPU_STAT_WATCHDOG_KILL",
            "Not all five actual kernel/host synthetic enforcement probes passed")
    require(report.get("per_fit_real_UCI_50k_resource_upper_bound_proven") is False and
            report.get("cgroup_quota_is_not_total_CPU_seconds_cap") is True and
            report.get("polling_is_not_guaranteed_to_catch_subinterval_transients") is True and
            report.get("Linux_resources_policy_approved") is False and
            report.get("P2_01_final_execution_toolchain_lock_pass") is False and
            report.get("P2_07_full_resources_gate_pass") is False and
            report.get("P2_execution_preflight_sha256") is None and
            report.get("P2_execution_authorized") is False and
            report.get("real_PCC_fits_executed")==0,
            "Synthetic cgroup test must never grant P2/real dataset permission")
    return True


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--existing-oci-artifact",type=Path,required=True)
    parser.add_argument("--ledger",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    require(not args.ledger.exists() and not args.output.exists(),
            "Cannot retry/overwrite existing append-only case ledger")
    host=verify_original_image(args.existing_oci_artifact.resolve())
    cases=[one_synthetic_case(mode,args.ledger) for mode in MODES]
    report={
        "schema":"pypcc-P2-07E-locked-OCI-Linux-cgroup-enforcement-synthetic-probes/v1",
        "status":"FIVE_STANDALONE_CGROUP_TESTS_PASS_IMPORT_ONLY_OCI_NO_PCC",
        "P1_scientific_sha256":P1_SHA,
        "original_locked_OCI_gzip_sha256":IMAGE_SHA,
        "verified_docker_image_config_SHA256":IMAGE_ID,
        "draft_resource_policy_SHA256":POLICY_SHA,
        "Linux_cgroup_v2_only":True,
        "host_candidate_capacity_observation_not_final_host":host,
        "cgroup_synthetic_tests":cases,
        "append_only_ledger_line_count":len(args.ledger.read_text().splitlines()),
        "actual_PCC_kernel_fits_executed":0,
        "UCI_datasets_loaded":0,
        "real_graphs_masks_reconstructed":0,
        "scientific_runner_present_in_OCI_image":False,
        "per_fit_real_UCI_50k_resource_upper_bound_proven":False,
        "cgroup_quota_is_not_total_CPU_seconds_cap":True,
        "polling_is_not_guaranteed_to_catch_subinterval_transients":True,
        "Linux_resources_policy_approved":False,
        "P2_01_final_execution_toolchain_lock_pass":False,
        "P2_07_full_resources_gate_pass":False,
        "P2_execution_preflight_sha256":None,
        "P2_execution_authorized":False,
        "real_PCC_fits_executed":0,
    }
    require(report["append_only_ledger_line_count"]==10,
            "Five STARTED and five completed source-free rows required")
    validate_report(report)
    args.output.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n",
                           encoding="utf-8")
    print(json.dumps({
        "status":report["status"],"cases":list(MODES),
        "image_id":IMAGE_ID,"P2_07_full_resources_gate_pass":False,
        "real_PCC_fits_executed":0,
    },sort_keys=True))


if __name__=="__main__":
    main()
