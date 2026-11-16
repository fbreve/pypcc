"""P2-01N — user-selected *candidate* Docker host admission, ZERO PCC fits.

Portable Python stdlib on Windows/Linux. Never downloads, builds, pulls,
publishes, loads, or edits an OCI image. Requires user-selected existing work
directory, already-backed-up EXACT original OCI archive and an ALREADY LOADED
image. Measures local host capacity and real Docker engine limits, then runs
ONLY original P2-01L strict 72-entrypoint IMPORT-ONLY audit in original image.

This is deliberately NOT final host selection or source-reproducible OS /
compiler / SDK / lazy DSO closure, P2-07C policy approval, resource upper
bound, P2-08 technical execution SHA, or permission to run real P1 fits.
"""
import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
import uuid

from experiments import p2_01l_frozen_72case_backend_guard_no_fit as guard
from experiments import p2_07c_consolidated_execution_envelope_no_fit as c
from experiments import p2_07m_exact_oci_original_graph_preparation_no_fit as m

ROOT=m.ROOT
SCHEMA="pypcc-P2-01N-user-selected-candidate-host-capacity-and-native-entrypoints-NO-FIT/v2"
STATUS="CANDIDATE_HOST_CPU_MEMORY_DISK_AND_IMPORT_ONLY_CHECKED_NOT_SELECTED"
WORKER_SCHEMA="pypcc-P2-01N-exact-original-OCI-import-only-cgroup-CPU-RAM-disk/v2"
IMAGE_SHA=guard.IMAGE_SHA
IMAGE_ID=guard.IMAGE_ID
IMAGE_BYTES=290796771
IMAGE_TAG=m.IMAGE_TAG
P1_SHA=guard.P1_SHA
RAM_MIN=8*1024**3
DISK_MIN=12*1024**3
CGROUP_MEM=2*1024**3
TMPFS=128*1024**2
DOCKER_VOLUME_MOUNT="/opt/p201n-local-disk-probe"
NO_GO=guard.NO_GO.copy()

class HostBlocked(ValueError):
    pass

def require(value,reason):
    if not value:
        raise HostBlocked(reason)

def canonical(obj):
    return json.dumps(obj,ensure_ascii=True,sort_keys=True,
                      separators=(",",":"),allow_nan=False).encode("ascii")

def digest(obj):
    return hashlib.sha256(canonical(obj)).hexdigest()

def sha_file(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda:stream.read(1<<20),b""):
            h.update(chunk)
    return h.hexdigest()

def physical_ram_bytes():
    system=platform.system()
    if system=="Linux":
        pages=os.sysconf("SC_PHYS_PAGES")
        size=os.sysconf("SC_PAGE_SIZE")
        return int(pages)*int(size)
    if system=="Windows":
        # GlobalMemoryStatusEx is a physical RAM measurement; it is not
        # Docker Desktop's independent WSL2 VM MemTotal.
        class MemoryStatus(ctypes.Structure):
            _fields_=[("dwLength",ctypes.c_ulong),
                      ("dwMemoryLoad",ctypes.c_ulong),
                      ("ullTotalPhys",ctypes.c_ulonglong),
                      ("ullAvailPhys",ctypes.c_ulonglong),
                      ("ullTotalPageFile",ctypes.c_ulonglong),
                      ("ullAvailPageFile",ctypes.c_ulonglong),
                      ("ullTotalVirtual",ctypes.c_ulonglong),
                      ("ullAvailVirtual",ctypes.c_ulonglong),
                      ("ullAvailExtendedVirtual",ctypes.c_ulonglong)]
        stats=MemoryStatus()
        stats.dwLength=ctypes.sizeof(stats)
        require(bool(ctypes.windll.kernel32.GlobalMemoryStatusEx(
            ctypes.byref(stats))),"PHYSICAL_RAM_QUERY_FAILED")
        return int(stats.ullTotalPhys)
    raise HostBlocked("ONLY_WINDOWS_OR_LINUX_USER_SELECTED_HOST")

def exact_original_archive(path):
    path=Path(path)
    require(path.is_file() and
            path.stat().st_size==IMAGE_BYTES,
            "ORIGINAL_ARCHIVE_MISSING_OR_WRONG_LENGTH")
    require(sha_file(path)==IMAGE_SHA,"ORIGINAL_ARCHIVE_SHA256_MISMATCH")
    return True

def run_cmd(arguments,*,timeout=35):
    try:
        result=subprocess.run(arguments,stdout=subprocess.PIPE,
                              stderr=subprocess.PIPE,text=True,
                              timeout=timeout,check=False)
    except (OSError,subprocess.TimeoutExpired) as exc:
        raise HostBlocked("LOCAL_DOCKER_COMMAND_UNAVAILABLE_OR_TIMED_OUT") from exc
    step=(arguments[1] if len(arguments)>1 and
          arguments[1] in ("info","context","image","run") else "unexpected")
    if result.returncode!=0:
        # Safe step name and return code only. For our own fully controlled
        # worker, extract ONLY its explicitly bounded failure sentinel.
        code=result.returncode
        label=f"LOCAL_DOCKER_{step.upper()}_EXIT_{code}"
        if step=="run":
            match=re.search(r"P201N_FAIL_CLOSED\|([A-Za-z0-9_]{1,100})",
                            result.stderr)
            if match is not None:
                label+="__"+match.group(1)
            safe_stage=re.findall(r"P201N_SAFE_STAGE\|([A-Z0-9_]{1,80})",
                                  result.stderr)
            if safe_stage:
                label+="__STAGE_"+safe_stage[-1]
        raise HostBlocked(label)
    require(len(result.stdout.encode())<=20000 and
            len(result.stderr.encode())<=20000,
            "LOCAL_DOCKER_"+step.upper()+"_UNEXPECTED_OUTPUT_SIZE")
    return result.stdout.strip()

def local_docker_endpoint(endpoint,override=None):
    require(type(endpoint) is str and endpoint.startswith((
        "unix:///","npipe:////./pipe/")),
        "REMOTE_DOCKER_ENGINE_NOT_ELIGIBLE_FOR_LOCAL_HOST_ATTESTATION")
    require(override in (None,"") or
            (type(override) is str and
             override.startswith(("unix:///","npipe:////./pipe/"))),
            "REMOTE_DOCKER_HOST_OVERRIDE_NOT_ALLOWED")
    return True

def validate_docker_info(info,context,override=None):
    require(type(info) is dict and type(context) is list and len(context)==1,
            "LOCAL_DOCKER_INFO_OR_CONTEXT_MISSING")
    endpoint=context[0].get("Endpoints",{}).get("docker",{}).get("Host","")
    local_docker_endpoint(endpoint,override)
    require(info.get("OSType")=="linux" and
            str(info.get("Architecture","")).lower() in ("x86_64","amd64") and
            str(info.get("CgroupVersion"))=="2",
            "DOCKER_ENGINE_NOT_LINUX_AMD64_CGROUP_V2")
    mem=info.get("MemTotal")
    require(type(mem) is int and mem>=RAM_MIN,
            "DOCKER_ENGINE_VM_MEMORY_BELOW_8GIB")
    require(type(info.get("ServerVersion")) is str and
            0<len(info["ServerVersion"])<64,
            "NO_VERIFIED_DOCKER_SERVER_VERSION")
    return {
        "engine_linux_amd64_cgroup_v2":True,
        "engine_VM_total_memory_bytes":mem,
        "engine_version":info["ServerVersion"],
        "daemon_proven_local_nonremote":True,
    }

def verify_image_inspect(inspection):
    require(type(inspection) is list and len(inspection)==1 and
            inspection[0].get("Id")==IMAGE_ID and
            inspection[0].get("Os")=="linux" and
            inspection[0].get("Architecture")=="amd64",
            "ORIGINAL_PRELOADED_OCI_IMAGE_ID_OR_ARCH_WRONG")
    return True

def check_candidate_host_archive_capacity(archive,work_dir,ram,free):
    require(Path(work_dir).is_dir() and type(ram) is int and
            type(free) is int and ram>=RAM_MIN and free>=DISK_MIN,
            "HOST_PHYSICAL_MEMORY_OR_EXISTING_WORKDIR_FREE_DISK_BELOW_DRAFT_MINIMA")
    exact_original_archive(archive)
    return {
        "host_physical_RAM_bytes":ram,
        "work_directory_free_disk_bytes_at_check":free,
        "original_OCI_archive_length_bytes":IMAGE_BYTES,
        "original_OCI_archive_SHA256":IMAGE_SHA,
        "original_OCI_archive_was_preexisting_and_unchanged":True,
    }

def parse_cpu_quota(cpu_max):
    parts=str(cpu_max).split()
    require(len(parts)==2 and
            all(part.isdigit() for part in parts) and
            0<int(parts[0])<=int(parts[1]),
            "DOCKER_WORKER_CPU_QUOTA_NOT_ENFORCED")
    return int(parts[0]),int(parts[1])

def checked_docker_storage_space(mountinfo, root_free, volume_free, tmp_free):
    """Read-only size validation: OCI overlay and Docker local-volume backing.

    This is a point-in-time *capacity observation*, not a guaranteed future
    write allowance, approval of a production evidence export destination,
    proof of host/WSL thin-provisioned physical disk, or OCI source closure.
    """
    require(type(mountinfo) is str and len(mountinfo)<1024**2 and
            type(root_free) is int and type(volume_free) is int and
            type(tmp_free) is int and
            root_free>=DISK_MIN and volume_free>=DISK_MIN and
            0<tmp_free<=TMPFS,
            "DOCKER_OCI_ROOT_OR_LOCAL_VOLUME_FREE_DISK_BELOW_12GIB")
    matches=[]
    for line in mountinfo.splitlines():
        parts=line.split(" - ",1)
        if len(parts)!=2:
            continue
        left,right=parts
        cols=left.split()
        if len(cols)>4 and cols[4]==DOCKER_VOLUME_MOUNT:
            fstype=right.split()[0] if right.split() else ""
            matches.append(fstype)
    require(len(matches)==1 and matches[0] not in
            ("","tmpfs","ramfs","devtmpfs","proc","sysfs","overlay"),
            "DOCKER_STORAGE_PROBE_NOT_AN_ACTUAL_DISK_BACKED_VOLUME_MOUNT")
    return {
        "OCI_root_overlay_free_disk_bytes_at_check":root_free,
        "Docker_local_anonymous_volume_free_disk_bytes_at_check":volume_free,
        "Docker_temporary_tmpfs_free_bytes_at_check":tmp_free,
        "local_anonymous_Docker_volume_mount_verified_non_tmpfs":True,
        "volume_is_temporary_auto_removed_by_docker_run_rm":True,
    }

def worker():
    require(platform.system()=="Linux" and
            platform.python_version()=="3.12.16" and
            os.environ.get("PYTHONPATH")==guard.k.CORRECT_PATH and
            os.environ.get("P201N_IMPORT_ONLY")=="1",
            "ONLY_EXACT_IMMUTABLE_OCI_IMPORT_ONLY_WORKER")
    # Explicit stable stage markers help diagnose errors without printing
    # Python traceback, source lines, local usernames or mount paths.
    print("P201N_SAFE_STAGE|BEFORE_ORIGINAL_72_IMPORTS",file=sys.stderr,flush=True)
    # Executes real original production backend RESOLUTION, not propagation.
    evidence=guard.worker("corrected_72_case")
    guard.validate_worker(evidence,"corrected_72_case")
    print("P201N_SAFE_STAGE|AFTER_ORIGINAL_72_IMPORTS",file=sys.stderr,flush=True)
    root=Path("/sys/fs/cgroup")
    memory=(root/"memory.max").read_text(encoding="ascii").strip()
    swap=(root/"memory.swap.max").read_text(encoding="ascii").strip()
    pids=(root/"pids.max").read_text(encoding="ascii").strip()
    cpu=(root/"cpu.max").read_text(encoding="ascii").strip()
    require(memory==str(CGROUP_MEM) and swap=="0" and
            pids=="128","ACTUAL_ORIGINAL_OCI_MEMORY_SWAP_PIDS_CAP_MISSING")
    numerator,denominator=parse_cpu_quota(cpu)
    fs=os.statvfs("/tmp")
    free=fs.f_bavail*fs.f_frsize
    require(0<free<=TMPFS and
            not Path("/tmp/p207r-release-fake-only").exists(),
            "EXPECTED_NO_FIT_TMPFS_OR_RELEASE_BARRIER_MISSING")
    # The original immutable OCI root is read-only. A *separate anonymous*
    # local Docker volume is mounted WITHOUT writes just to measure current
    # backing filesystem availability on the actual Docker daemon / WSL2 VM.
    # Docker run --rm removes this anonymous volume on exit. No named NAS,
    # host bind, external source or persistent result file is mounted.
    probe=Path(DOCKER_VOLUME_MOUNT)
    require(probe.is_dir(), "DOCKER_ANONYMOUS_STORAGE_PROBE_NOT_MOUNTED")
    def free_bytes(path):
        stat=os.statvfs(path)
        require(stat.f_frsize>0 and stat.f_bavail>=0,
                "DOCKER_STORAGE_FREE_SPACE_STAT_INVALID")
        return int(stat.f_bavail)*int(stat.f_frsize)
    store=checked_docker_storage_space(
        Path("/proc/self/mountinfo").read_text(encoding="utf-8"),
        free_bytes("/"),free_bytes(probe),free_bytes("/tmp"))
    print("P201N_SAFE_STAGE|AFTER_CGROUP_AND_DISK_CAPACITY",file=sys.stderr,flush=True)
    return {
        "schema":WORKER_SCHEMA,
        "original_frozen_P1_SHA256":P1_SHA,
        "actual_Linux_amd64_Python_3_12_16":True,
        "backend_entrypoint_identities_resolved_without_fallback":
            evidence["backend_cases"],
        "resolved_exact_original_P1_case_identifiers":72,
        "original_Cython_native_SHA256":
            guard.k.lock.NEW_NATIVE["production_pcc_step"],
        "original_isolated_Cython_observer_SHA256":
            guard.k.lock.NEW_NATIVE["isolated_pcc_step_live_research"],
        "real_OCI_worker_kernel_memory_limit_bytes":int(memory),
        "real_OCI_worker_swap_limit_bytes":int(swap),
        "real_OCI_worker_pids_limit":int(pids),
        "real_OCI_worker_cgroup_cpu_quota_numerator":numerator,
        "real_OCI_worker_cgroup_cpu_quota_denominator":denominator,
        "real_OCI_worker_tmpfs_free_bytes":int(free),
        **store,
        "real_OCI_worker_native_PCC_propagations_executed":0,
        "real_original_P1_fits_executed":0,
        "P2_execution_authorized":False,
    }

def validate_worker(data):
    require(type(data) is dict and
            data.get("schema")==WORKER_SCHEMA and
            data.get("original_frozen_P1_SHA256")==P1_SHA and
            data.get("actual_Linux_amd64_Python_3_12_16") is True and
            data.get("backend_entrypoint_identities_resolved_without_fallback")==
                {name:24 for name in c.BACKENDS} and
            data.get("resolved_exact_original_P1_case_identifiers")==72 and
            data.get("original_Cython_native_SHA256")==
                guard.k.lock.NEW_NATIVE["production_pcc_step"] and
            data.get("original_isolated_Cython_observer_SHA256")==
                guard.k.lock.NEW_NATIVE["isolated_pcc_step_live_research"] and
            data.get("real_OCI_worker_kernel_memory_limit_bytes")==CGROUP_MEM and
            data.get("real_OCI_worker_swap_limit_bytes")==0 and
            data.get("real_OCI_worker_pids_limit")==128 and
            data.get("real_OCI_worker_native_PCC_propagations_executed")==0 and
            data.get("real_original_P1_fits_executed")==0 and
            data.get("P2_execution_authorized") is False,
            "ORIGINAL_OCI_IMPORT_ONLY_CGROUP_RECEIPT_INVALID")
    n=data.get("real_OCI_worker_cgroup_cpu_quota_numerator")
    d=data.get("real_OCI_worker_cgroup_cpu_quota_denominator")
    require(type(n) is int and type(d) is int and
            n>0 and d>=n and
            type(data.get("real_OCI_worker_tmpfs_free_bytes")) is int and
            0<data["real_OCI_worker_tmpfs_free_bytes"]<=TMPFS and
            type(data.get("OCI_root_overlay_free_disk_bytes_at_check")) is int and
            data["OCI_root_overlay_free_disk_bytes_at_check"]>=DISK_MIN and
            type(data.get("Docker_local_anonymous_volume_free_disk_bytes_at_check")) is int and
            data["Docker_local_anonymous_volume_free_disk_bytes_at_check"]>=DISK_MIN and
            data.get("Docker_temporary_tmpfs_free_bytes_at_check")==
                data["real_OCI_worker_tmpfs_free_bytes"] and
            data.get("local_anonymous_Docker_volume_mount_verified_non_tmpfs") is True and
            data.get("volume_is_temporary_auto_removed_by_docker_run_rm") is True,
            "ORIGINAL_OCI_CPU_TMPFS_OR_DOCKER_LOCAL_DISK_CAPACITY_MISSING")
    return True

def docker_args(name):
    require(re.fullmatch(r"pypcc-p201n-[0-9a-f]{12}",name) is not None,
            "ONLY_FIXED_ORIGINAL_OCI_IMPORT_ONLY_CASE")
    return [
        "docker","run","--rm","--name",name,
        "--platform","linux/amd64","--network","none","--read-only",
        "--user","65534:65534","--cap-drop","ALL",
        "--security-opt","no-new-privileges","--pids-limit","128",
        "--memory","2g","--memory-swap","2g","--cpus","1",
        "--tmpfs","/tmp:rw,nosuid,nodev,noexec,mode=1777,size=128m",
        # Creates one anonymous Docker-local volume for *read-only statvfs*
        # inside the run; Docker --rm removes the anonymous volume.
        "--mount",f"type=volume,destination={DOCKER_VOLUME_MOUNT},volume-nocopy",
        "--env","P201N_IMPORT_ONLY=1",
        "--env",f"PYTHONPATH={guard.k.CORRECT_PATH}",
        "--env","PYTHONDONTWRITEBYTECODE=1",
        "--env","PYTHONNOUSERSITE=1",
        "--env","OPENBLAS_NUM_THREADS=1",
        "--env","OMP_NUM_THREADS=1",
        "--env","MKL_NUM_THREADS=1",
        "--env","NUMEXPR_NUM_THREADS=1",
        # Preserve ORIGINAL P2-01K/L read-only-container Numba cache locator.
        "--env","NUMBA_CACHE_DIR=/tmp/numba-cache",
        "--entrypoint",m.PYTHON_IN_OCI,
        "--mount",f"type=bind,src={ROOT.resolve()},dst=/workspace,readonly",
        IMAGE_TAG,"-m",
        "experiments.p2_01n_candidate_host_readiness_no_fit",
        "--worker",
    ]

def run_import_only():
    name="pypcc-p201n-"+uuid.uuid4().hex[:12]
    try:
        raw=run_cmd(docker_args(name),timeout=105)
        require(len(raw)<6000,
                "ORIGINAL_OCI_IMPORT_ONLY_WORKER_RETURNED_UNEXPECTED_OUTPUT")
        try:
            result=json.loads(raw)
        except ValueError as exc:
            raise HostBlocked("ORIGINAL_OCI_HOST_RETURNED_NONJSON") from exc
        validate_worker(result)
        return result
    finally:
        # A stuck import-only container must NOT survive a failed host check.
        try:
            result=subprocess.run(["docker","rm","-f",name],
                                  stdout=subprocess.PIPE,stderr=subprocess.PIPE,
                                  timeout=15,check=False)
            require(result.returncode in (0,1),
                    "ORIGINAL_OCI_HOST_CONTAINER_CLEANUP_UNVERIFIED")
        except OSError as exc:
            raise HostBlocked("ORIGINAL_OCI_HOST_CONTAINER_CLEANUP_UNAVAILABLE") from exc

def validate_report(report):
    require(type(report) is dict and
            report.get("schema")==SCHEMA and
            report.get("status")==STATUS and
            report.get("P1_scientific_sha256")==P1_SHA and
            report.get("exact_original_OCI_gzip_SHA256")==IMAGE_SHA and
            report.get("original_OCI_config_ID")==IMAGE_ID and
            report.get("candidate_thresholds_P2_07C_not_approved") is True and
            report.get("a_suitable_host_has_not_been_selected_or_approved") is True and
            report.get("source_compiler_SDK_OS_build_closure_proven") is False and
            report.get("real_original_50k_checkpoint_resource_bounds_proven") is False and
            report.get("Docker_engine_root_volume_free_disk_proven") is False and
            report.get("Docker_local_volume_and_OCI_root_disk_capacity_observed") is True and
            report.get("host_identity_paths_serials_tokens_exported") is False and
            all(report.get(key)==val for key,val in NO_GO.items()),
            "HOST_CAPACITY_RECEIPT_FALSE_FINAL_EXECUTION_GO")
    host=report.get("observed_local_host")
    require(type(host) is dict and type(host.get("host_physical_RAM_bytes")) is int and
            host["host_physical_RAM_bytes"]>=RAM_MIN and
            type(host.get("work_directory_free_disk_bytes_at_check")) is int and
            host["work_directory_free_disk_bytes_at_check"]>=DISK_MIN and
            host.get("original_OCI_archive_length_bytes")==IMAGE_BYTES and
            host.get("original_OCI_archive_SHA256")==IMAGE_SHA and
            host.get("original_OCI_archive_was_preexisting_and_unchanged") is True and
            host.get("engine_linux_amd64_cgroup_v2") is True and
            type(host.get("engine_VM_total_memory_bytes")) is int and
            host["engine_VM_total_memory_bytes"]>=RAM_MIN and
            host.get("daemon_proven_local_nonremote") is True,
            "HOST_CANDIDATE_CAPACITY_MISSING")
    require(type(host.get("engine_version")) is str and
            len(host["engine_version"])<64,
            "HOST_DOCKER_ENGINE_VERSION_NOT_RECORDED")
    validate_worker(report.get("exact_original_OCI_import_only_worker"))
    return report

def main():
    arg=argparse.ArgumentParser(description=__doc__)
    arg.add_argument("--worker",action="store_true")
    arg.add_argument("--archive",type=Path)
    arg.add_argument("--work-dir",type=Path)
    arg.add_argument("--output",type=Path)
    args=arg.parse_args()
    if args.worker:
        require(args.archive is None and args.work_dir is None and args.output is None,
                "HOST_WORKER_NEVER_ACCEPTS_DATA_FILES_OR_PATHS")
        print(canonical(worker()).decode("ascii"))
        return
    require(args.archive is not None and args.work_dir is not None and
            args.output is not None and
            platform.system() in ("Windows","Linux"),
            "HOST_MUST_PROVIDE_EXISTING_ARCHIVE_WORKDIR_NEW_RECEIPT")
    require(args.work_dir.is_dir() and args.output.parent.is_dir() and
            not args.output.exists(),
            "HOST_LOCAL_OUTPUT_MUST_BE_FRESH_AND_WORKDIR_ALREADY_EXISTS")
    require(m.verify_source_blobs() and guard.source_bytes_locked(),
            "ORIGINAL_PRODUCTION_SOURCE_SHA_INVALID")
    physical=physical_ram_bytes()
    free=shutil.disk_usage(args.work_dir).free
    capacities=check_candidate_host_archive_capacity(
        args.archive,args.work_dir,physical,free)
    require(not (os.environ.get("DOCKER_HOST","").startswith((
        "tcp://","ssh://","http://","https://"))),
        "REMOTE_DOCKER_HOST_OVERRIDE_NOT_ALLOWED")
    info=json.loads(run_cmd(["docker","info","--format","{{json .}}"]))
    context=json.loads(run_cmd(["docker","context","inspect"]))
    engine=validate_docker_info(info,context,os.environ.get("DOCKER_HOST"))
    original=json.loads(run_cmd(["docker","image","inspect",IMAGE_TAG]))
    verify_image_inspect(original)
    result=run_import_only()
    report={
        "schema":SCHEMA,"status":STATUS,
        "P1_scientific_sha256":P1_SHA,
        "exact_original_OCI_gzip_SHA256":IMAGE_SHA,
        "original_OCI_config_ID":IMAGE_ID,
        "observed_local_host":{**capacities,**engine},
        "exact_original_OCI_import_only_worker":result,
        "candidate_thresholds_P2_07C_not_approved":True,
        "a_suitable_host_has_not_been_selected_or_approved":True,
        "source_compiler_SDK_OS_build_closure_proven":False,
        "real_original_50k_checkpoint_resource_bounds_proven":False,
        # Probe is point-in-time anonymous Docker-local volume plus OCI
        # readonly root. Does NOT prove future image-store capacity or that
        # physical WSL2 VHD/host evidence export volume is suitably provisioned.
        "Docker_engine_root_volume_free_disk_proven":False,
        "Docker_local_volume_and_OCI_root_disk_capacity_observed":True,
        "host_identity_paths_serials_tokens_exported":False,
        **NO_GO,
    }
    validate_report(report)
    # Deliberately no hostname, username, file path, WSL distro serial, MAC
    # or per-process details in this receipt; local only, NEVER uploaded.
    with args.output.open("xb") as stream:
        stream.write(json.dumps(report,sort_keys=True,indent=2).encode("ascii")+b"\n")
        stream.flush()
        os.fsync(stream.fileno())
    print(json.dumps({
        "status":STATUS,
        "original_archive_sha256":IMAGE_SHA,
        "original_native_entrypoint_cases_without_propagation":72,
        "host_physical_RAM_GiB":round(physical/1024**3,2),
        "Docker_engine_VM_RAM_GiB":round(engine["engine_VM_total_memory_bytes"]/1024**3,2),
        "workdir_free_GiB":round(free/1024**3,2),
        "Docker_local_volume_free_GiB":round(result["Docker_local_anonymous_volume_free_disk_bytes_at_check"]/1024**3,2),
        "OCI_root_overlay_free_GiB":round(result["OCI_root_overlay_free_disk_bytes_at_check"]/1024**3,2),
        "local_review_receipt_SHA256_NOT_EXECUTION":digest(report),
        "P2_execution_preflight_sha256":None,
        "P2_execution_authorized":False,
        "real_PCC_fits_executed":0,
    },sort_keys=True))

if __name__=="__main__":
    try:
        main()
    except BaseException as exc:
        reason=str(exc)[:120] if isinstance(exc,HostBlocked) else type(exc).__name__
        print("P201N_FAIL_CLOSED|"+reason,file=sys.stderr)
        sys.exit(1)
