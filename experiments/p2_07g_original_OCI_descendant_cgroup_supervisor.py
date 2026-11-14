"""P2-07G: Linux host PID/starttime + Docker cgroup descendant cleanup proof.

Fixed three-Python-process stdlib-only probes inside EXACT old OCI image.
NO PCC imports, UCI, graphs, masks, synthetic or real PCC fits, image build,
registry downloads or P2 execution authorization. An observed PID's identity
is (host PID, /proc/PID/stat starttime), not the potentially reused PID alone.
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

from experiments import p2_07e_locked_oci_cgroup_resource_supervisor as e
from experiments import p2_07f_sealed_72case_no_fit_administrative_dryrun as f
from experiments import p2_01i_verify_exact_exported_runtime_image as image

ROOT=Path(__file__).resolve().parents[1]
FIXTURE=ROOT/"experiments/p2_07g_synthetic_descendant_tree_no_pcc.py"
MODES=("normal_tree","detached_tree","stubborn_tree","synthetic_crash")
PYTHON="/opt/hostedtoolcache/Python/3.12.16/x64/bin/python3.12"
P1_SHA=image.P1_SHA
IMAGE_SHA=image.EXPECTED_IMAGE_SHA
IMAGE_ID=image.EXPECTED_IMAGE_ID
F_REPORT_SHA="d6c7fedc5abb3bf48395a3b223bb1112c11ba6a21593075c55e4a02afabd1328"
GENESIS="0"*64

class TreeBlocked(ValueError):
    pass

def require(ok,reason):
    if not ok:
        raise TreeBlocked(reason)

def canonical(row):
    return json.dumps(row,sort_keys=True,separators=(",",":"),
                      ensure_ascii=True,allow_nan=False).encode("ascii")

def sha(row):
    return hashlib.sha256(canonical(row)).hexdigest()

def call(cmd,timeout=12,success=True):
    p=subprocess.run(cmd,stdout=subprocess.PIPE,stderr=subprocess.PIPE,
                     text=True,timeout=timeout)
    require(len(p.stdout)<16384 and len(p.stderr)<16384,
            "Process-output bound exceeded")
    if success:
        require(p.returncode==0,
                "Controlled container command failed: "+str(cmd[:3])+
                ", exit="+str(p.returncode)+" "+p.stderr[-200:])
    return p

def proc_identity(pid):
    require(type(pid) is int and 1<pid<2**31,"Invalid host PID")
    try:
        text=(Path("/proc")/str(pid)/"stat").read_text(encoding="ascii")
    except FileNotFoundError:
        return None
    parts=text.rsplit(") ",1)
    require(len(parts)==2 and text.startswith(str(pid)+" ("),
            "Malformed host /proc stat")
    fields=parts[1].split()
    require(len(fields)>19 and fields[0] in "RSDZTtIWPK",
            "Malformed host /proc PID starttime")
    ticks=int(fields[19])  # Linux stat field #22 (offset 19 from field #3)
    require(ticks>0,"No host process starttime identity")
    return {"pid":pid,"starttime_ticks":ticks,"state":fields[0]}

def process_still_present(original):
    latest=proc_identity(original["pid"])
    return latest is not None and latest["starttime_ticks"]==original["starttime_ticks"]

def parse_top(text):
    rows=[line.split() for line in text.strip().splitlines() if line.strip()]
    require(len(rows)==4 and rows[0][:3]==["PID","PPID","COMMAND"],
            "Exactly three synthetic Python processes required from docker top")
    require(all(len(row)>=3 and row[0].isdigit() and row[1].isdigit()
                and "python" in row[2].lower() for row in rows[1:]),
            "Unexpected non-Python or invalid host process in fixed tree")
    pids=[int(row[0]) for row in rows[1:]]
    require(len(set(pids))==3,"Repeated host process PID")
    return pids

def arguments(mode,name):
    require(mode in MODES and
            re.fullmatch(r"pypcc-p207g-[a-z_]{6,24}-[0-9a-f]{10}",name),
            "Only one of four named synthetic process trees permitted")
    require(FIXTURE.is_file(),"Missing stdlib-only tree fixture")
    return ["docker","run","-d","--name",name,
            "--platform","linux/amd64","--network","none",
            "--read-only","--user","65534:65534",
            "--cap-drop","ALL","--security-opt","no-new-privileges",
            "--pids-limit","128","--memory","2g","--memory-swap","2g",
            "--cpus","1.0",
            "--tmpfs","/tmp:rw,nosuid,nodev,noexec,mode=1777,size=64m",
            "--entrypoint",PYTHON,
            "--mount",f"type=bind,src={FIXTURE.resolve()},dst=/opt/p207g-probe.py,readonly",
            e.IMAGE_TAG,"/opt/p207g-probe.py",mode]

def state(cid):
    data=json.loads(call(["docker","inspect",cid]).stdout)
    require(len(data)==1,"Missing original Docker container")
    return data[0]["State"]

def ready(cid,mode):
    deadline=time.monotonic()+12
    while time.monotonic()<deadline:
        require(state(cid)["Running"] is True,"Synthetic tree root exited before readiness")
        for line in call(["docker","logs",cid],timeout=6).stdout.splitlines():
            try:
                data=json.loads(line)
            except json.JSONDecodeError:
                continue
            if data.get("kind")=="P2_07G_THREE_PROCESS_TREE_READY":
                require(data.get("mode")==mode and
                        data.get("PCC_kernel_fits")==0 and
                        data.get("official_UCI_source_rows")==0 and
                        len({data["root_ns_pid"],data["child_ns_pid"],
                             data["grandchild_ns_pid"]})==3 and
                        data.get("grandchild_has_new_session")==
                        (mode=="detached_tree"),
                        "Missing valid source-free 3-process readiness")
                return
        time.sleep(0.1)
    raise TreeBlocked("Three synthetic processes never became ready")

def observe_three(cid):
    pids=parse_top(call(["docker","top",cid,"-eo","pid,ppid,comm"]).stdout)
    require(state(cid)["Pid"] in pids,
            "Actual Docker root host PID not present in all three")
    observed=[]
    groups=[]
    for pid in pids:
        obj=proc_identity(pid)
        require(obj is not None,"One observed descendant disappeared")
        lines=(Path("/proc")/str(pid)/"cgroup").read_text().splitlines()
        matching=[line for line in lines if line.startswith("0::/")]
        require(len(matching)==1 and cid[:12] in matching[0],
                "Descendant PID escaped original exact Linux cgroup")
        groups.append(matching[0])
        observed.append(obj)
    require(len(set(groups))==1,
            "Root and detached descendants must share one Linux cgroup")
    return observed

def ensure_all_gone(originals):
    require(len(originals)==3,"Three distinct observed host PID identities required")
    deadline=time.monotonic()+8
    while time.monotonic()<deadline:
        if all(not process_still_present(x) for x in originals):
            return True
        time.sleep(0.1)
    raise TreeBlocked("Observed host process PID/starttime survived container cleanup")

def append(path,previous,mode,event,**extra):
    require(mode in MODES and event in (
        "STARTED","THREE_DESCENDANTS_SAME_CGROUP",
        "TERMINATION_REQUESTED","ROOT_EXIT_NONZERO",
        "ALL_ORIGINAL_PROCESS_IDENTITIES_GONE","FAIL_CLOSED"),
        "Unexpected journal operation (never allow a scientific fit here)")
    require(re.fullmatch("[0-9a-f]{64}",previous) is not None,
            "Malformed journal prior SHA")
    row={"mode":mode,"event":event,"previous_sha256":previous,
         "PCC_fits":0,"P1_cases":"ALL_72_NOT_RUN"}
    row.update(extra)
    row["record_sha256"]=sha(row)
    raw=canonical(row)
    require(len(raw)<2048,"Journal contains too much information")
    with Path(path).open("ab") as out:
        out.write(raw+b"\n")
        out.flush()
        os.fsync(out.fileno())
    return row["record_sha256"]

def validate_journal(path,*,completed=False):
    rows=[json.loads(s) for s in Path(path).read_bytes().splitlines()]
    previous=GENESIS
    for row in rows:
        actual=row.get("record_sha256")
        require(row.get("previous_sha256")==previous and
                actual==sha({k:v for k,v in row.items() if k!="record_sha256"}) and
                row.get("mode") in MODES and
                row.get("PCC_fits")==0 and row.get("P1_cases")=="ALL_72_NOT_RUN",
                "Source-free journal chain tampering or false scientific execution")
        previous=actual
    if completed:
        expected=("STARTED","THREE_DESCENDANTS_SAME_CGROUP",
                  "TERMINATION_REQUESTED","ROOT_EXIT_NONZERO",
                  "ALL_ORIGINAL_PROCESS_IDENTITIES_GONE")
        require(len(rows)==20,"Expected 20 immutable journal event lines")
        for i,mode in enumerate(MODES):
            block=rows[i*5:(i+1)*5]
            require([x["mode"] for x in block]==[mode]*5 and
                    tuple(x["event"] for x in block)==expected,
                    "No duplicate/retry/skip or silently continued synthetic mode")
    return {"count":len(rows),"tail_sha256":previous}

def one(mode,path,previous):
    name=f"pypcc-p207g-{mode}-{uuid.uuid4().hex[:10]}"
    cid=None
    originals=[]
    previous=append(path,previous,mode,"STARTED")
    try:
        cid=call(arguments(mode,name),timeout=25).stdout.strip()
        require(re.fullmatch("[0-9a-f]{64}",cid) is not None,
                "Docker did not issue valid exact container ID")
        ready(cid,mode)
        originals=observe_three(cid)
        previous=append(path,previous,mode,"THREE_DESCENDANTS_SAME_CGROUP",
                        host_descendants_observed=3)
        if mode=="synthetic_crash":
            command=["docker","exec","--user","65534:65534",cid,PYTHON,
                     "-c","from pathlib import Path;"
                          "Path('/tmp/p207g-trigger-crash').touch()"]
            reason="ROOT_EXIT_23"
        elif mode=="stubborn_tree":
            command=["docker","stop","--time","1",cid]
            reason="FORCED_STOP_AFTER_IGNORED_SIGTERM"
        else:
            command=["docker","kill",cid]
            reason="KILL_DETACHED_DESCENDANT" if mode=="detached_tree" else "KILL_NORMAL_TREE"
        previous=append(path,previous,mode,"TERMINATION_REQUESTED",reason=reason)
        call(command,timeout=20)
        deadline=time.monotonic()+12
        while time.monotonic()<deadline and state(cid)["Running"]:
            time.sleep(0.10)
        final=state(cid)
        require(final["Running"] is False and final["OOMKilled"] is False
                and final["ExitCode"]!=0,
                "Tree did not exit as failed source-free process group")
        if mode=="synthetic_crash":
            require(final["ExitCode"]==23,"Synthetic root error code was changed")
        previous=append(path,previous,mode,"ROOT_EXIT_NONZERO",
                        exit_code=final["ExitCode"])
        call(["docker","rm","-f",cid],timeout=15)
        ensure_all_gone(originals)
        require(call(["docker","inspect",cid],success=False).returncode!=0,
                "Container survived docker rm")
        previous=append(path,previous,mode,"ALL_ORIGINAL_PROCESS_IDENTITIES_GONE",
                        verified_original_host_PIDs=3)
        return previous,{"mode":mode,
                         "original_processes_observed":3,
                         "root_exit_code":final["ExitCode"],
                         "all_original_host_PID_starttime_identities_gone":True,
                         "all_original_processes_in_same_cgroup":True,
                         "extra_scientific_fits":0}
    except BaseException as exc:
        append(path,previous,mode,"FAIL_CLOSED",error_type=type(exc).__name__)
        raise
    finally:
        call(["docker","rm","-f",cid or name],success=False)

def validate_result(result):
    require(result.get("schema")=="pypcc-P2-07G-exact-OCI-descendant-termination-no-PCC/v1" and
            result.get("status")=="FOUR_SYNTHETIC_CONTAINER_PROCESS_TREES_CLEANED_NO_PCC" and
            result.get("P1_scientific_sha256")==P1_SHA and
            result.get("original_OCI_gzip_sha256")==IMAGE_SHA and
            result.get("original_OCI_image_config_id")==IMAGE_ID and
            result.get("original_P2_07F_admin_report_sha256")==F_REPORT_SHA,
            "Original frozen provenance is missing from descendant proof")
    cases=result.get("cases",[])
    require(len(cases)==4 and [x.get("mode") for x in cases]==list(MODES) and
            all(x.get("original_processes_observed")==3 and
                x.get("all_original_host_PID_starttime_identities_gone") is True and
                x.get("all_original_processes_in_same_cgroup") is True and
                x.get("extra_scientific_fits")==0 and
                type(x.get("root_exit_code")) is int and x["root_exit_code"]!=0
                for x in cases) and cases[-1]["root_exit_code"]==23,
            "Did not prove 4 real cgroup descendant terminations")
    require(result.get("journal_records")==20 and
            re.fullmatch("[0-9a-f]{64}",result.get("journal_tail_sha256","")) is not None,
            "No complete bounded crash-safe append-only attempt journal")
    for key,expected in (
        ("host_PID_reuse_checked_by_proc_starttime",True),
        ("every_original_root_child_grandchild_verified_gone",True),
        ("all_possible_future_process_escapes_proven_absent",False),
        ("final_scientific_runner_present",False),
        ("approved_real_graph_50k_resource_bounds",False),
        ("real_UCI_source_rows_loaded",0),
        ("graphs_or_masks_created",0),
        ("synthetic_PCC_fits",0),
        ("real_PCC_fits_executed",0),
        ("P2_01_final_execution_toolchain_lock_pass",False),
        ("P2_07_full_resources_gate_pass",False),
        ("P2_execution_preflight_sha256",None),
        ("P2_execution_authorized",False),
    ):
        require(result.get(key)==expected,"False scientific authorization or missing safety proof: "+key)
    return True

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--original-oci-artifact",type=Path,required=True)
    ap.add_argument("--original-admin-report",type=Path,required=True)
    ap.add_argument("--journal",type=Path,required=True)
    ap.add_argument("--output",type=Path,required=True)
    arg=ap.parse_args()
    require(platform.system()=="Linux" and
            platform.machine().lower() in ("x86_64","amd64"),
            "Host Linux AMD64 only for real /proc PIDs and cgroup")
    require(not arg.output.exists() and not arg.journal.exists(),
            "Never overwrite prior 72-case gate / 4-tree attempt journal")
    frozen=json.loads(arg.original_admin_report.read_text(encoding="utf-8"))
    require(f.validate_report(frozen) and f.sha(frozen)==F_REPORT_SHA and
            frozen["P2_execution_authorized"] is False and
            frozen["real_PCC_fits_executed"]==0,
            "P2-07G only accepts exact prior administrative P2-07F metadata")
    host=e.verify_original_image(arg.original_oci_artifact.resolve())
    require(host["loaded_exact_original_image_ID"]==IMAGE_ID,
            "Wrong original OCI image imported")
    previous=GENESIS
    cases=[]
    for mode in MODES:
        previous,row=one(mode,arg.journal,previous)
        cases.append(row)
    j=validate_journal(arg.journal,completed=True)
    require(j["tail_sha256"]==previous,"Journal hash end mismatched")
    result={
        "schema":"pypcc-P2-07G-exact-OCI-descendant-termination-no-PCC/v1",
        "status":"FOUR_SYNTHETIC_CONTAINER_PROCESS_TREES_CLEANED_NO_PCC",
        "P1_scientific_sha256":P1_SHA,
        "original_OCI_gzip_sha256":IMAGE_SHA,
        "original_OCI_image_config_id":IMAGE_ID,
        "original_P2_07F_admin_report_sha256":F_REPORT_SHA,
        "cases":cases,
        "journal_records":j["count"],
        "journal_tail_sha256":j["tail_sha256"],
        "host_PID_reuse_checked_by_proc_starttime":True,
        "every_original_root_child_grandchild_verified_gone":True,
        "all_possible_future_process_escapes_proven_absent":False,
        "final_scientific_runner_present":False,
        "approved_real_graph_50k_resource_bounds":False,
        "real_UCI_source_rows_loaded":0,
        "graphs_or_masks_created":0,
        "synthetic_PCC_fits":0,
        "real_PCC_fits_executed":0,
        "P2_01_final_execution_toolchain_lock_pass":False,
        "P2_07_full_resources_gate_pass":False,
        "P2_execution_preflight_sha256":None,
        "P2_execution_authorized":False,
    }
    validate_result(result)
    arg.output.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    print(json.dumps({"status":result["status"],"modes":list(MODES),
                      "original_descendant_PIDs_verified_gone":12,
                      "journal_records":j["count"],
                      "scientific_fits_executed":0,
                      "P2_execution_authorized":False},sort_keys=True))

if __name__=="__main__":
    main()
