"""Exact pending-only Qwen replacement control; preserve any running transition."""
from pathlib import Path
import json
import subprocess
from datetime import datetime,timezone
from official.experiments.prepare import file_sha,read,write_new

OLD=Path("/mnt/raid5/janghj/ODE-edit/local/qwen-baselines-server2-20261009/registration-r1")
EVAL=Path("/mnt/raid5/janghj/ODE-edit/local/official-baselines/server2/zsre-2k-reeval-20261009/registration-r1")
OUT=Path("/mnt/raid5/janghj/ODE-edit/local/qwen-baselines-server2-20261009/pending-eval-refresh-r1/control")
def now():return datetime.now(timezone.utc).isoformat()
def check(x,c):
    if not x:raise RuntimeError(c)
def cmd(argv):return subprocess.check_output(list(map(str,argv)),text=True,timeout=60).strip()
def state(item,root,kind):
    raw=cmd(["scontrol","show","job",item["job_id"],"-o"])
    f=dict(t.split("=",1) for t in raw.split() if "=" in t)
    script=(item["cell"]+"-"+item["kind"]) if kind=="qwen" else item["method"]
    check(f.get("UserId","").startswith("janghj(") and f.get("ReqNodeList")=="server2"
        and f.get("WorkDir")==str(root) and f.get("Command")==str(root/"scripts"/(script+".sh"))
        and f.get("JobName")==item["name"],"EXACT_OWNER_COMMAND_MISMATCH")
    return dict(at=now(),raw=raw,fields=f)
def run():
    check(not (OUT/"started.json").exists(),"NO_AUTOMATIC_REPEAT")
    check(read(OLD/"source-lock.json")["code_commit"]=="69bfbb2cdffe24072733950c671e47597ff9fbd5","OLD_SOURCE")
    check(read(EVAL/"source-lock.json")["code_commit"]=="ce8d536fa7eb4dfdd38f7024381e68d212f44ea0","EVAL_SOURCE")
    for root in (OLD,EVAL):
        lock=read(root/"input-lock.json")
        for m in lock["members"]:check(file_sha(m["path"])==m["sha256"],"OLD_INPUT_CHANGED")
    qjobs=read(OLD/"released.json")["jobs"];ejobs=read(EVAL/"released.json")["jobs"]
    before={j["job_id"]:state(j,OLD,"qwen") for j in qjobs}
    before_eval={j["job_id"]:state(j,EVAL,"eval") for j in ejobs}
    write_new(OUT/"started.json",dict(at=now(),authority="Direct USER: existing Qwen unstarted PENDING cancel/update/re-register; RUNNING KEEP",
        qwen=before,eval_resource_only=before_eval))
    # Protect downstream eval-only jobs from afterany cancellation release.
    held_eval=[]
    for j in reversed(ejobs):
        s=state(j,EVAL,"eval")
        check(s["fields"]["JobState"]=="PENDING","EVAL_NOT_PENDING_PRESERVE")
        cmd(["scontrol","hold",j["job_id"]])
        held_eval.append(j["job_id"])
        write_new(OUT/"eval-holds"/(j["job_id"]+".json"),dict(before=s,after=state(j,EVAL,"eval")))
    pending=[];kept=[]
    for j in reversed(qjobs):
        s=state(j,OLD,"qwen")
        if s["fields"]["JobState"]!="PENDING":
            kept.append(dict(job=j,state=s,action="KEEP_NOT_PENDING"));continue
        action=subprocess.run(["scontrol","hold",j["job_id"]],capture_output=True,text=True,timeout=30)
        after=state(j,OLD,"qwen")
        if after["fields"]["JobState"]!="PENDING":
            kept.append(dict(job=j,state=after,action="KEEP_TRANSITIONED"));continue
        check(action.returncode==0,"PENDING_HOLD_FAILED")
        pending.append(j)
        write_new(OUT/"qwen-holds"/(j["job_id"]+".json"),dict(before=s,after=after))
    cancelled=[]
    # Reverse registration order is collector, farthest archive/successor, heads last.
    for j in pending:
        s=state(j,OLD,"qwen")
        if s["fields"]["JobState"]!="PENDING":
            kept.append(dict(job=j,state=s,action="KEEP_TRANSITIONED"));continue
        cmd(["scancel","--state=PENDING",j["job_id"]])
        after=state(j,OLD,"qwen")
        check(after["fields"]["JobState"]=="CANCELLED","CANCELLATION_NOT_CONFIRMED")
        cancelled.append(j)
        write_new(OUT/"cancelled"/(j["job_id"]+".json"),dict(job=j,before=s,after=after))
    receipt=dict(at=now(),cancelled=cancelled,kept=kept,eval_temporarily_held=held_eval,
        protected_eval_source_unchanged=True,CP_raw_source_preserved=True)
    write_new(OUT/"cancellation.json",receipt)
    print(json.dumps(dict(cancelled=[j["job_id"] for j in cancelled],
        kept=[j["job"]["job_id"] for j in kept],eval_temporarily_held=held_eval)))
if __name__=="__main__":run()
