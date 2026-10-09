"""One explicit eval-only registration pass; no existing-job mutation or retry."""
import argparse
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import pwd
import re
import shlex
import shutil
import subprocess
import tarfile

from official.experiments.prepare import digest,file_sha,read,write_new
from official.runners.server2.zsre_reeval import METHODS,REPO,tracking_config,verify_source,require,member

PYTHON="/mnt/raid5/janghj/EasyEdit/.venv/bin/python"
QWEN=Path("/mnt/raid5/janghj/ODE-edit/local/qwen-baselines-server2-20261009/registration-r1")
ACTIVE={"PENDING","RUNNING","CONFIGURING","COMPLETING","SUSPENDED"}
TERMINAL={"COMPLETED","FAILED","CANCELLED","TIMEOUT","OUT_OF_MEMORY","NODE_FAIL","PREEMPTED","BOOT_FAIL","DEADLINE"}
def now():return datetime.now(timezone.utc).isoformat()
def cmd(args,**kwargs):
    return subprocess.check_output(list(map(str,args)),text=True,timeout=60,**kwargs).strip()
def fields(job):
    raw=cmd(["scontrol","show","job",job,"-o"])
    return raw,dict(x.split("=",1) for x in raw.split() if "=" in x)

def admission():
    prior=read(QWEN/"released.json")
    gpu={j["job_id"]:j for j in prior["jobs"] if j["kind"]=="gpu"}
    require(len(gpu)==12 and all(j["source"]=="69bfbb2cdffe24072733950c671e47597ff9fbd5"
        for j in gpu.values()),"PRIOR_QWEN_SOURCE_SCOPE")
    observed=[]
    for raw in cmd(["scontrol","show","jobs","-o"]).splitlines():
        f=dict(x.split("=",1) for x in raw.split() if "=" in x)
        if (f.get("UserId","").startswith("janghj(")
            and "server2" in (f.get("NodeList"),f.get("ReqNodeList"))
            and f.get("JobState") in ACTIVE and "gres/gpu" in f.get("ReqTRES","")):
            require(f["JobId"] in gpu,"UNKNOWN_OWN_GPU_FRONTIER_"+f["JobId"])
            old=gpu[f["JobId"]]
            require(f.get("Command")==str(QWEN/"scripts"/(old["cell"]+"-gpu.sh"))
                and f.get("WorkDir")==str(QWEN) and f.get("ReqNodeList")=="server2",
                "PRIOR_GPU_OWNER_COMMAND_SCOPE")
            observed.append(f)
    # Prior approved graph is four disjoint resource lanes after its two W0 heads.
    leaves=["61918","61920","61914","61916"]
    require(set(leaves)<=set(gpu),"FRONTIER_MANIFEST")
    children={j:[] for j in gpu}
    for j,item in gpu.items():
        for dep in item["dependencies"]:
            require(dep in gpu,"PRIOR_FOREIGN_GPU_EDGE")
            children[dep].append(j)
    require({j for j in gpu if not children[j]}==set(leaves),"PRIOR_LEAF_FRONTIER_CHANGED")
    leaf_receipts=[];frontier=[]
    for job in leaves:
        raw,f=fields(job);old=gpu[job]
        require(f.get("UserId","").startswith("janghj(") and
                f.get("Command")==str(QWEN/"scripts"/(old["cell"]+"-gpu.sh")),"LEAF_SCOPE")
        require(f.get("JobState") in ACTIVE|TERMINAL,"LEAF_UNKNOWN_STATE")
        frontier.append(job if f["JobState"] in ACTIVE else None)
        leaf_receipts.append(dict(job_id=job,raw=raw))
    # Compare live dependencies with registered AND-afterany DAG, not job-name patterns.
    for f in observed:
        deps=set(re.findall(r"afterany:(\d+)",f.get("Dependency","")))
        require(deps<=set(gpu[f["JobId"]]["dependencies"]),"PRIOR_DEPENDENCY_CHANGED")
    qos=cmd(["sacctmgr","-n","-P","show","qos","lab_gpu_s2","format=Name,MaxTRESPU"])
    require("gres/gpu=4" in qos,"QOS_CAP_CHANGED")
    cap=Path("/mnt/raid5/janghj/ODE-edit/servers/local/gpu-caps.tsv").read_text()
    own=[x.split() for x in cap.splitlines() if x.startswith("server2")]
    require(len(own)==1 and own[0][2]=="4","LOCAL_CAP_CHANGED")
    memory=cmd(["python3",str(REPO/"scripts/slurm_memory_policy.py"),"request",
        "--server","server2","--gpus","1","--mem","59392M","--local-limit-mib-per-gpu","60416"])
    return dict(at=now(),effective_cap=4,
        authority="Latest direct USER server2 cap4; historical canonical TSV2 superseded, no cap mutation",
        owned_active=observed,frontier=frontier,leaf_receipts=leaf_receipts,qos=qos,
        local_cap=cap,memory_check=memory,node=cmd(["scontrol","show","node","server2","-o"]),
        no_existing_job_mutation=True)

def prepare(root,preparation):
    require(not root.exists(),"ATTEMPT_EXISTS_NO_DUPLICATE")
    require(not cmd(["git","status","--porcelain"],cwd=REPO),"SOURCE_DIRTY")
    source=cmd(["git","rev-parse","HEAD"],cwd=REPO)
    require(subprocess.run(["git","merge-base","--is-ancestor",source,"origin/main"],cwd=REPO).returncode==0,
            "SOURCE_NOT_PUBLISHED")
    root.mkdir(parents=True)
    for name in ("logs","scripts","configs","source"):(root/name).mkdir()
    subprocess.run(["git","archive","--format=tar","--output="+str(root/"source.tar"),source,"official"],cwd=REPO,check=True)
    with tarfile.open(root/"source.tar") as tar:
        for entry in tar.getmembers():
            require(not Path(entry.name).is_absolute() and ".." not in Path(entry.name).parts
                and (entry.isfile() or entry.isdir()),"SOURCE_ARCHIVE_SCOPE")
            path=root/"source"/entry.name
            if entry.isdir():path.mkdir(parents=True,exist_ok=True)
            else:
                path.parent.mkdir(parents=True,exist_ok=True)
                with tar.extractfile(entry) as src,path.open("xb") as dst:shutil.copyfileobj(src,dst)
    write_new(root/"source-lock.json",dict(code_commit=source,
        official_git_tree=cmd(["git","rev-parse",source+":official"],cwd=REPO),
        members=[dict(path=str(p.relative_to(root/"source")),sha256=file_sha(p))
                 for p in sorted((root/"source").rglob("*")) if p.is_file()]))
    shutil.copyfile(preparation/"inputs.json",root/"inputs.json")
    inputs=read(root/"inputs.json")
    from official.tracking.schema import config as validate_config
    for row in inputs["rows"]:
        sha=digest(dict(row=row,source=source,proof=inputs["query_proof"],evaluation_batch_size=16))
        cfg=tracking_config(row,inputs,source,sha,"eval-r1-"+row["method"].lower())
        validate_config(cfg)
        write_new(root/"configs"/(row["method"]+".json"),cfg)
    for name in (*METHODS,"collector"):
        cpu="2" if name=="collector" else "6"
        env=dict(PYTHONPATH=str(root/"source"),PYTHONDONTWRITEBYTECODE="1",PYTHONUNBUFFERED="1",
            HOME=pwd.getpwuid(os.getuid()).pw_dir,USER="janghj",HF_HUB_OFFLINE="1",
            TRANSFORMERS_OFFLINE="1",TOKENIZERS_PARALLELISM="false",OMP_NUM_THREADS=cpu,
            MKL_NUM_THREADS=cpu,OPENBLAS_NUM_THREADS=cpu)
        if name=="collector":env["CUDA_VISIBLE_DEVICES"]=""
        args=[PYTHON,"-B","-m","official.runners.server2.zsre_reeval",
            "collect" if name=="collector" else "run","--root",str(root)]
        if name!="collector":args+=["--method",name]
        text="#!/bin/bash\nset -euo pipefail\n"
        text+="\n".join("export "+k+"="+shlex.quote(v) for k,v in env.items())+"\n"
        text+="exec "+shlex.join(args)+"\n"
        (root/"scripts"/(name+".sh")).write_text(text)
    members=[member(root/"inputs.json")]
    members += [member(p) for folder in ("configs","scripts") for p in sorted((root/folder).iterdir())]
    write_new(root/"input-lock.json",dict(members=members))
    require(shutil.disk_usage(root).free>=32*1024**3,"OUTPUT_STORAGE_RESERVE")
    write_new(root/"prepared.json",dict(status="SOURCE_FROZEN_NOT_SUBMITTED",source=source,
        cells=6,qualification="NOT_RUN_USER_DISABLED",checkpoint_writes=0,new_fits=0,W0=0,generation=0))
    print(json.dumps(read(root/"prepared.json")))

def inspect(root,item):
    raw,f=fields(item["job_id"]);gpu=item["method"]!="collector"
    expected=dict(JobId=item["job_id"],JobName=item["name"],JobState="PENDING",Reason="JobHeldUser",
        Requeue="0",Command=str(root/"scripts"/(item["method"]+".sh")),WorkDir=str(root),
        ReqNodeList="server2",QOS="lab_gpu_s2",Partition="gpu",TimeLimit="04:00:00",
        MinMemoryNode="58G" if gpu else "4G",**{"CPUs/Task":"6" if gpu else "2"})
    for k,v in expected.items():require(f.get(k)==v,"HELD_"+k)
    require(f.get("UserId","").startswith("janghj("),"HELD_OWNER")
    require(("gres/gpu=1" in f.get("ReqTRES","")) if gpu else
        "gres/gpu" not in f.get("ReqTRES",""),"HELD_GPU")
    require(set(re.findall(r"(afterany|afterok):(\d+)",f.get("Dependency","")))==
        {("afterany",dep) for dep in item["dependencies"]},"HELD_DEPENDENCY")
    return raw

def register(root,method,deps):
    gpu=method!="collector";name="s2-zsre-reeval-"+method.lower()
    argv=["sbatch","--parsable","--hold","--partition=gpu","--qos=lab_gpu_s2","--nodelist=server2",
        "--nodes=1","--ntasks=1","--cpus-per-task="+("6" if gpu else "2"),
        "--mem="+("59392M" if gpu else "4096M"),"--time=04:00:00","--export=NONE","--no-requeue",
        "--job-name="+name,"--chdir="+str(root),"--output="+str(root/"logs"/(method+"-%j.out")),
        "--error="+str(root/"logs"/(method+"-%j.err"))]
    if gpu:argv+=["--gres=gpu:a6000:1"]
    if deps:argv+=["--dependency=afterany:"+":".join(deps)]
    argv+=[str(root/"scripts"/(method+".sh"))]
    result=subprocess.run(argv,capture_output=True,text=True,timeout=60)
    write_new(root/"sbatch"/(method+".json"),dict(argv=argv,returncode=result.returncode,
        stdout=result.stdout,stderr=result.stderr,at=now()))
    require(result.returncode==0,"SBATCH_FAILED_NO_AUTORETRY")
    job=result.stdout.strip().split(";")[0];require(job.isdigit(),"SBATCH_INVALID_ID")
    item=dict(job_id=job,method=method,name=name,dependencies=deps,argv=argv,
        source=read(root/"source-lock.json")["code_commit"],
        config_sha256=read(root/"configs"/(method+".json"))["config_sha"] if gpu else None)
    write_new(root/"registrations"/(method+".json"),item)
    write_new(root/"inspections"/(method+".json"),dict(raw=inspect(root,item)))
    return item

def submit(root):
    verify_source(root)
    require(not (root/"registration-started.json").exists(),"NO_DUPLICATE_OR_AUTO_RETRY")
    receipt=admission()
    write_new(root/"registration-started.json",receipt)
    lanes=receipt["frontier"].copy();jobs=[]
    for i,method in enumerate(METHODS):
        deps=[lanes[i%4]] if lanes[i%4] else []
        item=register(root,method,deps);jobs.append(item);lanes[i%4]=item["job_id"]
    jobs.append(register(root,"collector",[j["job_id"] for j in jobs]))
    write_new(root/"submission.json",dict(status="ALL_HELD",jobs=jobs))
    verify_source(root)
    for item in jobs:inspect(root,item)
    for item in reversed(jobs):
        cmd(["scontrol","release",item["job_id"]])
        write_new(root/"released"/(item["job_id"]+".json"),dict(job_id=item["job_id"],at=now()))
    snapshot=cmd(["squeue","-j",",".join(j["job_id"] for j in jobs),"-h","-o","%i|%j|%T|%E"])
    value=dict(status="RELEASED",at=now(),jobs=jobs,snapshot=snapshot,
        WAndB="NOT_STARTED_UNTIL_ACTUAL_JOB_INIT",checkpoint_writes=0)
    write_new(root/"released.json",value)
    print(json.dumps(value))

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("command",choices=("prepare","submit"))
    p.add_argument("--root",type=Path,required=True);p.add_argument("--preparation",type=Path)
    args=p.parse_args()
    if args.command=="prepare":prepare(args.root,args.preparation)
    else:submit(args.root)
