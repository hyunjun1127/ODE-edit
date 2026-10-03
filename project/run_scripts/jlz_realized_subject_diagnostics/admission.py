"""Fresh server1 resource-only metadata, including admitted pending capacity."""
import argparse
import datetime
import fnmatch
import getpass
import json
import os
import re
import subprocess
from pathlib import Path
from .common import ROOT,NONCE,require,write,member


def run(argv,env=None):
    p=subprocess.run(argv,text=True,capture_output=True,env=env)
    require(p.returncode==0,str(argv)+p.stderr);return p.stdout.strip()


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    caps=Path('/mnt/raid5/janghj/ODE-edit/servers/local/gpu-caps.tsv')
    lines=[x.split('\t') for x in caps.read_text().splitlines() if x.startswith('server1\t')]
    require(len(lines)==1 and lines[0][1]=='devbox','LOCAL_CAP_BOUNDARY')
    local=int(lines[0][2]);tracked=[int(x.split('\t')[1]) for x in (ROOT/'control/gpu-concurrency-policy.tsv').read_text().splitlines() if x.startswith('server1\t')][0]
    cap=min(local,tracked,2);require(98304<=min(int(lines[0][3]),183296),'HOST_MEMORY_CAP')
    patterns=lines[0][4].split(',')
    # Node filter avoids remote-server tasks. Both R/P/held/configuring included.
    queue=run(['squeue','-h','-w','devbox','-o','%i|%u|%j|%T|%b|%N|%R'])
    selected=[];reserved=0;task_existing=0
    for line in queue.splitlines():
        cols=line.split('|');require(len(cols)==7,'QUEUE_SCHEMA')
        jid,user,name,status,tres,node,reason=cols
        if not any(fnmatch.fnmatchcase(name,pat) for pat in patterns):continue
        match=re.search(r'gpu(?::[^,:]+)?:([0-9]+)',tres)
        if 'gpu' in tres and match is None:raise RuntimeError('GPU_RESOURCE_SCHEMA:'+jid)
        ngpu=int(match.group(1)) if match else 0
        selected.append(dict(id=jid,owner=user,name=name,state=status,gpus=ngpu,reason=reason))
        reserved+=ngpu
        if name.startswith('odeedit_jlz_v10_a_b1_diag_s1'):task_existing+=ngpu
    require(task_existing==0,'DUPLICATE_NONCE_TASK_CAPACITY')
    require(reserved+1<=cap,'RESOURCE_PENDING_PROJECT_CAP_FULL')
    env=dict(os.environ,AGENT_GPU_CAPS_FILE=str(caps))
    helper=run([str(ROOT/'scripts/check-slurm-resource-cap.sh'),'server1','1','98304M'],env)
    node=run(['scontrol','show','node','devbox','-o'])
    partition=run(['scontrol','show','partition','gpu','-o'])
    require('State=UP' in partition and 'lab_gpu_s1' in partition,'PARTITION_QOS')
    require('MaxTime=30-00:00:00' in partition or 'MaxTime=UNLIMITED' in partition,'WALL_REVIEW_REQUIRED')
    info=run(['nvidia-smi','--query-gpu=name,memory.total,memory.free','--format=csv,noheader'])
    write(a.out,dict(instruction=NONCE,observed_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        effective_cap=cap,project_GPU_before=reserved,project_GPU_after=reserved+1,task_existing_GPU=task_existing,
        own_server_jobs=selected,resource_only=True,local_cap=member(caps),helper=helper,node=node,partition=partition,
        GPU_metadata=info,host_memory_MiB=98304,GPU=1,CPU=8,wall_hours=6,dependency=None,
        no_other_jobs_changed=True,no_other_task_result_read=True))
    print(json.dumps(dict(admission=str(a.out),project_GPU_before=reserved,project_GPU_after=reserved+1,effective_cap=cap)))


if __name__=='__main__':main()
