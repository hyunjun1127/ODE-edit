"""Explicit one-shot stage admission; no poller, no unrelated job actions."""
import argparse
import json
import os
from pathlib import Path
import subprocess
from .transfer import DEST
from project.run_scripts.single_layer_cumulative_risk.records import save

REPO=Path(__file__).resolve().parents[3]
def valid(root,arms):
    x=json.loads((root/'terminal.json').read_text())
    assert x['status']=='TERMINAL_VALID' and x['W0_restore'] and x['arms']==arms
    for arm in arms:
        t=json.loads((root/arm/'terminal.json').read_text());assert t['status']=='TERMINAL_VALID' and t['steps']==8
        assert len(json.loads((root/arm/'eval-008.json').read_text())['rows'])==3900

def main():
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['entries','auxiliary']);a=ap.parse_args()
    scientific=['H','R','EP'];valid(DEST/'output/Middle-main',scientific)
    if a.stage=='entries':plan=[('Early',scientific,'Early-main',None),('Late',scientific,'Late-main',None)]
    else:
        for e in ('Early','Late'):valid(DEST/f'output/{e}-main',scientific)
        plan=[('Middle',['EP-Free','EP-J4','EP-N16'],'Middle-auxiliary',DEST/'output/Middle-main/common.pt')]
    head=json.loads((DEST/'execution.lock.json').read_text())['source_head']
    assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()==head
    env=dict(os.environ,AGENT_GPU_CAPS_FILE='/mnt/raid5/janghj/ODE-edit/servers/local/gpu-caps.tsv')
    subprocess.run(['bash','scripts/check-slurm-resource-cap.sh','server2',str(len(plan)),str(60416*len(plan))+'M'],cwd=REPO,env=env,check=True)
    assert not (DEST/f'submission-{a.stage}.json').exists()
    jobs=[]
    for entry,arms,name,common in plan:
        output=DEST/'output'/name;assert not output.exists()
        command=['sbatch','--hold','--parsable','--output='+str(DEST/f'logs/{name}-%j.out'),
                 '--error='+str(DEST/f'logs/{name}-%j.err'),'project/run_scripts/blue_l4_progress_barrier/run.sbatch',str(REPO),head,
                 '--entry',entry,'--arms',*arms,'--output',str(output)]
        if common:command+=['--common',str(common)]
        job=subprocess.check_output(command,cwd=REPO,text=True).strip()
        inspection=subprocess.check_output(['scontrol','show','job',job,'-o'],text=True)
        assert 'JobState=PENDING' in inspection and 'Reason=JobHeldUser' in inspection
        assert 'MinMemoryNode=59G' in inspection and 'ReqNodeList=server2' in inspection and 'TresPerNode=gres/gpu:1' in inspection
        assert str(REPO) in inspection and str(output) in inspection and head in inspection
        jobs.append(dict(job=job,entry=entry,arms=arms,command=command,held_inspection=inspection))
    save(DEST/f'submission-{a.stage}.json',dict(stage=a.stage,source_head=head,jobs=jobs,other_job_actions=0))
    # Jobs remain held if a new admission appears in the intervening window.
    subprocess.run(['bash','scripts/check-slurm-resource-cap.sh','server2',str(len(plan)),str(60416*len(plan))+'M'],cwd=REPO,env=env,check=True)
    for row in jobs:subprocess.run(['scontrol','release',row['job']],check=True)
    save(DEST/f'release-{a.stage}.json',dict(status='RELEASED',jobs=[r['job'] for r in jobs],cap=2))
    print(json.dumps(dict(status='SUBMITTED_RELEASED',jobs=jobs)))

if __name__=='__main__':main()
