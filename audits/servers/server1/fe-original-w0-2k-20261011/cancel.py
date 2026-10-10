import subprocess,re
from pathlib import Path
from official.runners.server1.common import read,verify,member
from official.experiments.prepare import write_new
O=Path(__file__).parent
def job():
    s=subprocess.check_output(['scontrol','show','job','62530','-o'],text=True)
    return dict(re.findall(r'(\S+?)=(.*?)(?= \S+?=|$)',s.strip()))
s=read('/mnt/raid5/janghj/ODE-edit/local/official-baselines/server1/fe-author-hparams-2k-20261010/registration-r1/submission.json')
j=s['jobs']['zsre'];c=read(verify(j['config']));verify(j['script'])
assert j['job_id']=='62530' and c['method']=='MEMIT_FE_HISTORY' and c['dataset']=='zsre'
b=job()
for k in ['UserId','Command','WorkDir','JobName','ReqNodeList']:assert b[k]==s['initial_snapshot']['zsre'][k]
assert b['JobState']=='RUNNING'
write_new(O/'cancel-before.json',dict(job=b,config=j['config'],source=s['source'],authorization='USER-FE-ORIGINAL-W0-RESET-20261011-R1'))
assert job()['Command']==b['Command']
subprocess.run(['scancel','62530'],check=True)
write_new(O/'cancel-after.json',dict(job=job(),cancel_requested=['62530'],protected=['63125'],other_job_mutations=0))
print('SCANCEL_REQUESTED exact62530; 63125KEEP')
