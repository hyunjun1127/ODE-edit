"""Direct USER cap3: exact PENDING scheduling only, old scientific bytes intact."""
import argparse
import datetime
import getpass
import json
import os
import re
import shutil
from .repo_native_common import ROOT,LOCAL,member,read,write,verify,require,sha
from .repo_native_submit import resource_inventory,width,command

AUTHORITY=ROOT/'plans/updates/server1/gpu-cap3-20261008/user-override.json'
ATTEMPT=LOCAL/'attempt-native-repo-r1'
OUT=LOCAL/'native-repo-repair-r1/cap3'

def detail(job):
    text=command(['scontrol','show','job',job,'--oneliner'])
    return text,dict(re.findall(r'(?:^|\s)([A-Za-z][A-Za-z0-9_]*)=([^\s]*)',text))

def require_unallocated_pending(value):
    """Slurm's explicit '(null)' allocation is empty, not missing evidence."""
    require(value.get('JobState')=='PENDING' and value.get('RunTime')=='00:00:00'
        and value.get('StartTime') in ('Unknown','N/A','None')
        and 'NodeList' in value and value['NodeList'] in ('','(null)')
        and 'AllocTRES' in value and value['AllocTRES'] in ('','(null)'),
        'CAP3_EXACT_NOT_ALLOCATED_PENDING')

def apply():
    require(not (OUT/'resource-adjustment-started.json').exists(),'CAP3_ONE_DELIBERATE_RESOURCE_PASS')
    authority=read(AUTHORITY);require(authority['server1_combined_project_GPU_cap']==3,'CAP3_AUTHORITY')
    lock=read(ATTEMPT/'execution.lock.json');submission=read(ATTEMPT/'submission.json')
    require(lock['source_commit']=='adb244e6f9c86b54f73bd6d8fb833b338f470ded'
        and sha(ATTEMPT/'config.json')==lock['config_sha256']
        and submission['jobs']['ALPHAEDIT_BLUE']=='61521','CAP3_EXACT_ORIGINAL_ATTEMPT')
    for field in ('source_members','launchers'):
        for row in lock[field]:verify(row)
    rows=(ROOT/'servers/local/gpu-caps.tsv').read_text().splitlines()
    caprow=next(row for row in rows if row.startswith('server1\t')).split('\t')
    require(caprow[1:4]==['devbox','3','183296'],'CAP3_LOCAL_NODE_MEMORY_PRESERVED')
    before=resource_inventory();require(width(before['jobs'])<=3,'CAP3_EXISTING_PROJECT_DAG')
    node=command(['scontrol','show','node','devbox'])
    partition=command(['scontrol','show','partition','gpu'])
    qos=command(['sacctmgr','-n','-P','show','qos','lab_gpu_s1',
        'format=Name,MaxWall,MaxTRESPJ,MaxTRESPU,GrpTRES'])
    require('Gres=gpu' in node and 'PartitionName=gpu' in partition
        and 'State=UP ' in partition and qos.startswith('lab_gpu_s1|'),
        'CAP3_FRESH_NODE_PARTITION_QOS')
    qos_fields=qos.strip().split('|')
    limits=[int(value) for field in qos_fields[2:5]
        for value in re.findall(r'(?:^|,)gres/gpu=(\d+)(?:,|$)',field)]
    require(not limits or min(limits)>=3,'CAP3_STRICTER_QOS_GPU_LIMIT')
    fs=os.statvfs(ATTEMPT);raid=shutil.disk_usage(ATTEMPT);root=shutil.disk_usage('/')
    config=read(ATTEMPT/'config.json')
    require(raid.free>=config['resources']['reserve_bytes'] and fs.f_favail>32
        and root.free>0,'CAP3_DISK_INODE_RESERVE')
    text,d=detail('61521')
    require(d.get('UserId')==getpass.getuser()+'(1025)'
        and d.get('JobName')=='gpt2xl-baselines-native-generation-repair-ALPHAEDIT_BLUE'
        and d.get('Command')==str(ATTEMPT/'ALPHAEDIT_BLUE.sh')
        and d.get('WorkDir')==str(ATTEMPT/'source') and d.get('ReqNodeList')=='devbox',
        'CAP3_EXACT_PENDING_OWNER_SOURCE_COMMAND')
    receipt=dict(instruction_id=authority['instruction_id'],authority=member(AUTHORITY),
        at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        runtime_source=lock['source_commit'],config_sha256=lock['config_sha256'],
        before=before,target_before=text,effective_cap=3,per_job_GPU=1,
        node=node,partition=partition,qos=qos,RAID_available_bytes=raid.free,
        root_available_bytes=root.free,available_inodes=fs.f_favail,
        local_cap=member(ROOT/'servers/local/gpu-caps.tsv'),
        scientific_source_config_changed=False,RUNNING_mutated=False,other_jobs_mutated=False,
        scope='USER_SCOPED_SERVER1_RESOURCE_CAP3; canonical old cap2 historical, other servers unchanged')
    if d['JobState']!='PENDING':
        receipt.update(action='KEEP_ALREADY_STARTED_NO_MUTATION',observed_state=d['JobState'])
    else:
        require_unallocated_pending(d)
        require(d.get('Dependency') in ('afterany:61519(unfulfilled)','afterany:61519'),
            'CAP3_ONLY_ORIGINAL_RESOURCE_DEPENDENCY')
        projected=[dict(row,resource_detail=re.sub(r'\bDependency=[^ ]+',
            'Dependency=(null)',row['resource_detail'])) if row['job']=='61521' else row
            for row in before['jobs']]
        require(width(projected)<=3,'CAP3_PROJECTED_DAG_WIDTH')
        write(OUT/'resource-adjustment-started.json',receipt)
        fresh_text,fresh=detail('61521')
        require_unallocated_pending(fresh)
        require(fresh==d,'CAP3_STATE_SOURCE_NO_RACE')
        response=command(['scontrol','update','JobId=61521','Dependency='])
        after_text,after=detail('61521')
        require(after.get('Dependency')=='(null)' and after['Command']==d['Command']
            and after['WorkDir']==d['WorkDir'],'CAP3_RESOURCE_ONLY_AFTER')
        receipt.update(action='REMOVE_RESOURCE_ONLY_AFTERANY_61519',response=response,
            target_after=after_text,projected_GPU_width=width(projected))
    after=resource_inventory();require(width(after['jobs'])<=3,'CAP3_ACTUAL_ADMISSION_WIDTH')
    receipt['after']=after
    write(OUT/'receipt.json',receipt)
    print(json.dumps(dict(action=receipt['action'],projected_width=receipt.get('projected_GPU_width'),
        current_GPU_jobs=len(after['jobs']),effective_cap=3,source_unchanged=True)))
    return receipt

if __name__=='__main__':apply()
