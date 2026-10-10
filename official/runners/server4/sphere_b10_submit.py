"""One SH4 resume job after a resource lane; held inspection, cap 2, no old mutation."""
import getpass
import json
import os
import shlex
from pathlib import Path

from official.experiments.prepare import file_sha, write_new
from official.runners.server1.common import read, member, verify
from official.runners.server1 import submit as control
from .sphere_b10_resume import ROOT, CODE, AUTHORITY, validate


def parents(existing):
    jobs=existing['jobs'];tails=control.frontier(jobs)
    if existing['admitted_DAG_width'] > 2:
        result=tails  # Existing over-cap jobs must all drain before our one GPU.
    elif len(tails)<2:
        result=[] if existing['admitted_DAG_width']<2 else tails
    else:
        by_id={r['key']:r for r in jobs}
        def depth(key):return 1+max((depth(p) for p in by_id[key]['parents'] if p in by_id),default=0)
        result=[min(tails,key=lambda k:(depth(k),k))]
    width=control.graph_width(jobs+[dict(key='new-sphere-b10',gpus=1,parents=result)])
    assert width<=max(2,existing['admitted_DAG_width'])
    return result,width


def register():
    root=ROOT;c=read(root/'transfer.json');validate(c)
    assert not (root/'submitted.json').exists(),'REGISTERED_RECONCILE'
    metadata=read(root/'source-metadata.json')
    assert CODE==root/'source'
    assert metadata['main_membership_verified'] and file_sha(root/'official-source.tar')==metadata['archive_sha256']
    for m in metadata['members']:assert file_sha(CODE/m['relative'])==m['sha256']
    ready=read(root/'preparation/READY.json');assert ready['config']==member(root/'transfer.json')
    for m in c['evidence']:verify(m)
    assert file_sha(c['parent_checkpoint']['path'])==c['parent_checkpoint']['sha256']
    st=os.statvfs(root);assert st.f_bavail*st.f_frsize>=c['min_free_bytes']+2*2**30,'STORAGE_ADMISSION'
    budget=read(root/'preparation/storage.json')
    for i in (0,1):
        path=root/'output/checkpoint'/f'slot-{i}.pt';s=path.stat()
        assert s.st_size==budget['slot_capacity'] and s.st_blocks*512>=s.st_size,'CHECKPOINT_SLOT_NOT_RESERVED'
    existing=control.inventory(node='server4')
    assert not any('sphere-resume-b10' in row['original'] for row in existing['jobs']), 'DUPLICATE_ACTIVE_RESUME'
    deps,width=parents(existing)
    resources=dict(control.DEFAULT_RESOURCES,node='server4',qos='lab_gpu_s4',cpus=6,memory_MiB=59392,wall_seconds=48*3600)
    plan=dict(cap=2,resources=resources);physical=control.resource_preflight(plan)
    lock=dict(authority=AUTHORITY,code_commit=metadata['code_commit'],official_tree_sha256=metadata['official_tree_sha256'],
        source_directory=str(CODE),members=metadata['members'],config=member(root/'transfer.json'),parent_checkpoint=c['parent_checkpoint'])
    write_new(root/'execution-lock.json',lock)
    env=dict(PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
        TOKENIZERS_PARALLELISM='false',OMP_NUM_THREADS='6',MKL_NUM_THREADS='6',OPENBLAS_NUM_THREADS='6',
        WANDB_CONSOLE='off',WANDB_SAVE_CODE='false')
    argv=['/data/janghj/EasyEdit/.venv/bin/python','-B','-u','-m','official.runners.server4.sphere_b10_resume','run','--root',str(root)]
    script=root/'scripts/run.sh'
    text='#!/bin/bash\nset -euo pipefail\n'+''.join('export '+k+'='+shlex.quote(v)+'\n' for k,v in env.items())
    text+='cd '+shlex.quote(str(CODE))+'\nexec '+shlex.join(argv)+'\n'
    with script.open('x') as stream:stream.write(text)
    name='s4-qwen25-zsre-sphere-resume-b10'
    cmd=['sbatch','--parsable','--hold','--export=NONE','--no-requeue','--kill-on-invalid-dep=yes',
        '--partition=gpu','--qos=lab_gpu_s4','--nodelist=server4','--nodes=1','--ntasks=1','--cpus-per-task=6',
        '--gres=gpu:rtx_pro_6000:1','--mem=59392M','--time=2-00:00:00','--job-name='+name,
        '--chdir='+str(CODE),'--output='+str(root/'logs/main-%j.out'),'--error='+str(root/'logs/main-%j.err')]
    if deps:cmd.append('--dependency=afterany:'+':'.join(deps))
    cmd.append(str(script))
    write_new(root/'admission.json',dict(existing=existing,dependencies=deps,effective_user_cap=2,combined_width=width,
        resources=resources,physical=physical,storage=budget,available_after_reservation=st.f_bavail*st.f_frsize,old_jobs_changed=0))
    jid=control.command(cmd).split(';')[0];assert jid.isdigit()
    submitted=dict(job_id=jid,name=name,command=cmd,argv=argv,script=member(script),dependencies=deps,
        source=lock['code_commit'],official_tree=lock['official_tree_sha256'],config=member(root/'transfer.json'),
        parent_job='62538',parent_checkpoint=c['parent_checkpoint'],start_batch=10,next_batch=11,output=c['output'])
    write_new(root/'submitted.json',submitted)
    d=control.metadata(control.command(['scontrol','show','job',jid,'--oneliner']))
    assert d['UserId'].split('(')[0]==getpass.getuser() and d['JobState']=='PENDING' and d['Reason']=='JobHeldUser'
    assert d['Command']==str(script) and d['WorkDir']==str(CODE) and d['JobName']==name
    assert d['ReqNodeList']=='server4' and d['QOS']=='lab_gpu_s4' and d['Requeue']=='0' and d['Partition']=='gpu'
    assert control.gpu_count(d['ReqTRES'])==1 and not control.gpu_count(d.get('AllocTRES',''))
    assert control.requested_cpu_matches(d,6) and control.memory_MiB(d['MinMemoryNode'])==59392
    assert control.seconds(d['TimeLimit'])==48*3600 and control.dependencies(d['Dependency'])==sorted(('afterany',p) for p in deps)
    assert control.command(['scontrol','write','batch_script',jid,'-']).strip()==text.strip()
    fresh=control.inventory(node='server4');assert fresh['admitted_DAG_width']<=max(2,existing['admitted_DAG_width'])
    write_new(root/'held-inspection.json',dict(metadata=d,inventory=fresh,lock=member(root/'execution-lock.json'),script=member(script)))
    control.command(['scontrol','release',jid])
    state=control.metadata(control.command(['scontrol','show','job',jid,'--oneliner']))
    write_new(root/'submission.json',dict(submitted,held_inspected=True,released=True,initial=state,GPU_qualification='NOT_RUN_USER_DISABLED'))
    print(json.dumps(dict(job_id=jid,state=state['JobState'],dependencies=deps,cap=2,start_batch=10,next_batch=11)))


if __name__=='__main__':register()
