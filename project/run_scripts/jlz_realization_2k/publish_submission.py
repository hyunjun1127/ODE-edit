"""One bounded initial/resource handoff receipt. Never submit or monitor loops."""
import datetime
import json
import re
from .common import *
from .submit import command

def main():
    require(not (LOCAL/'initial-handoff.json').exists(),'ALREADY_PAUSED_NO_REPEAT_POLL')
    attempt=LOCAL/'attempt-r2';lock=json.loads((attempt/'execution.lock.json').read_text())
    sub=json.loads((attempt/'submission.json').read_text());held=json.loads(verify(sub['held']).read_text())
    config=json.loads((attempt/'config.json').read_text())
    require(sub['status']=='RELEASED' and len(held['jobs'])==3,'ALL_SCOPE_RELEASED')
    for name,job in sub['jobs'].items():
        release=json.loads((attempt/('released-'+name+'.json')).read_text())
        require(release['job']==job and release['command_succeeded'],'RELEASE_RECEIPT')
    details={k:command(['scontrol','show','job',v,'--oneliner']) for k,v in sub['jobs'].items()}
    node=command(['scontrol','show','node','server4'])
    states={k:re.search(r'\bJobState=([^ ]+)',v)[1] for k,v in details.items()}
    reasons={k:re.search(r'\bReason=([^ ]+)',v)[1] for k,v in details.items()}
    initial=[];startup=[]
    for arm in ('A','B'):
        root=attempt/('main-'+arm)
        if (root/'initial.json').exists():
            r=json.loads((root/'initial.json').read_text())
            require(r['source']==lock['source_commit'] and r['config']==digest(config),'INITIAL_IDENTITY')
            require(all(r[k] for k in ('main_B1_commit','all5_history','observer_restored','main_B2_ownentry')),'INITIAL_STRUCTURE')
            initial.append(member(root/'initial.json'))
        for name in ('runtime.json','initial-state.json','W0-reuse.json','actual-imports.json'):
            if (root/name).exists():startup.append(member(root/name))
    if not initial:
        require(all(states[k] in ('PENDING','RUNNING') for k in ('main-A','main-B')),'TECHNICAL_OR_TERMINAL_RECALL')
        require(any(states[k]=='PENDING' and reasons[k]=='Resources' for k in ('main-A','main-B')),'NOT_RESOURCE_PENDING')
        require('gres/gpu=8' in re.search(r'AllocTRES=([^\n]+)',node)[1],'NODE_GPU_SHORTAGE_NOT_PROVEN')
    snapshot=dict(observed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        instruction_id=INSTRUCTION,job_details=details,node=node,source=lock['source_commit'],
        initial=initial,startup=startup,main_initial='OBSERVED_REPRESENTATIVE' if initial else 'INITIAL_NOT_OBSERVED',
        resource_pause='한 경로 실행시작·다른 경로 Resources 대기; 모든 main held검사/release 완료' if not initial else None,
        monitoring_active=False,automatic_resume=False)
    write(LOCAL/'initial-handoff.json',snapshot)
    core=json.loads(verify(config['qualification_reuse']['receipt']).read_text())
    small=dict(instruction_id=INSTRUCTION,task_id=TASK,status='SUBMITTED_RESOURCE_PENDING' if not initial else 'MAIN_INITIAL',
        source=lock['source_commit'],source_tree=lock['source_tree'],lock=member(attempt/'execution.lock.json'),
        archive=lock['archive'],config=member(attempt/'config.json'),jobs=sub['jobs'],mapping=sub['mapping'],
        states=states,reasons=reasons,observed_utc=snapshot['observed_utc'],initial=snapshot['main_initial'],
        held=member(attempt/'held-inspection.json'),release=[member(attempt/('released-'+name+'.json')) for name in sub['jobs']],
        initial_handoff=member(LOCAL/'initial-handoff.json'),startup=startup,resources=config['resources'],
        cold=True,main_each=2000,total_occurrences=4000,Q1='기존 Q1 identity 재사용',new_Q1=0,new_Q2=0,new_baseline_fits=0,
        qualification_reuse=config['qualification_reuse']['receipt'],prior_Q1_B=core['prior_Q1_B_Slurm'],
        observer_identity=config['observer_identity'],W0_reuse=config['w0_reuse']['receipt'],
        baselines=[dict(name=b['name'],status=b['status'],scope=b.get('comparison_scope')) for b in config['baselines']],
        owner_audit=True,independent_reviewer=False,CPU_tests=22,registration_repair_tests=9,
        scientific_actual_initial=bool(initial),new_GPU_qualification=False,
        GPU_stop=False,monitoring_active=False,automatic_resume=False,other_task_mutation=False,
        no_checkpoint=True,exact_resume='NOT_AVAILABLE',
        broadcast='NO_BROADCAST_NOT_REQUIRED; 원 raw/tensor/prompt/fullstdout local KEEP, 신규 대형전송0',
        access_exception='명시 envelope가 exact runs namespace를 승인; generic helper PASS로 표기하지 않음')
    write(ROOT/'runs/odeedit_jlz_v9_2k_s4_20261004/submission-r2.server4-server-head.json',small)
    write(ROOT/f'audits/servers/server4/{TASK}/input-reuse-manifest.json',dict(
        instruction_id=INSTRUCTION,full_read=json.loads((LOCAL/'preparation-r2/full-read.json').read_text()),
        qualification=core,assets=config['assets'],native_reference=config['native_reference'],
        runtime=config['runtime'],baselines=config['baselines'],W0=config['w0_reuse'],
        packaging='metadata만 게시, 실제 raw/weights/token/prompt 없음'))
    print(json.dumps({k:small[k] for k in ('status','source','jobs','states','reasons','initial','monitoring_active')}))

if __name__=='__main__':main()
