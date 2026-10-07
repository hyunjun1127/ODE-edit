"""Explicit USER six-arm checkpoint retry; immutable failed attempts retained."""
import argparse,copy,getpass,json,os,shutil,unittest
from pathlib import Path
from .common import ROOT,LOCAL,TASK,CELLS,require,member,sha,verify,write,digest,rows_from,validate_rows
from .submit import command
from .replace_pending import boundary

OLD=LOCAL/'easyedit-hparams-20261007'
NEW=LOCAL/'checkpoint-repair-20261007'
AUDIT=LOCAL/'checkpoint-repair-audit-20261007'
SOURCE='ae507064421876af6fbf04231cf43abcb2cb0a81'
NONCE='USER-SH4-GPTJ-CHECKPOINT-REPAIR-20261007'
JOBS=dict(zip((*CELLS,'collector'),map(str,range(60134,60141))))

def check_failure():
    sub=json.loads((OLD/'submission.json').read_text());lock=json.loads((OLD/'execution.lock.json').read_text())
    require(sub['jobs']==JOBS and sub['source']==lock['source_commit']==SOURCE
        and sha(OLD/'config.json')==lock['config_sha256'],'EXACT_FAILED_SOURCE')
    for row in lock['launchers']+[lock['archive']]:verify(row)
    raw=command(['sacct','-X','-n','-P','-j',','.join(JOBS.values()),
        '--format=JobIDRaw,User,JobName%100,State,ExitCode,ElapsedRaw,Start,End,AllocTRES'])
    found={r.split('|')[0]:r.split('|') for r in raw.splitlines() if r.split('|')[0] in JOBS.values()}
    require(set(found)==set(JOBS.values()),'EXACT_PARENT_ACCOUNTING')
    parents=[];artifacts=[]
    for role,job in JOBS.items():
        r=found[job];require(r[1]==getpass.getuser() and r[2]==TASK+'-'+role,'OWNER_TASK')
        tres=dict(t.split('=',1) for t in r[8].split(',') if '=' in t)
        if role=='collector':require(r[3]=='COMPLETED' and int(tres.get('gres/gpu',0))==0,'CPU_TERMINAL')
        else:
            require(r[3]=='FAILED' and r[4]=='1:0' and int(tres['gres/gpu'])==1,'EXACT_FAILED_TERMINAL')
            terminal=json.loads((OLD/role/'terminal.json').read_text());error=json.loads((OLD/role/'first-error.json').read_text())
            require(terminal['job']==job and terminal['source']==SOURCE and terminal['commits']==0
                and terminal['status']=='TECHNICAL_BLOCKED' and not list((OLD/role).glob('batch-*/commit.json')),'NO_SUCCESSFUL_PREFIX')
            require(error['type']=='CheckpointError' and 'torch.Size([7])' in error['error']
                and 'torch.Size([1])' in error['error'],'COMMON_CHECKPOINT_CLOSURE_DEFECT')
            artifacts.extend(member(OLD/role/name) for name in ('terminal.json','first-error.json','runtime.json'))
        elapsed=int(r[5]);parents.append(dict(role=role,job=int(job),state=r[3],exit=r[4],elapsed_seconds=elapsed,
            allocated_GPU_seconds=0 if role=='collector' else elapsed,start=r[6],end=r[7],allocation=r[8]))
    return dict(status='SIX_ARMS_FAILED_CHECKPOINT_ZERO_COMMITS',source=SOURCE,config=member(OLD/'config.json'),
        lock=member(OLD/'execution.lock.json'),submission=member(OLD/'submission.json'),parents=parents,
        allocated_GPU_seconds=sum(r['allocated_GPU_seconds'] for r in parents),old_artifacts=artifacts,
        original_source_raw_config_KEEP=True,Llama_and_60001_mutations=0)

def prepare():
    boundary();AUDIT.mkdir(exist_ok=True)
    require(not NEW.exists() and not (AUDIT/'prepared-config.json').exists(),'CREATE_ONCE_RETRY_PREPARATION')
    failure=check_failure();write(AUDIT/'failure.json',failure)
    c=copy.deepcopy(json.loads((OLD/'config.json').read_text()))
    for row in c['assets']:
        s=Path(row['path']).stat();require((s.st_size,s.st_ino,s.st_mtime_ns)==(row['bytes'],row['inode'],row['mtime_ns']),'EXISTING_ASSET_FRESH_STAT')
    for row in c['runtime']['source_members']+c['native_reference']+c['dependency_sources']:verify(row)
    ready_member=member(OLD/'inputs/ready.json');ready=json.loads(verify(ready_member).read_text())
    for field in ('contexts_member','native_input_alignment','native_full_input_binding','observer_identity'):verify(ready[field])
    identities=json.loads(verify(ready['observer_identity']).read_text())['rows']
    raw=rows_from(OLD/'MEMIT_CAP075/W0',c['models']['MEMIT']['cold_W0_H0'])
    ids=[i for pack in ready['packs'] for i in pack['ids']]
    summary=validate_rows(raw,identities,ids,'W0')
    require(len(ids)==2000 and len(raw)==26000 and summary==json.loads((OLD/'MEMIT_CAP075/W0/summary.json').read_text())['summary'],'EXACT_W0_RAW_REDUCTION')
    reuse=dict(status='QUALIFIED_EXACT_REUSE',cold_state=c['models']['MEMIT']['cold_W0_H0'],
        chunks=[member(p) for p in sorted((OLD/'MEMIT_CAP075/W0').glob('chunk-*.json'))],
        summary=member(OLD/'MEMIT_CAP075/W0/summary.json'),runtime=member(OLD/'MEMIT_CAP075/runtime.json'),
        observation_identity=ready['observation_identity'],source_folder=str(OLD/'MEMIT_CAP075/W0'))
    for mc in c['models'].values():
        require(mc['model_asset_identity']==ready['model_asset_identity'] and c['ordered_ids_sha256']==ready['ordered_ids_sha256'],'NATIVE_MODEL_ORDER_IDENTITY')
        require(digest([mc['model_asset_identity'],c['runtime'],identities,mc['cold_W0_H0'],mc['evaluator_sources']])==ready['observation_identity'],'RECOMPUTED_W0_OBSERVATION_IDENTITY')
        mc['W0_reuse']=copy.deepcopy(reuse)
        for prof in mc['profiles'].values():require(prof['eligible_layers']==list(range(3,9)) and prof['lr']==.5 and prof['lambda_alpha']==10.,'NATIVE_HPARAMS_UNCHANGED')
    c['native_input_reuse']=ready_member
    c['parent_WandB_runs']={cell:json.loads((OLD/cell/'tracking-identity.json').read_text())['run_id'] for cell in CELLS}
    require(len(set(c['parent_WandB_runs'].values()))==6,'SIX_DISTINCT_OLD_RUNS')
    c['attempt']=str(NEW);c['run_instance'].update(attempt=NEW.name)
    c['execution_override']=dict(instruction_id=NONCE,parent_failure=member(AUDIT/'failure.json'),
        old_source=SOURCE,repair='subject lookup closure immutable; parity_pos separate',
        same_hparams_and_method=True,raw_W0_and_input_only_reuse=True,no_checkpoint_resume=True,automatic_retry=False)
    c['authority_members'].append(member(ROOT/'messages/acks/server4/jlz-price-gptj-checkpoint-repair.json'))
    free=shutil.disk_usage(LOCAL).free;stat=os.statvfs(LOCAL)
    require(free>=c['resources']['combined_reserve_bytes'] and stat.f_favail>10000,'RESOURCE_BLOCKED_COMBINED_STORAGE')
    c['resources'].update(free_bytes=free,free_inodes=stat.f_favail)
    from .test_checkpoint import CheckpointBindingTests
    suite=unittest.defaultTestLoader.loadTestsFromTestCase(CheckpointBindingTests)
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    require(result.wasSuccessful() and result.testsRun==4,'MECHANICAL_CHECKPOINT_REGRESSION')
    import torch
    require(not torch.cuda.is_initialized(),'CPU_ONLY_REGRESSION')
    component=dict(status='CHECKPOINT_PRODUCTION_COMPONENT_4_TESTS_PASS',tests=4,scientific_toy=False,
        actual_model_load=False,fit=0,GPU=False,source=[member(ROOT/'project/run_scripts/jlz_price_gptj'/name) for name in ('adapter.py','test_checkpoint.py')],
        source_review='worker gptj_checkpoint_repair; original failure reproduced; same-method fix',
        target_model_GPU_backward='NOT_OBSERVED',checkpoint_disabled_only_in_test_reference=True)
    write(AUDIT/'component-regression.json',component)
    from .preflight import check
    check(AUDIT/'static-source.json');c['cpu_preflight']=member(AUDIT/'static-source.json')
    # Tracking producer change only adds strict-whitelisted retry parent ID.
    review=json.loads(verify(c['tracking']['cpu_review']).read_text())
    for row in review['helper_sources']:verify(row)
    require(review['helper_ready'] and review['integration']=='FAKE_SDK_PAYLOAD_AXES_IDENTITY_PASS','EXISTING_TRACKING_EVIDENCE')
    from project.run_scripts.experiment_tracking import schema
    from .tracking import SCHEMA
    for cell,parent in c['parent_WandB_runs'].items():
        schema.config(dict(server='server4',task_id=TASK,arm=cell,attempt=NEW.name,source_sha='a'*40,
            config_sha='b'*64,model='gptj',model_family='GPTJ',writer='memit' if cell.startswith('MEMIT') else 'alphaedit',
            role='scientific',metric_schema=SCHEMA,parent_run_id=parent))
    write(AUDIT/'prepared-config.json',c)
    write(AUDIT/'reuse.json',dict(native_ready=ready_member,metadata_only_copy_bytes=ready_member['bytes'],
        original_context_and_pack_paths_KEEP=True,raw_copy=False,W0=reuse,
        rows=26000,R=2000,P=4000,N=20000,new_context_generation=False,new_W0_evaluation='0 when actual same runtime guard passes',
        exact_observation_identity=ready['observation_identity'],old_cost_not_counted_as_new=True,
        new_targetmodel_forward_for_review=0,noCP=True,exact_resume='NOT_AVAILABLE'))
    return dict(status='SAME_METHOD_REPAIR_CPU_CHECKED_INPUT_W0_REUSE_BOUND',config=member(AUDIT/'prepared-config.json'),
        old_allocated_GPU_seconds=failure['allocated_GPU_seconds'],new_GPU_qualification='NOT_OBSERVED')

if __name__=='__main__':
    argparse.ArgumentParser(description=__doc__).parse_args();print(json.dumps(prepare()))
