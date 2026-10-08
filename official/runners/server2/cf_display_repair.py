"""SH2 CF-only display repair; measured W0 provenance stays with its producer."""
import argparse
import json
from pathlib import Path
import subprocess
import tarfile
from official.experiments.prepare import read, write_new, file_sha, digest
from official.runners.server2 import submit as control
from official.runners.server2 import collect as reducer

INSTRUCTION='USER-SH1-GH-CF-DISPLAY-REPAIR-KEEP-HEALTHY-20261009-R1'
OLD=control.OUTPUT/'registration-no-gpu-qual-r1'
OLD_SOURCE='478464689231595eb63da368cc7c376e775eb559'
ROLES=['CF_ALPHAEDIT','CF_ALPHAEDIT_BLUE','CF_MEMIT_FE','CF_SPHERE']
KEPT={'CF_MEMIT':'61725','ZSRE_FT':'61726','ZSRE_MEMIT':'61728','ZSRE_ALPHAEDIT':'61730',
      'ZSRE_ALPHAEDIT_BLUE':'61732','ZSRE_MEMIT_FE':'61734','ZSRE_SPHERE':'61735'}
COMMON=('evaluation/factual.py','evaluation/reduce.py','evaluation/__init__.py',
        'hparams/cf-stream.lock.json')
require=control.require
member=control.member


def compatibility(old, new):
    for key in ('model','model_id','model_revision','model_snapshot','model_identity',
                'model_assets','tokenizer_sha256','tokenizer_files_sha256','runtime'):
        require(old[key]==new[key], 'W0_COMPATIBILITY_'+key)
    require(old['streams']['cf']['lock']==new['streams']['cf']['lock'], 'W0_ORDERED_STREAM_CHANGED')
    for name in COMMON:
        require(old['source_members'][name]==new['source_members'][name], 'W0_SCORER_SOURCE_CHANGED:'+name)
    return dict(producer_source=old['code_commit'], consumer_source=new['code_commit'],
        model_revision=old['model_revision'], tokenizer_sha256=old['tokenizer_sha256'],
        stream_sha256=old['streams']['cf']['lock']['stream_sha256'],
        runtime_sha256=digest(old['runtime']),
        consumed_source_sha256={name:old['source_members'][name] for name in COMMON},
        source_relabel=False, new_model_forward=0)


def bind_w0(manifest):
    old=read(OLD/'manifest.json')
    require(old['code_commit']==OLD_SOURCE,'EXACT_OLD_W0_PRODUCER')
    binding=dict(manifest=member(OLD/'manifest.json'),ready=member(OLD/'W0_CF/READY.json'),
        compatibility=compatibility(old,manifest))
    binding['factual']=read(binding['ready']['path'])['factual']
    reducer.verify_member(binding['factual'])
    return binding


def reused_w0(manifest, records):
    repair=manifest['cf_display_repair']
    require(repair['instruction_id']==INSTRUCTION and repair['roles']==ROLES,'EXACT_CF_REPAIR_SCOPE')
    binding=repair['W0_binding']
    old=read(reducer.verify_member(binding['manifest']))
    require(compatibility(old,manifest)==binding['compatibility'],'W0_COMPATIBILITY_RECEIPT_CHANGED')
    ready=read(reducer.verify_member(binding['ready']))
    require(ready['factual']==binding['factual'],'W0_ORIGINAL_FACTUAL_MEMBER')
    assets=read(old['asset_manifest'])
    reducer.validate_cold_w0(ready,old,assets,{'cf':records},role='W0_CF')
    return ready


def dependencies(role,jobs):
    if role=='CF_ALPHAEDIT': return []
    if role=='CF_ALPHAEDIT_BLUE': return [KEPT['CF_MEMIT']]
    if role=='CF_MEMIT_FE': return [jobs['CF_ALPHAEDIT']]
    if role=='CF_SPHERE': return [jobs['CF_ALPHAEDIT_BLUE']]
    if role=='collector': return list(jobs.values())+list(KEPT.values())
    raise ValueError('UNKNOWN_CF_REPAIR_ROLE')


def protected_detail():
    detail=control.command(['scontrol','show','job',KEPT['ZSRE_SPHERE'],'-o'])
    require(control.field(detail,'UserId')=='janghj(1025)' and control.field(detail,'ReqNodeList')=='server2'
        and control.field(detail,'JobState')=='PENDING'
        and control.field(detail,'Command')==str(OLD/'ZSRE_SPHERE.sh')
        and control.field(detail,'WorkDir')==str(OLD/'source'), 'PROTECTED_ZSRE_EXACT_PENDING_CONTROL')
    return detail


def submit(manifest_path, attempt):
    base=read(manifest_path)
    source=control.sealed_source(base['code_commit'],base['official_tree_sha256'])
    bound_tracking=control.tracking_binding(base)
    attempt=Path(attempt).resolve()
    require(attempt.parent==control.OUTPUT and not attempt.exists(),'NEW_IMMUTABLE_CF_REPAIR_ATTEMPT')
    manifest=dict(base,base_manifest_sha256=file_sha(manifest_path),registration_stage='cf_display_repair',
        registration_roles=ROLES,collector_mode='cf_display_repair',
        W0_cf_ready_path=str(OLD/'W0_CF/READY.json'))
    manifest['cf_display_repair']=dict(instruction_id=INSTRUCTION,roles=ROLES,kept=KEPT,
        old_manifest=member(OLD/'manifest.json'),W0_binding=bind_w0(manifest),
        healthy_old_source_not_hotpatched=True,historical_online_overwrite=False)
    reused_w0(manifest,read(manifest['streams']['cf']['path']))
    from official.runners.server2.run import tracking_config, configuration
    from official.tracking.schema import config as validate_config
    for role in ROLES:
        validate_config(tracking_config(manifest,configuration(role[3:],'cf'),'chain',attempt/role))
    stopped=read(control.OUTPUT/'cf-display-reconcile-r1/receipt.json')
    require(stopped['source']==OLD_SOURCE and set(stopped['cancelled'])==set(ROLES)|{'collector'},
        'EXACT_PENDING_REPLACEMENT_SCOPE_REQUIRED')
    admission=control.admission(manifest,control.OUTPUT)
    require(admission['cap']==4,'EXACT_FOUR_LANE_REPAIR_ADMISSION')
    require({r['job'] for r in admission['before']['project']}<=set(KEPT.values()),
        'UNEXPECTED_PROTECTED_FRONTIER_REQUIRES_NEW_SCHEDULE')
    for row in admission['before']['project']:
        role=next(r for r,j in KEPT.items() if j==row['job'])
        require(row['command']==str(OLD/(role+'.sh')),'KEPT_SOURCE_COMMAND_MISMATCH')
    pending_before=protected_detail()
    attempt.mkdir();(attempt/'source').mkdir()
    archive=attempt/'source.tar'
    control.command(['git','archive','--format=tar','--output='+str(archive),base['code_commit'],'official'])
    with tarfile.open(archive) as stream:
        require(control.safe_archive_members(stream.getmembers()),'SAFE_OFFICIAL_ARCHIVE')
        stream.extractall(attempt/'source',filter='data')
    write_new(attempt/'manifest.json',manifest)
    for role in [*ROLES,'collector']:
        script=control.launcher(attempt,role,manifest)
        (attempt/(role+'.sh')).write_text(script)
    write_new(attempt/'execution.lock.json',dict(source=source,manifest=member(attempt/'manifest.json'),
        archive=member(archive),launchers=[member(attempt/(r+'.sh')) for r in [*ROLES,'collector']],
        W0_binding=manifest['cf_display_repair']['W0_binding'],resources=manifest['resources'],
        qualification='NOT_RUN_USER_DISABLED',kept=KEPT,admission=admission))
    jobs={};deps={};inspections=[]
    try:
        for role in [*ROLES,'collector']:
            deps[role]=dependencies(role,jobs)
            argv=control.sbatch_argv(attempt,role,manifest,deps[role])
            answer=subprocess.run(argv,capture_output=True,text=True,timeout=45)
            write_new(attempt/('sbatch-'+role+'.json'),dict(argv=argv,returncode=answer.returncode,
                stdout=answer.stdout,stderr=answer.stderr))
            require(answer.returncode==0,'SBATCH_FAILED_NO_AUTORETRY:'+role)
            job=answer.stdout.strip().split(';')[0];require(job.isdigit(),'ACTUAL_JOB_ID')
            jobs[role]=job
            inspections.append(control.inspect_held(job,role,attempt,manifest,argv,deps[role]))
        fresh=control.inventory(exclude=jobs.values())
        require(not fresh['ambiguous'] and {r['job'] for r in fresh['project']}<=set(KEPT.values()),
            'ADMISSION_RACE_REMAIN_HELD')
        before=protected_detail()
        # Only this pending resource edge changes; frozen zsRE argv/source stays exact.
        control.command(['scontrol','update','JobId='+KEPT['ZSRE_SPHERE'],
            'Dependency=afterany:'+jobs['CF_MEMIT_FE']])
        after=protected_detail()
        require(control.typed_dependencies(control.field(after,'Dependency'))==
            control.typed_dependencies([jobs['CF_MEMIT_FE']]),'ZSRE_RESOURCE_EDGE_NOT_BOUND')
        write_new(attempt/'protected-zsre-control.json',dict(job=KEPT['ZSRE_SPHERE'],
            initial=pending_before,before=before,after=after,scientific_source_changed=False))
        write_new(attempt/'held-inspection.json',dict(jobs=inspections,fresh=fresh,
            qualification='NOT_RUN_USER_DISABLED',cap=4,online_PASS=False))
        for role in reversed([*ROLES,'collector']):
            control.command(['scontrol','release',jobs[role]])
            write_new(attempt/('released-'+role+'.json'),dict(job=jobs[role],released=True))
        control.command(['scontrol','release',KEPT['ZSRE_SPHERE']])
    except BaseException as error:
        write_new(attempt/'submission-failure.json',dict(jobs=jobs,error=str(error),automatic_retry=False))
        raise
    result=dict(status='SUBMISSION_HANDOFF',stage='cf_display_repair',source=source,jobs=jobs,
        dependencies=deps,kept=KEPT,cap=4,base_manifest_sha256=file_sha(manifest_path),
        manifest=member(attempt/'manifest.json'),lock=member(attempt/'execution.lock.json'),
        resources=manifest['resources'],qualification='NOT_RUN_USER_DISABLED',
        W_B='REPAIRED_DELIVERY_NOT_YET_OBSERVED_OLD_W0_REJECTION_KEPT',
        initial_snapshot=control.command(['squeue','-h','-j',','.join([*jobs.values(),*KEPT.values()]),
            '-o','%i|%j|%T|%r']),monitoring_active=False)
    write_new(attempt/'submission.json',result)
    return result


def collect(attempt):
    attempt=Path(attempt);manifest=read(attempt/'manifest.json');sub=read(attempt/'submission.json')
    require(not (attempt/'collector/result.json').exists(),'COLLECTOR_ALREADY_RECORDED')
    old=read(reducer.verify_member(manifest['cf_display_repair']['old_manifest']))
    sources={r:(attempt,manifest) for r in ROLES}
    sources.update({r:(OLD,old) for r in KEPT})
    summaries={};failures={};logging={}
    for role,(root,bound) in sources.items():
        path=root/role/'logging-transport.json'
        logging[role]=read(path) if path.is_file() else {'status':'NOT_RECORDED'}
        try:
            dataset,method=role.split('_',1);dataset=dataset.lower()
            records=read(bound['streams'][dataset]['path'])
            summaries[role]=reducer.validate_chain(read(root/role/'result.json'),bound,
                read(bound['asset_manifest']),records,method,dataset)
        except Exception as error:
            failures[role]=dict(type=type(error).__name__,code=str(error)[:300])
    accounting={}
    for label,jobs,stage in [('replacement',sub['jobs'],'cf_display_repair'),('kept',KEPT,'no_gpu_qual')]:
        try:
            accounting[label]=reducer.accounting(jobs,stage);accounting[label].pop('raw',None)
        except Exception as error:
            accounting[label]=dict(status='ACCOUNTING_NOT_RECORDED',error=str(error)[:300])
    complete=not failures and all(set(v.get('jobs',{}))==set(sub['jobs'] if k=='replacement' else KEPT)
        and all(r['state']=='COMPLETED' for role,r in v['jobs'].items() if role!='collector')
        for k,v in accounting.items())
    result=dict(status='SCIENTIFIC_COMPLETE' if complete else 'PARTIAL_OR_FAILED',
        scientific_complete=complete,summaries=summaries,failures=failures,accounting=accounting,
        logging=logging, SDK_accepted_is_not_remote_ACK=True,
        qualification='NOT_RUN_USER_DISABLED',kept_completed_CF_FT='61650',
        old_W0_logging_rejected_preserved=True,model_loads=0,GPU=0,
        historical_provenance_preserved=True,automatic_retry=False)
    write_new(attempt/'collector/result.json',result)
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--manifest');p.add_argument('--attempt',required=True)
    p.add_argument('--collect',action='store_true');args=p.parse_args()
    print(json.dumps(collect(args.attempt) if args.collect else submit(args.manifest,args.attempt)))
if __name__=='__main__':main()
