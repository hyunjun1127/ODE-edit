"""One-shot binding after Slurm afterok; no polling, sleeps, or submission.

The admission lock seals the producer/source/expected path, not nonexistent
teacher bytes. A new effective lock binds the completed READY before model load.
Neither original lock nor running preparation source is modified.
"""
import argparse
import copy
import json
from pathlib import Path
import re
import shlex
import shutil
from .preparation import ROOT, create_json, member, sha, storage_plan
from .model import require_lock
from project.run_scripts.single_layer_edit_preserving_correction.common import digest


def checked(item):
    p=Path(item['path'])
    if p.stat().st_size!=item['bytes'] or sha(p)!=item['sha256']:
        raise ValueError('DEPENDENT_INPUT_CHANGED:'+str(p))
    return json.loads(p.read_text())


def pending_spec(path):
    path=Path(path).resolve()
    if not path.is_relative_to(ROOT/'PREP') or path.name!='execution.lock.json':
        raise ValueError('EXACT_TASK_PREPARATION_LOCK_ONLY')
    prep=json.loads(path.read_text());require_lock(prep,historical_preparation=True)
    if prep['stage']!='GENERATED_REFERENCE_PREPARATION':raise ValueError('PREPARATION_STAGE')
    receipt=member(path.parent/'submission.json');submitted=checked(receipt)
    if (submitted['lock']['sha256']!=sha(path) or submitted.get('released') is not True or
        not re.fullmatch(r'[0-9]+',submitted['job'])):
        raise ValueError('SEALED_RELEASED_PREPARATION_SUBMISSION')
    return dict(job=submitted['job'],lock=member(path),submission=receipt,
        expected_ready=str(Path(prep['output'])/'READY.json'),
        preparation_source=prep['execution']['commit'],
        preparation_archive_sha256=prep['execution']['archive']['sha256'],
        ready_state_at_admission='NOT_REQUIRED_NOT_CLAIMED',
        failure_policy='AFTEROK_PREVENTS_START; MISSING_OR_INVALID_READY_FAILS_BEFORE_MODEL',
        no_watcher=True,no_automatic_resubmission=True)


def verify_pending_members(spec):
    prep=checked(spec['lock']);require_lock(prep,historical_preparation=True)
    submitted=checked(spec['submission'])
    if (prep['stage']!='GENERATED_REFERENCE_PREPARATION' or submitted['job']!=spec['job'] or
        submitted['lock']['sha256']!=spec['lock']['sha256'] or submitted['released'] is not True or
        spec['expected_ready']!=str(Path(prep['output'])/'READY.json') or
        spec['preparation_source']!=prep['execution']['commit'] or
        spec['preparation_archive_sha256']!=prep['execution']['archive']['sha256']):
        raise ValueError('PREPARATION_PRODUCER_IDENTITY')
    return prep


def compare_preparation_inputs(main,prep):
    keys=('snapshot','config4','blue_root','dataset_root','projector','cold_capsule',
          'reference_inputs','model_revision','records_digest','sample_order','torch','transformers',
          'model_config_sha256','model_weights_identity_sha256','tokenizer_identity_sha256','seed')
    for key in keys:
        if main[key]!=prep[key]:raise ValueError('PREPARATION_MAIN_INPUT_MISMATCH:'+key)


def inspect_preparation_job(text,spec):
    fields=dict(re.findall(r'(?:^|\s)([A-Za-z][A-Za-z0-9_]*)=([^\s]+)',text))
    required=dict(JobId=spec['job'],JobName='odeedit_en_reuse_g256_prep_s4',NumCPUs='8',
        Requeue='0',ReqNodeList='server4',TresPerNode='gres/gpu:rtx_pro_6000:1')
    if (any(fields.get(k)!=v for k,v in required.items()) or
        not re.fullmatch(r'janghj\([0-9]+\)',fields.get('UserId','')) or
        fields.get('JobState') not in ('RUNNING','PENDING','COMPLETED')):
        raise ValueError('PREPARATION_EXACT_JOB_STATE_RESOURCE_OWNER')
    tres=dict(item.split('=',1) for item in fields.get('ReqTRES','').split(',') if '=' in item)
    if tres.get('mem') not in ('59G','60416M') or tres.get('gres/gpu')!='1' or tres.get('cpu')!='8':
        raise ValueError('PREPARATION_RESOURCE')
    prep=verify_pending_members(spec)
    command=[str(Path(prep['execution']['source_root'])/'project/run_scripts/en_execution_reuse/run.sbatch'),
             prep['execution']['source_root'],spec['lock']['path'],'GENERATED_REFERENCE_PREPARATION']
    c=re.search(r'^\s*Command=(.*?)\s*$',text,re.M)
    s=re.search(r'^\s*SubmitLine=(.*?)\s*$',text,re.M)
    if c is None or s is None or shlex.split(c.group(1))!=command[:1] or shlex.split(s.group(1))[-4:]!=command:
        raise ValueError('PREPARATION_EXACT_SOURCE_LOCK_ARGS')


def resolve(lock):
    require_lock(lock)
    if lock['stage']!='MATCHED_B1':raise ValueError('B1_ONLY')
    if 'generated_ready_pending' not in lock:return lock,None
    if 'generated_ready' in lock:raise ValueError('AMBIGUOUS_READY_BINDING')
    prep=verify_pending_members(lock['generated_ready_pending'])
    compare_preparation_inputs(lock,prep)
    root=Path(prep['output'])
    if (root/'failure.json').exists():raise ValueError('PREPARATION_FAILURE_PRESENT')
    ready_member=member(root/'READY.json');ready=checked(ready_member)
    if (ready.get('status')!='GENERATED_REFERENCE_READY_NOT_CORRECTION_VALIDATION' or
        ready.get('generated_documents')!=640 or ready.get('source')!=prep['execution'] or
        ready.get('B2_authorized') is not False or ready.get('automatic_continuation') is not False):
        raise ValueError('READY_STATUS_SOURCE_COMPLETENESS')
    if (Path(ready['binding']['path'])!=root/'teacher-binding.json' or
        Path(ready['manifest']['path'])!=root/'generated/manifest.json'):
        raise ValueError('READY_EXACT_PRODUCER_PATHS')
    binding=checked(ready['binding']);manifest=checked(ready['manifest'])
    if (binding['source_sha256']!=prep['execution']['archive']['sha256'] or
        binding['runtime']['source']!=prep['execution']['commit'] or
        manifest.get('status')!='COMPLETE' or manifest.get('production_ready') is not True or
        manifest.get('document_counts')!={'R512':512,'Dev128':128} or
        manifest.get('upstream_cache_status')!='COMPLETE' or manifest.get('binding')!=binding or
        manifest.get('inputs_sha256')!=lock['reference_inputs']['sha256']):
        raise ValueError('READY_MANIFEST_IDENTITY_OR_COVERAGE')
    # The runner loads the full bank but skips payload numerical/hash audits
    # under the 2026-09-19 user override. Producer/scope binding is unchanged.
    plan=storage_plan([d['logp']['shape'][0] for d in manifest['documents']],current_valid_token_upper=10416)
    incremental=plan['estimated_bytes_without_unrecorded_overhead']-plan['payload_teacher_key_bytes']+8*2**30
    free=shutil.disk_usage(ROOT).free
    if free<incremental:raise ValueError('B1_INCREMENTAL_STORAGE_NO_WAIVER')
    effective=copy.deepcopy(lock)
    effective.pop('generated_ready_pending');effective.pop('lock_identity')
    effective.update(generated_ready=ready_member,admission_lock_identity=lock['lock_identity'],
        preparation_dependency_resolved=dict(job=lock['generated_ready_pending']['job'],
            source=prep['execution']['commit'],validation='READY_PRODUCER_AND_COMPLETE_MANIFEST_BEFORE_MODEL',
            actual_correction_validation='NOT_RUN',free_bytes=free,required_incremental_bytes=incremental))
    effective['lock_identity']=digest(effective);require_lock(effective)
    return effective,dict(ready=ready_member,manifest=ready['manifest'],binding=ready['binding'],
        admission_lock_identity=lock['lock_identity'],effective_lock_identity=effective['lock_identity'],
        max_batches=1,sequential_authorized=False,monitoring_process_created=False)


def main():
    p=argparse.ArgumentParser();p.add_argument('--lock',type=Path,required=True)
    p.add_argument('--max-batches',type=int,required=True);args=p.parse_args()
    if args.max_batches!=1:raise ValueError('B2_UNAUTHORIZED')
    lock=json.loads(args.lock.read_text());require_lock(lock)
    try:
        for item in lock['execution']['members']+lock['external_members']:
            if Path(item['path']).stat().st_size!=item['bytes'] or sha(item['path'])!=item['sha256']:
                raise ValueError('PREMODEL_SOURCE_ASSET_DRIFT:'+item['path'])
        effective,receipt=resolve(lock)
        if receipt is not None:
            create_json(args.lock.parent/'execution.ready.lock.json',effective)
            create_json(args.lock.parent/'preparation-binding-at-start.json',receipt)
    except BaseException as exc:
        create_json(args.lock.parent/'dependency-bootstrap-failure.json',dict(error=repr(exc),
            source=lock['execution']['commit'],model_loads=0,new_teacher=0,automatic_resubmission=False))
        raise
    from .matched_runner import run
    run(effective)


if __name__=='__main__':main()
