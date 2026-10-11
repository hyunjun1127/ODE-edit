"""Server1 (devbox) resubmission of the cancelled 63303 Qwen zsRE SPHERE B11-B20 resume.

User 2026-10-11, "server1 cap 남으니 그냥 올려": resubmit on server1 the 63303
`s4-qwen25-zsre-sphere-resume-b10` that was cancelled before it ran (07:05 KST stop of all
SPHERE jobs). The science is 63303's: the same server2 B10 state (62087 B1-B9 -> 62538 B10;
weights, native history, contexts, RNG and cursor), SPHERE hparams, stream order, the 23
native numeric source files byte for byte, per-batch two-slot checkpoints and the B15/B20
public-query zsRE evaluation of all seen requests (final W20 = first 2,000).

`validate`, `prepare` and `run` below are the text of official.runners.server4.sphere_b10_resume
with only the host bindings changed: the server label ('server4' -> 'server1') and the host
paths ('/data/janghj/' -> '/mnt/raid5/janghj/'), plus this task's ROOT/TASK/AUTHORITY constants.
official/tests/test_sphere_s1.py checks that equality. `validate_parent` and `save_reserved` are
imported unchanged. `stage`, `reserve` and `derive` are the devbox-only input, storage and
63303-derivation steps (the transfer itself is a read-only rsync pull with SHA-256 on both sides).
"""
import argparse
import copy
import importlib.metadata
import json
import os
from pathlib import Path
import re
import tempfile
import time
import traceback
import uuid

import torch
from official.baselines import registry
from official.experiments import checkpoint as cp
from official.experiments.prepare import digest, file_sha, write_new
from official.runners.server1.common import read, member, verify
from official.runners.server1.assets import member as asset_member
from official.runners.server1.cake_run import verify_asset
from official.runners.server2.qwen_run import _load_model, _model_seed, _tokenizer_receipt, validate_config
from official.runners.server2.qwen_eval_refresh import public_zsre
from official.runners.server2.sphere_b9_resume import equal_state
from official.runners.server4.qwen_native_state import NativeState
from official.runners.server1.sphere_s1_common import (AUTHORITY, LOCAL, TASK, differences, field_diff, host_only,
                                                       received, transfer_rows)
from official.runners.server4.sphere_b10_resume import (CONTEXT_SHA, PARENT_BYTES, PARENT_CONFIG, PARENT_SHA,
                                                        PARENT_SOURCE, require, save_reserved, validate_parent)
from official.tracking import init, official_zsre_metrics

ROOT = LOCAL / 'zsre-b10-resume'
CODE = Path(__file__).resolve().parents[3]
SERVER2_ROOT = '/mnt/raid5/janghj/ODE-edit/local/qwen-zsre-sphere-b9-resume-20261010/registration-r1'
SERVER4_ROOT = '/var/tmp/ode-edit-janghj-qwen-sphere-resume-20261011'  # 63303 root, read only
CANCELLED = dict(job_id='63303', name='s4-qwen25-zsre-sphere-resume-b10', server='server4',
                 source='16512a230aa8ceb319680839c01ee41a1bf67640',
                 transfer_file_sha256='14f6eb1d16e14ffe506429fe20b2c6d048787b8dfbf0ec7b7028c0659909ef00',
                 transfer_config_sha256='ba9451090eb9da9bbbc0483c93fab39c77c11f7df3bf74e79a816d6c989f6605',
                 script_sha256='286d49c07bce9a1643c06b7f10962183f09fd4c4505653e1025f5e9a9ec15a74',
                 cancelled='2026-10-11T07:05:24+09:00 audits/global/sphere-stop-20261011/cancellation.json')
SLOT_MARGIN = 64 * 2**20
RAW_BUDGET = 2 * 2**30
MIN_FREE_RUNTIME = 8 * 2**30
# 63303 transfer fields that are host bindings; every other field must be equal.
HOST_FIELDS = tuple(re.compile(p) for p in (
    r'authority', r'server', r'config_sha256', r'output', r'model_snapshot', r'tracking_env',
    r'(parent_checkpoint|scientific_config|stream|stream_lock|projector)\.path', r'evidence\[\d+\]\.path',
    r'storage\.(path|bytes|sha256)',
    r'model_members\[\d+\]\.(path|real_path|allocated_bytes|device|inode|nlink|mtime_ns|ctime_ns)'))


def validate(c):
    require(c['authority'] == AUTHORITY and c['server'] == 'server1' and c['start_batch'] == 10, 'TRANSFER_SCOPE')
    require(c['parent_checkpoint']['sha256'] == PARENT_SHA and c['parent_checkpoint']['bytes'] == PARENT_BYTES, 'PARENT_CHECKPOINT_ALLOWLIST')
    require(c['parent_identity']['code_commit'] == PARENT_SOURCE and c['parent_identity']['config_sha256'] == PARENT_CONFIG, 'PARENT_SOURCE_CONFIG')
    require(c['context_sha256'] == CONTEXT_SHA and c['target_batches'] == list(range(11,21)), 'NO_PREFIX_REPLAY')
    require(c['config_sha256'] == digest({k:v for k,v in c.items() if k != 'config_sha256'}), 'TRANSFER_CONFIG_SHA')
    science = read(verify(c['scientific_config'])); validate_config(science)
    require(science['config_sha256'] == PARENT_CONFIG and science['hparams'] == read(CODE/'official/hparams/SPHERE/qwen25.json'), 'SCIENTIFIC_HPARAMS_CHANGED')
    for row in c['numeric_sources']:
        require(file_sha(CODE/row['relative']) == row['sha256'], 'NATIVE_NUMERIC_SOURCE_CHANGED_'+row['relative'])
    return science


def prepare(root=ROOT):
    root=Path(root); require(root==ROOT and not (root/'transfer.json').exists(), 'NEW_TRANSFER_PREPARATION_REQUIRED')
    e=read(root/'inputs/parent-evidence.json');rows=e['files']
    inputs={r['path']:dict(path=str(root/'inputs/parent'/r['name']),bytes=r['bytes'],sha256=r['sha256']) for r in rows}
    for m in inputs.values():verify(m)
    def lookup(suffix):
        matches=[m for path,m in inputs.items() if path.endswith(suffix)]
        require(len(matches)==1,'EXACT_PARENT_METADATA_'+suffix);return matches[0]
    old=read(verify(lookup('/configs/qwen25-zsre-sphere.json')))
    receipt=read(verify(lookup('/asset-preflight.json')))
    commit=read(verify(lookup('/commits/b10.json')));pointer=read(verify(lookup('/checkpoint/latest.json')))
    require(commit['checkpoint_sha256']==pointer['sha256']==PARENT_SHA and pointer['batch']==10, 'B10_COMMIT_POINTER')
    original=read(verify(lookup('/resume-parent.json')))
    for n in range(1,10):
        require(read(verify(lookup(f'/commits/b{n:02d}.json')))['completed_batch']==n, 'MISSING_ANCESTOR_COMMIT')
    snapshot=Path('/mnt/raid5/janghj/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots')/old['model_identity']['revision']
    model_members=[asset_member(snapshot/name,allow_symlink=True,expected_sha=sha) for name,sha in receipt['asset_identity']['model_files_sha256'].items()]
    packages={k:importlib.metadata.version(k) for k in receipt['runtime']['packages']}
    require(packages==receipt['runtime']['packages'],'RUNTIME_PACKAGES_CHANGED')
    pinfo=dict(read(verify(lookup('/assets.json')))['projector'],path='/mnt/raid5/janghj/EasyEdit/examples/null_space_project_Qwen2.5-7B-Instruct.pt')
    require(member(pinfo['path'])['sha256']==pinfo['sha256'],'PROJECTOR_BYTES_CHANGED')
    partial=root/'inputs/parent-b10.pt.partial';parent=root/'inputs/parent-b10.pt'
    candidate=partial if partial.exists() else parent
    pm=member(candidate);require(pm['sha256']==PARENT_SHA and pm['bytes']==PARENT_BYTES,'TRANSFERRED_CP_CORRUPT')
    if partial.exists():os.rename(partial,parent)
    pm=dict(pm,path=str(parent))
    lock=read(verify(lookup('/source-lock.json')))
    # Entire used native implementation families, physical state adapter, model
    # loader and scorer must have the same bytes as the failed SH2 execution.
    roots=('official/baselines/easyedit/models/SPHERE/','official/baselines/easyedit/models/alphaedit/',
        'official/baselines/easyedit/util/')
    names=('official/runners/server4/qwen_native_state.py','official/runners/server2/qwen_eval_refresh.py',
        'official/runners/server2/qwen_run.py','official/evaluation/zsre_paper.py')
    numeric=[r for r in lock['members'] if r['relative'].startswith(roots) or r['relative'] in names]
    require(len(numeric)>20,'NUMERIC_SOURCE_BINDING_MISSING')
    matches=[m for path,m in inputs.items() if '/commits/pending-b10-' in path];require(len(matches)==1,'PARENT_PENDING_SINGLE')
    pending=read(verify(matches[0]))
    c=dict(authority=AUTHORITY,server='server1',start_batch=10,target_batches=list(range(11,21)),
        scientific_config=lookup('/configs/qwen25-zsre-sphere.json'),parent_checkpoint=pm,
        parent_identity=commit['checkpoint_identity'],parent_pending=pending,b9_binding_sha256=original['binding_sha256'],
        context_sha256=CONTEXT_SHA,stream=lookup('/streams/zsre-stream.json'),stream_lock=lookup('/streams/zsre-stream.lock.json'),
        evidence=list(inputs.values()),numeric_sources=numeric,model_snapshot=str(snapshot),model_members=model_members,
        model_identity=old['model_identity'],tokenizer_sha256=receipt['asset_identity']['tokenizer_files_sha256'],
        packages=packages,projector=pinfo,output=str(root/'output'),min_free_bytes=8*2**30,
        storage=member(root/'preparation/storage.json'),tracking_env='/mnt/raid5/janghj/ODE-edit/servers/local/wandb.env',
        checkpoint_policy='two_preallocated_slots_atomic_latest_every_batch_keep_final',
        origin=dict(server='server2',jobs=['62087','62538'],batches={'62087':[1,9],'62538':[10,10]},
            stop_reason='RESOURCE_BLOCKED_STORAGE_KEEP_CHECKPOINT',pretrained_qualification='NOT_RUN_USER_DISABLED'))
    c['config_sha256']=digest(c);validate(c)
    records=read(verify(c['stream']));require(isinstance(records,list) and len(records)==2000,'STREAM_2K')
    parent_payload=torch.load(parent,map_location='cpu',weights_only=False,mmap=True)
    validate_parent(parent_payload,c,records);del parent_payload
    from transformers import AutoTokenizer
    from official.evaluation.zsre_query_parity import compare_queries
    tok=AutoTokenizer.from_pretrained(snapshot,local_files_only=True,use_fast=True)
    if tok.pad_token_id is None:tok.pad_token=tok.eos_token
    t=_tokenizer_receipt(snapshot,tok,read(verify(c['stream_lock'])))
    require(t['tokenizer_sha256']==c['parent_identity']['tokenizer_sha256'],'TOKENIZER_PARENT_PARITY')
    proof=compare_queries(tok,records,model_family='qwen25');require(proof['status']=='PASS_CPU_QUERY_ONLY','ZSRE_QUERY_PARITY')
    write_new(root/'preparation/query-proof.json',proof)
    write_new(root/'transfer.json',c)
    write_new(root/'preparation/READY.json',dict(status='PASS_CPU_ASSETS_B10',config=member(root/'transfer.json'),
        actual_model_forward=0,start_batch=10,next_batch=11,numeric_files_unchanged=len(numeric),storage=read(root/'preparation/storage.json')))
    print('PASS_CPU_ASSETS_B10_READY',flush=True)


def run(root=ROOT,resume=False):
    root=Path(root); c=read(root/'transfer.json');science=validate(c);lock=read(root/'execution-lock.json')
    require(CODE==Path(lock['source_directory']).resolve(),'FROZEN_OFFICIAL_SOURCE_REQUIRED')
    for row in lock['members']:require(file_sha(CODE/row['relative'])==row['sha256'],'FROZEN_SOURCE_CHANGED')
    require(member(root/'transfer.json')==lock['config'],'FROZEN_TRANSFER_CONFIG')
    for m in c['evidence']:verify(m)
    for m in c['model_members']:verify_asset(m)
    require({k:importlib.metadata.version(k) for k in c['packages']}==c['packages'],'RUNTIME_PACKAGE_CHANGED')
    verify(c['parent_checkpoint'])
    records=read(verify(c['stream']));stream_lock=read(verify(c['stream_lock']))
    out=Path(c['output']); observations=out/'resumes'/uuid.uuid4().hex if resume else out
    observations.mkdir(parents=True,exist_ok=True)
    require(resume or not (out/'runtime.json').exists(),'NO_IMPLICIT_RETRY')
    identity=dict(config_sha256=c['config_sha256'],stream_sha256=c['stream']['sha256'],code_commit=lock['code_commit'],
        official_tree_sha256=lock['official_tree_sha256'],model_revision=c['model_identity']['revision'],
        tokenizer_sha256=c['parent_identity']['tokenizer_sha256'],assets_sha256=digest(dict(model=c['model_members'],projector=c['projector'])))
    values=dict(server='server1',task_id=TASK,arm='qwen25-zsre-sphere-resume-b10',attempt='r1'+('-'+observations.name if resume else ''),
        source_sha=lock['code_commit'],config_sha=c['config_sha256'],model='qwen25',model_family='qwen2',writer='SPHERE',baseline='SPHERE',role='scientific',
        metric_schema='official-baselines-scalar-v1',instruction_id='USER-OFFICIAL-BASELINES-20261008-R1',dataset='zsre',
        checkpoint_sha256=PARENT_SHA,source_run_id='job62538-b10')
    with init(env_file=c['tracking_env'],spool=observations/'tracking',config=values) as tracker:
        _model_seed(science['edit_seed']);model,tok=_load_model(c['model_snapshot'],c['model_identity']['revision'])
        t=_tokenizer_receipt(c['model_snapshot'],tok,stream_lock);require(t['tokenizer_sha256']==identity['tokenizer_sha256'],'TOKENIZER_RUNTIME_PARENT')
        hp=registry.hparams('SPHERE','qwen25',overrides=science['hparams']);native=NativeState('SPHERE',model,hp,dict(projector=c['projector']))
        payload=(cp.load(out/'checkpoint',identity) if resume else torch.load(c['parent_checkpoint']['path'],map_location='cpu',weights_only=False,mmap=True))
        if not resume:validate_parent(payload,c,records)
        start=payload['batch'];require(10<=start<20,'RESUME_BATCH_RANGE')
        cursor=native.restore_from_checkpoint(payload)
        require(equal_state(native.context_snapshot(),payload['contexts']) and equal_state(native.cache_for_checkpoint(),payload['cache_c']), 'RESTORE_CONTEXT_HISTORY')
        for name,w in payload['weights'].items():require(torch.equal(model.get_parameter(name).detach().cpu(),w),'RESTORE_WEIGHT')
        require(equal_state(cp.rng_snapshot(),payload['rng']) and cursor==payload['evaluation_cursor'],'RESTORE_RNG_CURSOR')
        del payload
        write_new(observations/'runtime.json',dict(identity=identity,parent=c['parent_checkpoint'],start_batch=start,
            actual_restore='W_H_CONTEXT_RNG_CURSOR_EXACT',source_numeric_bytes_unchanged=True,GPU=torch.cuda.get_device_name(),
            cross_hardware_future_bit_parity='NOT_CLAIMED',job_id=os.environ.get('SLURM_JOB_ID'),origin=c['origin']))
        try:
            for batch in range(start+1,21):
                st=os.statvfs(out);require(st.f_bavail*st.f_frsize>=c['min_free_bytes'],'RESOURCE_BLOCKED_STORAGE_KEEP_CHECKPOINT')
                before=time.monotonic();requests=registry.requests(records[(batch-1)*100:batch*100],'SPHERE','qwen25')
                with torch.enable_grad():native.apply(tok,requests)
                require(digest(native.context_snapshot())==CONTEXT_SHA,'CONTEXT_CHANGED')
                factual=None
                if batch in (15,20):
                    with torch.inference_mode():result=public_zsre(model,tok,records[:batch*100],identity)
                    path=observations/'evaluations'/f'w{batch:02d}.json';write_new(path,result)
                    factual=dict(member(path),summary=result['summary'])
                cursor=dict(completed_batch=batch,evaluated_endpoints=[0,5,10]+[b for b in (15,20) if b<=batch],
                    per_request_records=[dict(occurrence_index=r['occurrence_index'],case_id=r['case_id']) for r in records[:batch*100]],
                    ancestor_prefix_edits=1000,ancestor_checkpoint_sha256=PARENT_SHA,factual=factual)
                weights={name:model.get_parameter(name) for name in native.editable_parameter_names()}
                ref=save_reserved(out/'checkpoint',batch=batch,identity=identity,weights=weights,cache_c=native.cache_for_checkpoint(),
                    contexts=native.context_snapshot(),cursor=cursor,ancestor=c['parent_checkpoint'])
                write_new(observations/'commits'/f'b{batch:02d}.json',dict(checkpoint=ref,identity=identity,cursor=cursor,seconds=time.monotonic()-before))
                if factual:
                    values_out=official_zsre_metrics(factual['summary'],config_values=values,endpoint='all_seen/post',edits=batch*100,post_state_edits=batch*100)
                    require(tracker.log(values_out) is not False,'TRACKING_REJECTED')
            write_new(observations/'COMPLETE.json',dict(identity=identity,completed_edits=2000,new_edits=(20-start)*100,
                ancestor_jobs=['62087','62538'],ancestor_prefix_edits=1000,checkpoint=ref,factual=factual,source=c['origin']))
        except BaseException as error:
            write_new(observations/'FAILURE.json',dict(type=type(error).__name__,message=str(error),traceback=traceback.format_exc(),checkpoint_keep=True))
            raise




def stage(root=ROOT, receipt=None):
    """inputs/parent-evidence.json from the receipt; the evidence set must equal 63303's (path, bytes, SHA, name)."""
    root = Path(root); require(root == ROOT and not (root/'inputs/parent-evidence.json').exists(), 'NEW_STAGE_REQUIRED')
    rows = transfer_rows(receipt or (LOCAL/'transfer/receipt.tsv'))
    old = read(root/'inputs/server4-63303/inputs/parent-evidence.json')
    received(rows, root/'inputs/server4-63303/inputs/parent-evidence.json',
             SERVER4_ROOT + '/inputs/parent-evidence.json')
    files = []
    for previous in old['files']:
        local = root/'inputs/parent'/previous['name']
        row, found = received(rows, local, previous['path'])
        files.append(dict(path=row['source'], bytes=found['bytes'], sha256=found['sha256'], local=str(local),
                          name=previous['name']))
        require((files[-1]['path'], files[-1]['bytes'], files[-1]['sha256']) ==
                (previous['path'], previous['bytes'], previous['sha256']), 'EVIDENCE_NOT_63303_SET')
    require(len(files) == len(old['files']) == 22, 'EVIDENCE_COUNT')
    log = root/'inputs/62538-qwen25-zsre-sphere-resume-b9.out'
    received(rows, log, SERVER2_ROOT + '/logs/qwen25-zsre-sphere-resume-b9-62538.out')
    tail = log.read_text()[-8000:]
    require(tail == old['failure_tail'] and old['parent_job'] == '62538' and old['root'] == SERVER2_ROOT, 'PARENT_FAILURE_TAIL')
    _, parent = received(rows, root/'inputs/parent-b10.pt',
                         SERVER2_ROOT + '/runs/qwen25-zsre-sphere/checkpoint/batch-10-6ab0bea43a595f97.pt')
    require((parent['sha256'], parent['bytes']) == (PARENT_SHA, PARENT_BYTES), 'PARENT_B10_SHA')
    with (root/'inputs/parent/failure-tail.txt').open('x') as handle:
        handle.write(tail)
    write_new(root/'inputs/parent-evidence.json', dict(files=files, failure_tail=tail, parent_job='62538', root=SERVER2_ROOT))
    return files


def reserve(root=ROOT):
    """63303's checkpoint policy on this host: two preallocated slots, the same 64 MiB slot margin."""
    root = Path(root); require(root == ROOT and not (root/'preparation/storage.json').exists(), 'NEW_STORAGE_REQUIRED')
    folder = root/'output/checkpoint'; folder.mkdir(parents=True)
    slot = PARENT_BYTES + SLOT_MARGIN
    st = os.statvfs(root); free = st.f_bavail * st.f_frsize
    required = 3 * slot + RAW_BUDGET + MIN_FREE_RUNTIME
    require(free >= required, 'RESOURCE_BLOCKED_STORAGE_BEFORE_ALLOCATION')
    for index in (0, 1):
        with (folder/f'slot-{index}.pt').open('xb') as stream:
            os.posix_fallocate(stream.fileno(), 0, slot)
    (root/'preparation').mkdir(exist_ok=True)
    storage = dict(root=str(root), device=os.stat(root).st_dev, input_checkpoint_bytes=PARENT_BYTES, slot_capacity=slot,
                   slots=2, required_free_before_allocation=required, observed_free_before_allocation=free,
                   min_free_runtime=MIN_FREE_RUNTIME, raw_budget=RAW_BUDGET, old_files_deleted=0)
    write_new(root/'preparation/storage.json', storage)
    return storage


def derive(root=ROOT, receipt=None):
    """Field-level differences of this transfer config from the cancelled 63303's; only host bindings may differ."""
    root = Path(root); rows = transfer_rows(receipt or (LOCAL/'transfer/receipt.tsv'))
    copied = root/'inputs/server4-63303'
    _, found = received(rows, copied/'transfer.json', SERVER4_ROOT + '/transfer.json')
    require(found['sha256'] == CANCELLED['transfer_file_sha256'], 'NOT_63303_TRANSFER')
    old, c = read(copied/'transfer.json'), read(root/'transfer.json')
    require(old['config_sha256'] == CANCELLED['transfer_config_sha256']
            == digest({k: v for k, v in old.items() if k != 'config_sha256'}), 'NOT_63303_TRANSFER')
    _, script = received(rows, copied/'scripts/run.sh', SERVER4_ROOT + '/scripts/run.sh')
    require(script['sha256'] == CANCELLED['script_sha256'], 'NOT_63303_SCRIPT')
    validate(c)
    changed = host_only(field_diff(old, c), HOST_FIELDS)
    members = lambda x: [(Path(m['path']).name, m['sha256'], m['bytes']) for m in x['model_members']]
    require(members(old) == members(c), 'MODEL_FILES_DIFFER')
    result = dict(cancelled=CANCELLED, transfer_63303=member(copied/'transfer.json'), transfer=member(root/'transfer.json'),
                  changed_fields=changed, scientific_fields_changed=[],
                  differences=differences(old, c, [k for k in changed if not k.startswith(('evidence[', 'model_members['))]),
                  evidence_paths_changed=sum(k.startswith('evidence[') for k in changed),
                  model_member_stat_fields_changed=sum(k.startswith('model_members[') for k in changed),
                  model_files=[dict(name=n, sha256=s, bytes=z) for n, s, z in members(c)])
    write_new(root/'preparation/derivation-63303.json', result)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['stage','reserve','prepare','derive','run'])
    p.add_argument('--root',default=str(ROOT));p.add_argument('--receipt');p.add_argument('--resume',action='store_true')
    a=p.parse_args()
    if a.action=='stage':stage(Path(a.root),a.receipt)
    elif a.action=='reserve':reserve(Path(a.root))
    elif a.action=='prepare':prepare(a.root)
    elif a.action=='derive':print(json.dumps(derive(Path(a.root),a.receipt)['changed_fields']))
    else:run(a.root,a.resume)
