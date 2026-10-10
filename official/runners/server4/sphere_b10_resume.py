"""Move the exact SH2 SPHERE B10 state to SH4, keeping native science unchanged."""
import argparse
import copy
import importlib.metadata
import json
import os
from pathlib import Path
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
from official.tracking import init, official_zsre_metrics

TASK = 'qwen-sphere-server4-resume-20261011'
AUTHORITY = 'USER-QWEN-ZSRE-SPHERE-SERVER4-RESUME-20261011'
PARENT_SHA = '6ab0bea43a595f97df631176c0d5a6254eea90038b0c7c6eead8165c575abbd8'
PARENT_BYTES = 8535444185
PARENT_SOURCE = '78017702c1d3d5a6843e27e63fcf105206d22dbe'
PARENT_CONFIG = 'dcc7018f54b1f63c8c295ebe5d82b32fb933856e0d0b94b8659c9238ef41a814'
CONTEXT_SHA = '5c01bc1a91c0890af2897f18b7a5f95de011badc1eca5e27f44199acc0cb6515'
ROOT = Path('/var/tmp/ode-edit-janghj-qwen-sphere-resume-20261011')
CODE = Path(__file__).resolve().parents[3]


def require(ok, code):
    if not ok:
        raise ValueError(code)


def validate_parent(payload, c, records, *, layers=range(4,9), hidden=3584, width=18944):
    require(payload['schema'] == 'official-baseline-checkpoint-v1' and payload['method'] == 'SPHERE'
            and payload['batch'] == 10, 'EXACT_SPHERE_B10_REQUIRED')
    require(payload['identity'] == c['parent_identity'], 'PARENT_IDENTITY_CHANGED')
    require(set(payload['weights']) == {f'model.layers.{l}.mlp.down_proj.weight' for l in layers}
            and set(payload['cache_c']) == {str(l) for l in layers}, 'PARENT_PHYSICAL_LAYER_SET')
    for name, shape in [('weights',(hidden,width)), ('cache_c',(width,width))]:
        for value in payload[name].values():
            require(value.dtype == torch.float32 and value.device.type == 'cpu' and tuple(value.shape) == shape, 'PARENT_TENSOR_SCHEMA')
            for block in value.split(256):
                require(bool(torch.isfinite(block).all()), 'PARENT_NONFINITE')
    cursor = payload['evaluation_cursor']
    require(cursor['completed_batch'] == 10 and cursor['evaluated_endpoints'] == [0,5,10], 'PARENT_CURSOR')
    require(cursor['per_request_records'] == [dict(occurrence_index=r['occurrence_index'],case_id=r['case_id']) for r in records[:1000]], 'PARENT_FIRST1000_ORDER')
    require(digest(payload['contexts']) == c['context_sha256'], 'PARENT_CONTEXT')
    require(digest(c['parent_pending']) == cursor['pending_receipt_sha256'], 'PARENT_PENDING_BINDING')
    require(cursor['ancestor_prefix_edits'] == 900 and cursor['resume_parent_binding_sha256'] == c['b9_binding_sha256'], 'B9_TO_B10_ANCESTRY')
    rng = payload['rng']
    require(set(rng) == {'python','numpy','torch_cpu','torch_cuda'} and rng['torch_cpu'].dtype == torch.uint8
            and isinstance(rng['torch_cuda'],list) and len(rng['torch_cuda']) == 1
            and rng['torch_cuda'][0].dtype == torch.uint8, 'PARENT_RNG_SCHEMA')


def save_reserved(folder, *, batch, identity, weights, cache_c, contexts, cursor, ancestor):
    """Two preallocated slots: overwrite only inactive; publish pointer after fsync.

    Uses the common checkpoint payload/pointer format. Failed writes keep the
    active slot intact. Reserving both slots before admission avoids a new 8 GiB
    allocation on every atomic-save cycle on this nearly full host.
    """
    folder = Path(folder); cp.validate_identity(identity)
    require(type(batch) is int and 11 <= batch <= 20, 'CHILD_BATCH_RANGE')
    pointer = folder / 'latest.json'; previous = read(pointer) if pointer.exists() else None
    if previous:
        require(previous['identity_sha256'] == digest(identity), 'CHECKPOINT_IDENTITY_MISMATCH')
        require(previous['file'] in ('slot-0.pt','slot-1.pt') and previous['batch'] + 1 == batch, 'NONCONTIGUOUS_CHILD')
        slot = 1 - int(previous['file'][5])
    else:
        require(batch == 11, 'FIRST_CHILD_MUST_BE_B11'); slot = 0
    target = folder / f'slot-{slot}.pt'
    require(target.is_file() and not target.is_symlink(), 'PREALLOCATED_SLOT_REQUIRED')
    require(weights and cache_c and all(v.dtype == torch.float32 for v in [*weights.values(),*cache_c.values()]), 'CHILD_FP32_STATE')
    payload = dict(schema='official-baseline-checkpoint-v1', batch=batch, method='SPHERE', identity=identity,
        weights={k:v.detach().cpu() for k,v in weights.items()}, cache_c={k:v.detach().cpu() for k,v in cache_c.items()},
        contexts=contexts, rng=cp.rng_snapshot(), evaluation_cursor=cursor, resume_ancestor=ancestor)
    with target.open('r+b') as stream:
        stream.seek(0); torch.save(payload,stream)
        stream.truncate(stream.tell()); stream.flush(); os.fsync(stream.fileno())
    ref = dict(batch=batch,file=target.name,sha256=file_sha(target),final_W20=batch==20,identity_sha256=digest(identity))
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=folder,prefix='.latest-',mode='w',delete=False) as stream:
            temporary=Path(stream.name);json.dump(ref,stream,sort_keys=True);stream.flush();os.fsync(stream.fileno())
        os.replace(temporary,pointer);temporary=None;cp._sync_directory(folder)
    finally:
        if temporary is not None:temporary.unlink(missing_ok=True)
    return ref


def validate(c):
    require(c['authority'] == AUTHORITY and c['server'] == 'server4' and c['start_batch'] == 10, 'TRANSFER_SCOPE')
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
    snapshot=Path('/data/janghj/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots')/old['model_identity']['revision']
    model_members=[asset_member(snapshot/name,allow_symlink=True,expected_sha=sha) for name,sha in receipt['asset_identity']['model_files_sha256'].items()]
    packages={k:importlib.metadata.version(k) for k in receipt['runtime']['packages']}
    require(packages==receipt['runtime']['packages'],'RUNTIME_PACKAGES_CHANGED')
    pinfo=dict(read(verify(lookup('/assets.json')))['projector'],path='/data/janghj/EasyEdit/examples/null_space_project_Qwen2.5-7B-Instruct.pt')
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
    c=dict(authority=AUTHORITY,server='server4',start_batch=10,target_batches=list(range(11,21)),
        scientific_config=lookup('/configs/qwen25-zsre-sphere.json'),parent_checkpoint=pm,
        parent_identity=commit['checkpoint_identity'],parent_pending=pending,b9_binding_sha256=original['binding_sha256'],
        context_sha256=CONTEXT_SHA,stream=lookup('/streams/zsre-stream.json'),stream_lock=lookup('/streams/zsre-stream.lock.json'),
        evidence=list(inputs.values()),numeric_sources=numeric,model_snapshot=str(snapshot),model_members=model_members,
        model_identity=old['model_identity'],tokenizer_sha256=receipt['asset_identity']['tokenizer_files_sha256'],
        packages=packages,projector=pinfo,output=str(root/'output'),min_free_bytes=8*2**30,
        storage=member(root/'preparation/storage.json'),tracking_env='/data/janghj/ODE-edit/servers/local/wandb.env',
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
    values=dict(server='server4',task_id=TASK,arm='qwen25-zsre-sphere-resume-b10',attempt='r1'+('-'+observations.name if resume else ''),
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


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','run']);p.add_argument('--root',default=str(ROOT));p.add_argument('--resume',action='store_true')
    a=p.parse_args();prepare(a.root) if a.action=='prepare' else run(a.root,a.resume)
