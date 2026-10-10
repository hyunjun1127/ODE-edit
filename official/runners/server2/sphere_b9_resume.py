"""Explicit USER-authorized B9 ancestor -> B10..20 descendant, never relabel old W/H."""
import argparse
import json
import os
from pathlib import Path
import tempfile
import numpy as np
import torch
from official.experiments import checkpoint
from official.experiments.prepare import digest,file_sha,write_new
from official.runners.server1.common import read,member,verify

INSTRUCTION='USER-SH2-QWEN-ZSRE-SPHERE-B9-RESUME-20261010-R1'
CELL='qwen25-zsre-sphere'
OLD=Path('/mnt/raid5/janghj/ODE-edit/local/qwen-baseline-mask-cold-rerun-20261010/registration-r1')
PARENT=OLD/'runs'/CELL
CP_SHA='7e44f382ba1f3befcb9f4429f42fd328d55422306f9cef868a920c8bf7ec17b8'
SOURCE='7b5097aa447946e35de42229c22b0c0feabd11ae'
CONFIG='dcc7018f54b1f63c8c295ebe5d82b32fb933856e0d0b94b8659c9238ef41a814'
CONTEXT='5c01bc1a91c0890af2897f18b7a5f95de011badc1eca5e27f44199acc0cb6515'
LOCAL=Path('/mnt/raid5/janghj/ODE-edit/local/qwen-zsre-sphere-b9-resume-20261010')

def require(value,code):
    if not value:raise ValueError(code)

def validate_payload(payload,binding,records,*,width=18944,hidden=3584,layers=range(4,9)):
    require(payload['schema']=='official-baseline-checkpoint-v1','PARENT_SCHEMA')
    require(payload['batch']==9 and payload['method']=='SPHERE','PARENT_B9_METHOD')
    require(payload['identity']==binding['parent_identity'],'PARENT_IDENTITY_NOT_RELABELED')
    names={f'model.layers.{layer}.mlp.down_proj.weight' for layer in layers}
    require(set(payload['weights'])==names and set(payload['cache_c'])=={str(x) for x in layers},'PARENT_LAYERS')
    for kind,shape in [('weights',(hidden,width)),('cache_c',(width,width))]:
        for tensor in payload[kind].values():
            require(tensor.dtype==torch.float32 and tuple(tensor.shape)==shape and tensor.device.type=='cpu','PARENT_SHAPE_DTYPE')
            for block in tensor.split(256):require(bool(torch.isfinite(block).all()),'PARENT_NONFINITE')
    cursor=payload['evaluation_cursor']
    require(cursor['completed_batch']==9 and cursor['evaluated_endpoints']==[0,5],'PARENT_CURSOR')
    expected=[dict(occurrence_index=r['occurrence_index'],case_id=r['case_id']) for r in records[:900]]
    require(len(expected)==900 and cursor['per_request_records']==expected,'PARENT_900_ORDERED_PREFIX')
    require(digest(payload['contexts'])==binding['context_sha256'],'PARENT_CONTEXT')
    rng=payload['rng'];require(set(rng)=={'python','numpy','torch_cpu','torch_cuda'},'PARENT_RNG_SCHEMA')
    require(rng['torch_cpu'].dtype==torch.uint8 and rng['torch_cpu'].ndim==1,'PARENT_CPU_RNG')
    require(isinstance(rng['torch_cuda'],list) and len(rng['torch_cuda'])==1 and
            all(x.dtype==torch.uint8 and x.ndim==1 for x in rng['torch_cuda']),'PARENT_CUDA_RNG')
    return dict(start_batch=9,next_batch=10,remaining_batches=list(range(10,21)),
                parent_requests=900,remaining_requests=1100,shape_dtype_finite=True,
                prefix_and_context_verified=True,RNG_schema_verified=True)

def seal_parent(path):
    pointer=read(PARENT/'checkpoint/latest.json')
    require(pointer['batch']==9 and not pointer['final_W20'] and pointer['sha256']==CP_SHA,'EXACT_PARENT_POINTER')
    cp=member(PARENT/'checkpoint'/pointer['file'])
    require(cp['sha256']==CP_SHA and cp['bytes']==8535442009,'EXACT_PARENT_PAYLOAD')
    config=read(OLD/'configs'/f'{CELL}.json')
    require(config['config_sha256']==CONFIG,'EXACT_PARENT_CONFIG')
    identity=read(PARENT/'commits/b09.json')['checkpoint_identity']
    require(identity['code_commit']==SOURCE and identity['config_sha256']==CONFIG,'EXACT_PARENT_SOURCE')
    require(pointer['identity_sha256']==digest(identity),'PARENT_POINTER_IDENTITY')
    commits=[];raw=[]
    for batch in range(1,10):
        cm=member(PARENT/'commits'/f'b{batch:02d}.json');c=read(cm['path'])
        require(c['completed_batch']==batch and c['checkpoint_identity']==identity and
            c['code_commit']==SOURCE and c['config_sha256']==CONFIG,'PARENT_COMMIT_CHAIN')
        if c.get('factual'):
            fm=member(c['factual']['cases_path']);require(fm['sha256']==c['factual']['cases_sha256'],'PARENT_RAW_HASH');raw.append(fm)
        commits.append(cm)
    binding=dict(instruction_id=INSTRUCTION,parent_job_id='62087',replaced_cold_job='62534',
        parent_checkpoint=cp,parent_pointer=member(PARENT/'checkpoint/latest.json'),parent_identity=identity,
        parent_config=member(OLD/'configs'/f'{CELL}.json'),parent_assets=member(OLD/'asset-preflight.json'),
        parent_asset_paths=member(OLD/'assets.json'),parent_source_lock=member(OLD/'source-lock.json'),
        stream=member(OLD/'streams/zsre-stream.json'),stream_lock=member(OLD/'streams/zsre-stream.lock.json'),
        parent_commits=commits,parent_raw=raw,context_sha256=CONTEXT,
        W0_parent_binding=member(OLD/'w0-parent.json'),parent_prefix_edits=900,
        child_batches=list(range(10,21)),implicit_resume=False,old_payload_rewritten=False)
    payload=torch.load(cp['path'],map_location='cpu',weights_only=False,mmap=True)
    records=read(binding['stream']['path']);records=records['records'] if isinstance(records,dict) else records
    proof=validate_payload(payload,binding,records)
    pending=member(PARENT/'commits'/f"pending-b09-{payload['evaluation_cursor']['pending_receipt_sha256']}.json")
    require(digest(read(pending['path']))==payload['evaluation_cursor']['pending_receipt_sha256'],'PARENT_PENDING_DIGEST')
    binding['parent_pending_receipt']=pending;binding['CPU_payload_validation']=proof
    binding['binding_sha256']=digest(binding);write_new(path,binding)
    return binding

def validate_binding(path,config=None,*,hash_checkpoint=True):
    b=read(path)
    require(b['instruction_id']==INSTRUCTION and b['parent_job_id']=='62087','RESUME_AUTHORITY')
    require(b['binding_sha256']==digest({k:v for k,v in b.items() if k!='binding_sha256'}),'RESUME_BINDING_DIGEST')
    require(b['child_batches']==list(range(10,21)) and b['parent_prefix_edits']==900,'RESUME_BOUNDARY')
    require(b['parent_identity']['code_commit']==SOURCE and b['parent_identity']['config_sha256']==CONFIG and
            b['context_sha256']==CONTEXT,'PARENT_FIXED_IDENTITY')
    require(b['parent_checkpoint']['sha256']==CP_SHA and b['parent_checkpoint']['bytes']==8535442009,'PARENT_SHA_ALLOWLIST')
    require(Path(b['parent_checkpoint']['path'])==PARENT/'checkpoint/batch-09-7e44f382ba1f3bef.pt','PARENT_PATH_ALLOWLIST')
    for key in ('parent_pointer','parent_config','parent_assets','parent_asset_paths','parent_source_lock','stream','stream_lock','parent_pending_receipt','W0_parent_binding'):
        verify(b[key])
    for m in b['parent_commits']+b['parent_raw']:verify(m)
    require(len(b['parent_commits'])==9,'ANCESTOR_COMMIT_COUNT')
    for n,m in enumerate(b['parent_commits'],1):
        c=read(m['path'])
        require(c['completed_batch']==n and c['checkpoint_identity']==b['parent_identity'],'ANCESTOR_COMMIT_IDENTITY')
    require(read(b['parent_commits'][-1]['path'])['checkpoint_sha256']==CP_SHA,'ANCESTOR_COMMIT_CHECKPOINT')
    if hash_checkpoint:verify(b['parent_checkpoint'])
    if config is not None:require(config==read(b['parent_config']['path']),'PARENT_SCIENTIFIC_CONFIG_CHANGED')
    return b

def equal_state(a,b):
    if isinstance(a,torch.Tensor):return isinstance(b,torch.Tensor) and torch.equal(a.cpu(),b.cpu())
    if isinstance(a,np.ndarray):return isinstance(b,np.ndarray) and np.array_equal(a,b)
    if isinstance(a,dict):return set(a)==set(b) and all(equal_state(a[k],b[k]) for k in a)
    if isinstance(a,(tuple,list)):return type(a)==type(b) and len(a)==len(b) and all(equal_state(x,y) for x,y in zip(a,b))
    return a==b

def restore_parent(path,config,records,new_identity,native,out):
    b=validate_binding(path,config)
    old=b['parent_identity']
    require(old['code_commit']==SOURCE and new_identity['code_commit']!=SOURCE,'NEW_SOURCE_REQUIRED')
    require(all(old[k]==v for k,v in new_identity.items() if k not in ('code_commit','official_tree_sha256')),'RESUME_MODEL_ASSET_INPUT_IDENTITY')
    payload=torch.load(b['parent_checkpoint']['path'],map_location='cpu',weights_only=False,mmap=True)
    proof=validate_payload(payload,b,records)
    cursor=native.restore_from_checkpoint(payload)
    require(equal_state(native.context_snapshot(),payload['contexts']),'RESTORE_CONTEXT_MISMATCH')
    require(equal_state(native.cache_for_checkpoint(),payload['cache_c']),'RESTORE_HISTORY_MISMATCH')
    for name,value in payload['weights'].items():
        require(torch.equal(native.model.get_parameter(name).detach().cpu(),value),'RESTORE_WEIGHT_MISMATCH')
    require(equal_state(checkpoint.rng_snapshot(),payload['rng']),'RESTORE_RNG_MISMATCH')
    require(cursor==payload['evaluation_cursor'] and cursor['completed_batch']==9,'RESTORE_CURSOR_MISMATCH')
    write_new(Path(out)/'resume-provenance.json',dict(instruction_id=INSTRUCTION,
        parent_binding=member(path),parent_identity=old,child_identity=new_identity,actual_restore=proof,
        W_H_context_RNG_cursor_exact=True,parent_commits=b['parent_commits'],parent_raw=b['parent_raw'],
        original_source_unchanged=True,old_metrics_replayed_into_new_run=False))
    return 9

def save_descendant(folder,*,parent_binding,**kwargs):
    """Only first B10 needs a new-root bootstrap; later saves use stock checkpoint.save.

    No copied/symlinked B9 checkpoint or invented latest pointer. B10 is a real
    post-edit child payload with new identity and an explicit immutable ancestor.
    """
    folder=Path(folder)
    if (folder/'latest.json').exists():return checkpoint.save(folder,**kwargs)
    b=validate_binding(parent_binding,hash_checkpoint=False)
    require(kwargs['batch']==10 and kwargs['method']=='SPHERE' and kwargs['evaluation_complete'],'FIRST_CHILD_MUST_BE_B10')
    checkpoint.validate_identity(kwargs['identity'])
    cursor=kwargs['evaluation_cursor']
    require(cursor['completed_batch']==10 and cursor['resume_parent_binding_sha256']==b['binding_sha256'],'FIRST_CHILD_CURSOR')
    require(len(cursor['per_request_records'])==1000,'FIRST_CHILD_REQUEST_COUNT')
    require(all(t.dtype==torch.float32 for t in [*kwargs['weights'].values(),*kwargs['cache_c'].values()]),'CHILD_FP32')
    require(kwargs['identity']['code_commit']!=b['parent_identity']['code_commit'],'CHILD_NOT_PARENT_SOURCE')
    payload=dict(schema='official-baseline-checkpoint-v1',batch=10,identity=kwargs['identity'],method='SPHERE',
        weights={k:v.detach().cpu() for k,v in kwargs['weights'].items()},
        cache_c={k:v.detach().cpu() for k,v in kwargs['cache_c'].items()},contexts=kwargs['contexts'],
        rng=checkpoint.rng_snapshot(),evaluation_cursor=cursor,resume_ancestor=member(parent_binding))
    folder.mkdir(parents=True,exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=folder,prefix='.checkpoint-',delete=False) as f:
        temp=Path(f.name);torch.save(payload,f);f.flush();os.fsync(f.fileno())
    sha=file_sha(temp);target=folder/f'batch-10-{sha[:16]}.pt';os.replace(temp,target)
    require(file_sha(target)==sha,'CHILD_CHECKPOINT_HASH')
    ref=dict(batch=10,file=target.name,sha256=sha,final_W20=False,identity_sha256=digest(kwargs['identity']))
    with tempfile.NamedTemporaryFile(dir=folder,prefix='.latest-',mode='w',delete=False) as f:
        temp=Path(f.name);json.dump(ref,f,sort_keys=True);f.flush();os.fsync(f.fileno())
    os.replace(temp,folder/'latest.json');checkpoint._sync_directory(folder)
    return ref

def finish(root):
    root=Path(root);binding=validate_binding(root/'resume-parent.json',hash_checkpoint=False)
    out=root/'runs'/CELL;provenance=read(out/'resume-provenance.json')
    require(provenance['W_H_context_RNG_cursor_exact'],'ACTUAL_RESTORE_REQUIRED')
    children=[read(out/'commits'/f'b{x:02d}.json') for x in range(10,21)]
    identity=provenance['child_identity']
    require(provenance['parent_identity']==binding['parent_identity'] and
            provenance['parent_binding']==member(root/'resume-parent.json'),'FINAL_ANCESTRY_BINDING')
    require(all(c['completed_batch']==n and c['checkpoint_identity']==identity for n,c in zip(range(10,21),children)),'CHILD_COMMIT_CHAIN')
    require(not any((out/'commits'/f'b{n:02d}.json').exists() for n in range(1,10)),'ANCESTOR_RELABEL_FORBIDDEN')
    cp=read(out/'checkpoint/latest.json');require(cp['batch']==20 and cp['final_W20'] and cp['sha256']==children[-1]['checkpoint_sha256'],'FINAL_W20')
    require(cp['identity_sha256']==digest(identity) and
            file_sha(out/'checkpoint'/cp['file'])==cp['sha256'],'FINAL_PAYLOAD_IDENTITY_HASH')
    factual=children[-1]['factual'];require(file_sha(factual['cases_path'])==factual['cases_sha256'] and len(read(factual['cases_path']))==2000,'FINAL_FULL_2K')
    require(factual['work']['evaluation_profile']=='zsre-public-query-v1','FINAL_PUBLIC_QUERY')
    write_new(out/'terminal.json',dict(status='RESUMED_W20_COMPLETE',instruction_id=INSTRUCTION,actual_job_id=os.environ['SLURM_JOB_ID'],
        parent_job_id='62087',parent_batches=list(range(1,10)),child_batches=list(range(10,21)),
        parent_commits=binding['parent_commits'],child_commits=[member(out/'commits'/f'b{x:02d}.json') for x in range(10,21)],
        completed_edits=2000,new_edit_applications=1100,checkpoint=cp,checkpoint_identity=identity,
        factual=factual,resume_provenance=member(out/'resume-provenance.json'),old_history_not_rewritten=True))

def run(root):
    from official.runners.server2.qwen_pipeline import child
    root=Path(root);validate_binding(root/'resume-parent.json')
    for m in read(root/'input-lock.json')['members']:verify(m)
    for m in read(root/'source-lock.json')['members']:require(file_sha(root/'source'/m['relative'])==m['sha256'],'FROZEN_SOURCE_CHANGED')
    out=root/'runs'/CELL;require(not out.exists(),'NO_DUPLICATE_RESUMED_ATTEMPT')
    child(root,'execute',out,dataset='zsre',config=root/'configs'/f'{CELL}.json',
        extra=('--resume-parent',str(root/'resume-parent.json')),label=CELL+'-resume-b9')
    finish(root)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);a=p.parse_args();run(a.root)
