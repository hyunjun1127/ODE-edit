"""Post-terminal CPU file/tensor audit; not model replay or differentiation."""
import json
from pathlib import Path
import torch
from .preparation import member,sha
from .config import ARMS
from project.run_scripts.single_layer_edit_preserving_correction.common import digest,tensor_sha
from project.run_scripts.single_layer_edit_preserving_correction.alltoken import tensor_sha256
from project.run_scripts.single_layer_edit_preserving_correction.sequential_state import registry


def audit(root,lock):
    root=Path(root);end=json.loads((root/'terminal.json').read_text())
    if end['status']!='B1_COMPLETE':raise ValueError('TERMINAL_REQUIRED')
    manifest=json.loads((root/'artifact-manifest.json').read_text())
    files=[]
    for item in manifest['members']:
        p=Path(item['path'])
        if not p.resolve().is_relative_to(root.resolve()) or p.is_symlink():raise ValueError('ARTIFACT_SCOPE_OR_SYMLINK')
        if p.stat().st_size!=item['bytes'] or sha(p)!=item['sha256']:raise ValueError('ARTIFACT_FILE_BYTES')
        files.append(dict(path=str(p),bytes=item['bytes'],sha256=item['sha256']))
    checkpoints=[]
    for arm in ('N4',*ARMS):
        receipt=end['commits'][arm];cp=torch.load(receipt['checkpoint']['path'],weights_only=True,mmap=True,map_location='cpu')
        if cp['max_batches']!=1 or cp['batch']!=1 or cp['sequential_authorized'] is not False or cp['auto_continue'] is not False:
            raise ValueError('CHECKPOINT_SCOPE')
        if cp['sample_order']!=lock['sample_order'] or cp['source']!=lock['execution']:raise ValueError('CHECKPOINT_SOURCE_ORDER')
        identity=receipt['identity']
        for key,shape,ref in (('weight',(4096,14336),'W'),('M4',(1,14336,14336),'M')):
            t=cp[key]
            if tuple(t.shape)!=shape or t.dtype!=torch.float32 or not bool(torch.isfinite(t).all()) or tensor_sha(t)!=identity[ref]:
                raise ValueError('CHECKPOINT_'+key)
        if (digest(cp['rng'])!=identity['rng'] or digest(cp['context'])!=identity['context'] or
            digest(cp['received_ledger'])!=identity['ledger'] or cp['registry']!=registry(cp['received_ledger'])):
            raise ValueError('CHECKPOINT_NONWEIGHT_STATE')
        if len(cp['history'])!=1 or cp['history'][0]['history_append']!=1 or cp['history'][0]['layer']!=4:
            raise ValueError('CHECKPOINT_HISTORY_RECEIPT')
        checkpoints.append(dict(arm=arm,checkpoint=receipt['checkpoint'],identity=identity,shape_finite_hash='PASS',
            state_fields='CPU_VERIFIED',history='RECORDED_ONCE_NOT_NATIVE_RECOMPUTATION',GPU_continuation='NOT_TESTED'))
        del cp
    gradients={};grad_receipts=[]
    for arm in ARMS:
        exact=json.loads((root/'arms'/arm/'execution-exactness.json').read_text())
        if exact['gradient_sweeps']==0:continue
        h=exact['gradient_sha256'];paths=list((root/'arms'/arm/'gradients').glob(h+'.pt'))
        if len(paths)!=1:raise ValueError('GRADIENT_FILE_CARDINALITY')
        value=torch.load(paths[0],weights_only=True,mmap=True,map_location='cpu')['gradient']
        if value.shape!=(4096,14336) or value.dtype!=torch.float64 or not bool(torch.isfinite(value).all()) or tensor_sha256(value)!=h:
            raise ValueError('GRADIENT_TENSOR_BYTES')
        gradients[arm]=value;grad_receipts.append(dict(arm=arm,file=member(paths[0]),tensor_sha256=h))
    gradient_exact=None if len(gradients)!=2 else bool(torch.equal(gradients[ARMS[0]],gradients[ARMS[1]]))
    return dict(status='CPU_ARTIFACT_CHECKS_COMPLETE',files=len(files),logical_bytes=sum(x['bytes'] for x in files),
        inventory_root=digest(files),checkpoints=checkpoints,gradients=grad_receipts,independent_gradient_tensor_exact=gradient_exact,
        teacher_model_full_rehash='PRIOR_EXECUTION_VERIFIED_MANIFEST_AND_BINDING_REUSED',new_GPU=0,
        FD_or_native_replay='NOT_RUN',method_derivative_PASS_not_assigned=True)
