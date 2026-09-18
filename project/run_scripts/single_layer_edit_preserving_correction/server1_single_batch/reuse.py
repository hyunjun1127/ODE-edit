"""Read-only frozen M reuse identities; no fitting, observers, or selectors."""
import json
from pathlib import Path
import numpy as np
import torch
from ..common import member, tensor_sha, digest
from ..geometry import RightSpace
from ..alltoken import tensor_sha256 as header_sha

COMPLETED = ('N4', 'SCALE', 'CA', 'KL-P', 'EN-S')
MISSING = ('EN-F', 'EN-COV', 'EN-F4')

def read(path):
    return json.loads(Path(path).read_text())

def verify_endpoint(directory, arm, *, ids, W0, WN, context_tokens):
    root=Path(directory)/'arms'/arm
    seal=read(root/'selection-seal.json');ledger=read(root/'selection-ledger.json')
    saved=torch.load(root/'final-L4.pt',weights_only=True,map_location='cpu',mmap=True)
    weight=saved['weight']
    if weight.dtype!=torch.float32 or tuple(weight.shape)!=(4096,14336) or not torch.isfinite(weight).all():
        raise ValueError('REUSE_ENDPOINT_FP32_SHAPE_FINITE')
    if (saved['case_ids']!=ids or saved['W0_sha']!=W0 or saved['WN_sha']!=WN
        or saved['context_identity']!=context_tokens or saved['history']!=0
        or saved['episode']!=0 or saved['weight_name']!='model.layers.4.mlp.down_proj.weight'):
        raise ValueError('REUSE_ENDPOINT_ENTRY_CONTEXT_REQUEST')
    if (seal['status']!='SELECTION_SEALED' or seal['episode_id']!='b001' or seal['endpoint_id']!=arm
        or seal['request_order_sha256']!=digest(ids) or seal['endpoint_weight_sha256']!=tensor_sha(weight)
        or seal['selection_ledger_sha256']!=member(root/'selection-ledger.json')['sha256']
        or ledger['selected_weight_sha256']!=header_sha(weight)
        or ledger['endpoint']['sha256']!=member(root/'final-L4.pt')['sha256']):
        raise ValueError('REUSE_SELECTION_SEAL_OR_LEDGER')
    return weight,ledger,seal

def space_from(directory, name, allowed):
    root=Path(directory)/'geometry'
    tensors=torch.load(root/(name+'-factors.pt'),weights_only=True,map_location='cpu',mmap=True)
    evidence=read(root/(name+'.json'))
    basis=allowed.basis if tensors['basis'] is None else tensors['basis'].numpy()
    space=RightSpace(basis,tensors['blocked'].numpy(),tensors['status'],evidence)
    got=space.receipt()
    for key in ('basis_sha256','blocked_sha256','status','dimension'):
        if got[key]!=evidence[key]:raise ValueError('REUSE_GEOMETRY_'+key)
    return space

def validation_binding(lock, episode):
    if (episode!=0 or lock['instruction_id']!='ODEEDIT-S06-ENFC-SINGLE-BATCH-M-RESUME-SH1-V1'
        or lock['allowed_stages']!=['M_B001_ONLY'] or lock['native_fit_new_allowed'] is not False
        or lock['runtime_node']!='devbox' or lock['full_numerical_validation']!='NOT_ESTABLISHED'
        or lock['storage_waiver_inherited'] is not False):
        raise ValueError('S1_B001_SCOPE')
    evidence=read(lock['technical_evidence']['path'])
    if member(lock['technical_evidence']['path'])!=lock['technical_evidence']:
        raise ValueError('SOURCE_SKIP_T_EVIDENCE_CHANGED')
    if evidence['status']!='SKIPPED_USER_DIRECTED':raise ValueError('T_SKIP_STATUS')
    # Runtime path rebinding changes the teacher manifest SHA, never its payload.
    evidence['identity']['teacher']=lock['teacher_manifest']
    return evidence,False,True
