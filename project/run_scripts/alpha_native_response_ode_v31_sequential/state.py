"""Checks around the existing captured-endpoint transaction, plus real snapshots."""
import os
from pathlib import Path
import torch
from project.run_scripts.ordered_response_barrier_ode.fp32_overlay import tensor_set_sha256,tensor_sha256
from project.run_scripts.native_response_ode_v31.provenance import save
from project.run_scripts.ordered_response_barrier_ode.preflight import sha256_file


def commit_checked(family,endpoint):
    if endpoint['history_append_count']!=1:raise RuntimeError('HISTORY_APPEND_ONCE')
    if tensor_set_sha256(family.parameters)!=family.w0_sha256:raise RuntimeError('TRANSACTION_RESTORE_WEIGHT')
    if family.method_state_identity()!=family._prepared_method_state_identity:raise RuntimeError('TRANSACTION_RESTORE_HISTORY')
    captured_M=tensor_sha256(family._captured_endpoint_method_state)
    result=family.commit_captured_endpoint(expected_sha256=endpoint['selected_weight_endpoint_sha256'])
    if tensor_sha256(family.module.cache_c)!=captured_M:raise RuntimeError('CAPTURE_COMMIT_HISTORY_BYTES')
    if any(result[k]!=0 for k in ('writer_recompute_count','fixed_z_recompute_count','model_forward_count','evaluator_count')):
        raise RuntimeError('COMMIT_RECOMPUTATION')
    return dict(result,committed_M_content_sha256=captured_M,history_append_count=1)


def check_entry(family,previous):
    if previous is not None:
        if family.w0_sha256!=previous['committed_weight_sha256']:raise RuntimeError('SEQUENTIAL_WEIGHT_CONTINUITY')
        if tensor_sha256(family.module.cache_c)!=previous['committed_M_content_sha256']:raise RuntimeError('SEQUENTIAL_HISTORY_CONTINUITY')


def checkpoint(path,family,commit,metadata):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True,mode=0o700)
    payload=dict(weights={k:v.detach().cpu().clone() for k,v in family.parameters.items()},
        alpha_cache=family.module.cache_c.detach().cpu().clone(),cache_c_new=family.module.cache_c_new,
        commit=commit,metadata=metadata)
    fd=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
    with os.fdopen(fd,'wb') as f:torch.save(payload,f);f.flush();os.fsync(f.fileno())
    del payload
    restored=torch.load(path,map_location='cpu',weights_only=True)
    if tensor_set_sha256(restored['weights'])!=commit['committed_weight_sha256'] or tensor_sha256(restored['alpha_cache'])!=commit['committed_M_content_sha256']:
        raise RuntimeError('CHECKPOINT_RELOAD_BYTES')
    receipt=dict(path=str(path),bytes=path.stat().st_size,sha256=sha256_file(path),
        selected_weight_sha256=commit['committed_weight_sha256'],M_sha256=commit['committed_M_content_sha256'],
        actual_selected_weights_saved=True,actual_history_saved=True,reload_tensor_identity='PASS',
        full_pretrained_model_saved=False,journal_replay_parity='NOT_TESTED')
    save(path.with_suffix('.receipt.json'),receipt);return receipt
