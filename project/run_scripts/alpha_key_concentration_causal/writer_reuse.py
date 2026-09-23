"""Restore completed diagnostic branches in RAM, never refit/re-evaluate them.

Approved input checkpoint + original FP32 deltas/history keys are applied in
native order, checking every stored pre/post hash. This is not a saved resume
checkpoint. Missing exact post-z RNG is disclosed: the pinned zero-initialized
Adam/eval native target source has no stochastic operation, so entry RNG is
used, with source binding, rather than claiming an observed post-z snapshot.
"""
import copy
import hashlib
import inspect
import json
import marshal
from pathlib import Path

from .common import digest, file_sha, rng_get, save, tensor_sha
from .native_writer import _request_binding, _rng_sha


def verify_members(manifest):
    for row in manifest['members']:
        p=Path(row['path'])
        assert p.is_file() and not p.is_symlink(), 'REUSE_MEMBER_MISSING_OR_SYMLINK'
        assert p.stat().st_size==row['bytes'] and file_sha(p)==row['sha256'], 'REUSE_MEMBER_CORRUPTED'


def link_branch(source, destination):
    """Leaf-file links only: no mutable destination directory aliases."""
    source=Path(source);destination=Path(destination)
    for p in sorted(source.rglob('*')):
        assert not p.is_symlink(), 'REUSE_SOURCE_SYMLINK_FORBIDDEN'
        if p.is_file():
            relative=p.relative_to(source)
            if relative==Path('sham-control.json'):
                relative=Path('historical-sham-control.json')
            q=destination/relative;q.parent.mkdir(parents=True,exist_ok=True)
            q.symlink_to(p)


def restore_completed_branch(rt, entry_state, manifest, branch, out, pre_states):
    import torch
    from .writer_runner import _stage_snapshot
    source=Path(manifest['writers'])/'W050'/branch
    terminal=json.loads((source/'write/terminal.json').read_text())
    assert terminal['status']=='COMPLETED' and terminal['branch']==branch
    assert terminal['counters']['native_solves']==terminal['counters']['history_appends']==5
    saved=json.loads((source/'write/entry-binding.json').read_text())
    normalized,requests_sha=_request_binding(rt.current_requests(50))
    original=rt.native.apply_AlphaEdit_to_model
    binding=dict(entry=rt.signature(),requests_sha256=requests_sha,
        hparams_sha256=digest(vars(rt.hp)),contexts_sha256=digest(rt.contexts),
        native_code_sha256=hashlib.sha256(marshal.dumps(original.__code__)).hexdigest(),
        entry_rng_sha256=_rng_sha(rng_get()),z_layer=8)
    assert binding==saved['binding'] and digest(binding)==saved['binding_sha256'], 'REUSE_ENTRY_CONFIG_RNG_MISMATCH'
    assert rt.model.training is False, 'REUSE_REQUIRES_EVAL_MODEL'
    native_z_source=Path(inspect.getfile(rt.native.compute_z))
    assert file_sha(native_z_source)==manifest['compute_z_source']['sha256'], 'REUSED_Z_SOURCE_DRIFT'
    zpath=Path(manifest['writers'])/'W050/NATIVE/write/native-z.pt'
    z=torch.load(zpath,map_location='cpu',weights_only=True,mmap=True)
    assert z['binding_sha256']==saved['binding_sha256']
    assert z['case_ids']==[r['case_id'] for r in normalized]
    assert z['targets'].shape==(rt.weights[8].shape[0],100) and torch.isfinite(z['targets']).all()
    z=dict(z,binding=binding,rng_after_z=rng_get(),
           artifact=dict(path=str(zpath),bytes=zpath.stat().st_size,sha256=file_sha(zpath)))
    factors={};keys={};checks=[]
    for layer in (4,5,6,7,8):
        f=json.loads((source/f'write/L{layer}-factor-receipt.json').read_text())
        nf=torch.load(source/f'write/L{layer}-native-factors.pt',map_location='cpu',weights_only=True,mmap=True)
        df=torch.load(source/f'write/L{layer}-diagnostic-factors.pt',map_location='cpu',weights_only=True,mmap=True)
        assert nf['physical_layer']==df['physical_layer']==layer
        assert nf['request_ids']==z['case_ids'] and nf['divisor']==9-layer
        for name in ('K','R','delta'):
            assert torch.isfinite(nf[name]).all() and nf[name].dtype==torch.float32
            assert tensor_sha(nf[name])==f[name+'_sha256'], 'REUSE_FACTOR_TENSOR_DRIFT'
        assert tensor_sha(df['C'])==f['C_sha256']
        for name in ('C','G','H','response_gain'):
            assert torch.isfinite(df[name]).all(), 'REUSE_DIAGNOSTIC_NONFINITE'
        f.update(nf);f.update(df)
        f['pre_weight']=rt.weights[layer].detach().cpu().clone()
        assert tensor_sha(f['pre_weight'])==f['pre_weight_sha256'],'REUSE_PRE_WEIGHT_MISMATCH'
        with torch.no_grad():
            rt.weights[layer].copy_(rt.weights[layer]+f['delta'].to(rt.weights[layer].device))
        assert tensor_sha(rt.weights[layer])==f['post_weight_sha256'],'REUSE_POST_WEIGHT_MISMATCH'
        factors[layer]=f;checks.append(dict(layer=layer,weight_pre_post_exact=True))
        if branch=='NATIVE' and layer in (4,5):
            pre_states[layer+1]=_stage_snapshot(rt,entry_state['m'])
    for row in terminal['history']:
        layer=row['layer'];index=layer-4
        assert row['append_completed'] and tensor_sha(rt.M[index])==row['pre_M_sha256']
        key=torch.load(source/f'write/L{layer}-history-keys.pt',map_location='cpu',weights_only=True,mmap=True)
        assert key['case_ids']==z['case_ids'] and tensor_sha(key['K'])==row['key_sha256']
        rt.M[index]+=key['K']@key['K'].T
        assert tensor_sha(rt.M[index])==row['post_M_sha256'],'REUSE_HISTORY_POST_MISMATCH'
        keys[layer]=key['K']
    observed=json.loads((source/'stages/history/observation.json').read_text())
    assert rt.signature()==observed['seal'],'REUSE_SELECTED_ENDPOINT_SEAL_MISMATCH'
    rt.assert_nonselected()
    link_branch(source,out)
    save(Path(out)/'reuse-restoration.json',dict(status='EXACT_WEIGHT_HISTORY_RECONSTRUCTION',
        original_root=str(source),original_job_id='52565',checks=checks,
        observed_final_seal=observed['seal'],new_fit=0,new_evaluation=0,new_persistent_history_append=0,
        RAM_reconstruction_history_appends=5,post_z_rng_snapshot='NOT_RECORDED',
        RNG_reuse_basis='Pinned compute_z zeros/Adam/eval source, no stochastic operations; entry RNG reused',
        compute_z_source=manifest['compute_z_source'],new_resume_checkpoint_saved=False))
    return dict(z=z,factors=factors,receipts=terminal,finalkeys=keys)
