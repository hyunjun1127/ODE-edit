"""Allowlisted historical B020 selected-W adapter, never a training resume.

The historical payload/metadata keep their old schema and source. A separate
consumer binding names today's generation protocol/runtime, without claiming
old-runtime numerical parity or relabelling historical observations.
"""
import argparse,hashlib
from pathlib import Path
import torch
from official.experiments.prepare import digest,write_new
from official.runners.server1.common import read,member,verify
from .flucon_prepare import BASE,QWEN,REFERENCE,normalize_assets
from .flucon_eval import AUTHORITY,tracking_values
from official.runners.server1.flucon_eval import INSTRUCTION
from official.runners.server1.fe_history_prepare import runtime
from official.evaluation.generation.assets import load_assets

def legacy_hash(t):
    # Exact historical integrity.py tensor envelope (dtype/shape + raw bytes).
    x=t.detach().contiguous().cpu()
    h=hashlib.sha256(str((str(x.dtype),list(x.shape))).encode())
    h.update(memoryview(x.view(torch.uint8).numpy().reshape(-1)))
    return h.hexdigest()

def validate_payload(payload,binding):
    assert set(payload)=={'weights','cache_c','metadata'}
    m=payload['metadata'];assert m['batch']==20 and len(m['seen_ids'])==2000
    assert digest(m['seen_ids'])==binding['ordered_case_ids_sha256']
    for k in ('lock_sha256','source','base_model_revision','method','sample_root'):
        assert m[k]==binding['metadata'][k]
    assert m['state']['weights']==binding['selected_state_sha256']
    assert set(payload['weights'])==set(binding['selected_state_sha256'])
    for k,w in payload['weights'].items():
        assert w.dtype==torch.float32 and bool(torch.isfinite(w).all())
        assert legacy_hash(w)==binding['selected_state_sha256'][k]

def restore_historical(model,original):
    binding=read(verify(original['pointer']))
    payload=torch.load(verify(original['checkpoint']),map_location='cpu',mmap=True,weights_only=False)
    validate_payload(payload,binding)
    params=dict(model.named_parameters());selected=payload['weights']
    assert set(selected)=={original['hparams']['rewrite_module_tmp'].format(l)+'.weight' for l in original['hparams']['layers']}
    for k,w in selected.items():assert k in params and w.shape==params[k].shape and params[k].dtype==torch.float32
    before={k:(v.data_ptr(),v._version) for k,v in params.items() if k not in selected}
    with torch.no_grad():
        for k,w in selected.items():
            params[k].copy_(w);assert legacy_hash(params[k])==binding['selected_state_sha256'][k]
    assert before=={k:(v.data_ptr(),v._version) for k,v in params.items() if k not in selected}
    return dict(batch=20,actual_edit_calls=0,historical_payload_identity=binding['metadata'],
        selected_weights_restored=len(selected),nonselected_unchanged=True,original_H_context_RNG_preserved_in_CP=True,
        history_restored=False,reason='evaluation only; no editing state consumed',old_runtime_parity='NOT_CLAIMED')

def prepare(reference_path,output,standard_preparation):
    output=Path(output);assert not output.exists()
    handoff=read(reference_path);assert handoff['destination_owner']=='server2'
    closure=BASE/'checkpoint-archives/server4-migration-20260911-v1/new177-v1/closure/local'
    refs=load_assets(REFERENCE);configs=[];rows=[];assets=None
    for candidate in handoff['members']:
        row=dict(job_id=candidate['job_id'],method=candidate['method'],model='llama3')
        try:
            blue=candidate['method']=='ALPHAEDIT_BLUE'
            root=closure/('blue-lifelong-b100x100/attempt-checkpoint-r2' if blue else 'fixed10k-native-baselines/attempt-v1')
            lockpath=root/'execution.lock.json';lock=read(lockpath);lm=member(lockpath)
            cp=member(candidate['path']);assert cp['sha256']==candidate['recorded_sha256'] and cp['bytes']==candidate['bytes']
            payload=torch.load(cp['path'],map_location='cpu',mmap=True,weights_only=False);meta=payload['metadata']
            assert meta['lock_sha256']==lm['sha256'] and meta['source']==lock['local_source_sha256']
            assert member(root/'local-source.tar')['sha256']==meta['source']
            assert meta['base_model_revision']==lock['revision']==handoff['revision']
            stream=member(QWEN/'streams/cf-stream.json');records=read(verify(stream))
            assert meta['seen_ids']==[r['case_id'] for r in records]
            cell=next(c for c in lock['cells'] if c['cell']==(2 if candidate['method']=='MEMIT' else 1))
            if blue:
                hp_path=BASE/'blue-checkpoint-downstream/20260909-v1/imports/initial6-v1/source/closure/095-AlphaEdit-ORIGINAL.json'
                assert member(hp_path)['sha256']==cell['config_sha256'];hp=read(hp_path)
            else:hp=cell['hparams']
            original_members=[m for m in lock['members'] if m['path'].startswith(lock['snapshot']+'/')]
            assert original_members
            snapshot=Path(lock['snapshot'].replace('/data/janghj/','/mnt/raid5/janghj/',1))
            local_members=[dict(m,path=str(snapshot/Path(m['path']).name)) for m in original_members]
            tokenizer_members={Path(m['path']).name:m['sha256'] for m in original_members
                if not Path(m['path']).name.startswith('model') and Path(m['path']).name not in ('config.json','generation_config.json')}
            if assets is None:assets=normalize_assets(output,'llama3',snapshot,local_members)
            else:
                normalized=read(verify(assets))['model']['members']
                assert {Path(m['path']).name:m['sha256'] for m in normalized}=={Path(m['path']).name:m['sha256'] for m in original_members}
            binding=dict(schema='historical-B020-generation-consumer-binding-v1',
                checkpoint=cp,original_lock=lm,metadata={k:meta[k] for k in ('lock_sha256','source','base_model_revision','method','sample_root')},
                selected_state_sha256=meta['state']['weights'],ordered_case_ids_sha256=digest(meta['seen_ids']),
                old_runtime_source=lock['dependencies'],evaluation_runtime=runtime(),old_runtime_parity='NOT_CLAIMED')
            validate_payload(payload,binding);del payload
            receipt=output/'bindings'/f"{candidate['job_id']}.json";write_new(receipt,binding)
            identity=dict(schema='historical-consumer-binding-not-original-official-identity',model_revision=lock['revision'],
                tokenizer_sha256=digest(tokenizer_members),original_source=meta['source'],original_lock_sha256=lm['sha256'],
                original_config_sha256=cell['config_sha256'],original_checkpoint_sha256=cp['sha256'])
            original=dict(job_id=candidate['job_id'],method=candidate['method'],identity=identity,checkpoint=cp,
                checkpoint_schema='historical-W-method-state',pointer=member(receipt),terminal=member(receipt),
                config=lm,hparams=hp,ordered_case_ids_sha256=binding['ordered_case_ids_sha256'],
                expected_history=blue or candidate['method']=='ALPHAEDIT')
            key='llama3-'+candidate['method'].lower()+'-'+candidate['job_id']
            c=dict(instruction=INSTRUCTION,registration_authority=AUTHORITY,key=key,model='llama3',original=original,
                assets=assets,asset_schema='history',stream=stream,reference=member(REFERENCE),reference_identity=refs.sha,
                evaluator=member(Path(__file__).parents[2]/'evaluation/generation/native_observer.py'),runtime=runtime(),
                output=str(output/'runs'/key),tracking_env_file='/mnt/raid5/janghj/ODE-edit/servers/local/wandb.env')
            c['config_sha256']=digest(c);tracking_values(c,'a'*40)
            path=output/'configs'/f'{key}.json';write_new(path,c);configs.append(member(path))
            row.update(status='VERIFIED_HISTORICAL_B020_2000',checkpoint=cp,binding=member(receipt),config=member(path))
        except Exception as e:row.update(status='BLOCKED',error_type=type(e).__name__,reason=str(e))
        rows.append(row);print(row['job_id'],row['status'],row.get('reason',''),flush=True)
    standard=read(standard_preparation)
    write_new(output/'preparation.json',dict(standard,configs=standard['configs']+configs,
        inventory=standard['inventory']+rows,historical_reference=member(reference_path)))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--reference',required=True);p.add_argument('--output',required=True)
    p.add_argument('--standard-preparation',required=True);a=p.parse_args();prepare(a.reference,a.output,a.standard_preparation)
