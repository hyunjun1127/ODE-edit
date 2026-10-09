"""Read-only raw review under exact frozen runtime imports; no model forward."""
import argparse,datetime,json,sys
from pathlib import Path

p=argparse.ArgumentParser();p.add_argument('--registration',required=True);p.add_argument('--output',required=True)
a=p.parse_args();root=Path(a.registration);sys.path.insert(0,str(root/'source'))
from official.runners.server1.common import read,verify,member,validate_config,bindings
from official.runners.server1.audit import audit_commits,audit_factual
from official.runners.server1 import zsre_run
from official.experiments.prepare import write_new,digest
from transformers import AutoTokenizer
receipt=read(root/'submission.json');lock=read(verify(receipt['execution_lock']))
for m in lock['frozen_source']['members']:verify(m)
rows=[]
for profile in lock['profiles']:
    if profile['mode']!='chain':continue
    folder=Path(profile['output']);terminal=folder/'COMPLETE.json'
    row=dict(model='llama3',dataset=profile['dataset'],method=profile['method'],job_id=receipt['jobs'][profile['key']],
             source=lock['source'],config=profile['config'],complete=False)
    if not terminal.exists():
        row.update(status='NO_W20_TERMINAL',commits=len(list((folder/'commits').glob('batch-*.json'))));rows.append(row);continue
    try:
        cfg=validate_config(read(verify(profile['config'])))
        assets,records,identity,external=bindings(cfg,lock)
        tok=AutoTokenizer.from_pretrained(assets['model']['tokenizer_path'],local_files_only=True)
        tok.pad_token=tok.eos_token;tok.padding_side='right'
        final=read(terminal);assert final['identity']==identity
        if profile['dataset']=='cf':
            assert final['method']==profile['method'] and final['dataset']=='cf'
            assert final['actual_native_apply_calls']==20 and final['actual_applied_requests']==2000
            assert final['scientific_complete'] and final['checkpoint_W20_preserved']
            assert final['W20_generation_status']=='DEFERRED_TO_SAVED_W20_CHECKPOINT'
            reference=None
        else:
            assert final['method']==profile['method'] and final['requests']==2000
            assert final['native_apply_calls']==20 and final['generation_calls']==0 and final['checkpoint_W20_preserved']
            reference=zsre_run.reference(cfg,lock,external)
            audit_factual(reference['evaluation'],records,'zsre',tok,external,reference)
        ledger=audit_commits(folder,profile['method'],identity,20,records=records)
        commit=read(folder/'commits/batch-20.json');endpoint_member=commit['cursor']['factual']
        endpoint=read(verify(endpoint_member))
        result=audit_factual(endpoint,records,profile['dataset'],tok,external,reference)
        assert result['requests']==2000
        pointer=read(folder/'checkpoint/latest.json')
        assert pointer==commit['checkpoint'] and pointer['batch']==20 and pointer['final_W20']
        if profile['dataset']=='cf':assert final['final_cursor']==commit['cursor']
        else:assert final['final_checkpoint']==member(folder/'checkpoint/latest.json')
        scores={k:v for k,v in endpoint['summary'].items() if type(v) in (float,int)}
        row.update(complete=True,status='W20_FACTUAL_CPU_VERIFIED',summary=scores,
            generation='DEFERRED' if profile['dataset']=='cf' else 'NOT_APPLICABLE',
            endpoint=endpoint_member,terminal=member(terminal),checkpoint_metadata=member(folder/'checkpoint/latest.json'),
            checkpoint_SHA_verified=True,identity=identity,ordered_case_ids_sha256=digest([r['case_id'] for r in records]),
            raw_audit=result,commit_audit=ledger)
    except Exception as error:
        row.update(status='CPU_REVIEW_BLOCKED',error_type=type(error).__name__,error=str(error))
    rows.append(row);print(json.dumps({k:row[k] for k in ('job_id','method','dataset','status')}),flush=True)
write_new(a.output,dict(at=datetime.datetime.now(datetime.timezone.utc).isoformat(),rows=rows,
    review_source='FROZEN_RUNTIME_IMPORTS',registration=member(root/'submission.json'),
    actual_model_forward_calls=0,model_weights_deserialized=False,checkpoint_restore=False,
    GPU_resume_equivalence='NOT_RUN_USER_DISABLED',checkpoint_raw_transfer_delete=False))
