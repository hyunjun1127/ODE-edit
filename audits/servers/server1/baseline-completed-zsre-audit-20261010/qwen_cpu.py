"""Original-source CPU factual audit; no model/CP load or scheduler queries."""
import sys
from pathlib import Path
B=Path('/mnt/raid5/janghj/ODE-edit/local/qwen-baseline-mask-cold-rerun-20261010/server1')
sys.path.insert(0,str(B/'registration-r1/source'))
from official.runners.server1.common import read,verify,member
from official.runners.server1.audit import audit_factual
from official.experiments.prepare import digest,write_new
from official.baselines import registry
from transformers import AutoTokenizer
s=read(B/'registration-r1/submission.json');lock=read(verify(s['execution_lock']))
for m in lock['source_members']:verify(m)
cm=s['jobs']['qwen25']['config'];c=read(verify(cm));folder=Path(c['output'])
done=read(folder/'COMPLETE.json');identity=done['identity']
assert done['requests']==2000 and done['native_apply_calls']==done['history_appends_per_layer']==20
assert identity['code_commit']==s['source'] and identity['config_sha256']==c['config_sha256']
records=read(verify(c['stream_member']));records=records['records'] if isinstance(records,dict) else records
assert len(records)==2000 and digest(records)==c['stream_sha256']
previous=None
for b in range(1,21):
    commit=read(folder/'commits'/f'batch-{b:02d}.json');e=commit['cursor']['edit']
    assert commit['identity']==identity and commit['batch']==commit['cursor']['completed_batch']==b
    assert e['requests']==100 and e['history_appends_per_layer']==1
    assert e['request_sha256']==digest(registry.requests(records[(b-1)*100:b*100],'MEMIT_FE_HISTORY','qwen25'))
    assert e['before']['successful_calls']==b-1 and e['after']['successful_calls']==b
    if previous is not None:assert e['before']==previous
    previous=e['after']
assert done['final_cursor']==commit['cursor']
pointer=read(folder/'checkpoint/latest.json');assert pointer==commit['checkpoint'] and pointer['final_W20']
cp=member(folder/'checkpoint'/pointer['file']);assert cp['sha256']==pointer['sha256']
raw=read(verify(commit['cursor']['factual']));a=read(verify(c['assets']))
tok=AutoTokenizer.from_pretrained(a['model']['snapshot'],local_files_only=True);tok.pad_token=tok.eos_token;tok.padding_side='right'
external=dict(identity,instruction_id=c['instruction_id'],method=c['method'],model=c['model'])
proof=audit_factual(raw,records,'cf',tok,external)
result=dict(model='qwen25',method='MEMIT_FE_HISTORY',dataset='cf',job_id='62061',job_name=s['jobs']['qwen25']['name'],
    status='W20_FACTUAL_CPU_VERIFIED',complete=True,metrics=raw['summary'],source=s['source'],config=cm,
    endpoint=commit['cursor']['factual'],terminal=member(folder/'COMPLETE.json'),checkpoint=cp,
    checkpoint_receipt=member(folder/'checkpoint/latest.json'),identity=identity,commits=20,
    ordered_case_ids_sha256=digest([r['case_id'] for r in records]),raw_audit=proof,
    generator=member(Path(lock['source_directory'])/'official/baselines/easyedit/util/generate.py'),
    generation='DEFERRED',separate_history_variant=True,old_wrong_context_61975_excluded=True)
write_new(Path(__file__).parent/'qwen-history.json',result)
print(result['metrics'])
