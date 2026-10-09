"""CPU-only check of newly completed Llama history using its original factual code."""
import sys
from pathlib import Path
B=Path('/mnt/raid5/janghj/ODE-edit/local/official-baselines/server1')
sys.path.insert(0,str(B/'memit-fe-history-three-model-2k/registration-r1/source'))
from official.runners.server1.common import read,verify,member
from official.runners.server1.audit import audit_factual
from official.experiments.prepare import digest,write_new
from transformers import AutoTokenizer

cfg_path=B/'memit-fe-history-three-model-2k/preparation-r1/configs/llama3.json'
c=read(cfg_path);folder=Path(c['output']);done=read(folder/'COMPLETE.json')
assert done['requests']==2000 and done['native_apply_calls']==done['history_appends_per_layer']==20
commit=read(folder/'commits/batch-20.json');assert commit['cursor']==done['final_cursor']
raw_member=commit['cursor']['factual'];raw=read(verify(raw_member))
records=read(verify(c['stream_member']));records=records['records'] if isinstance(records,dict) else records
a=read(verify(c['assets']));t=AutoTokenizer.from_pretrained(a['model']['snapshot'],local_files_only=True)
t.pad_token=t.eos_token;t.padding_side='right'
external=dict(done['identity'],instruction_id=c['instruction_id'],method=c['method'],model=c['model'])
proof=audit_factual(raw,records,'cf',t,external)
r=dict(model='llama3',dataset='cf',method='MEMIT_FE_HISTORY',job_id='61928',
    job_name='official-s1-cf-llama3-memit-fe-history',complete=True,status='W20_FACTUAL_CPU_VERIFIED',
    summary=raw['summary'],identity=done['identity'],source=done['identity']['code_commit'],config=member(cfg_path),
    endpoint=raw_member,terminal=member(folder/'COMPLETE.json'),raw_audit=proof,
    ordered_case_ids_sha256=digest([x['case_id'] for x in records]),generation='DEFERRED',
    variant_separate_from_native_FE=True,commits=20)
write_new(B/'flucon-paper-scale-20261010/llama-history-refresh.json',r)
print(r['summary'])
