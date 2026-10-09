"""Bounded read-only actual-context/source audit; no model forward/job mutation."""
import ast
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import subprocess

LOCAL=Path('/mnt/raid5/janghj/ODE-edit/local')
OUT=Path(__file__).resolve().parent
OLD=LOCAL/'qwen-baselines-server2-20261009/registration-native-eval-r2'
NEW=LOCAL/'qwen-baseline-mask-cold-rerun-20261010/registration-r1'
REF=LOCAL/'native-context-import/server3-20261010/job-62101/qwen25'
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def digest(v):return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def write(p,v):
    with p.open('x') as f:json.dump(v,f,indent=2,sort_keys=True)

reference=read(REF/'contexts.json');ready=read(REF/'READY.json')
assert sha(REF/'contexts.json')==ready['context_sha256']=='caf43aaf6e04f8b894f49051cbca4312e51b63ac42466be4edd82a70d7589dea'
assert sha(REF/'context-token-ids.json')==ready['context_tokens_sha256']
assert ready['generator_sha256']=='35506690c41ecb7d59f11660da41dde50338f5a2ba9613735e045a8c7ab98db4'
assert ready['revision']=='a09a35458c702b33eeacc393d103063234e8bc28' and ready['seed']==0

def observed(root,cell,job):
    p=root/'logs'/f'{cell}-{job}.out'
    with p.open() as f:
        for lineno,line in enumerate(f,1):
            if line.startswith('Cached context templates '):
                context=ast.literal_eval(line.removeprefix('Cached context templates '))
                return dict(job_id=job,cell=cell,log_path=str(p),line=lineno,
                    context_line_sha256=hashlib.sha256(line.encode()).hexdigest(),
                    context_canonical_sha256=digest(context),equal_reference_exact=context==reference,
                    source=read(root/'source-lock.json')['code_commit'],
                    config_sha256=read(root/'configs'/f'{cell}.json')['config_sha256']),context
    raise ValueError('ACTUAL_CONTEXT_NOT_RECORDED:'+job)

rows=[];contexts=[]
for root,cell,job in [(OLD,'qwen25-cf-alphaedit_blue','61962'),(OLD,'qwen25-zsre-alphaedit_blue','61964'),
        (NEW,'qwen25-cf-memit','62073'),(NEW,'qwen25-cf-sphere','62079')]:
    row,context=observed(root,cell,job);assert row['equal_reference_exact']
    row['status']='ACTUAL_CONTEXT_MATCHES_NEW_VERIFIED_REFERENCE_KEEP';rows.append(row);contexts.append(context)
old=[]
for cell,job,replacement in [('qwen25-zsre-memit','61956','62081'),('qwen25-zsre-alphaedit','61960','62083'),('qwen25-zsre-memit_fe','61968','62085')]:
    row,context=observed(OLD,cell,job);assert not row['equal_reference_exact']
    row.update(status='OLD_REPETITIVE_CONTEXT_ALREADY_SUPERSEDED',replacement_job=replacement)
    old.append(row)

# Compare token IDs with the published local tokenizer, not model logits.
from transformers import AutoTokenizer
snapshot=read(NEW/'asset-preflight.json')['assets']['model_snapshot']['path']
tok=AutoTokenizer.from_pretrained(snapshot,local_files_only=True)
reference_tokens=read(REF/'context-token-ids.json')
for row,context in zip(rows,contexts):
    tokens=[[tok(x,add_special_tokens=True)['input_ids'] for x in group] for group in context]
    assert tokens==reference_tokens
    row['token_IDs_equal_reference_exact']=True

released=read(NEW/'released.json')
pending=[]
for j in released['jobs']:
    if j['job_id'] not in ('62075','62077','62081','62083','62085','62087'):continue
    config=read(NEW/'configs'/f"{j['cell']}.json")
    assert config['context_generator_sha256']==ready['generator_sha256']
    assert sha(NEW/'source/official/baselines/easyedit/util/generate.py')==ready['generator_sha256']
    pending.append(dict(job_id=j['job_id'],cell=j['cell'],source=j['source'],config_sha256=config['config_sha256'],
        status='REGISTERED_FIXED_SOURCE_CONTEXT_NOT_GENERATED_YET',old_context_injection=False))
assert len(pending)==6
ft_source=LOCAL/'qwen-baselines-server2-20261009/registration-r1/source/official/baselines/easyedit/models/ft/ft_main.py'
tree=ast.parse(ft_source.read_text())
assert not any(isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id in ('get_context_templates','generate_fast') for n in ast.walk(tree))
ids=['61898','61900','61962','61964','62072','62073','62075','62077','62079','62081','62083','62085','62087']
accounting=subprocess.check_output(['sacct','-X','-n','-P','-j',','.join(ids),'--format=JobIDRaw,JobName%100,User,State,NodeList,WorkDir%250'],text=True)
value=dict(observed_at_utc=datetime.now(timezone.utc).isoformat(),reference_job='62101',reference_file_sha256=ready['context_sha256'],
    reference_canonical_sha256=digest(reference),reference_provenance=ready,
    actual_matching=rows,historical_broken=old,pending_fixed=pending,
    FT=dict(jobs=['61898','61900'],status='NO_GENERATED_EDIT_CONTEXT_NATIVE_FT',source_sha256=sha(ft_source)),
    exact_accounting=accounting,additional_rerun_required=False,
    reason='Affected old trajectories already replaced by 62073/75/77/79/81/83/85/87; current observed contexts match new reference exactly.',
    model_forward_calls=0,job_mutations=0,checkpoint_loads=0,old_raw_CP_KEEP=True,raw_context_or_tokens_published=False)
write(OUT/'result.json',value)
print(json.dumps(dict(actual_matching=[r['job_id'] for r in rows],historical_broken=[r['job_id'] for r in old],
    pending_fixed=[r['job_id'] for r in pending],additional_rerun_required=False)))
