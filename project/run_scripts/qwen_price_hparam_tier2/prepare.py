"""CPU/tokenizer-only fixed held-out slice and source/config binding."""
import argparse
import json
from pathlib import Path
import torch
from transformers import AutoTokenizer
from official.ours.config import resolve,profile,plain,PRICE_KEYS
from official.ours.core.jlz_realization.inputs import CounterFactAdapter
from project.run_scripts.jlz_realization.inputs import CounterFactAdapter as PreviousAdapter
from project.run_scripts.jlz_realization.common import digest,member,require
from project.run_scripts.jlz_interference_l1.cap_storage import write,guard
from scripts.fixed_counterfact import load_prefix
from . import TASK,NONCE

OLD=Path('/data/janghj/ODE-edit/local/qwen-ours-m1-2k-20261008/preparation-v1')
DISPATCH=Path('/data/janghj/ODE-edit/local/qwen-hparam-tier2-dispatch-20261009')
ARMS=('Q0','Q1-lamN0001','Q2-beta150','Q3-beta250','Q4-beta400',
      'Q5-beta400-lamN05','Q6-beta400-c075','Q7-native')
RESERVE_BYTES=12*1024**3

def prepare(out):
    out=Path(out);guard(out,RESERVE_BYTES)
    original=json.loads((OLD/'config.json').read_text())['cells']['QWEN_M1_CAP075']
    manifest=json.loads((DISPATCH/'manifest.json').read_text())
    all_records=load_prefix(Path(original['stream']).parent,2500)
    train,held=all_records[:2000],all_records[2000:2500]
    ids=lambda rows:[r['case_id'] for r in rows]
    req=lambda rows:{digest(r['requested_rewrite']) for r in rows}
    require(not set(ids(train))&set(ids(held)),'HELDOUT_CASE_OVERLAP')
    require(not req(train)&req(held),'HELDOUT_REQUEST_OVERLAP')
    tokenizer=AutoTokenizer.from_pretrained(original['model'],local_files_only=True)
    tokenizer.pad_token=tokenizer.eos_token;tokenizer.padding_side='right'
    contexts=json.loads(Path(original['contexts']).read_text())
    bench=CounterFactAdapter(tokenizer,contexts);previous=PreviousAdapter(tokenizer,contexts)
    packs=[];affected=[]
    for i in range(5):
        records=held[i*100:(i+1)*100];p=bench.prepare(records);q=previous.prepare(records)
        require(p['identity']==q['identity'],'OFFICIAL_NATIVE_PACK_IDENTITY')
        require(torch.equal(p['targets'],q['targets']),'OFFICIAL_NATIVE_TARGETS')
        for k in p['tokens']:require(torch.equal(p['tokens'][k],q['tokens'][k]),'OFFICIAL_NATIVE_TOKENS')
        require(p['lookup']==q['lookup'],'OFFICIAL_NATIVE_LOOKUP')
        for owner,row in enumerate(p['canonical_rows']):
            if p['lookup'][row]==0:
                affected.append(dict(batch=i+1,owner=owner,case_id=records[owner]['case_id'],
                    prefix_count=5,extra_forward=0))
        packs.append(dict(batch=i+1,identity=p['identity'],ids=ids(records)))
    smoke=bench.prepare(train[:100])
    require(smoke['identity']==original['packs'][0]['identity'],'SMOKE_EVAL_B1_PACK')
    observer=[]
    for record in held:
        rewrite=record['requested_rewrite']
        for kind,prompts in bench.panels(record).items():
            for index,prompt in enumerate(prompts):
                row=dict(case_id=record['case_id'],kind=kind,prompt_index=index,
                    identity=digest([record['case_id'],kind,index,prompt,rewrite['target_new']['str'],rewrite['target_true']['str']]))
                for label in ('new','true'):
                    pt,tt=bench.evaluation_ids(prompt,rewrite['target_'+label]['str'])
                    row[label+'_token_identity']=digest([pt,tt])
                observer.append(row)
    write(out/'observer-identity.json',dict(rows=observer),limit=4*1024**2)
    base=original['arm_profiles']['QWEN_M1_CAP075']
    # Remove only obsolete duplicate knob aliases; all runtime axes retain their bytes/values.
    runtime={k:v for k,v in base.items() if k not in PRICE_KEYS|{'beta_max_native_scale',
        'clamp_factor','kl_factor','norm_factor','K_eval','eligible_layers','anchor_layer','nll_layer','lambda_C'}}
    resolved={arm:plain(profile(resolve('qwen25','qwen25-'+arm),runtime)) for arm in ARMS}
    write(out/'resolved-arms.json',resolved,limit=1024**2)
    rules=dict(Q7_extreme_N_collapse=dict(matched_W0_current_NS_loss_pp_exclusive=5.0,
        reason='Predeclared reference-only exclusion at more than five percentage points locality loss; ordinary admission remains 1.5pp.'),
        tier1_normal=dict(NS_loss_pp_max=1.5,PS_improvement_over_heldout_Q0_strict=True,candidates_min=2,candidates_max=3,
            never_fill_with_failed=True),
        conditional=dict(parent='Q4-beta400',last_shared_active_fraction_below=.8,
            name='Q4-beta400-lamN0',only_override={'lambda_N':0.}),
        tier2=dict(reference='Q0',cold_B1_B5_rerun_if_continuation_unproven=True),
        tier3_recommendation=dict(RS_min_pct=99.,NS_loss_pp_max=1.2,numeric_projection_failure_count=0,
            maximize='PS',no_pass='NO_PASS',submit=False))
    write(out/'selection-rules.json',rules)
    reference=OLD/'QWEN_M1_CAP075/batch-01/writer/commit.json'
    require(reference.is_file(),'61598_ACTUAL_B1_MISSING')
    result=dict(task_id=TASK,instruction_id=NONCE,stage='CPU_INPUT_READY_GPU_NOT_SUBMITTED',
        parent_config=member(OLD/'config.json'),dispatch_manifest=member(DISPATCH/'manifest.json'),
        stream=member(original['stream']),contexts=member(original['contexts']),model=original['model'],
        slice_start=2000,slice_stop=2500,cohort_role='heldout_tuning',records=500,
        slice_identity=digest(held),ordered_ids_sha256=digest(ids(held)),
        eval_first2k_case_overlap=0,eval_first2k_request_overlap=0,
        packs=packs,lookup_zero=affected,smoke_pack=dict(identity=smoke['identity'],ids=ids(train[:100])),
        smoke_reference=member(reference),resolved=member(out/'resolved-arms.json'),
        selection_rules=member(out/'selection-rules.json'),observer_identity=member(out/'observer-identity.json'),
        w0_status='NOT_MEASURED_HELDOUT500',reserve_bytes=RESERVE_BYTES,
        no_model_load=True,no_GPU=True,new_job_ids=[])
    write(out/'preparation.json',result)
    print(json.dumps(dict(stage=result['stage'],packs=len(packs),lookup_zero=len(affected),
        slice_identity=result['slice_identity'],overlap=0,job_ids=[])))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);prepare(p.parse_args().out)
