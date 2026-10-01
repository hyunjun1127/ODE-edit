"""Canonical NLL-pair observer: no writer feedback and no prompt serialization."""
import math
import time
import unicodedata
import torch
from project.run_scripts.blue_alphaedit_sequential_comparison.evaluation import bind_observation_only_package
bind_observation_only_package()
from project.run_scripts.alphaedit_strength_neutral_barrier.evaluator import PromptTarget, evaluate_pairs
from .state import digest, state_hash, parameter_guard

def active_flags(records):
    """Retain all denominators; mark prior differing-target versions superseded."""
    active = {int(r['case_id']):True for r in records}
    versions = {}
    for record in records:
        rw=record['requested_rewrite']
        key=(unicodedata.normalize('NFC',' '.join(rw['subject'].split())),rw['relation_id'])
        target=rw['target_new'].get('id',rw['target_new']['str'])
        for case,old in versions.get(key,[]):
            if old!=target: active[case]=False
        versions.setdefault(key,[]).append((int(record['case_id']),target))
    return active

def pair_identity(record,kind,index,prompt):
    rw=record['requested_rewrite']
    return digest([int(record['case_id']),kind,index,prompt,rw['target_new']['str'],rw['target_true']['str']])

def reduce_rows(rows):
    result={}
    for kind in ('R','P','N'):
        group=[r for r in rows if r['kind']==kind]
        if not group: continue
        if len({r['identity'] for r in group})!=len(group): raise RuntimeError('DUPLICATE_OBSERVATION')
        for row in group:
            if not all(math.isfinite(row[k]) for k in ('new_nll','true_nll')):
                raise FloatingPointError('NONFINITE_OBSERVATION')
        desired='true' if kind=='N' else 'new'
        successes=[r['true_nll']<r['new_nll'] if kind=='N' else r['new_nll']<r['true_nll'] for r in group]
        correct=sum(r[desired+'_token_correct'] for r in group)
        ntokens=sum(r[desired+'_token_count'] for r in group)
        result[kind]={'numerator':sum(successes),'denominator':len(group),'rate':sum(successes)/len(group),
                      'true_nll_mean':sum(r['true_nll'] for r in group)/len(group),
                      'new_nll_mean':sum(r['new_nll'] for r in group)/len(group),
                      'desired_nll_mean':sum(r[desired+'_nll'] for r in group)/len(group),
                      'desired_token_correct':correct,'desired_token_count':ntokens,
                      'desired_token_micro':correct/ntokens,
                      'desired_prompt_macro':sum(r[desired+'_token_correct']/r[desired+'_token_count'] for r in group)/len(group),
                      'desired_strict_numerator':sum(r[desired+'_strict'] for r in group),
                      'desired_strict_denominator':len(group)}
    return result

@torch.no_grad()
def observe(model,tokenizer,records,history,endpoint,current_start=0,w0=False):
    start=time.monotonic(); before=state_hash(model,history); guard=parameter_guard(model)
    rows=[]; chunks_count=0
    for kind in ('R','P','N'):
        selected=[r for i,r in enumerate(records) if kind!='N' or w0 or endpoint in (5,10) or i>=current_start]
        # Same endpoint's Current is a subset of these rows; never re-evaluate.
        for offset in range(0,len(selected),100):
            pairs_new=[]; pairs_true=[]; identities=[]
            for record in selected[offset:offset+100]:
                rw=record['requested_rewrite']
                prompts=([rw['prompt'].format(rw['subject'])] if kind=='R' else
                         record['paraphrase_prompts'] if kind=='P' else record['neighborhood_prompts'])
                if len(prompts)!={'R':1,'P':2,'N':10}[kind]: raise RuntimeError('PROMPT_DENOMINATOR_MISMATCH')
                for index,prompt in enumerate(prompts):
                    ids=(int(record['case_id']),index,prompt)
                    pairs_new.append(PromptTarget(ids[0],kind,index,prompt,rw['target_new']['str']))
                    pairs_true.append(PromptTarget(ids[0],kind,index,prompt,rw['target_true']['str']))
                    identities.append(pair_identity(record,kind,index,prompt))
            if not pairs_new: continue
            args=dict(device=next(model.parameters()).device,microbatch_size=2)
            a=evaluate_pairs(model,tokenizer,pairs_new,**args)
            b=evaluate_pairs(model,tokenizer,pairs_true,**args)
            chunks_count+=2*math.ceil(len(a)/2)
            for left,right,identity in zip(a,b,identities,strict=True):
                if (left['case_id'],left['prompt_index'],left['prompt'])!=(right['case_id'],right['prompt_index'],right['prompt']):
                    raise RuntimeError('PAIR_IDENTITY_MISMATCH')
                row={'case_id':left['case_id'],'kind':kind,'prompt_index':left['prompt_index'],
                     'identity':identity,'endpoint':endpoint,'new_nll':left['nll'],'true_nll':right['nll']}
                for label,value in (('new',left),('true',right)):
                    row[label+'_token_correct']=sum(value['token_correct'])
                    row[label+'_token_count']=len(value['token_correct'])
                    row[label+'_strict']=value['all_tokens_correct']
                rows.append(row)
            print({'event':'observer_progress','endpoint':endpoint,'kind':kind,'requests_done':offset+len(selected[offset:offset+100])},flush=True)
    flags=active_flags(records)
    for row in rows: row['active_at_endpoint']=flags[row['case_id']]
    if state_hash(model,history)!=before or parameter_guard(model)!=guard:
        raise RuntimeError('OBSERVER_STATE_MUTATION')
    summary=reduce_rows(rows)
    current_ids={r['case_id'] for r in records[current_start:]}
    return {'endpoint':endpoint,'state':before,'requests':len(records),'rows':rows,
            'summary':summary,'current':reduce_rows([r for r in rows if r['case_id'] in current_ids]),
            'seconds':time.monotonic()-start,'forward_microbatches':chunks_count,
            'no_mutation':True,'optimizer_feedback':False}
