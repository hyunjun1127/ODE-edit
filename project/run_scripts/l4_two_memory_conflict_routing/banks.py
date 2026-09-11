"""Outcome-blind active versions, candidate hashing, and fixed protection packing."""
import hashlib
import json
import math
import unicodedata
from .identity import digest

def fact(record):
    rw=record['requested_rewrite']
    return (' '.join(unicodedata.normalize('NFC',rw['subject']).split()),str(rw['relation_id']))

def latest(records,ordinals):
    table={}
    for i in ordinals:table[fact(records[i])]=i
    return sorted(table.values())

def rank_key(entry,kind,record,ordinal):
    data=dict(seed=20260911,entry_identity=entry,bank_kind=kind,
              fact_identity=list(fact(record)),version_ordinal=ordinal)
    raw=json.dumps(data,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode('utf-8')
    return hashlib.sha256(raw).hexdigest()

def candidate_inventory(records,offset,current_raw,panel,entry):
    effective=latest(records,current_raw)
    seen=latest(records,range(offset));current_facts={fact(records[i]) for i in effective}
    heldout={i for name,ii in panel['panels'].items() if not name.startswith('Current') for i in ii}
    heldout_facts={fact(records[i]) for i in heldout}
    all_eval=heldout|set(current_raw)
    all_eval_facts={fact(records[i]) for i in all_eval}
    historical_facts={fact(r) for r in records[:offset]}
    past=[i for i in seen if fact(records[i]) not in current_facts|heldout_facts]
    base=[i for i in latest(records,range(len(records)))
          if fact(records[i]) not in historical_facts|current_facts|all_eval_facts]
    def ordered(pool,kind):return sorted(pool,key=lambda i:(rank_key(entry,kind,records[i],i),i))
    pc=ordered(past,'Past')[:512];bc=ordered(base,'Base')[:512]
    audit=ordered([i for i in base if i not in set(bc)],'BaseAudit')[:128]
    return dict(entry_identity=entry,current_raw=list(current_raw),current_effective=effective,
                retired_seen=sorted(set(range(offset))-set(seen)),seen_active=seen,
                past_candidates=pc,base_candidates=bc,base_audit=audit,
                past_eligible=len(past),base_eligible=len(base),
                same_batch_retired=sorted(set(current_raw)-set(effective)),
                hash_encoding='UTF8 canonical JSON sort_keys=True ensure_ascii=False compact',
                fact_rule='NFC subject; split/join whitespace; relation_id; latest ordinal wins',
                controller_loss_based_selection=0)

def select(candidates,scores):
    if len(candidates)!=len(scores) or not all(math.isfinite(float(s)) for s in scores):
        raise ValueError('INVALID_STRUCTURAL_SELECTION_SCORES')
    n=min(128,len(candidates));first=(n+1)//2
    indices=list(range(first))+sorted(range(first,len(candidates)),key=lambda j:(-float(scores[j]),j))[:n-first]
    return [candidates[j] for j in indices]

def protection(records,indices,kind,tok):
    # Import through the existing namespace helper, avoiding eager unrelated code.
    from project.run_scripts.single_layer_cumulative_risk.evaluation import evaluator
    ev=evaluator();out=[]
    for i in indices:
        r=records[i];rw=r['requested_rewrite']
        prompts=[rw['prompt'].format(rw['subject'])]
        if kind=='Past' and r['paraphrase_prompts']:prompts+=r['paraphrase_prompts'][:1]
        target=rw['target_new' if kind=='Past' else 'target_true']['str']
        for ci,prompt in enumerate(prompts):
            pp,tt=ev._encode_pair(tok,ev.PromptTarget(r['case_id'],kind,ci,prompt,target))
            if kind!='Past':tt=tt[:8]
            if not pp or not tt:raise ValueError('EMPTY_PROTECTION_TOKENS')
            ids=(pp+tt)[:-1];positions=list(range(len(pp)-1,len(ids)))
            if len(positions)!=len(tt):raise ValueError('PROTECTION_POSITION_MISMATCH')
            out.append(dict(ordinal=i,case_id=r['case_id'],context_index=ci,kind=kind,
                input_ids=ids,target_ids=tt,positions=positions,context_count=len(prompts),
                context_weight=1/(len(indices)*len(prompts)),
                token_weight=1/(len(indices)*len(prompts)*len(tt)),
                identity=digest([i,kind,ci,ids,positions,tt])))
    return out
