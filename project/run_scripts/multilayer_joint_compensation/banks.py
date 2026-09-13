"""Outcome-independent fact/version banks and full-sequence prediction packing."""
import unicodedata
from .contracts import NAMESPACE,digest

def fact(r):
    w=r['requested_rewrite']
    return (' '.join(unicodedata.normalize('NFC',w['subject']).split()),str(w['relation_id']))

def latest(records,ordinals):
    version={}
    for i in ordinals:version[fact(records[i])]=i
    return sorted(version.values())

def hashed(records,ordinals,label):
    return sorted(ordinals,key=lambda i:(digest(dict(namespace=NAMESPACE,label=label,
        fact=list(fact(records[i])),ordinal=i,case_id=records[i]['case_id'])),i))

def select(records,start,current_raw,panel):
    current=latest(records,current_raw);currentfacts={fact(records[i]) for i in current}
    seen=latest(records,range(start));seenfacts={fact(records[i]) for i in seen}
    eval_ids=set(sum(panel['panels'].values(),[]));evalfacts={fact(records[i]) for i in eval_ids}
    pools=dict(Past=[i for i in seen if fact(records[i]) not in currentfacts|evalfacts],
      Base=[i for i in latest(records,range(len(records))) if fact(records[i]) not in seenfacts|currentfacts|evalfacts])
    chosen={};sizes={}
    for kind,pool in pools.items():
        ordered=hashed(records,pool,kind+'-pool');chosen[kind]=ordered[:128];chosen[kind+'Audit']=ordered[128:256]
        sizes[kind]=len(pool)
    keys=list(chosen);overlap=[]
    def strings(indices):
        out=set()
        for i in indices:
            r=records[i];w=r['requested_rewrite']
            out.update([w['prompt'].format(w['subject'])]+r['paraphrase_prompts']+r['neighborhood_prompts'])
        return out
    for j,a in enumerate(keys):
        for b in keys[j+1:]:
            overlap.append(dict(left=a,right=b,fact_overlap=len({fact(records[i]) for i in chosen[a]} & {fact(records[i]) for i in chosen[b]}),
                string_overlap=len(strings(chosen[a])&strings(chosen[b]))))
    return dict(namespace=NAMESPACE,current_raw=list(current_raw),current_effective=current,
      same_batch_retired=sorted(set(current_raw)-set(current)),seen_active=seen,
      retired_seen=sorted(set(range(start))-set(seen)),bank=chosen,eligible=sizes,overlaps=overlap,
      nested={'B1':current[:1],'B7':current[:7],'B100':current},
      raw_record_sha={str(i):digest(records[i]) for i in sorted(set(current+sum(chosen.values(),[])))},
      selection='canonical JSON SHA256, ascending; ordinal tie; no outcomes or native influence',
      current_fact_overlap={k:len(currentfacts&{fact(records[i]) for i in v}) for k,v in chosen.items()})

def current_rows(records,indices,contexts,tok):
    flat=[c for group in contexts for c in group];out=[]
    for i in indices:
        w=records[i]['requested_rewrite'];target=w['target_new']['str']
        target=target if target.startswith(' ') else ' '+target
        ids=tok(target)['input_ids']
        if ids and ids[0] in (tok.bos_token_id,tok.unk_token_id):ids=ids[1:]
        if not ids:raise ValueError('EMPTY_NATIVE_TARGET')
        for ci,c in enumerate(flat):
            prompt=(c.format(w['prompt'])+tok.decode(ids[:-1])).format(w['subject'])
            inp=tok(prompt)['input_ids'];pos=list(range(len(inp)-len(ids),len(inp)))
            if not pos or pos[0]<0:raise ValueError('NATIVE_TARGET_ALIGNMENT')
            out.append(dict(ordinal=i,case_id=records[i]['case_id'],context_index=ci,input_ids=inp,
              target_ids=ids,positions=pos,context_weight=1/(len(indices)*len(flat)),token_mean_weight=1/len(ids),
              identity=digest([i,'Current',ci,inp,pos,ids]),kind='Current'))
    return out

def protection_rows(records,indices,kind,tok):
    from project.run_scripts.single_layer_cumulative_risk.evaluation import evaluator
    ev=evaluator();out=[];past=kind.startswith('Past')
    for i in indices:
        r=records[i];w=r['requested_rewrite'];prompts=[w['prompt'].format(w['subject'])]
        if past:prompts+=r['paraphrase_prompts'][:1]
        target=w['target_new' if past else 'target_true']['str']
        for ci,prompt in enumerate(prompts):
            pp,tt=ev._encode_pair(tok,ev.PromptTarget(r['case_id'],kind,ci,prompt,target))
            if not past:tt=tt[:8]
            if not pp or not tt:raise ValueError('EMPTY_PROTECTION_TOKENS')
            inp=(pp+tt)[:-1];pos=list(range(len(pp)-1,len(inp)))
            out.append(dict(ordinal=i,case_id=r['case_id'],context_index=ci,input_ids=inp,target_ids=tt,
              positions=pos,context_weight=1/(len(indices)*len(prompts)),token_mean_weight=1/len(tt),
              identity=digest([i,kind,ci,inp,pos,tt]),kind=kind))
    return out
