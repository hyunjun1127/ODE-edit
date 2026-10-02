"""Observer-only paired NLL/TF, one shared sealed schedule, actual denominators."""
import math
import time
import unicodedata
import torch
from .common import require,digest,write
from .writer import state

def active_flags(records):
    flags={r['case_id']:True for r in records};versions={}
    for record in records:
        r=record['requested_rewrite']
        claim=(unicodedata.normalize('NFC',' '.join(r['subject'].split())),r['relation_id'])
        target=r['target_new'].get('id',r['target_new']['str'])
        for old,t in versions.get(claim,[]):
            if t!=target:flags[old]=False
        versions.setdefault(claim,[]).append((record['case_id'],target))
    return flags

def reduce_rows(rows):
    result={}
    require(len({r['identity'] for r in rows})==len(rows),'DUPLICATE_OBSERVER_ROW')
    for kind in sorted({r['kind'] for r in rows}):
        group=[r for r in rows if r['kind']==kind];desired='true' if kind=='N' else 'new'
        require(all(math.isfinite(r[k]) for r in group for k in ('new_nll','true_nll')),'NONFINITE_OBSERVER')
        success=[r['true_nll']<r['new_nll'] if kind=='N' else r['new_nll']<r['true_nll'] for r in group]
        count=sum(r[desired+'_token_count'] for r in group);correct=sum(r[desired+'_token_correct'] for r in group)
        result[kind]=dict(denominator=len(group),numerator=sum(success),rate=sum(success)/len(group),
            true_nll_mean=sum(r['true_nll'] for r in group)/len(group),new_nll_mean=sum(r['new_nll'] for r in group)/len(group),
            desired_token_count=count,desired_token_correct=correct,token_micro=correct/count,
            prompt_macro=sum(r[desired+'_token_correct']/r[desired+'_token_count'] for r in group)/len(group),
            strict_numerator=sum(r[desired+'_strict'] for r in group),strict_denominator=len(group),
            new_strict_numerator=sum(r['new_strict'] for r in group))
    return result

@torch.no_grad()
def scores(a,bench,pairs,microbatch):
    out=[]
    for start in range(0,len(pairs),microbatch):
        group=pairs[start:start+microbatch]
        encoded=[bench.evaluation_ids(prompt,target) for prompt,target in group]
        width=max(len(p)+len(t)-1 for p,t in encoded)
        require(width<=a.model.config.max_position_embeddings,'EVAL_LENGTH_OVERFLOW_NO_TRUNCATION')
        ids=torch.full((len(group),width),bench.tokenizer.pad_token_id,device=a.device,dtype=torch.long)
        mask=torch.zeros_like(ids);positions=[]
        for i,(p,t) in enumerate(encoded):
            row=(p+t)[:-1];offset=width-len(row)
            ids[i,offset:]=torch.tensor(row,device=a.device);mask[i,offset:]=1
            positions.append(list(range(offset+len(p)-1,width)))
        hidden=a.model.model(input_ids=ids,attention_mask=mask,use_cache=False).last_hidden_state
        selected=torch.cat([hidden[i,pos] for i,pos in enumerate(positions)])
        logits=a.model.lm_head(selected).float();logp=logits.log_softmax(-1);pred=logits.argmax(-1)
        cursor=0
        for p,t in encoded:
            target=torch.tensor(t,device=a.device);lp=logp[cursor:cursor+len(t)];pr=pred[cursor:cursor+len(t)]
            out.append(dict(nll=float(-lp.gather(1,target[:,None]).mean()),token_count=len(t),
                            token_correct=int((pr==target).sum()),strict=bool((pr==target).all()),
                            token_identity=digest([p,t])))
            cursor+=len(t)
    return out

@torch.no_grad()
def observe(a,bench,all_records,selected_records,history,endpoint,out,microbatch=2,current_ids=None):
    before=state(a,history);guard=a.guard();hooks=a.hook_signature();start=time.monotonic();rows=[]
    flags=active_flags(all_records)
    for startrow in range(0,len(selected_records),50):
        records=selected_records[startrow:startrow+50]
        specs=[];pairs=[]
        for record in records:
            r=record['requested_rewrite']
            for kind,prompts in bench.panels(record).items():
                for index,prompt in enumerate(prompts):
                    identity=digest([record['case_id'],kind,index,prompt,r['target_new']['str'],r['target_true']['str']])
                    specs.append(dict(case_id=record['case_id'],kind=kind,prompt_index=index,identity=identity,
                                      endpoint=endpoint,active_at_endpoint=flags[record['case_id']]))
                    pairs.extend([(prompt,r['target_new']['str']),(prompt,r['target_true']['str'])])
        values=scores(a,bench,pairs,microbatch)
        for i,row in enumerate(specs):
            for label,value in zip(('new','true'),values[2*i:2*i+2]):
                row.update({label+'_'+k:v for k,v in value.items()})
            row['margin_true_minus_new']=row['true_nll']-row['new_nll']
            rows.append(row)
        # Persist valid chunks before any later failure, no final overwrite.
        write(out/f'chunk-{startrow:04d}.json',dict(state=before,rows=specs,optimizer_feedback=False))
        print({'event':'observer','endpoint':endpoint,'requests_done':startrow+len(records)},flush=True)
    require(state(a,history)==before and a.guard()==guard and a.hook_signature()==hooks,'OBSERVER_MUTATION')
    summary=reduce_rows(rows)
    current=set(current_ids or [r['case_id'] for r in selected_records])
    result=dict(endpoint=endpoint,state=before,requests=len(selected_records),summary=summary,
                current=reduce_rows([r for r in rows if r['case_id'] in current]),row_count=len(rows),
                row_order=digest([r['identity'] for r in rows]),seconds=time.monotonic()-start,
                no_mutation=True,optimizer_feedback=False)
    write(out/'summary.json',result)
    return result
