"""E4 retains canonical left pad/group/positions; no singleton tie recheck."""
import math
import time
import torch
from project.run_scripts.jlz_sequential.observation import PromptTarget, pair_identity, reduce_rows
from project.run_scripts.alphaedit_strength_neutral_barrier.evaluator import _encode_pair

def pack(tok,group,device):
    encoded=[_encode_pair(tok,p) for p in group]
    width=max(len(p)+len(t)-1 for p,t in encoded)
    ids=torch.full((len(group),width),int(tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id),device=device,dtype=torch.long)
    mask=torch.zeros_like(ids);positions=[]
    for i,(p,t) in enumerate(encoded):
        inp=(p+t)[:-1];start=width-len(inp)
        ids[i,start:]=torch.tensor(inp,device=device);mask[i,start:]=1
        positions.append(list(range(start+len(p)-1,width)))
    return dict(input_ids=ids,attention_mask=mask),encoded,positions

@torch.no_grad()
def group_eval(model,tok,group,selected):
    device=next(model.parameters()).device;tokens,encoded,positions=pack(tok,group,device)
    if selected:
        hidden=model.model(**tokens,use_cache=False).last_hidden_state
        logits=[model.lm_head(hidden[i,p]).float() for i,p in enumerate(positions)]
    else:
        full=model(**tokens,use_cache=False).logits.float()
        # Preserve original evaluator's full-position logsoftmax/argmax work.
        # It is not a silently optimized reference benchmark.
        full_logp=full.log_softmax(-1);full_prediction=full.argmax(-1)
        logits=[full[i,p] for i,p in enumerate(positions)]
    rows=[]
    for i,((_,target),p,z) in enumerate(zip(encoded,group,logits)):
        if not bool(torch.isfinite(z).all()):raise FloatingPointError('NONFINITE_EVALUATOR')
        label=torch.tensor(target,device=device)
        pred=z.argmax(-1) if selected else full_prediction[i,positions[i]]
        logp=z.log_softmax(-1) if selected else full_logp[i,positions[i]]
        top=z.topk(2,dim=-1).values
        rows.append(dict(case_id=p.case_id,kind=p.kind,prompt_index=p.prompt_index,
            target_token_ids=target,nll=float(-logp.gather(1,label[:,None]).mean()),
            predictions=pred.cpu().tolist(),correct=(pred==label).cpu().tolist(),strict=bool((pred==label).all()),
            near_argmax=bool(((top[:,0]-top[:,1])<=2e-3).any()),logits=z.cpu()))
    work=dict(forwards=1,valid_tokens=int(tokens['attention_mask'].sum()),padded_tokens=tokens['input_ids'].numel(),
              attention_L2_positions=tokens['input_ids'].numel()*tokens['input_ids'].shape[1],
              head_rows=sum(len(p) for p in positions) if selected else tokens['input_ids'].numel())
    return rows,work

def panels(records,kind):
    news=[];trues=[];identities=[]
    for r in records:
        rw=r['requested_rewrite'];prompts=[rw['prompt'].format(rw['subject'])] if kind=='R' else r['paraphrase_prompts'] if kind=='P' else r['neighborhood_prompts']
        if len(prompts)!={'R':1,'P':2,'N':10}[kind]:raise RuntimeError('DENOMINATOR')
        for i,p in enumerate(prompts):
            news.append(PromptTarget(r['case_id'],kind,i,p,rw['target_new']['str']))
            trues.append(PromptTarget(r['case_id'],kind,i,p,rw['target_true']['str']))
            identities.append(pair_identity(r,kind,i,p))
    return news,trues,identities

@torch.no_grad()
def evaluate(model,tok,records,selected=False):
    torch.cuda.synchronize();start=time.monotonic();rows=[];work={};fallback=0;fallback_seconds=0.;captures=[]
    for kind in ('R','P','N'):
        news,trues,identities=panels(records,kind);a=[];b=[]
        for pairs,result in ((news,a),(trues,b)):
            for offset in range(0,len(pairs),2):
                values,w=group_eval(model,tok,pairs[offset:offset+2],selected)
                result.extend(values)
                for k,v in w.items():work[k]=work.get(k,0)+v
        if selected:
            groups={i//2 for i,(x,y) in enumerate(zip(a,b)) if x['near_argmax'] or y['near_argmax'] or abs(x['nll']-y['nll'])<=2e-4}
            t=time.monotonic()
            for g in sorted(groups):
                offset=g*2
                # Always rerun both original groups, same left-pad widths/order.
                for pairs,result in ((news,a),(trues,b)):
                    values,w=group_eval(model,tok,pairs[offset:offset+2],False)
                    result[offset:offset+len(values)]=values;fallback+=1
                    for k,v in w.items():work[k]=work.get(k,0)+v
            fallback_seconds+=time.monotonic()-t
        for x,y,identity in zip(a,b,identities):
            row=dict(case_id=x['case_id'],kind=kind,prompt_index=x['prompt_index'],identity=identity,new_nll=x['nll'],true_nll=y['nll'])
            for label,z in (('new',x),('true',y)):
                row[label+'_token_correct']=sum(z['correct']);row[label+'_token_count']=len(z['correct']);row[label+'_strict']=z['strict']
                row[label+'_predictions']=z['predictions'];captures.append(z['logits'])
            rows.append(row)
    torch.cuda.synchronize()
    return dict(rows=rows,summary=reduce_rows(rows),seconds=time.monotonic()-start,
        work=work,fallback_full_groups=fallback,fallback_seconds=fallback_seconds,logits=captures,
        instrument_overhead='Selected-logit CPU capture included in wall; not production throughput',
        selected=selected)

def compare(reference,candidate):
    errors=[];same=True;maxlog=0.;sumsq=0.;count=0
    for a,b in zip(reference['rows'],candidate['rows'],strict=True):
        same &= a['identity']==b['identity']
        for k in ('new','true'):
            errors.append(abs(a[k+'_nll']-b[k+'_nll']))
            same &= all(a[k+s]==b[k+s] for s in ('_token_count','_token_correct','_strict','_predictions'))
        same &= (a['new_nll']<a['true_nll'])==(b['new_nll']<b['true_nll'])
        same &= (a['true_nll']<a['new_nll'])==(b['true_nll']<b['new_nll'])
    for a,b in zip(reference['logits'],candidate['logits'],strict=True):
        d=(a-b).double();maxlog=max(maxlog,float(d.abs().max()));sumsq+=float(d.square().sum());count+=d.numel()
    rms=math.sqrt(sumsq/count)
    return dict(status='PASS' if same and max(errors)<=1e-4 and maxlog<=1e-3 and rms<=1e-4 else 'UNQUALIFIED',
        discrete_exact=bool(same),nll_maxabs=max(errors),logit_maxabs=maxlog,logit_rms=rms,
        denominator=len(reference['rows']),not_new_scientific_endpoint=True)
