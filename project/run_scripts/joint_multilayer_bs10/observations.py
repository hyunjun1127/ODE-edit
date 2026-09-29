"""Fixed TF rows and bounded one-row observation; no writer feedback."""
from pathlib import Path
import time
from .common import *


def pair(tok, record_, role, checkpoint, kind, index, prompt):
    q=record_['requested_rewrite']; pid=digest([role,checkpoint,kind,index,record_['case_id'],prompt])
    pids=tok(prompt,add_special_tokens=True)['input_ids']; out=[]
    for label in ('true','new'):
        text=q['target_'+label]['str']; text=text if text.startswith(' ') else ' '+text
        tids=tok.encode(text,add_special_tokens=False)
        while tids and tids[0] in (tok.bos_token_id,tok.unk_token_id):tids.pop(0)
        require(bool(pids) and bool(tids),'EMPTY_TOKENS')
        ids=(pids+tids)[:-1]; pos=list(range(len(pids)-1,len(pids)+len(tids)-1))
        require(len(ids)<=8192,'TOKEN_LENGTH_NO_TRUNCATION')
        row=dict(pair_id=pid,role=role,checkpoint=checkpoint,kind=kind,prompt_index=index,
            case_id=int(record_['case_id']),subject=q['subject'],label=label,
            input_ids=ids,target_ids=tids,positions=pos,prompt=prompt,target=text)
        row['row_id']=digest(row);out.append(row)
    return out


def catalog(config,tok,cp,contexts):
    data={int(r['case_id']):r for r in read(config['dataset']['path'])};base=Path(config['design']);panels=[];current={};native={};neighborhood={}
    for x in csv_rows(base/'panel-ids.csv'):
        if x['checkpoint'] not in ('all',cp):continue
        r=data[int(x['case_id'])];q=r['requested_rewrite']
        panels+=pair(tok,r,x['role'],x['checkpoint'],'R',0,q['prompt'].format(q['subject']))
    ids=config['execution_ids'];require(len(ids)==STEPS,'USER_EXECUTION_COUNT')
    for n,cid in enumerate(ids,1):
        r=data[cid];q=r['requested_rewrite'];current[cid]=[]
        for kind,j,p in [('R',0,q['prompt'].format(q['subject']))]+[('P',j,p) for j,p in enumerate(r['paraphrase_prompts'][:2])]:
            current[cid]+=pair(tok,r,'continuation','all',kind,j,p)
        if n in MILESTONES:
            neighborhood[cid]=sum([pair(tok,r,'neighborhood','all','N',j,p) for j,p in enumerate(r['neighborhood_prompts'][:10])],[])
        target=q['target_new']['str'];target=target if target.startswith(' ') else ' '+target
        tids=tok(target)['input_ids']
        if tids[0] in (tok.bos_token_id,tok.unk_token_id):tids=tids[1:]
        nr=[]
        for j,ctx in enumerate(c for group in contexts for c in group):
            prompt=(ctx.format(q['prompt'])+tok.decode(tids[:-1])).format(q['subject'])
            x=tok(prompt)['input_ids'];positions=list(range(len(x)-len(tids),len(x)))
            row=dict(role='current_constraint',kind='context',case_id=cid,label='new',prompt_index=j,
                 input_ids=x,target_ids=tids,positions=positions,checkpoint=cp)
            row['row_id']=digest(row);row['pair_id']=digest([cid,j,'native_context']);nr.append(row)
        require(len(nr)==6 and all(min(r['positions'])>=0 for r in nr),'SIX_NATIVE_CONTEXTS');native[cid]=nr
    return data,ids,panels,current,native,neighborhood


class Scorer:
    def __init__(self,rt):
        self.rt=rt;self.calls={};self.tokens={};self.seconds={};self.backwards=0;self.backward_tokens=0;self.backward_seconds=0.

    def forward(self,row,category='observer',capture=False):
        import torch
        rt=self.rt;dev=next(rt.model.parameters()).device;ids=torch.tensor([row['input_ids']],device=dev)
        require(len(row['input_ids'])<=rt.model.config.max_position_embeddings,'LENGTH_NO_TRUNCATION')
        hooks=[];caps={};start=time.monotonic()
        if capture:
            for l in LAYERS:
                def hook(m,args,out,l=l):caps[l]=(args[0][0].detach().cpu().clone(),out[0].detach().cpu().clone())
                hooks.append(rt.model.model.layers[l].mlp.down_proj.register_forward_hook(hook))
        try:
            logits=rt.model(input_ids=ids,attention_mask=torch.ones_like(ids),use_cache=False).logits[0]
            lp=logits[row['positions']].float().log_softmax(-1)
            require(bool(torch.isfinite(lp).all()),'SCORER_NONFINITE')
        finally:
            for h in hooks:h.remove()
        self.calls[category]=self.calls.get(category,0)+1;self.tokens[category]=self.tokens.get(category,0)+ids.numel()
        self.seconds[category]=self.seconds.get(category,0)+time.monotonic()-start
        return lp,caps

    def metric(self,row,teacher=None,capture=False,category='observer'):
        import torch
        rt=self.rt;versions=rt.versions();rng=rt.rng_get()
        with torch.no_grad():
            lp,caps=self.forward(row,category,capture)
            target=torch.tensor(row['target_ids'],device=lp.device);nll=-lp.gather(1,target[:,None])[:,0];pred=lp.argmax(-1)
            r={k:row[k] for k in ('row_id','pair_id','role','kind','case_id','label','prompt_index','checkpoint')}
            r.update(nll=float(nll.mean()),token_nll=nll.cpu().tolist(),token_predictions=pred.cpu().tolist(),
               token_correct=int((pred==target).sum()),strict=bool((pred==target).all()),target_count=len(target),
               input_sha=digest(row['input_ids']),target_sha=digest(row['target_ids']),position_sha=digest(row['positions']))
            if teacher is not None:
                p0=teacher.to(lp.device);require(p0.shape==lp.shape,'TEACHER_LAYOUT');r['w0_kl']=float((p0.exp()*(p0-lp)).sum(-1).mean())
        require(rt.versions()==versions,'OBSERVER_MUTATION')
        now=rt.rng_get()
        require(rng[0]==now[0] and rng[1][0]==now[1][0] and (rng[1][1]==now[1][1]).all() and rng[1][2:]==now[1][2:]
            and torch.equal(rng[2],now[2]) and all(torch.equal(a,b) for a,b in zip(rng[3],now[3])),'OBSERVER_RNG_MUTATION')
        return r,lp.detach().cpu(),caps

    def greedy(self,row):
        import torch
        rt=self.rt;start=row['positions'][0]+1;ids=torch.tensor([row['input_ids'][:start]],device='cuda');eos=rt.model.generation_config.eos_token_id
        eos=set(eos if isinstance(eos,list) else [eos]);generated=[]
        with torch.no_grad():
            for _ in range(rt.config['settings']['greedy_max_new_tokens']):
                out=rt.model(input_ids=ids,attention_mask=torch.ones_like(ids),use_cache=False).logits
                token=int(out[0,-1].argmax());generated.append(token)
                self.calls['greedy']=self.calls.get('greedy',0)+1;self.tokens['greedy']=self.tokens.get('greedy',0)+ids.numel()
                if token in eos:break
                ids=torch.cat((ids,torch.tensor([[token]],device=ids.device)),dim=1)
        actual=rt.tok.decode(generated,skip_special_tokens=True).strip().casefold();expected=rt.tok.decode(row['target_ids'],skip_special_tokens=True).strip().casefold()
        return dict(row_id=row['row_id'],case_id=row['case_id'],kind=row['kind'],generated_ids=generated,
             text=actual,expected=expected,exact_match=actual==expected,eos_terminated=generated[-1] in eos,censored=generated[-1] not in eos)
