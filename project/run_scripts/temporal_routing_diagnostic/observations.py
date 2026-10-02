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


def build_rows(config,tok):
    data={int(r['case_id']):r for r in read(config['dataset']['path'])}; base=Path(config['design']);rows=[]
    for x in csv_rows(base/'panel-ids.csv'):
        r=data[int(x['case_id'])]; q=r['requested_rewrite']
        rows+=pair(tok,r,x['role'],x['checkpoint'],'R',0,q['prompt'].format(q['subject']))
    for i,x in enumerate(csv_rows(base/'continuation-ids.csv'),1):
        r=data[int(x['case_id'])]; q=r['requested_rewrite']
        prompts=[('R',0,q['prompt'].format(q['subject']))]+[('P',j,p) for j,p in enumerate(r['paraphrase_prompts'][:2])]
        if i in MILESTONES: prompts += [('N',j,p) for j,p in enumerate(r['neighborhood_prompts'][:10])]
        for kind,j,p in prompts:
            rr=pair(tok,r,'continuation','all',kind,j,p)
            for v in rr:v['arrival']=i
            rows+=rr
    require(len({r['row_id'] for r in rows})==len(rows),'DUPLICATE_ROWS')
    return rows


def signed_random(x,seed):
    import torch
    g=torch.Generator(device='cpu');g.manual_seed(seed)
    perm=torch.randperm(x.shape[-1],generator=g).to(x.device)
    signs=(torch.randint(0,2,(x.shape[-1],),generator=g)*2-1).to(x.device,x.dtype)
    return x[...,perm]*signs


def norm_record(x):
    return dict(norm=float(x.double().norm()),rms=float(x.double().square().mean().sqrt()),elements=x.numel())


class Observer:
    def __init__(self,rt,output):
        self.rt=rt;self.output=Path(output);self.calls=0;self.seconds=0.;self.capture_bytes=0

    def evaluate(self,row,*,patch=None,teacher=None):
        import torch
        rt=self.rt; model=rt.model; device=next(model.parameters()).device
        ids=torch.tensor([row['input_ids']],device=device)
        pos=row['positions']; target=torch.tensor(row['target_ids'],device=device)
        hooks=[];keys={};values={};residual={};patch_measure={}
        begin=time.monotonic(); versions=rt.versions()
        rng_get=torch.cuda.get_rng_state if device.type=='cuda' else torch.get_rng_state
        rng=rng_get().clone()
        for l in range(rt.layer,9):
            layer=model.model.layers[l]
            def hook(m,args,out,l=l):
                keys[l]=args[0][0].detach().cpu().clone()
                if patch is not None and l==8:
                    entry=patch['entry_key'].to(out.device)
                    require(entry.shape==args[0][0].shape,'PATCH_POSITION_ALIGNMENT')
                    action=-torch.nn.functional.linear(args[0][0]-entry,patch['D8'].to(out.device))
                    targeted_norm=action.double().norm(dim=-1).detach().cpu()
                    if patch['mode']=='zero':action=torch.zeros_like(action)
                    elif patch['mode']=='random':action=signed_random(action,patch['seed'])
                    require(bool(torch.isfinite(action).all()),'PATCH_NONFINITE')
                    patch_measure.update(mode=patch['mode'],valid_positions=action.shape[0],
                        action_norm=float(action.double().norm()),targeted_token_norms=targeted_norm.tolist(),
                        applied_token_norms=action.double().norm(dim=-1).detach().cpu().tolist())
                    out=out+action.unsqueeze(0)
                values[l]=out[0].detach().cpu().clone()
                return out
            hooks.append(layer.mlp.down_proj.register_forward_hook(hook))
            def residual_pre(m,args,l=l):residual[l]=args[0][0].detach().cpu().clone()
            hooks.append(layer.post_attention_layernorm.register_forward_pre_hook(residual_pre))
        try:
            with torch.no_grad():
                logits=model(input_ids=ids,attention_mask=torch.ones_like(ids),use_cache=False).logits[0]
                lp=logits[pos].float().log_softmax(-1)
                nll=-lp.gather(1,target[:,None])[:,0]; pred=lp.argmax(-1); correct=pred.eq(target)
                summary={k:row[k] for k in ('row_id','pair_id','role','checkpoint','kind','case_id','label','prompt_index')}
                summary.update(nll=float(nll.mean()),token_nll=nll.cpu().tolist(),token_predictions=pred.cpu().tolist(),
                    target_count=len(target),token_correct=int(correct.sum()),strict=bool(correct.all()),
                    target_ids_sha=digest(row['target_ids']),input_sha=digest(row['input_ids']),position_sha=digest(pos))
                summary['answer_logp_sha256']=tensor_sha(lp)
                if patch is not None:summary['patch']=patch_measure
                if teacher is not None:
                    p0=teacher.to(device);require(p0.shape==lp.shape,'TEACHER_POSITION')
                    summary['w0_kl']=float((p0.double().exp()*(p0.double()-lp.double())).sum(-1).mean())
                summary['layer_norms']={str(l):dict(key=norm_record(keys[l]),mlp_readout=norm_record(values[l]),residual=norm_record(residual[l]),
                    readout_residual_norm_ratio=float(values[l].double().norm()/(residual[l].double().norm()+rt.config['observer']['epsilon']))) for l in keys}
                require(bool(torch.isfinite(lp).all()) and all(bool(torch.isfinite(x).all()) for x in list(keys.values())+list(values.values())),'OBSERVER_NONFINITE')
                payload=dict(row_id=row['row_id'],input_sha=summary['input_sha'],positions=pos,keys=keys,values=values,
                    teacher_logp=lp.detach().cpu().clone() if row['role'].startswith('base_') and row['label']=='true' else None)
        finally:
            for h in hooks:h.remove()
        require(rt.versions()==versions and torch.equal(rng,rng_get()),'OBSERVER_MUTATED_MODEL_OR_RNG')
        self.calls+=1;self.seconds+=time.monotonic()-begin
        return summary,payload


def risk(before,after,w0,k0,kt,epsilon):
    """Question-level raw A/B/C and shared normalization; signed B/C retained."""
    import torch
    with torch.no_grad():
        w=before.double();d=after.double()-w;orig=w0.to(w.device).double()
        k0=k0.to(w.device).double().T;kt=kt.to(w.device).double().T
        ref=orig@k0; delta0=d@k0;deltat=d@kt
        A=delta0.square().sum(); B=2*((w-orig)@k0*delta0).sum()+A
        C=2*((w@kt-ref)*deltat).sum()+deltat.square().sum();den=ref.square().sum()+epsilon
        return dict(A_raw=float(A),B_raw=float(B),C_raw=float(C),denominator=float(den),
                    A=float(A/den),B=float(B/den),C=float(C/den),valid_positions=k0.shape[1])
