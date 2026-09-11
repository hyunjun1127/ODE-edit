"""We-anchored protection only. No current edit loss enters this observer."""
import contextlib
import torch
from .identity import WEIGHT, tensor_save, tensor_sha, digest

def psi(d,tau=.01):
    return torch.where(d<=0,torch.zeros_like(d),torch.where(d<tau,d.square()/(2*tau),d-tau/2))

def pack(rows,pad,device):
    width=max(len(r['input_ids']) for r in rows)
    ids=torch.full((len(rows),width),pad,dtype=torch.long,device=device);mask=torch.zeros_like(ids)
    positions=[]
    for i,r in enumerate(rows):
        start=width-len(r['input_ids']);ids[i,start:]=torch.tensor(r['input_ids'],device=device);mask[i,start:]=1
        positions.append(torch.tensor(r['positions'],device=device)+start)
    return dict(input_ids=ids,attention_mask=mask,use_cache=False),positions

class View:
    def __init__(self,model,pad,ledger):
        self.model=model;self.pad=pad;self.ledger=ledger
        self.weight=dict(model.named_parameters())[WEIGHT]
        self.signature=[(n,p.data_ptr(),p._version,tuple(p.shape),p.dtype) for n,p in model.named_parameters()]

    def guard(self):
        now=[(n,p.data_ptr(),p._version,tuple(p.shape),p.dtype) for n,p in self.model.named_parameters()]
        if now!=self.signature:raise RuntimeError('LIVE_PARAMETER_MUTATION')

    def forward(self,state,rows,keys=False):
        self.guard();args,positions=pack(rows,self.pad,self.weight.device)
        if state.dtype!=torch.float32 or state.device!=self.weight.device:raise TypeError('VIEW_FP32_DEVICE')
        captured=[]
        handle=None
        if keys:
            def hook(mod,inp,out):
                captured.extend([inp[0][j,pos].detach().cpu() for j,pos in enumerate(positions)])
            handle=self.model.get_submodule(WEIGHT.removesuffix('.weight')).register_forward_hook(hook)
        try:
            result=torch.func.functional_call(self.model,{WEIGHT:state},(),args,strict=False).logits
            selected=[torch.log_softmax(result[j,pos].float(),dim=-1) for j,pos in enumerate(positions)]
            self.ledger.add('functional_model_forward');self.ledger.add('functional_sequences',len(rows))
            self.ledger.add('functional_input_tokens',int(args['attention_mask'].sum()))
        finally:
            if handle is not None:handle.remove()
            self.guard()
        return selected,captured

    def prepare(self,state,rows,output,*,keys=True,micro=2):
        teacher=[];factors=[]
        with self.ledger.time('teacher_and_key_capture'),torch.no_grad():
            for start in range(0,len(rows),micro):
                group=rows[start:start+micro];logps,kk=self.forward(state,group,keys)
                for j,(r,logp) in enumerate(zip(group,logps)):
                    targets=torch.tensor(r['target_ids'],device=logp.device)
                    nll=-logp.gather(1,targets[:,None]).squeeze(1)
                    teacher.append(dict(identity=r['identity'],log_probs=logp.cpu(),token_nll=nll.cpu()))
                    if keys:factors.append(kk[j].double().T*(r['token_weight']**.5))
        data=dict(rows=rows,teacher=teacher,weight_sha=tensor_sha(state),precision='FP32 full-vocab log_softmax',
                  key_positions='logit predicting target; first=last prompt token',packing_identity=digest(rows))
        tensor_save(output,data)
        return data,torch.cat(factors,dim=1) if factors else torch.empty((state.shape[1],0),dtype=torch.float64)

    def harm(self,state,prepared,kind,*,gradient=False,micro=2):
        rows=prepared['rows'];teachers=prepared['teacher']
        grad=torch.zeros(state.shape,dtype=torch.float64,device='cpu') if gradient else None
        values=[];total=0.;raw_total=0.;negative=0
        with self.ledger.time('functional_gradient' if gradient else 'functional_harm'):
            for start in range(0,len(rows),micro):
                group=rows[start:start+micro];tt=teachers[start:start+micro]
                w=state.detach().requires_grad_(gradient)
                with torch.set_grad_enabled(gradient):
                    logps,_=self.forward(w,group)
                    obj=torch.zeros((),device=w.device,dtype=torch.float64)
                    for r,teacher,logp in zip(group,tt,logps):
                        if teacher['identity']!=r['identity']:raise ValueError('TEACHER_PACKING_IDENTITY')
                        ids=torch.tensor(r['target_ids'],device=w.device)
                        token_nll=-logp.gather(1,ids[:,None]).squeeze(1)
                        delta=token_nll.double()-teacher['token_nll'].to(w.device).double()
                        if kind=='Past':
                            raw=delta.mean();harm=psi(raw)
                            token_raw=delta
                        else:
                            te=teacher['log_probs'].to(w.device)
                            # p_We || p_W; FP32 teacher retained, FP64 reduction.
                            token_raw=(te.double().exp()*(te.double()-logp.double())).sum(-1)
                            raw=token_raw.mean();harm=token_raw.clamp_min(0).mean()
                            negative+=int((token_raw<0).sum())
                        obj=obj+r['context_weight']*harm
                        raw_total+=r['context_weight']*float(raw.detach())
                        values.append(dict(ordinal=r['ordinal'],case_id=r['case_id'],context_index=r['context_index'],
                            identity=r['identity'],kind=kind,token_nll=token_nll.detach().cpu().tolist(),
                            token_raw=token_raw.detach().cpu().tolist(),raw=float(raw.detach()),harm=float(harm.detach()),
                            context_weight=r['context_weight'],negative_kl_tokens=int((token_raw<0).sum()) if kind!='Past' else 0))
                    total+=float(obj.detach())
                    if gradient:
                        gg,=torch.autograd.grad(obj,w)
                        if not torch.isfinite(gg).all():raise FloatingPointError('NONFINITE_FUNCTIONAL_GRADIENT')
                        grad+=gg.detach().cpu().double();self.ledger.add('functional_backward')
                del logps,w,obj
        return dict(value=total,raw_value=raw_total,negative_kl_tokens=negative,rows=values),grad
