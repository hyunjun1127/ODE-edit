"""Shared actual-materialized five-layer oracle, current+general+replay.

All-token dX is retained. No layer/quality gates or approximate vocabulary.
Dense reference is authoritative final evaluation; direct R is optional only.
"""
import time
import resource
import numpy as np
import torch
import torch.nn.functional as F
from project.run_scripts.jlz_pilot.run import prefix_cache, token_subset, chunks
from project.run_scripts.jlz_efficiency.core import effective_linear, crop_tokens, CertificationFailed
from project.run_scripts.jlz_sequential.oracle import suffix_hidden, selected_loss
from project.run_scripts.jlz_sequential.state import tensor_sha
from project.run_scripts.blue_alphaedit_sequential_comparison.evaluation import bind_observation_only_package
bind_observation_only_package()
from project.run_scripts.alphaedit_strength_neutral_barrier.contracts import prompt_token_ids, target_token_ids
from .common import LAYERS, digest, numerical, require
from .burden import build_burden_matrix, omega_and_grad

def reference_pack(tokenizer,records,target,device):
    enc=[]
    for rec in records:
        rw=rec['requested_rewrite']; p=prompt_token_ids(tokenizer,rw['prompt'].format(rw['subject']))
        y=target_token_ids(tokenizer,rw[target]['str'])
        enc.append((p,y))
    width=max((len(p)+len(y)-1 for p,y in enc),default=0)
    ids=torch.full((len(enc),width),tokenizer.pad_token_id,dtype=torch.long,device=device)
    mask=torch.zeros_like(ids);labels=torch.full_like(ids,-100)
    for i,(p,y) in enumerate(enc):
        row=(p+y)[:-1];ids[i,:len(row)]=torch.tensor(row,device=device);mask[i,:len(row)]=1
        labels[i,len(p)-1:len(row)]=torch.tensor(y,device=device)
    identity=digest({'case_ids':[r['case_id'] for r in records], 'target':target,
                     'input_ids':ids.cpu().tolist(),'mask':mask.cpu().tolist(),'labels':labels.cpu().tolist()})
    return dict(tokens={'input_ids':ids,'attention_mask':mask},targets=labels,
                case_ids=[r['case_id'] for r in records],identity=identity)

@torch.no_grad()
def prepare_teacher(model,tok,records,mb=4):
    """Called only in pristine W0; CPU FP32 full-vocab log probabilities."""
    result={}
    for start in range(0,len(records),mb):
        group=records[start:start+mb];pack=reference_pack(tok,group,'target_true',next(model.parameters()).device)
        h=model.model(**pack['tokens'],use_cache=False).last_hidden_state
        for i,r in enumerate(group):
            rowpack=reference_pack(tok,[r],'target_true','cpu')
            logp=model.lm_head(h[i,pack['targets'][i]!=-100]).log_softmax(-1).float().cpu()
            require(bool(torch.isfinite(logp).all()),'W0_TEACHER_NONFINITE')
            result[str(r['case_id'])]=dict(identity=rowpack['identity'],logp=logp,sha256=tensor_sha(logp))
    return result

def geometry(keys,history,anchors,stats):
    adj={};matrices={};scales={};receipt=[]
    for l in LAYERS:
        t=time.monotonic();path=stats/f'model.layers.{l}.mlp.down_proj_float32_mom2_100000.npz'
        with np.load(path,allow_pickle=False) as z:
            count=int(z['mom2.count']);raw=torch.from_numpy(z['mom2.mom2'].copy())
        require(count>0 and raw.dtype==torch.float32 and bool(torch.isfinite(raw).all()),'C0_INVALID')
        A=(raw/count).to(device=keys[l].device,dtype=torch.float64).mul_(15000)
        del raw
        A.add_(history[l].to(device=A.device,dtype=A.dtype));K=keys[l].double()
        system=A+K@K.T;P=torch.linalg.solve(system,K)
        require(bool(torch.isfinite(P).all()),'NONFINITE_GEOMETRY')
        residual=float((system@P-K).norm()/K.norm().clamp_min(1))
        M,check=build_burden_matrix(A,K,P)
        adj[l]=P;matrices[l]=M;scales[l]=anchors[l].double().square().sum(0).mean()
        require(float(scales[l])>0 and bool(torch.isfinite(scales[l])),'ANCHOR_SCALE_INVALID')
        receipt.append(dict(layer=l,count=count,normalization='FP32_raw/count_then_FP64',
                            solve=numerical({'relative':residual},{'relative':1e-7}),M=check,
                            s_squared=float(scales[l]),seconds=time.monotonic()-t))
        del A,system,K
    return adj,matrices,scales,receipt

class CommonOracle:
    def __init__(self,model,tok,spec,teacher,adj,matrices,scales,general,replay,w0teachers,eta,mb=4):
        self.model,self.tok,self.spec,self.teacher=model,tok,spec,teacher
        self.adj,self.matrices,self.scales,self.eta=adj,matrices,scales,eta
        self.B=len(spec['specs']);self.mb=mb;self.calls=0;self.records=[];self.initial_gradient=None
        self.entry={l:model.model.layers[l].mlp.down_proj.weight.detach().clone() for l in LAYERS}
        self.active=torch.ones(self.B,dtype=torch.bool,device=teacher.device)
        self.groups=[]
        for rows in chunks(spec,mb):
            tokens,width=crop_tokens(spec,rows,True)
            self.groups.append(dict(kind='current',rows=rows,width=width,tokens=tokens,
                                    cache=prefix_cache(model,tokens)))
        self.ref_ids={'general':[r['case_id'] for r in general],'replay':[r['case_id'] for r in replay]}
        for kind,records,target in [('general',general,'target_true'),('replay',replay,'target_new')]:
            for start in range(0,len(records),mb):
                recs=records[start:start+mb];pack=reference_pack(tok,recs,target,teacher.device)
                require(pack['tokens']['input_ids'].shape[1]<=model.config.max_position_embeddings,'REFERENCE_LENGTH')
                teachers=[]
                if kind=='general':
                    for r in recs:
                        saved=w0teachers[str(r['case_id'])]
                        identity=reference_pack(tok,[r],target,'cpu')['identity']
                        require(saved['identity']==identity,'REFERENCE_TEACHER_TOKEN_IDENTITY')
                        require(saved['logp'].dtype==torch.float32,'REFERENCE_TEACHER_DTYPE')
                        teachers.append(saved['logp'])
                self.groups.append(dict(kind=kind,pack=pack,tokens=pack['tokens'],count=len(records),
                                        teachers=teachers,cache=prefix_cache(model,pack['tokens'])))

    def effective(self,x):
        return {l:self.entry[l]+(x[i*self.B:(i+1)*self.B].T.double()@self.adj[l].T).float()
                for i,l in enumerate(LAYERS)}

    def __call__(self,x,route='dense'):
        start=time.monotonic();self.calls+=1
        direct=route=='direct';full=route=='original'
        with torch.no_grad(): eff=self.effective(x)
        if not all(bool(torch.isfinite(w).all()) for w in eff.values()):
            raise FloatingPointError('NONFINITE_MATERIALIZATION')
        leaves={l:w.detach().requires_grad_(not direct) for l,w in eff.items()}
        proxies={l:x[i*self.B:(i+1)*self.B].T.double().detach().requires_grad_() for i,l in enumerate(LAYERS)} if direct else None
        targets=proxies if direct else leaves
        accum={l:torch.zeros_like(v) for l,v in targets.items()}
        domain={l:dict(bound=torch.zeros((),device=x.device,dtype=torch.float64),finite=torch.ones((),device=x.device,dtype=torch.bool)) for l in LAYERS}
        mapping={l:(proxies[l],self.adj[l],domain[l]) for l in LAYERS} if direct else None
        nll=torch.zeros(self.B,device=x.device);kl=torch.zeros_like(nll)
        total=0.;general_rows=[];replay_rows=[];work=dict(forward_chunks=0,backward_chunks=0,valid_tokens=0,padded_tokens=0,head_positions=0)
        for group in self.groups:
            with effective_linear(self.model,leaves,mapping):
                hidden=(self.model.model(**group['tokens'],use_cache=False).last_hidden_state if full
                        else suffix_hidden(self.model,group['cache']))
                if group['kind']=='current':
                    spec=self.spec|{'targets':self.spec['targets'][:,:group['width']]}
                    loss,nr,kr=selected_loss(self.model,hidden,spec,self.teacher,self.active,group['rows'])
                    nll.add_(nr.detach());kl.add_(kr.detach())
                    work['head_positions']+=sum(int((spec['targets'][r]!=-100).sum()) if spec['row_kind'][r]=='rewrite' else 1 for r in group['rows'])
                else:
                    parts=[];pack=group['pack']
                    for i,case in enumerate(pack['case_ids']):
                        mask=pack['targets'][i]!=-100;labels=pack['targets'][i,mask]
                        logp=self.model.lm_head(hidden[i,mask]).log_softmax(-1)
                        if group['kind']=='general':
                            teacher=group['teachers'][i].to(x.device)
                            require(teacher.shape==logp.shape,'TEACHER_POSITION_VOCAB_IDENTITY')
                            value=F.kl_div(teacher,logp,log_target=True,reduction='sum')/len(labels)
                            general_rows.append({'case_id':case,'value':float(value.detach())})
                        else:
                            value=-logp.gather(1,labels[:,None]).mean()
                            replay_rows.append({'case_id':case,'value':float(value.detach())})
                        parts.append(value);work['head_positions']+=len(labels)
                    coefficient=self.B*(.0625 if group['kind']=='general' else 1.)/group['count']
                    loss=torch.stack(parts).sum()*coefficient
                if not bool(torch.isfinite(loss)): raise FloatingPointError('NONFINITE_COMMON_LOSS')
            gradients=torch.autograd.grad(loss,tuple(targets.values()))
            for l,g in zip(LAYERS,gradients):accum[l].add_(g.detach())
            total+=float(loss.detach());work['forward_chunks']+=1;work['backward_chunks']+=1
            work['valid_tokens']+=int(group['tokens']['attention_mask'].sum());work['padded_tokens']+=group['tokens']['input_ids'].numel()
            del gradients,hidden,loss
        if direct:
            if not all(bool(v['finite']) and float(v['bound'])<torch.finfo(torch.float32).max/2 for v in domain.values()):
                raise CertificationFailed('DIRECT_OVERFLOW_DOMAIN')
            gradient=torch.cat([accum[l].float().T for l in LAYERS])
        else:gradient=torch.cat([(accum[l].double()@self.adj[l]).float().T for l in LAYERS])
        omega=0.;layer_omega=[]
        for i,l in enumerate(LAYERS):
            value,og=omega_and_grad(x[i*self.B:(i+1)*self.B].T,self.matrices[l],self.scales[l],self.eta)
            omega+=float(value);layer_omega.append(float(value));gradient[i*self.B:(i+1)*self.B].add_(og.T.float())
        total+=self.eta*omega
        if not bool(torch.isfinite(gradient).all()) or not np.isfinite(total):raise FloatingPointError('NONFINITE_COMMON_GRADIENT')
        if self.initial_gradient is None and bool((x==0).all()):self.initial_gradient=gradient.clone()
        if x.is_cuda:torch.cuda.synchronize()
        row=dict(call=self.calls,route=route,seconds=time.monotonic()-start,**work,
                 peak_gpu_bytes=torch.cuda.max_memory_allocated() if x.is_cuda else None,
                 host_peak_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        self.records.append(row)
        payload=dict(nll=nll.cpu().tolist(),kl=kl.cpu().tolist(),general=general_rows,replay=replay_rows,
                     omega=omega,layer_omega=layer_omega,eta=self.eta,weights={l:w.detach() for l,w in leaves.items()},timing=row)
        return total,gradient,payload

    @torch.no_grad()
    def committed_losses(self):
        nll=torch.zeros(self.B,device=self.teacher.device);kl=torch.zeros_like(nll)
        for group in self.groups:
            if group['kind']!='current':continue
            hidden=self.model.model(**group['tokens'],use_cache=False).last_hidden_state
            spec=self.spec|{'targets':self.spec['targets'][:,:group['width']]}
            _,nr,kr=selected_loss(self.model,hidden,spec,self.teacher,self.active,group['rows'])
            nll.add_(nr);kl.add_(kr)
        return nll,kl
