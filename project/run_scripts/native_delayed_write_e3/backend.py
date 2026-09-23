"""Read-only BASE endpoint forward/hybrid/hook backend. No writer imports/calls."""
import contextlib
import hashlib
import os
from pathlib import Path
import random
import sys
import time
import numpy as np
from .common import ROOT, BASE, DEPS, MODEL, BATCHES, digest, read, sha


def tensor_sha(t):
    a = t.detach().cpu().contiguous().numpy()
    h=hashlib.sha256(str((str(t.dtype),list(t.shape))).encode())
    h.update(memoryview(a).cast('B'))
    return h.hexdigest()


def groups(rows):
    """Same composition/order in E1, factorial and patch. No length bucketing."""
    out = []; current = []; tag = None
    for r in rows:
        key = (r['panel'],r['kind'],r['label'])
        cap = 4 if r['kind']=='GENERAL' else 16
        if current and (key != tag or len(current) == cap):
            out.append(current); current=[]
        tag=key; current.append(r)
    if current: out.append(current)
    return out


class Backend:
    def __init__(self, config):
        sys.path.insert(0, str(DEPS))
        import torch
        import transformers
        from transformers import AutoModelForCausalLM
        self.torch=torch; self.config=config
        assert transformers.__version__=='4.44.2'
        assert torch.__version__=='2.9.1+cu128'
        random.seed(20260924); np.random.seed(20260924); torch.manual_seed(20260924)
        torch.set_num_threads(8)
        torch.backends.cuda.matmul.allow_tf32=False
        torch.backends.cudnn.allow_tf32=True
        self.started=time.monotonic(); self.calls=0; self.forward_seconds=0.; self.upload_bytes=0
        self.model=AutoModelForCausalLM.from_pretrained(MODEL, local_files_only=True,
            low_cpu_mem_usage=True, attn_implementation='eager').cuda().eval()
        for p in self.model.parameters(): p.requires_grad_(False)
        assert {p.dtype for p in self.model.parameters()}=={torch.float32}
        self.modules={l:self.model.model.layers[l].mlp.down_proj for l in range(4,9)}
        self.weights={l:m.weight for l,m in self.modules.items()}
        self.w0={l:w.detach().cpu().clone() for l,w in self.weights.items()}
        self.w0sha={l:tensor_sha(w) for l,w in self.w0.items()}
        self.fixed={n:(p.data_ptr(),p._version) for n,p in self.model.named_parameters() if n not in {f'model.layers.{l}.mlp.down_proj.weight' for l in range(4,9)}}
        self.states={}; self.state_metadata={}; self.name='W0'
        self.cp_stat={}
        for family in ('BASE_ALPHAEDIT','BASE_MEMIT'):
            for b in BATCHES:
                path=Path(config['checkpoints'][family][str(b)]['path'])
                st=path.stat(); self.cp_stat[str(path)]=(st.st_size,st.st_mtime_ns,st.st_ino)
                cp=torch.load(path,map_location='cpu',weights_only=True,mmap=True)
                md=cp['metadata']
                assert md['batch']==b and md['method']==('AlphaEdit' if family=='BASE_ALPHAEDIT' else 'MEMIT')
                assert md['base_model_revision']==MODEL.name and md['sample_root']==config['ordered_root']
                expected_ids=config['case_order'][:b*100]
                assert md['seen_ids']==expected_ids
                ws={l:cp['weights'][f'model.layers.{l}.mlp.down_proj.weight'] for l in range(4,9)}
                for l,w in ws.items():
                    assert w.shape==(4096,14336) and w.dtype==torch.float32
                    assert torch.isfinite(w).all()
                    assert tensor_sha(w)==md['state']['weights'][f'model.layers.{l}.mlp.down_proj.weight']
                assert cp['cache_c'].shape == ((5,14336,14336) if family=='BASE_ALPHAEDIT' else (0,))
                key=f'{family}_W{b:03d}'
                self.states[key]=ws  # mmap retained only selected W; never read/write M pages
                self.state_metadata[key]=dict(batch=b,method=md['method'],state=md['state'],context_sha=digest(md['contexts']),
                    rng_sha=digest(md['rng']),history_shape=list(cp['cache_c'].shape),checkpoint_sha=config['checkpoints'][family][str(b)]['sha256'])
        self.rng0=torch.get_rng_state().clone(); self.cuda_rng0=torch.cuda.get_rng_state().clone()
        self.model_load_seconds=time.monotonic()-self.started

    def set_weights(self, weights, name):
        with self.torch.no_grad():
            for l,w in weights.items():
                self.weights[l].copy_(w)
                self.upload_bytes+=w.numel()*w.element_size()
        self.name=name

    def endpoint(self, name):
        self.set_weights(self.w0 if name=='W0' else self.states[name],name)

    def hybrid(self, family, s, t, state):
        """A from prefix s; B changes only L4:7. CPU FP64 construction -> FP32 once.

        Corner11 directly copies the original endpoint, avoiding subtract/add drift.
        L4:7 use exact Ws/Wt endpoint copies; only L8 A-ablation needs subtraction.
        """
        ws=self.states[f'{family}_W{s:03d}']; wt=self.states[f'{family}_W{t:03d}']
        a,b=map(int,state)
        with self.torch.no_grad():
            for l in range(4,8): self.weights[l].copy_(wt[l] if b else ws[l])
            if a: self.weights[8].copy_(wt[8])
            else: self.weights[8].copy_((wt[8].double()-ws[8].double()+self.w0[8].double()).float())
        self.name=f'{family}_{s}_{t}_{state}'
        self.upload_bytes+=5*4096*14336*4

    def batch(self, rows):
        torch=self.torch
        n=max(len(r['input_ids']) for r in rows)
        ids=torch.full((len(rows),n),128009,dtype=torch.long,device='cuda')
        mask=torch.zeros_like(ids); positions=[]
        for i,r in enumerate(rows):
            start=n-len(r['input_ids']); ids[i,start:]=torch.tensor(r['input_ids'],device='cuda');mask[i,start:]=1
            positions.append([start+j for j in r['positions']])
        return ids,mask,positions

    def forward(self, rows, capture=True, patch=None, general_teacher=None, return_general=False):
        torch=self.torch; ids,mask,positions=self.batch(rows)
        hooks=[]; keys={}; values={}
        for l,m in self.modules.items():
            if capture or (patch is not None and l==8):
                def hook(module,args,out,layer=l):
                    if capture:
                        keys[layer]=args[0].detach().clone(); values[layer]=out.detach().clone()
                    if patch is not None and layer==8:
                        assert patch.shape==out.shape and patch.dtype==out.dtype and torch.isfinite(patch).all()
                        return out+patch
                hooks.append(m.register_forward_hook(hook))
        begin=time.monotonic()
        try:
            with torch.no_grad():
                logits=self.model(input_ids=ids,attention_mask=mask,use_cache=False).logits.float()
                # Identical full-logit then FP32 log_softmax route to frozen evaluator.
                logp=torch.log_softmax(logits,dim=-1); pred=logits.argmax(dim=-1)
                result=[]; teachers=[]
                for i,r in enumerate(rows):
                    ps=positions[i]; target=torch.tensor(r['target_ids'],device='cuda')
                    lp=logp[i,ps,:]; ls=logits[i,ps,:]
                    nll=-lp.gather(1,target[:,None]).mean()
                    top=pred[i,ps]; correct=top==target
                    target_logits=ls.gather(1,target[:,None])[:,0]
                    max_other=ls.clone(); max_other.scatter_(1,target[:,None],float('-inf'))
                    gap=target_logits-max_other.max(dim=1).values
                    row=dict(row_id=r['row_id'],pair_id=r['pair_id'],panel=r['panel'],kind=r['kind'],label=r['label'],
                        case_id=r['case_id'],subject=r['subject'],prompt_cluster=r['prompt_cluster'],
                        target_count=len(target),nll=float(nll),token_correct=int(correct.sum()),strict=bool(correct.all()),
                        token_predictions=top.cpu().tolist(),token_nll=(-lp.gather(1,target[:,None])[:,0]).cpu().tolist(),
                        target_gap_mean=float(gap.mean()),target_gap_min=float(gap.min()),target_logit_mean=float(target_logits.mean()),
                        logit_rms=float(ls.double().square().mean().sqrt()),input_sha=digest([r['input_ids'],r['positions'],r['target_ids']]))
                    if general_teacher is not None:
                        p0=general_teacher[i].to('cuda')
                        assert p0.shape==lp.shape and p0.dtype==torch.float32
                        row['w0_forward_kl']=float((p0.double().exp()*(p0.double()-lp.double())).sum(-1).mean())
                        row['w0_top1_agree']=int((p0.argmax(-1)==top).sum())
                    if return_general: teachers.append(lp.detach().cpu())
                    assert np.isfinite([row['nll'],row['target_gap_mean'],row['logit_rms']]).all()
                    result.append(row)
            torch.cuda.synchronize()
        finally:
            for h in hooks:h.remove()
        self.calls+=1; self.forward_seconds+=time.monotonic()-begin
        return result,keys,values,mask.bool(),teachers

    def assert_readonly(self):
        torch=self.torch
        for n,p in self.model.named_parameters():
            if n in self.fixed: assert (p.data_ptr(),p._version)==self.fixed[n], ('NONSELECTED_MUTATION',n)
        assert torch.equal(torch.get_rng_state(),self.rng0) and torch.equal(torch.cuda.get_rng_state(),self.cuda_rng0)
        for p,old in self.cp_stat.items():
            st=Path(p).stat(); assert (st.st_size,st.st_mtime_ns,st.st_ino)==old

    def restore(self):
        self.endpoint('W0')
        for l,w in self.weights.items():assert tensor_sha(w)==self.w0sha[l]
        self.assert_readonly()

    def cost(self):
        return dict(forward_calls=self.calls,forward_seconds=self.forward_seconds,selected_weight_H2D_bytes=self.upload_bytes,
            model_load_seconds=self.model_load_seconds,peak_gpu_bytes=self.torch.cuda.max_memory_allocated(),
            program_seconds=time.monotonic()-self.started,native_z=0,write=0,history_append=0,checkpoint_saved=False,
            M='READONLY_INPUT_NOT_MATERIALIZED_NOT_USED_BY_FORWARD',exact_new_resume='NOT_AVAILABLE')


def norms(x,mask):
    a=x[mask].double()
    return dict(norm=float(a.norm()),rms=float(a.square().mean().sqrt()),max_abs=float(a.abs().max()),elements=a.numel())


def per_row_norms(x,mask):
    return [norms(x[i:i+1],mask[i:i+1]) for i in range(len(x))]


def rotation(action, seed):
    """Fixed signed permutation is exactly token-RMS matched, no outcome fit."""
    import torch
    generator=torch.Generator(device='cpu');generator.manual_seed(seed)
    perm=torch.randperm(action.shape[-1],generator=generator).to(action.device)
    sign=(torch.randint(0,2,(action.shape[-1],),generator=generator)*2-1).to(device=action.device,dtype=action.dtype)
    return action[...,perm]*sign
