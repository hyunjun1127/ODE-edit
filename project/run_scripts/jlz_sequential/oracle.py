"""Shared FP32 materialization, selected full-vocabulary head, GPU accumulation.

The model suffix is full sequence. Only the output head's unused positions are
omitted. Accumulated dL/dW is pulled back through FP64 matmul / FP32 cast once.
The pinned pilot oracle is retained for the same-entry actual BS100 comparison.
"""
import time
import torch
import torch.nn.functional as F
from project.run_scripts.jlz_pilot.run import (Oracle as LegacyOracle, LAYERS,
    functional_weights, token_subset, require)

def suffix_hidden(model, cache):
    args, kwargs = cache['args'], cache['kwargs']
    hidden = args[0] if args else kwargs['hidden_states']
    for layer in model.model.layers[4:]:
        local = dict(kwargs)
        if args:
            out = layer(hidden, *args[1:], **local)
        else:
            local['hidden_states'] = hidden
            out = layer(**local)
        hidden = out[0] if isinstance(out, (tuple,list)) else out
    return model.model.norm(hidden)

def selected_loss(model, hidden, spec, teacher, active, rows):
    B = len(spec['specs'])
    zero = hidden.reshape(-1)[0] * 0.0
    ns, ks = [[] for _ in range(B)], [[] for _ in range(B)]
    # Per-row head retains the exact prediction position and full vocabulary.
    for local,row in enumerate(rows):
        r = spec['row_request'][row]
        if spec['row_kind'][row] == 'rewrite':
            mask = spec['targets'][row] != -100
            labels = spec['targets'][row,mask]
            logits = model.lm_head(hidden[local,mask])
            value = -logits.log_softmax(-1).gather(1,labels[:,None]).sum()
            ns[r].append(value / labels.numel() / spec['n_rw'])
        else:
            logits = model.lm_head(hidden[local,spec['lookup'][row]])
            ks[r].append(F.kl_div(teacher[r:r+1], logits.log_softmax(-1)[None],
                                 log_target=True, reduction='batchmean'))
    nll = torch.stack([torch.stack(v).sum() if v else zero for v in ns])
    kl = torch.stack([torch.stack(v).sum() if v else zero for v in ks])
    loss = (nll[active]+.0625*kl[active]).sum()
    if not all(bool(torch.isfinite(v).all()) for v in (nll,kl,loss)):
        raise FloatingPointError('NONFINITE_SELECTED_LOSS')
    return loss,nll,kl

class Oracle(LegacyOracle):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.seconds = 0.
        self.token_work = 0
        self.materializations = 0
        self.route = 'shared_selected'

    def __call__(self, x, route=None):
        if self.route == 'legacy' or route in ('full','suffix'):
            return super().__call__(x, route=route or 'suffix')
        start = time.monotonic()
        self.calls += 1
        with torch.no_grad():
            effective = self.effective(x)
        self.materializations += 1
        leaves = {l:w.detach().requires_grad_(True) for l,w in effective.items()}
        accum = {l:torch.zeros_like(w) for l,w in leaves.items()}
        nll,kl = torch.zeros(self.B,device=x.device),torch.zeros(self.B,device=x.device)
        total = 0.
        for rows,cache in zip(self.rows,self.prefixes):
            with functional_weights(self.model,leaves):
                hidden = suffix_hidden(self.model,cache)
                loss,nr,kr = selected_loss(self.model,hidden,self.spec,self.teacher,self.active,rows)
            gs = torch.autograd.grad(loss,tuple(leaves.values()))
            for l,g in zip(LAYERS,gs): accum[l].add_(g.detach())
            total += float(loss.detach()); nll.add_(nr.detach()); kl.add_(kr.detach())
            self.forward_calls += 1; self.backward_calls += 1
            self.token_work += int(self.spec['tokens']['attention_mask'][rows].sum())
            del gs,hidden,loss,nr,kr
        gradient = torch.cat([(accum[l].double() @ self.adj[l]).float().T for l in LAYERS])
        if not bool(torch.isfinite(gradient).all()): raise FloatingPointError('NONFINITE_GRADIENT')
        if self.initial_gradient is None and bool((x==0).all()): self.initial_gradient=gradient.clone()
        self.seconds += time.monotonic()-start
        if self.calls % 10 == 0:
            print({'event':'oracle', 'calls':self.calls, 'smooth':total, 'seconds':self.seconds},flush=True)
        return total,gradient,{'nll':nll.cpu().tolist(),'kl':kl.cpu().tolist(),
                               'weights':{l:w.detach() for l,w in leaves.items()}}

def actual_preflight(oracle, x0, rho, mask):
    """Exactly three additional whole-batch oracles; no optimization/write."""
    gen=torch.Generator(device=x0.device).manual_seed(20261001)
    probe=torch.randn(x0.shape,device=x0.device,generator=gen)
    probe *= (.02*rho/probe.norm(dim=1))[:,None]; probe[~mask]=0
    start=time.monotonic()
    f,g,p=oracle(probe,route='full')
    reference={k:p[k] for k in ('nll','kl')}; del p
    s,gs,ps=oracle(probe,route='suffix')
    fullsuffix={'loss_abs':abs(f-s),'gradient_relative':float((g-gs).norm()/g.norm().clamp_min(1))}
    fullsuffix['per_request_abs']=max(abs(a-b) for k in ('nll','kl') for a,b in zip(reference[k],ps[k]))
    del g,ps
    o,go,po=oracle(probe)
    optimized={'loss_abs':abs(o-s),'gradient_relative':float((go-gs).norm()/gs.norm().clamp_min(1)),
               'per_request_abs':max(abs(a-b) for k in ('nll','kl') for a,b in zip(reference[k],po[k]))}
    require(all(v<=1e-3 for v in fullsuffix.values()),'BS100_FULL_SUFFIX_PARITY_FAILED')
    optimized_pass=all(v<=1e-3 for v in optimized.values())
    # Explicitly authorized reference fallback is only computational, same formula.
    if not optimized_pass: oracle.route='legacy'
    elapsed=time.monotonic()-start
    return {'status':'PASS','BS':oracle.B,'extra_whole_batch_calls':3,
            'full_suffix':fullsuffix,'optimized':optimized,'optimized_pass':optimized_pass,
            'selected_route':oracle.route,'microbatch':2,'seconds':elapsed,
            'estimated_1200_oracle_seconds_from_comparison_mean':elapsed/3*1200,
            'estimate_excludes_geometry_observation_io':True,
            'peak_gpu_bytes':torch.cuda.max_memory_allocated() if x0.is_cuda else None}
