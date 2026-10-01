"""E1/E2/E3/E5/E6 adapters of reference7b4de31d, not a new method."""
from contextlib import contextmanager
import time
import resource
import torch
import torch.nn.functional as F
from project.run_scripts.jlz_pilot.run import (LAYERS, chunks, token_subset,
    prefix_cache, functional_weights, capture_entry, capture_keys)
from project.run_scripts.jlz_sequential.oracle import suffix_hidden, selected_loss

ROUTES=('REF_MB2','E1_MB2','E12_MB2','E123_MB2','E123_MB4','E123_MB8',
        'E123_SYNC_MB2','E123_DIRECT_R_MB2')

class CertificationFailed(RuntimeError): pass

@contextmanager
def effective_linear(model, effective, direct=None):
    saved=[]
    try:
        for layer in LAYERS:
            module=model.model.layers[layer].mlp.down_proj
            saved.append((module,'forward' in module.__dict__,module.__dict__.get('forward')))
            def forward(x, layer=layer, module=module):
                if direct is None: return F.linear(x,effective[layer],module.bias)
                proxy,adj,ledger=direct[layer]
                return DirectLinear.apply(x,proxy,effective[layer],adj,module.bias,ledger)
            module.forward=forward
        yield
    finally:
        for module,had,value in reversed(saved):
            if had: module.forward=value
            else: module.__dict__.pop('forward',None)

def crop_tokens(spec, rows, crop):
    tokens=token_subset(spec['tokens'],rows)
    width=tokens['input_ids'].shape[1]
    if crop:
        mask=tokens['attention_mask']; lengths=mask.sum(1)
        expected=torch.arange(width,device=mask.device)[None]<lengths[:,None]
        if not torch.equal(mask.bool(),expected): raise ValueError('RIGHT_PAD_REQUIRED')
        width=int(lengths.max())
        if not bool((spec['targets'][rows,width:]==-100).all()): raise ValueError('TARGET_CROP')
        if any(spec['lookup'][r]>=width for r in rows): raise ValueError('LOOKUP_CROP')
        tokens={k:v[:,:width] for k,v in tokens.items()}
    return tokens,width

class DirectLinear(torch.autograd.Function):
    @staticmethod
    def forward(ctx,x,r64,w,q,bias,ledger):
        z=x.reshape(-1,x.shape[-1]).double()@q
        ctx.save_for_backward(x,w,q,z);ctx.ledger=ledger
        ledger['finite'] &= torch.isfinite(x).all() & torch.isfinite(w).all() & torch.isfinite(q).all() & torch.isfinite(z).all()
        return F.linear(x,w,bias)
    @staticmethod
    def backward(ctx,g):
        x,w,q,z=ctx.saved_tensors; flat=g.reshape(-1,g.shape[-1])
        dx=(flat@w).reshape_as(x) if ctx.needs_input_grad[0] else None
        dr=flat.double().T@z
        bound=(flat.double().abs().amax(1)*x.reshape(-1,x.shape[-1]).double().abs().amax(1)).sum()
        ctx.ledger['bound'] += bound
        ctx.ledger['finite'] &= torch.isfinite(g).all() & torch.isfinite(dr).all() & torch.isfinite(bound)
        if dx is not None: ctx.ledger['finite'] &= torch.isfinite(dx).all()
        return dx,dr,None,None,None,None

def deferred_loss(model,hidden,spec,teacher,active,rows):
    # Same original row/reduction order, only host scalar synchronization deferred.
    B=len(spec['specs']); zero=hidden.reshape(-1)[0]*0
    ns,ks=[[] for _ in range(B)],[[] for _ in range(B)]
    for local,row in enumerate(rows):
        r=spec['row_request'][row]
        if spec['row_kind'][row]=='rewrite':
            mask=spec['targets'][row]!=-100;labels=spec['targets'][row,mask]
            logits=model.lm_head(hidden[local,mask])
            ns[r].append(-logits.log_softmax(-1).gather(1,labels[:,None]).sum()/labels.numel()/spec['n_rw'])
        else:
            logits=model.lm_head(hidden[local,spec['lookup'][row]])
            ks[r].append(F.kl_div(teacher[r:r+1],logits.log_softmax(-1)[None],log_target=True,reduction='batchmean'))
    nll=torch.stack([torch.stack(v).sum() if v else zero for v in ns])
    kl=torch.stack([torch.stack(v).sum() if v else zero for v in ks])
    return (nll[active]+.0625*kl[active]).sum(),nll,kl

class RouteOracle:
    def __init__(self,model,spec,teacher,adj,active,route):
        if route not in ROUTES: raise ValueError(route)
        self.model,self.spec,self.teacher,self.adj,self.active=model,spec,teacher,adj,active
        self.route=route;self.B=len(spec['specs']);self.mb=int(route.rsplit('MB',1)[1])
        self.rows=chunks(spec,self.mb);self.caches=[];self.widths=[];self.calls=0;self.records=[]
        self.entry={l:model.model.layers[l].mlp.down_proj.weight.detach().clone() for l in LAYERS}
        self.crop=route not in ('REF_MB2','E1_MB2');self.direct='DIRECT' in route;self.sync='SYNC' in route
        self.work=dict(valid_tokens=0,padded_tokens=0,attention_L2_positions=0,head_rows=0)
        for rows in self.rows:
            tokens,width=crop_tokens(spec,rows,self.crop)
            self.caches.append(prefix_cache(model,tokens));self.widths.append(width)
            self.work['valid_tokens']+=int(tokens['attention_mask'].sum())
            self.work['padded_tokens']+=tokens['input_ids'].numel()
            self.work['attention_L2_positions']+=len(rows)*width*width
            self.work['head_rows']+=sum(int((spec['targets'][r]!=-100).sum()) if spec['row_kind'][r]=='rewrite' else 1 for r in rows)

    def effective(self,x):
        return {l:self.entry[l]+(x[i*self.B:(i+1)*self.B].T.double()@self.adj[l].T).float() for i,l in enumerate(LAYERS)}

    def __call__(self,x):
        if x.is_cuda:torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
        start=time.monotonic();ev0=torch.cuda.Event(enable_timing=True) if x.is_cuda else None;ev1=torch.cuda.Event(enable_timing=True) if x.is_cuda else None
        if ev0:ev0.record()
        events=[]
        def mark(name):
            if x.is_cuda:
                event=torch.cuda.Event(enable_timing=True);event.record();events.append((name,event))
        mark('start')
        self.calls+=1
        with torch.no_grad():eff=self.effective(x)
        mark('materialization')
        leaves={l:w.detach().requires_grad_(not self.direct) for l,w in eff.items()}
        proxies={l:x[i*self.B:(i+1)*self.B].T.double().detach().requires_grad_() for i,l in enumerate(LAYERS)} if self.direct else None
        targets=proxies if self.direct else leaves
        accum={l:torch.zeros_like(v) for l,v in targets.items()}
        domain={l:dict(bound=torch.zeros((),device=x.device,dtype=torch.float64),finite=torch.ones((),device=x.device,dtype=torch.bool)) for l in LAYERS}
        direct={l:(proxies[l],self.adj[l],domain[l]) for l in LAYERS} if self.direct else None
        nll,kl=torch.zeros(self.B,device=x.device),torch.zeros(self.B,device=x.device)
        total=torch.zeros((),dtype=torch.float64,device=x.device) if self.sync else 0.
        finite=torch.ones((),dtype=torch.bool,device=x.device)
        mark('allocation')
        for rows,cache,width in zip(self.rows,self.caches,self.widths):
            cm=functional_weights(self.model,leaves) if self.route=='REF_MB2' else effective_linear(self.model,leaves,direct)
            with cm:
                hidden=suffix_hidden(self.model,cache)
                mark('suffix_forward')
                spec=self.spec|{'targets':self.spec['targets'][:,:width]}
                loss,nr,kr=(deferred_loss if self.sync else selected_loss)(self.model,hidden,spec,self.teacher,self.active,rows)
                mark('head_loss')
            gs=torch.autograd.grad(loss,tuple(targets.values()))
            mark('backward')
            for l,g in zip(LAYERS,gs):accum[l].add_(g.detach())
            if self.sync:
                total.add_(loss.detach().double())
                finite &= torch.isfinite(loss) & torch.isfinite(nr).all() & torch.isfinite(kr).all()
            else:total+=float(loss.detach())
            nll.add_(nr.detach());kl.add_(kr.detach());del gs,hidden,loss,nr,kr
            mark('accumulation_and_finite_sync')
        if self.direct:
            safe=all(bool(v['finite']) and float(v['bound'])<torch.finfo(torch.float32).max/2 for v in domain.values())
            if not safe:raise CertificationFailed('DIRECT_DENSE_OVERFLOW_DOMAIN_UNCERTIFIED; no hidden reference oracle')
            grad=torch.cat([accum[l].float().T for l in LAYERS])
        else:grad=torch.cat([(accum[l].double()@self.adj[l]).float().T for l in LAYERS])
        mark('pullback')
        if not bool(finite) or not bool(torch.isfinite(grad).all()):
            if self.direct:raise CertificationFailed('DIRECT_NONFINITE_UNCERTIFIED_WITHOUT_AUTHORITATIVE_REFERENCE')
            raise FloatingPointError('NONFINITE_ROUTE')
        if ev1:ev1.record();torch.cuda.synchronize()
        stages={}
        for (_,previous),(name,current) in zip(events,events[1:]):stages[name]=stages.get(name,0.)+previous.elapsed_time(current)/1000
        record=dict(route=self.route,call=self.calls,seconds=time.monotonic()-start,cuda_ms=ev0.elapsed_time(ev1) if ev0 else None,
            forward_chunks=len(self.rows),backward_chunks=len(self.rows),materialization_GEMMs=5,
            mapping_GEMMs=10*len(self.rows) if self.direct else 5,dense_gradient_GEMMs=0 if self.direct else 5*len(self.rows),
            peak_allocated=torch.cuda.max_memory_allocated() if x.is_cuda else None,peak_reserved=torch.cuda.max_memory_reserved() if x.is_cuda else None,
            host_RSS_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,**self.work)
        record['stage_cuda_seconds']=stages
        record['stage_interpretation']='sequential event intervals include host synchronization gaps; components partition this oracle, do not add again to total'
        if self.direct:record['overflow_bounds']={str(l):float(v['bound']) for l,v in domain.items()}
        self.records.append(record)
        return float(total),grad,dict(nll=nll.cpu().tolist(),kl=kl.cpu().tolist(),weights={l:w.detach() for l,w in leaves.items()},timing=record)

class KeyStop(Exception):pass

@torch.no_grad()
def early_keys(model,spec,contexts,mb=2):
    keys={l:[] for l in LAYERS};tokens=spec['key_tokens'];lookups=spec['key_lookup']
    for start in range(0,len(lookups),mb):
        rows=list(range(start,min(start+mb,len(lookups))));handles=[];seen=[]
        for l in LAYERS:
            def hook(module,args,layer=l):
                keys[layer].append(args[0][torch.arange(len(rows),device=args[0].device),torch.tensor([lookups[r] for r in rows],device=args[0].device)].detach().clone());seen.append(layer)
                if layer==8:raise KeyStop()
            handles.append(model.model.layers[l].mlp.down_proj.register_forward_pre_hook(hook))
        try:
            try:model.model(**token_subset(tokens,rows),use_cache=False)
            except KeyStop:
                if seen!=list(LAYERS):raise RuntimeError('KEY_STOP_INCOMPLETE')
            else:raise RuntimeError('KEY_STOP_NOT_REACHED')
        finally:
            for handle in handles:handle.remove()
    result={};B=len(spec['specs']);nc=sum(map(len,contexts))
    for l,parts in keys.items():
        raw=torch.cat(parts).reshape(B,nc,-1);groups=[];start=0
        for group in contexts:groups.append(raw[:,start:start+len(group)].mean(1));start+=len(group)
        result[l]=torch.stack(groups).mean(0).T.contiguous()
    return result

@torch.no_grad()
def selected_entry(model,spec,mb=2):
    B=len(spec['specs']);device=next(model.parameters()).device
    anchors={l:torch.zeros(model.config.hidden_size,B,device=device) for l in LAYERS}
    teacher=torch.empty(B,model.config.vocab_size,device=device);nll=torch.zeros(B,device=device)
    for rows in chunks(spec,mb):
        handles=[]
        for l in LAYERS:
            def hook(module,args,out,layer=l):
                hidden=out[0] if isinstance(out,(tuple,list)) else out
                for request,item in enumerate(spec['specs']):
                    if item['offset'] in rows:anchors[layer][:,request]=hidden[rows.index(item['offset']),item['lookup'][0]]
            handles.append(model.model.layers[l].register_forward_hook(hook))
        try:hidden=model.model(**token_subset(spec['tokens'],rows),use_cache=False).last_hidden_state
        finally:
            for handle in handles:handle.remove()
        for local,row in enumerate(rows):
            r=spec['row_request'][row]
            if spec['row_kind'][row]=='kl':teacher[r]=model.lm_head(hidden[local,spec['lookup'][row]]).log_softmax(-1)
            else:
                mask=spec['targets'][row]!=-100;ids=spec['targets'][row,mask]
                nll[r]+=-model.lm_head(hidden[local,mask]).log_softmax(-1).gather(1,ids[:,None]).sum()/ids.numel()/spec['n_rw']
    return anchors,teacher,nll

@torch.no_grad()
def selected_committed(model,spec,teacher,active,mb=2):
    nll=torch.zeros(len(spec['specs']),device=teacher.device);kl=torch.zeros_like(nll)
    for rows in chunks(spec,mb):
        hidden=model.model(**token_subset(spec['tokens'],rows),use_cache=False).last_hidden_state
        _,nr,kr=selected_loss(model,hidden,spec,teacher,active,rows)
        nll.add_(nr);kl.add_(kr)
    return nll,kl
