"""One native joint graph, selected full-vocabulary head, request SUM."""
import time
import torch
from .common import require
from .inputs import chunks, subset, pool
from .entry import PrefixCapture

class Oracle:
    def __init__(self, adapter, spec, microbatch=4, route='strict_prefix'):
        self.a, self.spec, self.microbatch, self.route = adapter,spec,microbatch,route
        self.rows = chunks(spec,microbatch)
        self.teacher = torch.empty(spec['n_requests'],adapter.model.config.vocab_size,device=adapter.device)
        self.anchors = {l:torch.zeros(d[0],spec['n_requests'],device=adapter.device) for l,d in adapter.dims.items()}
        self.rawkeys = {l:torch.empty(len(spec['key_lookup']),d[1],device=adapter.device) for l,d in adapter.dims.items()}
        self.rwmap = {row:k for k,row in enumerate(spec['rw_rows'])}
        self.caches, self.ready = [],False
        self.calls = dict(full_forward=0,suffix_forward=0,backward=0,head_positions=0,
                          original_tokens=0,computed_tokens=0,pruned_tokens=0)
        self.versions = adapter.versions()

    def readout_loss(self,nll_hidden,final_hidden,normalized,rows,offsets,initialize=False,full_head=False):
        a,s = self.a,self.spec
        selected,labels,owners,weights,kinds = [],[],[],[],[]
        for i,row in enumerate(rows):
            request=s['row_request'][row]
            if s['row_kind'][row]=='rewrite':
                positions=torch.where(s['targets'][row]!=-100)[0].tolist()
                for p in positions:
                    selected.append((i,p-offsets[i],False))
                    labels.append(int(s['targets'][row,p]));owners.append(request)
                    weights.append(1/len(positions)/s['specs'][request]['n_rw']);kinds.append('nll')
            else:
                selected.append((i,s['lookup'][row]-offsets[i],True))
                labels.append(0);owners.append(request);weights.append(1.);kinds.append('kl')
        # Readout31 == final for this profile, otherwise preserve separate paths.
        if full_head:
            nlog=a.head(nll_hidden,normalized)
            flog=nlog if normalized and nll_hidden is final_hidden else a.head(final_hidden,True)
            logits=torch.stack([flog[i,p] if k else nlog[i,p] for i,p,k in selected])
        else:
            nh=torch.stack([nll_hidden[i,p] if not k else final_hidden[i,p] for i,p,k in selected])
            if not normalized:
                ix=torch.tensor([not k for _,_,k in selected],device=a.device)
                nh=nh.clone();nh[ix]=a.model.model.norm(nh[ix])
            logits=a.model.lm_head(nh)
        logp=logits.float().log_softmax(-1)
        nll=torch.zeros(s['n_requests'],device=a.device)
        kl=torch.zeros_like(nll)
        for i,(kind,request,weight) in enumerate(zip(kinds,owners,weights)):
            if kind=='nll':
                nll[request]=nll[request]-logp[i,labels[i]]*weight
            else:
                if initialize:
                    self.teacher[request]=logp[i].detach()
                kl[request]=kl[request]+(logp[i].exp()*(logp[i]-self.teacher[request].detach())).sum()
        self.calls['head_positions']+=len(selected)
        return nll,kl

    def evaluate(self,deltas,backward=True,initialize=False,route=None,full_head=False):
        a,s=self.a,self.spec
        require(a.versions()==self.versions,'ENTRY_WEIGHT_MUTATED_DURING_FIT')
        route=route or self.route
        require(initialize or self.ready,'ENTRY_TEACHER_UNINITIALIZED')
        totaln=torch.zeros(s['n_requests'],device=a.device);totalk=torch.zeros_like(totaln)
        start=time.monotonic()
        for index,rows in enumerate(self.rows):
            lengths=[int(s['tokens']['attention_mask'][r].sum()) for r in rows]
            original=sum(lengths)
            if route!='full_reference':
                lengths=[s['lookup'][r]+1 if s['row_kind'][r]=='kl' else n for r,n in zip(rows,lengths)]
            tokens=subset(s['tokens'],rows,a.device,lengths)
            self.calls['original_tokens']+=original
            self.calls['pruned_tokens']+=original-sum(lengths)
            if not initialize and route=='strict_prefix':
                lh,fh,normalized=self.caches[index].replay(deltas)
                offsets=[s['lookup'][r] for r in rows]
                self.calls['suffix_forward']+=1
                self.calls['computed_tokens']+=sum(n-o for n,o in zip(lengths,offsets))
            else:
                def anchor(layer,hidden):
                    if initialize:
                        for i,row in enumerate(rows):
                            r=s['row_request'][row]
                            if row==s['canonical_rows'][r]:
                                self.anchors[layer][:,r]=hidden[i,s['lookup'][row]].detach()
                handles=[]
                if initialize and s['entry_key_prefix_exact']:
                    for layer in a.sites:
                        def keyhook(module,args,layer=layer):
                            for i,row in enumerate(rows):
                                if row in self.rwmap:
                                    self.rawkeys[layer][self.rwmap[row]]=args[0][i,s['lookup'][row]].detach()
                        handles.append(a.blocks[layer].mlp.down_proj.register_forward_pre_hook(keyhook))
                try:
                    if initialize and self.route=='strict_prefix':
                        with PrefixCapture(a,rows,s) as cache:
                            lh,fh,normalized=a.full_hidden(tokens,deltas,rows,s,anchor)
                        self.caches.append(cache.finish(tokens,self.versions))
                    else:
                        lh,fh,normalized=a.full_hidden(tokens,deltas,rows,s,anchor)
                finally:
                    for h in handles:h.remove()
                offsets=[0]*len(rows)
                self.calls['full_forward']+=1;self.calls['computed_tokens']+=sum(lengths)
            nll,kl=self.readout_loss(lh,fh,normalized,rows,offsets,initialize,full_head)
            loss=nll.sum()+a.profile['kl_factor']*kl.sum()
            if backward:
                loss.backward();self.calls['backward']+=1
            totaln+=nll.detach();totalk+=kl.detach()
            del loss,lh,fh,nll,kl
        require(torch.isfinite(totaln).all() and torch.isfinite(totalk).all(),'NONFINITE_NATIVE_LOSS')
        if initialize:
            require(all(torch.isfinite(x).all() and (x.square().sum(0)>0).all() for x in self.anchors.values()),'INVALID_ANCHOR')
            if s['entry_key_prefix_exact']:
                self.keys={l:pool(x,s) for l,x in self.rawkeys.items()}
            else:
                from .writer import capture_keys
                self.keys=capture_keys(a,s,self.microbatch)
            self.rawkeys.clear();self.ready=True
        return dict(nll=totaln,kl=totalk,seconds=time.monotonic()-start)
